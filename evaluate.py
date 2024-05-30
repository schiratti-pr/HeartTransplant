import os
import glob
import argparse
from omegaconf import OmegaConf
import collections
import pandas as pd
import numpy as np
import tqdm
from pathlib import Path
import random
import cv2
import cc3d
import copy
import math
import nibabel as nib


import pytorch_lightning as pl
import torch
import torch.backends.cudnn as cudnn
from monai.data import TestTimeAugmentation
from monai.transforms import (Compose, Resize, RandAffined)


from data.dataset import NLST_2D_NIFTI_Dataset, NLST_NIFTI_Dataset, NLST_2_5D_Dataset
from data.dataset import data_to_slices
from models.training_modules import binary_dice_coefficient
from utils.utils import get_model


from matplotlib.pyplot import figure, imshow, axis, savefig
import matplotlib.pyplot as plt


def bbox(array, point, radius):
    a = array[np.where(np.logical_and(array[:, 0] >= point[0] - radius, array[:, 0] <= point[0] + radius))]
    b = a[np.where(np.logical_and(a[:, 1] >= point[1] - radius, a[:, 1] <= point[1] + radius))]
    c = b[np.where(np.logical_and(b[:, 2] >= point[2] - radius, b[:, 2] <= point[2] + radius))]
    return c


def hausdorff(surface_a, surface_b):

    # Taking two arrays as input file, the function is searching for the Hausdorff distane of "surface_a" to "surface_b"
    dists = []

    l = len(surface_a)

    for i in range(l):

        # walking through all the points of surface_a
        dist_min = 1000.0
        radius = 0
        b_mod = np.empty(shape=(0, 0, 0))

        # increasing the cube size around the point until the cube contains at least 1 point
        while b_mod.shape[0] == 0:
            b_mod = bbox(surface_b, surface_a[i], radius)
            radius += 1

        # to avoid getting false result (point is close to the edge, but along an axis another one is closer),
        # increasing the size of the cube
        b_mod = bbox(surface_b, surface_a[i], radius * math.sqrt(3))

        for j in range(len(b_mod)):
            # walking through the small number of points to find the minimum distance
            dist = np.linalg.norm(surface_a[i] - b_mod[j])
            if dist_min > dist:
                dist_min = dist

        dists.append(dist_min)

    return np.max(dists)


def showPredictionContour(img, gt, pred, dice, patient_dir, slice_id):
    if len(img.shape) != 2:
        img = img[round(img.shape[0]/2), :, :]

    img = np.stack((img,) * 3, axis=-1).copy()
    patient = patient_dir.split('/')[-1]

    pred = pred.astype('uint8')
    contours_pred, _ = cv2.findContours(pred.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    gt[gt == 1] = 255
    gt = gt.astype('uint8')
    contours_gt, _ = cv2.findContours(gt.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours_pred:
        cv2.drawContours(img, [c], 0, (255, 0, 0), 3)
    for c in contours_gt:
        cv2.drawContours(img, [c], 0, (0, 255, 0), 3)

    fig, ax = plt.subplots()
    ax.imshow(img)
    proxy = [plt.Rectangle((0, 0), 1, 1, fc='green'), plt.Rectangle((0, 0), 1, 1, fc='red')]
    ax.set_title("Patient: {}, Slice: {}, \nDice score: {} %".format(patient, slice_id, round(dice.item()*100, 4)))
    plt.legend(proxy, ["mask", "prediction"])
    axis('off')
    plt.tight_layout()

    return fig


def main():
    SEED = 0
    cudnn.benchmark = True
    cudnn.deterministic = False
    torch.cuda.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    parser = argparse.ArgumentParser(description='Processing configuration for training')
    parser.add_argument('--exp_path', type=str, help='path to experiment path',
                        default=r't/best_nnUNet_split/UNet__2024-05-23-20-31-24')
    parser.add_argument('--save_fig', type=bool, help='save prediction figure', default=True)
    parser.add_argument('--device', type=str, default='cpu') # 'cuda:1' # 'cpu'
    parser.add_argument('--data_split', type=str, default='test')
    parser.add_argument('--tta', type=bool, help='run with tta', default=False)
    parser.add_argument('--result_save', type=str, help='path to folder', default='../results')
    parser.add_argument('--data_split_path', type=str, help='path to splits csv', default='../../data//NLST-train_val_test-split-120.csv')

    args = parser.parse_args()

    device = torch.device(args.device)

    pl.seed_everything(int(os.environ.get('LOCAL_RANK', 0)))

    # Load configuration file
    config_path = os.path.join(args.exp_path, 'config.yaml')
    config = OmegaConf.load(config_path)

    raw_directory = config['data']['raw_directory']
    annotations_dir = config['data']['annotations_dir']

    data_dict = {}
    for raw_sample in glob.glob(raw_directory + '**'):
        sample_id = raw_sample.split('/')[-1][:-4]
        for label_path in glob.glob(annotations_dir + '**'):
            if sample_id in label_path and '.nii' in label_path:
                data_dict.update({raw_sample: label_path})

    # Data split
    splits = pd.read_csv(args.data_split_path)

    data_dict_split = {}
    for index, row in splits.iterrows():
        patient_id = row['id']
        split = row['set'] # split

        for key in data_dict.keys():
            if str(patient_id) in key and split == args.data_split:
                data_dict_split.update({key: data_dict[key]})

    if len(config['data']['target_size']) == 2 and config['data'].get('window_step') is None:
        data_split = data_to_slices(collections.OrderedDict(data_dict_split), nifti=True)

        dataset = NLST_2D_NIFTI_Dataset(
            patients_paths=data_split,
            target_size=config['data']['target_size'],
            transform=None,
            eval_mode=True
        )
        number_channels = 1
    elif len(config['data']['target_size']) == 2:
        data_split = data_to_slices(collections.OrderedDict(data_dict_split), nifti=True)
        dataset = NLST_2_5D_Dataset(
            patients_paths=data_split,
            target_size=config['data']['target_size'],
            transform=None,
            window_step=config['data'].get('window_step'),
            eval_mode=True
        )
        number_channels = config['data']['window_step'] * 2 + 1
    else:
        dataset = NLST_NIFTI_Dataset(
            patients_paths=data_dict_split,
            target_size=config['data']['target_size'],
            transform=None,
            crop_heart=config['data'].get('crop_heart'),
            eval_mode=True
        )
        number_channels = 1

    # Init model
    model = get_model(config['model']['name'], spatial_dims=len(config['data']['target_size']),
                                                                num_in_channels=number_channels)
    model.to(device)

    # Load weights
    all_checkpoints = [str(x) for x in sorted(Path(args.exp_path).iterdir(), key=os.path.getmtime) if '.ckpt' in str(x)]
    checkpoint_path = all_checkpoints[-1]
    state_dict = torch.load(str(checkpoint_path), map_location=device)['state_dict']
    for key in list(state_dict.keys()):
        state_dict[key.replace('net.', '')] = state_dict.pop(key)

    model.load_state_dict(state_dict)
    model.eval()

    dice_scores, hd_scores, case_ids = [], [], []
    for vis_id in tqdm.tqdm(range(len(dataset))):
        img_ex = dataset[vis_id]
        img, mask = torch.unsqueeze(img_ex['data'], 0), img_ex['label'].squeeze()

        if args.tta:
            transforms = Compose([RandAffined(
                keys=["image", "label"],
                prob=1.0,
                spatial_size=(512, 512),
                rotate_range=(10, 10),
                scale_range=((0.8, 1), (0.8, 1)),
                padding_mode="zeros",
                mode=("bilinear", "nearest"),
            )])

            tt_aug = TestTimeAugmentation(
                transforms,
                batch_size=1,
                num_workers=0,
                inferrer_fn=lambda x: torch.sigmoid(model(x)),
                device=device
            )
            _, mean_tta, _, _ = tt_aug({"image": img_ex['data'].to(device),
                                                           'label': torch.unsqueeze(mask, 0)}, num_examples=5)
            class_pred = torch.round(mean_tta).cpu().squeeze()
        else:
            prediction = model(img.to(device)).squeeze().detach()
            sigmoid_pred = torch.sigmoid(prediction)
            class_pred = torch.round(sigmoid_pred).cpu()

        # Filtering out everything beyond the pre-calculated heart zone
        output_pred = np.zeros(class_pred.shape)
        if len(class_pred.shape) == 2 and config['data']['target_size'][0] == 512:
            output_pred[81:337, 158:430] = class_pred[81:337, 158:430]
            class_pred = torch.tensor(output_pred)
        elif config['data']['target_size'][0] == 256:
            # output_pred[40:168, 79:215, :] = class_pred[40:168, 79:215, :]
            print('In heart zone filtering!')
            output_pred[79:215, 40:168, :] = class_pred[79:215, 40:168, :]
            class_pred = torch.tensor(output_pred)
        elif config['data']['target_size'][0] == 128:
            output_pred[20:84, 39:107, ] = class_pred[20:84, 39:107, :]
            class_pred = torch.tensor(output_pred)

        # CCA
        connectivity = 6  # only 4,8 (2D) and 26, 18, and 6 (3D) are allowed
        labels_out, N = cc3d.connected_components(class_pred.numpy(), connectivity=connectivity, return_N=True)
        dict_components = {}
        for label, image in cc3d.each(labels_out, binary=False, in_place=True):
            dict_components.update({np.count_nonzero(image): copy.deepcopy(image)})
        try:
            class_pred = dict_components[max(dict_components.keys())]
            class_pred = torch.Tensor(class_pred.astype(np.float32))
        except:
            pass   # No components were found

        # Calculate Hausdorff distance and Dice score if the volumes are in 3D
        # if len(class_pred.shape) == 3:
        #     patient_hd_scores = []
        #     for sl in range(mask.shape[-1]):
        #         patient_hd_scores.append(hausdorff(mask[:, :, sl], class_pred[:, :, sl]))
        #     hd_scores.append(np.mean(patient_hd_scores))

        dice_coeff = binary_dice_coefficient(class_pred, mask)
        dice_scores.append(dice_coeff.item())
        case_ids.append(img_ex['patient_dir'].split('/')[-1][:-4])

        if args.save_fig:
            if len(config['data']['target_size']) == 2:
                showPredictionContour(img.squeeze().numpy(), mask.numpy(), class_pred.numpy(), dice_coeff,
                               img_ex['patient_dir'], img_ex['slice_id'])

                if not os.path.exists(args.result_save):
                    os.makedirs(args.result_save)
                savefig(args.result_save + '/{}_{}.png'.format(vis_id, round(dice_coeff.item()*100, 2)),
                        bbox_inches='tight')
            else:
                # For 3d saving the entire prediction mask
                if not os.path.exists(args.result_save):
                    os.makedirs(args.result_save)

                nifti_file = nib.Nifti1Image(class_pred.numpy(), np.eye(4))
                patient_id = img_ex['patient_dir'].split('/')[-1]
                scan_path_save = os.path.join(args.result_save,  str(patient_id))
                nib.save(nifti_file, scan_path_save)

    df = pd.DataFrame([dice_scores, case_ids]).T
    df = df.rename(columns={0: "dice", 1: "patient_id"})
    df.to_csv('3d_{}_dice_per_patient_{}.csv'.format(config['model']['name'], args.data_split), index=False)

    print('Mean Dice on {} : {}'.format(args.data_split, np.mean(dice_scores)))
    if len(hd_scores) > 0:
        print('Mean HD on {} : {}'.format(args.data_split, np.mean(hd_scores)))


if __name__ == '__main__':
    main()
