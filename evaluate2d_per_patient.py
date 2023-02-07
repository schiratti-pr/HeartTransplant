import os
import glob
import argparse
from omegaconf import OmegaConf
import pandas as pd
import numpy as np
import tqdm
from pathlib import Path
import random
import cc3d
import copy
import nibabel as nib


import pytorch_lightning as pl
import torch
import torch.backends.cudnn as cudnn
from monai.transforms import (Compose, RandAffined, EnsureChannelFirst, ScaleIntensity)


from models.training_modules import binary_dice_coefficient
from utils.utils import get_model


def array_to_tensor(img_array) -> torch.FloatTensor:
    return torch.FloatTensor(img_array)


def prepare_patient(patient_path, label_path):
    patient_id = patient_path.split('/')[-1][:-4]
    roi = nib.load(patient_path).get_fdata()
    roi = array_to_tensor(roi)

    label = nib.load(label_path).get_fdata()
    if int(patient_id) not in [100108, 100085, 100088, 100092, 100019, 100031, 100046, 100053, 100072, 100081]:
        label = np.flip(label, -1).copy()
    label = np.transpose(label, [1, 0, 2])

    mask = torch.Tensor(label)

    # Transform to 3D cube roi with same size for all dimensions
    seed = 1
    imtrans = Compose([ScaleIntensity(),EnsureChannelFirst()])
    imtrans.set_random_state(seed=seed)

    segtrans = Compose([EnsureChannelFirst()])
    segtrans.set_random_state(seed=seed)

    roi = imtrans(roi)
    mask = segtrans(mask)
    return roi, mask


def run_slice_prediction(x, model, device):
    img = torch.unsqueeze(x, 0)

    prediction = model(img.to(device)).squeeze().detach()
    sigmoid_pred = torch.sigmoid(prediction)
    class_pred = torch.round(sigmoid_pred).cpu()

    # Limit the heart region
    output_pred = np.zeros(class_pred.shape)
    output_pred[81:337, 158:430] = class_pred[81:337, 158:430]

    return output_pred


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
                        default='/home/mdo4009/transplant-rejection/logs/UNet-2D/UNet__2022-12-03-11-37-40/')
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--data_split', type=str, default='val')
    parser.add_argument('--tta', type=bool, help='run with tta', default=False)

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
    splits = pd.read_csv(config['data']['data_splits'])

    data_dict_split = {}
    for index, row in splits.iterrows():
        patient_id = row['id']
        split = row['set']

        for key in data_dict.keys():
            if str(patient_id) in key and split == args.data_split:
                data_dict_split.update({key: data_dict[key]})

    # Init model
    if config['data'].get('window_step') is not None:
        number_channels = config['data'].get('window_step') * 2 + 1
    else:
        number_channels = 1

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

    dice_scores, case_ids = [], []
    # Run prediction per patient
    for idx in tqdm.tqdm(range(len(data_dict_split))):
        patient_path, label_path = list(data_dict_split.items())[idx]
        case_id = patient_path.split('/')[-1][:-4]
        roi, mask = prepare_patient(patient_path, label_path)
        mask = mask.squeeze()

        # Iterate over depth and predict per slice
        prediction = []
        for slice_id in range(roi.shape[-1]):
            if slice_id >= 15 and slice_id <= 125 and slice_id <= roi.shape[-1] - 3:
                if config['data'].get('window_step') is None:
                    pred_slice = run_slice_prediction(roi[:, :, :, slice_id], model, device)
                else:
                    # Run prediction for 2.5D
                    window_step = config['data'].get('window_step')
                    input = roi[:, :, :, slice_id - window_step: slice_id + window_step + 1].squeeze()
                    input = input.permute(2, 0, 1)
                    pred_slice = run_slice_prediction(input, model, device)
            else:
                pred_slice = np.zeros((512, 512))
            prediction.append(pred_slice)

        prediction = np.stack(prediction, -1)

        # CCA on whole 3D volume prediction
        connectivity = 6
        labels_out, N = cc3d.connected_components(prediction, connectivity=connectivity, return_N=True)
        dict_components = {}
        for label, image in cc3d.each(labels_out, binary=False, in_place=True):
            dict_components.update({np.count_nonzero(image): copy.deepcopy(image)})
        try:
            prediction = dict_components[max(dict_components.keys())]
        except:
            # No components were found
            pass
        prediction = torch.Tensor(prediction.astype(np.float32))

        # Filtering out everything beyond the pre-calculated heart zone
        output_pred = np.zeros(prediction.shape)
        output_pred[81:337, 158:430, :] = prediction[81:337, 158:430, :]
        prediction = torch.tensor(output_pred)

        dice_coeff = binary_dice_coefficient(prediction, mask)
        dice_scores.append(dice_coeff.item()), case_ids.append(case_id)

    df = pd.DataFrame([dice_scores, case_ids]).T
    df = df.rename(columns={0: "dice", 1: "patient_id"})
    df.to_csv('25d_{}_dice_per_patient_{}.csv'.format(config['model']['name'], args.data_split), index=False)

    print('Mean Dice per patient on {} : {}'.format(args.data_split, np.mean(dice_scores)))


if __name__ == '__main__':
    main()
