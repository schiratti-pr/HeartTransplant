import os
import glob
import argparse
from omegaconf import OmegaConf
import collections
import pandas as pd

import pytorch_lightning as pl
import torch
from monai.networks.nets import SegResNet, SwinUNETR, UNet

from data.dataset import NLST_2D_NIFTI_Dataset
from data.dataset import data_to_slices

from matplotlib.pyplot import figure, imshow, axis, savefig
import matplotlib.pyplot as plt


def showImagesHorizontally(list_of_arrs):
    fig = figure()
    num_files = len(list_of_arrs)
    for i in range(num_files):
        a = fig.add_subplot(1, num_files, i+1)
        image = list_of_arrs[i]
        imshow(image, cmap='Greys_r')
        if i == 0:
            title = 'scan'
        elif i == 1:
            title = 'mask'
        else:
            title ='prediction'
        a.set_title(title)
        axis('off')
        plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description='Processing configuration for training')
    parser.add_argument('--config', type=str, help='path to config file', default='configs/config_2d.yaml')
    args = parser.parse_args()

    pl.seed_everything(int(os.environ.get('LOCAL_RANK', 0)))

    # Load configuration file
    config = OmegaConf.load(args.config)

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

    data_dict_train, data_dict_val = {}, {}
    for index, row in splits.iterrows():
        patient_id = row['id']
        split = row['set']

        for key in data_dict.keys():
            if str(patient_id) in key and split == 'train':
                data_dict_train.update({key: data_dict[key]})
            if str(patient_id) in key and split == 'val':
                data_dict_val.update({key: data_dict[key]})

    data_train = data_to_slices(collections.OrderedDict(data_dict_train), nifti=True)
    data_val = data_to_slices(collections.OrderedDict(data_dict_val), nifti=True)

    print('Train:', len(data_train), 'Val:', len(data_val))

    val_dataset = NLST_2D_NIFTI_Dataset(
        patients_paths=data_val,
        target_size=config['data']['target_size'],
        transform=None
    )

    # Init model
    model = UNet(
                spatial_dims=2,
                in_channels=1,
                out_channels=1,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
                num_res_units=2,
            )

    # Load weights
    checkpoint_path = '/home/mdo4009/transplant-rejection/logs/UNet-2D/UNet__2022-12-02-00-17-26/step=2300.ckpt'
    state_dict = torch.load(str(checkpoint_path), map_location='cpu')['state_dict']
    for key in list(state_dict.keys()):
        state_dict[key.replace('net.', '')] = state_dict.pop(key)

    model.load_state_dict(state_dict)
    model.eval()

    vis_id = 5
    img_ex = val_dataset[vis_id]
    img, mask = torch.unsqueeze(img_ex['data'], 0), img_ex['label'].squeeze().numpy()

    prediction = model(img).squeeze().detach()
    sigmoid_pred = torch.sigmoid(prediction)
    class_pred = torch.round(sigmoid_pred).numpy()

    f = showImagesHorizontally([img.squeeze().numpy(), mask, class_pred])
    savefig('result_{}.png'.format(vis_id), bbox_inches='tight')


if __name__ == '__main__':
    main()