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

from monai.transforms import (
    Activations,
    EnsureChannelFirst,
    AsDiscrete,
    Compose,
    LoadImage,
    RandSpatialCrop,
    Resize,
    ScaleIntensity,
    CenterSpatialCrop
)


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

def array_to_tensor(img_array) -> torch.FloatTensor:
    return torch.FloatTensor(img_array)

def tensor_to_array(tensor: torch.Tensor) -> np.ndarray:
    return tensor.numpy()

def pad_volume(vol, roi_size):
    """
    If the depth dimension is less than ROI_size, we add pading with zeros
    Padding the volume of size (1, W, H, D) to fit into a volume of size (1, ROI_size, ROI_size, ROI_size)
    """
    for d in range(1, 3):
        if vol.shape[d] < roi_size:
            pad_shape = [1] + [val if idx + 1 != d
                               else roi_size - val
                               for idx, val in enumerate(vol.shape[1:])]
            padding_patch = np.zeros(pad_shape)
            vol = np.concatenate([vol, padding_patch], axis=d)
    return vol

def transform_img(img_path, mask_path, crop_heart=False):
    
    target_size = [256,256,256]
    
    roi = nib.load(img_path).get_fdata()
    mask = nib.load(mask_path).get_fdata()
    
    roi = array_to_tensor(roi)
    mask = array_to_tensor(mask)

    if crop_heart:
        # Crop the region around the heart based on stats - [81:337, 158:430, :], adding padding of minimum 20
        roi = roi[50:360, 140:450, :]

    # Transform to 3D cube roi with same size for all dimensions
    seed = 1
    imtrans = Compose(
        [
            ScaleIntensity(),
            EnsureChannelFirst(),
            Resize(target_size)
            # CenterSpatialCrop(self.target_size),  # todo: limit the random crop
        ]
    )
    imtrans.set_random_state(seed=seed)

    segtrans = Compose(
        [
            EnsureChannelFirst(),
            Resize(target_size)
            # CenterSpatialCrop(self.target_size),
        ]
    )
    segtrans.set_random_state(seed=seed)

    roi = imtrans(roi)
    mask = segtrans(mask)
    
    roi = tensor_to_array(roi).squeeze()
    mask = tensor_to_array(mask).squeeze()

    # Pad depth dimension
    try:
        roi, mask = pad_volume(roi, target_size[0]).double(), pad_volume(mask, target_size[0])
    except:
        roi, mask = pad_volume(roi, target_size[0]), pad_volume(mask, target_size[0])

    return roi, mask

def get_middle_frame(img, mask):
    """
    Get the middle frame of the non-null frames in respect to each dimension.
    """
    middle_frames_img = []
    middle_frames_mask = []
    for dim in range(3):
        non_null_frames = np.where(np.any(img, axis=tuple(set(range(3)) - {dim})))[0]
        middle_frame_index = non_null_frames[len(non_null_frames) // 2]
        if dim == 0:
            middle_frames_img.append(img[middle_frame_index, :, :])
            middle_frames_mask.append(mask[middle_frame_index, :, :])
        elif dim == 1:
            middle_frames_img.append(img[:, middle_frame_index, :])
            middle_frames_mask.append(mask[:, middle_frame_index, :])
        else:
            middle_frames_img.append(img[:, :, middle_frame_index])
            middle_frames_mask.append(mask[:, :, middle_frame_index])
    return middle_frames_img, middle_frames_mask



def main():

    parser = argparse.ArgumentParser(description='Processing configuration for training')
    parser.add_argument('--exp_path', type=str, help='path to experiment path',
                        default=r'configs')
    parser.add_argument('--data_split', type=str, default='test')

    # args = parser.parse_args()
    args, unknown = parser.parse_known_args() # solution for jupyter run
    
    # Load configuration file
    config_path = os.path.join(args.exp_path, 'config.yaml')
    config = OmegaConf.load(config_path)

    raw_directory = r"C:\Users\pps21\Documents\Cornell\data\NLST_nifti\\"
    first_eval = r"C:\Users\pps21\Documents\Cornell\data\Annotations - VT - test\\"
    second_eval = r"C:\Users\pps21\Documents\Cornell\results\results_nlst_120\\"

    data_dict_first = {}
    for raw_sample in glob.glob(raw_directory + '**'):
        sample_id = raw_sample.split('\\')[-1][:-4]
        for label_path in glob.glob(first_eval + '**'):
            if sample_id in label_path and '.nii' in label_path:
                data_dict_first.update({raw_sample: label_path})
                
    data_dict_second = {}
    for raw_sample in glob.glob(raw_directory + '**'):
        sample_id = raw_sample.split('\\')[-1][:-4]
        for label_path in glob.glob(second_eval + '**'):
            if sample_id in label_path and '.nii' in label_path:
                data_dict_second.update({raw_sample: label_path})

    dataset_first = NLST_NIFTI_Dataset(
        patients_paths=data_dict_first,
        target_size=config['data']['target_size'],
        transform=None,
        crop_heart=config['data'].get('crop_heart'),
        eval_mode=True
    )
    
    dataset_second = NLST_NIFTI_Dataset(
        patients_paths=data_dict_second,
        target_size=config['data']['target_size'],
        transform=None,
        crop_heart=config['data'].get('crop_heart'),
        eval_mode=True
    )
    number_channels = 1

    fig, axs = plt.subplots(len(data_dict_first), 3, figsize=(30, len(data_dict_first)*5))
    
    dice_scores, hd_scores, case_ids = [], [], []
    for i, vis_id in enumerate(tqdm.tqdm(range(len(dataset_first)))):
        img_first = dataset_first[vis_id]
        img, mask_first = torch.unsqueeze(img_first['data'], 0), img_first['label'].squeeze()

        img_second = dataset_second[vis_id]
        img, mask_second = torch.unsqueeze(img_second['data'], 0), img_second['label'].squeeze()

        dice_coeff = binary_dice_coefficient(mask_first, mask_second)
        dice_scores.append(dice_coeff.item())
        
        subject_id = img_first['patient_dir'].split('\\')[-1][:-4]
        case_ids.append(subject_id)
        
        img = tensor_to_array(img).squeeze()
        mask_first = tensor_to_array(mask_first).squeeze()
        mask_second = tensor_to_array(mask_second).squeeze()
        
        # Get the middle frame of each dimension before flipping
        middle_frames_img, middle_frames_mask_first = get_middle_frame(img, mask_first)
        middle_frames_img, middle_frames_mask_second = get_middle_frame(img, mask_second)
        
        # Plot the middle frames before flipping
        for j in range(3):
            axs[i, j].imshow(middle_frames_img[j], cmap='gray')
            axs[i, j].imshow(middle_frames_mask_first[j], cmap='jet', alpha=0.2)
            axs[i, j].imshow(middle_frames_mask_second[j], cmap='viridis', alpha=0.2)
            axs[i, j].axis('off')

        
    # Set the title for the row on the right-most subplot, rotated 90 degrees clockwise
        axs[i, 2].text(1.15, 0.5, f"{subject_id}, Dice: {dice_scores[i]:.3f}", fontsize=25, rotation=-90, va='center', ha='right', transform=axs[i, 2].transAxes)

    # Display the figure
    plt.tight_layout()
    plt.show()

    df = pd.DataFrame([dice_scores, case_ids]).T
    df = df.rename(columns={0: "dice", 1: "patient_id"})
    # df.to_csv(r"C:\Users\pps21\Documents\Cornell\results\intra_user_valid_dice_per_patient.csv", index=False)

    print('Mean Dice on {} : {}'.format(args.data_split, np.mean(dice_scores)))
    print('Standard deviation Dice on {} : {}'.format(args.data_split, np.std(dice_scores)))
    
    if len(hd_scores) > 0:
        print('Mean HD on {} : {}'.format(args.data_split, np.mean(hd_scores)))


if __name__ == '__main__':
    main()
