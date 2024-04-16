# from monai.losses import DiceLoss, DiceFocalLoss, DiceCELoss
# from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss

import os
import nibabel as nib
import pandas as pd
import numpy as np
from monai.metrics import compute_meandice
import torch
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from monai.losses import DiceLoss, DiceFocalLoss, DiceCELoss
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

from data.aug import AUGMENTATIONS


def pad_volume(vol, roi_size):
    """
    If the depth dimension is less than ROI_size, we add pading with zeros
    Padding the volume of size (1, W, H, D) to fit into a volume of size (1, ROI_size, ROI_size, ROI_size)
    """
    for d in range(1, 4):
        if vol.shape[d] < roi_size:
            pad_shape = [1] + [val if idx + 1 != d
                               else roi_size - val
                               for idx, val in enumerate(vol.shape[1:])]
            padding_patch = np.zeros(pad_shape)
            vol = np.concatenate([vol, padding_patch], axis=d)
    return torch.tensor(vol)


def process_nii(nii_file, baseline_id):
    
    target_size = [256, 256, 256]
    transform = AUGMENTATIONS.get('basic3d')
    
    # Load the .nii files
    label = nib.load(nii_file).get_fdata()
    
    if int(baseline_id) not in [100108, 100085, 100088, 100092, 100019, 100031, 100046, 100053, 100072, 100081]:
        label = np.flip(label, -1).copy()
    label = np.transpose(label, [1, 0, 2])

    mask = torch.Tensor(label)
    
    seed=1
    segtrans = Compose(
        [
            EnsureChannelFirst(),
            Resize(target_size)
            # CenterSpatialCrop(self.target_size),
        ]
    )
    segtrans.set_random_state(seed=seed)

    # Transformations / augmenting input
    if transform:
        aug = transform({'image': mask, 'label': mask})
        mask = aug['label']

    mask = segtrans(mask)
    
    # Pad depth dimension
    mask = pad_volume(mask, target_size[0])
        
    return mask


# Define the directories
baseline_dir = "C:/Users/pps21/Box/NLST Chest CT/Annotations - VT"
repeated_dir = "C:/Users/pps21/Box/NLST Chest CT/Annotations - VT - Intra-user Repeat"

# Get the .nii files in the directories
baseline_files = [f for f in os.listdir(baseline_dir) if f.endswith('.nii')]
repeated_files = [f for f in os.listdir(repeated_dir) if f.endswith('.nii')]

# Initialize the lists to store the imported results
baseline_list = []
repeated_list = []

# Load the .nii files and check if the IDs match
used_ids = []
for baseline_file in baseline_files:
    baseline_id = baseline_file[:6]
    matching_files = [f for f in repeated_files if f.startswith(baseline_id)]
    
    if matching_files:
        used_ids.append(baseline_id)
        repeated_file = matching_files[0]
        
        baseline_file = os.path.join(baseline_dir, baseline_file)
        repeated_file = os.path.join(repeated_dir, repeated_file)
        
        mask_baseline = process_nii(baseline_file, baseline_id)
        mask_repeated = process_nii(repeated_file, baseline_id)
        
        # Store the imported results
        baseline_list.append(mask_baseline)
        repeated_list.append(mask_repeated)

baseline_torch = torch.stack(baseline_list)
repeated_torch = torch.stack(repeated_list)

self = DiceLoss(reduction='none')
loss = self(baseline_torch, repeated_torch)
loss = loss.numpy().reshape(12)

df = pd.DataFrame(data=[used_ids, loss.T], index=['ID', 'Dice Loss']).T
df.to_csv('../data/intra_user_validation.csv', index=False)