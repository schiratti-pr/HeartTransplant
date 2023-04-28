from monai.networks.nets import SegResNet, SwinUNETR, UNet

import torch
import numpy as np
import pydicom as dicom
import nibabel as nib
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


def array_to_tensor(img_array) -> torch.FloatTensor:
    return torch.FloatTensor(img_array)


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


def preprocess(path_to_scan):
    roi = nib.load(path_to_scan).get_fdata()

    roi = array_to_tensor(roi)
    # Transform to 3D cube roi with same size for all dimensions
    seed = 1
    imtrans = Compose(
        [
            ScaleIntensity(),
            EnsureChannelFirst(),
            Resize([256, 256, 256])
        ]
    )
    imtrans.set_random_state(seed=seed)

    roi = imtrans(roi)
    # Pad depth dimension
    roi = pad_volume(roi, 256).double().unsqueeze(0)

    return roi


def get_model(name, spatial_dims=2, num_in_channels=1):
    if name == 'SegResNet':
        net = SegResNet(
            spatial_dims=spatial_dims,
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
            init_filters=16,
            in_channels=num_in_channels,
            out_channels=1,
            dropout_prob=0.2,
        )
    elif name == 'SwinUNETR':
        net = SwinUNETR(
            img_size=512,
            in_channels=num_in_channels,
            out_channels=1,
            feature_size=48,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
            spatial_dims=spatial_dims
        )
    elif name == 'UNet':
        net = UNet(
            spatial_dims=spatial_dims,
            in_channels=num_in_channels,
            out_channels=1,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )
    if spatial_dims == 3:
        net = net.double()
    else:
        net = net

    return net
