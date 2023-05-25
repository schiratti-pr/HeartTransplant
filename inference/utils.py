from monai.networks.nets import UNet

import torch
import glob
import pydicom as dicom
import numpy as np

import nibabel as nib
from monai.transforms import (
    EnsureChannelFirst,
    Compose,
    Resize,
    ScaleIntensity
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


def get_pixels_hu(scans):
    image = np.stack([s.pixel_array for s in scans])
    image = image.astype(np.int32)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope

    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)

    image += np.int32(intercept)
    # 1000 = 1  > 500, hist
    return np.array(image, dtype=np.int16)


def load_sample(exam_path):
    slices_exam = glob.glob(exam_path + '/**')
    d = {sl_exam: int(sl_exam.split('/')[-1].split('-')[-1][:-4]) for sl_exam in slices_exam}
    sorted_dict = {k: v for k, v in sorted(d.items(), key=lambda item: item[1])}

    slices = [dicom.dcmread(sl) for sl in sorted_dict.keys()]
    # Hounsfield units
    hu_slices = get_pixels_hu(slices)
    return hu_slices


def load_sample_more_details(exam_path):
    slices_exam = glob.glob(exam_path + '/**')
    d = {sl_exam: int(sl_exam.split('/')[-1].split('-')[-1][:-4]) for sl_exam in slices_exam}
    sorted_dict = {k: v for k, v in sorted(d.items(), key=lambda item: item[1])}
    slices_full = [dicom.dcmread(sl) for sl in sorted_dict.keys()]

    # Find two consecutive slices to determine z spacing
    inst_num = [sl.InstanceNumber for sl in slices_full]
    i = 0
    for el in range(len(inst_num)):
        if inst_num[1] - inst_num[el] == 1:
            i = el

    first_scan = dicom.dcmread(list(sorted_dict.keys())[i])
    second_scan = dicom.dcmread(list(sorted_dict.keys())[1])
    z_spacing = abs(second_scan.ImagePositionPatient[-1] - first_scan.ImagePositionPatient[-1])

    return len(slices_full), first_scan, z_spacing


def preprocess(path_to_scan):
    # roi = nib.load(path_to_scan).get_fdata()
    orig_roi = load_sample(path_to_scan)

    roi = array_to_tensor(orig_roi)
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

    return roi, orig_roi


def get_model(spatial_dims=2, num_in_channels=1):
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
