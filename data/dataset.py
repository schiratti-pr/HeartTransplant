from typing import Dict, Any
import collections
import glob

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
)


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
    return vol


def array_to_tensor(img_array) -> torch.FloatTensor:
    return torch.FloatTensor(img_array / 255)


class NLSTDataset(torch.utils.data.Dataset):
    """
    Dataset for loading cardiac CTs
    Loads full patient volume per each CT
    """

    def __init__(self, patients_paths, target_size=None, transform=None):
        self.patients_paths = patients_paths
        self.target_size = tuple(target_size)
        self.transform = transform

    def __len__(self):
        return len(self.patients_paths)

    @staticmethod
    def load_sample(exam_path):
        slices_exam = glob.glob(exam_path + '/**')
        d = {sl_exam: int(sl_exam.split('/')[-1].split('-')[-1][:-4]) for sl_exam in slices_exam}
        sorted_dict = {k: v for k, v in sorted(d.items(), key=lambda item: item[1])}

        slices = [dicom.dcmread(sl).pixel_array for sl in sorted_dict.keys()]
        return np.stack(slices, -1)

    def __getitem__(self, idx) -> dict[str, Any]:
        patient_dir, label_path = list(self.patients_paths.items())[idx]
        roi = self.load_sample(patient_dir)

        roi = array_to_tensor(roi)
        mask = torch.DoubleTensor(nib.load(label_path).get_fdata())

        # Transform to 3D cube roi with same size for all dimensions
        seed = 1
        imtrans = Compose(
            [
                ScaleIntensity(),
                EnsureChannelFirst(),
                RandSpatialCrop(self.target_size, random_size=False),
            ]
        )
        imtrans.set_random_state(seed=seed)

        segtrans = Compose(
            [
                EnsureChannelFirst(),
                RandSpatialCrop(self.target_size, random_size=False),
            ]
        )
        segtrans.set_random_state(seed=seed)

        roi = imtrans(roi)
        mask = segtrans(mask)

        # Pad depth dimension
        roi, mask = pad_volume(roi,  self.target_size[0]), pad_volume(mask, self.target_size[0])

        return {'data': roi, 'label': mask}


if __name__ == '__main__':
    # Dictionary with directories to files in the format: {patient_dir} : {label_path}
    data_dict = {
        '/Users/mariadobko/Documents/Cornell/LAB/NLST First 60 Raw/NLST First 60 Raw - Part01 - 10Pats/100004/'
        '01-02-1999-NLST-LSS-63991/1.000000-0OPAGELSPLUSD4102.512080.00.10.75-24639':
            '/Users/mariadobko/Documents/Segmentation-Segment_1-label.nii'
    }

    data_dict = collections.OrderedDict(data_dict)

    ds = NLSTDataset(
        patients_paths=data_dict,
        target_size=[256, 256, 256],
    )
    test_roi, test_mask = ds[0]['data'], ds[0]['label']
    print(test_roi.shape, test_mask.shape)
