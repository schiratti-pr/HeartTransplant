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
    CenterSpatialCrop
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
    return torch.FloatTensor(img_array)


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


class NLST_NIFTI_Dataset(torch.utils.data.Dataset):
    def __init__(self, patients_paths, target_size=None, transform=None, eval_mode=False, crop_heart=False):
        self.patients_paths = patients_paths
        self.target_size = tuple(target_size)
        self.transform = transform
        self.evaluate_mode = eval_mode
        self.crop_heart = crop_heart

    def __len__(self):
        return len(self.patients_paths)

    def __getitem__(self, idx) -> dict[str, Any]:
        patient_path, label_path = list(self.patients_paths.items())[idx]
        roi = nib.load(patient_path).get_fdata()

        roi = array_to_tensor(roi)
        label = nib.load(label_path).get_fdata()
        label = np.flip(label, -1).copy()
        label = np.transpose(label, [1, 0, 2])

        mask = torch.Tensor(label)

        if self.crop_heart:
            # Crop the region around the heart based on stats - [81:337, 158:430, :], adding padding of minimum 20
            roi = roi[50:360, 140:450, :]

        # Transform to 3D cube roi with same size for all dimensions
        seed = 1
        imtrans = Compose(
            [
                ScaleIntensity(),
                EnsureChannelFirst(),
                Resize(self.target_size)
                # CenterSpatialCrop(self.target_size),  # todo: limit the random crop
            ]
        )
        imtrans.set_random_state(seed=seed)

        segtrans = Compose(
            [
                EnsureChannelFirst(),
                Resize(self.target_size)
                # CenterSpatialCrop(self.target_size),
            ]
        )
        segtrans.set_random_state(seed=seed)

        roi = imtrans(roi)
        mask = segtrans(mask)

        # Pad depth dimension
        try:
            roi, mask = pad_volume(roi, self.target_size[0]).double(), pad_volume(mask, self.target_size[0])
        except:
            roi, mask = pad_volume(roi, self.target_size[0]), pad_volume(mask, self.target_size[0])

        # Transformations / augmenting input
        if self.transform:
            aug = self.transform({'image': roi, 'label': mask})
            roi, mask = aug['image'], aug['label']

        if self.evaluate_mode:
            return {'data': roi, 'label': mask, 'patient_dir': patient_path}
        else:
            return {'data': roi.double(), 'label': mask}


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

        slices = [dicom.dcmread(sl) for sl in sorted_dict.keys()]
        # Hounsfield units
        hu_slices = get_pixels_hu(slices)
        return hu_slices

    def __getitem__(self, idx) -> dict[str, Any]:
        patient_dir, label_path = list(self.patients_paths.items())[idx]
        roi = self.load_sample(patient_dir)

        roi = array_to_tensor(roi)
        label = nib.load(label_path).get_fdata()
        label = np.flip(label, -1).copy()
        label = np.transpose(label, [1, 0, 2])

        mask = torch.DoubleTensor(label)

        # Transform to 3D cube roi with same size for all dimensions
        seed = 1
        imtrans = Compose(
            [
                ScaleIntensity(),
                EnsureChannelFirst(),
                RandSpatialCrop(self.target_size, random_size=False), # todo: limit the random crop
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


class NLST_2D_NIFTI_Dataset(torch.utils.data.Dataset):

    def __init__(self, patients_paths, target_size=None, transform=None, eval_mode=False):
        self.patients_paths = patients_paths
        self.target_size = target_size
        self.transform = transform
        self.evaluate_mode = eval_mode

    def __len__(self):
        return len(self.patients_paths)

    @staticmethod
    def load_slice(exam_path, slice_id):
        image = nib.load(exam_path).get_fdata()[:, :, slice_id]

        return image

    def __getitem__(self, idx) -> dict[str, Any]:
        patient_dir, label_path, slice_id = self.patients_paths[idx]
        roi = self.load_slice(patient_dir, slice_id)
        roi = array_to_tensor(roi)

        label = nib.load(label_path).get_fdata()
        label = np.flip(label, -1).copy()
        label = label[:, :, slice_id].T

        mask = torch.DoubleTensor(label)

        # Transform to 3D cube roi with same size for all dimensions
        seed = 1
        imtrans = Compose(
            [
                ScaleIntensity(),
                EnsureChannelFirst(),
                CenterSpatialCrop(self.target_size),   # todo: change to a limited random crop
            ]
        )
        imtrans.set_random_state(seed=seed)

        segtrans = Compose(
            [
                EnsureChannelFirst(),
                CenterSpatialCrop(self.target_size),
            ]
        )
        segtrans.set_random_state(seed=seed)

        roi = imtrans(roi)
        mask = segtrans(mask)

        # Transformations / augmenting input
        if self.transform:
            aug = self.transform({'image': roi, 'label': mask})
            roi, mask = aug['image'], aug['label']

        if self.evaluate_mode:
            return {'data': roi, 'label': mask, 'patient_dir': patient_dir, 'slice_id': slice_id}
        else:
            return {'data': roi, 'label': mask}


class NLST_2D_Dataset(torch.utils.data.Dataset):
    """
    Dataset for loading cardiac CTs
    Loads slices - 2D images
    """

    def __init__(self, patients_paths, target_size=None, transform=None):
        self.patients_paths = patients_paths
        self.target_size = target_size
        self.transform = transform

    def __len__(self):
        return len(self.patients_paths)

    @staticmethod
    def load_slice(exam_path, slice_id):
        slices_exam = glob.glob(exam_path + '/**')
        slice_path = [sl_path for sl_path in slices_exam if str(slice_id) + '.dcm' in sl_path][0]
        slice = dicom.dcmread(slice_path)

        image = slice.pixel_array
        image = image.astype(np.int16)
        image[image == -2000] = 0

        # Convert to Hounsfield units (HU)
        intercept = slice.RescaleIntercept
        slope = slice.RescaleSlope

        if slope != 1:
            image = slope * image.astype(np.float64)
            image = image.astype(np.int32)
        image += np.int32(intercept)

        return image

    def __getitem__(self, idx) -> dict[str, Any]:
        patient_dir, label_path, slice_id = self.patients_paths[idx]
        roi = self.load_slice(patient_dir, slice_id)
        roi = array_to_tensor(roi)

        label = nib.load(label_path).get_fdata()
        label = np.flip(label, -1).copy()
        label = label[:, :, slice_id].T

        mask = torch.DoubleTensor(label)

        # Transform to 3D cube roi with same size for all dimensions
        seed = 1
        imtrans = Compose(
            [
                ScaleIntensity(),
                EnsureChannelFirst(),
                CenterSpatialCrop(self.target_size),   # todo: change to a limited random crop
            ]
        )
        imtrans.set_random_state(seed=seed)

        segtrans = Compose(
            [
                EnsureChannelFirst(),
                CenterSpatialCrop(self.target_size),
            ]
        )
        segtrans.set_random_state(seed=seed)

        roi = imtrans(roi)
        mask = segtrans(mask)

        # Transformations / augmenting input
        sample = {'data': roi, 'label': mask}
        if self.transform:
            sample = self.transform(sample)

        return sample


class NLST_2_5D_Dataset(torch.utils.data.Dataset):
    """
    Dataset for loading cardiac CTs
    Loads slices - 2.5D images
    """
    def __init__(self, patients_paths, target_size=None, transform=None, eval_mode=False, crop_heart=False,
                 window_step=2):
        self.patients_paths = patients_paths
        self.target_size = list(target_size)
        self.transform = transform
        self.evaluate_mode = eval_mode
        self.crop_heart = crop_heart
        self.num_window_slices = window_step

    def __len__(self):
        return len(self.patients_paths)

    @staticmethod
    def load_slices_2_5(exam_path, slice_id, window_step):
        image = nib.load(exam_path).get_fdata()[:, :, slice_id - window_step : slice_id + window_step + 1]

        return image

    def __getitem__(self, idx) -> dict[str, Any]:

        patient_dir, label_path, slice_id = self.patients_paths[idx]
        roi = self.load_slices_2_5(patient_dir, slice_id, self.num_window_slices)
        roi = array_to_tensor(roi)

        label = nib.load(label_path).get_fdata()
        label = np.flip(label, -1).copy()
        label = label[:, :, slice_id].T

        mask = torch.DoubleTensor(label)

        if self.crop_heart:
            # Crop the region around the heart based on stats - [81:337, 158:430, :], adding padding of minimum 20
            roi = roi[50:360, 140:450, :]

        # Transform to 3D cube roi with same size for all dimensions
        seed = 1
        imtrans = Compose(
            [
                ScaleIntensity(),
                EnsureChannelFirst(),
                CenterSpatialCrop(self.target_size + [self.num_window_slices*2 + 1]),
            ]
        )
        imtrans.set_random_state(seed=seed)

        segtrans = Compose(
            [
                EnsureChannelFirst(),
                CenterSpatialCrop(self.target_size),
            ]
        )
        segtrans.set_random_state(seed=seed)

        roi = imtrans(roi).squeeze().permute(2, 0, 1)
        mask = segtrans(mask)

        # Transformations / augmenting input
        if self.transform:
            aug = self.transform({'image': roi, 'label': mask})
            roi, mask = aug['image'], aug['label']

        if self.evaluate_mode:
            return {'data': roi, 'label': mask, 'patient_dir': patient_dir, 'slice_id': slice_id}
        else:
            return {'data': roi, 'label': mask}


def data_to_slices(data, nifti=False):
    patient_slices = []

    for patient_path, label_path in data.items():
        annotation = nib.load(label_path).get_fdata()

        # Transpose and flip order because of nii format
        annotation = np.flip(annotation, -1)
        annotation = np.transpose(annotation, [1, 0, 2])

        for slice_ in range(annotation.shape[-1]):
            if np.sum(annotation[:, :, slice_]) > 0:
                if nifti:
                    patient_slices.append((patient_path, label_path, slice_))
                else:
                    patient_slices.append((patient_path, label_path, annotation.shape[-1] -slice_-1))

    return patient_slices


if __name__ == '__main__':
    # Dictionary with directories to files in the format: {patient_dir} : {label_path}
    data_dict = {
        '/Users/mariadobko/Documents/Cornell/Lab/NLST First 60 Raw/NLST First 60 Raw - Part01 - 10Pats/100005/'
        '01-02-1999-NLST-LSS-72969/2.000000-0OPAGELS16B3702.514060.00.11.375-43455':
            '/Users/mariadobko/Downloads/100005-label.nii'
    }

    data_dict = collections.OrderedDict(data_dict)

    ds = NLSTDataset(
        patients_paths=data_dict,
        target_size=[256, 256, 256],
    )
    test_roi, test_mask = ds[0]['data'], ds[0]['label']
    print('3D dataset', test_roi.shape, test_mask.shape)

    # For 2D images
    data_dict_2d = data_to_slices(data_dict)
    ds2d = NLST_2D_Dataset(
        patients_paths=data_dict_2d,
        target_size=[256, 256],
    )
    test_roi_slice, test_mask_slice = ds2d[0]['data'], ds2d[0]['label']
    print('2D dataset', test_roi_slice.shape, test_mask_slice.shape)

    # For 2.5D images
    data_dict_2d = data_to_slices(data_dict)
    ds2_5d = NLST_2_5D_Dataset(
        patients_paths=data_dict_2d,
        target_size=[256, 256],
        window_step=3,
    )
    test_roi_slice, test_mask_slice = ds2_5d[0]['data'], ds2_5d[0]['label']
    print('2.5-D dataset', test_roi_slice.shape, test_mask_slice.shape)

