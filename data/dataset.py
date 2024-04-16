from typing import Dict, Any
import glob
import random
from collections import namedtuple

import torch
from albumentations.core.transforms_interface import DualTransform
import albumentations

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
    return torch.tensor(vol)


class CustomCutout(DualTransform):
    """
    Custom Cutout augmentation with handling of bounding boxes
    Note: (only supports square cutout regions)

    Author: Kaushal28
    Reference: https://arxiv.org/pdf/1708.04552.pdf
    """

    def __init__(
            self,
            fill_value=0,
            bbox_removal_threshold=0.50,
            min_cutout_size=192,
            max_cutout_size=512,
            always_apply=False,
            p=0.5
    ):
        """
        Class construstor

        :param fill_value: Value to be filled in cutout (default is 0 or black color)
        :param bbox_removal_threshold: Bboxes having content cut by cutout path more than this threshold will be removed
        :param min_cutout_size: minimum size of cutout (192 x 192)
        :param max_cutout_size: maximum size of cutout (512 x 512)
        """
        super(CustomCutout, self).__init__(always_apply, p)  # Initialize parent class
        self.fill_value = fill_value
        self.bbox_removal_threshold = bbox_removal_threshold
        self.min_cutout_size = min_cutout_size
        self.max_cutout_size = max_cutout_size

    def _get_cutout_position(self, img_height, img_width, img_depth, cutout_size):
        """
        Randomly generates cutout position as a named tuple

        :param img_height: height of the original image
        :param img_width: width of the original image
        :param cutout_size: size of the cutout patch (square)
        :returns position of cutout patch as a named tuple
        """
        position = namedtuple('Point', 'x y z')

        # mask_region - [50:360, 140:450, :]
        return position(
            np.random.randint(50, 360),
            np.random.randint(140, 450),
            np.random.randint(0, img_depth - cutout_size + 1)
        )

    def _get_cutout(self, img_height, img_width, img_depth):
        """
        Creates a cutout pacth with given fill value and determines the position in the original image

        :param img_height: height of the original image
        :param img_width: width of the original image
        :returns (cutout patch, cutout size, cutout position)
        """
        cutout_size = np.random.randint(self.min_cutout_size, self.max_cutout_size + 1)
        cutout_position = self._get_cutout_position(img_height, img_width, img_depth, cutout_size)
        return np.full((cutout_size, cutout_size, cutout_size), self.fill_value), cutout_size, cutout_position

    def apply(self, image, **params):
        """
        Applies the cutout augmentation on the given image

        :param image: The image to be augmented
        :returns augmented image
        """
        image = image.copy()  # Don't change the original image
        self.img_height, self.img_width, img_depth = image.shape
        cutout_arr, cutout_size, cutout_pos = self._get_cutout(self.img_height, self.img_width, img_depth)

        # Set to instance variables to use this later
        self.image = image
        self.cutout_pos = cutout_pos
        self.cutout_size = cutout_size

        image[cutout_pos.y:cutout_pos.y + cutout_size, cutout_pos.x:cutout_size + cutout_pos.x, cutout_pos.z:cutout_size + cutout_pos.z] = cutout_arr

        return image

    # def apply_to_bbox(self, bbox, **params):
    #     """
    #     Removes the bounding boxes which are covered by the applied cutout
    #
    #     :param bbox: A single bounding box coordinates in pascal_voc format
    #     :returns transformed bbox's coordinates
    #     """
    #
    #     # Denormalize the bbox coordinates
    #     bbox = denormalize_bbox(bbox, self.img_height, self.img_width)
    #     x_min, y_min, x_max, y_max = tuple(map(int, bbox))
    #
    #     bbox_size = (x_max - x_min) * (y_max - y_min)  # width * height
    #     overlapping_size = np.sum(
    #         (self.image[y_min:y_max, x_min:x_max, 0] == self.fill_value) &
    #         (self.image[y_min:y_max, x_min:x_max, 1] == self.fill_value) &
    #         (self.image[y_min:y_max, x_min:x_max, 2] == self.fill_value)
    #     )
    #
    #     # Remove the bbox if it has more than some threshold of content is inside the cutout patch
    #     if overlapping_size / bbox_size > self.bbox_removal_threshold:
    #         return normalize_bbox((0, 0, 0, 0), self.img_height, self.img_width)
    #
    #     return normalize_bbox(bbox, self.img_height, self.img_width)

    def get_transform_init_args_names(self):
        """
        Fetches the parameter(s) of __init__ method
        :returns: tuple of parameter(s) of __init__ method
        """
        return ('fill_value', 'bbox_removal_threshold', 'min_cutout_size', 'max_cutout_size', 'always_apply', 'p')


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


class NLST_Reconstruct_Dataset(torch.utils.data.Dataset):
    def __init__(self, patients_paths, target_size=None, transform=None, eval_mode=False, crop_heart=False):
        self.patients_paths = patients_paths
        self.target_size = tuple(target_size)
        self.transform = transform
        self.evaluate_mode = eval_mode

    def __len__(self):
        return len(self.patients_paths)

    def __getitem__(self, idx) -> Dict[str, Any]:
        patient_path, label_path = list(self.patients_paths.items())[idx]
        patient_id = patient_path.split('/')[-1][:-4]
        roi = nib.load(patient_path).get_fdata()

        augmentation_cut = albumentations.Compose([
            CustomCutout(fill_value=0,
                         bbox_removal_threshold=0.50,
                         min_cutout_size=20,
                         max_cutout_size=60, always_apply=True, p=1)])

        mask = augmentation_cut(image=roi)['image']

        roi = array_to_tensor(roi)

        # Transform to 3D cube roi with same size for all dimensions
        seed = 1
        imtrans = Compose(
            [
                EnsureChannelFirst(),
                Resize(self.target_size)
            ]
        )
        imtrans.set_random_state(seed=seed)

        segtrans = Compose(
            [
                EnsureChannelFirst(),
                Resize(self.target_size)
            ]
        )
        segtrans.set_random_state(seed=seed)

        # Transformations / augmenting input
        if self.transform:
            aug = self.transform({'image': roi, 'label': mask})
            roi, mask = aug['image'], aug['label']

        roi = imtrans(roi)
        mask = segtrans(mask)

        # Pad depth dimension
        try:
            roi, original_roi = pad_volume(roi, self.target_size[0]).double(), pad_volume(mask, self.target_size[0])
        except:
            roi, original_roi = pad_volume(roi, self.target_size[0]), pad_volume(mask, self.target_size[0])

        return {'data': roi.double(), 'label': original_roi}


class NLST_NIFTI_Dataset(torch.utils.data.Dataset):
    def __init__(self, patients_paths, target_size=None, transform=None, eval_mode=False, crop_heart=False):
        self.patients_paths = patients_paths
        self.target_size = tuple(target_size)
        self.transform = transform
        self.evaluate_mode = eval_mode
        self.crop_heart = crop_heart

    def __len__(self):
        return len(self.patients_paths)

    def __getitem__(self, idx) -> Dict[str, Any]:
        patient_path, label_path = list(self.patients_paths.items())[idx]
        patient_id = patient_path.split('/')[-1][:-4]
        # patient_id = patient_path.split('\\')[-1][:6]
        roi = nib.load(patient_path).get_fdata()

        roi = array_to_tensor(roi)
        label = nib.load(label_path).get_fdata()
        original_z = label.shape[-1]
        if int(patient_id) not in [100108, 100085, 100088, 100092, 100019, 100031, 100046, 100053, 100072, 100081]:
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

        # Transformations / augmenting input
        if self.transform:
            aug = self.transform({'image': roi, 'label': mask})
            roi, mask = aug['image'], aug['label']

        roi = imtrans(roi)
        mask = segtrans(mask)

        # Pad depth dimension
        try:
            roi, mask = pad_volume(roi, self.target_size[0]).double(), pad_volume(mask, self.target_size[0])
        except:
            roi, mask = pad_volume(roi, self.target_size[0]), pad_volume(mask, self.target_size[0])

        if self.evaluate_mode:
            return {'data': roi, 'label': mask, 'patient_dir': patient_path, 'original_z':original_z}
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

    def __getitem__(self, idx) -> Dict[str, Any]:
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

    def __getitem__(self, idx) -> Dict[str, Any]:
        patient_dir, label_path, slice_id = self.patients_paths[idx]
        patient_id = patient_dir.split('/')[-1][:-4]
        roi = self.load_slice(patient_dir, slice_id)
        roi = array_to_tensor(roi)

        label = nib.load(label_path).get_fdata()
        if int(patient_id) not in [100108, 100085, 100088, 100092, 100019, 100031, 100046, 100053, 100072, 100081]:
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

    def __getitem__(self, idx) -> Dict[str, Any]:
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

    def __getitem__(self, idx) -> Dict[str, Any]:

        patient_dir, label_path, slice_id = self.patients_paths[idx]
        patient_id = patient_dir.split('/')[-1][:-4]
        roi = self.load_slices_2_5(patient_dir, slice_id, self.num_window_slices)
        roi = array_to_tensor(roi)

        label = nib.load(label_path).get_fdata()
        if int(patient_id) not in [100108, 100085, 100088, 100092, 100019, 100031, 100046, 100053, 100072, 100081]:
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


def data_to_slices(data, normal_sample=False, nifti=False):
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
            else:
                if normal_sample and slice_ >=2 and slice_<= annotation.shape[-1]-3 and slice_ % 3 == 0:
                    # Include slices that don't have annotations - non-heart
                    if nifti:
                        patient_slices.append((patient_path, label_path, slice_))
                    else:
                        patient_slices.append((patient_path, label_path, annotation.shape[-1] - slice_ - 1))

    return patient_slices


if __name__ == '__main__':
    # Dictionary with directories to files in the format: {patient_dir} : {label_path}

    raw_directory = '/athena/sablab/scratch/mdo4009/NLST_nifti_manual/'
    annotations_dir = '/athena/sablab/scratch/mdo4009/Annotations - VT/'

    data_dict = {}
    for raw_sample in glob.glob(raw_directory + '**'):
        sample_id = raw_sample.split('/')[-1][:-4]
        for label_path in glob.glob(annotations_dir + '**'):
            if sample_id in label_path and '.nii' in label_path:
                data_dict.update({raw_sample: label_path})

    train_dataset = NLST_NIFTI_Dataset(
        patients_paths=data_dict,
        target_size=[256, 256, 256],
        transform=None
    )

    r = train_dataset[0]
