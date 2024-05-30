import os
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
import glob
import torch

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

# def pad_volume(vol, target_size):
#     """
#     Pad every dimension of the numpy array to reach the specified dimension.
#     Padding the volume of size (W, H, D) to fit into a volume of size (target_size, target_size, target_size)
#     """
#     pad_shape = [(0, target_size - vol.shape[i]) if vol.shape[i] < target_size else (0, 0) for i in range(3)]
#     vol = np.pad(vol, pad_width=pad_shape, mode='constant', constant_values=0)
#     return vol

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

# Define the directory where the NIfTI files are located
img_dir = r"C:\Users\pps21\Documents\Cornell\data\NLST_nifti"
# mask_dir = r"C:\Users\pps21\Box\NLST Chest CT\Annotations - VT"
mask_dir = r"C:\Users\pps21\Documents\Cornell\data\Annotations - VT - Intra-user fix"
dest_dir = r"C:\Users\pps21\Documents\Cornell\data\Annotations - VT - Intra-user fix flip"

# Import the CSV file with flipped flag
csv_file_path = r"C:\Users\pps21\Documents\Cornell\data\flip_nifti_intra.csv"
df = pd.read_csv(csv_file_path)

# Get dictionary of img/mask pairs
data_dict = {}
for img_path in glob.glob(img_dir + '/*')[:]:
    subject_id = img_path.split('\\')[-1][:6]
    for mask_path in glob.glob(mask_dir + '/*'):
        if subject_id in mask_path and '.nii' in mask_path:
            data_dict.update({img_path: mask_path})
            s
# Create a figure for the subplots
fig, axs = plt.subplots(len(data_dict), 6, figsize=(30, len(data_dict)*5))

# Iterate over each file in the directory
for i, (img_path, mask_path) in enumerate(data_dict.items()):

    # Extract the subject ID from the file name
    subject_id = img_path.split('\\')[-1][:6]
    
    img, mask = transform_img(img_path, mask_path, crop_heart=False)
    
    # Get the middle frame of each dimension before flipping
    middle_frames_img, middle_frames_mask = get_middle_frame(img, mask)
    
    # Plot the middle frames before flipping
    for j in range(3):
        axs[i, j].imshow(middle_frames_img[j], cmap='gray')
        axs[i, j].imshow(middle_frames_mask[j], cmap='jet', alpha=0.5)
        axs[i, j].set_title(subject_id + ' before')
        axs[i, j].axis('off')
    
    # Check if the subject ID has a boolean 1 in the CSV file
    flip_flag = df.loc[df['subject_id'] == int(subject_id), 'flipped'].any()
    mask_original = nib.load(mask_path) # without any transformation, just flip
    mask_numpy = mask_original.get_fdata()
    if flip_flag:
        # Flip the array along the 3rd dimension
        flipped_data = np.flip(mask_numpy, axis=2).copy()
    else:
        flipped_data = mask_numpy
    
    # Convert the flipped array back to a NIfTI object
    flipped_nifti_obj = nib.Nifti1Image(flipped_data, mask_original.affine, mask_original.header)
    
    # Export the flipped NIfTI object to a new NIfTI file
    file_name = f'{subject_id}_fix_flip.nii'
    flipped_file_path = os.path.join(dest_dir, file_name)
    nib.save(flipped_nifti_obj, flipped_file_path)
    
    # Re-import the saved NIfTI file
    img, reimported_mask = transform_img(img_path, flipped_file_path, crop_heart=False)
    
    # reimported_data = flipped_data
    # Get the middle frame of each dimension after flipping
    middle_frames_img, middle_frames_mask_after = get_middle_frame(img, reimported_mask)
    
    # Plot the middle frames after flipping
    for j in range(3):
        axs[i, 3 + j].imshow(middle_frames_img[j], cmap='gray')
        axs[i, 3 + j].imshow(middle_frames_mask_after[j], cmap='jet', alpha=0.5)
        axs[i, 3 + j].set_title(subject_id + ' after' + ' flipped' if flip_flag else ' after')
        axs[i, 3 + j].axis('off')

# Display the figure
plt.tight_layout()
plt.show()