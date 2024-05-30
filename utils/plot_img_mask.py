import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

# Assuming 'img' and 'mask' are your 3D image and mask
# Convert tensors to numpy arrays if they are not already

patient_path = r"/home/prs247/data/NLST_nifti_60/100124.nii"
label_path = r'/home/prs247/data/Annotations - VT/100124-label.nii'

roi = nib.load(patient_path).get_fdata()
label = nib.load(label_path).get_fdata()
label = np.flip(label, 1).copy()

img_np = roi
mask_np = label

# img_np = img.numpy()
# mask_np = mask.numpy()
# img_np = img_np.squeeze()
# mask_np = mask_np.squeeze()


# Get the middle index of each dimension
middle = [dim // 2 for dim in img_np.shape]

fig, axs = plt.subplots(3, 2, figsize=(10, 15))

# Plot axial slices
axs[0, 0].imshow(img_np[middle[0], :, :], cmap='gray')
axs[0, 0].imshow(mask_np[middle[0], :, :], cmap='jet', alpha=0.5)

# Plot coronal slices
axs[1, 0].imshow(img_np[:, middle[1], :], cmap='gray')
axs[1, 0].imshow(mask_np[:, middle[1], :], cmap='jet', alpha=0.5)

# Plot sagittal slices
axs[2, 0].imshow(img_np[:, :, middle[2]-15], cmap='gray')
axs[2, 0].imshow(mask_np[:, :, middle[2]-15], cmap='jet', alpha=0.5)

# Set titles
axs[0, 0].set_title('Axial Image')
axs[1, 0].set_title('Coronal Image')
axs[2, 0].set_title('Sagittal Image')

plt.show()