import collections
import numpy as np
import nibabel as nib


def data_to_slices(data):
    patient_slices_dict = []

    for patient_path, label_path in data.items():
        annotation = nib.load(label_path).get_fdata()

        # Transpose and flip order because of nii format
        annotation = np.flip(annotation, -1)
        annotation = np.transpose(annotation, [1, 0, 2])

        for slice_ in range(annotation.shape[-1]):
            if np.sum(annotation[:, :, slice_]) > 0:
                patient_slices_dict.append((patient_path, label_path, annotation.shape[-1] -slice_-1))

    return patient_slices_dict


if __name__ == '__main__':
    data_dict = {
        '/Users/mariadobko/Documents/Cornell/LAB/NLST First 60 Raw/NLST First 60 Raw - Part01 - 10Pats/100004/'
        '01-02-1999-NLST-LSS-63991/1.000000-0OPAGELSPLUSD4102.512080.00.10.75-24639':
            '/Users/mariadobko/Downloads/Annotations - VT/100004.nii'
    }

    data_dict = collections.OrderedDict(data_dict)
    result = data_to_slices(data_dict)
    patient_dir, label_path, slice_id = result[0]

    print(patient_dir, label_path, slice_id)
