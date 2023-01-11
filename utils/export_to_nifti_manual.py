import os
import numpy as np
import pydicom as dicom
import glob
import nibabel as nib
import yaml


def get_first_scan(path_patient):
    all_scans = glob.glob(path_patient + '/**')
    years = [int(i.split('/')[-1].split('-')[2]) for i in all_scans]
    first_year = np.sort(years)[0]
    scan = [el for el in all_scans if '-' + str(first_year) + '-' in el][0]
    return scan


def load_sample(exam_path):
    slices_exam = glob.glob(exam_path + '/**')
    d = {sl_exam: int(sl_exam.split('/')[-1].split('-')[-1][:-4]) for sl_exam in slices_exam}
    sorted_dict = {k: v for k, v in sorted(d.items(), key=lambda item: item[1])}
    slices = [dicom.dcmread(sl).pixel_array for sl in sorted_dict.keys()]

    one_scan = dicom.dcmread(list(sorted_dict.keys())[0])
    intercept = one_scan.RescaleIntercept
    slope = one_scan.RescaleSlope

    return np.stack(slices, -1), intercept, slope


def covert_HU(image, intercept, slope):
    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)

    image += np.int32(intercept)
    # 1000 = 1  > 500, hist
    return np.array(image, dtype=np.int16)



if __name__ == '__main__':
    input_dir = '/Users/mariadobko/Documents/Cornell/LAB/NLST First 60 Raw'
    save_dir = '/Users/mariadobko/Documents/Cornell/LAB/NLST_nifti_manual'

    matches = '/Users/mariadobko/PycharmProjects/HeartTransplant/scan-mask-matches.yaml'
    matches_dict = yaml.safe_load(open(matches))

    patients_cases = glob.glob(input_dir + '/**/**')
    t = {}

    for case in patients_cases:
        # Create directory for a patient
        case_id = case.split('/')[-1]
        t.update({case_id: []})
        s_path = matches_dict[int(case_id)]

        scan_path = os.path.join(case, s_path)
        scans = glob.glob(scan_path + '/**')

        path_to_save = save_dir + '/' + case_id

        # Pick the correct Hounsfield units
        filtered_scans = []

        # Open the slices and save to NIFTI
        sample, intercept, slope = load_sample(scan_path)
        array = np.array(sample, dtype=np.float32)

        array = covert_HU(array, intercept, slope)

        affine = np.eye(4)
        nifti_file = nib.Nifti1Image(array, affine)

        scan_path_save = path_to_save + '.nii'
        nib.save(nifti_file, scan_path_save)
