import os
import numpy as np
import pydicom as dicom
import glob
import nibabel as nib
import dicom2nifti
import pandas as pd
import yaml


def get_first_scan(path_patient):
    all_scans = glob.glob(path_patient + '/**')
    years = [int(i.split('\\')[-1].split('-')[2]) for i in all_scans]
    first_year = np.sort(years)[0]
    scan = [el for el in all_scans if '-' + str(first_year) + '-' in el][0]
    return scan


def load_sample(exam_path):
    slices_exam = glob.glob(exam_path + '/**')
    d = {sl_exam: int(sl_exam.split('\\')[-1].split('-')[-1][:-4]) for sl_exam in slices_exam}
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
    input_dir = r"C:\Users\pps21\Documents\Cornell\data\NLST Raw Datasets"
    save_dir = r"C:\Users\pps21\Documents\Cornell\data\NLST_nifti"

    patients_cases = glob.glob(input_dir + '/**/**')
    failed_patients = []
    diacom_path = dict()
    t = {}
    for case in patients_cases:
        # Create directory for a patient
        case_id = case.split('\\')[-1]
        t.update({case_id: []})

        path_to_save = save_dir + '/' + case_id
        scan_path = get_first_scan(case)
        scans = glob.glob(scan_path + '/**')

        # Check number of dicoms
        scans = [s for s in scans if len(glob.glob(s + '/**')) > 10]
        sc = [len(glob.glob(scan + '/**')) for scan in scans]
        maxvalues_index = np.argwhere(sc == np.amax(sc)).flatten().tolist()
        scans = [scans[i] for i in maxvalues_index]
 
        if len(scans) != 1:
            # Pick the correct Hounsfield units
            filtered_scans = []

            for scan in scans:
                slice_test = glob.glob(scan + '/**')[0]
                dc = dicom.dcmread(slice_test)
                t[case_id].append((dc.WindowWidth, dc.WindowCenter))

                try:
                    if int(dc.WindowWidth[0]) == 400 or int(dc.WindowWidth[0]) == 350:
                        filtered_scans.append(scan)
                except:
                    if int(dc.WindowWidth) == 400 or int(dc.WindowWidth) == 350:
                        filtered_scans.append(scan)

            # If both scans have the same width, center, then save the one with more slices
            if t[case_id][0] == t[case_id][1]:
                sc = [len(glob.glob(scan + '/**')) for scan in scans]
                filtered_scans.append(scans[np.argmax(sc)])
            scans = filtered_scans

        if len(scans) == 0:
            failed_patients.append(case_id)
        else:
            chosen_scan = scans[0]
            diacom_path[case_id] = chosen_scan

            # Opens the DIACOM scans and convert directly to NIFTI
            scan_path_save = path_to_save + '.nii'
            # dicom2nifti.dicom_series_to_nifti(chosen_scan, scan_path_save, reorient_nifti=True)

            # # Open the scans and save to NIFTI (from intermediary array)
            # sample, intercept, slope = load_sample(chosen_scan)
            # array = np.array(sample, dtype=np.float32)

            # array = covert_HU(array, intercept, slope)

            # affine = np.eye(4)
            # nifti_file = nib.Nifti1Image(array, affine)

            # scan_path_save = path_to_save + '.nii'
            # nib.save(nifti_file, scan_path_save)

    # for c in failed_patients:
    #     print(c, t[c])
    print(len(failed_patients))

    # Export used DIACOM path
    diacom_path = {int(key): value.split('\\')[-2] + '/' + value.split('\\')[-1]  for key, value in diacom_path.items()}
    with open(r"C:\Users\pps21\Documents\Cornell\data\path_diacom.yaml", 'w') as file:
        yaml.dump(diacom_path, file)

    ## Compare diacom to previous study
    with open(r'C:\Users\pps21\Documents\Cornell\HeartTransplant\scan-mask-matches.yaml', 'r') as file:
        diacom_path_maria = yaml.safe_load(file)

    dict1=diacom_path_maria
    dict2=diacom_path

    # Get the union of all keys
    keys = set(dict1.keys()).union(set(dict2.keys()))

    # Create a DataFrame
    df = pd.DataFrame({
        'key': list(keys),
        'value1': [dict1.get(key, np.nan) for key in keys],
        'value2': [dict2.get(key, np.nan) for key in keys],
    })

    # Add a flag column
    df['flag'] = df['value1'] == df['value2']
    df.columns = ['subject_id', 'prev_study', 'new_study', 'is_equal']
    df.to_csv(r"C:\Users\pps21\Documents\Cornell\data\compare_diacom_path.csv", index=False)