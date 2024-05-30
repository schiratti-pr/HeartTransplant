import os
import glob
import re
import pydicom
from itertools import chain
import pandas as pd
import numpy as np
import yaml

# Define a function to extract the year from the filename
def extract_year(filename):
    match = re.search(r'-([0-9]{4})-', filename)
    return int(match.group(1)) if match else 0

def get_sorted_scan(path_patient):
    all_scans = glob.glob(path_patient + '/**')
    sorted_scans = sorted(all_scans, key=extract_year)
    return sorted_scans

# Define the hierarchies
kernel_hierarchy = [
    {'STANDARD': 6, 'DETAIL': 5, 'SOFT': 4, 'BONE': 3, 'EDGE': 2, 'BONEPLUS': 1, 'LUNG': 0},
    {'E': 6, 'C': 5, 'D': 4},
    {'B30f': 6, 'B45f': 5, 'B50f': 4, 'B60f': 3, 'B20f': 2},
    {'FC24': 6, 'FC65': 5, 'FC50': 4, 'FC51': 2, 'FC30': 3},
]

def select_best_key(d, hierarchies):
    # Start with all keys
    candidate_keys = list(d.keys())

    # Level 1: Try to select the key with the strictly smallest slice thickness
    min_value_0 = min((d[k][0] for k in candidate_keys), default=None)
    candidate_keys = [k for k in candidate_keys if d[k][0] == min_value_0]

    # If we found a unique best key, return it
    if len(candidate_keys) == 1:
        return candidate_keys[0]

    # Level 2: Try to select the key with the strictly highest FOV
    max_value_1 = max((d[k][1] for k in candidate_keys), default=None)
    candidate_keys = [k for k in candidate_keys if d[k][1] == max_value_1]

    # If we found a unique best key, return it
    if len(candidate_keys) == 1:
        return candidate_keys[0]

    # Level 3: kernel
    for hierarchy in hierarchies:
        intersect_values = set(d[k][2] for k in candidate_keys).intersection(set(hierarchy.keys()))
        if intersect_values:
            best_value = max(intersect_values, key=lambda v: hierarchy.get(v, -1))
            candidate_keys = [k for k in candidate_keys if d[k][2] == best_value]

            # If we found a unique best key, return it
            if len(candidate_keys) == 1:
                return candidate_keys[0]

    # If no unique best key was found, return None
    return None

### Get all the dicom paths and their characteristics
# Find the diacom folders
input_dir = r"C:\Users\pps21\Documents\Cornell\data\NLST Raw Datasets"

patients_cases = glob.glob(input_dir + '/**/**')
# diacom_path = dict()
kernel_list = []
for case in patients_cases:
    subject_id = case.split('\\')[-1]
    scan_paths = get_sorted_scan(case)
    
    # c = 0
    # scan_used = []
    for scan_path in scan_paths:
        scans = glob.glob(scan_path + '/**')
        dicom_examples = {scan:[glob.glob(scan + '/**')[0], glob.glob(scan + '/**')[1], len(glob.glob(scan + '/**'))] for scan in scans if len(glob.glob(scan + '/**')) > 10}
        
        if len(dicom_examples) > 0:
            for scan, double_example in dicom_examples.items():
                
                # Get slice thickness
                dicom_1 = pydicom.read_file(double_example[0])
                dicom_2 = pydicom.read_file(double_example[1])
                slice_thickness = np.abs(dicom_1.ImagePositionPatient[2] - dicom_2.ImagePositionPatient[2])
                number_of_slices = double_example[2]
                fov = slice_thickness * number_of_slices
                
                if 'ConvolutionKernel' in dicom_1:
                    kernel = dicom_1.ConvolutionKernel
                else:
                    kernel = None
                dicom_examples[scan] = [slice_thickness, fov, kernel]
            # scan_used = select_best_key(dicom_examples, kernel_hierarchy)
            # dicom_examples[scan_used] = dicom_examples[scan_used] + [1]
            kernel_list.append(dicom_examples)
        # c+=1
    # diacom_path[subject_id] = scan_used

kernels = dict(chain.from_iterable(d.items() for d in kernel_list))
data = []
for path, dicom_attr in kernels.items():
    subject_id = path.split('\\')[-3]
    path = '\\'.join(path.split('\\')[-2:])
    data.append([subject_id, path] + dicom_attr)
df = pd.DataFrame(data, columns=['subject_id', 'path', 'slice_thickness', 'fov', 'kernel'])
df['year_batch'] = df['path'].str.split('-').str[2]
# df.scan_used = df.scan_used.fillna(0)
df_choices = df
df_choices.to_csv(r"C:\Users\pps21\Documents\Cornell\data\dicom_all_path.csv", index=False)

### Not in use
# # Duplicated kernels
# duplicated_subject_ids = df[df.duplicated(['subject_id', 'kernel'], keep=False)]['subject_id'].unique()
# df_dup = df[df['subject_id'].isin(duplicated_subject_ids)]

# # Compare diacom path and export results
# diacom_path = {int(key): value.split('\\')[-2] + '/' + value.split('\\')[-1]  for key, value in diacom_path.items()}
# # with open(r"C:\Users\pps21\Documents\Cornell\data\diacom_120.yaml", 'w') as file:
# #     yaml.dump(diacom_path, file)

# df = pd.read_csv(r"C:\Users\pps21\Documents\Cornell\data\compare_diacom_path.csv").drop(columns='is_equal')
# df['path_kernel_priority'] = df['subject_id'].map(diacom_path)
# df.columns = ['subject_id', 'path_prev', 'path_actual_seg', 'path_kernel_priority']

# df['is_diff_prev'] = ((df['path_prev'] == df['path_actual_seg']) | (df['path_prev'] == df['path_kernel_priority'])).astype(int)
# df['is_diff_actual_seg'] = ((df['path_actual_seg'] == df['path_prev']) | (df['path_actual_seg'] == df['path_kernel_priority'])).astype(int)
# df['is_diff_kernel_priority'] = ((df['path_kernel_priority'] == df['path_prev']) | (df['path_kernel_priority'] == df['path_actual_seg'])).astype(int)
# # df.to_csv(r"C:\Users\pps21\Documents\Cornell\data\diacom_path.csv", index=False)

#### Get the optimal dicom path
# Combination of rules: first actual seg, second better kernel with same slice thickness and fov

actual_seg = pd.read_csv(r"C:\Users\pps21\Documents\Cornell\data\dicom_path_actual_seg.csv")
all_path = pd.read_csv(r"C:\Users\pps21\Documents\Cornell\data\dicom_all_path.csv")

all_path['path'] = all_path['path'].str.replace('\\', '/')

# Define the hierarchies
kernel_hierarchy = [
    {'STANDARD': 6, 'DETAIL': 5, 'SOFT': 4, 'BONE': 3, 'EDGE': 2, 'BONEPLUS': 1, 'LUNG': 0},
    {'E': 6, 'C': 5, 'D': 4},
    {'B30f': 6, 'B45f': 5, 'B50f': 4, 'B60f': 3, 'B20f': 2, 'B31f':1},
    {'FC24': 6, 'FC65': 5, 'FC50': 4, 'FC51': 3, 'FC30': 2, 'FC01':1, 'FC02':0},
]

# Merge the two DataFrames on subject_id
merged = pd.merge(all_path, actual_seg, on='subject_id')

# Function to select the best path for each group
def select_best_path(group):
    
    # Find the hierarchy that contains the kernels in the group
    for hierarchy in kernel_hierarchy:
        if group.loc[group['path'] == group['path_actual_seg'].iloc[0], 'kernel'].iloc[0] in hierarchy:
            break
    else:
        return pd.Series(False, index=group.index)  # return False for all paths if no matching hierarchy is found

    # If the actual_seg path is in the group, keep it unless there's a better kernel
    if group['path_actual_seg'].iloc[0] in group['path'].values:
        
        actual_kernel = group.loc[group['path'] == group['path_actual_seg'].iloc[0], 'kernel'].iloc[0]
        actual_fov = group.loc[group['path'] == group['path_actual_seg'].iloc[0], 'fov'].iloc[0]
        actual_slices = group.loc[group['path'] == group['path_actual_seg'].iloc[0], 'slice_thickness'].iloc[0]
        actual_year = group.loc[group['path'] == group['path_actual_seg'].iloc[0], 'year_batch'].iloc[0]
        # Check if actual_kernel is in hierarchy
        better_kernel_exists = any(hierarchy.get(k, -1) > hierarchy[actual_kernel] for k in group['kernel'])
        same_fov_and_slices_year = (group['fov'] == actual_fov) & (group['slice_thickness'] == actual_slices) & (group['year_batch'] == actual_year)
        
        if better_kernel_exists and same_fov_and_slices_year.sum() > 1:
            best_path = group[same_fov_and_slices_year].loc[group.loc[same_fov_and_slices_year,'kernel'].map(hierarchy).idxmax(), 'path']
        else:
            best_path = group['path_actual_seg'].iloc[0]
    # If the actual_seg path is not in the group, select the path with the best kernel
    else:
        return pd.Series(False, index=group.index)

    # Return a Series with True for the chosen path and False for all other paths
    return pd.Series(group['path'] == best_path, index=group.index)

# Apply the function to each group and join the result to the merged DataFrame
merged['chosen'] = merged.groupby('subject_id').apply(select_best_path).reset_index(level=0, drop=True)
merged.chosen = merged.chosen.astype(int)
# merged.path_actual_seg = merged.path_actual_seg.str.replace('/', '\\')
merged['actual_seg'] = (merged['path'] == merged['path_actual_seg']).astype(int)

merged.to_csv(r"C:\Users\pps21\Documents\Cornell\data\dicom_choices.csv", index=False)

dicom_120 = merged[merged.chosen==1]
dicom_120_dict = dict(zip(dicom_120.subject_id, dicom_120.path))
with open(r"C:\Users\pps21\Documents\Cornell\data\dicom_120.yaml", 'w') as file:
    yaml.dump(dicom_120_dict, file)
