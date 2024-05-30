import os
import shutil
import zipfile
import glob
import xml.etree.ElementTree as ET
import numpy as np
import yaml
import pandas as pd
import re
import nrrd

# Define a function to extract the year from the filename
def extract_year(filename):
    match = re.search(r'-([0-9]{4})-', filename)
    return int(match.group(1)) if match else 0

def get_sorted_scan(path_patient):
    all_scans = glob.glob(path_patient + '/**')
    sorted_scans = sorted(all_scans, key=extract_year)
    return sorted_scans

# Define source and destination directories
source_dir = r"C:\Users\pps21\Box\NLST Chest CT\Annotations - VT"
dest_dir = r"C:\Users\pps21\Documents\Cornell\data\temp_mrb"

# Get all .mrb files in the source directory
mrb_files = [f for f in os.listdir(source_dir) if f.endswith('.mrb')]

# Init results dictionary
uids_dict = {}
nrrd_sizes = {}

for mrb_file in mrb_files:
    
    subject_id = mrb_file.split('-')[0]
    
    # Copy the .mrb file to the destination directory
    shutil.copy(os.path.join(source_dir, mrb_file), dest_dir)

    # Remove the .mrb extension and add .zip
    zip_file = mrb_file.replace('.mrb', '.zip')
    os.rename(os.path.join(dest_dir, mrb_file), os.path.join(dest_dir, zip_file))

    # Create a new directory for the unzipped files
    unzip_dir = os.path.join(dest_dir, zip_file.replace('.zip', ''))
    os.makedirs(unzip_dir, exist_ok=True)

    # Unzip the .zip file
    with zipfile.ZipFile(os.path.join(dest_dir, zip_file), 'r') as zip_ref:
        zip_ref.extractall(unzip_dir)
        
    # Get the mrml file and convert it to a text file
    mrml_files = [f for f in glob.glob(unzip_dir + '/**/*.mrml', recursive=True)]
    if mrml_files:
        mrml_file = mrml_files[0]
        
    # Open the .mrml file and read its content
    with open(mrml_file, 'r') as file:
        mrml_content = file.read()

    # Parse the mrml_content
    root = ET.fromstring(mrml_content)

    # Parse first 60
    if len(uids_dict) < 60:
        
        # Store all SubjectHierarchy items in a dictionary with their dataNode attribute as the key
        subject_hierarchy_dict = {item.get('dataNode'): item for item in root.iter('SubjectHierarchyItem')}

        # Find the Segmentation object with id=vtkMRMLSegmentationNode1
        segmentation = next((item for item in root.iter('Segmentation') if item.get('id').startswith('vtkMRMLSegmentationNode')), None)

        if segmentation is not None:
            # Get the references attribute
            references = segmentation.get('references')

            # Find the referenceImageGeometryRef within the references
            referenceImageGeometryRef = next((ref.split(':')[1] for ref in references.split(';') if ref.startswith('referenceImageGeometryRef:')), None)

            if referenceImageGeometryRef is not None:
                # Find the SubjectHierarchy with dataNode=referenceImageGeometryRef
                subject_hierarchy = subject_hierarchy_dict.get(referenceImageGeometryRef)

                if subject_hierarchy is not None:
                    # Get the uids attribute
                    uids = subject_hierarchy.get('uids')
                    uids = uids.split('|')[0][-5:]
                    print(f"The uids attribute is: {uids}")
                    uids_dict[subject_id] = uids
                else:
                    print(f"No SubjectHierarchy found with dataNode={referenceImageGeometryRef}.")
            else:
                print("No referenceImageGeometryRef found in the references.")
        else:
            print("No Segmentation found with id=vtkMRMLSegmentationNode1.")
        
    # Parse next 60    
    else:
        
        # # Store all SubjectHierarchy items with a dataNode that starts with vtkMRMLScalarVolumeNode in a list
        subject_hierarchy_volumes = [item for item in root.iter('SubjectHierarchyItem') if item.get('dataNode', '').startswith('vtkMRMLScalarVolumeNode')]

        # Find the Segmentation object with id=vtkMRMLSegmentationNode1
        segmentation = next((item for item in root.iter('Segmentation') if item.get('id').startswith('vtkMRMLSegmentationNode')), None)

        if segmentation is not None:
            # Get the references attribute
            references = segmentation.get('references')

            # Find the referenceImageGeometryRef within the references
            referenceImageGeometryRef = next((ref.split(':')[1] for ref in references.split(';') if ref.startswith('referenceImageGeometryRef:')), None)

            if referenceImageGeometryRef is not None:
                # Find the Volume object with id=referenceImageGeometryRef
                matching_volume = next((volume for volume in root.iter('Volume') if volume.get('id') == referenceImageGeometryRef), None)
            
                if matching_volume is not None:
                    volume_name = matching_volume.get('name')
                    print(f"The name of the matching volume is: {volume_name}")
                else:
                    print(f"No Volume found with id={referenceImageGeometryRef}.")
            else:
                print("No referenceImageGeometryRef found in the references.")
        else:
            print("No Segmentation found with id=vtkMRMLSegmentationNode1.")
        
        # We cannot parse the mrml file for the second batch because some information were lost
        # We need to get the number of slices and compare 
        
        nrrd_file = [f for f in glob.glob(unzip_dir + '/**/*.nrrd', recursive=True) if volume_name == f.split('\\')[-1].split('.nrrd')[0]][0]
        
        data, header = nrrd.read(nrrd_file)
        nrrd_sizes[subject_id] = data.shape[2]

    # Remove the .zip file and unzipped file
    os.remove(os.path.join(dest_dir, zip_file))
    #os.remove(os.path.join(dest_dir, unzip_dir))

# Find the diacom folders
input_dir = r"C:\Users\pps21\Documents\Cornell\data\NLST Raw Datasets"

# Bug
nrrd_sizes['100226'] = 112

patients_cases = glob.glob(input_dir + '/**/**')
diacom_path = dict()
for case in patients_cases:
    # Create directory for a patient
    subject_id = case.split('\\')[-1]

    scan_paths = get_sorted_scan(case)
    
    c = 0
    if len(diacom_path) < 60:
        scan_used = []
        while not scan_used:
            scans = glob.glob(scan_paths[c] + '/**')
            scan_used = [s for s in scans if s.split('\\')[-1].split('-')[-1] == uids_dict[subject_id]]
            c+=1
        scan_used = scan_used[0]
    else:
        scan_used = None
        while not scan_used:
            scans = glob.glob(scan_paths[c] + '/**')
            for scan in scans:
                # Get the number of files in the scan
                num_files = len(glob.glob(scan + '/*'))
                # Check if the number of files matches the size in nrrd_sizes
                if nrrd_sizes.get(subject_id, 0) == num_files:
                    scan_used = scan
                    break
            c += 1
    diacom_path[subject_id] = scan_used
    
# Export used DIACOM path
diacom_path = {int(key): value.split('\\')[-2] + '/' + value.split('\\')[-1]  for key, value in diacom_path.items()}
    
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

df.columns = ['subject_id', 'path_prev', 'path_actual_seg']
df.to_csv(r"C:\Users\pps21\Documents\Cornell\data\compare_diacom_path.csv", index=False)



# # Previously used code to scrap second batch (does not work)
# # Store all SubjectHierarchy items with a dataNode that starts with vtkMRMLScalarVolumeNode in a list
# subject_hierarchy_volumes = [item for item in root.iter('SubjectHierarchyItem') if item.get('dataNode', '').startswith('vtkMRMLScalarVolumeNode')]

# # Find the Segmentation object with id=vtkMRMLSegmentationNode1
# segmentation = next((item for item in root.iter('Segmentation') if item.get('id').startswith('vtkMRMLSegmentationNode')), None)

# if segmentation is not None:
#     # Get the references attribute
#     references = segmentation.get('references')

#     # Find the referenceImageGeometryRef within the references
#     referenceImageGeometryRef = next((ref.split(':')[1] for ref in references.split(';') if ref.startswith('referenceImageGeometryRef:')), None)

#     if referenceImageGeometryRef is not None:
#         # Find the index of the SubjectHierarchyVolume that matches the referenceImageGeometryRef
#         matching_volume_index = next((i for i, volume in enumerate(subject_hierarchy_volumes) if volume.get('dataNode') == referenceImageGeometryRef), None)

#         if matching_volume_index is not None:
#             print(f"The index of the matching SubjectHierarchyVolume is: {matching_volume_index}")
#             index_diacom[subject_id] = matching_volume_index
#         else:
#             print(f"No SubjectHierarchyVolume found with dataNode={referenceImageGeometryRef}.")
#     else:
#         print("No referenceImageGeometryRef found in the references.")
# else:
#     print("No Segmentation found with id=vtkMRMLSegmentationNode1.")