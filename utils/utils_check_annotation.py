import numpy as np
import nibabel as nib
import glob


def check_annotation_quality(label_path, target_size=512):
    label_volume = nib.load(label_path).get_fdata()

    # Check the size of the label
    assert label_volume.shape[0] == label_volume.shape[1] == target_size

    # Check gaps in between slices
    annot_slices = []
    for sl in range(label_volume.shape[-1]):
        if np.sum(label_volume[:, :, sl]) > 0:
            annot_slices.append(sl)
    missing_elements = [x for x in range(annot_slices[0], annot_slices[-1] + 1) if x not in annot_slices]

    print('Number of annotated slices: {}, for {}'.format(len(annot_slices), label_path.split('/')[-1]))

    if len(missing_elements) == 0:
        return None

    return missing_elements


def check_annotation_directory(dir_path, target_size=512):
    all_files = glob.glob(dir_path + '/**')
    nii_files = [x for x in all_files if '.nii' in x]

    analysis_dict = {}
    for f in nii_files:
        check_result = check_annotation_quality(f, target_size=target_size)
        analysis_dict.update({f: check_result})

    return analysis_dict


if __name__ == '__main__':
    path_annotation = '/Users/mariadobko/Downloads/100005-label.nii'
    missing_labels = check_annotation_quality(path_annotation)
    print('In this volume the following slices are missing annotations : {}'.format(missing_labels))

    dir_annotations = '/Users/mariadobko/Downloads/Annotations - VT 2/'
    result = check_annotation_directory(dir_annotations)

    for k in result.keys():
        if result[k] is None:
            pass
        else:
            print('Some slices missed', result[k])
