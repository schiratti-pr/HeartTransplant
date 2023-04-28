import os
import argparse
import numpy as np
import random
import cc3d
import copy
import nibabel as nib

import torch
import torch.backends.cudnn as cudnn

from utils import get_model, preprocess


def main():
    SEED = 0
    cudnn.benchmark = True
    cudnn.deterministic = False
    torch.cuda.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    parser = argparse.ArgumentParser(description='Processing configuration for training')
    parser.add_argument('--scan_path', type=str)
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--result_save', type=str, help='path to folder', default='inference_results')

    args = parser.parse_args()

    device = torch.device(args.device)

    # Prepare data
    data_sample = preprocess(args.scan_path)

    # Init model
    model = get_model('UNet', spatial_dims=3, num_in_channels=1)
    model.to(device)

    # Load weights
    checkpoint_path = 'weights/best_3d_model.ckpt'
    state_dict = torch.load(str(checkpoint_path), map_location=device)['state_dict']
    for key in list(state_dict.keys()):
        state_dict[key.replace('net.', '')] = state_dict.pop(key)

    model.load_state_dict(state_dict)
    model.eval()

    prediction = model(data_sample.to(device)).squeeze().detach()
    sigmoid_pred = torch.sigmoid(prediction)
    class_pred = torch.round(sigmoid_pred).cpu()

    # Filtering out everything beyond the pre-calculated heart zone
    output_pred = np.zeros(class_pred.shape)
    output_pred[40:168, 79:215, :] = class_pred[40:168, 79:215, :]
    class_pred = torch.tensor(output_pred)

    # CCA
    connectivity = 6  # only 4,8 (2D) and 26, 18, and 6 (3D) are allowed
    labels_out, N = cc3d.connected_components(class_pred.numpy(), connectivity=connectivity, return_N=True)
    dict_components = {}
    for label, image in cc3d.each(labels_out, binary=False, in_place=True):
        dict_components.update({np.count_nonzero(image): copy.deepcopy(image)})
    try:
        class_pred = dict_components[max(dict_components.keys())]
        class_pred = torch.Tensor(class_pred.astype(np.float32))
    except:
        pass   # No components were found

    # Saving the entire prediction mask
    if not os.path.exists(args.result_save):
        os.makedirs(args.result_save)

    nifti_file = nib.Nifti1Image(class_pred.numpy(), np.eye(4))
    name_file = args.scan_path.split('/')[-1].split('.')[0] + '_pred'
    scan_path_save = os.path.join(args.result_save,  name_file)
    nib.save(nifti_file, scan_path_save)
    print('Result saved to: {}.nii'.format(scan_path_save))


if __name__ == '__main__':
    main()
