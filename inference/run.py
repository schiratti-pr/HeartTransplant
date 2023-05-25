import os
import argparse
import numpy as np
import random
import cc3d
import copy
import cv2
import nibabel as nib
from matplotlib.pyplot import figure, imshow, axis, savefig
import matplotlib.pyplot as plt

import torch
import torch.backends.cudnn as cudnn

from utils import get_model, preprocess, load_sample_more_details


def showPredictionContour(img, pred, slice_id):
    img = np.stack((img,) * 3, axis=-1).copy()

    pred = pred.numpy().astype('uint8')
    contours_pred, _ = cv2.findContours(pred.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours_pred:
        cv2.drawContours(img, [c], 0, (255, 0, 0), 2)

    fig, ax = plt.subplots()
    ax.imshow(img)
    proxy = [plt.Rectangle((0, 0), 1, 1, fc='red')]
    ax.set_title("Slice: {}".format(slice_id))
    plt.legend(proxy, ["prediction"])
    axis('off')
    plt.tight_layout()

    return fig



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
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--result_save', type=str, help='path to folder', default='inference_results')

    args = parser.parse_args()

    device = torch.device(args.device)

    # Prepare data
    data_sample, raw_sample = preprocess(args.scan_path)
    data_sample = np.swapaxes(data_sample, 2, -1).squeeze()
    data_sample = np.transpose(data_sample, [1, 0, 2]).unsqueeze(0).unsqueeze(0)

    # Init model
    model = get_model(spatial_dims=3, num_in_channels=1)
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

    # # Save Volume to NIFTI file
    # nifti_file = nib.Nifti1Image(class_pred.numpy(), np.eye(4))
    # name_file = args.scan_path.split('/')[-1].split('.')[0] + 'pred'
    # scan_path_save = os.path.join(args.result_save,  name_file)
    # nib.save(nifti_file, scan_path_save)
    # print('Result saved to: {}.nii'.format(scan_path_save))

    # Save Volume prediction
    num_slices, scan1, z_spacing = load_sample_more_details(args.scan_path)
    x_spacing, y_spacing = scan1.PixelSpacing[0], scan1.PixelSpacing[1]
    resize_scale = (512 * 512 * num_slices) / (256 * 256 * 256)
    heart_volume = np.sum(class_pred.numpy()) * x_spacing * y_spacing * z_spacing * 0.001 * resize_scale
    heart_volume = np.round(heart_volume, 5)

    name_file = 'volume_prediction.txt'
    txt_path_save = os.path.join(args.result_save, name_file)
    file_txt = open(txt_path_save, 'w')
    file_txt.write('Predicted heart volume is: {}'.format(heart_volume))
    file_txt.close()
    print('Predicted heart volume is {} and was saved to {}'.format(heart_volume, txt_path_save))

    # Save Volume as slices with overlay
    data_sample = data_sample.squeeze().numpy()

    for slice_id in range(data_sample.shape[0]):
        showPredictionContour(data_sample[:, :, slice_id], class_pred[:, :, slice_id], slice_id)
        savefig(args.result_save + '/{}.png'.format(slice_id), bbox_inches='tight')


if __name__ == '__main__':
    main()
