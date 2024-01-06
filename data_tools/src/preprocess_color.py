
from PIL import Image
import numpy as np
import os
from matplotlib import pyplot as plt


data_list_dp = '../../../datasets/config_new/'
train_labels_paths_list = os.path.join(data_list_dp, 'train.list')
# val_labels_list = os.path.join(data_list_dp, 'val.list')
test_labels_paths_list = os.path.join(data_list_dp, 'test.list')


def gray2rgb_color_replacer():
    '''
        Description:
            Converts from grayscale to RGB. Specifically replaces 0 pixels with [0, 128, 192] pixels and 1 pixels with [128, 0, 0] pixels. Practical only for segmentation mask images.
    '''

    value_to_replace0 = 0
    replacement_color0 = np.array([0, 128, 192], dtype = np.uint8)

    value_to_replace1 = 1
    replacement_color1 = np.array([128, 0, 0], dtype = np.uint8)

    with open(file = train_labels_paths_list, mode='r') as train_f:
        with open(file = test_labels_paths_list, mode='r') as test_f:

            fps = [line.split('\t')[1].strip() for line in train_f] + [line.split('\t')[1].strip() for line in test_f]

            for fp in fps:

                in_label = np.array(Image.open(fp))

                assert len(in_label.shape) == 2, 'E: Incompatible shape'

                # Create an empty RGB image (initializing with replacement color)
                out_label = np.ones((*in_label.shape, 3), dtype=np.uint8) * replacement_color0

                # Find indices where the specified value appears in the grayscale image
                indices = in_label == value_to_replace1

                # Replace pixels in the RGB image with the new color
                out_label[indices] = replacement_color1

                Image.fromarray(out_label).save(fp)

def rgb2gray_color_replacer():
    '''
        Description:
            Converts from RGB to grayscale. Specifically replaces [0, 128, 192] with 0 pixels and [128, 0, 0] pixels with 1 pixels. Practical only for segmentation mask images.
    '''

    value_to_replace0 = np.array([0, 128, 192], dtype = np.uint8)
    replacement_color0 = 0

    value_to_replace1 = np.array([128, 0, 0], dtype = np.uint8)
    replacement_color1 = 1

    with open(file = train_labels_paths_list, mode='r') as train_f:
        with open(file = test_labels_paths_list, mode='r') as test_f:

            fps = [line.split('\t')[1].strip() for line in train_f] + [line.split('\t')[1].strip() for line in test_f]

            for fp in fps:

                in_label = np.array(Image.open(fp))

                assert len(in_label.shape) == 3, 'E: Incompatible shape'

                # Create an empty grayscale image (initializing with replacement color)
                out_label = np.ones((in_label.shape[:2]), dtype=np.uint8) * replacement_color0

                # Find indices where the specified value appears in the grayscale image
                indices = np.all(in_label == value_to_replace1, axis = -1)

                # Replace pixels in the RGB image with the new color
                out_label[indices] = replacement_color1

                Image.fromarray(out_label).save(fp)


if __name__ == '__main__':

    rgb2gray_color_replacer()