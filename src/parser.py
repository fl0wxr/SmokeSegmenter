import json
import numpy as np
from PIL import Image
from copy import deepcopy

import visuals

from pdb import set_trace as pause


def preprocess_SMOKE5K(imgs: list[np.ndarray], masks: list[np.ndarray]) \
    -> tuple[list[np.ndarray], list[np.ndarray]]:
    '''
        Args:
            imgs. Size N.
                imgs[i]. Shape (H, W, C).
            masks. Size N.
                masks[i]. Shape (H, W, C).
    '''

    for i in range(len(masks)):
        masks[i] = masks[i] // np.max(masks[i])

    return imgs, masks

def instance_pair_integrity_check_raw(imgs: list[np.ndarray], masks: list[np.ndarray]):
    '''
        Description:
            Used right after the data parsing.

        Args:
            imgs. Size N.
                imgs[i]. Shape (H, W, C).
            masks. Size N.
                masks[i]. Shape (H, W, C).
    '''

    ## Lengths
    assert len(imgs) == len(masks), 'E: Inconsistency in number of instances.'
    for img, mask in zip(imgs, masks):
        assert img.shape[:-1] == mask.shape, 'E: Image and mask arrays do not share the same shape.'

def instance_pair_integrity_check_preprocessed(imgs: list[np.ndarray], masks: list[np.ndarray], class_idcs: dict):
    '''
        Description:
            Used right after the initial preprocessing of data.

        Args:
            imgs. Size N.
                imgs[i]. Shape (H, W, C).
            masks. Size N.
                masks[i]. Shape (H, W, C).
            class_idcs. Size `n_classes`.
    '''

    class_idcs_value_set = {val for val in class_idcs.values()}

    ## Outer length
    assert len(imgs) == len(masks), 'E: Inconsistency in number of instances.'
    for img, mask in zip(imgs, masks):

        ## Inner lengths
        assert img.shape[:-1] == mask.shape, 'E: Image and mask arrays do not share the same shape.'

        ## Data type
        assert type(img.flat[0]) == np.uint8, 'E: Image array data type is not np.uint8.'
        assert type(mask.flat[0]) == np.uint8, 'E: Image array data type is not np.uint8.'

        ## Mask
        masks_range = set(np.unique(masks).tolist())
        assert masks_range == class_idcs_value_set, 'E: Mask range and class indices are inconsistent.'

def test_one_instance():

    class_idcs = {'background': 0, 'smoke': 1}
    n_classes = len(class_idcs)

    with open('../paths.json', 'r') as json_file:
        json_content = json.load(json_file)
        datasets_dp = json_content['datasets_dp']

    ## Dummy dataset
    imgs_SMOKE5K = [np.array(Image.open(datasets_dp + '/1_1.jpg'))]

    ## 0 -> Background, 255 -> Smoke. Assumes that multiple values can be taken for multiple classes.
    masks_SMOKE5K = [np.array(Image.open(datasets_dp + '1_1.png'))]

    instance_pair_integrity_check_raw(imgs = imgs_SMOKE5K, masks = masks_SMOKE5K)
    imgs_SMOKE5K, masks_SMOKE5K = preprocess_SMOKE5K(imgs = imgs_SMOKE5K, masks = masks_SMOKE5K)
    instance_pair_integrity_check_preprocessed(imgs = imgs_SMOKE5K, masks = masks_SMOKE5K, class_idcs = class_idcs)

    image_plot = visuals.SegmentationVisuals()
    image_plot.build_plt(img = imgs_SMOKE5K[0], mask = masks_SMOKE5K[0], fig_title = 'Testing')

    image_plot.store_fig(fp = './fig0.jpg')

if __name__ == '__main__':

    test_one_instance()


