import numpy as np
from PIL import Image
from copy import deepcopy
from glob import glob
import json
import os

import visuals

from pdb import set_trace as pause


class SegmData:
    '''
        Description:
            Segmentation data for smoke class. Labels are mask images.
    '''

    def __init__(self, dataset_dp):

        self.class_idcs = {'background': 0, 'smoke': 1}
        self.classes = [None for class_idx in range(len(self.class_idcs))]
        for class_idx in range(len(self.classes)):
            for (key, value) in self.class_idcs.items():
                if value == class_idx:
                    self.classes[class_idx] = key
                    break

        self.n_classes = len(self.class_idcs)
        self.dataset_dp = dataset_dp

    def preprocess_smoke5k(self, imgs: list[np.ndarray], masks: list[np.ndarray]) \
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

    def instance_pair_integrity_check_raw(self, imgs: list[np.ndarray], masks: list[np.ndarray]):
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

    def instance_pair_integrity_check_preprocessed(self, imgs: list[np.ndarray], masks: list[np.ndarray], class_idcs: dict):
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

    def get_one_segm_instance_smoke5k_dp(self):

        imgs_smoke5k = [np.array(Image.open(self.dataset_dp + 'SMOKE5K/train/img/1_23.jpg'))]

        ## 0 -> Background, 255 -> Smoke. Assumes that multiple values can be taken for multiple classes.
        masks_smoke5k = [np.array(Image.open(self.dataset_dp + 'SMOKE5K/train/gt/1_23.png'))]

        self.instance_pair_integrity_check_raw(imgs = imgs_smoke5k, masks = masks_smoke5k)
        imgs_smoke5k, masks_smoke5k = self.preprocess_smoke5k(imgs = imgs_smoke5k, masks = masks_smoke5k)
        self.instance_pair_integrity_check_preprocessed(imgs = imgs_smoke5k, masks = masks_smoke5k, class_idcs = self.class_idcs)

        return imgs_smoke5k[0], masks_smoke5k[0]

class DetData:
    '''
        Description:
            Object detection data parsing with YOLO's bounding box label format. Classes: 0 -> Smoke, 1 -> Fire.
    '''

    def __init__(self, dataset_dp, processed_dataset_dp = None, multiple_instances = False):
        self.pause_ = False

        self.class_idcs = {'smoke': 0, 'fire': 1}
        self.classes = [None for class_idx in range(len(self.class_idcs))]
        for class_idx in range(len(self.classes)):
            for (key, value) in self.class_idcs.items():
                if value == class_idx:
                    self.classes[class_idx] = key
                    break

        self.n_classes = len(self.class_idcs)
        self.dataset_dp = dataset_dp
        self.processed_dataset_dp = processed_dataset_dp
        self.multiple_instances = multiple_instances

        assert (self.multiple_instances and processed_dataset_dp != None) or (not self.multiple_instances and processed_dataset_dp == None), 'E: Fix it.'
        if self.multiple_instances:

            self.imgs_fnames, self.bboxes_fnames = self.generate_data_paths()
            self.n_instances = len(self.imgs_fnames)

            self.INSTANCE_IDX = -1

    def __next__(self):

        if self.multiple_instances == False:
            print('W: This only works for directory copies.')
            return None

        self.INSTANCE_IDX += 1
        print('List index:', self.INSTANCE_IDX)
        if self.INSTANCE_IDX <= self.n_instances - 1:
            img_fname = self.imgs_fnames[self.INSTANCE_IDX]
            bboxes_fname = self.bboxes_fnames[self.INSTANCE_IDX]
            assert img_fname.split('/')[-1].split('.')[-2] == bboxes_fname.split('/')[-1].split('.')[-2], 'E: Image bbox file name pair incompatibility'
            self.img, self.bboxes = self.get_one_det_instance_dfire(img_fp = img_fname, bbox_fp = bboxes_fname)
            self.imgs_fnames_out = self.processed_dataset_dp + self.imgs_fnames[self.INSTANCE_IDX].split('/')[-1]
            self.bboxes_fnames_out = '.'.join((self.processed_dataset_dp + self.bboxes_fnames[self.INSTANCE_IDX].split('/')[-1]).split('.')[:-1] + ['json'])

            return True

        else:

            return False

    def save_pos_smoke(self):

        if self.multiple_instances == False:
            print('W: This only works for directory copies.')
            return None

        if self.bboxes != [] and not os.path.exists(self.imgs_fnames_out):
            img_pil = Image.fromarray(self.img)
            img_pil.save(self.imgs_fnames_out)

            with open(file = self.bboxes_fnames_out, mode = 'w') as json_file:
                json.dump(self.bboxes, json_file)

    def get_yolo_bboxes(self, path: str) -> list[list]:
        '''
            Description:
                Reads bounding boxes file based on YOLO format, and produces a list.

            Args:
                path. Path to bounding boxes file.

            Returns:
                yolo_bboxes. Length equal to the instances number of bounding boxes.
                    yolo_bboxes[i] -> Bounding box with index i. The first value is the class, the following 2 values are the normalized coordinates in [0, 1] of the bounding boxes center, and the last 2 are width and height. Hence you expect each bounding box value format to be in
                        yolo_bboxes[i][0] -> class index
                        yolo_bboxes[i][1] -> x center
                        yolo_bboxes[i][2] -> y center
                        yolo_bboxes[i][3] -> width
                        yolo_bboxes[i][4] -> height
        '''

        with open(file = path, mode = 'r') as yolo_bboxes_file:
            yolo_bboxes_str = yolo_bboxes_file.read().split('\n')

        if yolo_bboxes_str[-1] == '': yolo_bboxes_str = yolo_bboxes_str[:-1]
        yolo_bboxes_str = [yolo_bbox_str.split(' ') for yolo_bbox_str in yolo_bboxes_str]
        yolo_bboxes = []
        for yolo_bbox_str in yolo_bboxes_str:
            yolo_bbox = 5 * [None]
            yolo_bbox[0] = int(yolo_bbox_str[0])
            for coord_idx in range(1, len(yolo_bbox)):
                yolo_bbox[coord_idx] = float(yolo_bbox_str[coord_idx])
            if min(yolo_bbox[1:]) < 0 or 1 < max(yolo_bbox[1:]):
                print('W: Invalid coordinates.')
            else:
                yolo_bboxes.append(yolo_bbox)
            assert yolo_bbox[0] in self.class_idcs.values(), 'E: Invalid class label index'

        return yolo_bboxes

    def yolo_bboxes_to_vertex_bboxes(self, yolo_bboxes: list[list]) -> list[list]:
        '''
            Args:
                yolo_bboxes. Length equal to the instances number of bounding boxes.
                    yolo_bboxes[i][0] -> class index
                    yolo_bboxes[i][1] -> x center
                    yolo_bboxes[i][2] -> y center
                    yolo_bboxes[i][3] -> width
                    yolo_bboxes[i][4] -> height

            Returns:
                norm_vertex_bboxes. Length equal to the instances number of bounding boxes. Coordinates are normalized.
                    norm_vertex_bboxes[i][0] -> class index
                    norm_vertex_bboxes[i][1] -> x left
                    norm_vertex_bboxes[i][2] -> x right
                    norm_vertex_bboxes[i][3] -> y up
                    norm_vertex_bboxes[i][4] -> y down
        '''

        norm_vertex_bboxes = [5 * [None] for bbox_idx in range(len(yolo_bboxes))]
        for bbox_idx, yolo_bbox in enumerate(yolo_bboxes):
            norm_vertex_bboxes[bbox_idx][0] = yolo_bbox[0]
            norm_vertex_bboxes[bbox_idx][1] = yolo_bbox[1] - (yolo_bbox[3] / 2)
            norm_vertex_bboxes[bbox_idx][2] = yolo_bbox[1] + (yolo_bbox[3] / 2)
            norm_vertex_bboxes[bbox_idx][3] = yolo_bbox[2] - (yolo_bbox[4] / 2)
            norm_vertex_bboxes[bbox_idx][4] = yolo_bbox[2] + (yolo_bbox[4] / 2)

        return norm_vertex_bboxes

    def norm_bboxes_to_image_bboxes(self, img_shape: tuple[int], norm_vertex_bboxes: list[list]) -> list[list]:
        '''
            Args:
                norm_vertex_bboxes. Length equal to the instances number of bounding boxes. Coordinates are normalized.
                    norm_vertex_bboxes[i][0] -> class index
                    norm_vertex_bboxes[i][1] -> x left
                    norm_vertex_bboxes[i][2] -> x right
                    norm_vertex_bboxes[i][3] -> y up
                    norm_vertex_bboxes[i][4] -> y down

            Returns:
                norm_vertex_bboxes. Takes `norm_vertex_bboxes` discretizes and scales its coordinates to match that of an image with shape `img_shape`.
        '''

        vertex_bboxes = [5 * [None] for bbox_idx in range(len(norm_vertex_bboxes))]
        for bbox_idx, norm_vertex_bbox in enumerate(norm_vertex_bboxes):
            vertex_bboxes[bbox_idx][0] = norm_vertex_bbox[0]
            vertex_bboxes[bbox_idx][1] = int(norm_vertex_bbox[1] * img_shape[1])
            vertex_bboxes[bbox_idx][2] = int(norm_vertex_bbox[2] * img_shape[1])
            vertex_bboxes[bbox_idx][3] = int(norm_vertex_bbox[3] * img_shape[0])
            vertex_bboxes[bbox_idx][4] = int(norm_vertex_bbox[4] * img_shape[0])

        return vertex_bboxes

    def label_drop_all_classes_except_first(self, norm_vertex_bboxes):

        norm_vertex_bboxes_filtered = []
        for norm_vertex_bbox in norm_vertex_bboxes:
            if norm_vertex_bbox[0] == 0:
                norm_vertex_bboxes_filtered.append(norm_vertex_bbox)

        return norm_vertex_bboxes_filtered

    def generate_data_paths(self):

        imgs_paths = glob\
        (
            pathname = self.dataset_dp + 'train/images/**/*.jpg',
            recursive = True
        ) + \
        glob\
        (
            pathname = self.dataset_dp + 'test/images/**/*.jpg',
            recursive = True
        )

        bboxes_paths = glob\
        (
            pathname = self.dataset_dp + 'train/labels/**/*.txt',
            recursive = True
        ) + \
        glob\
        (
            pathname = self.dataset_dp + 'test/labels/**/*.txt',
            recursive = True
        )

        return sorted(imgs_paths), sorted(bboxes_paths)

    def get_one_det_instance_dfire(self, img_fp: str = None, bboxes_fp: str = None) -> tuple[np.ndarray, list[list]]:
        '''
            Description:
                Receives an image path and a JSON file that contains its bounding boxes in YOLO format and returns the respective objects.

            Args:
                img_fp. File path of the input image.
                bbox
        '''

        if img_fp == None or bboxes_fp == None:
            img_fp = self.dataset_dp + 'train/images/' + 'WEB09440' + '.jpg'
            bboxes_fp = self.dataset_dp + 'train/labels/' + 'WEB09440' + '.txt'

        imgs_dfire = np.array(Image.open(img_fp).convert("RGB"))

        yolo_bboxes_dfire = self.get_yolo_bboxes(path = bboxes_fp)

        norm_vertex_bboxes_dfire = self.yolo_bboxes_to_vertex_bboxes(yolo_bboxes = yolo_bboxes_dfire)

        norm_vertex_bboxes_dfire = self.label_drop_all_classes_except_first(norm_vertex_bboxes_dfire)

        vertex_bboxes_dfire = self.norm_bboxes_to_image_bboxes(img_shape = imgs_dfire.shape, norm_vertex_bboxes = norm_vertex_bboxes_dfire)

        return imgs_dfire, vertex_bboxes_dfire