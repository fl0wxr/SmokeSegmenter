import numpy as np
from PIL import Image
from copy import deepcopy
from glob import glob
import json
import os
import re

import visuals


class SegmData:
    '''
        Description:
            Segmentation data for smoke class. Labels are mask images.
    '''

    def __init__(self, dataset_dp):

        self.raw_class_idcs = {'background': 0, 'smoke': 1}
        self.classes = [None for class_idx in range(len(self.raw_class_idcs))]
        for class_idx in range(len(self.classes)):
            for (key, value) in self.raw_class_idcs.items():
                if value == class_idx:
                    self.classes[class_idx] = key
                    break

        self.n_classes = len(self.raw_class_idcs)
        self.dataset_dp = dataset_dp

        self.imgs_fps, self.masks_fps = self.generate_data_paths()

        self.n_instances = len(self.imgs_fps)

        self.INSTANCE_IDX = -1

    def __next__(self):

        self.INSTANCE_IDX += 1
        if self.INSTANCE_IDX <= self.n_instances - 1:

            print('Path index:', self.INSTANCE_IDX)

            self.img_fp = self.imgs_fps[self.INSTANCE_IDX]
            self.mask_fp = self.masks_fps[self.INSTANCE_IDX]

            assert self.img_fp.split('/')[-1].split('.')[-2] == self.mask_fp.split('/')[-1].split('.')[-2], 'E: Image bounding box file name pair incompatibility'

            self.img, self.mask = self.get_one_segm_instance_ssmoke_dp(img_fp = self.img_fp, mask_fp = self.mask_fp)

            self.n_smoke_pixels = np.sum(self.mask == self.raw_class_idcs['smoke'])

            self.instance_name = self.mask_fp.split('/')[-1].split('.')[-2]

            return True

        else:

            return False

    def generate_data_paths(self) -> (tuple[str], tuple[str]):
        '''
            Returns:
                A tuple of 2 values where the first is
                    proper_imgs_paths. All image paths that correspond to a mask path.
                and the second is
                    proper_masks_paths. Each position of this tuple corresponds to the same indice's position inside proper_imgs_paths.
        '''

        imgs_paths = glob\
        (
            pathname = os.path.join(self.dataset_dp, 'images', '*.jpg'),
            recursive = True
        )

        masks_paths = glob\
        (
            pathname = os.path.join(self.dataset_dp, 'seg_labels', '*.png'),
            recursive = True
        )

        imgs_paths = sorted(imgs_paths, reverse = True)
        masks_paths = sorted(masks_paths, reverse = True)

        instance_pair_paths = {}
        print('Matching all instance pair (image & mask) paths')

        imgs_filenames = []
        for img_path in imgs_paths:
            imgs_filenames.append(img_path.split('/')[-1].split('.')[-2])

        masks_filenames = []
        for mask_path in masks_paths:
            masks_filenames.append(mask_path.split('/')[-1].split('.')[-2])

        for img_filename, img_path in zip(imgs_filenames, imgs_paths):
            for mask_filename, mask_path in zip(masks_filenames, masks_paths):
                if img_filename == mask_filename:
                    instance_pair_paths[img_path] = mask_path

        return tuple(instance_pair_paths.keys()), tuple(instance_pair_paths.values())

    def preprocess_smoke5k(self, img: np.ndarray, masks: list[np.ndarray]) \
        -> tuple[list[np.ndarray], list[np.ndarray]]:
        '''
            Args:
                imgs. Shape (H, W, C).
                masks. Size N.
                    masks[i]. Shape (H, W, C).
        '''

        for i in range(len(masks)):
            masks[i] = masks[i] // np.max(masks[i])

        return img, masks

    def instance_pair_integrity_check_raw(self, img: np.ndarray, mask: list[np.ndarray]):
        '''
            Description:
                Used right after the data parsing.

            Args:
                img. Shape (H, W, C).
                mask. Shape (H, W, C).
        '''

        ## Lengths
        assert img.shape[:-1] == mask.shape, 'E: Image and mask arrays do not share the same shape.'

    def instance_pair_integrity_check_preprocessed(self, img: np.ndarray, mask: list[np.ndarray], raw_class_idcs: dict):
        '''
            Description:
                Used right after the initial preprocessing of data.

            Args:
                img. Shape (H, W, C).
                mask. Shape (H, W, C).
                raw_class_idcs. Size `n_classes`.
        '''

        raw_class_idcs_value_set = {val for val in raw_class_idcs.values()}

        ## Inner lengths
        assert img.shape[:-1] == mask.shape, 'E: Image and mask arrays do not share the same shape.'

        ## Data type
        assert type(img.flat[0]) == np.uint8, 'E: Image array data type is not np.uint8.'
        assert type(mask.flat[0]) == np.uint8, 'E: Image array data type is not np.uint8.'

        ## Mask
        mask_range = set(np.unique(mask).tolist())
        assert 0 in mask_range and mask_range.issubset(raw_class_idcs_value_set), 'E: Mask range and class indices are inconsistent.'

    def get_one_segm_instance_smoke5k_dp(self):

        img_smoke5k = np.array(Image.open(self.dataset_dp + 'SMOKE5K/train/img/1_23.jpg'))

        ## 0 -> Background, 255 -> Smoke. Assumes that multiple values can be taken for multiple classes.
        mask_smoke5k = np.array(Image.open(self.dataset_dp + 'SMOKE5K/train/gt/1_23.png'))

        self.instance_pair_integrity_check_raw(img = img_smoke5k, mask = mask_smoke5k)
        img_smoke5k, mask_smoke5k = self.preprocess_smoke5k(img = img_smoke5k, mask = mask_smoke5k)
        self.instance_pair_integrity_check_preprocessed(img = img_smoke5k, mask = mask_smoke5k, raw_class_idcs = self.raw_class_idcs)

        return img_smoke5k, mask_smoke5k

    def get_one_segm_instance_ssmoke_dp(self, img_fp: str = None, mask_fp: str = None):

        if img_fp == None or mask_fp == None:
            img_fp = self.dataset_dp + 'images/' + 'AoF06978' + '.jpg'
            mask_fp = self.dataset_dp + 'seg_labels/' + 'AoF06978' + '.png'

        ## ! Parse: Begin

        img_ssmoke = np.array(Image.open(img_fp).convert("RGB"))

        ## 0 -> Background, 1 -> Smoke. Assumes that multiple values can be taken for multiple classes.
        mask_ssmoke = np.array(Image.open(mask_fp))

        ## ! Parse: End

        self.instance_pair_integrity_check_raw(img = img_ssmoke, mask = mask_ssmoke)
        self.instance_pair_integrity_check_preprocessed(img = img_ssmoke, mask = mask_ssmoke, raw_class_idcs = self.raw_class_idcs)

        return img_ssmoke, mask_ssmoke

class DetData:
    '''
        Description:
            Object detection data parsing with YOLO's bounding box label format. Classes: 0 -> Smoke, 1 -> Fire.
    '''

    def __init__(self, dataset_dp, save_dp = None):

        self.raw_class_idcs = {'smoke': 0, 'fire': 1}
        self.classes = [None for class_idx in range(len(self.raw_class_idcs))]
        for class_idx in range(len(self.classes)):
            for (key, value) in self.raw_class_idcs.items():
                if value == class_idx:
                    self.classes[class_idx] = key
                    break

        self.n_classes = len(self.raw_class_idcs)
        self.dataset_dp = dataset_dp
        self.save_dp = save_dp

        self.imgs_fps, self.bboxes_fps = self.generate_data_paths()
        self.n_instances = len(self.imgs_fps)

        self.INSTANCE_IDX = -1

    def __next__(self):

        self.INSTANCE_IDX += 1
        if self.INSTANCE_IDX <= self.n_instances - 1:
            # print('Debugging 0x4ec472765a: Adjust previous line')
            print('Path index:', self.INSTANCE_IDX)

            img_fp = self.imgs_fps[self.INSTANCE_IDX]
            bboxes_fp = self.bboxes_fps[self.INSTANCE_IDX]
            assert img_fp.split('/')[-1].split('.')[-2] == bboxes_fp.split('/')[-1].split('.')[-2], 'E: Image bounding box file name pair incompatibility'

            self.img, self.bboxes = self.get_one_det_instance_ssmoke(img_fp = img_fp, bboxes_fp = bboxes_fp)
            if self.save_dp is not None:
                self.save_image_fp = os.path.join(self.save_dp, 'images', os.path.basename(img_fp))
                self.mask_fp = os.path.join(self.save_dp, 'seg_labels', os.path.splitext(os.path.basename(bboxes_fp))[0]) + '.png'
            else:
                self.save_image_fp = None
                self.mask_fp = None

            return True

        else:

            return False

    def generate_data_paths(self):

        imgs_paths = glob\
        (
            pathname = os.path.join(self.dataset_dp, '**/*.jpg'),
            recursive = True
        )

        bboxes_paths = glob\
        (
            pathname = os.path.join(self.dataset_dp, '**/*.txt'),
            recursive = True
        )

        assert len(imgs_paths) == len(bboxes_paths), 'E: Image-label mismatch'

        return sorted(imgs_paths, reverse = True), sorted(bboxes_paths, reverse = True)

    def save_segm_labels(self, mask: np.ndarray):

        if self.save_image_fp is None:
            print('W: Files were not saved')
            return False

        out_img = Image.fromarray(self.img)
        out_img.save(self.save_image_fp)
        mask_ = Image.fromarray(mask)
        mask_.save(self.mask_fp)

        return True

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
            assert yolo_bbox[0] in self.raw_class_idcs.values(), 'E: Invalid class label index'

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

    def get_one_det_instance_ssmoke(self, img_fp: str = None, bboxes_fp: str = None) -> tuple[np.ndarray, list[list]]:
        '''
            Description:
                Receives an image path and a JSON file that contains its bounding boxes in YOLO format and returns the respective objects.
        '''

        if img_fp == None or bboxes_fp == None:
            img_fp = self.dataset_dp + 'train/images/' + 'WEB09440' + '.jpg'
            bboxes_fp = self.dataset_dp + 'train/det_labels/' + 'WEB09440' + '.txt'

        ## Parse data
        img_ssmoke = np.array(Image.open(img_fp).convert("RGB"))
        yolo_bboxes_ssmoke = self.get_yolo_bboxes(path = bboxes_fp)

        ## Preprocess
        norm_vertex_bboxes_ssmoke = self.yolo_bboxes_to_vertex_bboxes(yolo_bboxes = yolo_bboxes_ssmoke)
        norm_vertex_bboxes_ssmoke = self.label_drop_all_classes_except_first(norm_vertex_bboxes_ssmoke)
        vertex_bboxes_ssmoke = self.norm_bboxes_to_image_bboxes(img_shape = img_ssmoke.shape, norm_vertex_bboxes = norm_vertex_bboxes_ssmoke)

        return img_ssmoke, vertex_bboxes_ssmoke