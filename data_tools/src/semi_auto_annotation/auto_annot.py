import sys

sys.path.append('../utils')

import visuals
import data_utils
import bbox2segm_mask
import json
import os
import time
import numpy as np
import torch


def one_segm_instance(paths_fp = '../config/paths.json'):

    with open(file = paths_fp, mode = 'r') as json_file:
        paths_json = json.load(json_file)

    dataset_dp = paths_json['raw_ssmoke_data_dp']
    if dataset_dp[-1] != '/': dataset_dp += '/'

    data = data_utils.SegmData(dataset_dp = dataset_dp)
    img, mask = data.get_one_segm_instance_ssmoke_dp()

    image_plot = visuals.SegmVisuals(classes = data.classes)
    image_plot.build_plt(img = img, mask = mask, fig_title = 'Smoke Segmentation')
    image_plot.store_fig(fp = './fig0.jpg')

def one_det_instance(paths_fp = '../config/paths.json'):

    with open(file = paths_fp, mode = 'r') as json_file:
        paths_json = json.load(json_file)

    dataset_dp = paths_json['dfire_data_dp']
    if dataset_dp[-1] != '/': dataset_dp += '/'

    data = data_utils.DetData(dataset_dp = dataset_dp)
    img, bboxes = data.get_one_det_instance_ssmoke()

    image_plot = visuals.DetVisuals(classes = data.classes)
    image_plot.build_plt(img = img, bboxes = bboxes, fig_title = 'Detection')
    image_plot.store_fig(fp = './fig0.jpg')

def one_convert_bboxes_to_segm_mask(paths_fp = '../config/paths.json'):

    with open(file = paths_fp, mode = 'r') as json_file:
        paths_json = json.load(json_file)

    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    dataset_dp = paths_json['dfire_data_dp']
    if dataset_dp[-1] != '/': dataset_dp += '/'
    sam_fp = paths_json['sam_fp']

    data = data_utils.DetData(dataset_dp = dataset_dp)
    img, bboxes = data.get_one_det_instance_ssmoke()

    mask = bbox2segm_mask.bbox2segm_mask(img = img, bboxes = bboxes, sam_fp = sam_fp, DEVICE = DEVICE)

    image_plot = visuals.SegmVisuals(classes = data.classes)
    image_plot.build_plt(img = img, mask = mask, bboxes = bboxes, fig_title = 'Smoke Segmentation')
    # image_plot.display()
    image_plot.store_fig(fp = './fig0.jpg')
    print('Operation completed')

def convert_bboxes_to_segm_mask(paths_fp = '../config/paths.json'):

    def DetectNSave():
        print('Number of bounding boxes: %d'%(len(data.bboxes)))
        mask = bbox2segm_mask.bbox2segm_mask(img = data.img, bboxes = data.bboxes, sam_fp = sam_fp, DEVICE = DEVICE)
        data.save_segm_labels(mask = mask)
        print('Action: Mask saved at\n%s'%(data.mask_fp))

    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    t_total_conversion_init = time.time()
    delta_t_iter_list = []

    with open(file = paths_fp, mode = 'r') as json_file:
        paths_json = json.load(json_file)

    dataset_dp = paths_json['dfire_data_dp']
    if dataset_dp[-1] != '/': dataset_dp += '/'
    save_dp = paths_json['raw_ssmoke_data_dp']
    if save_dp[-1] != '/': save_dp += '/'
    sam_fp = paths_json['sam_fp']

    overwrite_choices = {'O': 'Overwrite', 'I': 'Ignore'}
    inp = input('If already generated output label files are found, how should they be handled?\n[I] Ignore files\n[O] Overwrite files with newly processed ones\n> ')#; inp = 'I'
    while inp not in overwrite_choices.keys():
        inp = input('Select a valid input:\n> ')
    print('Proceeding with:\n%s\n---\n'%overwrite_choices[inp])
    overwrite_switch = inp == 'O'

    data = data_utils.DetData(dataset_dp = dataset_dp, save_dp = save_dp)
    while next(data):

        t_iter_i = time.time()

        if os.path.isfile(data.mask_fp):
            print('W: Mask file already exists')
            if overwrite_switch:
                print('W: Overwritting file')
                DetectNSave()
            else:
                print('W: Ignoring instance\n%s'%(data.mask_fp))
        else:
            DetectNSave()

        delta_t_iter = time.time() - t_iter_i
        delta_t_iter_list.append(delta_t_iter)

        print('Instance conversion period: %.2f s'%(delta_t_iter))
        print('Completion status: %.2f%%'%(100*(data.INSTANCE_IDX+1)/data.n_instances))
        print('---', end = 2 * '\n')

    delta_t_total_conversion = time.time() - t_total_conversion_init

    delta_t_iter_list = np.array(delta_t_iter_list)

    print('\nOperation completed\n')
    print('Conversion per image\navg: %.2f s\nmax: %.2f s\nmin: %.2f s'%(np.mean(delta_t_iter_list), np.max(delta_t_iter_list), np.min(delta_t_iter_list)))
    print('\nTotal conversion period: %.1f s'%(delta_t_total_conversion))


if __name__ == '__main__':

    convert_bboxes_to_segm_mask()
