import sys

sys.path.append('../utils')

import visuals
import data_utils
import json
import os
import numpy as np


def combine_input_with_seg_mask(paths_fp = '../config/paths.json'):

    with open(file = paths_fp, mode = 'r') as json_file:
        paths_json = json.load(json_file)

    dataset_dp = paths_json['curated_ssmoke_data_dp']
    if dataset_dp[-1] != '/': dataset_dp += '/'

    train_dp = os.path.join(dataset_dp, 'train')
    test_dp = os.path.join(dataset_dp, 'test')

    for data_dp in (train_dp, test_dp):

        print('Set @ ' + data_dp)
        data = data_utils.SegmData(dataset_dp = data_dp)

        while next(data):

            image_plot = visuals.SegmVisuals(classes = data.classes)
            image_plot.build_plt(img = data.img, mask = data.mask, fig_title = 'Smoke Segmentation')
            # image_plot.display()

            image_plot.store_fig(fp = os.path.join(data_dp, 'combined', os.path.basename(data.img_fp)))


if __name__ == '__main__':

    combine_input_with_seg_mask()
    print('Operation completed')




