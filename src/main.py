import visuals
import data_tools
import bbox2segm_mask
import json

from pdb import set_trace as pause


def one_segm_instance(json_fp = '../paths.json'):

    with open(file = json_fp, mode = 'r') as json_file:
        paths_json = json.load(json_file)

    dataset_dp = paths_json['smoke5k_dp']
    if dataset_dp[-1] != '/': dataset_dp += '/'

    data = data_tools.SegmData(dataset_dp = dataset_dp)
    img, mask = data.get_one_segm_instance_smoke5k_dp()

    image_plot = visuals.SegmVisuals(classes = data.classes)
    image_plot.build_plt(img = img, mask = mask, fig_title = 'Segmentation Testing')
    image_plot.store_fig()

def one_det_instance(json_fp = '../paths.json'):

    with open(file = json_fp, mode = 'r') as json_file:
        paths_json = json.load(json_file)

    dataset_dp = paths_json['dfire_dp']
    if dataset_dp[-1] != '/': dataset_dp += '/'

    data = data_tools.DetData(dataset_dp = dataset_dp)
    img, bboxes = data.get_one_det_instance_dfire()

    image_plot = visuals.DetVisuals(classes = data.classes)
    image_plot.build_plt(img = img, bboxes = bboxes, fig_title = 'Detection Testing')
    image_plot.store_fig()

def one_convert_bboxes_to_segm_mask(json_fp = '../paths.json'):

    with open(file = json_fp, mode = 'r') as json_file:
        paths_json = json.load(json_file)

    dataset_dp = paths_json['dfire_dp']
    if dataset_dp[-1] != '/': dataset_dp += '/'

    sam_fp = paths_json['sam_fp']

    data = data_tools.DetData(dataset_dp = dataset_dp)
    img, bboxes = data.get_one_det_instance_dfire()

    mask = bbox2segm_mask.bbox2segm_mask(img = img, bboxes = bboxes, sam_fp = sam_fp)

    image_plot = visuals.SegmVisuals(classes = data.classes)
    image_plot.build_plt(img = img, mask = mask, bboxes = bboxes, fig_title = 'Segmentation Testing')
    # image_plot.display()
    image_plot.store_fig()

def convert_bboxes_to_segm_mask(json_fp = '../paths.json'):

    with open(file = json_fp, mode = 'r') as json_file:
        paths_json = json.load(json_file)

    dataset_dp = paths_json['dfire_dp']
    if dataset_dp[-1] != '/': dataset_dp += '/'

    sam_fp = paths_json['sam_fp']

    processed_dataset_dp = paths_json['dfire_smoke_only_pos_dp']
    if processed_dataset_dp[-1] != '/': processed_dataset_dp += '/'

    data = data_tools.DetData(dataset_dp = dataset_dp, multiple_instances = True, processed_dataset_dp = processed_dataset_dp)

    while next(data):
        data.save_pos_smoke()


if __name__ == '__main__':

    convert_bboxes_to_segm_mask()