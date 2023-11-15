import visuals
import data_tools
import bbox2segm_mask
import json

from pdb import set_trace as pause


def one_segm_instance(paths_fp = '../paths.json'):

    with open(file = paths_fp, mode = 'r') as json_file:
        paths_json = json.load(json_file)

    dataset_dp = paths_json['smoke5k_dp']
    if dataset_dp[-1] != '/': dataset_dp += '/'

    data = data_tools.SegmData(dataset_dp = dataset_dp)
    img, mask = data.get_one_segm_instance_smoke5k_dp()

    image_plot = visuals.SegmVisuals(classes = data.classes)
    image_plot.build_plt(img = img, mask = mask, fig_title = 'Smoke Segmentation')
    image_plot.store_fig()

def one_det_instance(paths_fp = '../paths.json'):

    with open(file = paths_fp, mode = 'r') as json_file:
        paths_json = json.load(json_file)

    dataset_dp = paths_json['ssmoke_data_dp']
    if dataset_dp[-1] != '/': dataset_dp += '/'

    data = data_tools.DetData(dataset_dp = dataset_dp)
    img, bboxes = data.get_one_det_instance_ssmoke()

    image_plot = visuals.DetVisuals(classes = data.classes)
    image_plot.build_plt(img = img, bboxes = bboxes, fig_title = 'Detection')
    image_plot.store_fig()

def one_convert_bboxes_to_segm_mask(paths_fp = '../paths.json'):

    with open(file = paths_fp, mode = 'r') as json_file:
        paths_json = json.load(json_file)

    dataset_dp = paths_json['ssmoke_data_dp']
    if dataset_dp[-1] != '/': dataset_dp += '/'

    sam_fp = paths_json['sam_fp']

    data = data_tools.DetData(dataset_dp = dataset_dp)
    img, bboxes = data.get_one_det_instance_ssmoke()

    mask = bbox2segm_mask.bbox2segm_mask(img = img, bboxes = bboxes, sam_fp = sam_fp)

    image_plot = visuals.SegmVisuals(classes = data.classes)
    image_plot.build_plt(img = img, mask = mask, bboxes = bboxes, fig_title = 'Smoke Segmentation')
    # image_plot.display()
    image_plot.store_fig()
    print('Operation completed')

def convert_bboxes_to_segm_mask(paths_fp = '../paths.json'):

    with open(file = paths_fp, mode = 'r') as json_file:
        paths_json = json.load(json_file)

    dataset_dp = paths_json['ssmoke_data_dp']
    if dataset_dp[-1] != '/': dataset_dp += '/'

    sam_fp = paths_json['sam_fp']

    data = data_tools.DetData(dataset_dp = dataset_dp)
    while next(data):
        print('W: Debugging mode is on')
        mask = bbox2segm_mask.bbox2segm_mask(img = data.img, bboxes = data.bboxes, sam_fp = sam_fp)
        data.save_segm_labels(mask = mask)
        print('Completion status: %.2f%%'%(100*(data.INSTANCE_IDX+1)/(2+1)))
        print('---', end = 2 * '\n')

    print('Operation completed')


if __name__ == '__main__':

    CONVERT_BBOXES_TO_SEGMENTATION_MASKS = True

    if CONVERT_BBOXES_TO_SEGMENTATION_MASKS:
        convert_bboxes_to_segm_mask()