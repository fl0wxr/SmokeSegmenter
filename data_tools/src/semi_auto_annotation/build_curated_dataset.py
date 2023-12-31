import os
import shutil
import json
from glob import glob


def build_curated_dataset(SRC_DP, TGT_DP, SPLIT_RATIO, BLACKLIST_FP = '../blacklist/blacklisted_instances.list', SESSION_FP = '../blacklist/session.path'):

    ## These are to be deleted
    old_tgt_img_fps = glob\
    (
        pathname = os.path.join(TGT_DP, '**/*.jpg'),
        recursive = True
    )
    old_tgt_label_fps = glob\
    (
        pathname = os.path.join(TGT_DP, '**/*.png'),
        recursive = True
    )

    old_tgt_img_fps = sorted(old_tgt_img_fps, reverse = True)
    old_tgt_label_fps = sorted(old_tgt_label_fps, reverse = True)

    ## ! Get all relevant paths: Begin

    src_img_fps = glob\
    (
        pathname = os.path.join(SRC_DP, '**/*.jpg'),
        recursive = True
    )
    src_label_fps = glob\
    (
        pathname = os.path.join(SRC_DP, '**/*.png'),
        recursive = True
    )

    src_img_fps = sorted(src_img_fps, reverse = True)
    src_label_fps = sorted(src_label_fps, reverse = True)

    if os.path.exists(BLACKLIST_FP):
        with open(BLACKLIST_FP, 'r') as file:
            blacklist_str = file.read()
    else:
        print('W: No blacklisted file list found; proceeding with unconstrained copy')

    blacklist_paths = blacklist_str.split('\n')
    if blacklist_paths[-1] == '': blacklist_paths = blacklist_paths[:-1]
    blacklist_img_fps = []
    blacklist_label_fps = []
    for blacklist_fp_pair in blacklist_paths:
        blacklist_img_fp, blacklist_label_fp = blacklist_fp_pair.split(', ')
        blacklist_img_fps.append(blacklist_img_fp)
        blacklist_label_fps.append(blacklist_label_fp)
        assert os.path.splitext(os.path.basename(blacklist_img_fp))[0] == os.path.splitext(os.path.basename(blacklist_label_fp))[0], 'E: Inconsistency in recorded blacklist filepath pairs'
        assert blacklist_img_fp in src_img_fps and blacklist_label_fp in src_label_fps, 'E: There are blacklisted paths that are missing from the filesystem'

    ## ! Get all relevant paths: End

    del blacklist_fp_pair

    ## Clean target directories
    for fp in old_tgt_img_fps + old_tgt_label_fps:
        os.remove(fp)

    with open(file = SESSION_FP, mode = 'r') as f:
        checkpoint = f.read()

    ## Drop all blacklisted file paths
    filtered_src_img_fps = []
    filtered_src_label_fps = []
    for src_instance_idx in range(len(src_img_fps)):
        src_img_fp = src_img_fps[src_instance_idx]
        src_label_fp = src_label_fps[src_instance_idx]

        ## Stopping criterion considers only those files that were checked until the manual evaluation's checkpoint; this implementation is based on the ordering of the path lists
        if src_label_fp < checkpoint:
            break

        if src_label_fp not in blacklist_label_fps:
            filtered_src_img_fps.append(src_img_fp)
            filtered_src_label_fps.append(src_label_fp)

    ## ! Split and save: Begin

    test_instances_idx_offset = int(SPLIT_RATIO * len(filtered_src_label_fps))

    test_filtered_src_img_fps = filtered_src_img_fps[:test_instances_idx_offset]
    test_filtered_src_label_fps = filtered_src_label_fps[:test_instances_idx_offset]
    train_filtered_src_img_fps = filtered_src_img_fps[test_instances_idx_offset:]
    train_filtered_src_label_fps = filtered_src_label_fps[test_instances_idx_offset:]

    ## Test set
    for src_instance_idx in range(len(test_filtered_src_img_fps)):
        src_img_fp = test_filtered_src_img_fps[src_instance_idx]
        src_label_fp = test_filtered_src_label_fps[src_instance_idx]
        tgt_img_fp = os.path.join(TGT_DP, 'test', 'images', os.path.basename(src_img_fp))
        tgt_label_fp = os.path.join(TGT_DP, 'test', 'seg_labels', os.path.basename(src_label_fp))
        shutil.copy(src = src_img_fp, dst = tgt_img_fp)
        shutil.copy(src = src_label_fp, dst = tgt_label_fp)

    ## Train set
    for src_instance_idx in range(len(train_filtered_src_img_fps)):
        src_img_fp = train_filtered_src_img_fps[src_instance_idx]
        src_label_fp = train_filtered_src_label_fps[src_instance_idx]
        tgt_img_fp = os.path.join(TGT_DP, 'train', 'images', os.path.basename(src_img_fp))
        tgt_label_fp = os.path.join(TGT_DP, 'train', 'seg_labels', os.path.basename(src_label_fp))
        shutil.copy(src = src_img_fp, dst = tgt_img_fp)
        shutil.copy(src = src_label_fp, dst = tgt_label_fp)

    ## ! Split and save: End


if __name__ == '__main__':

    paths_fp = '../config/paths.json'
    with open(file = paths_fp, mode = 'r') as json_file:
        paths_json = json.load(json_file)

    build_curated_dataset(SRC_DP = paths_json['raw_ssmoke_data_dp'], TGT_DP = paths_json['curated_ssmoke_data_dp'], SPLIT_RATIO = 0.3)
