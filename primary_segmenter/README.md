# Smoke Segmenter

This directory is based on a modified version of TorchSeg [[link](https://github.com/ycszen/TorchSeg)] with commit hash `62eeb159aee77972048d9d7688a28249d3c56735`. It is utilized to train and apply estimations for the task of smoke segmentation on the S-Smoke dataset.

## Setup

Navigate inside `./datasets/config_new` and apply
```
python3 gen_ssmoke_path_lists.py
```
to generate all instance file paths in order for TorchSeg to be able to find the dataset's paths.

## Available Pretrained Models

- `cityscapes.bisenet.R18`