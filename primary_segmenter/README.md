# Smoke Segmenter

This directory is heavily based on a modified version of the TorchSeg [[link](https://github.com/ycszen/TorchSeg)] project, with git commit hash `62eeb159aee77972048d9d7688a28249d3c56735`. It is utilized to train and apply estimations for the task of smoke segmentation on the S-Smoke dataset.

## Setup

Navigate inside `./datasets/config_new` and apply
```
python3 gen_ssmoke_path_lists.py
```
to generate all instance file paths in order for TorchSeg to be able to find the dataset's paths.

## Available Pretrained Models

- `R18`. Path `./primary_segmenter/TorchSeg/model/bisenet/cityscapes.bisenet.R18`.

### Notable Functionality

This section assumes that you have followed the instructions of `/README.md` and `./data_tools/README.md`.

#### R18

Navigate to the respective model directory (e.g. `cityscapes.bisenet.R18`).

To train a pretrained model on your variant of `S-Smoke`, simply apply
```
python3 train.py
```
If a previous training session was interrupted, you could also override the model's initialization by applying
```
python3 train.py -c ./log/snapshot/epoch-last.pth
```
To use the model for prediction/estimation on the validation set apply
```
python3 eval.py
```
You can use the `-s` option to cycle through the validation set and display a slideshow of prediction plots.

<!-- {fig0.png} -->

## Added Utilities

- During training, the mean IU plot (computed on the validation set) is updated epoch-wise and stoed as `metrics.png`. This image file is located inside the respective model directory.