# Primary Segmenters - Fine Tuning

This directory holds all segmentation scripts used to apply fine tuning on the S-Smoke dataset. These scripts and their directories are based on modified versions of

- TorchSeg [[link](https://github.com/ycszen/TorchSeg)] project, with git commit hash `62eeb159aee77972048d9d7688a28249d3c56735`
- PIDNet [[link](https://github.com/XuJiacong/PIDNet)] project, with git commit hash `fefa51716bddc13a4321af2c70a074367100645a`

This section assumes that you have followed the instructions of `/README.md` and `./data_tools/README.md`.

## Setup

Navigate inside `./data_tools/src` and apply
```
python3 gen_ssmoke_path_lists.py
```
to generate all instance file paths for the primary segmenters to be able to find the corresponding instance files.

## TorchSeg

### Available Pretrained Models

- BiSeNet - R18. Path `./primary_segmenter/TorchSeg/model/cityscapes.bisenet.R18` with configuration file `config.py`.

#### R18

Navigate inside the respective model directory (e.g. `cityscapes.bisenet.R18`).

First the `config.py` has to be examined and be set correctly.

To train a model on S-Smoke apply
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
You can use the `-s` option to enable a slideshow of figures showcasing the prediction of each individual validation instance. Each such figure is composed of 3 stacked images (input - prediction - ground truth).

All the predictions of validation instances, along with their mentioned prediction results, can be stored at a directory path set by the argument `-p` right after their corresponding segmentation masks are produced. An example follows
```
python3 eval.py -p ../../predicted_results
```

### Added Utilities

- During training, the mean IU plot (computed on the validation set) is updated epoch-wise and stored as `metrics.png` helping the AI developer to monitor the training's progress. This image file is located inside the respective model directory.

## PIDNet

### Available Pretrained Models

- Small PIDNet. Configuration file: `./configs/camvid/pidnet_small_camvid.yaml`.

#### Small PIDNET

Navigate in `./primary_segmenters/PIDNet`.

To train a model on S-Smoke apply
```
python3 tools/train.py --cfg configs/camvid/pidnet_small_camvid.yaml GPUS "(0,)"
```
To use the model for prediction/estimation on the validation set apply
```
python3 tools/eval.py --cfg configs/camvid/pidnet_small_camvid.yaml GPUS "(0,)"
```
To produce segmentation mask files and their combined images one may use `tools/custom.py`.