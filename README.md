# Smoke Segmentation

The objective of this repository is to provide the means to create a large image dataset for the task of smoke segmentation. It also provides the associated tools for image data curation and a guide to automatically convert a large amount of image detection-based labels to image segmentation labels.

### Upcoming Release Plans

`v0.7.0`: The repository will encompass the implementation of a proposed architecture for the segmenter, its training algorithm, a pre-trained segmenter for smoke image segmentation, as well as a demonstration of the process.

## Dataset

### External Dataset

#### D-Fire

D-Fire [[link](https://github.com/gaiasd/DFireDataset)] is an image dataset of fire and smoke occurrences designed for machine learning and object detection algorithms with more than 21,000 images. The bounding box labels are stored inside `.txt` files in YOLO format (class + cxcywh).

### Directory

```
.
├── fig0.jpg
├── LICENSE
├── models
├── paths.json [Developer must create this]
├── README.md
└── src
    ├── bbox2segm_mask.py
    ├── data_tools.py
    ├── main.py
    └── visuals.py
```

### S-Smoke

Consisted of all D-Fire's images, with segmentation mask labels for the task of image smoke segmentation. The ground truth labels were generated from a pre-trained SAM model, guided by the bounding box labels of the D-Fire dataset.

#### Preparing to Convert D-Fire to S-Smoke

To replicate the steps involved for the generation of image labels, download and decompress the D-Fire dataset; with the following directory structure
```
D-Fire
├── test
|   ├── labels
|   └── images
└── train
    ├── labels
    └── images
```
To align the paths with the currently provided scripts, you should rename the directories with names `labels` to `det_labels`. Additionally add two directories inside `test` and `train` respectively, both with names `segm_labels`. After the proper application of the preceding instructions, the resulting dataset's structure will look exactly like this
```
D-Fire
├── test
|   ├── det_labels
|   ├── images
|   └── segm_labels
└── train
    ├── det_labels
    ├── images
    └── segm_labels
```
Install `segment-anything` [[link](https://github.com/facebookresearch/segment-anything)] through
```
python3 -m pip install git+https://github.com/facebookresearch/segment-anything.git
```
which is based on a pre-trained segmentation model called SAM. Now download SAM through [[link](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)] and place it at the `./models` directory.

#### Setting Up All Paths

It is assumed that the user has downloaded and decompressed the corresponding data files. The paths of the data's directory along with the SAM model's file path should be specified inside `./paths.json`.
```
{
    "ssmoke_data_dp": "<INSERT D-FIRE DATASET DIRECTORY (e.g. ~/Downloads/D-Fire)>",
    "sam_fp": "<SAM MODEL FILE PATH (e.g. ../models/sam_vit_h_4b8939.pt)>"
}
```

#### Conversion from D-Fire to S-Smoke

Inside `main.py`, set `CONVERT_BBOXES_TO_SEGMENTATION_MASKS` to `True`. Navigate in `./src` and apply
```
python3 main.py
```
This process will potentially take at least one day to complete.

## Citation

- <p align="justify">Pedro Vinícius Almeida Borges de Venâncio, Adriano Chaves Lisboa, Adriano Vilela Barbosa: <a href="https://link.springer.com/article/10.1007/s00521-022-07467-z"> An automatic fire detection system based on deep convolutional neural networks for low-power, resource-constrained devices. </a> In: Neural Computing and Applications, 2022.</p>

- Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A., Lo, W.Y., Dollar, P., & Girshick, R. (2023). Segment Anything. arXiv:2304.02643.


