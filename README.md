# Forest Smoke Segmentation

The objective of this repository is to serve as a central reference point for accessible raw data, both labeled and unlabeled, specifically tailored for image segmentation and detection tasks related to the smoke class. In addition to the smoke image segmentation data, it also provides tools for image data curation. Also a guide to automatically convert a large amount of image detection-based labels to image segmentation labels. All accompanied by concise documentation of the associated pipeline.

### Next Updates

The repository will encompass the implementation of a proposed architecture for the segmenter, its training algorithm, a pre-trained segmenter for smoke image segmentation, as well as a demonstration of the process.

## Dataset

### External Datasets

#### 1. UNETSMOKE

UNETSMOKE [[link](https://github.com/sonvbhp199/Unet-Smoke)] is an image dataset containing 2,000 images in total. Each image is paired with a binary segmentation mask that splits the image into smoke vs non-smoke area.

#### 2. SMOKE5K

SMOKE5K [[link](https://github.com/SiyuanYan1/Transmission-BVM)] is an image dataset containing 5,400 images in total. The dataset is consisted of 1,400 real and 4,000 synthetic images. Each image is paired with a binary segmentation mask that splits the image into smoke vs non-smoke area.

![](./fig0.jpeg)

#### 3. D-Fire

D-Fire [[link](https://github.com/gaiasd/DFireDataset)] is an image dataset of fire and smoke occurrences designed for machine learning and object detection algorithms with more than 21,000 images.

### Directory Structure

```
.
├── datasets
|   ├── UNETSMOKE
|   └── SMOKE5K
├── README.md
├── src
|   ├── parser.py
|   └── visuals.py
└── paths.json
```

### Installation

If you seek to convert a set of bounding box labels into segmentation masks, then we shall install `segment-anything` [[link](https://github.com/facebookresearch/segment-anything)] through
```
python3 -m pip install git+https://github.com/facebookresearch/segment-anything.git
```
which is based on a pre-trained segmentation model called SAM. Download this model [[link](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)] and place it at the `./models` directory.

### Usage

Before executing any code, first navigate in `./src`. It is assumed that the user has downloaded and decompressed the corresponding data files. The paths of the resulting data directories along with the SAM model's file path should be specified inside `./paths.json`.
```
{
    "smoke5k_dp": "<INSERT SMOKE 5K DATASET DIRECTORY>",
    "dfire_dp": "<INSERT D-FIRE DATASET DIRECTORY>",
    "sam_fp": "<SAM MODEL FILE>"
}
```

<!-- #### Convert Detection Labels to Segmentation Labels through SAM

\[Empty\] -->



## Citation

- <p align="justify">Pedro Vinícius Almeida Borges de Venâncio, Adriano Chaves Lisboa, Adriano Vilela Barbosa: <a href="https://link.springer.com/article/10.1007/s00521-022-07467-z"> An automatic fire detection system based on deep convolutional neural networks for low-power, resource-constrained devices. </a> In: Neural Computing and Applications, 2022.</p>

- Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A., Lo, W.Y., Dollar, P., & Girshick, R. (2023). Segment Anything. arXiv:2304.02643.


