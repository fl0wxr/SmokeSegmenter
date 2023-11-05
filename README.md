# Forest Smoke Segmentation

The objective of this repository is to serve as a central resource for accessible raw data, both labeled and unlabeled, specifically tailored for image segmentation tasks related to the smoke class. In addition to the smoke image segmentation data, it also provides tools for data curation. The repository encompasses the implementation of the current proposed architecture for the segmenter, its training algorithm, a pre-trained segmenter for smoke image segmentation, as well as a demonstration of the process, all accompanied by concise documentation of the associated pipeline.

## Dataset

### External Datasets

#### 1. UNETSMOKE

UNETSMOKE [[link](https://github.com/sonvbhp199/Unet-Smoke)] is an image dataset containing 2,000 images in total. Each image is paired with a binary segmentation mask that splits the image into smoke vs non-smoke area.

#### 2. SMOKE5K

SMOKE5K [[link](https://github.com/SiyuanYan1/Transmission-BVM)] is an image dataset containing 5,400 images in total. The dataset is consisted of 1,400 real and 4,000 synthetic images. Each image is paired with a binary segmentation mask that splits the image into smoke vs non-smoke area.

An example of a segmented image with resolution (256, 256).

![](./fig0.jpeg)

## Installation

It is assumed that the user has downloaded and decompressed the corresponding data files. The paths of the resulting directories should be specified inside `./paths.json` that contains all the paths needed to get the data.

```
{
    "UNETSMOKE_dp": "<INSERT DATASET DIRECTORY>",
    "SMOKE5K_dp": "<INSERT DATASET DIRECTORY>"
}
```

### Directory Structure

```
.
├── datasets
|   ├── UNETSMOKE
|   └── SMOKE5K
├── README.md
└── src
    ├── parser.py
    └── visuals.py
```

### Usage

Before executing any code, first navigate in `./src`.

