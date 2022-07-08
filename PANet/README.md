# About

This subdirectory utilizes the official implementation of [PANet](https://github.com/kaixin96/PANet) as part of explorations of existing few-shot learning architectures for the task of biological cell image segmentation. The immediate ongoing goal is to adapt PANet to accept the novel HPA dataset from this project; subsequent steps may include experimentation with various CNN backbones (to replace the default VGG-16) and/or investigating other few-shot architectures.


# Setup

This part of the project uses [this Docker Hub repo](https://hub.docker.com/repository/docker/teresahgao/superurop-panet).

## Instructions for running on CHTC

Training jobs have been run on free GPU resources provided by [CHTC](https://chtc.cs.wisc.edu). Use of CHTC computing is not required, though the repository already contains the necessary `.sub` and related files to run these jobs on the server. To run training jobs locally, you will need to manually set up the Docker Hub containers [teresahgao/superurop-grayscale-cnn](https://hub.docker.com/repository/docker/teresahgao/superurop-grayscale-cnn) and [teresahgao/superurop-panet](https://hub.docker.com/repository/docker/teresahgao/superurop-panet).

0. Create `.txt` file called `neptune-api-token.txt` whose contents are your API token for Neptune.ai.

> Note: if you are not running on CHTC, then you can permanently configure the Neptune API token using a command line call.

### Original PANet

The following are instructions to replicate the published results from the [original PANet paper](https://arxiv.org/abs/1908.06391) via the [official code repo](https://github.com/kaixin96/PANet). The dataset used is VOCdevkit, and the backbone used is a pretrained VGG16.

1. Follow instructions from original PANet paper repo (**Data Preparation for VOC Dataset** and **Usage**, reproduced below) to prepare the VOC Dataset.
2. Zip and transfer `VOCdevkit` and `pretrained_model.zip` to `.` (`superurop/PANet/`).
3. Create output directory `outputs/` by running `mkdir outputs/` and submit the job to run using `condor_submit panet-voc.sub`.

#### Data Preparation for VOC Dataset

0. Download VOC dataset from [here](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit).

1. Download `SegmentationClassAug`, `SegmentationObjectAug`, `ScribbleAugAuto` from [here](https://drive.google.com/drive/folders/1N00R9m9qe2rKZChZ8N7Hib_HR2HGtXHp?usp=sharing) and put them under `VOCdevkit/VOC2012/`.

2. Download `Segmentation` from [here](https://drive.google.com/drive/folders/1N00R9m9qe2rKZChZ8N7Hib_HR2HGtXHp?usp=sharing) and use it to replace `VOCdevkit/VOC2012/ImageSets/Segmentation`.

#### Usage

1. Download the ImageNet-pretrained weights of VGG16 network from `torchvision`: [https://download.pytorch.org/models/vgg16-397923af.pth](https://download.pytorch.org/models/vgg16-397923af.pth) and put it under `PANet/pretrained_model/` folder.

2. Change configuration via `config.py`, then train the model using `python train.py` or test the model using `python test.py`. You can use `sacred` features, e.g. `python train.py with gpu_id=2`.

#### Citation of original authors
```
@InProceedings{Wang_2019_ICCV,
author = {Wang, Kaixin and Liew, Jun Hao and Zou, Yingtian and Zhou, Daquan and Feng, Jiashi},
title = {PANet: Few-Shot Image Semantic Segmentation With Prototype Alignment},
booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
month = {October},
year = {2019}
}
```

### HPA

The current task is adapting the PANet for the HPA dataset, still with the same default VGG16 backbone. While a successful adaptation has not yet been achieved, the details of the debugging process so far can be found at https://github.com/broadinstitute/ip-superurop-t_gao/discussions/11.

1. Move grayscale annotation masks (`Greyscale_Images`) and RGB images (e.g,. `PbMgNr`). Run `generate_png_from_tiff.py` and `generate_jpg_from_tiff.py` locally.
2. Zip and transfer `Greyscale_Images_png`, `PbMgNr_jpg` (or whatever other RGB versions of the images are being used), and `pretrained_model.zip` to `.` (`superurop/PANet/`).
3. Create output directory `outputs/` by running `mkdir outputs/` and submit the job to run using `condor_submit panet.sub`.


# Making Changes

## CHTC files

Each CHTC job requires a `.sub` file for submission to the server. This `.sub` file transfers the specified input files to the server and runs the `.sh` file, which in turn runs the training/testing `.py` file.

For more information, see [`../CHTC_guidebook.md`](../CHTC_guidebook.md).

### `config` files

The default `config` file, `config.py`, runs PANet with the HPA dataset.

The alternate `config` file, `config_voc.py`, runs PANet with the VOC dataset.

The correct `config` file is loaded by default when submitting `panet.sub` versus `panet-voc.sub`, respectively.

## `train.py`

`train.py` loads the dataset from `dataloaders/` and the fewshot model from `models/`.

The dataloader for the VOC dataset, implemented as the method `voc_fewshot()` in `dataloaders/customized.py`, should work successfully.

Debugging efforts are ongoing for the dataloader for the HPA dataset, implemented in two parts: first as transforms are applied in `dataloaders/hpa.py` and then within the method `hpa_fewshot()` in `dataloaders/customized.py` to create subdatasets.
