# Instructions for running on CHTC

0. Create `.txt` file called `neptune-api-token.txt` whose contents are your API token for Neptune.ai.

## Original PANet

1. Follow instructions from original PANet paper repo (**Data Preparation for VOC Dataset** and **Usage**, reproduced below) to prepare the VOC Dataset.
2. Zip and transfer `VOCdevkit` and `pretrained_model.zip` to `.` (`superurop/PANet/`).
3. Create output directory `outputs/` by running `mkdir outputs/` and submit the job to run using `condor_submit panet-voc.sub`.

### Data Preparation for VOC Dataset

0. Download VOC dataset from [here](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit).

1. Download `SegmentationClassAug`, `SegmentationObjectAug`, `ScribbleAugAuto` from [here](https://drive.google.com/drive/folders/1N00R9m9qe2rKZChZ8N7Hib_HR2HGtXHp?usp=sharing) and put them under `VOCdevkit/VOC2012/`.

2. Download `Segmentation` from [here](https://drive.google.com/drive/folders/1N00R9m9qe2rKZChZ8N7Hib_HR2HGtXHp?usp=sharing) and use it to replace `VOCdevkit/VOC2012/ImageSets/Segmentation`.

### Usage

1. Download the ImageNet-pretrained weights of VGG16 network from `torchvision`: [https://download.pytorch.org/models/vgg16-397923af.pth](https://download.pytorch.org/models/vgg16-397923af.pth) and put it under `PANet/pretrained_model/` folder.

2. Change configuration via `config.py`, then train the model using `python train.py` or test the model using `python test.py`. You can use `sacred` features, e.g. `python train.py with gpu_id=2`.

### Citation
Please consider citing our paper if the project helps your research. BibTeX reference is as follows.
```
@InProceedings{Wang_2019_ICCV,
author = {Wang, Kaixin and Liew, Jun Hao and Zou, Yingtian and Zhou, Daquan and Feng, Jiashi},
title = {PANet: Few-Shot Image Semantic Segmentation With Prototype Alignment},
booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
month = {October},
year = {2019}
}
```

## HPA

1. Move grayscale annotation masks (`Greyscale_Images`) and RGB images (e.g,. `PbMgNr`). Run `generate_png_from_tiff.py` and `generate_jpg_from_tiff.py` locally.
2. Zip and transfer `Greyscale_Images_png`, `PbMgNr_jpg` (or whatever other RGB versions of the images are being used), and `pretrained_model.zip` to `.` (`superurop/PANet/`).
3. Create output directory `outputs/` by running `mkdir outputs/` and submit the job to run using `condor_submit panet.sub`.
