# Start here

## Summary

### High-level goal

Adapt existing few-shot learning architectures such as [PANet](https://arxiv.org/abs/1908.06391) to biological cell images from sources such as the [Human Protein Atlas (HPA)](https://www.proteinatlas.org/humanproteome/pathology). In terms of implementation, this might mean using a CNN backbone pre-trained on biological cell images.

### Current progress

Some work has been done investigating the use of a ResNet architecture to replace the default PANet backbone. More recent work has focused on adapting the vanilla PANet architecture to accept an original image dataset from the Human Protein Atlas.

#### Tools

The dataset used is generated from HPA images. The directory, `2021_10_06_HumanProteinAtlas_CiminiLab_TeresaGao`, may be accessed by connecting to the server `smb://hydrogen/imaging_analysis` via the [Broad Institute VPN](vpn.broadinstitute.org). In this directory are the JPG images as originally downloaded from HPA, grayscale masks, and color rotations (e.g., an image with blue nuclei, red microtubules, and green actin filaments recolored to have green nuclei, blue microtubules, and red actin filaments to counteract any possible unwanted effect of color on model training).

Training jobs have been run on free GPU resources provided by [CHTC](https://chtc.cs.wisc.edu). Use of CHTC computing is not required, though the repository already contains the necessary `.sub` and related files to run these jobs on the server. To run training jobs locally, you will need to manually set up the Docker Hub containers [teresahgao/superurop-grayscale-cnn](https://hub.docker.com/repository/docker/teresahgao/superurop-grayscale-cnn) and [teresahgao/superurop-panet](https://hub.docker.com/repository/docker/teresahgao/superurop-panet).

#### CNN backbone

Instructions for setting up and running the CNN locally are included in `CNN-backbone/README.md`.

A CNN backbone was initially implementing as an AlexNet. The AlexNet overfit as expected, with about 0.7 accuracy, when presented with a very small amount of data (about 6 images per class. However, an inexplicable plateauing accuracy around 0.4 was observed when the number of images per class was increased, even when the training images used was increased.

Due to this performance plateau, as well as the relative age of AlexNet, ResNet was next considered. Debugging this network was also time-consuming; for the sake of project progress, because of the amount of time already spent on AlexNet was nontrivial, it was decided to switch focus to investigating adaptations to PANet and double back to the issue of the CNN backbone should it become necessary to improve PANet performance.

See `CNN-backbone/README.md` for more information.

#### PANet adaptation

Thus far, the original results from the [PANet paper](https://arxiv.org/abs/1908.06391) have been successfully replicated via the [official repo](https://github.com/kaixin96/PANet). The work that remains

See https://github.com/broadinstitute/ip-superurop-t_gao/discussions/11 and `PANet/README.md` for more information.


## Detailed motivation

Machine learning, which has shown promise in everything from spam filters to music recommendations, has the potential to become a powerful ubiquitous tool. But while it can be used to perform simple or repetitive tasks, it is less successful at automating more complicated processes such as those necessitated by researchers. This limitation is especially frustrating for many biologists, who must often laboriously annotate large quantities of data before deep learning can be applied to tasks. This is because **most projects using biological images require object detection and/or segmentation so that the component objects can be measured and/or classified to describe the biological phenotype: any extra burden at the step of finding objects decreases the likelihood an image can be used for biological discovery.**

Fortunately, techniques such as one-shot and few-shot learning can significantly reduce the amount of annotated data needed to train a network. **Few-shot learning aims to learn with just a few examples, or with a single example in the case of one-shot learning** — more efficient compared to methods in standard supervised learning, which typically require a large number of ground-truth data points to successfully train deep neural networks. Methods that solve the few-shot learning problem propose algorithms tailored to learn to solve a new task with a limited number of data points.

Although one-shot and few-shot learning are well-researched for deep learning classification tasks, the application of these techniques to segmentation and object detection has been less explored. Therefore, **the goal of this project is to research one-shot and few-shot learning strategies that exist in the deep learning literature with the aim of generalizing them to new classes of architectures**, transforming them from classification tools to segmentation or detection tools. Once implemented, the degree of performance improvement achieved on segmentation tasks in comparison to current machine learning techniques can then be evaluated using real biological image data from a variety of domains, such as fluorescent images and unstained label-free cells.



# Repo structure

## directories

### Human-Protein-Atlas/

Download image data from the [Human Protein Atlas](https://www.proteinatlas.org).

### CNN-backbone/

Implement AlexNet CNN backbone. (ResNet backbone implementation is in progress on branch [resnet-backbone](https://github.com/broadinstitute/ip-superurop-t_gao/tree/resnet-backbone).)

### PANet/

Run [PANet](https://github.com/kaixin96/PANet) using VOCdevkit or HPA dataset.

## files

### CHTC_guidebook.md

Basic setup instructions for [CHTC](https://chtc.cs.wisc.edu)
