# Start here

## Summary

### High-level goal

Adapt existing few-shot learning architectures such as [PANet](https://arxiv.org/abs/1908.06391) to biological cell images from sources such as the [Human Protein Atlas](https://www.proteinatlas.org/humanproteome/pathology). In terms of implementation, this might mean using a CNN backbone pre-trained on biological cell images.

### Current progress

Some work has been done investigating the use of a ResNet architecture to replace the default PANet backbone. More recent work has focused on adapting the vanilla PANet architecture to accept an original image dataset from the Human Protein Atlas.

> For more information, see https://github.com/broadinstitute/ip-superurop-t_gao/discussions/11


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
