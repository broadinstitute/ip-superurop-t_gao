# Summary

Adapt existing few-shot learning architectures such as [PANet](https://arxiv.org/abs/1908.06391) to biological cell images from sources such as the [Human Protein Atlas (HPA)](https://www.proteinatlas.org/humanproteome/pathology). In terms of implementation, this may mean using a CNN backbone pre-trained on biological cell images in such architectures.

Some work has been done investigating the use of a ResNet architecture to replace the default PANet backbone. More recent work has focused on adapting the vanilla PANet architecture to accept an original image dataset from the Human Protein Atlas.

For more information, see [SuperUROP project proposal](https://docs.google.com/document/d/1xxzvYFNDUMMzaYmyn2nCEQ5F0Bjo43BPgvgW6_6UEkg/edit#) for a detailed background and research vision.

## Links

### Subprojects

- [Human-Protein-Atlas/README.md](Human-Protein-Atlas/README.md): **COMPLETED**
- [CNN-backbone/README.md](CNN-backbone/README.md): **INACTIVE**
- [PANet/README.md](PANet/README.md): **ACTIVE**

### Other

- [log (part 1)](https://docs.google.com/document/d/1OHJpOZrEiuWCtvU7-S1mA5YAt8x8PVPNHAKtJnNuG6I/edit): high-level progress summaries and weekly tasks
- [log (part 2)](https://github.com/broadinstitute/ip-superurop-t_gao/discussions/11): debugging details
- [CHTC_guidebook.md](CHTC_guidebook.md): instructions for using CHTC resources


# Details

## Motivation

Machine learning, which has shown promise in everything from spam filters to music recommendations, has the potential to become a powerful ubiquitous tool. But while it can be used to perform simple or repetitive tasks, it is less successful at automating more complicated processes such as those necessitated by researchers. This limitation is especially frustrating for many biologists, who must often laboriously annotate large quantities of data before deep learning can be applied to tasks. This is because **most projects using biological images require object detection and/or segmentation so that the component objects can be measured and/or classified to describe the biological phenotype: any extra burden at the step of finding objects decreases the likelihood an image can be used for biological discovery.**

Fortunately, techniques such as one-shot and few-shot learning can significantly reduce the amount of annotated data needed to train a network. **Few-shot learning aims to learn with just a few examples, or with a single example in the case of one-shot learning** — more efficient compared to methods in standard supervised learning, which typically require a large number of ground-truth data points to successfully train deep neural networks. Methods that solve the few-shot learning problem propose algorithms tailored to learn to solve a new task with a limited number of data points.

Although one-shot and few-shot learning are well-researched for deep learning classification tasks, the application of these techniques to segmentation and object detection has been less explored. Therefore, **the goal of this project is to research one-shot and few-shot learning strategies that exist in the deep learning literature with the aim of generalizing them to new classes of architectures**, transforming them from classification tools to segmentation or detection tools. Once implemented, the degree of performance improvement achieved on segmentation tasks in comparison to current machine learning techniques can then be evaluated using real biological image data from a variety of domains, such as fluorescent images and unstained label-free cells.

## HPA dataset

> STATUS: COMPLETED

This project uses a novel dataset generated from on the [pathology.tsv](https://www.proteinatlas.org/about/download) dataset of the [Human Protein Atlas](https://www.proteinatlas.org).

## CNN backbone

> STATUS: INACTIVE
> To be resumed should the need arise for a more specialized CNN backbone in PANet

Initially, several CNNs were investigated as potential backbones for existing one- and few-shot architectures. Since most of those architectures rely on CNNs pre-trained on large but general datasets such as ImageNet, the idea was that a CNN pre-trained on HPA or related biological cell images might produce better results for a model categorizing

A CNN backbone was first implemented as an AlexNet. The AlexNet overfit as expected, with about 0.7 accuracy, when presented with a very small amount of data (about 6 images per class. However, an inexplicable plateauing accuracy around 0.4 was observed when the number of images per class was increased, even when the training images used was increased.

Due to this performance plateau, as well as the relative age of AlexNet, ResNet was next considered. Debugging this network was also time-consuming; for the sake of project progress, because of the amount of time already spent on AlexNet was nontrivial, it was decided to switch focus to investigating adaptations to PANet and double back to the issue of the CNN backbone should it become necessary to improve PANet performance.

For more information, see [CNN-backbone/README.md](CNN-backbone/README.md).

## PANet adaptation

> STATUS: ACTIVE
> Debugging in process

The current few-shot learning architecture being investigated is PANet. Thus far, the original results from [*PANet: Few-Shot Image Semantic Segmentation with Prototype Alignment*](https://arxiv.org/abs/1908.06391) have been successfully replicated via the [official repo](https://github.com/kaixin96/PANet). Next steps include adapting PANet to accept the [HPA dataset](Human-Protein-Atlas/) and experimenting with various CNN backbones.

For more information, see https://docs.google.com/document/d/1OHJpOZrEiuWCtvU7-S1mA5YAt8x8PVPNHAKtJnNuG6I/edit and https://github.com/broadinstitute/ip-superurop-t_gao/discussions/11 for status updates and [PANet/README.md](PANet/README.md) for implementation instructions.
