"""
Load Human Protein Atlas (HPA) dataset
"""

import os

import numpy as np
from PIL import Image
import torch

from .common import BaseDataset

class HPA(BaseDataset):
    """
    Base Class for HPA Dataset

    Args:
        base_dir:
            directory containing HPA and grayscale mask datasets
        split:
            which split to use
            choose from ('train', 'val', 'trainval', 'trainaug')
        transform:
            transformations to be performed on images/masks
        to_tensor:
            transformation to convert PIL Image to tensor
    """
    def __init__(self, base_dir, rgb_dir, grayscale_dir, split, transforms=None, to_tensor=None):
        super().__init__(base_dir)
        self.split = split
        self.transforms = transforms
        self.to_tensor = to_tensor
        self.samples = {}
        self.labels = {'all': set()}
        self.sub_ids = []

        class_counter = 0
        for class_dir in os.listdir(os.path.join(base_dir, rgb_dir)):
            self.labels[class_counter] = set()
            for img_file in os.list_dir(os.path.join(base_dir, rgb_dir, class_dir)):
                self.labels['all'].add(img_file)
                self.labels[class_counter].add(img_file)
                self.sub_ids.append(img_file)

                samples[img_file] = {
                    'image': Image.open(os.path.join(base_dir, rgb_dir, class_dir, img_file)),
                    'label': Image.open(os.path.join(base_dir, grayscale_dir, class_dir, img_file))
                }
            class_counter += 1

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, img_file):
        # Fetch data
        sample = self.samples[img_file]

        # Image-level transformation
        if self.transforms is not None:
            sample = self.transforms(sample)

        # Save the original image (without normalization)
        image_t = torch.from_numpy(np.array(sample['image']).transpose(2, 0, 1))

        # Transform to tensor
        if self.to_tensor is not None:
            sample = self.to_tensor(sample)

        sample['id'] = id_
        sample['image_t'] = image_t

        return sample

    def get_labels(self):
        return self.labels

    def get_sub_ids(self):
        return self.sub_ids[]
