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
            HPA dataset directory
        split:
            which split to use
            choose from ('train', 'val', 'trainval', 'trainaug')
        transform:
            transformations to be performed on images/masks
        to_tensor:
            transformation to convert PIL Image to tensor
    """
    def __init__(self, base_dir, split, transforms=None, to_tensor=None):
        super().__init__(base_dir)
        self.split = split
        self._base_dir = base_dir
        self.transforms = transforms
        self.to_tensor = to_tensor

        self.ids = # TODO: set equal to filenames (without extensions) of images
        #       for VOC, this looks at trainaug.txt (as data_split is set in config.py)
        # TODO: delete the below (prior assignment from VOC pascal.py)
        # with open(os.path.join(self._id_dir, f'{self.split}.txt'), 'r') as f:
        #     self.ids = f.read().splitlines()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # Fetch data
        id_ = self.ids[idx]
        image = # TODO: fix given self._base_dir (as set in config.py, something like '../../data/HPA/') and id_ (image filename without extension); this should be the RGB image
        semantic_mask = # TODO: similarly fix; this should be the grayscale mask
        # TODO: delete the below (prior assignments from VOC pascal.py)
        # image = Image.open(os.path.join(self._image_dir, f'{id_}.jpg'))
        # semantic_mask = Image.open(os.path.join(self._label_dir, f'{id_}.png'))
        sample = {'image': image,
                  'label': semantic_mask}

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
