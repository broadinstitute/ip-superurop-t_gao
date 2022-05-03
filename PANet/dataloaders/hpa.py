"""
Load Human Protein Atlas (HPA) dataset
"""

import os

import numpy as np
from PIL import Image
import torch
from torchvision import transforms as tv_transforms

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
        self.ids = []
        self.sub_ids = []

        class_counter = 0
        for class_dir in os.listdir(os.path.join(base_dir, rgb_dir)):
            if class_dir == '.DS_Store':
                continue
            self.labels[class_counter] = set()
            class_sub_ids = []
            for img_file in os.listdir(os.path.join(base_dir, rgb_dir, class_dir)):
                if img_file == '.DS_Store':
                    continue

                # try:
                #     img_file_dict = {
                #         'image': Image.open(os.path.join(base_dir, rgb_dir, class_dir, img_file)),
                #         'label': Image.open(os.path.join(base_dir, grayscale_dir, class_dir, img_file))
                #     }
                # except FileNotFoundError:
                #     continue
                try:
                    image = Image.open(os.path.join(base_dir, rgb_dir, class_dir, img_file))
                except FileNotFoundError:
                    continue

                try:
                    label = Image.open(os.path.join(base_dir, grayscale_dir, class_dir, img_file))
                except FileNotFoundError:
                    continue

                # label = np.stack((label,)*3, axis=-1) # label = label.convert('RGB') # to ensure 3 channels

                #print('IMAGE TYPE', type(image))
                #print('LABEL TYPE', type(label))
                #image = np.array(image)
                #label = np.array(label)
                #print('IMAGE ARRAY', image)
                #print('LABEL ARRAY', label)
                #image = tv_transforms.ToTensor()(image).unsqueeze_(0) # np.array(image) #image.float() # .to(dtype='float')
                #label = tv_transforms.ToTensor()(label).unsqueeze_(0) # np.array(label) #label.float() # .to(dtype='float')

                # No scribble/instance mask
                img_dimensions = (256, 256) # TODO: link to config
                img_as_zeros = np.zeros(img_dimensions) #, dtype=np.uint8)
                # img_as_zeros = img_as_zeros.astype(np.uint8)
                #print('img_as_zeros', img_as_zeros)
                # instance_mask = Image.fromarray(img_as_zeros) #, dtype='uint8'))
                # scribble_mask = Image.fromarray(img_as_zeros) #, dtype='uint8'))
                instance_mask = Image.fromarray(np.zeros(img_dimensions, dtype='uint8'))
                scribble_mask = Image.fromarray(np.zeros(img_dimensions, dtype='uint8'))

                img_file_dict = {
                    'image': image,
                    'label': label,
                    'inst': instance_mask,
                    'scribble': scribble_mask
                }

                self.labels['all'].add(img_file)
                self.labels[class_counter].add(img_file)
                self.ids.append(img_file)
                class_sub_ids.append(img_file)
                self.samples[img_file] = img_file_dict

                # self.samples[img_file] = {
                #     'image': Image.open(os.path.join(base_dir, rgb_dir, class_dir, img_file)),
                #     'label': Image.open(os.path.join(base_dir, grayscale_dir, class_dir, img_file))
                # }
            self.sub_ids.append(class_sub_ids)
            class_counter += 1

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # Fetch data
        img_file = self.ids[idx]
        sample = self.samples[img_file]

        # Image-level transformation
        if self.transforms is not None:
            sample = self.transforms(sample)

        # # Save the original image (without normalization)
        # image_t = torch.from_numpy(np.array(sample['image']).transpose(2, 0, 1))

        # Transform to tensor
        if self.to_tensor is not None:
            sample = self.to_tensor(sample)

        sample['id'] = img_file
        # sample['image_t'] = image_t

        return sample

    def get_labels(self):
        return self.labels

    def get_sub_ids(self):
        return self.sub_ids
