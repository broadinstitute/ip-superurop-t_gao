"""
Customized dataset
"""

import os
import random

import torch
import numpy as np

from .pascal import VOC
from .coco import COCOSeg
from .common import PairedDataset
from .hpa import HPA


def attrib_basic(_sample, class_id):
    """
    Add basic attribute
    Args:
        _sample: data sample
        class_id: class label asscociated with the data
            (sometimes indicting from which subset the data are drawn)
    """
    return {'class_id': class_id}


def getMask(label, scribble, class_id, class_ids):
    """
    Generate FG/BG mask from the segmentation mask

    Args:
        label:
            semantic mask
        scribble:
            scribble mask
        class_id:
            semantic class of interest
        class_ids:
            all class id in this episode
    """
    # Dense Mask
    fg_mask = torch.where(label == class_id,
                          torch.ones_like(label), torch.zeros_like(label))
    bg_mask = torch.where(label != class_id,
                          torch.ones_like(label), torch.zeros_like(label))
    for class_id in class_ids:
        bg_mask[label == class_id] = 0

    # Scribble Mask
    bg_scribble = scribble == 0
    fg_scribble = torch.where((fg_mask == 1)
                              & (scribble != 0)
                              & (scribble != 255),
                              scribble, torch.zeros_like(fg_mask))
    scribble_cls_list = list(set(np.unique(fg_scribble)) - set([0,]))
    if scribble_cls_list:  # Still need investigation
        fg_scribble = fg_scribble == random.choice(scribble_cls_list).item()
    else:
        fg_scribble[:] = 0

    return {'fg_mask': fg_mask,
            'bg_mask': bg_mask,
            'fg_scribble': fg_scribble.long(),
            'bg_scribble': bg_scribble.long()}


def fewShot(paired_sample, n_ways, n_shots, cnt_query, dataset='VOC', coco=False):
    """
    Postprocess paired sample for fewshot settings

    Args:
        paired_sample:
            data sample from a PairedDataset
        n_ways:
            n-way few-shot learning
        n_shots:
            n-shot few-shot learning
        cnt_query:
            number of query images for each class in the support set
        dataset:
            VOC, COCO, or HPA
    """


    ###### Compose the support and query image list ######
    cumsum_idx = np.cumsum([0,] + [n_shots + x for x in cnt_query])

    # support class ids
    # class_ids = [paired_sample[cumsum_idx[i]]['basic_class_id'] for i in range(n_ways)]
    class_ids = []
    for i in range(n_ways):
        try:
            class_ids.append(paired_sample[cumsum_idx[i]]['basic_class_id'])
        except TypeError:
            continue
    # class_ids = [paired_sample[cumsum_idx[i]]['basic_class_id'] for i in range(n_ways)]

    # support images
    support_images = []
    for i in range(n_ways):
        support_images_entry = []
        for j in range(n_shots):
            try:
                support_images_entry.append(paired_sample[cumsum_idx[i] + j]['image'])
            except TypeError:
                continue
            support_images.append(support_images_entry)
    # support_images = [[paired_sample[cumsum_idx[i] + j]['image'] for j in range(n_shots)] for i in range(n_ways)]
    # support_images_t = [[paired_sample[cumsum_idx[i] + j]['image_t'] for j in range(n_shots)] for i in range(n_ways)]

    # support image labels
    if dataset == 'COCO':
        support_labels = [[paired_sample[cumsum_idx[i] + j]['label'][class_ids[i]]
                           for j in range(n_shots)] for i in range(n_ways)]
    else:
        support_labels = []
        for i in range(n_ways):
            support_labels_entry = []
            for j in range(n_shots):
                try:
                    support_labels_entry.append(paired_sample[cumsum_idx[i] + j]['label'])
                except TypeError:
                    continue
            support_labels_entry.append(support_labels)
        # support_labels = [[paired_sample[cumsum_idx[i] + j]['label'] for j in range(n_shots)] for i in range(n_ways)]
    
    support_scribbles = []
    for i in range(n_ways):
        support_scribbles_entry = []
        for j in range(n_shots):
            try:
                support_scribbles_entry.append(paired_sample[cumsum_idx[i] + j]['scribble'])
            except TypeError:
                continue
        support_scribbles.append(support_scribbles_entry)
    #support_scribbles = [[paired_sample[cumsum_idx[i] + j]['scribble'] for j in range(n_shots)] for i in range(n_ways)]

    support_insts = []
    for i in range(n_ways):
        support_insts_entry = []
        for j in range(n_shots):
            try:
                support_insts_entry.append(paired_sample[cumsum_idx[i] + j]['inst'])
            except TypeError:
                continue
        support_insts.append(support_insts_entry) 
    # support_insts = [[paired_sample[cumsum_idx[i] + j]['inst'] for j in range(n_shots)] for i in range(n_ways)]



    # query images, masks and class indices
    query_images = []
    for i in range(n_ways):
        for j in range(cnt_query[i]):
            query_image_index = cumsum_idx[i+1] - j - 1
            query_sample = paired_sample[query_image_index]
            if query_sample is None:
                continue
            query_image = query_sample['image']
            if query_image is None:
                continue
            query_images.append(query_image)
    #query_images = [paired_sample[cumsum_idx[i+1] - j - 1]['image'] for i in range(n_ways)
    #                for j in range(cnt_query[i])]
    # query_images_t = [paired_sample[cumsum_idx[i+1] - j - 1]['image_t'] for i in range(n_ways) for j in range(cnt_query[i])]

    query_labels = []
    if coco:
        for i in range(n_ways):
            for j in range(cnt_query[i]):
                query_label_index = cumsum_idx[i+1] - j - 1
                query_label_sample = paired_sample[query_label_index]
                query_label = query_label_sample['label']
                class_id = class_ids[i]
                query_label_i = query_label[class_id]
                query_labels.append(query_label_i)
        # query_labels = [paired_sample[cumsum_idx[i+1] - j - 1]['label'][class_ids[i]]
        #                 for i in range(n_ways) for j in range(cnt_query[i])]
    else:
        for i in range(n_ways):
            for j in range(cnt_query[i]):
                try:
                    query_label_index = cumsum_idx[i+1] - j - 1
                    query_label_sample = paired_sample[query_label_index]
                    query_label = query_label_sample['label']
                    query_labels.append(query_label)
                except TypeError:
                    continue
        # query_labels = [paired_sample[cumsum_idx[i+1] - j - 1]['label'] for i in range(n_ways)
        #                for j in range(cnt_query[i])]
    query_cls_idx = [sorted([0,] + [class_ids.index(x) + 1
                                    for x in set(np.unique(query_label)) & set(class_ids)])
                     for query_label in query_labels]


    ###### Generate support image masks ######
    support_mask = []
    for way in range(n_ways):
        support_mask_entry = []
        for shot in range(n_shots):
            try:
                support_mask_entry.append(getMask(support_labels[way][shot], support_scribbles[way][shot], class_ids[way], class_ids))
            except IndexError:
                continue
        support_mask.append(support_mask_entry)
    # support_mask = [[getMask(support_labels[way][shot], support_scribbles[way][shot], class_ids[way], class_ids) for shot in range(n_shots)] for way in range(n_ways)]


    ###### Generate query label (class indices in one episode, i.e. the ground truth)######
    query_labels_tmp = [torch.zeros_like(x) for x in query_labels]
    for i, query_label_tmp in enumerate(query_labels_tmp):
        query_label_tmp[query_labels[i] == 255] = 255
        for j in range(n_ways):
            try:
                query_label_tmp[query_labels[i] == class_ids[j]] = j + 1
            except IndexError:
                continue

    ###### Generate query mask for each semantic class (including BG) ######
    # BG class
    query_masks = [[torch.where(query_label == 0,
                                torch.ones_like(query_label),
                                torch.zeros_like(query_label))[None, ...],]
                   for query_label in query_labels]
    # Other classes in query image
    for i, query_label in enumerate(query_labels):
        for idx in query_cls_idx[i][1:]:
            mask = torch.where(query_label == class_ids[idx - 1],
                               torch.ones_like(query_label),
                               torch.zeros_like(query_label))[None, ...]
            query_masks[i].append(mask)


    return {'class_ids': class_ids,

            # 'support_images_t': support_images_t,
            'support_images': support_images,
            'support_mask': support_mask,
            'support_inst': support_insts,

            # 'query_images_t': query_images_t,
            'query_images': query_images,
            'query_labels': query_labels_tmp,
            'query_masks': query_masks,
            'query_cls_idx': query_cls_idx,
           }


def coco_fewshot(base_dir, split, transforms, to_tensor, labels, n_ways, n_shots, max_iters, n_queries=1):
    """
    Args:
        base_dir:
            COCO dataset directory
        split:
            which split to use
            choose from ('train', 'val')
        transform:
            transformations to be performed on images/masks
        to_tensor:
            transformation to convert PIL Image to tensor
        labels:
            labels of the data
        n_ways:
            n-way few-shot learning, should be no more than # of labels
        n_shots:
            n-shot few-shot learning
        max_iters:
            number of pairs
        n_queries:
            number of query images
    """
    cocoseg = COCOSeg(base_dir, split, transforms, to_tensor)
    cocoseg.add_attrib('basic', attrib_basic, {})

    # Load image ids for each class
    cat_ids = cocoseg.coco.getCatIds()
    sub_ids = [cocoseg.coco.getImgIds(catIds=cat_ids[i - 1]) for i in labels]
    # Create sub-datasets and add class_id attribute
    subsets = cocoseg.subsets(sub_ids, [{'basic': {'class_id': cat_ids[i - 1]}} for i in labels])

    # Choose the classes of queries
    cnt_query = np.bincount(random.choices(population=range(n_ways), k=n_queries),
                            minlength=n_ways)
    # Set the number of images for each class
    n_elements = [n_shots + x for x in cnt_query]
    # Create paired dataset
    paired_data = PairedDataset(subsets, n_elements=n_elements, max_iters=max_iters, same=False,
                                pair_based_transforms=[
                                    (fewShot, {'n_ways': n_ways, 'n_shots': n_shots,
                                               'cnt_query': cnt_query, 'coco': True})])
    return paired_data


def voc_fewshot(base_dir, split, transforms, to_tensor, labels, n_ways, n_shots, max_iters, n_queries=1):
    """
    Args:
        base_dir:
            VOC dataset directory
        split:
            which split to use
            choose from ('train', 'val', 'trainval', 'trainaug')
        transform:
            transformations to be performed on images/masks
        to_tensor:
            transformation to convert PIL Image to tensor
        labels:
            object class labels of the data
        n_ways:
            n-way few-shot learning, should be no more than # of object class labels
        n_shots:
            n-shot few-shot learning
        max_iters:
            number of pairs
        n_queries:
            number of query images
    """
    voc = VOC(base_dir=base_dir, split=split, transforms=transforms, to_tensor=to_tensor)
    voc.add_attrib('basic', attrib_basic, {})

    # Load image ids for each class
    sub_ids = []
    for label in labels:
        with open(os.path.join(voc._id_dir, voc.split,
                               'class{}.txt'.format(label)), 'r') as f:
            sub_ids.append(f.read().splitlines())
    # Create sub-datasets and add class_id attribute
    subsets = voc.subsets(sub_ids, [{'basic': {'class_id': cls_id}} for cls_id in labels])

    # Choose the classes of queries
    cnt_query = np.bincount(random.choices(population=range(n_ways), k=n_queries), minlength=n_ways)
    # Set the number of images for each class
    n_elements = [n_shots + x for x in cnt_query]
    # Create paired dataset
    paired_data = PairedDataset(subsets, n_elements=n_elements, max_iters=max_iters, same=False,
                                pair_based_transforms=[
                                    (fewShot, {'n_ways': n_ways, 'n_shots': n_shots,
                                               'cnt_query': cnt_query})])
    return paired_data


def hpa_fewshot(base_dir, grayscale_dir, rgb_dir, split, transforms, to_tensor, n_ways, n_shots, max_iters, n_queries=1):
    """
    Args:
        base_dir:
            HPA dataset directory
        split:
            which split to use
            choose from ('train', 'val')
        transform:
            transformations to be performed on images/masks
        to_tensor:
            transformation to convert PIL Image to tensor
        labels:
            labels of the data
        n_ways:
            n-way few-shot learning, should be no more than # of labels
        n_shots:
            n-shot few-shot learning
        max_iters:
            number of pairs
        n_queries:
            number of query images
    """

    hpa = HPA(base_dir, grayscale_dir, rgb_dir, split, transforms, to_tensor)
    hpa.add_attrib('basic', attrib_basic, {})

    # Load image ids for each class
    sub_ids = hpa.get_sub_ids()
    labels = hpa.get_labels()

    # Create sub-datasets and add class_id attribute
    # subsets = hpa.subsets(sub_ids)
    subsets = hpa.subsets(sub_ids, [{'basic': {'class_id': cls_id}} for cls_id in labels])

    # Choose the classes of queries
    cnt_query = np.bincount(random.choices(population=range(n_ways), k=n_queries), minlength=n_ways)

    # Set the number of images for each class
    n_elements = [n_shots + x for x in cnt_query]

    # Create paired dataset
    paired_data = PairedDataset(
        subsets,
        n_elements,
        max_iters,
        same=False,
        pair_based_transforms=[(
            fewShot, {
                'n_ways': n_ways,
                'n_shots': n_shots,
                'cnt_query': cnt_query
            }
        )]
    )

    return paired_data, labels
