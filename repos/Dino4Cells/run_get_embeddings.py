import os
import sys
import argparse

import random
import colorsys
import numpy as np
from io import BytesIO
import umap
from sklearn.preprocessing import StandardScaler
from file_dataset import pandas_reader_no_labels, ImageFileList_with_filenames, pandas_reader_website, no_reticulum_loader, single_cell_resize_loader, single_cell_mirror_loader, single_cell_resize_keep_aspect_loader, single_cell_resize_no_mask_loader, single_cell_mirror_no_gap_loader
from file_dataset import ImageFileList, RGBA_loader, pandas_reader, protein_loader, both_loader, protein_transparency_loader, RGBA_norm_loader, threshold_loader, threshold_new_channel_loader, single_cell_resize_with_noise_loader
import numpy as np
from cell_dataset import dino_dataset

from tqdm import tqdm
import skimage.io
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Polygon
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
import numpy as np
from PIL import Image
from tqdm import tqdm

import utils
from main_dino import DataAugmentationDINO
import vision_transformer as vits
from archs import xresnet as cell_models #(!)
from vision_transformer import DINOHead
from pathlib import Path
import cell_utils
import yaml
from functools import partial #(!)

# Unnecessary imports?
# import cv2
# import requests
# import umap
# from sklearn.preprocessing import StandardScaler

# for those who haven't downloaded wair pretrained model weights:
try: from get_wair_model import get_wair_model
except: pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Get embeddings from model')
    parser.add_argument('--config', type=str, default='.', help='path to config file')

    args = parser.parse_args()
    config = yaml.safe_load(open(args.config, 'r'))

    #TODO: fix these temp compatibility patches:
    if not 'HEAD' in list(config['embedding'].keys()):
        print('Please see line 55 in run_get_embeddings.py for additional arguments that can be used to run the full backbone+HEAD model')
    config['embedding']['HEAD'] = True if 'HEAD' in list(config['embedding'].keys()) else False
    config['embedding']['crops'] = True if 'crops' in list(config['embedding'].keys()) else False

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # build model
    wair_model_name_list = ['DenseNet121_change_avg_512_all_more_train_add_3_v2',
                'DenseNet121_change_avg_512_all_more_train_add_3_v3',
                'DenseNet121_change_avg_512_all_more_train_add_3_v5',
                'DenseNet169_change_avg_512_all_more_train_add_3_v5',
                'se_resnext50_32x4d_512_all_more_train_add_3_v5',
                'Xception_osmr_512_all_more_train_add_3_v5',
                'ibn_densenet121_osmr_512_all_more_train_add_3_v5_2']

    if config['model']['model_type'] == 'DINO':
        if config['model']['arch'] in vits.__dict__.keys():
            model = vits.__dict__[config['model']['arch']](patch_size=config['model']['patch_size'], num_classes=0, in_chans=config['model']['num_channels'])
            embed_dim = model.embed_dim
        elif config['model']['arch'] in cell_models.__dict__.keys():
            model = partial(cell_models.__dict__[config['model']['arch']], c_in=config['model']['num_channels'])(False)
            embed_dim = model[-1].in_features
            model[-1] = nn.Identity()

        if config['embedding']['HEAD'] == True:
            model = utils.MultiCropWrapper(model,DINOHead(embed_dim, config['model']['out_dim'], config['model']['use_bn_in_head']))
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        model.to(device)
        pretrained_weights = config['embedding']['pretrained_weights']
        if os.path.isfile(pretrained_weights):
            state_dict = torch.load(pretrained_weights, map_location="cpu")
            teacher = state_dict['teacher']
            if not config['embedding']['HEAD'] == True:
                teacher = {k.replace("module.", ""): v for k, v in teacher.items()}
                teacher = {k.replace("backbone.", ""): v for k, v in teacher.items()}
            msg = model.load_state_dict(teacher, strict=False)
            print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))
        else:
            print("Please use the `--pretrained_weights` argument to indicate the path of the checkpoint to evaluate.")
            quit()
    elif config['model']['model_type'] in wair_model_name_list:
        model = get_wair_model(config['model']['model_type'], fold=0)
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        model.to(device)

    if config['model']['datatype'] in ['HPA','HPA_threshold_preprocessing','HPA_single_cell_mirror_no_gap','HPA_single_cell_mirror','HPA_single_cell_resize_keep_aspect','HPA_single_cell_resize_noise','HPA_single_cell_resize_no_mask','HPA_single_cell_resize_rotate_dihedral','HPA_single_cell_resize_rotate','HPA_single_cell_resize','HPA_single_cell_resize_combo_v1', 'HPA_single_cell_resize_jigsaw','HPA_threshold_protein','HPA_threshold_protein_low','HPA_combo_v1', 'HPA_remove_blur_grey_solar_aug', 'HPA_combined_aug','HPA_modify_jitter','HPA_rescaling_protein_aug', 'HPA_min_max_norm', 'HPA_rotation', 'HPA_remove_grey_aug','HPA_remove_channel','HPA_remove_flip_aug','HPA_remove_color_jitter_aug','HPA_remove_solarization_aug','HPA_remove_blur_aug']:
        transform = transforms.Compose([
            transforms.Resize((config['embedding']['image_size'], config['embedding']['image_size'])),
            transforms.ToTensor(),
            transforms.Normalize((0.2145, 0.2145, 0.2145, 0.2145), (0.1483, 0.1482, 0.1482, 0.1482))
        ])
    elif config['model']['datatype'] in ['HPA_single_cell_resize_partial','HPA_single_cell_random_resize_partial_warp','HPA_single_cell_resize_partial_warp','HPA_single_cell_resize_partial_random','HPA_tile','HPA_single_cell_resize_warp']:
        transform = transforms.Compose([
            cell_utils.Single_cell_Resize(),
            # transforms.Resize(config['embedding']['image_size']),
            transforms.ToTensor(),
            transforms.Normalize((0.2145, 0.2145, 0.2145, 0.2145), (0.1483, 0.1482, 0.1482, 0.1482))
        ])
    elif config['model']['datatype'] in ['HPA_both', 'HPA_alpha_mask_protein','HPA_no_reticulum']:
        transform = transforms.Compose([
            transforms.Resize(config['embedding']['image_size']),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    elif config['model']['datatype'] == 'wair':
        transform = transforms.Compose([
            transforms.Resize(config['embedding']['image_size']),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    elif config['model']['datatype'] in ['protein','HPA_combo_protein_v1']:
        transform = transforms.Compose([
            transforms.Resize(config['embedding']['image_size']),
            cell_utils.Get_specific_channel(1),
            transforms.ToTensor(),
            transforms.Normalize((0.485), (0.229)),
        ])
    elif config['model']['datatype'] in ['HPA_threshold_preprocessing_new_channel']:
        transform = transforms.Compose([
            cell_utils.Permute(),
            transforms.Resize(config['embedding']['image_size']),
            transforms.Normalize((0.2145, 0.2145, 0.2145, 0.2145, 0.2145), (0.1483, 0.1482, 0.1482, 0.1482, 0.1482))
        ])

    dataset_path = Path(config['embedding']['df_path'])
    Path(config['embedding']['output_path']).parent.absolute().mkdir(exist_ok=True)
    if config['model']['datatype'] in ['HPA','HPA_single_cell_resize_partial','HPA_single_cell_random_resize_partial_warp','HPA_single_cell_resize_partial_warp','HPA_single_cell_resize_partial_random','HPA_remove_blur_grey_solar_aug','HPA_combined_aug','HPA_threshold_protein', 'HPA_threshold_protein_low','HPA_combo_v1', 'HPA_rescaling_protein_aug','HPA_modify_jitter','HPA_rotation','HPA_remove_grey_aug','HPA_remove_channel','HPA_remove_flip_aug','HPA_remove_color_jitter_aug','HPA_remove_solarization_aug','HPA_remove_blur_aug']:
        loader = RGBA_loader
    if config['model']['datatype'] == 'HPA_min_max_norm':
        loader = RGBA_norm_loader
    if config['model']['datatype'] == 'wair':
        loader = both_loader
    elif config['model']['datatype'] in ['protein','HPA_combo_protein_v1']:
        loader = protein_loader
    elif config['model']['datatype'] == 'HPA_both':
        loader = both_loader
    elif config['model']['datatype'] == 'HPA_no_reticulum':
        loader = no_reticulum_loader
    elif config['model']['datatype'] == 'HPA_alpha_mask_protein':
        loader = protein_transparency_loader
    elif config['model']['datatype'] == 'HPA_threshold_preprocessing':
        loader = threshold_loader
    elif config['model']['datatype'] == 'HPA_threshold_preprocessing_new_channel':
        loader = threshold_new_channel_loader
    elif config['model']['datatype'] in ['HPA_single_cell_resize_rotate','HPA_single_cell_resize_warp','HPA_single_cell_resize_rotate_dihedral','HPA_tile','HPA_single_cell_resize','HPA_single_cell_resize_combo_v1']:
        loader = single_cell_resize_loader
    elif config['model']['datatype'] == 'HPA_single_cell_resize_jigsaw':
        loader = single_cell_resize_loader
    elif config['model']['datatype'] == 'HPA_single_cell_mirror':
        loader = single_cell_mirror_loader
    elif config['model']['datatype'] == 'HPA_single_cell_mirror_no_gap':
        loader = single_cell_mirror_no_gap_loader
    elif config['model']['datatype'] == 'HPA_single_cell_resize_keep_aspect':
        loader = single_cell_resize_keep_aspect_loader
    elif config['model']['datatype'] == 'HPA_single_cell_resize_no_mask':
        loader = single_cell_resize_no_mask_loader
    elif config['model']['datatype'] == 'HPA_single_cell_resize_noise':
        loader = single_cell_resize_with_noise_loader
    elif config['model']['datatype'] == 'HPA_rotation':
        loader = RGBA_loader
    elif config['model']['datatype'] == 'CellNet':
        loader = RGBA_loader
    if config['model']['datatype'] == 'CellNet':
        dataset = dino_dataset(dataset_path, transform=transform, training=False)
    else:
        dataset = ImageFileList(dataset_path, transform=transform,
                                flist_reader=pandas_reader_website if config['embedding']['embedding_has_labels'] else pandas_reader_no_labels,
                                loader = loader,
                                training=False)
    sampler = torch.utils.data.SequentialSampler(dataset)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=config['model']['batch_size_per_gpu'],
        num_workers=config['embedding']['num_workers'],
        pin_memory=True,
    )

    all_classes = []
    protein_locations = []
    cell_types = []
    if config['model']['model_type'] == 'DINO':
        if config['embedding']['HEAD'] == True:
            all_features = torch.zeros(0, state_dict['args'].out_dim)
        else: all_features = torch.zeros(0, embed_dim)
    elif config['model']['model_type'] in wair_model_name_list:
        if config['model']['model_type'] == 'Xception_osmr_512_all_more_train_add_3_v5':
            all_features = torch.zeros(0, 2048)
        elif config['model']['model_type'] == 'se_resnext50_32x4d_512_all_more_train_add_3_v5':
            all_features = torch.zeros(0, 2048)
        elif config['model']['model_type'] == 'DenseNet169_change_avg_512_all_more_train_add_3_v5':
            all_features = torch.zeros(0, 1664)
        else:
            all_features = torch.zeros(0, 1024)
    # for images, class_ids, filename in tqdm(data_loader):
    i = 0
    for images, protein_location, cell_type in tqdm(data_loader):
    # for images, class_ids in tqdm(data_loader):
        if config['model']['model_type'] in wair_model_name_list:
            images = images[:,:3,:,:]
        if config['model']['datatype'] == 'HPA_both':
            images = images[:,:3,:,:]
        if isinstance(images, list):
            # compatibility for crops:
            images = [i.to(device) for i in images]
            features = model(images)
        else: features = model(images.to(device))
        all_features = torch.cat((all_features, features.cpu()))
        cell_types.append(cell_type)
        protein_locations.append(protein_location)
        if (i % 100) == 0:
            torch.save((all_features, protein_locations, cell_types), f'{config["embedding"]["output_path"]}')
        i += 1
        # all_classes.extend(class_ids)

    # torch.save((all_features, all_classes), f'{config["embedding"]["output_path"]}/features.pth')
    torch.save((all_features, protein_locations, cell_types), f'{config["embedding"]["output_path"]}')


