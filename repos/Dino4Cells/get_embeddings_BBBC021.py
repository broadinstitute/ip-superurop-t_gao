import os
import sys
import argparse
import cv2
import random
import colorsys
import requests
from io import BytesIO
import umap
from sklearn.preprocessing import StandardScaler
from file_dataset import pandas_reader_no_labels, ImageFileList_with_filenames, pandas_reader_website, pandas_reader_bbbc021
from file_dataset import ImageFileList, ImageFileList_BBBC021, RGBA_loader, pandas_reader, protein_loader, both_loader, RGB_loader
import numpy as np

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
import vision_transformer as vits
from pathlib import Path
from get_wair_model import get_wair_model
import cell_utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualize Self-Attention maps')
    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base'], help='Architecture (support only ViT atm).')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='', type=str,
        help="Path to pretrained weights to load.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument("--dataset_path", default=None, type=str, help="Path of the dataset.")
    parser.add_argument("--image_size", default=(480, 480), type=int, nargs="+", help="Resize image.")
    parser.add_argument('--output_dir', default='.', help='Path where to save visualizations.')
    parser.add_argument('--batch_size_per_gpu', default=64, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--num_channels', default=3, type=int, help='Number data channels')
    parser.add_argument('--with_labels', action='store_true', help='Whether to include a label target matrix')
    parser.add_argument('--channel_type', type=str, default='RGBA', help='Type of channels')
    parser.add_argument('--model_type', type=str, default='DINO', help='Type of channels')

    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # build model
    wair_model_name_list = ['DenseNet121_change_avg_512_all_more_train_add_3_v2',
                'DenseNet121_change_avg_512_all_more_train_add_3_v3',
                'DenseNet121_change_avg_512_all_more_train_add_3_v5',
                'DenseNet169_change_avg_512_all_more_train_add_3_v5',
                'se_resnext50_32x4d_512_all_more_train_add_3_v5',
                'Xception_osmr_512_all_more_train_add_3_v5',
                'ibn_densenet121_osmr_512_all_more_train_add_3_v5_2']
    if args.model_type == 'DINO':
        model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0, in_chans=args.num_channels)
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        model.to(device)
        if os.path.isfile(args.pretrained_weights):
            state_dict = torch.load(args.pretrained_weights, map_location="cpu")
            if args.checkpoint_key is not None and args.checkpoint_key in state_dict:
                print(f"Take key {args.checkpoint_key} in provided checkpoint dict")
                state_dict = state_dict[args.checkpoint_key]
            # remove `module.` prefix
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            # remove `backbone.` prefix induced by multicrop wrapper
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            msg = model.load_state_dict(state_dict, strict=False)
            print('Pretrained weights found at {} and loaded with msg: {}'.format(args.pretrained_weights, msg))
        else:
            print("Please use the `--pretrained_weights` argument to indicate the path of the checkpoint to evaluate.")
            quit()
    elif args.model_type in wair_model_name_list:
        model = get_wair_model(args.model_type, fold=0)
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        model.to(device)

    if args.channel_type == 'RGBA':
        transform = transforms.Compose([
            transforms.Resize(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.2145, 0.2145, 0.2145, 0.2145), (0.1483, 0.1482, 0.1482, 0.1482)) if args.channel_type == 'RGBA' else transforms.Normalize((0.2145, 0.2145, 0.2145), (0.1483, 0.1482, 0.1482)),
        ])
    elif args.channel_type == 'HPA_both':
        transform = transforms.Compose([
            transforms.Resize(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    elif args.channel_type == 'RGB':
        transform = transforms.Compose([
            transforms.Resize(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    elif args.channel_type == 'wair':
        transform = transforms.Compose([
            transforms.Resize(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(args.image_size),
            cell_utils.Get_specific_channel(1),
            transforms.ToTensor(),
            transforms.Normalize((0.485), (0.229)),
        ])
    dataset_path = Path(args.dataset_path)
    Path(args.output_dir).mkdir(exist_ok=True)
    if args.channel_type == 'RGBA':
        loader = RGBA_loader
    if args.channel_type == 'RGB':
        loader = RGB_loader
    if args.channel_type == 'wair':
        loader = RGBA_loader
    elif args.channel_type == 'protein':
        loader = protein_loader
    elif args.channel_type == 'HPA_both':
        loader = both_loader


    dataset = ImageFileList_BBBC021(args.dataset_path, transform=transform,
                            flist_reader=pandas_reader_bbbc021,
                            # flist_reader=pandas_reader if args.with_labels else pandas_reader_no_labels,
                            loader = loader,
                            training=False)
    sampler = torch.utils.data.SequentialSampler(dataset)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    all_classes = []
    MOAs = []
    if args.model_type == 'DINO':
        all_features = torch.zeros(0, 384)
    elif args.model_type in wair_model_name_list:
        if args.model_type == 'Xception_osmr_512_all_more_train_add_3_v5':
            all_features = torch.zeros(0, 2048)
        elif args.model_type == 'se_resnext50_32x4d_512_all_more_train_add_3_v5':
            all_features = torch.zeros(0, 2048)
        elif args.model_type == 'DenseNet169_change_avg_512_all_more_train_add_3_v5':
            all_features = torch.zeros(0, 1664)
        else:
            all_features = torch.zeros(0, 1024)
    # for images, class_ids, filename in tqdm(data_loader):
    i = 0
    for images, moa in tqdm(data_loader):
    # for images, class_ids in tqdm(data_loader):
        # update weight decay and learning rate according to their schedule
        if args.model_type in wair_model_name_list:
            images = images[:,:3,:,:]
        features = model(images.to(device))
        all_features = torch.cat((all_features, features.cpu()))
        MOAs.append(moa)
        if (i % 100) == 0:
            torch.save((all_features, MOAs), f'{args.output_dir}/features.pth')
            np.save(f'{args.output_dir}/features.npy', (all_features, MOAs))
        i += 1
        # all_classes.extend(class_ids)

    # torch.save((all_features, all_classes), f'{args.output_dir}/features.pth')
    torch.save((all_features, MOAs), f'{args.output_dir}/features.pth')
    np.save(f'{args.output_dir}/features.npy', (all_features, MOAs))
    # np.save(f'{args.output_dir}/features.npy', (all_features, all_classes))



