import argparse
import cv2
import random
import colorsys
import requests
from io import BytesIO
import umap
from sklearn.preprocessing import StandardScaler
from file_dataset import ImageFileList, RGBA_loader, pandas_reader
import numpy as np
import pandas as pd

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
import yaml

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Get embeddings from model')
    parser.add_argument('--config', type=str, default='.', help='path to config file')

    args = parser.parse_args()
    config = yaml.load(open(args.config, 'r'))

    (all_features, protein_locations, cell_types) = torch.load(config['embedding']['features_path'])
    df = pd.read_csv(config['embedding']['df_path'])
    ids = np.array(df.ID)[:len(all_features)]
    protein_locations = np.array([b for i in range(len(protein_locations)) for b in protein_locations[i]])
    cell_types = np.array([b for i in range(len(cell_types)) for b in cell_types[i]])

    cell_types = np.array(cell_types)
    protein_locations = np.array(protein_locations)
    all_protein_locations = np.array([b for i in range(len(protein_locations)) for b in eval(protein_locations[i])])

    # category = cell_types
    # unique_categories = np.unique(category)
    # # unique_categories = np.unique(all_protein_locations)
    # # unique_categories = np.unique(['cytosol','vesicles','mitochondria','nucleoplasm','plasma membrane'])

    # category_inds = []
    # category_inds = {ind : [] for ind in unique_categories}

    # # for i,c in enumerate(category):
    # #     for class_ind in eval(c):
    # #         if class_ind in unique_categories:
    # #             category_inds[(class_ind)].append(i)
    # # category_inds = {k : np.array(v) for k,v in category_inds.items()}

    # for unique_category in unique_categories:
    #     inds = np.where(category == unique_category)[0]
    #     category_inds[unique_category] = inds

    # inds = category_inds['U-2 OS']
    # all_features = all_features[inds, :]
    # cell_types = cell_types[inds]
    # protein_locations = protein_locations[inds]

    category = cell_types
    unique_categories = np.unique(category)
    # unique_categories = np.unique(all_protein_locations)
    # unique_categories = np.unique(['cytosol','vesicles','mitochondria','nucleoplasm','plasma membrane'])

    category_inds = []
    category_inds = {ind : [] for ind in unique_categories}

    # for i,c in enumerate(category):
    #     for class_ind in eval(c):
    #         if class_ind in unique_categories:
    #             category_inds[(class_ind)].append(i)
    # category_inds = {k : np.array(v) for k,v in category_inds.items()}

    for unique_category in unique_categories:
        inds = np.where(category == unique_category)[0]
        category_inds[unique_category] = inds
    reducer = umap.UMAP()
    scaled_features = all_features.numpy()
    # scaled_features = StandardScaler().fit_transform(all_features.numpy())
    embedding = reducer.fit_transform(scaled_features)

    print(f'number of features: {embedding.shape[0]}')

    f, ax = plt.subplots(1,1,figsize=(20,20))
    for unique_category in unique_categories:
        ax.scatter(embedding[category_inds[unique_category], 0], embedding[category_inds[unique_category], 1], s=1, label=unique_category)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    lgnd = ax.legend(loc="upper right", fontsize=18, frameon=False)
    for l in lgnd.legendHandles: l._sizes = [50]

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('none')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(f'{config["plot_embedding"]["output_name"]}/cell_type_all_colors.png')
    plt.close('all')


    num_classes = len(category_inds.keys())
    num_rows = int(np.ceil(np.sqrt(num_classes)))
    num_cols = int(np.ceil(num_classes / num_rows))
    num_rows = np.max([np.max((num_rows, num_cols)), 2])
    num_cols = np.max([np.max((num_rows, num_cols)), 2])

    cmap = cm.nipy_spectral
    f, axes = plt.subplots(num_rows, num_cols, figsize=(20,20))

    for c_prime in range(num_rows * num_cols):
        if c_prime in list(range(num_classes)):
            for c in unique_categories:
                if list(unique_categories).index(c) != c_prime:
                    axes[int(np.floor(c_prime / num_rows))][int(np.floor(c_prime % num_rows))].scatter(embedding[category_inds[c], 0],
                                                                                    embedding[category_inds[c], 1],
                                                                                    s=1, color='lightgrey')
            axes[int(np.floor(c_prime / num_rows))][int(np.floor(c_prime % num_rows))].scatter(embedding[category_inds[list(unique_categories)[c_prime]], 0],
                                                                            embedding[category_inds[list(unique_categories)[c_prime]], 1],
                                                                            s=1, color=cmap(c_prime / num_classes))
            axes[int(np.floor(c_prime / num_rows))][int(np.floor(c_prime % num_rows))].set_title(list(unique_categories)[c_prime])
        axes[int(np.floor(c_prime / num_rows))][int(np.floor(c_prime % num_rows))].spines['right'].set_visible(False)
        axes[int(np.floor(c_prime / num_rows))][int(np.floor(c_prime % num_rows))].spines['top'].set_visible(False)
        axes[int(np.floor(c_prime / num_rows))][int(np.floor(c_prime % num_rows))].spines['left'].set_visible(False)
        axes[int(np.floor(c_prime / num_rows))][int(np.floor(c_prime % num_rows))].spines['bottom'].set_visible(False)

        # Only show ticks on the left and bottom spines
        axes[int(np.floor(c_prime / num_rows))][int(np.floor(c_prime % num_rows))].yaxis.set_ticks_position('none')
        axes[int(np.floor(c_prime / num_rows))][int(np.floor(c_prime % num_rows))].xaxis.set_ticks_position('none')
        axes[int(np.floor(c_prime / num_rows))][int(np.floor(c_prime % num_rows))].set_xticks([])
        axes[int(np.floor(c_prime / num_rows))][int(np.floor(c_prime % num_rows))].set_yticks([])

    plt.tight_layout()
    plt.savefig(f'{config["plot_embedding"]["output_name"]}/cell_type_umap_per_class.png')
    plt.close('all')

    category = protein_locations
    # unique_categories = np.unique(category)
    unique_categories = np.unique(all_protein_locations)
    # unique_categories = np.unique(['cytosol','vesicles','mitochondria','nucleoplasm','plasma membrane'])

    category_inds = []
    category_inds = {ind : [] for ind in unique_categories}

    for i,c in enumerate(category):
        for class_ind in eval(c):
            if class_ind in unique_categories:
                category_inds[(class_ind)].append(i)
    category_inds = {k : np.array(v) for k,v in category_inds.items()}


    # for unique_category in unique_categories:
    #     inds = np.where(category == unique_category)[0]
    #     category_inds[unique_category] = inds

    # (all_features, target_matrix) = np.load(config['embedding']['features_path'], allow_pickle=True)
    # df = pd.read_csv(args.csv_path)
    # small_ids = np.where(df.mask_size < (16.5 * 16.5))
    # large_ids = np.where(df.mask_size >= (16.5 * 16.5))

    f, ax = plt.subplots(1,1,figsize=(20,20))
    for unique_category in unique_categories:
        ax.scatter(embedding[category_inds[unique_category], 0], embedding[category_inds[unique_category], 1], s=1, label=unique_category)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    lgnd = ax.legend(loc="upper right", fontsize=18, frameon=False)
    for l in lgnd.legendHandles: l._sizes = [50]

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('none')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(f'{config["plot_embedding"]["output_name"]}/protein_all_colors.png')
    plt.close('all')


    num_classes = len(category_inds.keys())
    num_rows = int(np.ceil(np.sqrt(num_classes)))
    num_cols = int(np.ceil(num_classes / num_rows))
    num_rows = np.max((num_rows, num_cols))
    num_cols = np.max((num_rows, num_cols))

    cmap = cm.nipy_spectral
    f, axes = plt.subplots(num_rows, num_cols, figsize=(20,20))

    for c_prime in range(num_rows * num_cols):
        if c_prime in list(range(num_classes)):
            for c in unique_categories:
                if list(unique_categories).index(c) != c_prime:
                    axes[int(np.floor(c_prime / num_rows))][int(np.floor(c_prime % num_rows))].scatter(embedding[category_inds[c], 0],
                                                                                    embedding[category_inds[c], 1],
                                                                                    s=1, color='lightgrey')
            axes[int(np.floor(c_prime / num_rows))][int(np.floor(c_prime % num_rows))].scatter(embedding[category_inds[list(unique_categories)[c_prime]], 0],
                                                                            embedding[category_inds[list(unique_categories)[c_prime]], 1],
                                                                            s=1, color=cmap(c_prime / num_classes))
            axes[int(np.floor(c_prime / num_rows))][int(np.floor(c_prime % num_rows))].set_title(list(unique_categories)[c_prime])
        axes[int(np.floor(c_prime / num_rows))][int(np.floor(c_prime % num_rows))].spines['right'].set_visible(False)
        axes[int(np.floor(c_prime / num_rows))][int(np.floor(c_prime % num_rows))].spines['top'].set_visible(False)
        axes[int(np.floor(c_prime / num_rows))][int(np.floor(c_prime % num_rows))].spines['left'].set_visible(False)
        axes[int(np.floor(c_prime / num_rows))][int(np.floor(c_prime % num_rows))].spines['bottom'].set_visible(False)

        # Only show ticks on the left and bottom spines
        axes[int(np.floor(c_prime / num_rows))][int(np.floor(c_prime % num_rows))].yaxis.set_ticks_position('none')
        axes[int(np.floor(c_prime / num_rows))][int(np.floor(c_prime % num_rows))].xaxis.set_ticks_position('none')
        axes[int(np.floor(c_prime / num_rows))][int(np.floor(c_prime % num_rows))].set_xticks([])
        axes[int(np.floor(c_prime / num_rows))][int(np.floor(c_prime % num_rows))].set_yticks([])

    plt.tight_layout()
    plt.savefig(f'{config["plot_embedding"]["output_name"]}/protein_umap_per_class.png')
    plt.close('all')

    # category = ids
    # unique_categories = np.unique(category)
    # category_inds = []
    # category_inds = {ind : [] for ind in unique_categories}
    # for unique_category in unique_categories:
    #     inds = np.where(category == unique_category)[0]
    #     category_inds[unique_category] = inds


    # f, ax = plt.subplots(1,1,figsize=(20,20))
    # for unique_category in unique_categories:
    #     ax.scatter(embedding[category_inds[unique_category], 0], embedding[category_inds[unique_category], 1], s=1, label=unique_category)
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # # Only show ticks on the left and bottom spines
    # ax.yaxis.set_ticks_position('none')
    # ax.xaxis.set_ticks_position('none')
    # ax.set_xticks([])
    # ax.set_yticks([])
    # plt.tight_layout()
    # plt.savefig(f'{config['plot_embedding']['output_name']}/ID_all_colors.png')
    # plt.close('all')


    # num_classes = len(category_inds.keys())
    # num_rows = int(np.ceil(np.sqrt(num_classes)))
    # num_cols = int(np.ceil(num_classes / num_rows))
    # num_rows = np.max([np.max((num_rows, num_cols)), 2])
    # num_cols = np.max([np.max((num_rows, num_cols)), 2])

    # cmap = cm.nipy_spectral
    # f, axes = plt.subplots(num_rows, num_cols, figsize=(20,20))

    # for c_prime in range(num_rows * num_cols):
    #     if c_prime in list(range(num_classes)):
    #         for c in unique_categories:
    #             if list(unique_categories).index(c) != c_prime:
    #                 axes[int(np.floor(c_prime / num_rows))][int(np.floor(c_prime % num_rows))].scatter(embedding[category_inds[c], 0],
    #                                                                                 embedding[category_inds[c], 1],
    #                                                                                 s=1, color='lightgrey')
    #         axes[int(np.floor(c_prime / num_rows))][int(np.floor(c_prime % num_rows))].scatter(embedding[category_inds[list(unique_categories)[c_prime]], 0],
    #                                                                         embedding[category_inds[list(unique_categories)[c_prime]], 1],
    #                                                                         s=1, color=cmap(c_prime / num_classes))
    #     axes[int(np.floor(c_prime / num_rows))][int(np.floor(c_prime % num_rows))].spines['right'].set_visible(False)
    #     axes[int(np.floor(c_prime / num_rows))][int(np.floor(c_prime % num_rows))].spines['top'].set_visible(False)
    #     axes[int(np.floor(c_prime / num_rows))][int(np.floor(c_prime % num_rows))].spines['left'].set_visible(False)
    #     axes[int(np.floor(c_prime / num_rows))][int(np.floor(c_prime % num_rows))].spines['bottom'].set_visible(False)

    #     # Only show ticks on the left and bottom spines
    #     axes[int(np.floor(c_prime / num_rows))][int(np.floor(c_prime % num_rows))].yaxis.set_ticks_position('none')
    #     axes[int(np.floor(c_prime / num_rows))][int(np.floor(c_prime % num_rows))].xaxis.set_ticks_position('none')
    #     axes[int(np.floor(c_prime / num_rows))][int(np.floor(c_prime % num_rows))].set_xticks([])
    #     axes[int(np.floor(c_prime / num_rows))][int(np.floor(c_prime % num_rows))].set_yticks([])

    # plt.tight_layout()
    # plt.savefig(f'{config['plot_embedding']['output_name']}/ID_umap_per_class.png')
    # plt.close('all')
