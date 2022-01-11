import torch.utils.data as data
import torch

from PIL import Image
import os
import numpy as np
import os.path
import pandas as pd

def default_loader(path):
    return Image.open(path).convert('RGB')

def RGB_loader(path):
    return Image.open(path).convert('RGB')

def RGBA_loader(path):
    return Image.open(path).convert('RGBA')

def single_cell_resize_keep_aspect_loader(path):
    mask = np.array(Image.fromarray(np.load(path.replace(
        'Dino4Cells/datasets/HPA/data/','human_protein_atlas/website_small_subset_masks/mask_').replace('jpg','npy')) == 1).resize((512,512)))
    img  = np.array(Image.open(path))
    cell_ind = 1
    cell_image_size = 224
    mask = mask == 1
    cell_mask = ((mask == cell_ind) * 255).astype(int)
    left_limit = np.where(cell_mask.sum(axis=0) > 0)[0][0]
    right_limit = np.where(cell_mask.sum(axis=0) > 0)[0][-1]
    upper_limit = np.where(cell_mask.sum(axis=1) > 0)[0][0]
    lower_limit = np.where(cell_mask.sum(axis=1) > 0)[0][-1]
    center = (int(upper_limit + (lower_limit - upper_limit) / 2)), int((left_limit + (right_limit - left_limit) / 2))
    new_img = np.copy(img)[upper_limit : lower_limit, left_limit : right_limit, :]
    for c in range(4):
        new_img[:,:,c] = np.where(cell_mask[upper_limit : lower_limit, left_limit : right_limit], new_img[:,:,c], 0)

    ratio = new_img.shape[0] / new_img.shape[1]
    if ratio > 1:
        new_x_shape = min(int(cell_image_size / ratio), cell_image_size)
        new_y_shape = int(cell_image_size)
    else:
        new_x_shape = int(cell_image_size)
        new_y_shape = min(int(cell_image_size * ratio), cell_image_size)
    new_img = np.array(Image.merge('RGBA', [c.resize((new_x_shape,new_y_shape)) for c in Image.fromarray(new_img).split()]))

    img_shape = new_img.shape
    upper_pad = int((max((cell_image_size - (img_shape[0])) / 2, 0)))
    lower_pad = int((max((cell_image_size - (img_shape[0])) / 2, 0)))
    left_pad  = int((max((cell_image_size - (img_shape[1])) / 2, 0)))
    right_pad = int((max((cell_image_size - (img_shape[1])) / 2, 0)))

    new_img = np.pad(new_img, ((upper_pad, lower_pad),
                            (left_pad, right_pad),
                            (0, 0)), 'constant').astype(np.uint8)
    new_img = new_img[:224, :224, :]
    new_img = Image.fromarray(new_img)

    return new_img

def single_cell_resize_no_mask_loader(path):
    mask = np.array(Image.fromarray(np.load(path.replace('Dino4Cells/datasets/HPA/data/','human_protein_atlas/website_small_subset_masks/mask_').replace('jpg','npy')) == 1).resize((512,512)))
    img  = np.array(Image.open(path))
    cell_ind = 1
    cell_image_size = 224
    mask = mask == 1
    cell_mask = ((mask == cell_ind) * 255).astype(int)
    left_limit = np.where(cell_mask.sum(axis=0) > 0)[0][0]
    right_limit = np.where(cell_mask.sum(axis=0) > 0)[0][-1]
    upper_limit = np.where(cell_mask.sum(axis=1) > 0)[0][0]
    lower_limit = np.where(cell_mask.sum(axis=1) > 0)[0][-1]
    center = (int(upper_limit + (lower_limit - upper_limit) / 2)), int((left_limit + (right_limit - left_limit) / 2))
    new_img = np.copy(img)[upper_limit : lower_limit, left_limit : right_limit, :]

    pil_img = Image.fromarray(new_img)
    pil_img = Image.merge('RGBA', [c.resize((cell_image_size,cell_image_size)) for c in pil_img.split()])
    return pil_img

def single_cell_resize_with_noise_loader(path):
    mask = np.array(Image.fromarray(np.load(path.replace('Dino4Cells/datasets/HPA/data/','human_protein_atlas/website_small_subset_masks/mask_').replace('jpg','npy')) == 1).resize((512,512)))
    img  = np.array(Image.open(path))
    cell_ind = 1
    cell_image_size = 224
    mask = mask == 1
    cell_mask = ((mask == cell_ind) * 255).astype(int)
    left_limit = np.where(cell_mask.sum(axis=0) > 0)[0][0]
    right_limit = np.where(cell_mask.sum(axis=0) > 0)[0][-1]
    upper_limit = np.where(cell_mask.sum(axis=1) > 0)[0][0]
    lower_limit = np.where(cell_mask.sum(axis=1) > 0)[0][-1]
    center = (int(upper_limit + (lower_limit - upper_limit) / 2)), int((left_limit + (right_limit - left_limit) / 2))
    new_img = np.copy(img)[upper_limit : lower_limit, left_limit : right_limit, :]
    for c in range(4):
        new_img[:,:,c] = np.where(cell_mask[upper_limit : lower_limit, left_limit : right_limit], new_img[:,:,c], 0)
    pil_img = Image.fromarray(new_img)
    pil_img = np.array(Image.merge('RGBA', [c.resize((cell_image_size,cell_image_size)) for c in pil_img.split()]))
    random_img = np.random.rand(pil_img.shape[0], pil_img.shape[1], pil_img.shape[2])
    for c in range(4):
        random_img[:, :, c] = np.where(pil_img.mean(axis=(2)) == 0, (random_img[:, :, c] * 255).astype(np.uint8), 0)
    pil_img += random_img.astype(np.uint8)
    pil_img = Image.fromarray(pil_img)
    return pil_img

def tile_loader(path):
    img  = np.array(Image.open(path))
    new_img = img[:128, :128, :]
    pil_img = Image.fromarray(new_img)
    return pil_img

def single_cell_loader(path):
    mask = np.array(image.fromarray(np.load(path.replace('dino4cells/datasets/hpa/data/','human_protein_atlas/website_small_subset_masks/mask_').replace('jpg','npy')) == 1).resize((512,512)))
    img  = np.array(image.open(path))
    cell_ind = 1
    mask = mask == 1
    cell_mask = ((mask == cell_ind) * 255).astype(int)
    left_limit = np.where(cell_mask.sum(axis=0) > 0)[0][0]
    right_limit = np.where(cell_mask.sum(axis=0) > 0)[0][-1]
    upper_limit = np.where(cell_mask.sum(axis=1) > 0)[0][0]
    lower_limit = np.where(cell_mask.sum(axis=1) > 0)[0][-1]
    center = (int(upper_limit + (lower_limit - upper_limit) / 2)), int((left_limit + (right_limit - left_limit) / 2))
    new_img = np.copy(img)[upper_limit : lower_limit, left_limit : right_limit, :]
    for c in range(4):
        new_img[:,:,c] = np.where(cell_mask[upper_limit : lower_limit, left_limit : right_limit], new_img[:,:,c], 0)

    pil_img = image.fromarray(new_img)
    return pil_img

def single_cell_resize_loader(path):
    mask = np.array(Image.fromarray(np.load(path.replace('Dino4Cells/datasets/HPA/data/','human_protein_atlas/website_small_subset_masks/mask_').replace('jpg','npy')) == 1).resize((512,512)))
    img  = np.array(Image.open(path))
    cell_ind = 1
    cell_image_size = 224
    mask = mask == 1
    cell_mask = ((mask == cell_ind) * 255).astype(int)
    left_limit = np.where(cell_mask.sum(axis=0) > 0)[0][0]
    right_limit = np.where(cell_mask.sum(axis=0) > 0)[0][-1]
    upper_limit = np.where(cell_mask.sum(axis=1) > 0)[0][0]
    lower_limit = np.where(cell_mask.sum(axis=1) > 0)[0][-1]
    center = (int(upper_limit + (lower_limit - upper_limit) / 2)), int((left_limit + (right_limit - left_limit) / 2))
    new_img = np.copy(img)[upper_limit : lower_limit, left_limit : right_limit, :]
    for c in range(4):
        new_img[:,:,c] = np.where(cell_mask[upper_limit : lower_limit, left_limit : right_limit], new_img[:,:,c], 0)

    pil_img = Image.fromarray(new_img)
    # pil_img = Image.merge('RGBA', [c.resize((cell_image_size,cell_image_size)) for c in pil_img.split()])
    return pil_img

def single_cell_mirror_no_gap_loader(path):
    mask = np.array(Image.fromarray(np.load(path.replace('Dino4Cells/datasets/HPA/data/','human_protein_atlas/website_small_subset_masks/mask_').replace('jpg','npy')) == 1).resize((512,512)))
    img  = np.array(Image.open(path))
    cell_ind = 1
    cell_image_size = 224
    mask = mask == 1
    cell_mask = ((mask == cell_ind) * 255).astype(int)
    left_limit = np.where(cell_mask.sum(axis=0) > 0)[0][0]
    right_limit = np.where(cell_mask.sum(axis=0) > 0)[0][-1]
    upper_limit = np.where(cell_mask.sum(axis=1) > 0)[0][0]
    lower_limit = np.where(cell_mask.sum(axis=1) > 0)[0][-1]
    center = (int(upper_limit + (lower_limit - upper_limit) / 2)), int((left_limit + (right_limit - left_limit) / 2))
    new_img = np.copy(img)[upper_limit : lower_limit, left_limit : right_limit, :]
    for c in range(4):
        new_img[:,:,c] = np.where(cell_mask[upper_limit : lower_limit, left_limit : right_limit], new_img[:,:,c], 0)
    img_shape = new_img.shape
    if (img_shape[0] >= img_shape[1]) and (img_shape[0] > cell_image_size):
        new_img = np.array(Image.fromarray(new_img).resize((int(img_shape[1] * cell_image_size / img_shape[0]), cell_image_size)))
    elif (img_shape[1] >= img_shape[0]) and (img_shape[1] > cell_image_size):
        new_img = np.array(Image.fromarray(new_img).resize((cell_image_size, int(img_shape[0] * cell_image_size / img_shape[1]))))
    img_shape = new_img.shape
    pad_border = 0
    upper_pad = int(min(max((cell_image_size - (img_shape[0] + pad_border)) / 2, 0), pad_border))
    lower_pad = int(min(max((cell_image_size - (img_shape[0] + pad_border)) / 2, 0), pad_border))
    left_pad  = int(min(max((cell_image_size - (img_shape[1] + pad_border)) / 2, 0), pad_border))
    right_pad = int(min(max((cell_image_size - (img_shape[1] + pad_border)) / 2, 0), pad_border))
    new_img = np.pad(new_img, ((upper_pad, lower_pad),
                            (left_pad, right_pad),
                            (0, 0)), 'constant').astype(np.uint8)
    img_shape = new_img.shape
    upper_pad = int((cell_image_size - img_shape[0]) / 2)
    lower_pad = max(cell_image_size - img_shape[0] - upper_pad, int((cell_image_size - img_shape[0]) / 2))
    left_pad  = int((cell_image_size - img_shape[1]) / 2)
    right_pad = max(cell_image_size - img_shape[1] - left_pad, int((cell_image_size - img_shape[1]) / 2))
    padded_img = np.pad(new_img, ((upper_pad, lower_pad),
                            (left_pad, right_pad),
                            (0, 0)), 'reflect').astype(np.uint8)
    pil_img = Image.fromarray(padded_img)
    return pil_img

def single_cell_mirror_loader(path):
    mask = np.array(Image.fromarray(np.load(path.replace('Dino4Cells/datasets/HPA/data/','human_protein_atlas/website_small_subset_masks/mask_').replace('jpg','npy')) == 1).resize((512,512)))
    img  = np.array(Image.open(path))
    cell_ind = 1
    cell_image_size = 224
    mask = mask == 1
    cell_mask = ((mask == cell_ind) * 255).astype(int)
    left_limit = np.where(cell_mask.sum(axis=0) > 0)[0][0]
    right_limit = np.where(cell_mask.sum(axis=0) > 0)[0][-1]
    upper_limit = np.where(cell_mask.sum(axis=1) > 0)[0][0]
    lower_limit = np.where(cell_mask.sum(axis=1) > 0)[0][-1]
    center = (int(upper_limit + (lower_limit - upper_limit) / 2)), int((left_limit + (right_limit - left_limit) / 2))
    new_img = np.copy(img)[upper_limit : lower_limit, left_limit : right_limit, :]
    for c in range(4):
        new_img[:,:,c] = np.where(cell_mask[upper_limit : lower_limit, left_limit : right_limit], new_img[:,:,c], 0)
    img_shape = new_img.shape
    if (img_shape[0] >= img_shape[1]) and (img_shape[0] > cell_image_size):
        new_img = np.array(Image.fromarray(new_img).resize((int(img_shape[1] * cell_image_size / img_shape[0]), cell_image_size)))
    elif (img_shape[1] >= img_shape[0]) and (img_shape[1] > cell_image_size):
        new_img = np.array(Image.fromarray(new_img).resize((cell_image_size, int(img_shape[0] * cell_image_size / img_shape[1]))))
    img_shape = new_img.shape
    pad_border = 10
    upper_pad = int(min(max((cell_image_size - (img_shape[0] + pad_border)) / 2, 0), pad_border))
    lower_pad = int(min(max((cell_image_size - (img_shape[0] + pad_border)) / 2, 0), pad_border))
    left_pad  = int(min(max((cell_image_size - (img_shape[1] + pad_border)) / 2, 0), pad_border))
    right_pad = int(min(max((cell_image_size - (img_shape[1] + pad_border)) / 2, 0), pad_border))
    new_img = np.pad(new_img, ((upper_pad, lower_pad),
                            (left_pad, right_pad),
                            (0, 0)), 'constant').astype(np.uint8)
    img_shape = new_img.shape
    upper_pad = int((cell_image_size - img_shape[0]) / 2)
    lower_pad = max(cell_image_size - img_shape[0] - upper_pad, int((cell_image_size - img_shape[0]) / 2))
    left_pad  = int((cell_image_size - img_shape[1]) / 2)
    right_pad = max(cell_image_size - img_shape[1] - left_pad, int((cell_image_size - img_shape[1]) / 2))
    padded_img = np.pad(new_img, ((upper_pad, lower_pad),
                            (left_pad, right_pad),
                            (0, 0)), 'reflect').astype(np.uint8)
    pil_img = Image.fromarray(padded_img)
    return pil_img

def threshold_new_channel_loader(path):
    # return Image.open(path).convert('RGBA')
    img = Image.open(path).convert('RGBA')
    img = np.array(img)
    protein_channel = 1
    # min_img = img[:, :, protein_channel].min()
    # max_img = img[:, :, protein_channel].max()
    # threshold = min_img + (max_img - min_img) / 4
    threshold = 30
    # img = np.concatenate((img, 255 * np.where( img[:, :, protein_channel] > threshold, img[:, :, protein_channel], 0)[:, :, np.newaxis]), axis=2)
    img = np.concatenate((img, np.where( img[:, :, protein_channel] > threshold, 255, 0)[:, :, np.newaxis]), axis=2)
    # img = Image.fromarray(img)
    return torch.Tensor(img)

def threshold_loader(path):
    img = Image.open(path).convert('RGBA')
    img = np.array(img)
    protein_channel = 1
    # min_img = img[:, :, protein_channel].min()
    # max_img = img[:, :, protein_channel].max()
    # threshold = min_img + (max_img - min_img) / 4
    threshold = 30
    img[:, :, protein_channel] = np.where( img[:, :, protein_channel] > threshold, img[:, :, protein_channel], 0)
    img = Image.fromarray(img)
    return img

def RGBA_norm_loader(path):
    img = Image.open(path)
    img = np.array(img).astype(float)
    img -= img.min(axis=(0,1))
    img *= (255 / img.max(axis=(0,1)))
    return Image.fromarray(img.astype(np.uint8))

def protein_transparency_loader(path):
    img = Image.open(path)
    new_img = Image.fromarray(255 * np.zeros((512,512,3)).astype(np.uint8))
    # new_img.paste(Image.fromarray(np.array(img)[:, :, [0,2,3]], 'RGB'), mask = Image.fromarray(np.array(img)[:, :, 1], 'L'))
    new_img.paste(Image.fromarray(np.array(img)[:, :, [0,2,3]], 'RGB'), mask = Image.fromarray((np.array(img)[:, :, 1] > 30).astype(np.uint8) * 255, 'L'))
    return new_img

def no_reticulum_loader(path):
    img = Image.open(path)
    img = np.array(img)[:, :, :3].astype(float)
    return Image.fromarray(img.astype(np.uint8))

def both_loader(path):
    img = Image.open(path)
    img = np.array(img)[:, :, :3].astype(float)
    img -= img.min(axis=(0,1))
    img *= (255 / img.max(axis=(0,1)))
    return Image.fromarray(img.astype(np.uint8))

def protein_loader(path):
    microtubules, protein, nucleus, endoplasmic_reticulum  = Image.open(path).split()
    return protein.convert('RGB')

def red_channel_loader(path):
    red, green, blue, yellow  = Image.open(path).split()
    return red.convert('RGB')

def green_channel_loader(path):
    red, green, blue, yellow  = Image.open(path).split()
    return green.convert('RGB')

def blue_channel_loader(path):
    red, green, blue, yellow  = Image.open(path).split()
    return blue.convert('RGB')

def yellow_channel_loader(path):
    red, green, blue, yellow  = Image.open(path).split()
    return yellow.convert('RGB')

def pandas_reader_bbbc021(flist):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    files = pd.read_csv(flist)[['file','MOA']]
    files = np.array(files.to_records(index = False))
    imlist = []
    for impath, moa in files:
        imlist.append( (impath, moa) )
    return imlist

def pandas_reader_website(flist):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    files = pd.read_csv(flist)[['file','protein_location','cell_type']]
    files = np.array(files.to_records(index = False))
    imlist = []
    for impath, protein_location, cell_type in files:
        if ('30871' in impath) or ('27093' in impath) or ('35134' in impath): continue
        imlist.append( (impath, protein_location, cell_type) )
    return imlist

def pandas_reader_no_labels(flist):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    files = pd.read_csv(flist)[['file','ID']]
    print(len(files))
    files = np.array(files.to_records(index = False))
    imlist = []
    for impath, imlabel in files:
        imlist.append( (impath, imlabel) )
    return imlist

def pandas_reader_with_filenames(flist):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    files = pd.read_csv(flist)[['file','label']]
    files = np.array(files.to_records(index = False))
    imlist = []
    for impath, imlabel in files:
        imlist.append( (impath, imlabel) )
    return imlist

def pandas_reader(flist):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    files = pd.read_csv(flist)[['file','label']]
    files = np.array(files.to_records(index = False))
    imlist = []
    for impath, imlabel in files:
        imlist.append( (impath, imlabel) )
    return imlist

def default_flist_reader(flist):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath, imlabel = line.strip().split(',')
            imlist.append( (impath, imlabel) )
    return imlist

class ImageFileList(data.Dataset):
    def __init__(self, flist, transform=None, target_transform=None,
            flist_reader=pandas_reader, loader=default_loader, training=True):
        self.imlist = flist_reader(flist)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.training = training

    def __getitem__(self, index):
        impath, protein, cell = self.imlist[index]
        # impath, target = self.imlist[index]
        img = self.loader(impath)
        # channels = img.split()
        # for c in channels:
        #     c = np.array(c)
        #     c -= c.mean()
        #     c /= c.std()

        if self.transform is not None:
            img = self.transform(img)
        if self.training:
            return img, protein
        else:
            return img, protein, cell

        # if self.transform is not None:
        #     img = self.transform(img)
        # return img, target

    # def __getitem__(self, index):
    #     impath, target = self.imlist[index]
    #     img = self.loader(impath)
    #     if self.transform is not None:
    #         img = self.transform(img)
    #     if self.target_transform is not None:
    #         target = self.target_transform(target)
    #     return img, target

    def __len__(self):
        return len(self.imlist)

class ImageFileList_BBBC021(data.Dataset):
    def __init__(self, flist, transform=None, target_transform=None,
            flist_reader=pandas_reader, loader=default_loader, training=True):
        self.imlist = flist_reader(flist)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.training = training

    def __getitem__(self, index):
        impath, moa = self.imlist[index]
        img = self.loader(impath)
        # channels = img.split()
        # for c in channels:
        #     c = np.array(c)
        #     c -= c.mean()
        #     c /= c.std()

        if self.transform is not None:
            img = self.transform(img)
        return img, moa

    # def __getitem__(self, index):
    #     impath, target = self.imlist[index]
    #     img = self.loader(impath)
    #     if self.transform is not None:
    #         img = self.transform(img)
    #     if self.target_transform is not None:
    #         target = self.target_transform(target)
    #     return img, target

    def __len__(self):
        return len(self.imlist)

class ImageFileList_with_filenames(data.Dataset):
    def __init__(self, flist, transform=None, target_transform=None,
            flist_reader=pandas_reader, loader=default_loader):
        self.imlist = flist_reader(flist)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        impath, target = self.imlist[index]
        img = self.loader(impath)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, impath

    def __len__(self):
        return len(self.imlist)

