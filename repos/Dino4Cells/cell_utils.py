import numbers
from typing import Tuple, List, Optional
import torchvision
import math
from torch import Tensor
import warnings
from collections.abc import Sequence
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from skimage.filters import threshold_otsu
from PIL import Image, ImageFilter, ImageOps
import kornia as K

_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
    Image.HAMMING: 'PIL.Image.HAMMING',
    Image.BOX: 'PIL.Image.BOX',
}

################### Fast-AI inspired transforms ###################

class Warp_cell(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img):
        img = torchvision.transforms.ToTensor()(img).unsqueeze(0)
        points_src = torch.Tensor([[
            [0., 0.],
            [float(img.shape[2]), 0.],
            [float(img.shape[2]), float(img.shape[3])],
            [0, float(img.shape[3])],
        ]])

        shift = max(int(min(img.shape[2], img.shape[3]) / 4), 1)
        points_dst = (points_src).clone()
        corner_ind_0 = np.random.randint(4)
        for corner_ind_1 in [0,1]:
            val = points_dst[0, corner_ind_0, corner_ind_1]
            if val == 0:
                points_dst[0, corner_ind_0, corner_ind_1] = points_dst[0, corner_ind_0, corner_ind_1] + np.random.randint(0, shift)
            else:
                points_dst[0, corner_ind_0, corner_ind_1] = points_dst[0, corner_ind_0, corner_ind_1] - np.random.randint(0, shift)

        M = K.get_perspective_transform(points_src, points_dst)
        img_warp = K.warp_perspective(img.float(), M, dsize=(img.shape[2], img.shape[3]))
        return Image.fromarray((np.array(img_warp[0, :, :].permute(1,2,0) * 255)).astype(np.uint8))


def dihedral(x:torch.Tensor, k):
        if k in [1,3,4,7]: x = x.flip(-1)
        if k in [2,4,5,7]: x = x.flip(-2)
        if k in [3,5,6,7]: x = x.transpose(-1,-2)
        return x

class rnd_dihedral(object):
    def __call__(self, x):
        k = np.random.randint(0,7)
        return dihedral(x, k)

class Rotate_dihedral(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img):
        x = transforms.ToTensor()(img)
        k = np.random.randint(0,7)
        x = (dihedral(x, k).numpy() * 255).astype(np.uint8).transpose(1,2,0)
        x = (Image.fromarray(np.array(x)))
        return x

class self_normalize(object):
    def __call__(self, x):
        m = x.mean((-2,-1), keepdim=True)
        s = x.std((-2,-1), unbiased=False, keepdim=True)
        x -= m
        x /= (s + 1e-7)
        return x

class to_np(object):
    def __call__(self, x):
        return np.array(x)

def FA2_grid_sample(x, coords, mode='bilinear', padding_mode='reflection', align_corners=None):
    "Resample pixels in `coords` from `x` by `mode`, with `padding_mode` in ('reflection','border','zeros')."
    #coords = coords.permute(0, 3, 1, 2).contiguous().permute(0, 2, 3, 1) # optimize layout for grid_sample
    if mode=='bilinear': # hack to get smoother downwards resampling
        mn,mx = coords.min(),coords.max()
        # max amount we're affine zooming by (>1 means zooming in)
        z = 1/(mx-mn).item()*2
        # amount we're resizing by, with 100% extra margin
        d = min(x.shape[-2]/coords.shape[-2], x.shape[-1]/coords.shape[-1])/2
        # If we're resizing up by >200%, and we're zooming less than that, interpolate first
        if d>1 and d>z:
            # Pytorch > v1.4.x needs an extra argument when calling nn.functional.interpolate to preserve previous behaviour
            if (int(torch.__version__[0:4].replace(".", "")) > 14):
                x = F.interpolate(x, scale_factor=1/d, mode='area', recompute_scale_factor=True)
            else:
                x = F.interpolate(x, scale_factor=1/d, mode='area')
    return F.grid_sample(x, coords, mode=mode, padding_mode=padding_mode, align_corners=align_corners)

def _init_mat(x):
    mat = torch.eye(3, device=x.device).float()
    return mat.unsqueeze(0).expand(x.size(0), 3, 3).contiguous()

def FA2affine_grid(theta, size, align_corners=None):
#     return TensorFlowField(F.affine_grid(theta, size, align_corners=align_corners))
    return torch.Tensor(F.affine_grid(theta, size, align_corners=align_corners))

def FA2affine_coord(x, mat=None, coord_tfm=None, sz=None, mode='bilinear',
                 pad_mode='reflection', align_corners=True):
    if mat is None and coord_tfm is None and sz is None: return x
    size = tuple(x.shape[-2:]) if sz is None else (sz,sz) if isinstance(sz,int) else tuple(sz)
    if mat is None: mat = _init_mat(x)[:,:2]
    coords = FA2affine_grid(mat, x.shape[:2] + size, align_corners=align_corners)
#     if coord_tfm is not None: coords = coord_tfm(coords)
#     return TensorImage(FA2_grid_sample(x, coords, mode=mode, padding_mode=pad_mode, align_corners=align_corners))
    return torch.Tensor(FA2_grid_sample(x, coords, mode=mode, padding_mode=pad_mode, align_corners=align_corners))

class FA_resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, x):
        im = FA2affine_coord(x[None,...], sz=224)
        return im[0]

################### PIL/Torch transforms ###################

def resize_for_5_channels(image, size):
    return Image.merge('RGBA', [c.resize(size) for c in image.split()])

def resize_RGBA(image, size):
    return Image.merge('RGBA', [c.resize(size) for c in image.split()])

def _setup_size(size, error_msg):
    if isinstance(size, numbers.Number):
        return int(size), int(size)

    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)

    return size


class Permute(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img):
        return img.permute(2,0,1)


def resized_crop_for_5_channels(
        img: Tensor, top: int, left: int, height: int, width: int, size: List[int], interpolation: int = Image.BILINEAR
) -> Tensor:

    img = torchvision.transforms.functional.resized_crop(img.permute(2,0,1),
                                                         top = top,
                                                         left = left,
                                                         height = height,
                                                         width = width,
                                                         size = size).permute(1,2,0)
    return img

def resized_crop(
        img: Tensor, top: int, left: int, height: int, width: int, size: List[int], interpolation: int = Image.BILINEAR
) -> Tensor:
    img = img.crop((top, left, top + height, left + width))
    img = resize_RGBA(img, size)
    return img

class RandomResizedCrop_for_5_channels(torch.nn.Module):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR):
        super().__init__()
        self.size = _setup_size(size, error_msg="Please provide only two dimensions (h, w) for size.")

        if not isinstance(scale, Sequence):
            raise TypeError("Scale should be a sequence")
        if not isinstance(ratio, Sequence):
            raise TypeError("Ratio should be a sequence")
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("Scale and ratio should be of kind (min, max)")

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(
            img: Tensor, scale: List[float], ratio: List[float]
    ) -> Tuple[int, int, int, int]:
        width, height = torchvision.transforms.functional._get_image_size(img)
        area = height * width

        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            log_ratio = torch.log(torch.tensor(ratio))
            aspect_ratio = torch.exp(
                torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
            ).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w


    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped and resized.

        Returns:
            PIL Image or Tensor: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return resized_crop_for_5_channels(img, i, j, h, w, self.size, self.interpolation)


    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string

class RandomResizedCenterCrop(torch.nn.Module):
    """
    Crop an image to a given size. Coordinates for crop are randomly drawn from a guassian
    distribution and will hence primarily target the center of the image.

    The image can be a PIL Image or a Tensor, in which case it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions

    Args:
        size (int or sequence): expected output size of each edge. If size is an
            int instead of sequence like (h, w), a square output size ``(size, size)`` is
            made. If provided a tuple or list of length 1, it will be interpreted as (size[0], size[0]).
        scale (tuple of float): range of size of the origin size cropped
        interpolation (int): Desired interpolation enum defined by `filters`_.
            Default is ``PIL.Image.BILINEAR``. If input is Tensor, only ``PIL.Image.NEAREST``, ``PIL.Image.BILINEAR``
            and ``PIL.Image.BICUBIC`` are supported.
        depth (int or float): sampling depth for the generated distribution.
        s (float): desired standard deviation for the generated guassian distribution. Use this to control the degree to which
            crops are biased to originate from the center of the image (lower s -> stronger bias).
    """
    def __init__(self, size, scale, interpolation=Image.BILINEAR, depth=1e6, s=1.0):
        super().__init__()
        self.size = _setup_size(size, error_msg="Please provide only two dimensions (h, w) for size.")
        self.scale = scale
        self.dist = self.getdistrib(depth, s)
        self.interpolation = interpolation

    def getdistrib(self, depth, s):
        d = torch.randn(int(depth))
        mx = torch.max(d)
        mn = torch.abs(torch.min(d))
        d = d*s
        d = (d + mn) / (mn+mx)
        return d

    @staticmethod
    def get_params(img, scale, dist):
        sz = img.size()[-1]
        target_area = sz * torch.empty(1).uniform_(scale[0], scale[1]).item()
        tsz = int(round(target_area))

        draw = torch.randint(0, dist.size()[0], size=(2,))
        i = int(torch.round(dist[draw[0]] * (sz - tsz)))
        j = int(torch.round(dist[draw[1]] * (sz - tsz)))
        return i, j, tsz, tsz

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped and resized. Expected in [...,h,w]

        Returns:
            PIL Image or Tensor: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.dist)
        return transforms.functional.resized_crop(img, i, j, h, w, self.size, self.interpolation)

class RandomResizedCrop(torch.nn.Module):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR):
        super().__init__()
        self.size = _setup_size(size, error_msg="Please provide only two dimensions (h, w) for size.")

        if not isinstance(scale, Sequence):
            raise TypeError("Scale should be a sequence")
        if not isinstance(ratio, Sequence):
            raise TypeError("Ratio should be a sequence")
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("Scale and ratio should be of kind (min, max)")

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(
            img: Tensor, scale: List[float], ratio: List[float]
    ) -> Tuple[int, int, int, int]:
        width, height = torchvision.transforms.functional._get_image_size(img)
        area = height * width

        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            log_ratio = torch.log(torch.tensor(ratio))
            aspect_ratio = torch.exp(
                torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
            ).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w


    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped and resized.

        Returns:
            PIL Image or Tensor: Randomly cropped and resized image.
        """

        i, j, h, w = self.get_params(img, self.scale, self.ratio)
#         return transforms.functional.resized_crop(img, i, j, h, w, self.size, self.interpolation)
        return resized_crop(img, i, j, h, w, self.size, self.interpolation)


    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string

def solarize_for_RGBA(image, threshold=128):
    lut = []
    for i in range(256):
        if i < threshold:
            lut.append(i)
        else:
            lut.append(255-i)
    lut = lut + lut + lut + lut
    return image.point(lut)

class Solarization_for_5_channels(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.random() < self.p:
            partial_image = Image.fromarray(img[:, :, :4].numpy().astype(np.uint8))
            partial_image = solarize_for_RGBA(partial_image)
            return torch.Tensor(np.concatenate((np.array(partial_image), img[:, :, [4]].numpy()), axis=2))
        else:
            return img

class Solarization_for_RGBA(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.random() < self.p:
            img = solarize_for_RGBA(img)
            return img
        else:
            return img

class GaussianBlur_for_5_channels(torch.nn.Module):
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = np.random.random() <= self.prob
        if not do_it:
            return img
        partial_image = Image.fromarray(img[:, :, :4].numpy().astype(np.uint8))

        partial_image = partial_image.filter(
            ImageFilter.GaussianBlur(
                radius=np.random.uniform(self.radius_min, self.radius_max)))
        return torch.Tensor(np.concatenate((np.array(partial_image), img[:, :, [4]].numpy()), axis=2))

class RandomGrayscale_for_5_channels(torch.nn.Module):
    def __init__(self, p=0.1):
        super().__init__()
        self.p = p

    def forward(self, img):
        if torch.rand(1) < self.p:
            return torch.Tensor(np.concatenate((np.repeat((img[:, :, :4]).mean(axis=2)[:, :, np.newaxis], 4, 2).numpy().astype(np.uint8), img[:, :, [4]]), axis=2))
        return img

class Single_cell_random_resize(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img):
        row_cell_image_size = np.random.randint(180, 224)
        col_cell_image_size = np.random.randint(180, 224)
        img = Image.merge('RGBA', [c.resize((row_cell_image_size,col_cell_image_size)) for c in img.split()])
        cell_image_size = 224
        img_shape = (row_cell_image_size, col_cell_image_size)
        upper_pad = int((max((cell_image_size - (img_shape[1])) / 2, 0)))
        lower_pad = int((max((cell_image_size - (img_shape[1])) / 2, 0)))
        left_pad  = int((max((cell_image_size - (img_shape[0])) / 2, 0)))
        right_pad = int((max((cell_image_size - (img_shape[0])) / 2, 0)))
        upper_pad += 224 - (upper_pad + lower_pad + img_shape[1])
        left_pad += 224 - (right_pad + left_pad + img_shape[0])
        new_img = np.pad(img, ((upper_pad, lower_pad),
                            (left_pad, right_pad),
                            (0, 0)), 'constant').astype(np.uint8)
        return Image.fromarray(new_img)

class RandomGrayscale_for_RGBA(torch.nn.Module):
    def __init__(self, p=0.1):
        super().__init__()
        self.p = p

    def forward(self, img):
        if torch.rand(1) < self.p:
            return Image.fromarray(np.repeat(np.array(img).mean(axis=2)[:, :, np.newaxis], 4, 2).astype(np.uint8))
        return img

class remove_channel(torch.nn.Module):
    def __init__(self, p = 0.2):
        super().__init__()
        self.p = p

    def forward(self, img):
        if len(img.mode) < 4:
            return img
        img_size = np.array(img).shape
        if np.random.rand() <= self.p:
            channel_to_blacken = np.random.choice(np.array([0,2,3]), 1, replace=False)[0]
            channels = list(img.split())
            channels[channel_to_blacken] = Image.fromarray(np.zeros((img_size[0], img_size[0])).astype(np.uint8))
            new_img = Image.merge('RGBA', [channels[i] for i in [0,1,2,3]])
            return new_img
        else:
            return img

class rescale_protein(torch.nn.Module):
    def __init__(self, p = 0.2):
        super().__init__()
        self.p = p

    def forward(self, img):
        if np.random.rand() <= self.p:
            channels = list(img.split())
            protein_channel = 1
            random_factor = np.random.rand() * 2
            channels[protein_channel] = np.array(channels[protein_channel]).astype(float)
            channels[protein_channel] = Image.fromarray((channels[protein_channel] * random_factor).astype(np.uint8))
            new_img = Image.merge('RGBA', [channels[i] for i in [0,1,2,3]])
            return new_img
        else:
            return img

def change_contrast(img, level):
    mean_img = np.array(img).mean()
    def contrast(pixel):
        pixel = pixel + (mean_img - pixel) * (1 - level)
        return max(0, min(255, pixel))
    return img.point(contrast)

class Threshold_protein(torch.nn.Module):
    def __init__(self, p=0.2):
        super().__init__()
        self.p = p

    def forward(self, img):
        if np.random.rand() < self.p:
            channels = list(img.split())
            protein_channel = 1
            arr = np.array(channels[protein_channel])
            # thresh = threshold_otsu(arr)
            thresh = np.random.randint(0, 50)
            channels[protein_channel] = Image.fromarray(np.where(arr > thresh, arr, 0))
            return Image.merge('RGBA', channels)
        return img

class Jigsaw(torch.nn.Module):
    def __init__(self, p=0.2):
        super().__init__()
        self.p = p
    def forward(self, img):
        img_array = np.array(img)
        if np.random.rand() < self.p:
            rows = np.linspace(0, img_array.shape[0], 5).astype(int)
            cols = np.linspace(0, img_array.shape[0], 5).astype(int)
            all_tiles = []
            for r, _ in enumerate(rows[:-1]):
                for c, _ in enumerate(cols[:-1]):
                    all_tiles.append(img_array[rows[r] : rows[r + 1], cols[c] : cols[c + 1], :])
            np.random.shuffle(all_tiles)
            new_img_array = np.zeros(img_array.shape)
            for r, _ in enumerate(rows[:-1]):
                for c, _ in enumerate(cols[:-1]):
                    new_img_array[rows[r] : rows[r + 1], cols[c] : cols[c + 1], :] = all_tiles[r + c * 4]
            new_img = Image.fromarray(new_img_array.astype(np.uint8))
            return new_img
        else:
            return img

class Change_contrast(torch.nn.Module):
    def __init__(self, p=0.2):
        super().__init__()
        self.p = p

    def forward(self, img):
        channels = list(img.split())
        for ind in range(len(channels)):
            if np.random.rand() < (self.p / len(channels)):
                contrast_factor = np.random.rand() * 2
                channels[ind] = change_contrast(channels[ind], contrast_factor)
        return Image.merge('RGBA', channels)

class Single_cell_Resize(torch.nn.Module):
    def __init__(self, size=224):
        super().__init__()
        self.cell_image_size = size

    def forward(self, img):
        return Image.merge('RGBA', [c.resize((self.cell_image_size, self.cell_image_size)) for c in img.split()])

class Rotate_single_cell(torch.nn.Module):
    def __init__(self, p=0.8):
        super().__init__()
        self.p = p

    def forward(self, img):
        if np.random.rand() < (self.p):
            angle = np.random.randint(0, 360)
            img = img.rotate(angle)
        return img

class Change_brightness(torch.nn.Module):
    def __init__(self, p=0.2):
        super().__init__()
        self.p = p

    def forward(self, img):
        channels = list(img.split())
        for ind in range(len(channels)):
            if np.random.rand() < (self.p / len(channels)):
                brightness_factor = np.random.rand() * 2
                channels[ind] = torchvision.transforms.functional_pil.adjust_brightness(
                    channels[ind], brightness_factor)
        return Image.merge('RGBA', channels)

class ColorJitter_for_5_channels(torch.nn.Module):
    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1):
        super().__init__()
        self.trans = transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    def forward(self, tensor):
        img = Image.fromarray(tensor[:, :, :4].numpy().astype(np.uint8))
        channel_indices = np.random.choice(range(4), 3, replace=False)
        channels = list(img.split())
        jittered_image = Image.merge('RGB', [channels[i] for i in channel_indices])
        jittered_image = self.trans(jittered_image)
        jittered_channels = jittered_image.split()
        for ind, channel_ind in enumerate(channel_indices):
            channels[channel_ind] = jittered_channels[ind]
        for ind, c in enumerate(channels):
            channels[ind] = np.array(channels[ind])[:, :, np.newaxis]
        return torch.Tensor(np.concatenate((np.concatenate(channels, axis=2), tensor[:, :, 4].numpy()[:, :, np.newaxis]), axis=2))

class ColorJitter_for_RGBA(torch.nn.Module):
    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1):
        super().__init__()
        self.trans = transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    def forward(self, img):
        channel_indices = np.random.choice(range(4), 3, replace=False)
        channels = list(img.split())
        jittered_image = Image.merge('RGB', [channels[i] for i in channel_indices])
        jittered_image = self.trans(jittered_image)
        jittered_channels = jittered_image.split()
        for ind, channel_ind in enumerate(channel_indices):
            channels[channel_ind] = jittered_channels[ind]
        return Image.merge('RGBA', channels)

class Get_specific_channel(torch.nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c

    def forward(self, img):
        channels = img.split()
        return channels[self.c]

