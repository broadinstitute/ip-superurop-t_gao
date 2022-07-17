#!/usr/bin/env python

import os
import pathlib
from PIL import Image #, TiffImagePlugin
import tifffile as tiff
import numpy as np

def generate_jpg_from_rgb_tiff(source_dir, verbose=True):

    print('calling generate_jpg_from_tiff for source_dir', source_dir)

    if not os.path.exists(source_dir + '_jpg/'):
        os.mkdir(source_dir + '_jpg/')

    for child in os.listdir(source_dir):
        if verbose:
            print('\nchild:', child)
        if os.path.isdir(os.path.join(source_dir, child)):
            if verbose:
                print('\tchild is dir')
            subdirectory = child
            jpg_subdirectory = os.path.join(source_dir + '_jpg/', subdirectory)
            if not os.path.exists(jpg_subdirectory):
                print('jpg_subdirectory', jpg_subdirectory, 'does not exist, so creating')
                os.mkdir(jpg_subdirectory)
            file_names = os.listdir(os.path.join(source_dir, subdirectory))

            for tiff_file in file_names:
                if tiff_file == '.DS_Store':
                    continue

                tiff_array = tiff.imread(os.path.join(source_dir, subdirectory, tiff_file))
                normalized_tiff_array = (tiff_array - np.min(tiff_array)) / (np.max(tiff_array) - np.min(tiff_array))

                jpg_filepath = os.path.join(jpg_subdirectory, tiff_file + '.jpg')
                # tiff.imsave(jpg_filepath, original_image)

                # jpg_image = tiff.imread(jpg_filepath)
                jpg_image = Image.fromarray(np.uint8(normalized_tiff_array*255))
                jpg_image.thumbnail((256,256)) # resize image to 256x256
                jpg_image.save(jpg_filepath, format='JPEG')

                if verbose:
                    print('\tprocessed', jpg_filepath)
                #     print('\n\n\ntiff file is', tiff_file)
                #     print('\n\n\nsaved ' + os.path.join('./Greyscale_Images_jpg', subdirectory, tiff_file + '.jpg'))

generate_jpg_from_rgb_tiff('./PbMgNr')
