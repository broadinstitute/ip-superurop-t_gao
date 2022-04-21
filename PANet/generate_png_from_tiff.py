#!/usr/bin/env python

import os
import pathlib
from PIL import Image #, TiffImagePlugin
import tifffile as tiff
import numpy as np

print('entered generate_png_from_tiff.py')

def generate_png_from_grayscale_tiff(source_dir, verbose=True):

    print('\n-------------------------')
    print('calling generate_png_from_tiff for source_dir', source_dir)

    if not os.path.exists(source_dir + '_png/'):
        os.mkdir(source_dir + '_png/')

    for child in os.listdir(source_dir):
        if verbose:
            print('\nchild:', child)
        if os.path.isdir(os.path.join(source_dir, child)):
            if verbose:
                print('\tchild is dir')
            subdirectory = child
            png_subdirectory = os.path.join(source_dir + '_png/', subdirectory)
            if not os.path.exists(png_subdirectory):
                print('png_subdirectory', png_subdirectory, 'does not exist, so creating')
                os.mkdir(png_subdirectory)
            file_names = os.listdir(os.path.join(source_dir, subdirectory))

            for tiff_file in file_names:
                if tiff_file == '.DS_Store':
                    continue
                original_image = Image.open(os.path.join(source_dir, subdirectory, tiff_file))
                png_filepath = os.path.join(png_subdirectory, tiff_file + '.png')
                original_image.save(png_filepath, format='png')

                png_image = Image.open(png_filepath)
                png_image.thumbnail((256,256)) # resize image to 256x256
                png_image.save(png_filepath, format='png')

                if verbose:
                    print('\tprocessed', png_filepath)
                #     print('\n\n\ntiff file is', tiff_file)
                #     print('\n\n\nsaved ' + os.path.join('./Greyscale_Images_png', subdirectory, tiff_file + '.png'))

generate_png_from_grayscale_tiff('./Greyscale_Images')
# generate_png_from_tiff('./PbMgNr')

print('done with generate_png_from_tiff.py')

##############################################

print('\n-------------------------\n')
print('entered generate_png_from_tiff.py')

def generate_png_from_rgb_tiff(source_dir, verbose=True):

    print('calling generate_png_from_tiff for source_dir', source_dir)

    if not os.path.exists(source_dir + '_png/'):
        os.mkdir(source_dir + '_png/')

    for child in os.listdir(source_dir):
        if verbose:
            print('\nchild:', child)
        if os.path.isdir(os.path.join(source_dir, child)):
            if verbose:
                print('\tchild is dir')
            subdirectory = child
            png_subdirectory = os.path.join(source_dir + '_png/', subdirectory)
            if not os.path.exists(png_subdirectory):
                print('png_subdirectory', png_subdirectory, 'does not exist, so creating')
                os.mkdir(png_subdirectory)
            file_names = os.listdir(os.path.join(source_dir, subdirectory))

            for tiff_file in file_names:
                if tiff_file == '.DS_Store':
                    continue

                tiff_array = tiff.imread(os.path.join(source_dir, subdirectory, tiff_file))
                normalized_tiff_array = (tiff_array - np.min(tiff_array)) / (np.max(tiff_array) - np.min(tiff_array))

                png_filepath = os.path.join(png_subdirectory, tiff_file + '.png')
                # tiff.imsave(png_filepath, original_image)

                # png_image = tiff.imread(png_filepath)
                png_image = Image.fromarray(np.uint8(normalized_tiff_array*255))
                png_image.thumbnail((256,256)) # resize image to 256x256
                png_image.save(png_filepath, format='png')

                if verbose:
                    print('\tprocessed', png_filepath)
                #     print('\n\n\ntiff file is', tiff_file)
                #     print('\n\n\nsaved ' + os.path.join('./Greyscale_Images_png', subdirectory, tiff_file + '.png'))

generate_png_from_rgb_tiff('./PbMgNr')

print('done with generate_png_from_tiff.py')
print('\n-------------------------\n')
