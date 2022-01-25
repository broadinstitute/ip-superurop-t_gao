#!/usr/bin/env python

import os
import pathlib
from PIL import Image

source_dir = "./Greyscale_Images"
verbose = True

print('entered generate_png_from_tiff.py')

if not os.path.exists(source_dir):
    os.mkdir('./Greyscale_Images_png/')

for child in os.listdir(source_dir):
    if verbose:
        print('\nchild:', child)
    if os.path.isdir(os.path.join(source_dir, child)):
        if verbose:
            print('\tchild is dir')
        subdirectory = child
        png_subdirectory = os.path.join('./Greyscale_Images_png', subdirectory)
        if not os.path.exists(png_subdirectory):
            os.mkdir(png_subdirectory)
        file_names = os.listdir(os.path.join(source_dir, subdirectory))

        for tiff_file in file_names:
            if tiff_file == '.DS_Store':
                continue
            original_image = Image.open(os.path.join(source_dir, subdirectory, tiff_file))
            png_filepath = os.path.join('./Greyscale_Images_png', subdirectory, tiff_file + '.png')
            original_image.save(png_filepath, format='png')

            png_image = Image.open(png_filepath)
            png_image.thumbnail((256,256)) # resize image to 256x256
            png_image.save(png_filepath, format='png')

            if verbose:
                print('\tprocessed', png_filepath)
            #     print('\n\n\ntiff file is', tiff_file)
            #     print('\n\n\nsaved ' + os.path.join('./Greyscale_Images_png', subdirectory, tiff_file + '.png'))

print('done with generate_png_from_tiff.py')
