#!/usr/bin/env python

import os
import pathlib
from PIL import Image

source_dir = "./Greyscale_Images"
verbose = False

print('entered generate_png_from_tiff.py')

os.mkdir('./Greyscale_Images_png/');

for child in os.listdir(source_dir):
    if verbose:
        print('\nchild:', child)
    if os.path.isdir(os.path.join(source_dir, child)):
        if verbose:
            print('\n\nchild is dir')
        subdirectory = child
        os.mkdir(os.path.join('./Greyscale_Images_png', subdirectory))
        file_names = os.listdir(os.path.join(source_dir, subdirectory))

        for tiff_file in file_names:
            if tiff_file == '.DS_Store':
                continue
            original_image = Image.open(os.path.join(source_dir, subdirectory, tiff_file))
            original_image.save(os.path.join('./Greyscale_Images_png', subdirectory, tiff_file + '.png'), format='png')
            if verbose:
                print('\n\n\ntiff file is', tiff_file)
                print('\n\n\nsaved ' + os.path.join('./Greyscale_Images_png', subdirectory, tiff_file + '.png'))

print('done with generate_png_from_tiff.py')
