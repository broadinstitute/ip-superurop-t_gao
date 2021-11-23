#!/usr/bin/env python

import os
import pathlib
from PIL import Image

source_dir = "./Greyscale_Images/"

print('entered generate_png_from_tiff.py')
for child in os.listdir(source_dir):
    print('\nchild:', child)
    if os.path.isdir(os.path.join(source_dir, child)):
        print('\n\nchild is dir')
        subdirectory = child
        file_names = os.listdir(os.path.join(source_dir, subdirectory))

        for tiff_file in file_names:
            print('\n\n\ntiff file is', tiff_file)
            original_image = Image.open(os.path.join(source_dir, subdirectory, tiff_file))
            # print('\topened image is', original_image)
            original_image.save(os.path.join(source_dir, subdirectory, tiff_file + '.png'), format='png')

print('done with generate_png_from_tiff.py')
