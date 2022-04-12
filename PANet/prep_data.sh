#!/bin/bash
#
# panet.sh
#
python3 move_grayscale_under_ten.py
python3 generate_png_from_tiff.py
#
zip PbMgNr
zip Greyscale_Images
