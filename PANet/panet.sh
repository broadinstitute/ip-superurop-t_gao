#!/bin/bash
#
# panet.sh
#
echo "Beginning CHTC Job $1"
#
export NEPTUNE_API_TOKEN=`cat neptune-api-token.txt`
#
mkdir f1/
mkdir f1/f2/
mv train.py f1/f2/
# mv train.py f1/f2/
mv config.py f1/f2/
mv util f1/f2/
mv dataloaders/ f1/f2/
mv models/ f1/f2/
#
mkdir data/
unzip PbMgNr_png.zip -d data/
unzip Greyscale_Images_png.zip -d data/
unzip pretrained_model.zip -d f1/f2/ # -d pretrained_model
echo "unzipped all ZIP files"
#
cd f1/f2
# python3 move_grayscale_under_ten.py
# python3 generate_png_from_tiff.py
# echo "ran move_grayscale_under_ten.py and generate_png_from_tiff.py"
#
python3 train.py
#
echo "Finished CHTC Job $1"
