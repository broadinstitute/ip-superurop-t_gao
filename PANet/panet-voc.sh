#!/bin/bash
#
export NEPTUNE_API_TOKEN=`cat neptune-api-token.txt`
#
mkdir f1/
mkdir f1/f2/
mv train-voc.py f1/f2/
mv config_voc.py f1/f2/
mv util f1/f2/
mv dataloaders/ f1/f2/
mv models/ f1/f2/
#
mkdir data/
mkdir data/Pascal/
unzip VOCdevkit.zip -d data/Pascal/ # -d VOCdevkit
unzip pretrained_model.zip -d f1/f2/ # -d pretrained_model
#
cd f1/f2
python3 train-voc.py
#
echo "Finished CHTC Job $1"
