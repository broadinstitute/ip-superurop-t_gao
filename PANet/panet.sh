#!/bin/bash
#
# chtc-test.sh
#
echo "Beginning CHTC Job $1"
#
#which git
#whereis git
#git --version
#git --exec-path
#whereis git
#where is git
#
export NEPTUNE_API_TOKEN=`cat neptune-api-token.txt`
#
mkdir f1/
mkdir f1/f2/
mv train.py f1/f2/
mv config.py f1/f2/
mv util f1/f2/
mv dataloaders/ f1/f2/
mv models/ f1/f2/
#
mkdir data/
mkdir data/Pascal/
unzip VOCdevkit.zip -d data/Pascal/ # -d VOCdevkit
unzip pretrained_model.zip -d f1/f2/ # -d pretrained_model
# ls -lt
#ls -lt
#echo "----------"
#ls -lt VOCdevkit
#echo "----------"
#ls -lt pretrained_model
#
# docker system prune -a -f
#
cd f1/f2
python3 train.py
#
#mkdir donttransfer
#FILES="*.hdf5
#*.png"
#for f in $FILES
#do
#  echo "processing file $f"
#  cp $f $1_$(basename -- $f)
#  mv $f donttransfer
#done
#
echo "Finished CHTC Job $1"
