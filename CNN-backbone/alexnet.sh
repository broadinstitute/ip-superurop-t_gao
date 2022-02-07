#!/bin/bash
#
# chtc-test.sh
#
echo "Beginning CHTC Job $1"
#
export NEPTUNE_API_TOKEN=`cat neptune-api-token.txt`
python3 main.py
#
mkdir donttransfer
FILES="*.hdf5
*.png"
for f in $FILES
do
  echo "processing file $f"
  cp $f $1_$(basename -- $f)
  mv $f donttransfer
done
#
echo "Finished CHTC Job $1"
