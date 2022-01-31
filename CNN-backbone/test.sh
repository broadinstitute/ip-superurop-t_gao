#!/bin/bash
#
# chtc-test.sh
#
# echo "Beginning CHTC test Job $1 running on `whoami`@`hostname`"
#
echo "cluster $1"
echo "process $2"
#mkdir outputs/$(Cluster)_$(Process)
#echo "in home directory" > homedir.txt
#cd outputs
#echo "in outputs" > outputs.txt
#cd $(Cluster)_$(Process)
#echo "in cluster_process $(Cluster)_$(Process)" > clusterprocess.txt
#cd ../..
mkdir outputs/
echo "first file" > outputs/first.txt
echo "second file" > outputs/second.txt
cd outputs/
echo "third file" > third.txt
cd ..
for f in outputs/*.txt
do
  echo "processing file $f"
  #fullfilename = $f
  # copy to main directory by getting filename and extension without filepath
  mv $f $1_$(basename -- $f)
done
#tar -czf $(Cluster)_$(Process)_output.tar.gz outputs/$(Cluster)_$(Process)_output/
# pwd
# ls -lt
# du -lhs *
# more /proc/cpuinfo | grep flags
# du -lhs Greyscale_Images/centrosome/*
# echo "... about to run generate_png_from_tiff.py"
#echo "===================================="
#python3 generate_png_from_tiff.py
# du -lhs Greyscale_Images/centrosome/*
# echo "... finished running generate_png_from_tiff.py"
#echo ""
#du -lhs *
# echo "... about to run main.py"
#python3 main.py
# echo "... finished running main.py"
#
# keep this job running for a few minutes so you'll see it in the queue:
# sleep 60
