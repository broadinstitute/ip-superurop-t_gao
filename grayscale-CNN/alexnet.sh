#!/bin/bash
#
# chtc-test.sh
#
# echo "Beginning CHTC test Job $1 running on `whoami`@`hostname`"
#
# mkdir outputs/$(Cluster)_$(Process)
# pwd
# ls -lt
du -lhs *
# more /proc/cpuinfo | grep flags
# du -lhs Greyscale_Images/centrosome/*
# echo "... about to run generate_png_from_tiff.py"
echo "===================================="
python3 generate_png_from_tiff.py
# du -lhs Greyscale_Images/centrosome/*
# echo "... finished running generate_png_from_tiff.py"
echo ""
du -lhs *
# echo "... about to run main.py"
python3 main.py
# echo "... finished running main.py"
#
# keep this job running for a few minutes so you'll see it in the queue:
# sleep 60
