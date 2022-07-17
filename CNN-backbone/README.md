# About

This subdirectory implements a CNN for image classification trained on images from the [Human Protein Atlas (HPA)](https://www.proteinatlas.org). In theory, the CNN impmlemented here is used to replace the VGG-16 CNN backbone currently used by ResNet.

As of now, the CNN used is a basic AlexNet, with performance as described on the [main page README](../README.md); **code for a WIP implementation of ResNet is available at [this draft pull request](https://github.com/broadinstitute/ip-superurop-t_gao/pull/7)**; see also https://github.com/broadinstitute/ip-superurop-t_gao/issues/8.


# Setup

This part of the project uses [this Docker Hub repo](https://hub.docker.com/repository/docker/teresahgao/superurop-grayscale-cnn).

## Running locally

1. Navigate to `repos/grayscale-CNN`. If `generate_png_from_tiff.py` has never been run, then run `python3 generate_png_from_tiff.py`. This converts the `.tiff` HPA images to `.png`s, which are smaller and also processable by Python.

2. Run `docker build -t local-grayscale-cnn .` to build from `Dockerfile`. You will need to rerun this in order for changes to propagate from `main.py` and other files to what you see in the container.

3. Run `docker run -it local-grayscale-cnn /bin/bash`. You are now in the Docker container!

4. Run `python3 main.py`. This is where the CNN is implemented.

5. To copy an output file from Docker container to host machine, run `docker cp <containerId>:/file/path/within/container /host/path/target`.


## Running on CHTC

> For instructions on getting started with CHTC, see [../CHTC_guidebook.md](../CHTC_guidebook.md).

0. Connect to CHTC by running `ssh <your-chtc-username>@submit1.chtc.wisc.edu` on your local terminal.

1. Clone this repo (https://github.com/broadinstitute/superurop-broad-cimini) and navigate to `repos/grayscale-CNN`.

2. Transfer `Greyscale_Images` to this directory. If `generate_png_from_tiff.py` has never been run, then run `python3 generate_png_from_tiff.py`.

3. Create the outputs directory by running `mkdir outputs`.

4. Run `condor_submit run.sub`.

5. To copy an output file from CHTC to host machine, run `scp <yourchtcusername>@submit1.chtc.wisc.edu:path/to/data local/destination/path/to/data`. To transfer directories, use the recursive tag `-r`.
