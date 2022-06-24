# running locally

1. Navigate to `repos/grayscale-CNN`. If `generate_png_from_tiff.py` has never been run, then run `python3 generate_png_from_tiff.py`.

2. Run `docker build -t local-grayscale-cnn .` to build from `Dockerfile`. You will need to rerun this in order for changes to propagate from `main.py` and other files to what you see in the container.

3. Run `docker run -it local-grayscale-cnn /bin/bash`. You are now in the Docker container!

4. Run `python3 main.py`.

5. To copy an output file from Docker container to host machine, run `docker cp <containerId>:/file/path/within/container /host/path/target`.


# running on CHTC

> For instructions on getting started with CHTC, see this unofficial [CHTC guidebook](https://docs.google.com/document/d/1arRuX7-QuKWpS1xej4o_pZevHEmNcbl7WsQGi13qI8Q/edit#)

0. Connect to CHTC by running `ssh <your-chtc-username>@submit1.chtc.wisc.edu` on your local terminal.

1. Clone this repo (https://github.com/broadinstitute/superurop-broad-cimini) and navigate to `repos/grayscale-CNN`.

2. Transfer `Greyscale_Images` to this directory. If `generate_png_from_tiff.py` has never been run, then run `python3 generate_png_from_tiff.py`.

3. Create the outputs directory by running `mkdir outputs`.

4. Run `condor_submit alexnet.sub`.

5. To copy an output file from CHTC to host machine, run `scp <yourchtcusername>@submit1.chtc.wisc.edu:path/to/data local/destination/path/to/data`. To transfer directories, use the recursive tag `-r`.
