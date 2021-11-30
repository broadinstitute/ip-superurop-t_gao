# running on CHTC

1. Navigate to `repos/grayscale-CNN`.

2. Transfer `Greyscale_Images` to this directory. If `generate_png_from_tiff.py` has never been run, then run `python3 generate_png_from_tiff.py`.

3. Create the outputs directory by running `mkdir outputs`.

4. Run `condor_submit alexnet.sub`.

# running locally

1. Navigate to `repos/grayscale-CNN`. If `generate_png_from_tiff.py` has never been run, then run `python3 generate_png_from_tiff.py`.

2. Run `docker build -t local-grayscale-cnn .` to build from `Dockerfile`.

3. Run `docker run -it local-grayscale-cnn /bin/bash`. You are now in the Docker container!

4. Run `python3 main.py`.
