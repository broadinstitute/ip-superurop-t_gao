# running on CHTC

1. Navigate to `repos/grayscale-CNN`.

2. Transfer `Greyscale_Images` to this directory. If `generate_png_from_tiff.py` has never been run, then run `python3 generate_png_from_tiff.py`.

3. Create the outputs directory by running `mkdir outputs`.

4. Run `condor_submit alexnet.sub`.
