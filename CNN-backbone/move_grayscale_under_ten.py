import os
import shutil

# IMPORTANT: first, ensure connection to Broad Institute IP group VPN!

source_dir = '/Volumes/imaging_analysis/2021_10_06_HumanProteinAtlas_CiminiLab_TeresaGao/Greyscale_Images/'
target_dir = '/Volumes/imaging_analysis/2021_10_06_HumanProteinAtlas_CiminiLab_TeresaGao/Greyscale_Images_under_10/'

if not os.path.exists(target_dir):
    os.mkdir(target_dir)

for child in os.listdir(source_dir):
    print('\nchild:', child)
    if os.path.isdir(os.path.join(source_dir, child)):
        print('child is dir')
        subdirectory = child
        file_names = os.listdir(os.path.join(source_dir, subdirectory));

        # move directories with less than 10 image files out of Greyscale_Images source directory
        if len(file_names) < 10:
            print('contents less than 10')
            shutil.move(os.path.join(source_dir, subdirectory), os.path.join(target_dir, subdirectory))
