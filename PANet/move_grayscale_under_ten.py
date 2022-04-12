import os
import shutil

# IMPORTANT: first, ensure connection to Broad Institute IP group VPN!

print('entered move_grayscale_under_ten.py')

def move_grayscale_under_ten(source_dir, target_dir):

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

move_grayscale_under_ten(source_dir='./Greyscale_Images/', target_dir='./Greyscale_Images_under_10/')
move_grayscale_under_ten(source_dir='./PbMgNr/', target_dir='./PbMgNr_under_10/')
