import numpy as np
import PIL.ImageOps

from PIL import Image

print('entered chth-test.py')

im = Image.open('../../../midbody/ENSG00000134152_U-2-OS_1345_E8_1_blue_red_green.jpg')
im_array = np.array(im)
print('\nim_array')
print(im_array)

im_inverted = PIL.ImageOps.invert(im)
im_inverted_array = np.array(im_inverted)
print('\nim_inverted_array')
print(im_array_inverted)

im_inverted.save('chtc-test-output.jpg')
print('\nsaved inverted image to chtc-test-output.jpg')

print('\ndone :)')
