# input_path = '/home/doantientai/Projects/DeepRoot/root-unet/Experiments/root_64_rgb/TestResults/root_64_rgb_1_predict.png'

#input_path = '/home/doantientai/Projects/DeepRoot/root-unet/Experiments/root_64_rgb_crop256_flrt/TestResults/root_64_rgb_crop256_flrt_0_predict.png'

input_path = '/home/doantientai/Projects/DeepRoot/root-unet/Experiments/root_db1_bin_pat32_cr15p_flrt_500e/TestResults/root_db1_bin_pat32_cr15p_flrt_500e_0_origin.png'

gt_path = input_path.replace("_predict", "_gtruth")

from PIL import Image
from matplotlib.pyplot import imshow, imsave
import numpy as np
# %matplotlib inline
import scipy.ndimage


# import os
# os.sys.path

# import sys
# sys.path.append('/home/doantientai/anaconda3/envs/keras1.1/lib/python2.7/site-packages')
import cv2

from skimage import morphology

# input: grayscale predicted image
# convert to binary, invert
# remove small objects

# im = Image.open(input_path)
im = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
# imshow(np.asarray(im),cmap="gray")
# imsave(im, "img_gray.png")
cv2.imwrite("im_pred.png", im)

im_gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
cv2.imwrite("im_gt.png", im_gt)

im_bin = cv2.adaptiveThreshold(im,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,31,2)
# cv2.imwrite("im_bin.png", im_bin)

im_inv = cv2.bitwise_not(im_bin)
# cv2.imwrite("im_inv.png", im_inv)

kernel_erode = np.ones((2,2),np.uint8)
im_ero = cv2.erode(im_inv,kernel_erode,iterations = 1)
# cv2.imwrite('im_ero.png', im_ero)

kernel_close = np.ones((5,5),np.uint8)
im_close = closing = cv2.morphologyEx(im_ero, cv2.MORPH_CLOSE, kernel_close)
# cv2.imwrite('im_close.png', im_close)

im_dn = morphology.remove_small_objects(im_close.astype('bool'),32)
im_dn = np.array(im_dn, dtype=np.uint8)

kernel_close = np.ones((5,5),np.uint8)
im_dn = closing = cv2.morphologyEx(im_dn, cv2.MORPH_CLOSE, kernel_close)
# cv2.imwrite('im_close.png', im_close)


im_dn*=255

# cv2.imwrite('im_dn.png', im_dn)

im_dn_inv = cv2.bitwise_not(im_dn)
cv2.imwrite("im_dn_inv.png", im_dn_inv)

# print np.amax(im_dn.astype('uint8'))
# imshow(im_dn,cmap='binary')