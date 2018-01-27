###################################################
#
#   Script to pre-process the original imgs
#
##################################################


import numpy as np
from PIL import Image
import cv2

from help_functions import *


#My pre processing (use for both training and testing!)
def my_PreProc(data):
    assert(len(data.shape)==4)
    assert (data.shape[1]==3)  #Use the original images
    #black-white conversion
    #train_imgs = rgb2gray(data)
    train_imgs = data
    #my preprocessing:
#     print("train_imgs: ", train_imgs.shape)
    train_imgs = dataset_normalized(train_imgs)
    train_imgs = clahe_equalized(train_imgs)
    train_imgs = adjust_gamma(train_imgs, 1.2)
#     print("AVG value of layer R: ", np.average(train_imgs[0,0,:,:]))
#     print("AVG value of layer G: ", np.average(train_imgs[0,1,:,:]))
#     print("AVG value of layer B: ", np.average(train_imgs[0,2,:,:]))
    train_imgs = train_imgs/255.  #reduce to 0-1 range
    return train_imgs


#============================================================
#========= PRE PROCESSING FUNCTIONS ========================#
#============================================================

#==== histogram equalization
def histo_equalized(imgs):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[1]==1)  #check the channel is 1
    imgs_equalized = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        imgs_equalized[i,0] = cv2.equalizeHist(np.array(imgs[i,0], dtype = np.uint8))
    return imgs_equalized


# CLAHE (Contrast Limited Adaptive Histogram Equalization)
#adaptive histogram equalization is used. In this, image is divided into small blocks called "tiles" (tileSize is 8x8 by default in OpenCV). Then each of these blocks are histogram equalized as usual. So in a small area, histogram would confine to a small region (unless there is noise). If noise is there, it will be amplified. To avoid this, contrast limiting is applied. If any histogram bin is above the specified contrast limit (by default 40 in OpenCV), those pixels are clipped and distributed uniformly to other bins before applying histogram equalization. After equalization, to remove artifacts in tile borders, bilinear interpolation is applied
def clahe_equalized(imgs):
    assert (len(imgs.shape)==4)  #4D arrays
    #assert (imgs.shape[1]==1)  #check the channel is 1
    assert (imgs.shape[1]==3)  #check the channel is 3
    #create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#     print("imgs.shape: ", imgs.shape)
    imgs_equalized = np.empty(imgs.shape)
#     print("imgs_equalized.shape: ", imgs_equalized.shape)
    
    
#     print("AVG value of layer R: ", np.average(imgs[0,0,0,0]))
#     print("AVG value of layer G: ", np.average(imgs[0,1,0,0]))
#     print("AVG value of layer B: ", np.average(imgs[0,2,0,0]))
    
    for i in range(imgs.shape[0]):
        for c in range(imgs.shape[1]):
#             print("shape imgs_equalized[i,c]", imgs_equalized[i,c].shape)
#             print("shape imgs[i,c]", imgs[i,c].shape)
            imgs_equalized[i,c] = clahe.apply(np.array(imgs[i,c,:,:], dtype = np.uint8))
            
#             print("AVG imgs_equalized[i,c]:", np.average(imgs_equalized[i,c]))
#             print("AVG imgs[i,c]:", np.average(imgs[i,c,:,:]))
    return imgs_equalized


# ===== normalize over the dataset
def dataset_normalized(imgs):
    assert (len(imgs.shape)==4)  #4D arrays
    #assert (imgs.shape[1]==1)  #check the channel is 1
    assert (imgs.shape[1]==3)  #check the channel is 3 (RGB)
    imgs_normalized = np.empty(imgs.shape)
    imgs_std = np.std(imgs)
    imgs_mean = np.mean(imgs)
#     print("imgs_std:")
#     print(imgs_std)
#     print("imgs_mean:")
#     print(imgs_mean)
#     print("imgs:")
#     print(imgs.shape)
    imgs_normalized = (imgs-imgs_mean)/imgs_std
#     imgs_normalized = imgs
#     print("Passed!")
    for i in range(imgs.shape[0]):
        imgs_normalized[i] = ((imgs_normalized[i] - np.min(imgs_normalized[i])) / (np.max(imgs_normalized[i])-np.min(imgs_normalized[i])))*255
    return imgs_normalized


def adjust_gamma(imgs, gamma=1.0):
    assert (len(imgs.shape)==4)  #4D arrays
    #assert (imgs.shape[1]==1)  #check the channel is 1
    assert (imgs.shape[1]==3)  #check the channel is 3
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    new_imgs = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        for c in range(imgs.shape[1]):
            new_imgs[i,c] = cv2.LUT(np.array(imgs[i,c,:,:], dtype = np.uint8), table)
    return new_imgs
