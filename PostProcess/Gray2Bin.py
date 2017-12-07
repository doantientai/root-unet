
# coding: utf-8

# In[43]:


input_path = '/home/doantientai/Projects/DeepRoot/root-unet/Experiments/root_64_rgb/TestResults/root_64_rgb_1_predict.png'

from PIL import Image
from matplotlib.pyplot import imshow, imsave
import numpy as np
get_ipython().magic('matplotlib inline')
import scipy.ndimage
import cv3


# In[37]:


im = Image.open(input_path)
# imshow(np.asarray(im),cmap="gray")
# imsave(im, "img_gray.png")
im.save("im_gray.png")


# In[38]:


im_array = np.asarray(im)
print(im_array.shape)
# THRESHOLD_VALUE = 225

# im_array_bin = (im_array > THRESHOLD_VALUE) * 255.0

# scipy.ndimage.morphology.binary_fill_holes(im_array_bin)

# im_bin = Image.fromarray(im_array_bin).convert('L')
# # im_bin
# # im_array_bin.shape


th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,            cv2.THRESH_BINARY,11,2)


im_bin.save("img_bin.png")
# imshow(im_bin)
# im_bin
im_bin


# In[32]:


10%2

