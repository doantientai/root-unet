###################################################
#
#   Script to:
#   - Load the images and extract the patches
#   - Define the neural network
#   - define the training
#
##################################################


import numpy as np
import ConfigParser

from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Reshape, core, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
#from keras.utils.visualize_util import plot
from keras.utils.vis_utils import plot_model
from keras.optimizers import SGD

import keras.backend as K
EPSILON = 1e-7

from theano import tensor as T

import sys
sys.path.insert(0, './lib/')
from help_functions import *

#function to obtain data for training/testing (validation)
from extract_patches import get_data_training


#========= Load settings from Config file
config = ConfigParser.RawConfigParser()
config.read('configuration.txt')
#patch to the datasets
path_data = config.get('data paths', 'path_local')
#Experiment name
name_experiment = config.get('experiment name', 'name')
result_dir = "Experiments/" + name_experiment
#training settings
N_epochs = int(config.get('training settings', 'N_epochs'))
batch_size = int(config.get('training settings', 'batch_size'))
learning_rate_ini = float(config.get('training settings', 'learning_rate_ini'))


#========= Tai's custom loss-function
def weighted_categorical_crossentropy(weights):
    """
    Source:
    https://gist.github.com/wassname/ce364fddfc8a025bfab4348cf5de852d
    
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    
    weights = K.variable(weights)
        
    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    
    return loss


def tai_loss(weight):   
    w = K.variable(weight)
    
    def loss(y_true, y_pred):
        new_loss = ((y_true*w) + (1-y_true)*(1-w))*K.abs(y_true - y_pred)
        return new_loss
    return loss

# def Tai_loss_01(y_true, y_pred):
# #     new_loss = K.categorical_crossentropy(y_true, y_pred)
#     w = 0.5
#     new_loss = ((y_true*w) + (1-y_true)*(1-w))*K.abs(y_true - y_pred)
#     return new_loss

# def Tai_loss(target, output):
# #     if from_logits:
# #         output = T.nnet.softmax(output)
# #     else:
# #         # scale preds so that the class probas of each sample sum to 1
# #         output /= output.sum(axis=-1, keepdims=True)
#     # avoid numerical instability with _EPSILON clipping
#     output = T.clip(output, EPSILON, 1.0 - EPSILON)    
#     new_loss = T.nnet.categorical_crossentropy(output, target)
    
#     return new_loss


#Define the neural network
def get_unet(n_ch,patch_height,patch_width):
    inputs = Input(shape=(n_ch,patch_height,patch_width))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    #
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    #
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_first')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv3)

    up1 = UpSampling2D(size=(2, 2))(conv3)
    up1 = concatenate([conv2,up1],axis=1)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(up1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv4)
    #
    up2 = UpSampling2D(size=(2, 2))(conv4)
    up2 = concatenate([conv1,up2], axis=1)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv5)
    #
    conv6 = Conv2D(2, (1, 1), activation='relu',padding='same',data_format='channels_first')(conv5)
    conv6 = core.Reshape((2,patch_height*patch_width))(conv6)
    conv6 = core.Permute((2,1))(conv6)
    ############
    conv7 = core.Activation('softmax')(conv6)

    model = Model(input=inputs, output=conv7)

    sgd = SGD(lr=learning_rate_ini, decay=1e-6, momentum=0.3, nesterov=False)
    model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'])
    
#     model.compile(optimizer='sgd', loss=tai_loss(0.75),metrics=['accuracy'])

        
#     weights = np.array([1 1])
#     loss = weighted_categorical_crossentropy(weights)
#     model.compile(loss=loss,optimizer='sgd')
    
    return model

#Define the neural network gnet
#you need change function call "get_unet" to "get_gnet" in line 166 before use this network
# def get_gnet(n_ch,patch_height,patch_width):
#     inputs = Input((n_ch, patch_height, patch_width))
#     conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
#     conv1 = Dropout(0.2)(conv1)
#     conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
#     up1 = UpSampling2D(size=(2, 2))(conv1)
#     #
#     conv2 = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(up1)
#     conv2 = Dropout(0.2)(conv2)
#     conv2 = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(conv2)
#     pool1 = MaxPooling2D(pool_size=(2, 2))(conv2)
#     #
#     conv3 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(pool1)
#     conv3 = Dropout(0.2)(conv3)
#     conv3 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv3)
#     pool2 = MaxPooling2D(pool_size=(2, 2))(conv3)
#     #
#     conv4 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool2)
#     conv4 = Dropout(0.2)(conv4)
#     conv4 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv4)
#     pool3 = MaxPooling2D(pool_size=(2, 2))(conv4)
#     #
#     conv5 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool3)
#     conv5 = Dropout(0.2)(conv5)
#     conv5 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv5)
#     #
#     up2 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
#     conv6 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up2)
#     conv6 = Dropout(0.2)(conv6)
#     conv6 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv6)
#     #
#     up3 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
#     conv7 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up3)
#     conv7 = Dropout(0.2)(conv7)
#     conv7 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv7)
#     #
#     up4 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
#     conv8 = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(up4)
#     conv8 = Dropout(0.2)(conv8)
#     conv8 = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(conv8)
#     #
#     pool4 = MaxPooling2D(pool_size=(2, 2))(conv8)
#     conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(pool4)
#     conv9 = Dropout(0.2)(conv9)
#     conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv9)
#     #
#     conv10 = Convolution2D(2, 1, 1, activation='relu', border_mode='same')(conv9)
#     conv10 = core.Reshape((2,patch_height*patch_width))(conv10)
#     conv10 = core.Permute((2,1))(conv10)
#     ############
#     conv10 = core.Activation('softmax')(conv10)

#     model = Model(input=inputs, output=conv10)

#     # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
# #     model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'])
    
#     weights = np.array([0.25,1]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
#     loss = weighted_categorical_crossentropy(weights)
#     model.compile(loss=loss,optimizer='adam')

    
#     return model



#============ Load the data and divided in patches
patches_imgs_train, patches_masks_train = get_data_training(
    DRIVE_train_imgs_original = path_data + config.get('data paths', 'train_imgs_original'),
    DRIVE_train_groudTruth = path_data + config.get('data paths', 'train_groundTruth'),  #masks
    patch_height = int(config.get('data attributes', 'patch_height')),
    patch_width = int(config.get('data attributes', 'patch_width')),
    N_subimgs = int(config.get('training settings', 'N_subimgs')),
    inside_FOV = False
)


#========= Save a sample of what you're feeding to the neural network ==========
N_sample = min(patches_imgs_train.shape[0],40)
visualize(group_images(patches_imgs_train[0:N_sample,:,:,:],5),result_dir+'/'+"sample_input_imgs")#.show()
visualize(group_images(patches_masks_train[0:N_sample,:,:,:],5),result_dir+'/'+"sample_input_masks")#.show()


#=========== Construct and save the model arcitecture =====
n_ch = patches_imgs_train.shape[1]
patch_height = patches_imgs_train.shape[2]
patch_width = patches_imgs_train.shape[3]
model = get_unet(n_ch, patch_height, patch_width)  #the U-net model
print "Check: final output of the network:"
print model.output_shape
plot_model(model, to_file=result_dir+'/'+name_experiment + '_model.png')   #check how the model looks like
json_string = model.to_json()
open(result_dir+'/'+name_experiment +'_architecture.json', 'w').write(json_string)



#============  Training ==================================
checkpointer = ModelCheckpoint(filepath=result_dir+'/'+name_experiment +'_best_weights.h5', verbose=1, monitor='val_loss', mode='auto', save_best_only=True) #save at each epoch if the validation decreased


# def step_decay(epoch):
#     lrate = 0.01 #the initial learning rate (by default in keras)
#     if epoch==100:
#         return 0.005
#     else:
#         return lrate
#
# lrate_drop = LearningRateScheduler(step_decay)

patches_masks_train = masks_Unet(patches_masks_train)  #reduce memory consumption
model.fit(patches_imgs_train, patches_masks_train, nb_epoch=N_epochs, batch_size=batch_size, verbose=2, shuffle=True, validation_split=0.1, callbacks=[checkpointer])


#========== Save and test the last model ===================
model.save_weights(result_dir+'/'+name_experiment +'_last_weights.h5', overwrite=True)
#test the model
# score = model.evaluate(patches_imgs_test, masks_Unet(patches_masks_test), verbose=0)
# print('Test score:', score[0])
# print('Test accuracy:', score[1])


















#
