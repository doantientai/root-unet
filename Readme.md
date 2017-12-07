Hello world!
In this directory, 

## Tasks:
### Part 0: Setup the neural network
In this part, I try to make the Unet (along with it's source code) in https://github.com/orobix/retina-unet running on our data of tree's root. There are a few differences in the data-structure, so I need to modify the source code and re-organize the dataset structure so that they can match each other.

Todo list:
1. ~~Modify the readme.md~~
2. ~~PreProcessing: check pairs' names, resize~~
3. ~~Reorganize data images~~
4. ~~Modify the *prepare_datasets_DRIVE.py*~~
5. ~~Run *prepare_datasets_DRIVE.py* to make new data files~~
6. ~~Run *run_training.py*~~
7. ~~Run *run_testing.py*~~

### Part 1: Training the network with the first dataset
The goal of this part is to figure out the set of hyperparametters of the Unet which help it learn most effectively, which means the loss of the predicted output is minimized. As the first collection of data, images are very collective and have good quality. The next datasets will be more challenged.

In this part, I need to train the Unet using different
1. ~~Batch sizes~~
2. ~~Input sizes~~
3. Number of conv layers
4. Conv kernels sizes
5. Hidden layers sizes

The most effective batch size is 64. This value also depends on the computer's configuration.
The best input size is 64x64 as it helps the validation loss to be more converged, comparing to the other input sizes.

![hehe](https://raw.githubusercontent.com/doantientai/root-unet/master/Experiments/input-sizes.png)

Below is what we get so far (Original Image, Groudtruch Image, Predicted Image)
<p align="center">
  <img src="https://raw.githubusercontent.com/doantientai/root-unet/input-rgb/Experiments/root_64/root_64_Original_GroundTruth_Prediction1.png" width="350"/>
</p>

Obviously, the predicted result needs some post-process.

### Part 2: Post-processing predicted results 
Source: https://github.com/doantientai/root-unet/tree/input-rgb/PostProcess
After trying a set of image processing functions, I end up with this series of processes on the output:
- cv2.adaptiveThreshold()
- cv2.bitwise_not()
- cv2.erode()
- cv2.MORPH_CLOSE
- morphology.remove_small_objects
- cv2.MORPH_CLOSE again

Below is what we get so far (hand-draw version, predicted verion, predicted verion after post-processing)
<p align="center">
  <img src="https://raw.githubusercontent.com/doantientai/root-unet/input-rgb/Experiments/root_64_rgb/TestResults/root_64_rgb_1_gtruth.png" width="500"/>
  <img src="https://raw.githubusercontent.com/doantientai/root-unet/input-rgb/Experiments/root_64_rgb/TestResults/root_64_rgb_1_predict.png" width="500"/>
  <img src="https://raw.githubusercontent.com/doantientai/root-unet/input-rgb/PostProcess/im_dn_inv.png" width="500"/>
</p>

![alt text](http://i0.kym-cdn.com/photos/images/newsfeed/000/531/557/a88.jpg "We need to go deeper")

