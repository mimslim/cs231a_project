## CS231A project -- (Lim/Mims) Extreme Low Light Target Recognition and Tracking

### Introduction
The Google Colaboratory notebooks within are intended to provide a way to generate synthetic low-light images with bounding box information, so that a network can be trained to detect boxes in extremely low light.  The synthetic images are of 3D objects which are of random size and aspect ratio, and rotated and translated before transformation to 2D.  After the 2D transformation, temporal and fixed pattern noise is added, which includes signal shot noise, electrical read noise, column FPN, and hot pixels.

### This repository contains five directories.


- __Colaboratory notebooks__ - One notebook is for generating synthetic images.   The other is for training and detection

- __Python_code__ - Holds various modified versions of Yolov3 code modules, along with utilities to prepare data and configuration files for use with Yolo.

- __data__ - Training and validation datasets.

- __output_images__ - Saved images from earlier runs.

- __saved_weights__ - Saved weights from earlier runs.


### Setting up to run training or prediction

The code and utilities will be expecting a file structure like this on Google Drive:
My Drive
   yolov3
     checkpoints
     test
        output
        samples
           bbox
   CS231A_project
      Data
         test_3D_yolov3
         train_3D_yolov3


To use the model for training or prediction:
- Create a directory structure that includes the above
- download the Colaboratory notebook for training and utilize.
- for _training_, download the Python code, and data files.
- for _detection_, download the Python code and the saved weights
- to _generate_ more clean/noisy images, download the training notebook (syn_create_3D_yolov3_training.ipynb)


```
### Detection Outputs
There will be images created during detection which include bounding boxes.  These will be placed in the yolov3/test/samples/bbox directory:
