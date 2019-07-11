# DOCUMENT EDGE DETECTION USING U-Net ARCHITECTURE IN KERAS, to have accurate document scanning

The architecture was inspired by [U-Net: Convolutional Networks for Biomedical Image Segmentation](http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/).

---

## Overview

A small application built on top of U-Net segmentation architecture, that segments documents from images,
and performs perspective transform on it.

Running the document through the model yields a mask of the document.

The mask is the preprocessed, to further isolate the document from the background.

The Canny edge detector is used to detect the edges of the document in the mask, and then the optimum contour is identified.

Perspective transform is used on the original image with the 4 corner points obtained from the contours.

This is an improvement over the previous model, where direct canny is applied without image segmentation. It had very poor accuracy, in terms of detecting the edges of the document.

### Pre-processing

The images are 3-D volume tiff, you should transfer the stacks into images first.

The data for training contains 30 256*256 images.

Create the following directory structure: 
  |-data 
        |- npydata
        |- train
                |- image
                |- label
        |- test

  |- results
  |- static
           |- FinalTransformedDoc

  |- templates
  |- uploads

### Model

This deep neural network is implemented with Keras functional API, which makes it extremely easy to experiment with different interesting architectures.

The output from the network is a 256*256 which represents a mask that should be learned. Sigmoid activation function
makes sure that mask pixels are in \[0, 1\] range.

### Training

The model is trained for 10 epochs.

After 10 epochs, the calculated accuracy is about 0.91.

The loss function for the training is basically just a binary cross-entropy

---

## How to use

### Dependencies

This tutorial depends on the following libraries:

* Tensorflow
* Keras >= 1.0
* libtiff(optional)
* OpenCv
* Numpy
* OS

Also, this code should be compatible with Python versions 2.7-3.6.



### Prepare the data

First transfer 3D volume tiff to 30 256*256 images.

To do so, run ```python compress.py```, providing the right input and output directory.

The labels have to black-n-white masks, 256*256, named serially from 0.


### Define the model

* Check out ```get_unet()``` in ```unet.py``` to modify the model, optimizer and loss function.

### Train the model and generate masks for test images

* Run ```python train.py``` to train the model.

After this script finishes, in ```imgs_mask_test.npy``` masks for corresponding images in ```imgs_test.npy```
should be generated. I suggest you examine these masks for getting further insight into your model's performance.

### Generating masks of documents from the trained model

* Run ```python test.py``` to get the masks of the images

After it's done, the resultant masks are saved in ```results```.

### Generating the final document

* Run ```python document-edge-detect.py```

The final document is saved in ```FinalTransformedDoc```.

If both mask generation and edge detection should happen together, uncomment the first 3 comments under the main function.

## About Keras

Keras is a minimalist, highly modular neural networks library, written in Python and capable of running on top of either TensorFlow or Theano. It was developed with a focus on enabling fast experimentation. Being able to go from idea to result with the least possible delay is key to doing good research.

Use Keras if you need a deep learning library that:

allows for easy and fast prototyping (through total modularity, minimalism, and extensibility).
supports both convolutional networks and recurrent networks, as well as combinations of the two.
supports arbitrary connectivity schemes (including multi-input and multi-output training).
runs seamlessly on CPU and GPU.
Read the documentation [Keras.io](http://keras.io/)

Keras is compatible with: Python 2.7-3.5.