# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 22:35:42 2019

@author: Kenil Shah
"""


import keras 

# Building Convolution Neural Network

from keras.model import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialize Convolution Neural Network

model = Sequential()

# Adding Convolution Layer - Get the Feature Map. Input Image X Feature Detector = Feature Map

model.add(Convolution2D(nb_filter,nb_row,nb_col,border_mode = 'same',input_shape = (3,256,256),activation = 'relu'))

""" nb_filter is the number of feature detectors. Start with 32~64~128~256
    nb_row is the number of rows in feature detector matrix Start with 3
    nb_col is the number of column in feature detector matrix Start with 3 
    border_mode how the feature detector handle border value
    input_shape tells us about what is the expected format of the images. Color image is a 3d image
    3~colors(red,blue,green),256~pixels,256~pixels 
    (3,256,256) is for Theano (256,256,3) is for Tensorflow 
    activation is the activation function for layers. Here we are using Rectifier Function """

    
# Pooling - Max Pool. Will be applied on each of the feature map. And will reduce the size of the feature maps
    
model.add(MaxPooling2D(poolsize = (2,2))    

#Second Convolution Layer 

model.add(Convolution2D(nb_filter,nb_row,nb_col,border_mode = 'same',activation = 'relu'))
model.add(MaxPooling2D(poolsize = (2,2))    

""" poolsize is the size of the pool we use to extreact features from the feature map """   
    
#Flattening the pooled map

model.add(Flatten()) # No paramters. Keras understood by itself it needs to flatten the previous layer

# We nov have a one dimension vector of the image

# Full Connection. We will create a hidden layer just like ANN

model.add(Dense(output_dim = 128,activation = 'relu'))
model.add(Dense(output_dim = 1,activation = 'sigmoid'))


""" output_dim is to set the number of nodes in the hidden layer
    activation is the activation function for the hidden layer. relu is for rectifier function. sigmoid is for sigmoid function 
"""    

# Compiling CNN

model.compiler(optimizer = 'adam',loss = 'binary_crossentropy',metrics = ['accuracy'])

""" optimizer is the algorithm we want to use to find the weights . Here we are using 'adam' for Stocastic Gradient Descent
loss is the loss function with the adam algorithm  which we need to optimize the Gradient Descent to find the optimal weights 
binary_crossentropy is the function for variable with two levels.
categorical_crossentropy for variables with more than two levels.
metrics  is the criterion we use to evaluate our model """

# Fitting the CNN on the dataset

# Image Augmentation

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,   # Rescaling pixel values
        shear_range=0.2,  # Geometrical Transformation
        zoom_range=0.2,   # Zoom our image
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train = train_datagen.flow_from_directory(
                                        'Dataset Path',          # Directory Name
                                        target_size=(256, 256),  # Size of the image expected in our mode
                                        batch_size=32,           # Size of the batch after which batch will be updated
                                        class_mode='binary')     # Show the class mode of the result

test = test_datagen.flow_from_directory(
                                        'Dataset Path',         # Directory Name
                                        target_size=(256, 256), # Image of our test set
                                        batch_size=32,          # Size of the batch after which batch will be updated
                                        class_mode='binary')    # Show the class mode of the result

model.fit_generator(
                    train,
                    steps_per_epoch=2000,  # Number of images in our train dataset
                    epochs=25,             # Number of epoch we need
                    validation_data= test, # Test set 
                    validation_steps=800)  # Number of images in our test dataset
                    

""" To improve the model you can
Add a convolution Layer
Increase the target size in train and test dataset so that more information of the pixel pattern is captured 
"""


                    

