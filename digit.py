# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 17:38:21 2019

@author: Dhruv Gangwani
"""
#Import Libraries
from keras.datasets import mnist
import numpy as np
from keras.utils import np_utils


#importing dataset
(x_train, y_train), (x_test,y_test) = mnist.load_data()
#Reshaping inorder to make it acceptable by keras layers
x_train = x_train.reshape(60000,28,28,1)

x_test = x_test.reshape(10000,28,28,1)
#Converting to float
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
#Normalizing the data (0 to 1)
x_train= x_train/255
x_test = x_test/255

#Implementing One Hot Encoding
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

img_rows = x_train[0].shape[0]
img_cols = x_train[1].shape[0]
input_shape = (img_rows, img_cols,1)
num_classes = 10
num_pixels = 784

#import libraries
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.optimizers import SGD

model = Sequential()
#Convolutional layer with 32 filters each of size 3*3 and activation function as RELU
model.add(Conv2D(32, 3, activation="relu", input_shape = input_shape ))
#Convolutional layer with 64 filters each of 3*3 size

model.add(Conv2D(64,3,activation="relu"))
#Applying Maxpooling with filter of size 2*2
model.add(MaxPooling2D(2))
model.add(Dropout(0.25))
#Flattening to create a vector of inputs
model.add(Flatten())
#Hidden layer with 128 nodes and RELU as activation function
model.add(Dense(128,activation="relu"))
model.add(Dropout(0.5))
#Output layer with softmax as activation function
model.add(Dense(num_classes, activation="softmax"))
#Compiling model
model.compile(loss="categorical_crossentropy", optimizer=SGD(lr=0.01), metrics=["accuracy"])
print(model.summary())
batch_size=32
epoch=5
#fitting training set to model
model.fit(x_train, y_train ,batch_size= batch_size, epochs = epoch,  validation_data=(x_test,y_test))
#predicting model accuracy on testing set
score = model.evaluate(x_test,y_test )