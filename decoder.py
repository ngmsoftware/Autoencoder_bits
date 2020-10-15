#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 13:03:16 2018

@author: ninguem
"""

import sys
import os

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.executing_eagerly()

BATCH_SIZE = 16
MAX_EPOCH = 10000




HEX = {'0':[0,0,0,0], '1':[1,0,0,0], '2':[0,1,0,0], '3':[1,1,0,0], '4':[0,0,1,0], '5':[1,0,1,0], '6':[0,1,1,0], '7':[1,1,1,0], '8':[0,0,0,1], '9':[1,0,0,1], 'A':[0,1,0,1], 'B':[1,1,0,1], 'C':[0,0,1,1], 'D':[1,0,1,1], 'E':[0,1,1,1], 'F':[1,1,1,1] }


X_train = np.ndarray((0,28,28,1))
Y_train = np.ndarray((0,4))

files = os.listdir('./digits/imgs2')
for file in files:
    name, ext = file.split('.')
    digit, _ = name.split('-')

    if (ext == 'png'):

        _I = 1.0 - plt.imread('./digits/imgs2/'+file).reshape((1,28,28,1))
    
        X_train = np.concatenate((X_train, _I), 0)
    
        Y_train = np.concatenate((Y_train, np.array(HEX[digit]).reshape(1,4)), 0)




compiled = False


NTrain = X_train.shape[0]

   
modelDecoder = tf.keras.models.Sequential()
modelDecoder.add( tf.keras.layers.Input(4))  
modelDecoder.add(tf.keras.layers.Dense(28*28*1, activation='sigmoid'))
modelDecoder.add(tf.keras.layers.Reshape((28, 28, 1)))
modelDecoder.add( tf.keras.layers.Conv2D(16, kernel_size=(8,8), activation='relu', padding='same') ) 
modelDecoder.add( tf.keras.layers.Conv2D(1, kernel_size=(1,1), activation='sigmoid', padding='same') ) 



# 8. Compile model
modelDecoder.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
 


# 9. Fit model on training data
modelDecoder.fit(Y_train, X_train, batch_size=BATCH_SIZE, epochs=MAX_EPOCH, verbose=1)




for digitHex, digitsBin in zip(HEX.keys(), HEX.values()):
    
   plt.figure()
    
   features = np.array(digitsBin).reshape((1,4));
    
   IHat = modelDecoder.predict(features).reshape((28,28))

   plt.imshow(IHat)

   # plt.subplot(2,2,1)
   # plt.imshow(I)
   # plt.subplot(2,2,2)
   # plt.imshow(IHat);
   # plt.subplot(2,1,2)
   # plt.imshow(features);
    
   plt.pause(0.01)        
    
   






