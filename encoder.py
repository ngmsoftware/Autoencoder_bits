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
MAX_EPOCH = 2000




HEX = {'0':[0,0,0,0], '1':[1,0,0,0], '2':[0,1,0,0], '3':[1,1,0,0], '4':[0,0,1,0], '5':[1,0,1,0], '6':[0,1,1,0], '7':[1,1,1,0], '8':[0,0,0,1], '9':[1,0,0,1], 'A':[0,1,0,1], 'B':[1,1,0,1], 'C':[0,0,1,1], 'D':[1,0,1,1], 'E':[0,1,1,1], 'F':[1,1,1,1] }


X_train = np.ndarray((0,28,28))
Y_train = np.ndarray((0,4))

files = os.listdir('./digits/imgs2')
for file in files:
    name, ext = file.split('.')
    digit, _ = name.split('-')

    if (ext == 'png'):

        _I = 1.0 - plt.imread('./digits/imgs/'+file).reshape((1,28,28))
    
        X_train = np.concatenate((X_train, _I), 0)
    
        Y_train = np.concatenate((Y_train, np.array(HEX[digit]).reshape(1,4)), 0)




compiled = False


NTrain = X_train.shape[0]

   
modelEncoder = tf.keras.models.Sequential()
modelEncoder.add( tf.keras.layers.Input((28,28)) )  
modelEncoder.add( tf.keras.layers.Flatten() )  
#modelEncoder.add( tf.keras.layers.Dense(50, activation='sigmoid') )
modelEncoder.add( tf.keras.layers.Dense(4, activation='sigmoid') )



# 8. Compile model
modelEncoder.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
 


# 9. Fit model on training data
modelEncoder.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=MAX_EPOCH, verbose=1)


for X, Y in zip(X_train, Y_train):
    yHat = np.round( modelEncoder.predict(X.reshape((1,28,28))) )
    y = Y
    
    print((y-yHat)**2)
    


   






