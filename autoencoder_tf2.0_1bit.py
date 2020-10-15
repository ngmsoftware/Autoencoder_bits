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
import math
import matplotlib as mpl

norm = mpl.colors.Normalize(vmin=0.,vmax=1.)

tf.executing_eagerly()

BATCH_SIZE = 8
MAX_EPOCH = 80000



lossFnc = tf.keras.losses.mean_squared_error

optimizer = optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.03,  momentum=0.1)


folderName = './digits/imgs2/'


X_train = np.ndarray((0,28,28,1))

files = os.listdir(folderName)
for file in files:
    name, ext = file.split('.')
    digit, _ = name.split('-')

    if digit == '0' or digit == '1' or digit == '2' or digit == '3' or digit == '4' or digit == '5' or digit == '6' or digit == '7':

        if (ext == 'png'):
    
            _I = 1.0 - plt.imread(folderName+file).reshape((1,28,28,1))
        
            X_train = np.concatenate((X_train, _I), 0)
    


NTrain = X_train.shape[0]

initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=2.0)

# 7. build the model
modelEncoder = tf.keras.models.Sequential()
modelEncoder.add( tf.keras.layers.Input(shape=(28, 28, 1)) ) 
modelEncoder.add(tf.keras.layers.Flatten())
modelEncoder.add(tf.keras.layers.Dense(10, activation='sigmoid', kernel_initializer = initializer ) ) 
modelEncoder.add(tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer = initializer ) )


# FEATURES!!!


modelDecoder = tf.keras.models.Sequential()
modelDecoder.add( tf.keras.layers.Input(1))  
modelDecoder.add(tf.keras.layers.Dense(28*28*1, activation='linear'))
modelDecoder.add(tf.keras.layers.Reshape((28, 28, 1)))
modelDecoder.add( tf.keras.layers.Conv2D(40, kernel_size=(3,3), activation='linear', padding='same') ) 
modelDecoder.add(tf.keras.layers.LeakyReLU())
modelDecoder.add( tf.keras.layers.Conv2D(1, kernel_size=(1,1), activation='sigmoid', padding='same') ) 




# load last 
modelDecoder = tf.keras.models.load_model('decoder_1bit')
modelEncoder = tf.keras.models.load_model('encoder_1bit')




factor = np.exp(np.log(50)/MAX_EPOCH)
expo = 0.5   

_loss = []

for epoch in range(MAX_EPOCH):
    print('epoch : %d of %d'%(epoch, MAX_EPOCH))

    p = np.random.permutation(NTrain)

    X_train = X_train[p]

    X_train_dataset = tf.data.Dataset.from_tensor_slices(X_train).batch(batch_size=BATCH_SIZE)
    
    i = 0    
    
    expo = expo*factor
    expo = 1
    
    for X in X_train_dataset:

        with tf.GradientTape(persistent=True) as tape:
            
            features = modelEncoder(X)
            XHat = modelDecoder(features)
            loss = tf.reduce_mean(lossFnc(X, XHat))
                
           

        gradsEncoder = tape.gradient(loss, modelEncoder.trainable_variables)
        gradsDecoder = tape.gradient(loss, modelDecoder.trainable_variables)

        optimizer.apply_gradients(zip(gradsEncoder, modelEncoder.trainable_variables))
        optimizer.apply_gradients(zip(gradsDecoder, modelDecoder.trainable_variables))


        _loss.append(loss)

        print('batch : %d of %d (loss = %.4f)'%(i,int(NTrain/BATCH_SIZE),loss))
    
        i = i + 1
        
        # if len(_loss)%100 == 0:
        #     plt.cla()
        #     plt.plot(_loss)
        #     plt.plot([0, len(_loss)],[0.001, 0.001],color='k')
        #     plt.axis([0, len(_loss), 0.0, 0.002])
        #     plt.pause(0.000001)


plt.close('all')
i = 1
for X in X_train:
    features = modelEncoder.predict(X.reshape((1,28,28,1)))

    print(features)

    IHat = modelDecoder.predict(features).reshape((28,28))

    plt.subplot(2,16,2*(i-1)+1);
    plt.imshow(X.reshape((28,28)))
    plt.axis('off')
    plt.subplot(2,16,2*(i-1)+2);
    plt.imshow(IHat)
    plt.axis('off')
    plt.subplot(2,8,8+i);
    plt.imshow(features, norm = norm)
    plt.axis('off')
    
    plt.pause(0.0001)

    i = i+1



for i in range(100):
    plt.cla()
    plt.imshow(modelDecoder.predict(np.array([i/100.0])).reshape((28,28,1)))
    plt.title('%.2f'%(i/100))
    plt.pause(0.1)
    
    
    