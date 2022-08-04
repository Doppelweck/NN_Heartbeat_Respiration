#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 11:29:01 2020

@author: sebastianfriedrich
"""

#simple expample: approximation of sin+rand by RNN
from matplotlib import pyplot as plt
import numpy as np
 
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
#from tensorflow.data import Dataset
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import optimizers


#generate data
v1=np.sin(np.linspace(0,4000,4000*5))+0.4*np.sin(4*np.linspace(0,4000,4000*5))
v1=v1+0.1*np.random.randn(len(v1))
v2=v1
 
#init parameter
train_split=int(len(v1)/2) #how to split training and validation (no test data here)
samp_size=20 #length of a sequence the RNN is supposed to process
 
batch_size = 300
epochs=8
buffer_size = 100
 
#Helper function reshape data
def univariate_data(dataset, target_data, start_index, end_index, history_size, target_size):
  data = []
  labels = []
 
  start_index = start_index + history_size
  if end_index is None:
    end_index = len(dataset) - target_size
 
  for i in range(start_index, end_index):
    indices = range(i-history_size, i)
    # Reshape data from (history_size,) to (history_size, 1)
    data.append(np.reshape(dataset[indices], (history_size, 1)))
    labels.append(target_data[i+target_size])
  return np.array(data), np.array(labels)
 
 
x_train, y_train= univariate_data(v1,v2, 0, train_split,
                                           samp_size,
                                           5)
x_val, y_val= univariate_data(v1,v2, train_split,None,
                                       samp_size,
                                       5)
 
print(x_train.shape)
print(y_train.shape)
print ('First signal')
print (x_train[0])
print ('\n First Value to predict')
print (y_train[0])
 
 
 
#use keras tool to prepare dataset for training and validation
train_univariate = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_univariate = train_univariate.cache().shuffle(buffer_size).batch(batch_size).repeat()
 
val_univariate = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_univariate = val_univariate.batch(batch_size).repeat()
 
 
 
#build LSTM model by means of keras
 
model = Sequential()
 
 
#model.add(layers.GRU(20,activation='tanh',input_shape=x_train.shape[-2:])) #A_1 in upper diagram
model.add(layers.LSTM(6,activation='tanh',input_shape=x_train.shape[-2:])) #A_1 in upper diagram
model.add(layers.Dense(1,activation='tanh')) #A_2 in the upper diagram
 
# setup optimizer and define loss
opt=optimizers.SGD(lr=0.001)
model.compile(loss='mae', optimizer=opt)
 
model.summary()
 
model.fit(train_univariate, epochs=epochs,
                      steps_per_epoch=500,
                      validation_data=val_univariate, validation_steps=50)
 
 
 
 
#plot prediction v.s true values
pred_y=model.predict(x_val)
 
plt.figure()
plt.plot(y_val[0:600],label="true")
plt.plot(pred_y[0:600],label="pred")
plt.legend()
plt.show()