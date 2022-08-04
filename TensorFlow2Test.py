#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 11:16:33 2020

@author: sebastianfriedrich
"""

print('Start App') 
# =============================================================================
#    Import Files, Folders and Modules
# =============================================================================
print('Load Modules') 
from matplotlib import pyplot as plt
import numpy as np
import os
import sys
import scipy
 
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
#from tensorflow.data import Dataset
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import optimizers



#set working directory
os.chdir('/Users/sebastianfriedrich/Documents/Hochschule Trier/Module/Masterprojekt (LAROS)/Python/')
#os.chdir('/Users/sebastianfriedrich/Desktop/MA LaROS/Python/')

#import Folders and Files
sys.path.insert(0, '/Users/sebastianfriedrich/Documents/Hochschule Trier/Module/Masterprojekt (LAROS)/Python/Python Files')
#sys.path.insert(0, '/Users/sebastianfriedrich/Desktop/MA LaROS/Python/Python Files')
import dataFunctionsTF2
import modelFunctions

SaveSessionLog = True;
OnlyForTesting = True; #Only Load 5 Data Files

print('Open Matlab Files') 
try:
    TrainingDataMatrixes
    NoOfTrainingDataSets
except:
    dataDir = "/Users/sebastianfriedrich/Documents/Hochschule Trier/Module/Masterprojekt (LAROS)/TrainingData/"
    #dataDir = "/Users/sebastianfriedrich/Desktop/MA LaROS/TrainingData/"
    TrainingDataMatrixes,NoOfTrainingDataSets = dataFunctionsTF2.openTraingsDataFiles(dataDir,OnlyForTesting);
    
# =============================================================================
#    Prepare Data
# =============================================================================
print('Prepare Data') 
NoInputCells = 150;     #No of Samples used as Input for the Network
NoOutputCells = 150;
batch_size = 40
epochs=1
buffer_size = 200

NoOfTrainingSets = int(NoOfTrainingDataSets*0.6)
NoOfValidationSets = NoOfTrainingDataSets-NoOfTrainingSets

print('Get Traing Data') 
train_P, train_S, train_H, train_R,  NoTrainSamples = dataFunctionsTF2.getAndStackDataFromFiles(TrainingDataMatrixes,NoInputCells,NoOutputCells,NoOfTrainingSets,0);
print('Get Validation Data') 
val_P, val_S, val_H, val_R,  NoValSamples = dataFunctionsTF2.getAndStackDataFromFiles(TrainingDataMatrixes,NoInputCells,NoOutputCells,NoOfTrainingDataSets,NoOfTrainingSets);

#use keras tool to prepare dataset for training and validation
train_univariate = tf.data.Dataset.from_tensor_slices((train_S, train_H))
# train_univariate = train_univariate.cache().shuffle(buffer_size).batch(batch_size).repeat()
train_univariate = train_univariate.cache().batch(batch_size)
 
val_univariate = tf.data.Dataset.from_tensor_slices((val_S, val_H))
val_univariate = val_univariate.batch(batch_size)
 

# =============================================================================
#    Create Model
# =============================================================================
print('Create Model') 

model = Sequential()
 
 
#model.add(layers.GRU(20,activation='tanh',input_shape=x_train.shape[-2:])) #A_1 in upper diagram
#model.add(layers.LSTM(50,activation='tanh',input_shape=train_S.shape[-2:])) #A_1 in upper diagram
model.add(layers.Conv1D(filters=10, kernel_size=30, activation='tanh',padding='same',input_shape=train_S.shape[-2:]))
#model.add(layers.LSTM(50,activation='tanh',recurrent_dropout=0.1,return_sequences=True)) 
#model.add(layers.Dropout(0.5))
#model.add(layers.LSTM(50,activation='tanh',recurrent_dropout=0.1,return_sequences=True)) 
#model.add(layers.Dropout(0.5))
#model.add(layers.MaxPooling1D(pool_size=3))
model.add(layers.Flatten())
model.add(layers.Dense(150,activation='tanh')) #A_2 in the upper diagram
#A_1 in upper diagram
 
# setup optimizer and define loss
opt=optimizers.SGD(lr=0.001)
model.compile(loss='mae', optimizer=opt)
 
model.summary()
 
model.fit(train_univariate, epochs=epochs,
                      validation_data=val_univariate)



#plot prediction v.s true values
pred_y=model.predict(val_S)
 
plt.figure()
plt.plot(val_H[600,:],label="true")
plt.plot(pred_y[600,:],label="pred")
plt.legend()
plt.show()



#Signal_R, Signal_H, Signal_S, Signal_P, NoOfSamples = dataFunctionsTF2.extractDataset(TrainingDataMatrixes,NoInputCells,NoOutputCells);






# x_train = X_Phase_NN[0:int(len(X_Phase_NN)/2)];

# x_val = X_Phase_NN[int(len(X_Phase_NN)/2):int(len(X_Phase_NN))];

# y_train = y_LabelR[0:int(len(y_LabelR)/2)];

# y_val = y_LabelR[int(len(y_LabelR)/2):int(len(y_LabelR))];


# #use keras tool to prepare dataset for training and validation
# train_univariate = tf.data.Dataset.from_tensor_slices((x_train, y_train))
# train_univariate = train_univariate.cache().shuffle(buffer_size).batch(batch_size).repeat()
 
# val_univariate = tf.data.Dataset.from_tensor_slices((x_val, y_val))
# val_univariate = val_univariate.batch(batch_size).repeat()



# model = Sequential()
 
 
# #model.add(layers.GRU(20,activation='tanh',input_shape=x_train.shape[-2:])) #A_1 in upper diagram
# model.add(layers.LSTM(6,activation='tanh',input_shape=X_Phase_NN.shape[-2:])) #A_1 in upper diagram
# model.add(layers.Dense(NoOfOutputSamples/2,activation='tanh')) #A_2 in the upper diagram
 
# # setup optimizer and define loss
# opt=optimizers.SGD(lr=0.001)
# model.compile(loss='mae', optimizer=opt)
 
# model.summary()
 
# model.fit(train_univariate, epochs=epochs,
#                       steps_per_epoch=500,
#                       validation_data=val_univariate, validation_steps=50)



# #plot prediction v.s true values
# pred_y=model.predict(x_val)
 
# plt.figure()
# plt.plot(y_val[0:600],label="true")
# plt.plot(pred_y[0:600],label="pred")
# plt.legend()
# plt.show()




























    