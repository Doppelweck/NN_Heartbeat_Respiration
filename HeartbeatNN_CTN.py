# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 10:41:10 2019
@author: sebastianfriedrich
"""
print('Start App') 
# =============================================================================
#    Import Files, Folders and Modules
# =============================================================================
print('Load Modules') 
import matplotlib.pyplot as plt
import matplotlib
import tensorflow as tf
import numpy as np
import os
import sys
import scipy
from scipy import signal
from scipy.io import loadmat
import keras
#from keras.models import Sequential
#from keras.layers import Dense
from datetime import datetime
from tensorflow.keras.callbacks import EarlyStopping

#set working directory
os.chdir('/Users/sebastianfriedrich/Documents/Hochschule Trier/Module/Masterprojekt (LAROS)/Python/')

#import Folders and Files
sys.path.insert(0, '/Users/sebastianfriedrich/Documents/Hochschule Trier/Module/Masterprojekt (LAROS)/Python/Python Files')
import dataFunctions
import modelFunctions

timestamp = datetime.now()
tf.reset_default_graph()   # To clear the defined variables and operations of the previous cell

SaveSessionLog = True;
OnlyForTesting = True; #Only Load 5 Data Files

# =============================================================================
#    Open Matlab Files
# =============================================================================
print('Open Matlab Files') 
try:
    TraningDataMatrixes
    NoOfTrainingDataSets
except:
    dataDir = "/Users/sebastianfriedrich/Documents/Hochschule Trier/Module/Masterprojekt (LAROS)/TrainingData/"
    TraningDataMatrixes,NoOfTrainingDataSets = dataFunctions.openTraingsDataFiles(dataDir,OnlyForTesting);

# =============================================================================
#    Ckeck Selected Data
# =============================================================================
dataFunctions.plotSpectrogram(TraningDataMatrixes,1) #Plot Spectrogram of given Dataset
dataFunctions.plotFrequenzyProfile(TraningDataMatrixes,1) #Plot Spectrogram of given Dataset
#(x1,x2,x3)=dataFunctions.test()

# =============================================================================
#    Create Model
# =============================================================================
print('Create Model') 
NoOfUsedInputChannels = 1;      #No of Channels used as Input for the Network
NoOfUsedInputSamples = 300;     #No of Samples per Channel used as Input for the Network
NoOfUsedOutputSamplesPerSignal = 300;
NoOfUsedOutputSignals = 2; #No of Outputsignals Can be 1 or 2 for Heartbeat and/or Respiration
NoOfOutputSamples = NoOfUsedOutputSamplesPerSignal*NoOfUsedOutputSignals;        #No of Sampels used for Output Signal

with tf.name_scope('Placeholders'):    
    x_in = tf.placeholder(tf.float32, [None,NoOfUsedInputSamples,NoOfUsedInputChannels],name='x_in') #Define Input Data Structure [batchSize, inputDim1, inputDim2]
    y_Label_R = tf.placeholder(tf.float32, [None,NoOfUsedOutputSamplesPerSignal,1],name='y_Label_R') #Define Output Data Structure for Respiration [batchSize, inputDim1, inputDim2]
    y_Label_H = tf.placeholder(tf.float32, [None,NoOfUsedOutputSamplesPerSignal,1],name='y_Label_H') #Define Output Data Structure for Hertbeat [batchSize, inputDim1, inputDim2]
#    y_Label = tf.placeholder(tf.float32, [None,NoOfOutputSamples,1],name='y_Out') #Define Output Data Structure for Hertbeat and Respiration[batchSize, inputDim1, inputDim2]    
X_input_NN, X_Phase_NN, Y_Label_NN, NoOfSamples, y_LabelR, y_LabelHB = dataFunctions.randomizeAllData(TraningDataMatrixes,NoOfTrainingDataSets,NoOfUsedInputSamples,NoOfOutputSamples,False);

with tf.name_scope("NeuralNetwork"):
    
     model = modelFunctions.DC_CNN_Model(NoOfUsedInputSamples)
     model.compile(loss='mean_squared_error', optimizer='adam')
     
     TimeStamp = timestamp.strftime("Date_%d%m%Y_%H%M%S")
     logdir = "'/Users/sebastianfriedrich/Documents/Hochschule Trier/Module/Masterprojekt (LAROS)/Python/" + "logs/train/"+TimeStamp + "'"
     print("tensorboard --logdir='/Users/sebastianfriedrich/Documents/Hochschule Trier/Module/Masterprojekt (LAROS)/Python/" + "logs/train/"+TimeStamp + "'")
     tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)  
     monitor = EarlyStopping(monitor='loss', min_delta=1e-3, patience=5, verbose=1, mode='auto', restore_best_weights=True)
     model.fit(X_input_NN,y_LabelR,callbacks=[monitor, tensorboard_callback],verbose=2,epochs=1000)

    
