# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 10:41:10 2019
@author: sebastianfriedrich
"""
# =============================================================================
#    Import Files, Folders and Modules
# =============================================================================
import matplotlib.pyplot as plt
import matplotlib
import tensorflow as tf
import numpy as np
import os
import sys
import scipy
from scipy import signal
from scipy.io import loadmat

#set working directory
os.chdir('/Users/sebastianfriedrich/Documents/Hochschule Trier/Module/Masterprojekt (LAROS)/Python/')

#import Folders and Files
sys.path.insert(0, '/Users/sebastianfriedrich/Documents/Hochschule Trier/Module/Masterprojekt (LAROS)/Python/Python Files')
import checkDataFunctions
import modelFunctions

# =============================================================================
#    Open Matlab Files
# =============================================================================
dataDir = "/Users/sebastianfriedrich/Documents/Hochschule Trier/Module/Masterprojekt (LAROS)/TrainingData/"


TraningDataMatrixes = []
NoOfTrainingDataSets = 0

for file in os.listdir(dataDir):
    if file.endswith('.mat'): #Load only if .mat file (.ds_Stroe file causes error)
        TraningDataMatrixes.append(scipy.io.loadmat(dataDir+file, struct_as_record=False, squeeze_me = True))
        print("load Data from file "+file)
        NoOfTrainingDataSets = NoOfTrainingDataSets+1
 
# =============================================================================
#    Ckeck Selected Data
# =============================================================================
checkDataFunctions.plotSpectrogram(TraningDataMatrixes,1) #Plot Spectrogram of given Dataset
checkDataFunctions.plotFrequenzyProfile(TraningDataMatrixes,1) #Plot Spectrogram of given Dataset
#(x1,x2,x3)=checkDataFunctions.test()

# =============================================================================
#    Create Model
# =============================================================================
NoOfUsedInputChannels = 1;      #No of Channels used as Input for the Network
NoOfUsedInputSamples = 10;     #No of Samples per Channel used as Input for the Network
NoOfOutputSamples = NoOfUsedInputSamples;        #No of Sampels used for Output Signal

batch_Size = 1
    
x_in=tf.placeholder(tf.float32, [None,NoOfUsedInputSamples,NoOfUsedInputChannels]) #Define Input Data Structure [batchSize, inputDim1, inputDim2]
y_outR=tf.placeholder(tf.float32, [None,NoOfOutputSamples,1]) #Define Output Data Structure for Respiration [batchSize, inputDim1, inputDim2]
y_outH=tf.placeholder(tf.float32, [None,NoOfOutputSamples,1]) #Define Output Data Structure for Hertbeat [batchSize, inputDim1, inputDim2]
    
#Layer1
LenghtConv_L1 = 30;
NoFilters_L1 = 20;

#Layer2
LenghtConv_L2 = 10;
NoFilters_L2 = 10;

## setup computational graph
#layer1 = modelFunctions.newConvoulution1DLayer(x_in,(LenghtConv_L1,1,NoFilters_L1)) #[filter_width, in_channels, out_channels(No of Filters)]
#layer2 = modelFunctions.newConvoulution1DLayer(layer1,(LenghtConv_L2,NoFilters_L1,10))
#full_layer_one = tf.nn.relu(normal_full_layer(layer2,NoOfOutputSamples))
#
#num_hidden = 24
#cell = tf.nn.rnn_cell.LSTMCell(num_hidden,state_is_tuple=True);
#
## setup Learning/Cost Functions
#loss = tf.reduce_mean(tf.square(full_layer_one - y_outH)) #MSE
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
#train = optimizer.minimize(loss)
#
## create Session
#init = tf.global_variables_initializer() #Initalize Global Variables and Placeholders 

#Prepare Data
X_input, y_Label=checkDataFunctions.splitDataIntoTrainingExamples1D(TraningDataMatrixes[1]['Data'],NoOfUsedInputSamples,NoOfOutputSamples,False)

with tf.Session() as sess:
    sess.run(init)
    
    SignalsRaw = TraningDataMatrixes[1]['Data'].Radar.SignalsRaw;
    RespirationSignalTrue = TraningDataMatrixes[1]['Data'].Model.SignalRespirationHub;
    feed_dict_train = {x_in: SignalsRaw,y_outH:RespirationSignalTrue}; #Assign Values to Placeholders 
    
    out_data=session.run(full_layer_one,feed_dict=feed_dict_train);
    
    
