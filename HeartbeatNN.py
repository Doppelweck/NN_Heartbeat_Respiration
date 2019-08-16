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
from keras.models import Sequential
from keras.layers import Dense

#set working directory
os.chdir('/Users/sebastianfriedrich/Documents/Hochschule Trier/Module/Masterprojekt (LAROS)/Python/')

#import Folders and Files
sys.path.insert(0, '/Users/sebastianfriedrich/Documents/Hochschule Trier/Module/Masterprojekt (LAROS)/Python/Python Files')
import dataFunctions
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
dataFunctions.plotSpectrogram(TraningDataMatrixes,1) #Plot Spectrogram of given Dataset
dataFunctions.plotFrequenzyProfile(TraningDataMatrixes,1) #Plot Spectrogram of given Dataset
#(x1,x2,x3)=dataFunctions.test()

# =============================================================================
#    Create Model
# =============================================================================
NoOfUsedInputChannels = 1;      #No of Channels used as Input for the Network
NoOfUsedInputSamples = 10;     #No of Samples per Channel used as Input for the Network
NoOfUsedOutputSignals = 2; #No of Outputsignals Can be 1 or 2 for Heartbeat and/or Respiration
NoOfOutputSamples = NoOfUsedInputSamples*NoOfUsedOutputSignals;        #No of Sampels used for Output Signal

    
x_in = tf.placeholder(tf.float32, [None,NoOfUsedInputSamples,NoOfUsedInputChannels]) #Define Input Data Structure [batchSize, inputDim1, inputDim2]
y_outR = tf.placeholder(tf.float32, [None,NoOfOutputSamples,1]) #Define Output Data Structure for Respiration [batchSize, inputDim1, inputDim2]
y_outH = tf.placeholder(tf.float32, [None,NoOfOutputSamples,1]) #Define Output Data Structure for Hertbeat [batchSize, inputDim1, inputDim2]
y_Label = tf.placeholder(tf.float32, [None,NoOfOutputSamples,1]) #Define Output Data Structure for Hertbeat and Respiration[batchSize, inputDim1, inputDim2]    
#Layer1
LenghtConv_L1 = 5;
NoFilters_L1 = 3;

#Layer2
LenghtConv_L2 = 10;
NoFilters_L2 = 10;

#Prepare Data
#X_input, y_Label = dataFunctions.splitDataIntoTrainingExamples1D(TraningDataMatrixes[1]['Data'],NoOfUsedInputSamples,NoOfOutputSamples,False)


# setup computational graph
#layer1 = modelFunctions.newConvoulution1DLayer(x_in,(LenghtConv_L1,1,NoFilters_L1)) #[filter_width, in_channels, out_channels(No of Filters)]
#layer2 = modelFunctions.newConvoulution1DLayer(layer1,(LenghtConv_L2,NoFilters_L1,10))
#full_layer_one = tf.nn.relu(normal_full_layer(layer1,NoOfOutputSamples))
layer1 = modelFunctions.newLinearReLULayer(x_in, NoOfUsedInputSamples, NoOfUsedInputSamples)
print(layer1)
layer2 = modelFunctions.newLinearReLULayer(layer1, 30, NoOfUsedInputSamples)
print(layer2)
y_outHR = modelFunctions.newLinearReLULayer(layer2, NoOfOutputSamples, 30)
print(y_outHR)
#
#num_hidden = 24
#cell = tf.nn.rnn_cell.LSTMCell(num_hidden,state_is_tuple=True);
#
## setup Learning/Cost Functions
n_iterations = 100;
batch_size = 1; #Anzahl gleichzeitiger Samples im Netz
learning_Rate = 0.0005;

loss = tf.reduce_mean(tf.square(y_outHR - y_Label)) #MSE
optimizer = tf.train.AdamOptimizer(learning_rate=learning_Rate)
train = optimizer.minimize(loss)
teta=tf.trainable_variables()
gradLoss = tf.gradients(loss,teta)
NormGradLoss = tf.linalg.global_norm(gradLoss)
#
## create Session
init = tf.global_variables_initializer() #Initalize Global Variables and Placeholders 



#model = Sequential()
#model.add(Dense(10, activation='relu', input_dim=n_input))
#model.add(Dense(20))
#model.compile(optimizer='adam', loss='mse')
#model.fit(X_input, y_Label, epochs=2000, verbose=0)

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('./logs/1/train ')

with tf.Session() as sess:
    
    sess.run(init)
    for k in range(NoOfTrainingDataSets-1):
        #Prepare Data
        X_input, Y_Label = dataFunctions.splitDataIntoTrainingExamples1D(TraningDataMatrixes[k]['Data'],NoOfUsedInputSamples,NoOfOutputSamples,False)
        feed_dict_train = {x_in: X_input,y_Label:Y_Label};
        
        for i in range(n_iterations):
    #       SignalsRaw = TraningDataMatrixes[1]['Data'].Radar.SignalsRaw;
    #       RespirationSignalTrue = TraningDataMatrixes[1]['Data'].Model.SignalRespirationHub;
    #       feed_dict_train = {x_in: SignalsRaw,y_outH:RespirationSignalTrue}; #Assign Values to Placeholders
            out_data=sess.run([train], feed_dict=feed_dict_train);
#            summary = sess.run(merged,feed_dict=feed_dict_train);
#            writer.add_summary(summary)
    #       train_writer = tf.summary.FileWriter( './logs/1/train ', sess.graph)
            if i %10==0:
                acc, Ngrad = sess.run([loss ,NormGradLoss],feed_dict=feed_dict_train)
#                session.run(loss, feed_dict=feed_dict_train) 
#                mse=loss.eval(feed_dict=feed_dict_train)
#                print(k,i, "\tMSE",mse)
                print(k,i,acc,Ngrad)
                
    pr=sess.run([y_outHR], feed_dict=feed_dict_train);                
#writer.add_graph(sess.graph)
        
