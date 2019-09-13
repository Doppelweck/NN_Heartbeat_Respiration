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
from datetime import datetime

#set working directory
os.chdir('/Users/sebastianfriedrich/Documents/Hochschule Trier/Module/Masterprojekt (LAROS)/Python/')

#import Folders and Files
sys.path.insert(0, '/Users/sebastianfriedrich/Documents/Hochschule Trier/Module/Masterprojekt (LAROS)/Python/Python Files')
import dataFunctions
import modelFunctions

timestamp = datetime.now()
tf.reset_default_graph()   # To clear the defined variables and operations of the previous cell

SaveSessionLog = True;
# =============================================================================
#    Open Matlab Files
# =============================================================================
dataDir = "/Users/sebastianfriedrich/Documents/Hochschule Trier/Module/Masterprojekt (LAROS)/TrainingData/"
TraningDataMatrixes,NoOfTrainingDataSets = dataFunctions.openTraingsDataFiles(dataDir);

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
NoOfUsedInputSamples = 300;     #No of Samples per Channel used as Input for the Network
NoOfUsedOutputSamples = 100;
NoOfUsedOutputSignals = 2; #No of Outputsignals Can be 1 or 2 for Heartbeat and/or Respiration
NoOfOutputSamples = NoOfUsedOutputSamples*NoOfUsedOutputSignals;        #No of Sampels used for Output Signal

with tf.name_scope('Placeholders'):    
    x_in = tf.placeholder(tf.float32, [None,NoOfUsedInputSamples,NoOfUsedInputChannels],name='x_in') #Define Input Data Structure [batchSize, inputDim1, inputDim2]
    #y_outR = tf.placeholder(tf.float32, [None,NoOfOutputSamples,1]) #Define Output Data Structure for Respiration [batchSize, inputDim1, inputDim2]
    #y_outH = tf.placeholder(tf.float32, [None,NoOfOutputSamples,1]) #Define Output Data Structure for Hertbeat [batchSize, inputDim1, inputDim2]
    y_Label = tf.placeholder(tf.float32, [None,NoOfOutputSamples,1],name='y_Label') #Define Output Data Structure for Hertbeat and Respiration[batchSize, inputDim1, inputDim2]    
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
#layer1 = modelFunctions.newLinearReLULayer(x_in, NoOfUsedInputSamples, NoOfUsedInputSamples)
#print(layer1)
#layer2 = modelFunctions.newLinearReLULayer(layer1, 200, NoOfUsedInputSamples)
#print(layer2)
#layer3 = modelFunctions.newLinearReLULayer(layer2, 300, 200)
#print(layer3)
#y_outHR = modelFunctions.newLinearReLULayer(layer3, NoOfOutputSamples, 300)
#print(y_outHR)
#
with tf.name_scope("NeuralNetwork"):
    hidden1,W1 = modelFunctions.neuron_Layer_FullyConnected(x_in,NoOfUsedInputSamples,"hidden1",activation=tf.nn.tanh)
    hidden2,W2 = modelFunctions.neuron_Layer_TimeForwardConnected(hidden1,260,"hidden2",activation=tf.nn.tanh)
    hidden3,W3 = modelFunctions.neuron_Layer_TimeForwardConnected(hidden2,230,"hidden3",activation=tf.nn.tanh)
    y_outHR,W4 = modelFunctions.neuron_Layer_TimeForwardConnected(hidden3,NoOfOutputSamples,"outputLayer",activation=tf.nn.tanh)
#
## setup Learning/Cost Functions
n_iterations = 2;
batch_size = 30; #Anzahl gleichzeitiger Samples im Netz
learning_Rate = 0.0001;
DataInRandomOrder = False;

loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y_outHR ,y_Label)))) #RMSE
optimizer = tf.train.AdamOptimizer(learning_rate=learning_Rate)
train = optimizer.minimize(loss)
teta=tf.trainable_variables()
gradLoss = tf.gradients(loss,teta)
gradW1 = tf.gradients(loss,W1)
NormGradLoss = tf.linalg.global_norm(gradLoss)
#
## create Session
init = tf.global_variables_initializer() #Initalize Global Variables and Placeholders 

if SaveSessionLog:
    merged = tf.summary.merge_all()
    TimeStamp = timestamp.strftime("Date_%d%m%Y_%H%M%S")
    LogPath = './logs/train/'+TimeStamp;
    RMSE_summary = tf.summary.scalar('RMSE',loss)
    NormGradient_summary = tf.summary.scalar('Normalized Gradient',NormGradLoss)
    writer = tf.summary.FileWriter(LogPath)
    step = 0

with tf.Session() as sess:
    
    sess.run(init)
    for i_iteration in range(n_iterations):
        
        for i_TrainDataSet in range(NoOfTrainingDataSets-1): #Get new Set of Training Data
            #Prepare Data
            X_input_NN, Y_Label_NN, NoOfSamples, y_LabelR, y_LabelHB = dataFunctions.splitDataIntoTrainingExamples1D(TraningDataMatrixes[i_TrainDataSet]['Data'],NoOfUsedInputSamples,NoOfOutputSamples,False)
            
            if DataInRandomOrder: #Shuffle Data Pairs in Random order
                X_input_NN, Y_Label_NN = dataFunctions.shuffleCompleteDataBatch([X_input_NN, Y_Label_NN])
                print('!!!!!!!!!!!! Data is in random order !!!!!!!!!!!!!!')
            
            # get total number of differnt Batches. One Batch contains n samles of Training Data with Size batch_size to feed into the Netowrk
            i_Batches = dataFunctions.getNumberOfBatches(NoOfSamples,batch_size); #Total Number of different Batches
            Total_No_Batches = i_Batches;
            
            for i_Batches in range(i_Batches): 
                #get next batch
                X_input_Batch, Y_Label_Batch = dataFunctions.nextBatch([X_input_NN, Y_Label_NN],batch_size,Total_No_Batches,i_Batches)
                #feed batch into NN    
                feed_dict_train = {x_in: X_input_Batch,y_Label:Y_Label_Batch};
                #train the NN
                out_data=sess.run([train], feed_dict=feed_dict_train);
    
                if i_Batches %10==0:
                    Loss, Ngrad = sess.run([loss ,NormGradLoss],feed_dict=feed_dict_train)
                    print('Iteration:',i_iteration,'TrainSet:',i_TrainDataSet,' Batch:',i_Batches,' Loss:',Loss,' NormGrad:',Ngrad)
                    if SaveSessionLog:
                        summary_str = RMSE_summary.eval(feed_dict=feed_dict_train)
                        summary_grad = NormGradient_summary.eval(feed_dict=feed_dict_train)
                        summary_gradW1 = sess.run(gradW1,feed_dict=feed_dict_train)
                        print(summary_gradW1)
                        fig1 = plt.figure()
                        plt.imshow(np.squeeze(np.array(summary_gradW1)),[])
                        plt.show()
                        step=step+1
                        writer.add_summary(summary_str, step)
                        writer.add_summary(summary_grad, step)
                        
    pr=sess.run([y_outHR], feed_dict=feed_dict_train);                
    
#    figPred = plt.figure()
#    plt.plot(pr)
#    plt.show()
#    plt.plot(Y_Label_Batch)
#    plt.show()
    
    if SaveSessionLog:
        writer.add_graph(sess.graph)

if SaveSessionLog:    
    tensorboardString = "tensorboard --logdir='/Users/sebastianfriedrich/Documents/Hochschule Trier/Module/Masterprojekt (LAROS)/Python/" + "logs/train/"+TimeStamp + "'"
    pathString = "/Users/sebastianfriedrich/Documents/Hochschule Trier/Module/Masterprojekt (LAROS)/Python/" + "logs/train/"+TimeStamp + "/tensorboard.txt"
    print('') 
    print('!!!! Tensorboard comand for terminal:')       
    print(tensorboardString)
    textFile = open(pathString,"w+")
    textFile.write(tensorboardString)
    textFile.close()
    writer.close()
































