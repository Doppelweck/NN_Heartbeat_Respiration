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

SaveSessionLog = False;
OnlyForTesting = True; #Only Load 5 Data Files
# =============================================================================
#    Open Matlab Files
# =============================================================================
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


with tf.name_scope("NeuralNetwork"):
    Layer1,W1 = modelFunctions.neuron_Layer_FullyConnected(x_in,NoOfUsedInputSamples,"InputLayer",activation=tf.nn.tanh)
    Layer2,W2 = modelFunctions.neuron_Layer_TimeForwardConnected(Layer1,260,"hidden2",activation=tf.nn.tanh)
    Layer3,W3 = modelFunctions.neuron_Layer_TimeForwardConnected(Layer2,230,"hidden3",activation=tf.nn.tanh)
    y_outHR,W4 = modelFunctions.neuron_Layer_TimeForwardConnected(Layer3,NoOfOutputSamples,"outputLayer",activation=tf.nn.tanh)
#
## setup Learning/Cost Functions
n_iterations = 5;
batch_size = 20; #Anzahl gleichzeitiger Samples im Netz
learning_Rate = 0.0008;
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
        
#        for i_TrainDataSet in range(NoOfTrainingDataSets-1): #Get new Set of Training Data
            #Prepare Data
            X_input_NN, Y_Label_NN, NoOfSamples, y_LabelR, y_LabelHB = dataFunctions.randomizeAllData(TraningDataMatrixes,NoOfTrainingDataSets,NoOfUsedInputSamples,NoOfOutputSamples,False)
#            X_input_NN, Y_Label_NN, NoOfSamples, y_LabelR, y_LabelHB = dataFunctions.splitDataIntoTrainingExamples1D(TraningDataMatrixes[i_TrainDataSet]['Data'],NoOfUsedInputSamples,NoOfOutputSamples,False)
            
            # get total number of differnt Batches. One Batch contains n samles of Training Data with Size batch_size to feed into the Netowrk
            i_Batches = dataFunctions.getNumberOfBatches(NoOfSamples,batch_size); #Total Number of different Batches
            Total_No_Batches = i_Batches;
            
            for i_Batches in range(i_Batches): 
                #get next batch
                X_input_Batch, Y_Label_Batch, y_LabelR_Batch, y_LabelHB_Batch = dataFunctions.nextBatch([X_input_NN, Y_Label_NN, y_LabelR, y_LabelHB],batch_size,Total_No_Batches,i_Batches)
                #feed batch into NN    
                feed_dict_train = {x_in: X_input_Batch,y_Label:Y_Label_Batch};
                #train the NN
                out_data=sess.run([train], feed_dict=feed_dict_train);
    
                if i_Batches %10==0:
                    
                    Loss, Ngrad = sess.run([loss ,NormGradLoss],feed_dict=feed_dict_train)
                    print('Iteration:',i_iteration,' Batch:',i_Batches,' Loss:',Loss,' NormGrad:',Ngrad)
                    
                    if SaveSessionLog:
                        
                        summary_str = RMSE_summary.eval(feed_dict=feed_dict_train)
                        summary_grad = NormGradient_summary.eval(feed_dict=feed_dict_train)
                        summary_gradW1 = sess.run(gradW1,feed_dict=feed_dict_train)
#                        print(summary_gradW1)
#                        fig1 = plt.figure()
#                        plt.imshow(np.squeeze(np.array(summary_gradW1)),[])
#                        plt.show()
                        
                        step=step+1
                        writer.add_summary(summary_str, step)
                        writer.add_summary(summary_grad, step)
                 
                if i_Batches %200==0:    
                        
                    pr=sess.run([y_outHR], feed_dict=feed_dict_train);
                    pre=np.array(pr)
                    pre=np.squeeze(pre)
                    y_pre = pre.reshape(pre.shape[0],100,2)
                    
                    Y_Label_Test = Y_Label_Batch.reshape(Y_Label_Batch.shape[0],100,2)
                    
                    figPred = plt.figure()
                    plt.plot(y_pre[0,:,0])
                    plt.plot(Y_Label_Test[0,:,0])
                    plt.show()
                    
                    figPred = plt.figure()
                    plt.plot(y_pre[0,:,1])
                    plt.plot(Y_Label_Test[0,:,1])
                    plt.show()
                    
                    
#                    Y_Label_Test = Y_Label_Batch.reshape(30,100,2)
#                    figPred = plt.figure()
#                    plt.plot(Y_Label_Test[0,:,0])
#                    plt.show()
    
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
































