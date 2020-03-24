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
#from keras.models import Sequential
#from keras.layers import Dense
from datetime import datetime

#set working directory
os.chdir('/Users/sebastianfriedrich/Documents/Hochschule Trier/Module/Masterprojekt (LAROS)/Python/')
#os.chdir('/Users/sebastianfriedrich/Desktop/MA LaROS/Python/')

#import Folders and Files
sys.path.insert(0, '/Users/sebastianfriedrich/Documents/Hochschule Trier/Module/Masterprojekt (LAROS)/Python/Python Files')
#sys.path.insert(0, '/Users/sebastianfriedrich/Desktop/MA LaROS/Python/Python Files')
import dataFunctions
import modelFunctions

timestamp = datetime.now()
tf.reset_default_graph()   # To clear the defined variables and operations of the previous cell

SaveSessionLog = False;
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
    #dataDir = "/Users/sebastianfriedrich/Desktop/MA LaROS/TrainingData/"
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


with tf.name_scope("NeuralNetwork"):
#    X_meanFree = modelFunctions.meanFree(x_in,'MeanFreeLayer')
#    Layer1,W1 = modelFunctions.neuron_Layer_FullyConnected(X_meanFree,NoOfUsedInputSamples,"InputLayer",activation=tf.nn.relu)
#    Layer2,W2 = modelFunctions.neuron_Layer_TimeForwardConnected(Layer1,260,"hidden2",activation=tf.nn.tanh)
#    Layer3,W2 = modelFunctions.neuron_Layer_TimeForwardConnected(Layer2,230,"hidden3",activation=tf.nn.tanh)
#   Split NN for Heartbeat
    Layer1_HB,W1_HB = modelFunctions.neuron_Layer_FullyConnected(x_in,300,"hidden1_HB",activation=tf.nn.tanh)
    Layer2_HB,W2_HB = modelFunctions.neuron_Layer_FullyConnected(Layer1_HB,300,"hidden2_HB",activation=tf.nn.tanh)
#    Layer3_HB,W3_HB = modelFunctions.neuron_Layer_TimeForwardConnected(Layer2_HB,350,"hidden3_HB",activation=tf.nn.tanh)
    y_Out_H,WOut_HB = modelFunctions.neuron_Layer_FullyConnected(Layer2_HB,NoOfUsedOutputSamplesPerSignal,"outputLayer_HB",None)
#   Split NN for Respiration
    Layer1_R,W1_R = modelFunctions.neuron_Layer_FullyConnected(x_in,300,"hidden1_R",activation=tf.nn.tanh)
    Layer2_R,W2_R = modelFunctions.neuron_Layer_FullyConnected(Layer1_R,300,"hidden2_R",activation=tf.nn.tanh)
#    Layer3_R,W3_R = modelFunctions.neuron_Layer_TimeForwardConnected(Layer2_R,350,"hidden3_R",activation=tf.nn.tanh)
    y_Out_R,WOut_R = modelFunctions.neuron_Layer_FullyConnected(Layer2_R,NoOfUsedOutputSamplesPerSignal,"outputLayer_R",None)    
#
## setup Learning/Cost Functions
n_Epoch = 3;
batch_size = 30; #Anzahl gleichzeitiger Samples im Netz
learning_Rate = 0.0001;
DataInRandomOrder = False;

lossR = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y_Out_R ,y_Label_R)))) #RMSE for Respiration
lossH = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y_Out_H ,y_Label_H)))) #RMSE for Heartbeat
lossTotal = tf.add(lossR,lossH)

optimizerR = tf.train.AdamOptimizer(learning_rate=learning_Rate,beta1=0.9, beta2=0.999, epsilon=1e-08)
optimizerH = tf.train.AdamOptimizer(learning_rate=learning_Rate,beta1=0.9, beta2=0.999, epsilon=1e-08)
trainR = optimizerR.minimize(lossR)
trainH = optimizerH.minimize(lossH)
train = tf.group(trainR,trainH)

teta=tf.trainable_variables()
gradLoss = tf.gradients(lossTotal,teta)
#gradW1 = tf.gradients(lossTotal,W1)
NormGradLoss = tf.linalg.global_norm(gradLoss)
#
## create Session
init = tf.global_variables_initializer() #Initalize Global Variables and Placeholders 

print('Create Session') 
if SaveSessionLog:
    merged = tf.summary.merge_all()
    TimeStamp = timestamp.strftime("Date_%d%m%Y_%H%M%S")
    LogPath = './logs/train/'+TimeStamp;
    RMSE_HB_summary = tf.summary.scalar('RMSE_Heartbeat',lossH)
    RMSE_R_summary = tf.summary.scalar('RMSE_Respiration',lossR)
    TotalLoss_summary = tf.summary.scalar('Total_Loss',lossTotal)
    
    NormGradient_summary = tf.summary.scalar('Normalized_Gradient',NormGradLoss)
    writer = tf.summary.FileWriter(LogPath)
    step = 0

print('Run Session') 
with tf.Session() as sess:
    
    sess.run(init)
    for i_Epoch in range(n_Epoch):
#            learning_Rate = learning_Rate/(10*(i_Epoch+1));
#            optimizer = tf.train.AdamOptimizer(learning_rate=learning_Rate)
#        for i_TrainDataSet in range(NoOfTrainingDataSets-1): #Get new Set of Training Data
            #Prepare Data
            X_input_NN, X_Phase_NN, Y_Label_NN, NoOfSamples, y_LabelR, y_LabelHB = dataFunctions.randomizeAllData(TraningDataMatrixes,NoOfTrainingDataSets,NoOfUsedInputSamples,NoOfOutputSamples,False);
#            X_input_NN, Y_Label_NN, NoOfSamples, y_LabelR, y_LabelHB = dataFunctions.splitDataIntoTrainingExamples1D(TraningDataMatrixes[i_TrainDataSet]['Data'],NoOfUsedInputSamples,NoOfOutputSamples,False)
            
            # get total number of differnt Batches. One Batch contains n samles of Training Data with Size batch_size to feed into the Netowrk
            n_Batches = dataFunctions.getNumberOfBatches(NoOfSamples,batch_size); #Total Number of different Batches
            Total_No_Batches = n_Batches;
            
            for i_Batches in range(n_Batches): 
                #get next batch
                X_input_Batch, Y_Label_Batch, y_LabelR_Batch, y_LabelHB_Batch = dataFunctions.nextBatch([X_Phase_NN, Y_Label_NN, y_LabelR, y_LabelHB],batch_size,Total_No_Batches,i_Batches)
                #feed batch into NN    
                feed_dict_train = {x_in: X_input_Batch, y_Label_R:y_LabelR_Batch, y_Label_H:y_LabelHB_Batch};
                #train the NN
                out_data=sess.run([train], feed_dict=feed_dict_train);
    
                if i_Batches %10==0:
                    
                    LossR, LossH= sess.run([lossR ,lossH],feed_dict=feed_dict_train)
                    print('Epoch:',i_Epoch,' Batch:',i_Batches,' LossR:',LossR,' LossH:',LossH)
                    testWeights = sess.run(W2_HB)
                    
                    if SaveSessionLog:
                        
                        RMSE_HB_summary_str = RMSE_HB_summary.eval(feed_dict=feed_dict_train)
                        RMSE_R_summary_str = RMSE_R_summary.eval(feed_dict=feed_dict_train)
#                        summary_grad = NormGradient_summary.eval(feed_dict=feed_dict_train)
#                        summary_gradW1 = sess.run(gradW1,feed_dict=feed_dict_train)
#                        print(summary_gradW1)
#                        fig1 = plt.figure()
#                        plt.imshow(np.squeeze(np.array(summary_gradW1)),[])
#                        plt.show()
                        
                        step=step+1
                        writer.add_summary(RMSE_HB_summary_str, step)
                        writer.add_summary(RMSE_R_summary_str, step)
#                        writer.add_summary(summary_grad, step)
                 
                if i_Batches %100==0:    
                        
#                    prH=sess.run([y_Out_H], feed_dict=feed_dict_train);
#                    preH=np.array(prH)
#                    y_pre_H=np.squeeze(preH)
#                    
#                    prR=sess.run([y_Out_R], feed_dict=feed_dict_train);
#                    preR=np.array(prR)
#                    y_pre_R=np.squeeze(preR)                    
#                    
#                    figPred = plt.figure()
#                    plt.plot(y_pre_H[0,:])
#                    plt.plot(y_LabelHB_Batch[0,:,0])
#                    plt.show()
#                    
#                    figPred = plt.figure()
#                    plt.plot(y_pre_R[0,:])
#                    plt.plot(y_LabelR_Batch[0,:,0])
#                    plt.show()
#                    
                    figPred = plt.figure()
                    plt.plot(X_input_Batch[0,:]/((2*3.141592/6)*(2*2.45)))
#                    plt.plot(Y_Label_Batch[0,:,0])
                    plt.show()
#                    
#                    
#                    batchIn = sess.run([X_meanFree], feed_dict=feed_dict_train);
#                    batchInArray=np.array(batchIn)
#                    batchInArray=np.squeeze(batchInArray)
##                    
#                    figPred = plt.figure()
#                    plt.plot(batchInArray[0,:])
#                    plt.show()
#                    figPred = plt.figure()
#                    plt.plot(y_LabelR_Batch[0,:,0])
#                    plt.plot(X_input_Batch[0,:,0])
#                    plt.show()
                    
                    
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
    
print('') 
print('!!!! END')

































