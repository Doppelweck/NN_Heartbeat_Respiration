#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 11:13:48 2019

@author: sebastianfriedrich
"""


import tensorflow as tf
import numpy as np
import os
import scipy
from scipy import signal
from scipy.io import loadmat
import collections
import pandas as pd

#from keras.layers.core import Activation, Dense, Dropout, RepeatVector, SpatialDropout1D
##from keras.layers.embedding import Embedding
#from keras.layers.recurrent import GRU
#from keras.layers.wrapper import TimeDistributed
#from keras.utils import np_utils
#from sklearn.model_selection import train_test_split
#from keras.models import Model
#from keras.layers import *
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.layers import Conv1D, Input, Add, Activation, Dropout
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import TruncatedNormal
from keras.layers.advanced_activations import LeakyReLU, ELU
from tensorflow.keras import optimizers


def createSeqentialModel():
    NoOfUsedInputChannels = 8;      #No of Channels used as Input for the Network
    NoOfUsedInputSamples = 300;     #No of Samples per Channel used as Input for the Network
    NoOfOutputSamples = 300;        #No of Sampels used for Output Signal
    
    x_in=tf.placeholder(tf.float32, [None,NoOfUsedInputSamples,NoOfUsedInputChannels]) #Define Input Data Structure [batchSize, inputDim1, inputDim2]
    y_outR=tf.placeholder(tf.float32, [None,NoOfOutputSamples,1]) #Define Output Data Structure for Respiration [batchSize, inputDim1, inputDim2]
    y_outH=tf.placeholder(tf.float32, [None,NoOfOutputSamples,1]) #Define Output Data Structure for Hertbeat [batchSize, inputDim1, inputDim2]
    
    layer1 = newConvoulution1DLayer(x_in,(LenghtConv,NoFilters_1,1))
    layer12 = newConvoulution1DLayer(layer1,(NoOfUsedInputSamples,NoFilters_2,NoFilters_1))
    #reshape
    layer2 = newLinearReLULayer(layer1,NoOfOutputSamples,(1,100))
    y_pred=tf.nn.softmax(layer2)
    
#    inpTensor = Input((NoOfUsedInputChannels,NoOfUsedInputSamples))  
    
    
def newLinearReLULayer(indata, number_of_neurons, in_dim):
    # returns a new linear unit with ReLu activation
    shape_w = (number_of_neurons, in_dim)
    weight = tf.Variable(tf.truncated_normal(shape_w, stddev=0.5))
    bias = tf.Variable(tf.zeros([number_of_neurons,1]))
    activation = tf.transpose(tf.nn.relu(tf.matmul(weight, tf.transpose(indata)) + bias))
    return activation  

def meanFree(X,name):
    with tf.name_scope(name):
        n_inputs = int(X.get_shape()[1])
        W = tf.ones([n_inputs,1])
        mean = tf.reduce_sum( tf.multiply( W, X ))/n_inputs
        X_meanFree = X-mean
        return X_meanFree

def neuron_Layer_FullyConnected(X,n_neurons,name,activation=None):
    with tf.name_scope(name):
        n_inputs = int(X.get_shape()[1])
        stddev = 2/np.sqrt(n_inputs)
        init = tf.truncated_normal((n_neurons,n_inputs),stddev=stddev)
        with tf.name_scope('weights'):
            W = tf.Variable(init, name="kernel")
        with tf.name_scope('biasses'):   
            bias = tf.Variable(tf.zeros([n_neurons,1]),name="bias")
        Z = tf.matmul(W,X) + bias
        if activation is not None:
            return activation(Z),W
        else:
            return Z,W
        
def neuron_Layer_TimeForwardConnected(X,n_neurons,name,activation=None):
    n_inputs = int(X.get_shape()[1])
    if n_neurons>n_inputs:
        #Use fully Connected Layer
        with tf.name_scope(name):
            n_inputs = int(X.get_shape()[1])
            stddev = 0.333
            init = tf.truncated_normal((n_neurons,n_inputs),stddev=stddev,mean=0)
            with tf.name_scope('weights'):
                W = tf.Variable(init, name="kernel")
            with tf.name_scope('biasses'):     
                bias = tf.Variable(tf.zeros([n_neurons,1]),name="bias")
            Z = tf.matmul(W,X) + bias
            if activation is not None:
                return activation(Z),W
            else:
                return Z,W
    else: 
        # Use Time Forward Connected Layer
        with tf.name_scope(name):
            n_inputs = int(X.get_shape()[1])
            stddev = 2/np.sqrt(n_inputs)
            stddev = 0.333
            with tf.name_scope('mask'):
                mask = np.ones([n_neurons, n_inputs])
                mask = np.tril(mask,-(n_neurons-n_inputs))
            
            init = tf.truncated_normal((n_neurons,n_inputs),stddev=stddev,mean=0)
            with tf.name_scope('weights'):
                W = tf.math.multiply(tf.Variable(init, name="kernel"),mask)
            with tf.name_scope('biasses'):    
                bias = tf.Variable(tf.zeros([n_neurons,1]),name="bias")
            Z = tf.matmul(W,X) + bias
            if activation is not None:
                return activation(Z),W
            else:
                return Z,W      

def newConvoulution1DLayer(indata,ConvCore):
    #indata:    Input Tensor for Conv Layer [batchSize, inputDim1, inputDim2]
    #ConvCore:  Shape of Convolution Core (filter) [filter_width, in_channels, out_channels(No of Filters)]
    init_random_dist = tf.truncated_normal(ConvCore, stddev=0.1)
    weights = tf.Variable(init_random_dist)
    init_bias_vals = tf.constant(0.1, shape=ConvCore)
    bias = tf.Variable(init_bias_vals)
    activation = tf.nn.conv1d(indata,weights,stride=1,padding="SAME")+bias
    return tf.nn.relu(activation)

def newRNNLayer(n_inputs, n_neurons, n_steps):
    #
    X = tf.placeholder(tf.float32,[None, n_steps, n_inputs])
    basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
    outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)
    return outputs,states

def newLSTMLayer(n_inputs, n_neurons, n_steps):
    #
    X = tf.placeholder(tf.float32,[None, n_steps, n_inputs])
    basic_cell = tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons)
    outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)
    return outputs,states

def DC_CNN_Block(nb_filter, filter_length, dilation, l2_layer_reg):
   def f(input_):

       residual =    input_

       layer_out =   Conv1D(filters=nb_filter, kernel_size=filter_length, 
                     dilation_rate=dilation, 
                     activation='linear', padding='causal', use_bias=False,
                     kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05, 
                     seed=42), kernel_regularizer=l2(l2_layer_reg))(input_)

       layer_out =   Activation('selu')(layer_out)

       skip_out =    Conv1D(1,1, activation='linear', use_bias=False, 
                     kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05, 
                     seed=42), kernel_regularizer=l2(l2_layer_reg))(layer_out)

       network_in =  Conv1D(1,1, activation='linear', use_bias=False, 
                     kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05, 
                     seed=42), kernel_regularizer=l2(l2_layer_reg))(layer_out)

       network_out = Add()([residual, network_in])

       return network_out, skip_out

   return f

def DC_CNN_Model(length):

   input = Input(shape=(length,1))
#   input = Input(shape=(1,8)) # AKRAIN

   l1a, l1b = DC_CNN_Block(32,2,1,0.001)(input)    
   l2a, l2b = DC_CNN_Block(32,2,2,0.001)(l1a) 
   l3a, l3b = DC_CNN_Block(32,2,4,0.001)(l2a)
   l4a, l4b = DC_CNN_Block(32,2,8,0.001)(l3a)
   l5a, l5b = DC_CNN_Block(32,2,16,0.001)(l4a)
   l6a, l6b = DC_CNN_Block(32,2,32,0.001)(l5a)
   l6b = Dropout(0.8)(l6b) #dropout used to limit influence of earlier data
   l7a, l7b = DC_CNN_Block(32,2,64,0.001)(l6a)
   l7b = Dropout(0.8)(l7b) #dropout used to limit influence of earlier data

   l8 =   Add()([l1b, l2b, l3b, l4b, l5b, l6b, l7b])

   l9 =   Activation('relu')(l8)

   l21 =  Conv1D(1,1, activation='linear', use_bias=False, 
          kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05, seed=42),
          kernel_regularizer=l2(0.001))(l9)

   model = Model(inputs=input, outputs=l21)
   # model = Model(output=l21)

   adam = optimizers.Adam(lr=0.00075, beta_1=0.9, beta_2=0.999, epsilon=None, 
                          decay=0.0, amsgrad=False)

   model.compile(loss='mae', optimizer=adam, metrics=['mse'])

   return model
    
    
    
    
    
    