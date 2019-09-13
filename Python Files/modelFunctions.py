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

def neuron_Layer_FullyConnected(X,n_neurons,name,activation=None):
    with tf.name_scope(name):
        n_inputs = int(X.get_shape()[1])
        stddev = 2/np.sqrt(n_inputs)
        init = tf.truncated_normal((n_neurons,n_inputs),stddev=stddev)
        W = tf.Variable(init, name="kernel")
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
            stddev = 2/np.sqrt(n_inputs)
            init = tf.truncated_normal((n_neurons,n_inputs),stddev=stddev)
            W = tf.Variable(init, name="kernel")
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
            stddev = 1
            
            mask = np.ones([n_neurons, n_inputs])
            mask = np.tril(mask,-(n_neurons-n_inputs))
            
            init = tf.truncated_normal((n_neurons,n_inputs),stddev=stddev)
            W = tf.Variable(init, name="kernel")
            bias = tf.Variable(tf.zeros([n_neurons,1]),name="bias")
            Z = tf.matmul(tf.math.multiply(W,mask),X) + bias
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
    
    
    
    
    
    