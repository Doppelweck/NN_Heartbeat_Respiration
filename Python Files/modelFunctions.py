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
import nltk

from keras.layers.core import Activation, Dense, Dropout, RepeatVector, SpatialDropout1D
from keras.layers.embedding import Embedding
from keras.layers.recurrent import GRU
from keras.layers.wrapper import TimeDistributed
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import *
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


def createSeqentialModel():
    NoOfUsedInputChannels = 8;      #No of Channels used as Input for the Network
    NoOfUsedInputSamples = 300;     #No of Samples per Channel used as Input for the Network
    NoOfOutputSamples = 300;        #No of Sampels used for Output Signal
    
    x_in=tf.placeholder(tf.float32, [None,NoOfUsedInputSamples,NoOfUsedInputChannels]) #Define Input Data Structure
    y_outR=tf.placeholder(tf.float32, [None,NoOfOutputSamples]) #Define Output Data Structure for Respiration
    y_outH=tf.placeholder(tf.float32, [None,NoOfOutputSamples]) #Define Output Data Structure for Hertbeat
    
    layer1 = newConvoulution1DLayer(x_in,(1,NoOfUsedInputSamples))
    layer2 = newLinearReLULayer(layer1,NoOfOutputSamples,(1,100))
    y_pred=tf.nn.softmax(layer2)
    
#    inpTensor = Input((NoOfUsedInputChannels,NoOfUsedInputSamples))  
    
    
def newLinearReLULayer(indata, number_of_neurons, in_dim):
    # returns a new linear unit with ReLu activation
    
    shape_w = (number_of_neurons, in_dim)

    weight = tf.Variable(tf.truncated_normal(shape_w, stddev=0.05))

    bias = tf.Variable(tf.zeros([number_of_neurons,1]))

    activation = tf.transpose(tf.nn.relu(tf.matmul(weight, tf.transpose(indata)) + bias))

    return activation  

def newConvoulution1DLayer(indata,ConvCore):

    init_random_dist = tf.truncated_normal(ConvCore, stddev=0.1)
    weights = tf.Variable(init_random_dist)
    init_bias_vals = tf.constant(0.1, shape=ConvCore)
    bias = tf.Variable(init_bias_vals)
    activation = tf.nn.conv1d(indata,weights,strides=1,padding='SAME')+bias
    return tf.nn.relu(activation)

def newLSTMLayer(indata,memoryLength):
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    