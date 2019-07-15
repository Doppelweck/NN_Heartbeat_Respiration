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
(x1,x2,x3)=checkDataFunctions.test()

# =============================================================================
#    Create Model
# =============================================================================

