#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 14:51:10 2019
@author: sebastianfriedrich
"""

# =============================================================================
#    Data Struct Info .mat files
# =============================================================================
#    Data.Setup = Setup; %All Values for Traningdata Model
#    Data.Setup.HB_GainSmooth = HB_GainSmooth;
#    Data.Setup.HB_max_Hz = HB_max_Hz;
#    Data.Setup.HB_min_Hz = HB_min_Hz;
#    Data.Setup.HB_Init = HB_Init; %Init Value for random walk profile
#    Data.Setup.HB_Varianz = HB_Varianz;
#    Data.Setup.R_GainSmooth = R_GainSmooth;
#    Data.Setup.R_max_Hz = R_max_Hz;
#    Data.Setup.R_min_Hz = R_min_Hz;
#    Data.Setup.R_Init = R_Init; %Init Value for random walk profile
#    Data.Setup.R_Varianz = R_Varianz;
#    Data.Setup.FreqProfileHB = FPH;
#    Data.Setup.FreqProfileR = FPR;
#        
#    Data.Model.SignalSpeaker = yRHB_Speaker;                                       %Normalized Singal for Speaker [-1 to 1]
#    Data.Model.SignalSpeakerNoise = yRHB_SpeakerHub_Noise;                         %in +- mm Hub
#    Data.Model.SignalRespirationHub = yR_HubMeanFree;                              %Signal Respiration in mm Hub
#    Data.Model.SignalHeartBeatHub = yHB_HubMeanFree;                               %Signal Heartbeat in mm Hub
#    Data.Model.SignalRespirationSpeaker = yR_Speaker;                              %Signal Respiration Normalized for Speaker
#    Data.Model.SignalHeartBeatSpeaker = yHB_Speaker;                               %Signal Heartbeat Normalized for Speaker
#    Data.Model.TimeVectorSimulation = timeVec;                                     %Timevector simulation at f_s
#        
#    Data.Radar.TimeVectorRadar = timeVector_DownSample;                            %Timevector Rador
#    Data.Radar.SignalsRaw = RadarSignal_Raw;
#                           (Channel(8), SamplesPerRamp(128), NoOfRamps(n))
#    Data.Radar.SignalsRawVoltage = RadarSignal_Raw_Voltage;
#    Data.Radar.SignalRespirationHub_DownSample = SignalRespirationHub_DownSample;
#    Data.Radar.SignalHeartbeatHub_DownSample = SignalHeartbeatHub_DownSample;
#    Data.Radar.SignalRHB_Hub_DownSample = SignalRHB_Hub_DownSample;
#    Data.Radar.SignalRHB_Hub_Noise_DownSample = SignalRHB_Hub_Noise_DownSample;
#    Data.Radar.FFToverRamps = FFT_RadarSignal_Raw;

import matplotlib.pyplot as plt
import matplotlib
import tensorflow as tf
import numpy as np
import os
import scipy
from scipy import signal
from scipy.io import loadmat

def splitTrainigData(DataMatrix,NoOfTestData):
    """
    Extract all Channels from the Radar Data Matrix into single Time Series Vectors for one given Datafile. 
    """
   

def plotSpectrogram(DataMatrix,NoOfTestData):
    
    windowSize = 400
    windowOverlap = 200
    window = scipy.signal.get_window('hamming',windowSize,0);
    samplingFrequency = 32
    
    SignalSpeakerNoise = DataMatrix[NoOfTestData]['Data'].Model.SignalSpeakerNoise
    TimeVectorSimulation = DataMatrix[NoOfTestData]['Data'].Model.TimeVectorSimulation
    SignalsRaw = DataMatrix[NoOfTestData]['Data'].Radar.SignalsRaw;
#    test=np.squeeze(SignalsRaw[1,1,:])
#    fig1 = plt.figure()
#    plt.plot(test)
    
    
    f, t, Sxx = scipy.signal.spectrogram(np.squeeze(SignalsRaw[1,1,:]),samplingFrequency,window,windowSize,windowOverlap)
    
    figSpectrogram = plt.figure()
    plt.pcolormesh(t, f, Sxx)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()
    
def plotFrequenzyProfile(DataMatrix,NoOfTestData):
    
    FreqProfHB = DataMatrix[NoOfTestData]['Data'].Setup.FreqProfileHB
    FreqProfR = DataMatrix[NoOfTestData]['Data'].Setup.FreqProfileR
    timeVecSimulation = DataMatrix[NoOfTestData]['Data'].Model.TimeVectorSimulation
    
    figFreqProfileHB = plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(timeVecSimulation,FreqProfHB)
    plt.xlabel('time (s)')
    plt.ylabel('f (Hz)')
    plt.title('Frequenzy Profile of Heartbeat')
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(timeVecSimulation,FreqProfR)
    plt.xlabel('time (s)')
    plt.ylabel('f (Hz)')
    plt.title('Frequenzy Profile of Respiration')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    