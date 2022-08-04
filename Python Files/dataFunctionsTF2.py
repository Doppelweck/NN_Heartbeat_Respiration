#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 13:18:24 2020

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
#
#    Data.Radar.SignalRespiration_Speaker_DownSample;  Normalized Label
#    Data.Radar.SignalRHB_Speaker_Noise_DownSample;  Speaker Signal Normalized 

import matplotlib.pyplot as plt
import matplotlib
import tensorflow as tf
import numpy as np
import os
import scipy
from scipy import signal
from scipy.io import loadmat
import random

def openTraingsDataFiles(dataDir,OnlyForTesting):
    """
    Opens all Matlab Files 
    """
    TraningDataMatrixes = []
    NoOfTrainingDataSets = 0
    if OnlyForTesting: #Only Read 4 Files max
        if len(os.listdir(dataDir)) <=4:
        
            for file in os.listdir(dataDir):
                if file.endswith('.mat'): #Load only if .mat file (.ds_Stroe file causes error)
                    TraningDataMatrixes.append(scipy.io.loadmat(dataDir+file, struct_as_record=False, squeeze_me = True)['Data'])
                    print("load Data from file "+file)
                    NoOfTrainingDataSets = NoOfTrainingDataSets+1
            
            return TraningDataMatrixes,NoOfTrainingDataSets
        else: 
            List = os.listdir(dataDir);
            List = List[0:6]
            for file in List:
                if file.endswith('.mat') and 'TrainingData' in file : #Load only if .mat file (.ds_Stroe file causes error)
                    TraningDataMatrixes.append(scipy.io.loadmat(dataDir+file, struct_as_record=False, squeeze_me = True)['Data'])
                    print("load Data from file "+file)
                    NoOfTrainingDataSets = NoOfTrainingDataSets+1
            
            return TraningDataMatrixes,NoOfTrainingDataSets
        
    else: #Read all Files 
        
        for file in os.listdir(dataDir):
            if file.endswith('.mat'): #Load only if .mat file (.ds_Stroe file causes error)
                TraningDataMatrixes.append(scipy.io.loadmat(dataDir+file, struct_as_record=False, squeeze_me = True)['Data'])
                print("load Data from file "+file)
                NoOfTrainingDataSets = NoOfTrainingDataSets+1
        
        return TraningDataMatrixes,NoOfTrainingDataSets   
    
def splitDataInto1DVectors(DataMatrix,NoInputCells,NoOutputCells):
    
    NoOfTrainingDataSets=DataMatrix.shape[0];
    """
    Open one Matlab File and extract all Informations
    """

def getAndStackDataFromFiles(TrainingDataMatrix,NoInputCells,NoOutputCells,NoOfDataSets,OffsetDatasets):
    """
    Open one Matlab File and extract all Informations
    """
    TotalNoOfSamples = 0;
    
    SIGNAL_P = np.empty((0,NoInputCells,1)); #Phase gemessen mit Radar (1 Channel only)
    SIGNAL_S = np.empty((0,NoInputCells,1)); #Signal Speaker (ideal synthetisch)
    SIGNAL_R = np.empty((0,NoOutputCells,1)); #Respiration
    SIGNAL_H = np.empty((0,NoOutputCells,1)); #Heartbeat
    
    for i in range(OffsetDatasets,NoOfDataSets): #Intern i_max = NoOfDataSets-1 
        print(i)
        DataMatrix = TrainingDataMatrix[i]
        
        #Split Every Traningset in 1D Examples according to the Number of Input/Output Cells
        Signal_R,  Signal_H,  Signal_S,  Signal_P, NoOfSamples = reshapeDataIntoTrainLabelPairs(DataMatrix,NoInputCells,NoOutputCells);

        SIGNAL_P = np.vstack((SIGNAL_P, Signal_P)) #Phase gemessen mit Radar (1 Channel only)
        SIGNAL_S = np.vstack((SIGNAL_S, Signal_S)) #Signal auf Lautsprecher (Signal_R + Signal_H)
        SIGNAL_H = np.vstack((SIGNAL_H, Signal_H)) #Signal Heartbeat (Label)
        SIGNAL_R = np.vstack((SIGNAL_R, Signal_R)) #Signal Respiration (Label)
        TotalNoOfSamples = TotalNoOfSamples + NoOfSamples;    
        
    return     SIGNAL_P, SIGNAL_S, SIGNAL_H, SIGNAL_R, TotalNoOfSamples
 
    
def reshapeDataIntoTrainLabelPairs(DataMatrix,NoInputCells,NoOutputCells):
    """
    Open one Matlab File and extract all Informations
    """
    Channel = 1; #Selected Radar Channel

    
    RespirationSignalTrue = np.array(DataMatrix.Radar.SignalRespiration_Speaker_DownSample);
    RespirationSignalTrue = RespirationSignalTrue.reshape((len(RespirationSignalTrue), 1));
    
    HeartbeatSignalTrue = np.array(DataMatrix.Radar.SignalHeartbeat_Speaker_DownSample);
    HeartbeatSignalTrue = HeartbeatSignalTrue.reshape((len(HeartbeatSignalTrue), 1));
    
    SpeakerSignal = np.array(DataMatrix.Radar.SignalRHB_Speaker_Noise_DownSample);
    SpeakerSignal = SpeakerSignal.reshape((len(SpeakerSignal), 1));
    
    PhaseUnwrap = np.array(DataMatrix.Radar.PhaseUnwrap[Channel-1,:])
    PhaseUnwrap = PhaseUnwrap.reshape((len(PhaseUnwrap), 1))*(-1)
    
    Dataset = np.hstack((RespirationSignalTrue,HeartbeatSignalTrue,PhaseUnwrap,SpeakerSignal))
    
    if NoInputCells == NoOutputCells: #Same lenght for Input Signal and each Output Signal
    # split Dataset sequence into samples for Training
        yR, yH, yP, yS= list(), list(), list(), list()
        for i in range(len(RespirationSignalTrue)): #Split Input Sequence
            # find end of Seqeunce
            end_ix = i+ NoInputCells #No of Input cells = No od Output cells
            if end_ix > len(RespirationSignalTrue): #End of Sequence
                break
            seq_R, seq_HB , seq_P, seq_S = Dataset[i:end_ix, 0:1], Dataset[i:end_ix, 1:2], Dataset[i:end_ix, 2:3], Dataset[i:end_ix, 3:4];
            
            yR.append(seq_R) #Respiration
            yH.append(seq_HB) #Heartbeat
            yP.append(seq_P) #Phase Radar
            yS.append(seq_S) #Signal Speaker
          
        Signal_R = np.array(yR); #Signal Respiration (Label)
        Signal_H = np.array(yH); #Signal Heartbeat (Label)
        Signal_P = np.array(yP); #Phase gemessen mit Radar (1 Channel only)
        Signal_S = np.array(yS); #Signal auf Lautsprecher (Signal_R + Signal_H)
#
        NoOfSamples = Signal_R.shape[0]
        
        return  Signal_R, Signal_H, Signal_S, Signal_P, NoOfSamples
    
    elif NoInputCells > NoOutputCells:
        yDelay = int(NoInputCells-NoOutputCells);
        yR, yH, yP, yS= list(), list(), list(), list()
        for i in range(len(RespirationSignalTrue)): #Split Input Sequence
            # find end of Seqeunce
            end_ix = i+ NoInputCells #No of Input cells = No od Output cells
            if end_ix > len(RespirationSignalTrue): #End of Sequence
                break
            seq_R, seq_HB , seq_P, seq_S = Dataset[i+yDelay:end_ix, 0:1], Dataset[i+yDelay:end_ix, 1:2], Dataset[i:end_ix, 2:3], Dataset[i:end_ix, 3:4];
            
            yR.append(seq_R) #Respiration
            yH.append(seq_HB) #Heartbeat
            yP.append(seq_P) #Phase Radar
            yS.append(seq_S) #Signal Speaker
          
        Signal_R = np.array(yR); #Signal Respiration (Label)
        Signal_H = np.array(yH); #Signal Heartbeat (Label)
        Signal_P = np.array(yP); #Phase gemessen mit Radar (1 Channel only)
        Signal_S = np.array(yS); #Signal auf Lautsprecher (Signal_R + Signal_H)
#
        NoOfSamples = Signal_R.shape[0]
        
        return  Signal_R, Signal_H, Signal_S, Signal_P, NoOfSamples
    else:
        return False
            
    






    
    