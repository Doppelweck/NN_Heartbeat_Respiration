# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 07:09:44 2019

@author: mantsven
"""
import scipy.io as sio

# Mat file als python object einlesen, dadurch kann man es ähndlich verwenden wie in Matlab. 
# Die oberste Struktur ist ein dictionary also kannst (musst) dann die gleichen Feldnamen benutzen wie in dem Mat file also Data.
mat_file = sio.loadmat('TrainingData_Tsim_240s_2019-5-29_13-8.mat', struct_as_record=False, squeeze_me = True)

# Hier kannst du jetzt auf die darunterliegenden Structs zugreifen wie bei Matlab
HB_GainSmooth = mat_file['Data'].Setup.HB_GainSmooth
HB_max_Hz = mat_file['Data'].Setup.HB_max_Hz
HB_min_Hz = mat_file['Data'].Setup.HB_min_Hz
HB_Init = mat_file['Data'].Setup.HB_Init
HB_Varianz = mat_file['Data'].Setup.HB_Varianz
'''
.
.
.
.
.
'''
TimeVectorRadar = mat_file['Data'].Radar.TimeVectorRadar
SignalsRaw = mat_file['Data'].Radar.SignalsRaw
SignalsRawVoltage = mat_file['Data'].Radar.SignalsRawVoltage