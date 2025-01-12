# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 21:15:15 2024

@author: robur
"""

import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils import data


class SinWaveDataset():
    
    def __init__(self,
                 data_path = './data/sine_waves.json',
                 normalize = False):
        
        self.data_path = data_path
        self.normalize= normalize
        
    def loadData(self):

        if (not os.path.isfile(self.data_path)):
            print("Sinusoidal Waves dataset can not be found")
            
        else:
            with open(self.data_path) as f:
                df = json.load(f)
                df = np.array(df)
                df = np.expand_dims(df, axis=2) #add new dimension to array to be compatible with model architecture
                print('Input data loaded successfully')
                
        # Train/Test/Validation split
        x_train, x_test = train_test_split(df, test_size=0.3, shuffle=True, random_state=34)
        print(f'Train Set is of shape {x_train.shape}')
        print(f'Test Set is of shape {x_test.shape}')
        
        if self.normalize:
            x_train = self.normalization(x_train)
            x_test = self.normalization(x_test)
            print('Data is normalized')
                 
        return x_train, x_test
    
    def _normalize(self, epoch):
        """ A helper method for the normalization method.
            Returns
                result: a normalized epoch
        """
        e = 1e-10
        result = (epoch - epoch.mean(axis=0)) / ((np.sqrt(epoch.var(axis=0)))+e)
        return result
    
    def _min_max_normalize(self, epoch):
        
        result = (epoch - min(epoch)) / (max(epoch) - min(epoch))
        return result
    
    def normalization(self, df):
        """ Normalizes each epoch e s.t mean(e) = 0 and var(e) = 1
            Args:
                epochs - Numpy structure of epochs
            Returns:
                epochs_n - mne data structure of normalized epochs (mean=0, var=1)
        """
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                df[i,j,0,:] = self._normalize(df[i,j,0,:])
                #df[i,j,0,:] = self._min_max_normalize(df[i,j,0,:])
    
        return df




