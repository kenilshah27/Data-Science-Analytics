# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 17:58:32 2019

@author: Kenil Shah
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv("Filename")

#diving the dataset into independent and dependent variables
X = dataset.iloc[:,:].values
Y = dataset.iloc[:,:].values


#Splitting the data into training set and validation set

from sklearn.cross_validation import train_test_split
train_X,test_X,train_Y,test_Y = train_test_split(X,Y,test_size = 0.2, random_state = 0) 
# test_size to set the size of validation set. random_state to set seed 0 to have the same answer everytime

#Feature Scaling

from sklearn.preprocessing import StandardScaler
scaler_X = StandardScaler()
train_X = scaler_X.fit_transform(train_X) # scale the data for train_X
test_X = scaler_X.transform(test_X) # no need to fit again as we already fit the model above
scaler_Y = StandardScaler()
train_Y = scaler_Y.fit_transform(train_Y) # scale the target variable

#if you predict the answer you will get the scaled value need to convert it into original
y_pred = scaler_Y.inverse_transform("Replace the predict function here")