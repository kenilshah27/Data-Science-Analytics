# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 18:11:22 2019

@author: Kenil Shah
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

X = dataset.iloc[:].values
Y = dataset.iloc[:].values

from sklearn.preprocessing import Imputer
imputer = Imputer()
imputer = Imputer(missing_values = 'NaN', axis = 0 , strategy = 'strategy' ) # axis 0 for column. Replace the strategy you want to use in place of strategy
X = imputer.fit_transform(X) # can replace X with whatever columns you want to fill the mising values in 