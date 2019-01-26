# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 18:15:03 2019

@author: Kenil Shah
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv("Filename")

#diving the dataset into independent and dependent variables
X = dataset.iloc[:,:].values
Y = dataset.iloc[:,:].values

#Encoding Categorical Variable

from sklearn.preprocessing import LabelEncoder,OneHotEncoder

LE_X = LabelEncoder()  # replace the values with number for x
X = LE_X.fit_transform(X)
OHE_X = OneHotEncoder(categorical_variables = [column numbers]) # convert the number into individual columns
X = OHE_X.fit_transform(X).toarray()
LE_Y = LabelEncoder()  # replace the values with number for y
Y = LE_Y.fit_transform(Y) 
