# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 02:13:10 2019

@author: Kenil Shah
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#Importing Dataset
dataset = pd.read_csv("File Name")

transaction = []

# for each row in the table
for i in range(0,n): # N is the number of rows in our dataset
    transaction.append([str(dataset.values[i,j]) for j in range(0,20)])  # convert the dataset into a list of list
    
#Training Apriori on the dataset

from apyori import apriori
rules = apriori(transactions,min_support = , min_confidence = , min_lift = , min_length = 2)  # The constraints of our model

#Visualising rules
result = list(rules) # Already sorted with respect to support,confidence,lift . sorted by relevance criterion

# Look at the results to find the top rules recommended by the model
