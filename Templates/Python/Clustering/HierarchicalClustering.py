# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 13:58:36 2019

@author: Kenil Shah
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#Importing Dataset
dataset = pd.read_csv("Salary_Data.csv")

#Gives us a summary of the dataset
dataset.describe()

#to give column names
dataset.columns = []

#Diving the dataset into independent and dependent variables
X = dataset.iloc[:,:].values
Y = dataset.iloc[:,:].values

# Column names
list(dataset)

#Correlation Matrix
corr_matrix = dataset.corr()
f, ax = plt.subplots(figsize=(16,10))
ax = sns.heatmap(corr_matrix)
ax.set_title("correlation between all features")
figure = ax.get_figure()    

#Do feature scaling if required.
#Convert into categorical variable if required

#Using dendogram to find optimal number of cluster
import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(X,method = 'ward'))   # ward try to minimize the within cluster variance
plt.title('Dendogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distance')
plt.show()

#Model
from sklearn.cluster import AgglomerativeClustering
model = AgglomerativeClustering(n_clusters = 5,affinity = 'euclidean',linkage = 'ward') # n_clusters are the number of clusters
                                                                     # affinity tells about the type of distance we want to use
pred = model.fit_predict(X)

