# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 02:13:10 2019

@author: Kenil Shah
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#Importing Dataset
dataset = pd.read_csv("File Name")

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

#Finding optimal number of clusters using elbow method
from sklearn.cluster import KMeans
wcss = []

for i in range(1,11):
    model = KMeans(n_clusters = i,init = 'k-means++',max_iter = 300, n_init = 10,random_state = 0) 
    #n_clusters is the number of clusters we want. init is the initial choice we want to make about centroids.
    #max_iter is the numbers of iterations you want . n_init is the number of times this algorithm will be run with different centroids
    model.fit(dataset)
    wcss = wcss.append(kmeans.inertia_)

#Plotting the Elbow graph
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number Of CLusters')
plt.ylabel('The WCSS')
plt.show()

#The Final Kmeans clustering
finalmodel = KMeans(n_clusters = 5,init = 'k-means++',max_iter = 300, n_init = 10,random_state = 0) 
predictedclusters = finalmodel.fit_predict(X)


    
