# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 20:51:41 2019

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

#Splitting the data into training set and validation set
from sklearn.cross_validation import train_test_split
train_X,test_X,train_Y,test_Y = train_test_split(X,Y,test_size = 0.2, random_state = 0) 

#Correlation Matrix
corr_matrix = train_X.corr()
f, ax = plt.subplots(figsize=(16,10))
ax = sns.heatmap(corr_matrix)
ax.set_title("correlation between all features")
figure = ax.get_figure()    

#Do feature scaling if required.
#Convert into categorical variable if required

#Model
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion = 'entropy',random_state = 0) # entropy is for information gain 
#You can vary the arguments to get better results

model.fit(train_X,train_Y)

# Predicting the Test set results
pred_Y = model.predict(test_X)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_Y, pred_Y)

from sklearn.metrics import classification_report,precision_recall_curve,roc_curve,
#There are a lot of other metrics that can be used depending on our requirement

#Generated a classification reports that has precision rate,recall rate and f1-score. Look into help for more details.
classification_report(test_Y,pred_Y,target_names = target_names)

#Learn more about ROC AUC Curve and Precision curve on this website
#https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/

#Precision - Recall Curve
precision,recall,threshold = precision_recall_curve(test_Y,pred_Y)
plt.plot([0,1],[0.5,0.5],linestyle = '--')
plt.plot(recall,precision,color = 'b',marker = '.')
plt.show()

#ROC - AUC Curve
plt.plot([0, 1], [0, 1], linestyle='--')
plt.plot(fpr, tpr, marker='.')
plt.show()
