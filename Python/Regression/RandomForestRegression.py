# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 19:30:51 2019

@author: Kenil Shah
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators = 10,random_state = 0)
# n_estimators sets the number of trees we want to create. We can also change other parameters for better results 

model.fit(train_X,train_Y)

#Predicting the values
y_pred = model.predict(test_X)

#If feature scaling is used on Y remember to inverse transform the y_pred variable

# Gives the coefficient values
model.coef_

#Get the intercept
model.intercept_

from sklearn.metrics import mean_squared_error,r2_score

#Mean Square Error
mse = mean_squared_error(test_Y,y_pred)

#Adjusted R Square
r2 = r2_score(test_Y,y_pred)

#Residual Plot to check how linear the model is
plt.scatter(model.predict(train_X),model.predict(train_X) - train_Y, c = 'b')
plt.hlines(y = 0,xmin = min(model.predict(train_X)),xmax = max(model.predict(train_X)))

#Saving the output file
y_pred = pd.Series(y_pred, name = 'target variable name')
submission = pd.DataFrame(data='The unique indentity of test dataset')
submission['Targer variable name'] = pred_y
submission.to_csv('submissionPath', sep='\t', index = False)




