# -*- coding: utf-8 -*-
"""
Spyder Editor

@Author : Kenil Shah
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Data Processing

dataset = pd.read_csv('File name')
X = dataset.iloc[:,:].values  #Independent Variables
Y = dataset.iloc[:,:].values  #Dependent Variable

#Encoding the categorical data in our dataset
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X1 = LabelEncoder()
X[:, 1] = labelencoder_X1.fit_transform(X[:, 1])
labelencoder_X2 = LabelEncoder()
X[:,2] = labelencoder_X2.fit_transform(X[:,2])

#Keep on creating the LabelEncoders and transforming the column into categorical values for each of the categorical variables

onehotencoder = OneHotEncoder(categorical_features = [1]) # Do this for all the categorical variables with more than 2 labels
X = onehotencoder.fit_transform(X).toarray()

X = X[:,1:] # Avoiding the Dummy trap from all the categorical variables

#Splitting the dataset
from sklearn.model_selection import train_test_split
train_X,test_X,train_Y,test_Y = train_test_split(X,Y,test_size = 0.2,random_state = 0) #Training 80% Test 20%

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
train_X = sc.fit_transform(train_X)
test_X = sc.fit_transform(test_X)

"""ANN starts here
Make sure you have all the required librarires downloaded
Importing Keras libraries"""

import keras
from keras.models import Sequential  #intialize neural network
from keras.layers import Dense       #build on the layers neural network

#ANN with sequence of layers
model = Sequential() # Initializing ANN by defining it as a sequence of layers

#Adding the input layer and the first hidden layer
model.add(Dense(output_dim = 6,init = 'Uniform',activation = 'relu',input_dim = 11))

"""Choose hidden level nodes that is an average of the input and the output nodes
output_dim is to set the number of nodes in the hidden layer
init is use to initialize the weights  
activation is the activation function for the hidden layer. relu is for rectifier function. sigmoid is for sigmoid function 
input_dim is the number of input variables or the number of independent variables
from the next hidden layers we need not mention the input_dim variables as the model knows what to expect"""

#Adding the second hidden layer
model.add(Dense(output_dim = 6,init = 'Uniform',activation = 'relu'))

#Adding the final or the Output layer
model.add(Dense(output_dim = 1,init = 'Uniform',activation = 'sigmoid'))

""" If you have more than two levels in the output then change the output_dim to the number of levels you have
Change the activation function to softmax which is similar to sigmoid but for variable with more than two levels """

#Applying stocastic gradient descent of the network.Compiling the ANN 

model.compile(optimizer = 'adam', loss = 'binary_crossentropy',metrics = ['accuracy'])

""" optimizer is the algorithm we want to use to find the weights . Here we are using 'adam' for Stocastic Gradient Descent
loss is the loss function with the adam algorithm  which we need to optimize the Gradient Descent to find the optimal weights 
binary_crossentropy is the function for variable with two levels.
categorical_crossentropy for variables with more than two levels.
metrics  is the criterion we use to evaluate our model """ 

#Fitting the ANN to the dataset
model.fit(train_X,train_Y,batch_size = 10,np_epoch = 100)
""" batch size is the number of observations after which we want to update our weights
nb_epoch is the number of times we want to pass the entire training set through the ANN """ 

#Predicting the test results
pred = model.predict(test_X)
pred = (pred > 0.5)

#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_Y,pred)
