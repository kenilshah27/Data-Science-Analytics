# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 22:02:44 2019

@author: Kenil Shah
"""

import pandas as pd
import numpy as np
import matplotlib as plt
import nltk
nltk.download('stopwords')  # Downloading a list of all the stopwords
from nltk.corpus import stopwords
import re 
from nltk.stem import PorterStemmer

#Importing the dataset
dataset = pd.read_csv('File Name', delimiter = '\t', quoting = 3) # Take a tab seperated file as sentences may have commas in them 
                                                            #quoting - 3 ignores double quotes in the sentences

corpus = [] # Contains all the clean text which we have created

for i in range(len(dataset)):
    #Cleaning the texts from the dataset
    review = re.sub('[^a-zA-Z]', ' ', dataset['Text column'][i]) # Here text column is the column containing text for your dataset 
    #Remove numbers and puntuation and replace those position with a space(second argument)
                                         
    review = review.lower() # Changes all the character to lowercase
    
    #Remove the Non significant words like the,and,articles,prepositions etc
    
    review = review.split()  # review will become a list of all the words
    ps = PorterStemmer() #to do stemming on our review variable
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))] #Removing all the words that are part of stopwords package
                                                                # we convert it into a set because functions are faster on set then on lists
    #ps.stem does Stemming taking the root of the word i.e. words like loved,loving,love will all be mapped to love
    
    #Converting the review back into string
    review = ' '.join(review)  
    corpus.append(review)  # To add our cleaned text to the corpus list

#Creating Bag of Words Model . Take all the words from the reviews and create a column for each words.
#We will create a table with all reviews as row and words as column and the value will be 1 if the word is present in review
#We actually created a Sparse Matrix
#Using the process of tokenization

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)  # has parameter stop_words that can be used to remove stopwords and lowercase to change to lowercase
                                           # token pattern to give it a list of letters and symbol we want to keep in the reviews
                                           # max_features to set a limit on the number of features to remove less frequent or rare words
                                           # We can also reduce the max features using the dimensionality reduction methods
 
X = cv.fit_transform(corpus).toarray()  # Creating the matrix
                                        # to array() to convert it into a matrix

Y = dataset['Liked'].values   # or dataset.iloc[:,1].values
                              # .values to convert it into an array from Series

# Choose a classification model and decide the best model
# Most common models used are Naive Bayes, Decision Tree and Random Forest

#Splitting the dataset into training and test dataset
from sklearn.cross_validation import train_test_split
train_X,test_X,train_Y,test_Y = train_test_split(X, Y, test_size = 0.20, random_state = 0)

#Applying Feature Scaling if required
 
#Naive Bayes Model 
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(train_X,train_Y)

#Predicting
pred_Y = model.predict(test_X)
    
#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_Y,pred_Y)

#Accuracy
sum(np.diag(cm))*100/200
