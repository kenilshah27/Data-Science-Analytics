# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 23:51:16 2019

@author: Kenil Shah
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

#Importing the dataset
dataset = pd.read_csv('File Name')

#Implementing Upper Confidence Bound

#Here entity is machine we want to predict 
#Define d the number of entities and N the number of rounds you want

number_of_selection = [0]* d # vector of size d. 
#number_of_selection is use to store how many times each entity in question is selected

sums_of_rewards = [0] * d # use to store the value of rewards till i for each entity

entity_selected = [] # ads selected after each round
total_reward = 0 # to calculate total reward after all the rounds
for n in range(0,N):# N is the total number of rounds
    entity = 0    
    max_upper_bound = 0 # highest upper_bound among all entities
    for i in range(0,d):# d is the number of 
        if(number_of_selection[i] > 0): #this is done because initally we do not have much idea about the 
                                        #entities and hence we are selction each of them initially
            average_reward = sums_of_rewards[i]/number_of_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n + 1)/number_of_selection[i]) # use to find the upper bound
            upper_bound = average_reward + delta_i
       else:
            upper_bound = 1e200   # so that we are able to select each of the entity intially
             
             #Selecting the entity with maximum upper bound 
       if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            entity = i # take up the entity number if we find a higher upper bound then max 
    entity_selected.append(entity)
    number_of_selection[entity] = number_of_selection[entity] + 1
    reward = dataset[n][entity]
    total_reward = total_reward + reward    # updating total reward
    sum_of_reward[entity] = sum_of_rewards[entity] + reward # Adding reward to the particular entity sum_of_reward
    
#Visualising the entities and results

plt.hist(entity_selected)
plt.title('Histogram of Entity Selection')
plt.xlabel('Entity')
plt.ylabel('Frequency of entity selection')
plt.show()


    
       
        