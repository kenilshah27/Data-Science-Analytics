# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 02:04:05 2019

@author: Kenil Shah
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

#Importing the dataset
dataset = pd.read_csv('File Name')

#Implementing Thompson Sampling 

#Here entity is machine we want to predict 
#Define d the number of entities and N the number of rounds you want

number_of_reward0 = [0]* d # vector of size d. 
#number_of_reward0 is use to select how many times entity got reward up to round n

number_of_reward1 = [0] * d # use to store the value of rewards till i for each entity
#number_of_reward1 is use to select how many times entity got reward up to round n


entity_selected = [] # ads selected after each round
total_reward = 0 # to calculate total reward after all the rounds

for n in range(0,N):# N is the total number of rounds
    entity = 0    
    max_random = 0 # highest upper_bound among all entities
    for i in range(0,d):# d is the number of 
        random_beta =  random.betavariate(number_of_reward1[entity]+1,number_of_reward0[entity]+1)# correspond to different random draws taken from beta distribution for the entity                
                
        #Selecting the entity with maximum upper bound 
       if random_beta > max_random:
            max_random = random_beta
            entity = i # take up the entity number if we find a higher upper bound then max 
    entity_selected.append(entity)
    reward = dataset[n][entity]
    total_reward = total_reward + reward    # updating total reward
    if reward == 0:
        number_of_reward0[entity] = number_of_reward)[entity] + 1
    else:
        number_of_reward1[entity] = number_of_reward1[entity] + 1 # updating the rewards count for each entity

    
#Visualising the entities and results

plt.hist(entity_selected)
plt.title('Histogram of Entity Selection')
plt.xlabel('Entity')
plt.ylabel('Frequency of entity selection')
plt.show()


