
dataset <- read.csv('File Name')

#Implementing UCB

number_of_selections = integer(d) #Number of times a particular entity was selected
sum_of_rewards = integer(d)       #Number of times we got a positive result for an entity that was selected

total_reward = 0 # To store total rewards 

entity_selected = integer() #Store the entity selected after each round
for(n in 1:N) # N is Number of rounds we need for to come up with a solution of the algorithm
{
  max_upper_bound = 0
  entity = 0
  
  for(i in 1:d)# d is the number of entity
  {
    if(number_of_selections[i] > 0) 
    {
      average_reward = sum_of_rewards[i]/number_of_selections[i]
      delta_i = sqrt(3/2*log(n)/number_of_selections[i])
      upper_bound = average_reward + delta_i         
    }
    else
    {
      max_upper_bound = 1e400    
    }
    if(upper_bound > max_upper_bound)  # If we get a better upper bound we replace it
    {
      max_upper_bound = upper_bound
      entity = i  #Selecting the entity with max upper bound
    }
  }
  entity_selected = append(entity_selected,entity)
  number_of_selections[entity] = number_of_selections[entity] + 1
  reward = dataset[n,entity]
  if(reward == 1)
  {
    sum_of_rewards[entity] = sum_of_rewards[entity] + 1
  }
  else
  {
    sum_of_rewards[entity] = sum_of_rewards[entity] + 0
  }
  total_reward = total_reward + reward
}


#Visualising the histogram

hist(number_of_selections)


