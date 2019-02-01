
dataset <- read.csv('File Name')

#Implementing Thompson Sampling

number_of_reward1 = integer(d) #Number of times the ad was rewarded with 1
number_of_reward0 = integer(d) #Number of times the ad was rewarded with 0

total_reward = 0 # To store total rewards 

entity_selected = integer() #Store the entity selected after each round
for(n in 1:N) # N is Number of rounds we need for to come up with a solution of the algorithm
{
  max_random = 0
  entity = 0
  
  for(i in 1:d)# d is the number of entity
  {
    random_beta = rbeta(n = 1, 
                        number_of_reward1[i] + 1,
                        number_of_reward0[i] + 1)  # Generating a random number from beta distribution
    if(random_beta > max_random)  # If we get a better random number we replace it
    {
      max_random = random_beta
      entity = i  #Selecting the entity with max random number that we generate
    }
  }
  entity_selected = append(entity_selected,entity)
  reward = dataset[n,entity]
  if(reward == 1)
  {
    number_of_reward1[entity] = number_of_reward1[entity] + 1
  }
  else
  {
    number_of_reward0[entity] = number_of_reward0[entity] + 0
  }
  total_reward = total_reward + reward
}


#Visualising the histogram

hist(number_of_selections)


