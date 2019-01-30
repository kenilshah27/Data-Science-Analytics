
#Apriori

#Data Processing

dataset = read.csv('File Name')

#Transform dataset into a sparse matrix

library(arules)

dataset = read.transactions('Fine Name',sep = ",",rm.duplicates = TRUE)

#Get info about the created sparse matrix
summary(dataset)

itemFrequencyPlot(dataset,topN = 100) # where topN is the number of top items you want

#Training Apriori on the dataset 
rules = apriori(data = dataset,parameter = list(support = ,confidence = )) # choice of minimum support and confidence support

# give the support(default = 0.1) and confidence(default = 0.8) you need depending on your business goal

#Visualising
inspect(rules[1:10]) #Gives 10 first rules found by our apriori model

inspect(sort(rules,by = 'lift')[1:10]) # sort the rules with respect ot its lift and then give the first 10 rules with
                                        #highest lift

# We can look at the rules and then decide whether our intial parameters are good or not and can rerun the model again

