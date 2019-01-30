
#It is like the Apriori model but is very simplified

#It is well when you do not want to change much parameters and also want an idea of the rules that can be used


#Data Processing

dataset = read.csv('File Name')

#Transform dataset into a sparse matrix

library(arules)

dataset = read.transactions('Fine Name',sep = ",",rm.duplicates = TRUE)

#Get info about the created sparse matrix
summary(dataset)

itemFrequencyPlot(dataset,topN = 100) # where topN is the number of top items you want

#Training Eclat on the dataset 
rules = eclat(data = dataset,parameter = list(support = ,minlen = 2)) 
              # choice of minimum support and confidence support
              # want atleast two items in the basket so set minlen to 2

# give the support(default = 0.1) and confidence(default = 0.8) you need depending on your business goal

#Visualising
inspect(sort(rules,by = 'support')[1:10] #Gives 10 first rules found by our apriori model

