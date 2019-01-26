
dataset <- read.csv('dataset name')

library(caTools)

set.seed(273) # any random number

datasplit <- sample.split(dataset$DependentVariable,SplitRatio = 0.8) #size of training dataset

train <- subset(dataset,datasplit = TRUE)
test <- sbuset(dataset,datasplit = FALSE)

#Feature Scaling

#Z-score Normalization
train <- scale(train)
test <- scale(test)  

#Normalization based on Range

normalize <- function(x) {
  return ((x - min(x))/max(x) - min(x))
}

dataset$column <- as.data.frame(lapply(dataset$column,normalize)) # since lapply would result a list


