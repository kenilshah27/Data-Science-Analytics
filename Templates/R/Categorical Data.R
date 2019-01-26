
#One Code Encoder is not required in R as the functions takes care of that

dataset <- read.csv('Dataset name')

dataset$columnname <- as.factor(datase$columnname,levels = c('a','b','c'),labels = c(0,1,2)) # we can assign a label to each level