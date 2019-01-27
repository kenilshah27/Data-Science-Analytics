
dataset <- read.csv('filename')
summary(dataset) # provides a summary of the dataset

str(dataset) # gives us a structure of the dataset

names(dataset) # names of the column

#check for correlation 

#tocheck for two columns

cor(dataset&column1,dataset&column2,method = 'pearson') # other methods can be used

#between all the variables

library(ggpubr)
ggscatter(dataset,x= 'column 1',y = 'column 2',add = 'reg.line',
          conf.int = TRUE , cor.coef = TRUE , cor.method = 'pearson',
          xlab = 'Column 1',ylab = 'Column 2')

#Correltion diagram

library(corrplot)
x <- cor(dataset)
corrplot(x, type="upper", order="hclust")

#Correlation Matrix

corr.test(dataset)

#Correlation of one variable will every variable

library(corrr)
dataset %>% correlate() %>% focus(columnname)

#Do scaling if needed
#Convert into categorical variable if required

#Before creating the model, you might need to bin few values . You can use mapvalue function for that
library(plyr)
dataset$column <- mapvalues(dataset$column,from = c('A','B','C'),to = c('D','E','F')) # to map A to D, B to E,c to F

#Using dendogram to find the optimal number of clusters
dendogram = hcclust(dist(X,method = 'euclidean',method = 'ward.D'))  # first argument is the dissimilarity structure 
                                                  #method ward tries to minimize the within cluser variances

plot(dendogram,main = paste('Dendogram'),xlab = 'Customers', ylab = 'Euclidean distances')

#Model
model <- hclust(dist(X,method = 'euclidean'),method = 'ward.D')
finalmodel <- cutree(model,5)










