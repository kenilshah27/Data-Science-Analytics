
dataset <- read.csv('Dataset Name')

summary(dataset) # provides a summary of the dataset

str(dataset) # gives us a structure of the dataset

names(dataset) # names of the column

library(caTools)

datasplit <- sample.split(dataset$DependentVariable,SplitRatio = 0.8)
train <- subset(dataset,datasplit = TRUE)
test <- sbuset(dataset,datasplit = FALSE)

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
train$column <- mapvalues(train$column,from = c('A','B','C'),to = c('D','E','F')) # to map A to D, B to E,c to F
test$column <- mapvalues(test$column,from = c('A','B','C'),to = c('D','E','F')) # we need to map values in test as well

#Model
library(rpart)
model <- rpart(DependentVariable ~.,data = train,control = rpart.control(minsplit = 1))
                                                              # set various controls using rpart.control
?rpart.control # to check for all the control parameters

#Predicting values

predictedvalues <- predict(model,test)

#coefficients of the independent variables
model$coefficients

#Summary of the model will have the coefficients value, mean square error, adjusted r square and many other metrics

summary(model)

#Residual plot to check how linear the model is
plot(model$fitted.values,model$residuals)

#Visualing Decision Tree
library(rpart.plot)
rpart.plot(tree, box.palette="RdBu", shadow.col="gray", nn=TRUE)

#Saving the output file

write.csv(predictedvalues,'Filename',sep = '\t')
