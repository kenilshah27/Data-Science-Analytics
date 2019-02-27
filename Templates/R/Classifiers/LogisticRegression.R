
dataset <- read.csv('filename')
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

model <- glm(DependentVariable~.,family = 'binomial',dataset = train)

#Predicting Values

predictedvalues <- predict(model,type = 'response',test)
pred_Y <- ifelse(predictedvalues > 0.5,1,0)

#coefficients of the independent variables
model$coefficients

#Summary of the model will have the coefficients value, mean square error, adjusted r square and many other metrics
summary(model)

#Metrics to check the model performance

#Confusion Matrix
cm = table(test$DependentVariable,pred_Y)

#Learn more about precision = recall curve and roc curve here
#https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/

#ROC Curve
library(ROCR)

pred <- prediction(pred_Y,test$DependentVariable)
perf <- performance(pred,'tpr','fpr' )
plot(perf)

#using pROC
library(pROC)
roc_rounded <- roc(test_set$DependentVariable, pred_Y)
plot(roc_rounded, print.auc=TRUE)

#PR and ROC curve using PRROC
library(PRROC)
fg <- predictedvalues[test$DependentVariable == 1]
bg <- predictedvalues[test$DependentVariable == 0]

# ROC Curve    
roc <- roc.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
plot(roc)

# PR Curve
pr <- pr.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
plot(pr)

