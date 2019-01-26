
#Import data

dataset <- read.csv("Dataset Name")


#Summary check to see if any column has missing value
summary(dataset)

# Gives true or false whether a that value is NA or not
is.na(dataset)

#Gives position of NAs
which(is.na((x)))


#Replacing the missing values with mean,mode or median
dataset$'Replace column name here' <- ifelse(is.na(dataset$'Replace column name here'),
                                             FUN = function(x) mean(x,na.rm = TRUE),
                                             dataset$'Replace column name here')

#We use FUN in the method to be used for replacing to show that we can replace the mean and use any other function to
#replace the missing value in the dataset

#Predicition
#1.KNN neighbours

library(DMwR)
knnoutput <- knnImputation(dataset)

#Knn may not be useful when the missing value come from a categorical variable and so rpart and mice is used.
#2.rpart

library(rpart)
class_mod <- rpart('column with missing value'~.-'target variable',
                   data = dataset('minus the column with missing values'),
                   method = class,na.action = na.omit)  # for a categorical variable

anova_mod <- rpart('column with missing value'~.-'target variable',
                   data = dataset('minus the column with missing values'),
                   method = anova,na.action = na.omit)  # for a numeric variable

pred <- predict(class_mod,dataset('with the missing value'))  # dataset(is.na(datasetwithmissingvalue))
pred <- predict(anova_mod,dataset('with the missing value'))  # dataset(is.na(datasetwithmissingvalue))

#3.mice
library(mice)

micemodel <- mice(dataset,method = 'rf') # perform mice imputation based on random forest
Output <- complete(micemodel)


