
dataset <- read.csv('File Name')

dataset <- dataset[,] # Taking only the variables that can have an impact on the Dependent Variable

#Converting the categorical variables into factor and then into numeric as it is 
#required by our depp learning package

dataset$CategoricalVariable = as.numeric(factor(dataset$CategoricalVariable,labels = c()))

# Do this for all the categorical variables

#Splitting the dataset into train and test

library(caTools)
set.seed(123)

split = sample.split(dataset$DependentVariable,SplitRatio = 0.8)
train = subset(dataset,split == TRUE)
test = subset(dataset,split == FALSE)

#Feature Scaling
train$IndependentVariables = scale(train$IndependentVariables)
test$IndependentVariables = scale(test$IndependentVariables)

  #Do this for all the independent variable

#Fitting ANN to the train dataset

library(h2o) 
#other libraries for neural network.
#neuralnet for regressors only.
#nnet but only one hidden layer.
#deepnet many hidden layers

h2o.init(nthreads = -1) 

# To initialize our h2o model. nthreads is the number of cores to use. 
#-1 indicates all the cores available

model = h2o.deeplearning(y = 'DependentVariableName',training_frame = as.h2o(train),
                         activation = 'Rectifier',
                         hidden = c(number_of_nodes,number_of_nodes),
                         epochs = 100,
                         train_samples_per_iteration = -2)  

# we do as.h2o to conver the traning set into a type the function deeplearning requires
#activation is the activation function we need to use in our ANN
#hidden use to mention the number of hidden layers and the number of nodes in each hidden layer
#Choose hidden level nodes that is an average of the input and the output nodes
#epochs is the number of times we want to pass the entire training set through the ANN
# train_samples_per_iteration is the number of observations after which we want to update our weights
#-2 signifies auto-tuning and gives the model to select the best

#Predicting the test results
pred = h2o.predict(model,newdata = as.h2o(test$AllIndependentVariables))
pred_y <- ifelse(pred > 0.5,1,0)
pred_y <- as.vector(pred_y) # Converting the h2o datatype into vector

#Confusion Matrix
cm = table(test$DependentVariable,pred_y)

h2o.shutdown()

  

