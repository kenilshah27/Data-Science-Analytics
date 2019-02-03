
#Importing the dataset
dataset_original <-read.delim('File Name',quote = '',stringsAsFactors = FALSE) #taking a tsv file
dataset = dataset_original
# Take a tab seperated file as sentences may have commas in them
# quoting - 3 ignores double quotes in the sentences
# stringAsFactors identify each string as factor


summary(dataset) 
 
# Cleaning the dataset
#install.packages('tm') #Package required
#install.packages('SnowballC') # PAckage required for stopwords

library(tm)
library(SnowballC)

corpus = VCorpus(VectorSource(x = dataset$TextColumn)) 
# Contains all the clean text which we have created 
#Text column is the column that has the text in your dataset

corpus = tm_map(x = corpus,content_transformer(FUN = tolower)) #Transform the corpus words into lower cases

corpus = tm_map(x = corpus,removeNumbers) #Remove all the numbers from the text
corpus = tm_map(x = corpus,removePunctuation) #Remove all the punctuations

corpus = tm_map(x = corpus,removeWords,stopwords()) 

#Remove all the insignificant words like and,the,this,prepositions etc
#stopwords contains a list of nonrelevant words a default list that can be used anywhere

#We will perform stemming. 
#Stemming is taking the root of the word i.e. words like loved,loving,love will all be mapped to love

corpus = tm_map(x = corpus,stemDocument)
corpus = tm_map(x = corpus,stripWhitespace) # Removing the extra spaced in the text

# Creating the Bag of World Model
# Take all the words from the reviews and create a column for each words.
# We will create a table with all reviews as row and words as column 
# the value will be 1 if the word is present in review
# We actually created a Sparse Matrix

dtm = DocumentTermMatrix(corpus) # Create the sparse matrix
dtm = removeSparseTerms(dtm,0.999) 

# to keep a proportion of the most frequent words
# contains 99.9% of the words that have the most ones

#Now we will be applying the classification model

dataset = as.data.frame(as.matrix(dtm)) # Transfrom the matrix into a data frame
dataset$Response = dataset_original$Response # Response column is the column of dependent variable in your dataset

dataset$Response = factor(dataset$Response,levels = c()) # set the number of levels your dependent variable has

library(caTools)
set.seed(123)

split = sample.split(dataset$Response,SplitRatio = 0.8)
train = subset(dataset,split == TRUE )
test = subset(dataset,split == FALSE )

library(randomForest)
model = randomForest(x = train$AlltheIndependentcolumnsoftext,y = train$Response,ntree = 100)

#Predicting the results
pred = predict(model,newdata = test$AlltheIndependentcolumnsoftext)

#Confusion Matrix
cm = table(test$Response,pred)

