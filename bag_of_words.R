# Akhil Anand
# CSIS 4560
# Big Data Analytics
# some bag of words text mining

# Semantic parsing might be something to look at
# we lose ordering using bag of words

# Libraries 
library(tidytext)
library(glmnet)
library(reshape2)
library(tidyverse)
library(ggplot2)
library(dplyr)
library(topicmodels)
library(tm)
library(wordcloud)
library(qdap)
library(e1071)
library(caret)
library(gmodels)
library(pROC)
library(ROCR)
library(randomForest)
library(SnowballC)
library(caTools)
library(randomForest)
library(rpart)


all.data <- read_csv("/Users/akhil/CSIS 4560/Stock_Sentiment_Analytics/all-data.csv", col_names = F)
stock.data <- read_csv("/Users/akhil/CSIS 4560/Stock_Sentiment_Analytics/stock_data.csv")

# lets create a corpus
stock_text <- all.data[,2]
stock_text2 <- stock_data[,1]
View(stock.data)
View(all.data)

# need to do some moving around of sentiment
# we have test sentiment as well
stockdata_sentiment <- stock_data[,2]
test_sentiment <- as_tibble(test_sentiment)
names(test_sentiment)[1] <- "Sentiment"
test_sentiment <- rbind(test_sentiment, stockdata_sentiment)
stock_text <- as_tibble(stock_text)
stock_text <- cbind(stock_text, test_sentiment)
stock_text$Sentiment <- as.factor(stock_text$Sentiment)

names(stock_text)[1] <- "Text"
stock_text <- rbind(stock_text, stock_text2)

head(stock_text,10)
str(stock_text)

clean_corpus <- function(corpus) {
  stock_corpus <- tm_map(corpus, removePunctuation)
  stock_corpus <- tm_map(corpus, removeWords, words = c(stopwords("en")))
  stock_corpus <- tm_map(corpus, stripWhitespace)
  
  stock_corpus <- tm_map(corpus, content_transformer(replace_number))
  stock_corpus <- tm_map(corpus, content_transformer(replace_abbreviation))
  stock_corpus <- tm_map(corpus, content_transformer(replace_contraction))
  stock_corpus <- tm_map(corpus, content_transformer(replace_symbol))
  return(corpus)
}

# testing out the function
# it should work hopefully
# it didnt work
# test <- all.data[,2]
# test2 <- stock.data[,1]
# names(test)[1] <- "Text"
# 
# test <- rbind(test, test2)
# test$Text <- str_replace_all(test$Text,"[^[:graph:]]", " ") 
# 
# test_source <- VectorSource(test$Text)
# test_corpus <- VCorpus(test_source)
# 
# clean_test <- clean_corpus(test_corpus)
# content(clean_test[[1]])

# tolower(), removePunctuation(), removeNumbers(), stripWhiteSpace(), removeWords()
# bracketX(): Remove all text within brackets (e.g. "It's (so) cool" becomes "It's cool")
# replace_number(): Replace numbers with their word equivalents (e.g. "2" becomes "two")
# replace_abbreviation(): Replace abbreviations with their full text equivalents (e.g. "Sr" becomes "Senior")
# replace_contraction(): Convert contractions back to their base words (e.g. "shouldn't" becomes "should not")
# replace_symbol() Replace common symbols with their word equivalents (e.g. "$" becomes "dollar")
# DO NOT replace numbers
stock_text$Text <- str_replace_all(stock_text$Text,"[^[:graph:]]", " ") 
stock_text$Text <- removeWords(stock_text$Text, c(stopwords("en"),"will","finnish","eur","aap","mn","oyj"))
stock_text$Text <- tolower(stock_text$Text)
stock_text$Text <- removePunctuation(stock_text$Text)
stock_text$Text <- stripWhitespace(stock_text$Text)
stock_text$Text <- removeNumbers(stock_text$Text)

# Replace abbreviations
stock_text$Text <- replace_abbreviation(stock_text$Text)

# Replace number with text
stock_text$Text <- replace_number(stock_text$Text)


# Replace contractions
stock_text$Text <- replace_contraction(stock_text$Text)

# Replace symbols with words
stock_text$Text <- replace_symbol(stock_text$Text)

# i probably want to do some stem completion 
# words like mn and eur will become million and euro

# Volatile corpus, stores the corpus into RAM instead of in memory
# More memory efficient
# It is also possible to make it from a data frame using the dataFrameSource()
stock_source <- VectorSource(stock_text$Text)
stock_corpus <- VCorpus(stock_source)

# Viewing the content of the 10th text in the corpus
content(stock_corpus[[10]])
stock_text$Text[10]

# lets remove stop words using english stop words
# converting into DTM and TDM 
stock_tdm <- TermDocumentMatrix(stock_corpus)
stock_dtm <- DocumentTermMatrix(stock_corpus)

stock_tdm_m <- as.matrix(stock_tdm)
stock_dtm_m <- as.matrix(stock_dtm)

# lets take a look at the dimensions
dim(stock_dtm_m)
dim(stock_tdm_m)

stock_dtm_m[25:45, c("profit", "the")]

# let's get into some visualizations now
# text mining plots

# sum rows of term document matrix and sort by frequency 
term_frequency <- rowSums(stock_tdm_m)
term_frequency <- sort(term_frequency,
                       decreasing = TRUE)

# lets get term frequency using qdap
frequency <- freq_terms(
  stock_text$Text,
  top=25,
  at.least = 3,
  stopwords = stopwords("en")
)

plot(frequency)

# Create a barplot
barplot(term_frequency[1:25],
        col = "tan")

# Creating word clouds
word_frequencies <- data.frame(term = names(term_frequency),
                               num = term_frequency)

wordcloud(
  word_frequencies$term,
  word_frequencies$num,
  max.words = 100,
  colors = "darkblue"
)

# better word cloud
wordcloud(
  word_frequencies$term,
  word_frequencies$num,
  max.words = 100,
  colors = c("darkgoldenrod1", "slateblue4", "seagreen")
)

# maybe a word network
word_associate(
  stock_text$Text,
  match.string = c("profit"),
  stopwords = c(stopwords("en"), "will", "oyj", "eur", "aap","mn","finnish","finland"),
  network.plot = TRUE,
  cloud.colors = c("darkred", "darkgreen")
)

# lets use the stock dtm or tdm to do some learning
inspect(stock_dtm)

# remove sparse terms
clean_stock_dtm <- removeSparseTerms(stock_dtm, 0.99)
clean_stock_dtm
inspect(clean_stock_dtm)

stocks <- as.data.frame(as.matrix(clean_stock_dtm))
colnames(stocks) <- make.names(colnames(stocks))

# creating data partition
stocks$Sentiment <- stock_text$Sentiment
train_obs <- createDataPartition(stocks$Sentiment,p=0.85,list = FALSE)

stock_dtm_train <- stocks[train_obs,]
stock_dtm_test <- stocks[-train_obs,]
prop.table(table(stock_dtm_train$Sentiment))
prop.table(table(stock_dtm_train$Sentiment))

####Cross validation control
Control<-trainControl(method="cv",number=10)

#Random Forest
#train model with 10 k fold cross validation
stock_RF_Model<-train(Sentiment~.,data=stock_dtm_train,method="rf",
                      parms = list(split = "information"),
                      trControl=Control)

#train model with out 10 k fold cross validation
#tweet_RF_Model <- randomForest(label~ . , data = tweet_dtm_train,na.action=na.roughfix)

####Cross validation control
predict_stock_RF <- predict(stock_RF_Model, newdata = stock_dtm_test)

###CROSS TABLE
CrossTable(predict_stock_RF,
           stock_dtm_test$Sentiment,
           prop.chisq = FALSE,
           prop.t = FALSE,
           prop.r=FALSE,
           dnn=c('predicted','actual'))

# Tree
cartModel <- rpart(Sentiment ~ ., data=stock_dtm_train, method="class")
plot(cartModel)
text(cartModel)

# predictions
predictCART <- predict(cartModel, newdata=stock_dtm_test, type="class")
table(stock_dtm_test$Sentiment, predictCART)

# Accuracy
(52+64+687)/nrow(stock_dtm_test)

#Cross validation
numFolds <- trainControl(method = "cv", number = 10)
cpGrid <- expand.grid(.cp=seq(0.001, 0.01, 0.001))
model <- train(Sentiment ~ ., 
               data = stock_dtm_train, 
               method = "rpart", 
               trControl = numFolds, 
               tuneGrid = cpGrid)
plot(model)
predictModel <- predict(model, newdata=stock_dtm_test, type="prob")
predictClass <- NULL
for(i in 1:nrow(predictModel)) {
  if(predictModel$`0`[i] > predictModel$`1`[i] &&predictModel$`0`[i] > predictModel$`-1`[i]) {
    predictClass[i] <- 0
  } else if(predictModel$`1`[i] > predictModel$`0`[i] &&predictModel$`1`[i] > predictModel$`-1`[i]) {
    predictClass[i] <- 1
  } else{
    predictClass[i] <- -1
  }
}
table(stock_dtm_test$Sentiment, predictClass)
sum(ifelse(predictClass==stock_dtm_test$Sentiment,1,0))/nrow(stock_dtm_test)

plot(model)
text(model)

# Improved
cartModelImproved <- rpart(Sentiment ~ ., data=stock_dtm_train, method="class", cp= 0.001)
plot(cartModelImproved)
text(cartModelImproved)

# Baseline Accuracy

4291/nrow(stock_dtm_train)

# Prediction 
predictCARTImproved <- predict(cartModelImproved, newdata=stock_dtm_test, type="class")
table(stock_dtm_test$Sentiment, predictCARTImproved)
sum(ifelse(predictCARTImproved==stock_dtm_test$Sentiment,1,0))/nrow(stock_dtm_test)



#exp model
cartModelAnova <- rpart(Sentiment ~ ., data=stock_dtm_train, method="anova", cp= 0.001)
plot(cartModelAnova)
text(cartModelAnova)

# predictions
predictCARTAnova <- predict(cartModelAnova, newdata=stock_dtm_test, type="vector")
table(stock_dtm_test$Sentiment, predictCARTAnova)

anova_class <- NULL
index <- 1
for(i in predictCARTAnova) {
  if(i <= 1.67) {
    anova_class[index] <- -1 
  }
  else if(i > 1.67 && i <= 2.33){
    anova_class[index] <- 0
  }
  else{
    anova_class[index] <- 1
  }
  index <- index + 1
}
table(stock_dtm_test$Sentiment, anova_class)
sum(ifelse(anova_class==stock_dtm_test$Sentiment,1,0))/nrow(stock_dtm_test)


# random forest
rf_classifier <- randomForest(x = stock_dtm_train,
                             y = stock_dtm_train$Sentiment,
                             ntree = 300)

rf_classifier

# random forest predictions
rf_pred <- predict(rf_classifier, newdata = stock_dtm_test)
rf_train_pred <- predict(rf_classifier, newdata = stock_dtm_train)
confusionMatrix(table(rf_pred,stock_dtm_test$Sentiment))
confusionMatrix(table(rf_train_pred,stock_dtm_train$Sentiment))

table(stock_dtm_test$Sentiment, rf_pred)

# classification
rf.trees <- randomForest(Sentiment~.,data=stock_dtm_train,importance=TRUE)
rf.trees
importance(rf.trees)
varImpPlot(rf.trees)

rf_preds <- predict(rf.trees,newdata=stock_dtm_test,type="class")
rf_train_pred <- predict(rf.trees, newdata = stock_dtm_train, type="class")
table(stock_dtm_train$Sentiment, rf_train_pred)
sum(ifelse(rf_train_pred==stock_dtm_train$Sentiment,1,0))/nrow(stock_dtm_train)


table(stock_dtm_test$Sentiment,rf_preds)
sum(ifelse(rf_preds==stock_dtm_test$Sentiment,1,0))/nrow(stock_dtm_test)
plot(rf.trees)


# Naive Bayes 
control <- trainControl(method="repeatedcv", number=10, repeats=3, classProbs = T)
system.time( classifier_nb <- naiveBayes(stock_dtm_train, stock_dtm_train$Sentiment, laplace = 3,
                                         trControl = control,tuneLength = 101) )

nb_pred <- predict(classifier_nb, type = 'class', newdata = stock_dtm_test)
confusionMatrix(nb_pred,stock_dtm_test$Sentiment)
table(stock_dtm_test$Sentiment,nb_pred)
sum(ifelse(nb_pred==stock_dtm_test$Sentiment,1,0))/nrow(stock_dtm_test)


# Support Vector Machine
svm_classifier <- svm(Sentiment~., data=stock_dtm_train)
svm_classifier
# Predictions
svm_pred <- predict(svm_classifier,stock_dtm_test)
svm_predt <- predict(svm_classifier,stock_dtm_train)

confusionMatrix(svm_pred,stock_dtm_test$Sentiment)

table(stock_dtm_train$Sentiment,svm_predt)
sum(ifelse(svm_predt==stock_dtm_train$Sentiment,1,0))/nrow(stock_dtm_train)

table(stock_dtm_test$Sentiment,svm_pred)
sum(ifelse(svm_pred==stock_dtm_test$Sentiment,1,0))/nrow(stock_dtm_test)
summary(svm_classifier)


# XGBoost

fitControl <- trainControl(method = "cv",
                           number = 5,
                           search = "random",
                           classProbs = TRUE,
                           sampling = "up",
                           verboseIter = FALSE
)

model <- train(Sentiment ~ .,
               data = stock_dtm_train,
               preProcess = c("center", "scale"),
               method = "xgbTree",
               tuneLength = 50,
               verbose = FALSE
)
plot(model)
xg_pred <- predict(model,stock_dtm_test)
xg_pred
table(stock_dtm_test$Sentiment,xg_pred)
sum(ifelse(xg_pred==stock_dtm_test$Sentiment,1,0))/nrow(stock_dtm_test)


cm_train <- confusionMatrix(predict(model, newdata = train[Index,]), train[Index,]$target)
cm_test <- confusionMatrix(predict(model, newdata = train[-Index,]), train[-Index,]$target)


ggplot(model) + ggtitle("Results of the xgbTree model")+
  geom_point(colour = "#6633FF")


# CNN 
# lets try this

model <- keras_model_sequential() %>% 
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = "relu", 
                input_shape = c(32,32,3)) %>% 
  layer_max_pooling_2d(pool_size = c(2,2)) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2,2)) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = "relu")

summary(model)

model %>% 
  layer_flatten() %>% 
  layer_dense(units = 64, activation = "relu") %>% 
  layer_dense(units = 10, activation = "softmax")


# Should we try glm?
logit_model <- glm(Sentiment~., data = stock_dtm_train, family = "binomial")

logit_pred <- predict(logit_model, newdata = stock_dtm_test, type = "response")
logit_class <- NULL
index <- 1
for(i in logit_pred) {
  if(i <= 0.33) {
    logit_class[index] <- -1 
  }
  else if(i > 0.66){
    logit_class[index] <- 1
  }
  else{
    logit_class[index] <- 0 
  }
  index <- index + 1
}
logit_class <- as.factor(logit_class)
confusionMatrix(logit_class,stock_dtm_test$Sentiment)

table(stock_dtm_test$Sentiment, logit_class)
sum(ifelse(logit_class==stock_dtm_test$Sentiment,1,0))/nrow(stock_dtm_test)

#RNN

# set some parameters for our model
max_len <- 6 # the number of previous examples we'll look at
batch_size <- 32 # number of sequences to look at at one time during training
total_epochs <- 15 # how many times we'll look @ the whole dataset while training our model

# set a random seed for reproducability
set.seed(123)
rnn_model <- keras_model_sequential()
rnn_model %>%
  layer_dense(input_shape = dim(stock_dtm_train), units = max_len)

rnn_model %>% 
  layer_simple_rnn(units = 6)

rnn_model %>%
  layer_dense(units = 1, activation = 'sigmoid') # output

summary(rnn_model)

rnn_model %>% compile(loss = 'categorical_crossentropy', 
                  optimizer = optimizer_rmsprop(), 
                  metrics = 'accuracy')

# Actually train our model! This step will take a while
x <- array(stock_dtm_train[,-156])
rnn_history <- rnn_model %>% fit(
  x, stock_dtm_train$Sentiment,
  batch_size = batch_size,
  epochs = total_epochs,
  validation_split = 0.01
)
