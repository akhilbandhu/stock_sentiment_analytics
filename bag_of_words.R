# Akhil Anand
# CSIS 4560
# Big Data Analytics
# some bag of words text mining

# Semantic parsing might be something to look at
# we lose ordering using bag of words

# Libraries 
library(tidytext)
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

# lets create a corpus
stock_text <- all.data[,2]
stock_text2 <- stock.data[,1]


# need to do some moving around of sentiment
# we have test sentiment as well
stockdata_sentiment <- stock.data[,2]
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
# Replace abbreviations
stock_text$Text <- replace_abbreviation(stock_text$Text)

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
content(stock_corpus[[1]])
stock_text$Text[1]

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
train_obs <- createDataPartition()
