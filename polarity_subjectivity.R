## Using the text blob package
## Akhil Anand

#Libraries Used
library(vader)
library(textblob)
# remotes::install_github("news-r/textblob")

# compund is the most useful metric
text <- text_blob(all.data$X2[1])
vader <- get_vader(text)

# if "compound" between 0.3 and -0.3, then neutral
# else if "compound" between 0.3 and 1.0, then positive
# else "compound" between -0.3 and -1.0, then negative
text_sentiment <- as.factor(all.data$V1)
train_sentiment <- NULL
test_sentiment <- NULL

# this will be our testing data
for(i in 1:nrow(all.data)) {
  if(text_sentiment[i] == "negative") {
    test_sentiment[i] <- -1
  }
  else if(text_sentiment[i] == "neutral") {
    test_sentiment[i] <- 0
  }
  else{
    test_sentiment[i] <- 1
  }
}

# this will be our modeling data that needs to be tested
# this is done without any cleaning and the accuracy comes out to 59.24%
for(i in 1:nrow(all.data)) {
  vader_score <- get_vader(all.data$X2[i])
  compound <- as.numeric(vader_score[2])
  # print(compound)
  if(is.na(compound)) {
    train_sentiment[i] <- 0
  }
  else if(compound <= 0.3 && compound >= -0.3) {
    train_sentiment[i] <- 0
  }
  else if(compound > 0.3 && compound < 1.0) {
    train_sentiment[i] <- 1
  }
  else{
    train_sentiment[i] <- -1
  }
}

# lets get how many we get correct
correct_predictions <- sum(ifelse(test_sentiment == train_sentiment, 1, 0))

accuracy <- correct_predictions/nrow(all.data)
