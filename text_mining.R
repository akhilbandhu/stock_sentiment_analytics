# Akhil Anand
# Big Data Analytics
# CSIS 4560 
# Sentiment Analysis Project

# The all.data file has 3 levels of sentiment
# negative, neutral, positive

# packages used
library(tidytext)
library(reshape2)
library(tidyverse)
library(ggplot2)
library(dplyr)
library(topicmodels)
library(tm)
library(wordcloud)
library(sentimentr)

all.data <- read_csv("/Users/akhil/CSIS 4560/Stock_Sentiment_Analytics/all-data.csv", col_names = F)
stock.data <- read_csv("/Users/akhil/CSIS 4560/Stock_Sentiment_Analytics/stock_data.csv")
View(all.data)
View(stock.data)
all.data$V1 <- as.factor(all.data$V1)

summary(stock_data)
stock_data$Sentiment <- as.factor(stock_data$Sentiment)

# let's look at tokenizing data, only working with all_data file first
tidy_all_data <- all.data %>%
                    unnest_tokens(word, X2)

# let's count the number of words
tidy_all_data %>% 
  count(word) %>%
  arrange(desc(n))

# lets remove stop words
# words that are not useful for us
tidy_all_data_2 <- all.data %>% 
                    mutate(id = row_number()) %>%
                    unnest_tokens(word,X2) %>%
                    anti_join(stop_words)

# count the data from this better data set now
word_counts <- tidy_all_data_2 %>% 
  count(word) %>%
  arrange(desc(n))

# dont need all the words
word_counts2 <- tidy_all_data_2 %>% 
  count(word) %>%
  filter(n>100) %>%
  mutate(word2 = fct_reorder(word,n)) %>%
  arrange(desc(n))


# let's plot the word count using ggplot
ggplot(
  word_counts2, aes(x = word2, y = n)
) + geom_col() +
  coord_flip() +
  ggtitle("Stock Word Counts")

# we will have to create a custom stop tibble to add to stop words data frame
# word clouds now
wordcloud(
  word = word_counts2$word,
  freq = word_counts2$n,
  max.words = 70
)

# now for sentiment dictionaries
# sentiment lexicon 
# bing, loughran, afinn, nrc dictionaries
# need to remember to cite the NRC emotion lexicon
# need to go through the NRC lexicon
loughran <- get_sentiments("loughran")
summary(as.factor(loughran$sentiment))

nrc <- get_sentiments("nrc")
summary(as.factor(nrc$sentiment))

sentiment_counts <- get_sentiments("nrc") %>% 
  count(sentiment) %>% 
  mutate(sentiment2 = fct_reorder(sentiment, n))

# Visualize sentiment_counts using the new sentiment factor column
ggplot(sentiment_counts, aes(sentiment2, n)) +
  geom_col() +
  coord_flip() +
  # Change the title to "Sentiment Counts in NRC", x-axis to "Sentiment", and y-axis to "Counts"
  labs(
    title = "Sentiment Counts in NRC",
    x = "Sentiment",
    y = "Counts"
  )

# lets append the nrc dictionary using inner join 
sentiment_all_data <- tidy_all_data_2 %>%
                  inner_join(get_sentiments("nrc"))

sentiment_all_data %>% 
  count(word, sentiment) %>%
  arrange(desc(n))

# now lets look at positive and negative data only
sentiment_all_data_2 <- sentiment_all_data %>%
  filter(sentiment %in% c("positive", "negative"))

# word count now 
word_counts3 <- sentiment_all_data_2 %>%
  count(word, sentiment) %>%
  group_by(sentiment) %>% 
  top_n(10, n) %>%
  ungroup() %>%
  mutate(word2 = fct_reorder(word, n))

# lets plot the top 10 positive and negative words
ggplot(
  word_counts3, aes(word2, n,fill=sentiment)) + 
  geom_col() + 
  coord_flip() +
  facet_wrap(~sentiment, scales = "free") +
  labs(
    title = "Sentiment Word Counts",
    x = "Word"
  )

# lets do some unsupervised learning 
# Topic modeling (discrete measure)
# Clustering (distance between objects (knns), word frequency)
# Latent Dirichlet Learning (LDA)

# do some document term matices
dtm_all <- tidy_all_data_2 %>% 
  count(word, id) %>%
  cast_dtm(id,word,n) %>%
  as.matrix()

lda_out <- LDA(
  dtm_all,
  k=2,
  method = "Gibbs",
  control = list(seed=42)
)

glimpse(lda_out)

lda_topics <- lda_out %>% 
  tidy(matrix = "beta")
  
lda_topics <- lda_topics %>% 
  arrange(desc(beta))
  
# lets get the word probabilities now
word_probs <- lda_topics %>%
  group_by(topic) %>%
  top_n(15, beta) %>%
  ungroup() %>%
  mutate(term2 = fct_reorder(term, beta))
  
# plotting word probabilities
ggplot(
  word_probs,
  aes(
    x = term2,
    y = beta,
    fill = as.factor(topic)
  )
) + 
  geom_col(show.legend = F) + 
  facet_wrap(~topic, scales = "free") + 
  coord_flip()

# the topics are very subjective, I need to go through and see what makes sense
lda_topics2 <- LDA(
  dtm_all,
  k=4,
  metho3 = "Gibbs",
  control = list(seed=42)
) %>% 
  tidy(matrix = "beta") %>%
  arrange(desc(beta))

# lets get the word probabilities now
word_probs2 <- lda_topics2 %>%
  group_by(topic) %>%
  top_n(15, beta) %>%
  ungroup() %>%
  mutate(term2 = fct_reorder(term, beta))

# plotting word probabilities
ggplot(
  word_probs2,
  aes(
    x = term2,
    y = beta,
    fill = as.factor(topic)
  )
) + 
  geom_col(show.legend = F) + 
  facet_wrap(~topic, scales = "free") + 
  coord_flip()

# when topic gets duplicated basically we know there's two many values of k in LDA model
# adding different topics is good

?sentimentr()
emotion(all.data$X1[1])  
