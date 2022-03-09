# Akhil Anand
# CSIS 4560
# Big Data Analytics

# Libraries used 
library(dendextend)
library(qdap)
library(tm)
library(RWeka)

# lets make a hierarchical cluster on the term document matrix
sparse_tdm <- removeSparseTerms(stock_tdm, sparse = 0.975)
sparse_tdm_m <- as.matrix(sparse_tdm)
dist_tdm <- dist(sparse_tdm_m)

hc <- hclust(dist_tdm)
plot(hc)

# using the dendextend library
hcd <- as.dendrogram(hc)
labels(hcd)


# making some n-gram tokens
# bi-gram tokenizer
tokenizer <- function(x) {
  NGramTokenizer(x, Weka_control(min=2,max=2))
}

bigram_tdm <- TermDocumentMatrix(stock_corpus,
                                 control = list(tokenize = tokenizer))
bigram_tdm


