library(rjson)
library(dplyr)
require(magrittr)
library(quanteda)
library(stm)
library(tm)
library(NLP)
library(openNLP)
library(ggplot2)
library(ggdendro)
library(cluster)
library(fpc)  

#read json file
setwd("C:/Users/weiyi/Desktop/R/Assignment 8")
path <- "Automotive_5.json"
data <- fromJSON(sprintf("[%s]", paste(readLines(path),collapse=",")))
review <-sapply(data, function(x) x[[5]])

#Generate DFM
help(corpus)
corpus <- corpus(review)
corpus <- toLower(corpus, keepAcronyms = FALSE) 
cleancorpus <- tokenize(corpus, 
                        removeNumbers=TRUE,  
                        removePunct = TRUE,
                        removeSeparators=TRUE,
                        removeTwitter=FALSE,
                        verbose=TRUE)

dfm <- dfm(cleancorpus,
          toLower = TRUE, 
          ignoredFeatures =stopwords("SMART"), 
          verbose=TRUE, 
          stem=TRUE)
# Reviewing top features
topfeatures(dfm, 50)     # displays 50 features

#Cleaning corpus
stop_words <- stopwords("SMART")
## additional junk words showing up in the data
stop_words <- c(stop_words, "just", "get", "will", "can", "also", "much","need")
stop_words <- tolower(stop_words)


cleancorpus <- gsub("'", "", cleancorpus) # remove apostrophes
cleancorpus <- gsub("[[:punct:]]", " ", cleancorpus)  # replace punctuation with space
cleancorpus <- gsub("[[:cntrl:]]", " ", cleancorpus)  # replace control characters with space
cleancorpus <- gsub("^[[:space:]]+", "", cleancorpus) # remove whitespace at beginning of documents
cleancorpus <- gsub("[[:space:]]+$", "", cleancorpus) # remove whitespace at end of documents
cleancorpus <- gsub("[^a-zA-Z -]", " ", cleancorpus) # allows only letters
cleancorpus <- tolower(cleancorpus)  # force to lowercase

## get rid of blank docs
cleancorpus <- cleancorpus[cleancorpus != ""]

# tokenize on space and output as a list:
doc.list <- strsplit(cleancorpus, "[[:space:]]+")

# compute the table of terms:
term.table <- table(unlist(doc.list))
term.table <- sort(term.table, decreasing = TRUE)


# remove terms that are stop words or occur fewer than 5 times:
del <- names(term.table) %in% stop_words | term.table < 5
term.table <- term.table[!del]
term.table <- term.table[names(term.table) != ""]
vocab <- names(term.table)

# now put the documents into the format required by the lda package:
get.terms <- function(x) {
  index <- match(x, vocab)
  index <- index[!is.na(index)]
  rbind(as.integer(index - 1), as.integer(rep(1, length(index))))
}
documents <- lapply(doc.list, get.terms)


# Compute some statistics related to the data set:
D <- length(documents)  # number of documents (1)
W <- length(vocab)  # number of terms in the vocab (8941L)
doc.length <- sapply(documents, function(x) sum(x[2, ]))  # number of tokens per document [46, 27, 106 ...]
N <- sum(doc.length)  # total number of tokens in the data (863558L)
term.frequency <- as.integer(term.table) 

# MCMC and model tuning parameters:
K <- 10
G <- 3000
alpha <- 0.02
eta <- 0.02

# Fit the model:
library(lda)
set.seed(357)
t1 <- Sys.time()
fit <- lda.collapsed.gibbs.sampler(documents = documents, K = K, vocab = vocab, 
                                   num.iterations = G, alpha = alpha, 
                                   eta = eta, initial = NULL, burnin = 0,
                                   compute.log.likelihood = TRUE)
t2 <- Sys.time()
## display runtime
t2 - t1  

theta <- t(apply(fit$document_sums + alpha, 2, function(x) x/sum(x)))
phi <- t(apply(t(fit$topics) + eta, 2, function(x) x/sum(x)))

reviews.LDA <- list(phi = phi,
                     theta = theta,
                     doc.length = doc.length,
                     vocab = vocab,
                     term.frequency = term.frequency)

library(LDAvis)
library(servr)

# create the JSON object to feed the visualization:
json <- createJSON(phi = reviews.LDA$phi, 
                   theta = reviews.LDA$theta, 
                   doc.length = reviews.LDA$doc.length, 
                   vocab = reviews.LDA$vocab, 
                   term.frequency = reviews.LDA$term.frequency)

serVis(json, out.dir = 'vis', open.browser = TRUE)

