
#Business Analytics - Assignment 8
##Amazon Customer Reviews
###Yixi Wei
========================================================

In this project, I run an LDA topic models analysis for Amazon's Customer Reviews, more specifically, the Automotive reviews (available from: http://jmcauley.ucsd.edu/data/amazon/). I used the R package [lda](http://cran.r-project.org/web/packages/lda/) and I visualize the output using [LDAvis](https://github.com/cpsievert/LDAvis).

### The data

First, I manually download the review .json file to my work dir, and load the .json file in R studio. Note that this review files contains 9 categories, while the 5th column in the list is the review text, so I extract the the review text only.

```r
#read json file
setwd("C:/Users/weiyi/Desktop/R/Assignment 8")
path <- "Automotive_5.json"
data <- fromJSON(sprintf("[%s]", paste(readLines(path),collapse=",")))
review <-sapply(data, function(x) x[[5]])
```




### Pre-processing

Before fitting a topic model, we need to tokenize the text, and remove all the punctuations and spaces. In particular, we use the english stop words from the "SMART" and several customized stop words.

```r
#Cleaning corpus
stop_words <- stopwords("english")
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
```

### Using the R package 'lda' for model fitting

The object `documents` is a super large list where each element represents one document. After creating this list, we compute a few statistics about the corpus, such as length and vocabulary counts:

```r
# Compute some statistics related to the data set:
D <- length(documents)  # number of documents (1)
W <- length(vocab)  # number of terms in the vocab (8941L)
doc.length <- sapply(documents, function(x) sum(x[2, ]))  # number of tokens per document [46, 27, 106 ...]
N <- sum(doc.length)  # total number of tokens in the data (863558L)
term.frequency <- as.integer(term.table) 
```

Next, we set up a topic model with 10 topics, relatively diffuse priors for the topic-term distributions ($\eta$ = 0.02) and document-topic distributions ($\alpha$  = 0.02), and we set the collapsed Gibbs sampler to run for 3,000 iterations (slightly conservative to ensure convergence). A visual inspection of `fit$log.likelihood` shows that the MCMC algorithm has converged after 3,000 iterations. This block of code takes about 10 minutes to run on a laptop using a 2.5GHz i7 processor (and 8GB RAM).


```r
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
t2 - t1  #10 min runtime on my laptop
```

### Visualizing the fitted model with LDAvis

To visualize the result using [LDAvis](https://github.com/cpsievert/LDAvis/), we'll need estimates of the document-topic distributions, which we denote by the $D \times K$ matrix $\theta$, and the set of topic-term distributions, which we denote by the $K \times W$ matrix $\phi$. We estimate the "smoothed" versions of these distributions ("smoothed" means that we've incorporated the effects of the priors into the estimates) by cross-tabulating the latent topic assignments from the last iteration of the collapsed Gibbs sampler with the documents and the terms, respectively, and then adding pseudocounts according to the priors. A better estimator might average over multiple iterations of the Gibbs sampler (after convergence, assuming that the MCMC is sampling within a local mode and there is no label switching occurring), but we won't worry about that for now.

We've already computed the number of tokens per document and the frequency of the terms across the entire corpus. We save these, along with $\phi$, $\theta$, and `vocab`, in a list as the data object `reviews.LDA`.
```r
theta <- t(apply(fit$document_sums + alpha, 2, function(x) x/sum(x)))
phi <- t(apply(t(fit$topics) + eta, 2, function(x) x/sum(x)))

reviews.LDA <- list(phi = phi,
                     theta = theta,
                     doc.length = doc.length,
                     vocab = vocab,
                     term.frequency = term.frequency)
```





Now we're ready to call the `createJSON()` function in **LDAvis**. This function will return a character string representing a JSON object used to populate the visualization. The `createJSON()` function computes topic frequencies, inter-topic distances, and projects topics onto a two-dimensional plane to represent their similarity to each other. It also loops through a grid of values of a tuning parameter, $0 \leq \lambda \leq 1$, that controls how the terms are ranked for each topic, where terms are listed in decreasing of *relevance*, where the relevance of term $w$ to topic $t$ is defined as $\lambda \times p(w \mid t) + (1 - \lambda) \times p(w \mid t)/p(w)$. Values of $\lambda$ near 1 give high relevance rankings to *frequent* terms within a given topic, whereas values of $\lambda$ near zero give high relevance rankings to *exclusive* terms within a topic. The set of all terms which are ranked among the top-`R` most relevant terms for each topic are pre-computed by the `createJSON()` function and sent to the browser to be interactively visualized using D3 as part of the JSON object.


```r
library(LDAvis)
library(servr)
# create the JSON object to feed the visualization:
json <- createJSON(phi = reviews.LDA$phi, 
                   theta = reviews.LDA$theta, 
                   doc.length = reviews.LDA$doc.length, 
                   vocab = reviews.LDA$vocab, 
                   term.frequency = reviews.LDA$term.frequency)

serVis(json, out.dir = 'vis', open.browser = TRUE)
```

The `serVis()` function can take `json` and serve the result in a variety of ways. Here we'll write `json` to a file within the 'vis' directory (along with other HTML and JavaScript required to render the page). You can see the result at: https://dduwill.github.io/Product-Review/vis.
