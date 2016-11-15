
#Business Analytics - Assignment 8
##Amazon Customer Reviews
###Yixi Wei
========================================================

### Summary

In this project, I run an LDA topic models analysis for Amazon's Customer Reviews, more specifically, the Automotive reviews (available from: http://jmcauley.ucsd.edu/data/amazon/). I used the R package [lda](http://cran.r-project.org/web/packages/lda/) and I visualize the output using [LDAvis](https://github.com/cpsievert/LDAvis). Because my dataset is a json file, so I used the [rjson] (http://cran.r-project.org/web/packages/lda/) to process this dataset.

All the relevant files are on Github repository: https://github.com/dduwill/Product-Review

The R Source Code is Available at: https://github.com/dduwill/Product-Review/blob/master/Assignment%208.R

The LDA topic models analysis for top 10 relavant topics is Available at: https://dduwill.github.io/Product-Review/vis

### Result Analysis
Note that, all my analysis are using a relevance setting of $\lambda = 0.5$.

According to my result, topic 3, 5, 6 have some overlaps, these customer reviews are more focused on detailed information of the vehicles, such as engines, oil filters, batteries, lights, etc.

Topic 3  has the most relevant term: light bulbs, and also some other related terms, like leds, bright, information. 
https://dduwill.github.io/Product-Review/vis/#topic=3&lambda=0.5&term=

Topic 5 has the most relevant term: oil, and some other related terms, like engine, oil filter, and fuel/gas. So this topic is related to detailed mechanical issues. 
https://dduwill.github.io/Product-Review/vis/#topic=5&lambda=0.5&term=

Topic 6 has the most relevant term: battery, or charger/power. This topic is related to electrical system of the cars.
https://dduwill.github.io/Product-Review/vis/#topic=6&lambda=0.5&term=

Topic 4, 8 have a great overlap, mostly focused on exterior apperance of the vehicles, like leathers, wax, cleanness, plaints.

Topic 4 has the most relevant term: wax and paint.
https://dduwill.github.io/Product-Review/vis/#topic=4&lambda=0.5&term=

Topic 8 has the most relevant term: leather.
https://dduwill.github.io/Product-Review/vis/#topic=8&lambda=0.5&term=

Topic 7 is very close to topic 4, 8, but more focused on minor exterior condition, for example, whether the vehicle is carefully washed or not. Most relevant terms are: towels, wash, cleaning, brush, etc.
https://dduwill.github.io/Product-Review/vis/#topic=7&lambda=0.5&term=

Topic 9, 2, 1 are interconnected with each other, these topics focused on detailed parts of the vehicles, like trailer, lock, hose, tire, etc. 

https://dduwill.github.io/Product-Review/vis/#topic=1&lambda=0.5&term=

https://dduwill.github.io/Product-Review/vis/#topic=2&lambda=0.5&term=

https://dduwill.github.io/Product-Review/vis/#topic=9&lambda=0.5&term=

Topic 10 is isolated from all the other topics, which only focused on windshield and wiper conditions. 
https://dduwill.github.io/Product-Review/vis/#topic=10&lambda=0.5&term=

From the topics' sizes, we could conclude that three major factors affect the customer reviews - exterior apperance, mechanical conditions and detailed parts. 



### The data

First, I manually download the review .json file to my work dir, and load the .json file in R studio. Because .json file is a large list instead of data frame, so after loading the original .json data, I also need to extract the review text to my review corpus. Note that this review files contains 9 categories, while the 5th column in the list is the review text.

```r
#read json file
setwd("C:/Users/weiyi/Desktop/R/Assignment 8")
path <- "Automotive_5.json"
data <- fromJSON(sprintf("[%s]", paste(readLines(path),collapse=",")))
review <-sapply(data, function(x) x[[5]])
```

### Pre-processing

Before fitting a topic model, we need to tokenize the text, and remove all the punctuations and spaces. In particular, we use the english stop words from the "SMART" and several customized stop words, like just, get, will, can, etc. These customized stop words are based on dfm analysis, which is not showing here. Then I also formated my review data into the format required by the lda package.

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

Then I compute a few statistics about the corpus, such as length and vocabulary counts for lda:

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

To visualize the result, I used the package `LDAvis`, which would estimate the document-topic distributions. I computed the number of tokens per document and the frequency of the terms across the entire corpus from previous steps. And I will use them to create the `reviews.LDA`, along with $\phi$, $\theta$, and `vocab`.
```r
theta <- t(apply(fit$document_sums + alpha, 2, function(x) x/sum(x)))
phi <- t(apply(t(fit$topics) + eta, 2, function(x) x/sum(x)))

reviews.LDA <- list(phi = phi,
                     theta = theta,
                     doc.length = doc.length,
                     vocab = vocab,
                     term.frequency = term.frequency)
```

Now we're ready to call the `createJSON()` function in `LDAvis` package. This function will return a character string representing a JSON object used to populate the visualization. The `createJSON()` function computes topic frequencies, inter-topic distances, and projects topics onto a two-dimensional plane to represent their similarity to each other. 

It has a feature of tuning parameter - lambda (0-1), that controls how the terms are ranked for each topic, where terms are listed in decreasing of relevance. Values of lambda near 1 give high relevance rankings to frequent terms within a given topic, whereas values of lambda near zero give high relevance rankings to exclusive terms (or not requent used) within a topic. Note that readers can interact with any of these topics to view the relevant terms.

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

The `serVis()` function can take `json` and create a user interactive webpage. As a result, I write `json` to the `vis` directory along with other HTML and JavaScript required to render the page. Note that the `vis` directory was orginally saved in my local drive, but I uploaded to Github for sharing purpose. You can see this LDA page at: https://dduwill.github.io/Product-Review/vis.
