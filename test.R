rm(list = ls())

source("ann.R")

load("data/test.RData")
load("data/model.RData")

test_set <- list()
for (i in seq_len(length(test_labels))) {
  test_set[[i]] <- list()
  test_set[[i]]$input <- scale(as.vector(test_images[i, , ]))
  output <- rep(0, 10)
  output[test_labels[i] + 1] <- 1
  test_set[[i]]$output <- output
}

x <- test(trained_model, test_set)
