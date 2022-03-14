rm(list = ls())

source("ann.R")

load("data/test.RData")
load("data/model.RData")

test_set <- list()
for (i in seq_len(length(test_labels))) {
  test_set[[i]] <- list()
  test_set[[i]]$input <- as.vector(test_images[i, , ])
  output <- rep(0, 10)
  output[test_labels[i] + 1] <- 1
  test_set[[i]]$output <- output
}

round(mlp(trained_model, test_set[[100]]$input)$params[[4]]$a, 4)

test_set[[100]]$output

cost(trained_model, test_set)$cost

