rm(list = ls())

source("ann.R")

load("data/train.RData")

min_max <- function(x) (x - min(x)) / (max(x) - min(x))

training_set <- list()
for (i in seq_len(length(train_labels))) {
  training_set[[i]] <- list()
  training_set[[i]]$input <- min_max(as.vector(train_images[i, , ]))
  output <- rep(0, 10)
  output[train_labels[i] + 1] <- 1
  training_set[[i]]$output <- output
}

model <- mlp(c(784, 16, 10), seed = 1337)

trained_model <- train(model,
                       training_set,
                       control = list(trace = 2))

save(trained_model, file = "data/model.RData")
