rm(list = ls())

source("ann.R")

load("data/train.RData")

min_max <- function(x) (x - min(x)) / (max(x) - min(x))

scaled_images <- apply(apply(train_images, 1, as.vector), 1,
                       function(x) {
                         if (max(x) - min(x) != 0) {
                           y <- (x - mean(x)) / (max(x) - min(x))
                         } else {
                           y <- rep(0, length(x))
                         }
                         return(y)
                       })

training_set <- list()
for (i in seq_len(length(train_labels))) {
  training_set[[i]] <- list()
  training_set[[i]]$input <- scaled_images[i, ]
  output <- rep(0, 10)
  output[train_labels[i] + 1] <- 1
  training_set[[i]]$output <- output
}

model <- mlp(c(784, 16, 16, 10),
             c("relu", "relu", "softmax"),
             seed = 1337)

trained_model <- train_sgd(model,
                           training_set,
                           batch_size = 1000)

gradient_check(model, training_set[1:100])

cost(model, training_set)
cost(trained_model, training_set)

save(trained_model, file = "data/model.RData")
