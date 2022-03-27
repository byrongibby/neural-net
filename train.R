rm(list = ls())

source("ann.R")

load("data/train.RData")

train_labels[1:9]

par(mfrow = c(3, 3))
image(t(apply(train_images[1, , ], 2, rev)))
image(t(apply(train_images[2, , ], 2, rev)))
image(t(apply(train_images[3, , ], 2, rev)))
image(t(apply(train_images[4, , ], 2, rev)))
image(t(apply(train_images[5, , ], 2, rev)))
image(t(apply(train_images[6, , ], 2, rev)))
image(t(apply(train_images[7, , ], 2, rev)))
image(t(apply(train_images[8, , ], 2, rev)))
image(t(apply(train_images[9, , ], 2, rev)))

par(mfrow = c(1, 1))
hist(train_labels, breaks = -0.5:9.5)

min_max <- function(x) (x - min(x)) / (max(x) - min(x))

training_set <- list()
for (i in seq_len(length(train_labels))) {
  training_set[[i]] <- list()
  training_set[[i]]$input <- min_max(as.vector(train_images[i, , ]))
  output <- rep(0, 10)
  output[train_labels[i] + 1] <- 1
  training_set[[i]]$output <- output
}

model <- mlp(c(784, 16, 16, 10),
             c("logistic", "logistic", "logistic"),
             seed = 1337)

cost(model, list(list(input = matrix(rnorm(784), nrow = 28),
                      output = rep(0, 10))))

gradient_check(model, training_set[1, drop = FALSE])

# trained_model <- train_opt(model,
#                            training_set,
#                            control = list(trace = 2))

trained_model <- train_sgd(model,
                           training_set,
                           batch_size = 100)

predict(trained_model, training_set)[sample(1:60000, 100)]

save(trained_model, file = "data/model.RData")
