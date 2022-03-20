sigmoid_fn <- function(fn = "relu") {
  switch(fn,
         "logistic" = function(x, derivative = FALSE) {
           if (!derivative) {
             1 / (1 + exp(-x))
           } else {
             y <- exp(-x) / (1 + exp(-x))^2
             ifelse(is.nan(y), 0, y)
           }
         },
         "relu" = function(x, derivative = FALSE) {
           if (!derivative) {
             ifelse(x > 0, x, 0)
           } else {
             ifelse(x > 0, 1, 0)
           }
         })
}

mlp <- function(n, sigmoid = "relu", seed = NULL) {
  w <- list()
  if (!is.null(seed)) set.seed(seed)
  for (l in seq_len(length(n) - 1)) {
    w[[l]] <- matrix(rnorm(n[l + 1] * (n[l] + 1), 0, 0.1),
                     nrow = n[l + 1],
                     ncol = n[l] + 1)
  }
  model <- list()
  model$sigmoid <- sigmoid
  model$weights <- w
  return(model)
}

cost <- function(model, training_set) {
  samples <- sapply(training_set,
    function(x, w, sig) {
      a <- list()
      z <- list()
      a[[1]] <- w[[1]] %*% c(1, x$input)
      z[[1]] <- sig(a[[1]])
      for (l in seq(2, length(w))) {
        a[[l]] <- w[[l]] %*% c(1, z[[l - 1]])
        z[[l]] <- sig(a[[l]])
      }
      sum(0.5 * (x$output - z[[length(w)]])^2)
    },
    w = model$weights,
    sig = sigmoid_fn(model$sigmoid))
  return(mean(samples))
}

del_cost <- function(model, training_set) {
  samples <- sapply(training_set,
    function(x, w, sig) {
      a <- list()
      z <- list()
      a[[1]] <- w[[1]] %*% c(1, x$input)
      z[[1]] <- sig(a[[1]])
      for (l in seq(2, length(w))) {
        a[[l]] <- w[[l]] %*% c(1, z[[l - 1]])
        z[[l]] <- sig(a[[l]])
      }
      d <- list()
      gr <- list()
      d[[length(w)]] <- sig(a[[length(w)]], T) * (z[[length(w)]] - x$output)
      for (l in seq(length(w) - 1, 1)) {
        d[[l]] <- sig(a[[l]], T) * crossprod(w[[l + 1]][, -1], d[[l + 1]])
        gr[[l + 1]] <- tcrossprod(d[[l + 1]], c(1, z[[l]]))
      }
      gr[[1]] <- tcrossprod(d[[l]], c(1, x$input))
      unlist(sapply(gr, as.vector))
    },
    w = model$weights,
    sig = sigmoid_fn(model$sigmoid))
  apply(simplify2array(samples), 1, mean)
}

gradient_check <- function(model, x, eps = 1e-5) {
  grn <- rep(NA, sum(sapply(model$weights, function(x) prod(dim(x)))))
  index_base <- 0
  for (l in seq_len(length(model$weights))) {
    for (i in seq_len(prod(dim(model$weights[[l]])))) {
      print(i)
      m_pos <- model
      m_neg <- model
      m_pos$weights[[l]][i] <- model$weights[[l]][i] + eps
      m_neg$weights[[l]][i] <- model$weights[[l]][i] - eps
      y <- (cost(m_pos, x) - cost(m_neg, x)) / (2 * eps)
      grn[index_base   + i] <- y
    }
    index_base <- index_base + i
  }
  gra <- del_cost(model, x)
  diff <- norm(gra - grn, type = "2") /
    max(norm(gra, type = "2"), norm(grn, type = "2"))
  return(diff)
}

model2vec <- function(model) {
  unlist(sapply(model$weights, as.vector))
}

vec2model <- function(params_vec, model) {
  start_index <- 1
  for (l in seq_len(length(model$weights))) {
    nweights <- nrow(model$weights[[l]]) * ncol(model$weights[[l]])
    index_range <- start_index:(start_index + nweights - 1)
    model$weights[[l]] <- matrix(params_vec[index_range],
                                 nrow(model$weights[[l]]),
                                 ncol(model$weights[[l]]))
    start_index <- start_index + nweights
  }
  return(model)
}

train <- function(model, training_set, control = list()) {
  opt <- optim(par = model2vec(model),
               fn = function(params_vec, model, data) {
                 cost(vec2model(params_vec, model), data)
               },
               gr =  function(params_vec, model, data) {
                 del_cost(vec2model(params_vec, model), data)
               },
               model = model,
               data = training_set,
               method = "BFGS",
               control = control)
  model <- vec2model(opt$par, model)
  model$opt <- opt
  return(model)
}

test <- function(model, test_set) {
  samples <- sapply(test_set,
    function(x, w, sig) {
      a <- list()
      z <- list()
      a[[1]] <- w[[1]] %*% c(1, x$input)
      z[[1]] <- sig(a[[1]])
      for (l in seq(2, length(w))) {
        a[[l]] <- w[[l]] %*% c(1, z[[l - 1]])
        z[[l]] <- sig(a[[l]])
      }
      x$prediction <- z[[length(w)]]
      return(x)
    },
    w = model$weights,
    sig = sigmoid_fn(model$sigmoid))
  return(samples)
}
