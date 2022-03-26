act_fn <- function(fns = "relu") {
  lapply(as.list(fns), function(fn) {
          switch(fn,
                 "logistic" = function(x, k, derivative = FALSE) {
                   if (!derivative) {
                     1 / (1 + exp(-x))
                   } else {
                     y <- exp(-x) / (1 + exp(-x))^2
                     ifelse(is.nan(y), 0, y)
                   }
                 },
                 "relu" = function(x, k, derivative = FALSE) {
                   if (!derivative) {
                     ifelse(x > 0, x, 0)
                   } else {
                     ifelse(x > 0, 1, 0)
                   }
                 },
                 "softmax" = function(x, k, derivative = FALSE) {
                   if (!derivative) {
                     exp(x) / sum(exp(x))
                   } else {
                     y <- exp(x) / sum(exp(x))
                     y * (1 - y)
                   }
                 })
        })
}

mlp <- function(n, activation = "relu", seed = NULL) {
  w <- list()
  if (!is.null(seed)) set.seed(seed)
  for (l in seq_len(length(n) - 1)) {
    w[[l]] <- matrix(rnorm(n[l + 1] * (n[l] + 1), 0, n[l + 1]^-0.5),
                     nrow = n[l + 1],
                     ncol = n[l] + 1)
  }
  if (length(activation) != length(n) - 1)
    act <- rep(act_fn(activation[1]), length(n) - 1)
  else
    act <- act_fn(activation)
  model <- list()
  model$activations <- act
  model$weights <- w
  return(model)
}

cost <- function(model, training_set) {
  samples <- sapply(training_set,
    function(x, w, h) {
      L <- length(w)
      a <- list()
      z <- list()
      a[[1]] <- w[[1]] %*% c(1, x$input)
      z[[1]] <- h[[1]](a[[1]])
      for (l in seq(2, L)) {
        a[[l]] <- w[[l]] %*% c(1, z[[l - 1]])
        z[[l]] <- h[[l]](a[[l]])
      }
      sum(0.5 * (x$output - z[[L]])^2)
    },
    w = model$weights,
    h = model$activations)
  return(mean(samples))
}

del_cost <- function(model, training_set) {
  samples <- sapply(training_set,
    function(x, w, h) {
      L <- length(w)
      a <- list()
      z <- list()
      a[[1]] <- w[[1]] %*% c(1, x$input)
      z[[1]] <- h[[1]](a[[1]])
      for (l in seq(2, L)) {
        a[[l]] <- w[[l]] %*% c(1, z[[l - 1]])
        z[[l]] <- h[[l]](a[[l]])
      }
      d <- list()
      gr <- list()
      d[[L]] <- h[[L]](a[[L]], T) * (z[[L]] - x$output)
      for (l in seq(L - 1, 1)) {
        d[[l]] <- h[[l]](a[[l]], T) * crossprod(w[[l + 1]][, -1], d[[l + 1]])
        gr[[l + 1]] <- tcrossprod(d[[l + 1]], c(1, z[[l]]))
      }
      gr[[1]] <- tcrossprod(d[[l]], c(1, x$input))
      unlist(sapply(gr, as.vector))
    },
    w = model$weights,
    h = model$activations)
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

train_opt <- function(model, training_set, method = "CG", control = list()) {
  opt <- optim(par = model2vec(model),
               fn = function(params_vec, model, data) {
                 cost(vec2model(params_vec, model), data)
               },
               gr =  function(params_vec, model, data) {
                 del_cost(vec2model(params_vec, model), data)
               },
               model = model,
               data = training_set,
               method = method,
               control = control)
  model <- vec2model(opt$par, model)
  model$opt <- opt
  return(model)
}

train_sgd <- function(model, training_set, batch_size) {
  n <- seq_len(length(training_set))
  params <- model2vec(model)
  loss <- 1
  tol <- 1
  m <- 2
  nu <- 0.05 * c(1 * rep(sqrt(784), 785 * 16),
                 4 * rep(sqrt(16), 17 * 16),
                 6 * rep(sqrt(16), 17 * 10))
  while (loss > 1e-2 && tol > 1e-4) {
    model <- vec2model(params, model)
    data <- training_set[sample(n, batch_size), drop = FALSE]
    prev_params <- params
    params <- params - del_cost(model, data)
     + 0.0 * (params - prev_params)
    print(loss <- cost(model, data))
    tol <- norm(params - prev_params, type = "2") /
      max(norm(params, type = "2"), norm(prev_params, type = "2"))
    m <- m + 1
    if (batch_size < length(training_set))
      batch_size <- batch_size + 1
  }
  return(model)
}

test <- function(model, test_set) {
  samples <- sapply(test_set,
    function(x, w, h) {
      L <- length(w)
      a <- list()
      z <- list()
      a[[1]] <- w[[1]] %*% c(1, x$input)
      z[[1]] <- h[[1]](a[[1]])
      for (l in seq(2, L)) {
        a[[l]] <- w[[l]] %*% c(1, z[[l - 1]])
        z[[l]] <- h[[l]](a[[l]])
      }
      return(z[[L]])
    },
    w = model$weights,
    h = model$activations)
  return(samples)
}
