ann <- function(model, activations, seed = NULL) {
  if (is.vector(model)) {
    params <- list(); layers <- model

    # Weights and biases
    if (!is.null(seed)) set.seed(seed)
    l <- 1
    while (l < length(layers)) {
      params[[l]] <- list()
      params[[l]]$w <- matrix(rnorm(layers[l] * layers[l + 1], 0, 0.1),
                              layers[l + 1],
                              layers[l])
      params[[l]]$b <- matrix(runif(layers[l + 1], 0, 1),
                              layers[l + 1],
                              1)
      l <- l + 1
    }
    params[[l]] <- list()
  } else {
    params <- model$params
  }

  # Sigmoid function
  sig <- function(x) 1 / (1 + exp(-x))

  # Activations
  params[[1]]$a <- matrix(activations, length(activations), 1)
  for (l in seq_len(length(params) - 1)) {
    params[[l + 1]]$a <- sig(params[[l]]$w %*% params[[l]]$a + params[[l]]$b)
    params[[l]]$s_z <- matrix(double(), layers[l + 1], 1)
    params[[l]]$dc_da <- matrix(double(), layers[l + 1], 1)
  }

  model$params <- params
  return(model)
}

cost <- function(model, training_set) {
  rmse <- function(y, yhat) sqrt(mean((y - yhat)^2))

  samples <- sapply(train, function(x) {
    pred <- ann(x$input, model$params)
    rmse(x$output, pred$params[[length(pred)]]$a)
  })

  model$cost <- mean(samples)
  return(model)
}

del_cost <- function(model, training_set) {
  sig_prime <- function(x) exp(-x) / (1 + exp(-x))^2

  s_z <- function(model, l, j) {
    sig_prime(model$params[[l]]$w[j, ] %*%
              model$params[[l - 1]]$a +
              model$params[[l]]$b)
  }

  dc_da <- function(model, n, l, j) {
    out <- 0
    if (l < length(n)) {
      for (j in seq_len(n[l + 1] - 1)) {
        out <- out +
          model$params[[l + 1]]$w[j, k]  *
          model$params[[l + 1]]$s_z[j, 1] *
          model$params[[l + 1]]$dc_da[j, 1]
      }
    } else {
      out <- 2 * (model$params[[l]]$a[j, 1] - y)
    }
  }

  n <- sapply(model$params, function(x) nrow(x$a))
  npar <- sapply(model$params, function(x) nrow(x$w) * ncol(x$w) + nrow(x$b))
  delc <- rep(double(), sum(npar))
  for (l in rev(seq_len(length(n)))) {
    for (j in seq_len(nrow(model$params[[l]]$w))) {
      for (k in seq_len(ncol(model$params[[l]]$w))) {
        model$params[[l]]$dc_da[j, 1] <- dc_da(model, n, l, j)
        model$params[[l]]$s_z[j, 1] <- s_z(model,  l, j)
        delc[] <- model$params[[l - 1]]$a[k, 1] *
          model$params[[l]]$s_z[j, 1] *
          model$params[[l]]$dc_da[j, 1]
      }
    }
  }
}

train <- function(model, training_set, control = list()) {
  f <- function(model) {
    model_vec <- model
    return(model_vec)
  }

  finv <- function(model_vec) {
    model <- model_vec
    return(model)
  }

  opt <- optim(par = f(model),
               fn = function(model_vec, training_set) {
                 cost(finv(model_vec), training_set)
               },
               gr =  function(model_vec, training_set) {
                 del_cost(finv(model_vec), training_set)
               },
               training_set = training_set,
               method = "BFGS",
               lower = -Inf,
               upper = Inf,
               control = control,
               hessian = FALSE)

  return(finv(opt$par))
}
