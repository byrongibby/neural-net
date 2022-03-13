mlp <- function(model, activations, seed = NULL) {
  if (!is.list(model)) {
    params <- list()
    layers <- model

    # Weights and biases
    if (!is.null(seed)) set.seed(seed)
    l <- 1
    while (l <= length(layers)) {
      params[[l]] <- list()
      if (l > 1) {
        params[[l]]$w <- matrix(rnorm(layers[l] * layers[l - 1], 0, 1),
                                layers[l],
                                layers[l - 1])
        params[[l]]$b <- matrix(runif(layers[l], -1, 1),
                                layers[l],
                                1)
      }
      l <- l + 1
    }
  } else {
    layers <- sapply(model$params, function(x) nrow(x$a))
    params <- model$params
  }

  # Sigmoid function
  sig <- function(x) 1 / (1 + exp(-x))

  # Activations and initialization vectors for sig'(z) and dc/da
  params[[1]]$a <- matrix(activations, length(activations), 1)
  for (l in seq(2, length(params))) {
    params[[l]]$a <- sig(params[[l]]$w %*% params[[l - 1]]$a + params[[l]]$b)
    params[[l]]$sig_prime_z <- matrix(NaN, layers[l], 1)
    params[[l]]$dc_da <- matrix(NaN, layers[l], 1)
    params[[l]]$grad <- matrix(NaN,
                               nrow(params[[l]]$w),
                               ncol(params[[l]]$w) + 1)
  }

  model <- list()
  model$params <- params
  return(model)
}

cost <- function(model, training_set) {
  rmse <- function(y, yhat) sqrt(mean((y - yhat)^2))

  samples <- sapply(training_set, function(x) {
    pred <- mlp(model, x$input)
    rmse(x$output, pred$params[[length(pred$params)]]$a)
  })

  model$cost <- mean(samples)
  return(model)
}

del_cost <- function(model, training_set) {
  require(parallel)

  sig_prime <- function(x) exp(-x) / (1 + exp(-x))^2

  sig_prime_z <- function(model, l, j) {
    sig_prime(model$params[[l]]$w[j, ] %*%
              model$params[[l - 1]]$a +
              model$params[[l]]$b[j])
  }

  dc_da <- function(model, n, l, j,  y) {
    out <- 0
    if (l < length(n)) {
      k <- j
      for (j in seq_len(nrow(model$params[[l + 1]]$w))) {
        out <- out +
          model$params[[l + 1]]$w[j, k]  *
          model$params[[l + 1]]$sig_prime_z[j, 1] *
          model$params[[l + 1]]$dc_da[j, 1]
      }
    } else {
      out <- 2 * (model$params[[l]]$a[j, 1] - y[j])
    }
    return(out)
  }

  cl <- makeCluster(detectCores(), setup_strategy = "sequential")

  clusterExport(cl,
                varlist = c("mlp",
                            "model",
                            "sig_prime",
                            "sig_prime_z",
                            "dc_da"),
                envir = environment())

  samples <- parLapply(cl, training_set, function(x) {
    pred <- mlp(model, x$input)
    n <- sapply(pred$params, function(x) nrow(x$a))
    grad <- NULL
    for (l in rev(seq(2, length(pred$params)))) {
      for (j in seq_len(nrow(pred$params[[l]]$w))) {
          pred$params[[l]]$dc_da[j, 1] <- dc_da(pred, n, l, j, x$output)
          pred$params[[l]]$sig_prime_z[j, 1] <- sig_prime_z(pred, l, j)
          pred$params[[l]]$grad[j, ncol(pred$params[[l]]$w) + 1] <-
            pred$params[[l]]$sig_prime_z[j, 1] * pred$params[[l]]$dc_da[j, 1]
        for (k in seq_len(ncol(pred$params[[l]]$w))) {
          pred$params[[l]]$grad[j, k] <-
            pred$params[[l - 1]]$a[k, 1] *
            pred$params[[l]]$grad[j, ncol(pred$params[[l]]$w) + 1]
        }
      }
      grad <- c(grad, as.vector(pred$params[[l]]$grad))
    }
    return(grad)
  })

  stopCluster(cl)

  npar <- sapply(model$params, function(x) {
    y <- nrow(x$w) * ncol(x$w) + nrow(x$b)
    if (length(y) == 0) NA else y
  })

  model$grad <- apply(simplify2array(samples), 1, mean)
  return(model)
}

train <- function(model, training_set, control = list()) {

  model2vec <- function(model) {
    params <- model$params
    npar <- sapply(params, function(x) {
      y <- nrow(x$w) * ncol(x$w) + nrow(x$b)
      if (length(y) == 0) NA else y
    })
    params_vec <- rep(NaN, sum(npar, na.rm = TRUE))
    start_index <- 1
    for (l in seq(2, length(params))) {
      nweights <- nrow(params[[l]]$w) * ncol(params[[l]]$w)
      nbiases <- nrow(params[[l]]$b)
      range_1 <- start_index:(start_index + nweights - 1)
      range_2 <- (start_index + nweights):(start_index + nweights + nbiases - 1)
      params_vec[range_1] <- params[[l]]$w
      params_vec[range_2] <- params[[l]]$b
      start_index <- start_index + nweights + nbiases
    }
    return(params_vec)
  }

  vec2model <- function(params_vec, model) {
    start_index <- 1
    params <- model$params
    for (l in seq(2, length(params))) {
      nweights <- nrow(params[[l]]$w) * ncol(params[[l]]$w)
      nbiases <- nrow(params[[l]]$b)
      range_1 <- start_index:(start_index + nweights - 1)
      range_2 <- (start_index + nweights):(start_index + nweights + nbiases - 1)
      params[[l]]$w <- matrix(params_vec[range_1],
                              nrow(params[[l]]$w),
                              ncol(params[[l]]$w))
      params[[l]]$b <- matrix(params_vec[range_2],
                              nrow(params[[l]]$b),
                              1)
      start_index <- start_index + nweights + nbiases
    }
    model$params <- params
    return(model)
  }

  opt <- optim(par = model2vec(model),
               fn = function(params_vec, model, data) {
                 cost(vec2model(params_vec, model), training_set)$cost
               },
               gr =  function(params_vec, model, data) {
                 del_cost(vec2model(params_vec, model), training_set)$grad
               },
               model = model,
               data = training_set,
               method = "BFGS",
               lower = -Inf,
               upper = Inf,
               control = control,
               hessian = FALSE)

  model <- vec2model(opt$par, model)
  model$opt <- opt

  return(model)
}
