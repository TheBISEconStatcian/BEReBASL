#' @title Synthetic data generation
#' 
#' @description Generates synthetic credit scoring data with a binary target.
#' 
#' @details 
#' This function generates synthetic credit scoring data with a binary target (GOOD/BAD).
#'
#' The number of variables and sample size is set by a user. The simulated data contains:
#' - continuous features (e.g. income) - generated using multivariate Gaussin distribution;
#' - categrocial features (e.g. PLZ) - generated using latent variables based on continuous features;
#' - binary features (e.g. gender) - generated using binomial distribution; 
#'
#' The complexity of the data generating process is controlled by:
#' - number of non-linear target-feature relationships (squared and logged terms);
#' - number and complexity of categorical and binary features;
#' - distribution differences between GOOD and BAD cases.
#'
#' The data is generated separately for GOOD and BAD cases, using multivariate distributions
#' with different vectors of means and/or different variance/covariance matrices. The data 
#' generation framework is implemented as follows:
#' 1. Generate binary features and find all unique combinations of values appearing in the data.
#' 2. Generate continuous features based on specified differences between classes and dummies.
#' 3. Generate categrocial features based on latent variables computed from the continuous features.
#' 4. Add random noise to continuous features.
#'
#' The ouput of the function is a list with data.frame and distribution parameters.
#' 
#' Argument groups:
#' - dimenionality: n, k_con, k_cat, k_bin, k_noise, bad_ratio
#' - continuous features: con_mean_bad_dif, con_var_bad_dif, covars, con_nonlinear, con_noise_var, iid
#' - categrocial features: cat_levels, cat_var_share, cat_nonlinear, cat_noise_var
#' - binary features: bin_prob, bin_bad_ratio, bin_mean_con_dif, bin_mean_bad_dif, bin_var_bad_dif, bin_noise_var
#' - other parameters: replicate, seed, verbose, encode_factors
#' 
#' @param n (integer): total number of generated examples.
#' @param k_con (integer): number of generated continuous features.
#' @param k_cat (integer): number of generated categorical features.
#' @param k_bin (integer): number of generated binary features.
#' @param k_noise (integer): number of white-noise features: e ~ N(0, 1).
#' @param bad_ratio (float): percentage of BAD examples.
#'
#' @param con_mean_bad_dif (float/vector): difference between distribution means between classes.
#' @param con_var_bad_dif (float): percentage of var/covar matrix elements that are different between classes.
#' @param covars (list): list with specified var-covar matrices. Default is NULL.
#' @param con_nonlinear (float): percentage of continuous features, for which the specified mean and covariance are set for the nonlinear transformation of the feature.
#' @param con_noise_var (float): set con_noise_var > 0 to add e ~ N(0, con_noise_var) to each feature.
#' @param iid (logical): set iid = T to override covariance-related paramaters and generate IID features.
#'
#' @param cat_levels (integer/vector): number of unique values for categorical features.
#' @param cat_var_share (float/vector): percentage of continuous features used to construct latent variables.
#' @param cat_nonlinear (float): percentage of continuous features that have a nonlinear relationship with the latent variable.
#' @param cat_noise_var (float): set cat_noise_var > 0 to add e ~ N(0, cat_noise_var) to each latent variable before computing to a categorcal feature.
#'
#' @param bin_prob (float/vector): P(dummy = 1) for the underlying binomial distribution. 
#' @param bin_bad_ratio (float/vector): percentage of BAD examples for examples with dummy = 1.
#' @param bin_mean_con_dif (float/vector): shifts in distribution means for examples with different combination of values of binary features for both classes.
#' @param bin_mean_bad_dif (float/vector): additional differences between distribution means between classes for examples with different combination of values of binary features.
#' @param bin_var_bad_dif (float/vector): percentage of var/covar matrix elements that are different for examples with different combination of values of binary features.
#' @param bin_noise_var (float): set bin_noise_var > 0 to flip value of each dummy variable with probability = bin_noise_var.
#' 
#' @param replicate (list): object exported during the previous simulation. Can be used for generating a new sample with the same properties as the saved data. If not NULL, overrieds all function arguments except for dimensionality.
#' @param seed (integer): random seed for the data generator.
#' @param verbose (logical): set to TRUE to display feedback during data generation.
#' @param encode_factors (character): set to 'dummy' or 'label' to encode factors. Default is FALSE.
#' 
#' @return 
#' List with 5 elements:
#' - data: data.frame containg the generated synthetic data
#' - params: list of synthtic data parameters
#' - call: saved function call
#' - arguments: saved fuction arguments
#' - time: time to generate the synthetic data
#' 
genCreditData <- function(n                = 1000, 
                          k_con            = 10, 
                          k_cat            = 5, 
                          k_bin            = 2, 
                          k_noise          = 0, 
                          bad_ratio        = 0.5,
                          con_mean_bad_dif = 1, 
                          con_var_bad_dif  = 0, 
                          con_nonlinear    = 0.5, 
                          con_noise_var    = 0.1,
                          covars           = NULL,
                          iid              = F,
                          mixture          = F,
                          mix_mean_dif     = NULL,
                          mix_var_dif      = NULL,
                          cat_levels       = 5,
                          cat_nonlinear    = 0,
                          cat_var_share    = 0.1, 
                          cat_noise_var    = 0.1,
                          bin_prob         = 0.5, 
                          bin_mean_bad_dif = 0, 
                          bin_bad_ratio    = bad_ratio,
                          bin_mean_con_dif = 0, 
                          bin_var_bad_dif  = con_var_bad_dif, 
                          bin_noise_var    = 0.1, 
                          encode_factors   = F, 
                          verbose          = T, 
                          seed             = NULL,
                          replicate        = NULL) {
  
  
  ###################################
  #                                 
  #         1. PREPARATIONS         
  #                                 
  ###################################
  
  # random seed
  if (!is.null(seed)) {set.seed(seed)}
  
  # start timer
  t.start <- proc.time()
  
  # set arguments if replicating old DGP
  if (!is.null(replicate)) {
    n_new    <- n
    seed_new <- seed
    for (obj in 1:length(replicate$arguments)) {
      assign(names(replicate$arguments)[obj], replicate$arguments[[obj]])
    }
    n    <- n_new
    seed <- seed_new
  }
  
  # check length errors
  if (length(con_mean_bad_dif) != 1 & length(con_mean_bad_dif) != k_con) {stop('con_mean_bad_dif requires a vector of length 1 or k_con')}
  if (length(cat_levels)       != 1 & length(cat_levels)       != k_cat) {stop('cat_levels requires a vector of length 1 or k_cat')}
  if (length(cat_var_share)    != 1 & length(cat_var_share)    != k_cat) {stop('cat_var_share requires a vector of length 1 or k_cat')}
  if (length(bin_prob)         != 1 & length(bin_prob)         != k_bin) {stop('bin_prob requires a vector of length 1 or k_bin')}
  if (length(bin_bad_ratio)    != 1 & length(bin_bad_ratio)    != k_bin) {stop('bin_bad_ratio requires a vector of length 1 or k_bin')}
  if (length(bin_mean_con_dif) != 1 & length(bin_mean_con_dif) != k_bin) {stop('bin_mean_con_dif requires a vector of length 1 or k_bin')}
  if (length(bin_mean_bad_dif) != 1 & length(bin_mean_bad_dif) != k_bin) {stop('bin_mean_bad_dif requires a vector of length 1 or k_bin')}
  if ((mixture == T) & (is.null(mix_mean_dif) | is.null(mix_var_dif)))   {stop('mix_mean_dif and mix_var_dif have to be set if mixture = True')}
  
  # check other errors
  if (k_con < 1) {stop('At least one continuous feature is required')}
  
  # display info
  k <- k_con + k_cat + k_bin + k_noise
  if (verbose == T) {print(paste0('Simulating (', n, ' x ', k, ') data set:'))}
  
  
  
  ###################################
  #                                 
  #     2. GENERATIUNG NEW DATA     
  #                                 
  ###################################
  
  ###################################
  #                                 #
  #       2.1 BINARY FEATURES       #
  #                                 #
  ###################################
  
  # display info
  if (verbose == T & k_bin > 0) {print('- Generating dummy features...')}
  
  # placeholder
  bin_params <- NULL
  
  # generate matrix with dummies
  if (k_bin > 0) {
    
    # create vector with probs
    if (length(bin_prob) == 1) {
      bin_prob <- rep(bin_prob, k_bin)
    }
    
    # generate data
    dummy <- NULL
    for (d in 1:k_bin) {
      dummy <- cbind(dummy, rbinom(n, 1, bin_prob[d]))
    }
    
    # convert to data.frame
    dummy <- as.data.frame(dummy)
    colnames(dummy) <- paste0('D', 1:k_bin)
    
    # inject noise
    for (var in colnames(dummy)) {
      dummy[[var]] <- abs(dummy[[var]] - rbinom(n, 1, bin_noise_var))
    }
    
    # find unique combinations
    combos <- unique(apply(dummy, 1, function(x) paste(x, collapse = '')))
  }
  
  # create artificial dummy if necessary
  if (k_bin == 0) {
    dummy  <- data.frame(D1 = rep(1, n))
    combos <- '1' 
  }
  
  
  ###################################
  #                                 
  #     2.2. CONTINUOUS FEATURES    
  #                                 
  ###################################
  
  # display info
  if (verbose == T & k_con > 0) {print('- Generating continuous features...')}
  
  # placeholder
  con_params <- list(combo = list(), means = list(), covar = list(), nonlinearities = NULL)
  
  # create vectors with mean differences
  if (length(bin_mean_con_dif) == 1) {bin_mean_con_dif <- rep(bin_mean_con_dif, k_bin)}
  if (length(bin_mean_bad_dif) == 1) {bin_mean_bad_dif <- rep(bin_mean_bad_dif, k_bin)}
  if (length(con_mean_bad_dif) == 1) {con_mean_bad_dif <- rep(con_mean_bad_dif, k_con)}
  if (length(bin_bad_ratio)    == 1) {bin_bad_ratio    <- rep(bin_bad_ratio,    k_bin)}
  
  # loop for all combinations of dummies
  for (combo in combos) {
    
    # distribution parameters
    combo_vals      <- as.numeric(unlist(strsplit(combo, split = '')))
    combo_idx       <- apply(dummy, 1, function(x) all(x == combo_vals))
    combo_count     <- sum(combo_idx)
    if (sum(combo_vals) > 0 & k_bin != 0) {
      combo_bad_ratio <- mean(bin_bad_ratio * combo_vals)
    }else{
      combo_bad_ratio <- bad_ratio
    }
    
    # compute sample size
    combo_n1 <- round(combo_count * combo_bad_ratio)
    combo_n2 <- round(combo_count * (1 - combo_bad_ratio))
    if (combo_n1 + combo_n2 != combo_count) {
      cid <- rbinom(1, 1, 0.5)
      if (cid == 0) {
        combo_n1 <- combo_count - combo_n2
      }else{
        combo_n2 <- combo_count - combo_n1
      }
    } 
    
    # generate specified or load saved means
    if (is.null(replicate)) {
      mu1 <- rep(sum(combo_vals * bin_mean_con_dif), k_con)
      mu2 <- mu1 + con_mean_bad_dif + sum(combo_vals * bin_mean_bad_dif)
      if (mixture == T) {
        mu1x <- mu1 + mix_mean_dif
        mu2x <- mu2 + mix_mean_dif
      }
    }else{
      mu1 <- replicate$params$continuous$means[[which(combos == combo)]][1, ]
      mu2 <- replicate$params$continuous$means[[which(combos == combo)]][2, ]
      if (mixture == T) {
        mu1x <- replicate$params$continuous$means_x[[which(combos == combo)]][1, ]
        mu2x <- replicate$params$continuous$means_x[[which(combos == combo)]][2, ]
      }
    }
    
    # generate diagonal covariance matrices
    if (iid == T & is.null(replicate)) {
      sigma1 <- diag(k_con)
      sigma2 <- diag(k_con)
      if (mixture == T) {
        sigma1x <- diag(k_con)
        sigma2x <- diag(k_con)
      }
    }
    
    # generate random covariance matrices
    if (iid == F & is.null(replicate) & is.null(covars)) {
      
      # generate matrices
      sigma1 <- genPositiveDefMat(k_con, 'unifcorrmat', rangeVar = c(0, 1))$Sigma
      sigma2 <- genPositiveDefMat(k_con, 'unifcorrmat', rangeVar = c(0, 1))$Sigma
      
      # replace some values for class 2
      if (sum(combo_vals) > 0) {
        combo_var_dif <- mean(bin_var_bad_dif * combo_vals)
      }else{
        combo_var_dif <- con_var_bad_dif
      }
      sigma_index <- as.logical(rbinom((k_con^2-k_con)/2, 1, prob = combo_var_dif))
      upperTriangle(sigma2, byrow = F)[!sigma_index] <- upperTriangle(sigma1, byrow = F)[!sigma_index]
      lowerTriangle(sigma2, byrow = T)[!sigma_index] <- lowerTriangle(sigma1, byrow = T)[!sigma_index]
      sigma_index <- as.logical(rbinom(k_con, 1, prob = combo_var_dif))
      diag(sigma2)[!sigma_index] <- diag(sigma1)[!sigma_index]
      
      # loop until sigma2 becomes positive semi-definite
      ev <- eigen(sigma2, symmetric = T)$values
      while (!all(ev >= -sqrt(.Machine$double.eps) * abs(ev[1]))) {
        sigma2 <- genPositiveDefMat(k_con, 'unifcorrmat', rangeVar = c(0, 1))$Sigma
        sigma_index <- as.logical(rbinom((k_con^2-k_con)/2, 1, prob = combo_var_dif))
        upperTriangle(sigma2, byrow = F)[!sigma_index] <- upperTriangle(sigma1, byrow = F)[!sigma_index]
        lowerTriangle(sigma2, byrow = T)[!sigma_index] <- lowerTriangle(sigma1, byrow = T)[!sigma_index]
        sigma_index <- as.logical(rbinom(k_con, 1, prob = combo_var_dif))
        diag(sigma2)[!sigma_index] <- diag(sigma1)[!sigma_index]
        ev <- eigen(sigma2, symmetric = T)$values
      }
    }
    
    # load saved covariance matrices
    if (!is.null(replicate)) {
      sigma1 <- replicate$params$continuous$covar[[which(combos == combo)]][[1]]
      sigma2 <- replicate$params$continuous$covar[[which(combos == combo)]][[2]]
      if (mixture == T) {
        sigma1x <- replicate$params$continuous$covar_x[[which(combos == combo)]][[1]]
        sigma2x <- replicate$params$continuous$covar_x[[which(combos == combo)]][[2]]
      }
    }
    
    # load saved covariance matrices
    if (!is.null(covars)) {
      sigma1 <- covars[[1]]
      sigma2 <- covars[[2]]
    }
    
    # add mixture covariances
    if (mixture == T) {
      sigma1x <- sigma1 + mix_var_dif
      sigma2x <- sigma2 + mix_var_dif
    }
    
    # generate feature matrix
    set.seed(seed)
    X1 <- NULL
    X2 <- NULL
    if (combo_n1 > 0) {
      X1 <- as.data.frame(rmvnorm(round(combo_n1/2), mean = mu1, sigma = sigma1))
      if (mixture == T) {
        X1x <- as.data.frame(rmvnorm(round(combo_n1/2), mean = mu1x, sigma = sigma1x))
        X1  <- rbind(X1, X1x)
      }
    }
    if (combo_n2 > 0) {
      X2 <- as.data.frame(rmvnorm(round(combo_n2/2), mean = mu2, sigma = sigma2))
      if (mixture == T) {
        X2x <- as.data.frame(rmvnorm(round(combo_n2/2), mean = mu2x, sigma = sigma2x))
        X2  <- rbind(X2, X2x)
      }
    }
    
    # generate classes
    if (!is.null(X1)) {X1$BAD <- 'BAD'}
    if (!is.null(X2)) {X2$BAD <- 'GOOD'}
    
    # merge data frames
    X <- rbind(X1, X2)
    X$BAD <- as.factor(X$BAD)
    
    # merge dummies
    if (which(combos == combo) == 1) {
      df <- cbind(X, dummy[combo_idx, ])
    }else{
      df <- rbind(df, cbind(X, dummy[combo_idx, ]))
    }
    
    # save parameters
    con_params$combo[[which(combos == combo)]] <- combo_vals
    con_params$means[[which(combos == combo)]] <- rbind(mu1, mu2)
    con_params$covar[[which(combos == combo)]] <- list(sigma1, sigma2)
    if (mixture == T) {
      con_params$means_x[[which(combos == combo)]] <- rbind(mu1x, mu2x)
      con_params$covar_x[[which(combos == combo)]] <- list(sigma1x, sigma2x)
    }
  }
  
  # adjust colnames
  if (k_bin == 1) {
    colnames(df)[colnames(df) == 'dummy[combo_idx, ]'] <- 'D1'
  }
  if (k_bin == 0) {
    df['dummy[combo_idx, ]'] <- NULL
  }
  
  # inject nonlinearity
  if (con_nonlinear > 0 & round(con_nonlinear * k_con) > 0) {
    
    # find features and treatment
    nonlinear_vars <- sample(k_con)[1:round(con_nonlinear * k_con)]
    nonlinear_vars <- paste0('V', nonlinear_vars)
    nonlinear_type <- 1 + rbinom(length(nonlinear_vars), 1, 0.5)
    nonlinear_desc <- c('square', 'cube')[nonlinear_type]
    nonlinearities <- paste0(nonlinear_vars, ': ', nonlinear_desc, collapse = ', ')
    
    # transform features
    for (var in nonlinear_vars) {
      if (nonlinear_type[which(nonlinear_vars == var)] == 1) {df[[var]] <- df[[var]]^2}
      if (nonlinear_type[which(nonlinear_vars == var)] == 2) {df[[var]] <- df[[var]]^3}
    }
  }else{
    nonlinearities <- NULL
  }
  
  # save parameters
  con_params$nonlinearities <- nonlinearities
  
  # inject noise
  for (var in colnames(df)[grepl('V', colnames(df))]) {
    df[[var]] <- df[[var]] + rnorm(n, 0, con_noise_var)
  }
  
  
  ###################################
  #                                 
  #     2.3. CATEGORICAL FEATURES    
  #                                 
  ###################################
  
  # display info
  if (verbose == T & k_cat > 0) {print('- Generating categorical features...')}
  
  # placeholder
  cat_params <- list(equation = list(), breaks = list())
  
  # generate categorical features
  if (k_cat > 0) {
    
    # create vectors
    if (length(cat_var_share) == 1) {
      cat_var_share <- rep(cat_var_share, k_cat)
    }
    if (length(cat_levels)    == 1) {
      cat_levels <- rep(cat_levels, k_cat)
    }
    
    # loop for categrocial features
    for (c in 1:k_cat) {
      
      # select random set of continuous variables
      num_vars <- data.frame(df[, colnames(df)[grepl('V', colnames(df))]])
      num_var_names <- sample(k_con)[1:round(cat_var_share[c] * k_con)]
      num_var_names <- paste0('V', num_var_names)
      num_vars <- data.frame(num_vars[, num_var_names])
      
      # inject nonlinearity
      if (cat_nonlinear > 0 & round(cat_nonlinear * ncol(num_vars)) > 0) {
        
        # find features and treatment
        nonlinear_vars <- sample(ncol(num_vars))[1:round(cat_nonlinear * ncol(num_vars))]
        nonlinear_type <- 1 + rbinom(length(nonlinear_vars), 1, 0.5)
        nonlinear_desc <- c('square', '1/x')
        for (v in 1:length(nonlinear_vars)) {
          if (nonlinear_desc[nonlinear_type[v]] == 'square') {
            num_var_names[nonlinear_vars[v]] <- paste0('(', num_var_names[nonlinear_vars[v]], ')^2')
          }
          if (nonlinear_desc[nonlinear_type[v]] == 'cube') {
            num_var_names[nonlinear_vars[v]] <- paste0('(', num_var_names[nonlinear_vars[v]], ')^3')
          }
        }
        colnames(num_vars) <- num_var_names
        
        # transform features
        for (v in 1:length(nonlinear_vars)) {
          if (grepl('^2', var) | grepl('^3', var)) {
            if (nonlinear_type[v] == 1) {num_vars[[var]] <- num_vars[[num_var_names[nonlinear_vars[v]]]]^2}
            if (nonlinear_type[v] == 2) {num_vars[[var]] <- 1/num_vars[[num_var_names[nonlinear_vars[v]]]]}
          }
        }
      }      
      
      # compute latent variables with random coefficients
      coefs   <- round(rnorm(ncol(num_vars), mean = 0, sd = 1), digits = 4)
      lat_var <- as.matrix(num_vars) %*% coefs + rnorm(nrow(num_vars), 0, cat_noise_var)
      cat_var <- cut(lat_var, breaks = cat_levels[c])
      
      # save information
      cat_params$breaks[[c]] <- levels(cat_var)
      cat_params$equation[[c]] <- paste0('C', c, ' = ', paste0(coefs, '*', num_var_names, collapse = ' + '), 
                                         ' + E~(0,', cat_noise_var, ')')
      
      # add categrocial feature to data
      levels(cat_var) <- paste0('L', 1:cat_levels[c])
      df[[paste0('C', c)]] <- cat_var
    }
  }
  
  
  ###################################
  #                                 
  #       2.4. NOISY FEATURES       
  #                                 
  ###################################
  
  # display info
  if (verbose == T & k_noise > 0) {print('- Generating noisy features...')}
  
  # generate noise
  if (k_noise > 0) {
    for (k in 1:k_noise) {
      df[[paste0('N', k)]] <- rnorm(n, 0, 1)
    }
  }
  
  
  
  ###################################
  #                                 
  #       3. FINAL PROCESSING       
  #                                 
  ###################################
  
  # sort columns
  df <- df[, c(colnames(df)[!colnames(df) %in% 'BAD'], 'BAD')]
  rownames(df) <- paste0('id', 1:nrow(df))
  
  # encoding factors
  if (encode_factors != F) {
    df <- encodeFactors(df, target = 'BAD', method = encode_factors)
  }
  
  # renacme V to F
  colnames(df)[colnames(df) %in% paste0('V', 1:k_con)] <- paste0('X', 1:k_con)
  
  # save simulation parameters
  pars <- list(continuous = con_params, categorical = cat_params, binary = bin_params)
  call <- match.call()
  args <- list(n                = n,
               k_con            = k_con, 
               k_cat            = k_cat, 
               k_bin            = k_bin, 
               k_noise          = k_noise, 
               bad_ratio        = bad_ratio,
               con_mean_bad_dif = con_mean_bad_dif, 
               con_var_bad_dif  = con_var_bad_dif, 
               con_nonlinear    = con_nonlinear, 
               con_noise_var    = con_noise_var,
               mixture          = mixture,
               mix_mean_dif     = mix_mean_dif,
               mix_var_dif      = mix_var_dif,
               cat_levels       = cat_levels, 
               cat_var_share    = cat_var_share, 
               cat_nonlinear    = cat_nonlinear,
               cat_noise_var    = cat_noise_var,
               bin_prob         = bin_prob, 
               bin_mean_bad_dif = bin_mean_bad_dif, 
               bin_bad_ratio    = bin_bad_ratio,
               bin_mean_con_dif = bin_mean_con_dif, 
               bin_var_bad_dif  = bin_var_bad_dif, 
               bin_noise_var    = bin_noise_var, 
               encode_factors   = encode_factors, 
               verbose          = verbose, 
               seed             = seed)
  
  # stop timer
  t.all <- paste0(round((((proc.time() - t.start))[3]), digits = 0), ' seconds')
  if (verbose == T) {print(paste0('Finished. Elapsed time: ', t.all))}
  
  # construct list
  data <- list(data      = df, 
               params    = pars, 
               arguments = args, 
               call      = call, 
               time      = t.all)
  return(data)
}
