##################################
#                                
#        BAYESIAN METRICS
#                                
##################################

#' @title Compute Bayesian metric using Monte-carlo simulations
#' 
#' @description Computes evaluation metrics on accepts and rejects using Monte-Carlo 
#' simulation where P(BAD) for unlabeled examples is drawn from a specified distribution 
#' in each simulation.
#' 
#' @param metric (function): evaluation metric in the mlr package format
#' @param model (mlr model): trained model in the mlr package format
#' @param accepts (data.frame): data set of accepts (validation sample)
#' @param rejects (data.frame): data set of rejects
#' @param rejects_prior (vector): vector of scores of rejects assuming BAD = 1
#' @param min_iterations (int): min number of Monte-Carlo iterations
#' @param max_iterations (int): max number of Monte-Carlo iterations
#' @param epsilon (float): stop simulations if metric does not change by epsilon
#' @param seed (int): random seed
#' 
#' @return metric value
#' 
bayesianMetric <- function(metric, 
                            model, 
                            accepts, 
                            rejects, 
                            rejects_prior, 
                            min_iterations = 10^2,
                            max_iterations = 10^4,
                            epsilon        = 10^(-4),
                            seed = 1) {
  
  # seed
  set.seed(seed)
  
  # predictions
  preds_acc <- predict(model, newdata = accepts)
  preds_rej <- predict(model, newdata = rejects)
  
  # monte-carlo simulations
  vals <- rep(NA, max_iterations)
  for (n in 1:max_iterations) {
    
    # assign labels to rejects
    preds_tmp <- preds_rej
    preds_tmp$data$truth <- rbinom(n    = nrow(rejects),
                                   size = 1,
                                   prob = rejects_prior)
    preds_tmp$data$truth <- ifelse(preds_tmp$data$truth == 1, 'BAD', 'GOOD')
    
    # merge accepts and rejects
    preds_tmp$data <- rbind(preds_acc$data, preds_tmp$data)
    
    # compute metric
    vals[n] <- mlr::performance(preds_tmp, metric)
    
    # check early stop
    if (n >= min_iterations) {
      dmean <- abs(mean(vals[1:n]) - mean(vals[1:(n-1)]))
      if (dmean < epsilon) {
        vals <- vals[1:n]
        break
      }
    }
  }

  # return vector of values
  return(mean(vals))
}
