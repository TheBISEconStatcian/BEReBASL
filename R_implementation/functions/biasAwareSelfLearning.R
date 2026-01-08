#' @title Bias-Aware Self-Learning
#' 
#' @description Performs reject inference with bias-aware self-learning
#' 
#' @details 
#' This function performs reject inference using bias-aware self-learning framework.
#' 
#' Bias-aware self-learning (BASL) is a reject inference framework that mitigates the effect 
#' of sampling bias on model training. BASL is based on semi-supervised self-learning framework
#' and introduces extensions to adapt it to a setting with sampling bias. BASL introduces 
#' a filtering stage before labeling and incorporates distinct regimes for labeling rejects 
#' and training a scorecard to ensure that the scorecard is model-agnostic. The labeling 
#' regime includes multiple steps to account for bias and reduce the risk of error propagation
#' such as: sampling rejects on each labeleing iteration, using imbalance multiplier to 
#' account for a higher bad ratio among rejects, perfoemance-based early stopping using a 
#' Bayesian evaluation framework.
#' 
#' @param accepts (data.frame): data set containing accepted applications
#' @param rejects (data.frame): data set containing rejected applications
#' @param target (character): name of the target variable
#' @param filtering_beta (float): percentiles of rejects to be filtered
#' @param weak_learner (character): weak base learner (mlr format)
#' @param strong_learner (character): strong base learner (mlr format)
#' @param sampling_percent (float): percents of rejects to be sampled
#' @param labeling_percent (float): percent of rejects to be labeled 
#' @param multiplier (float): imbalance multiplier
#' @param max_iterations (int): max. number of labeling iteration
#' @param early_stop (logical): whether to use performance-based early stopping
#' @param holdout_percent (float): percentage of data in the holdout sample
#' @param silent (logical): whether to print training details
#' 
#' @return labels of rejected applications
#' 
biasAwareSelfLearning <- function(accepts,
                                  rejects,
                                  target           = 'BAD', 
                                  filtering_beta   = c(0, 1),
                                  weak_learner     = 'classif.logreg',
                                  strong_learner   = 'classif.logreg',
                                  holdout_percent  = 0.1,
                                  sampling_percent = 1,
                                  labeling_percent = 0.01, 
                                  multiplier       = 1, 
                                  max_iterations   = 3,
                                  early_stop       = F,
                                  silent           = F) {
  
  ####### PREPARATION
  
  # random seed
  set.seed(7)
  
  # iterations
  iter <- 1
  
  # percentage
  per_b <- labeling_percent
  per_g <- labeling_percent / multiplier
  
  # performance vector
  perf_vector <- rep(0, max_iterations + 1)
  
  # data partitioning
  rejects[[target]] <- NA
  if (early_stop == T) {
    holdout_idx       <- sample(nrow(accepts))[1:round(holdout_percent*nrow(accepts))]
    holdout_accepts   <- accepts[ holdout_idx, ]
    accepts           <- accepts[-holdout_idx, ]
    holdout_idx       <- sample(nrow(rejects))[1:round(holdout_percent*nrow(rejects))]
    holdout_rejects   <- rejects[ holdout_idx, ]
    rejects           <- rejects[-holdout_idx, ]
  }
  train    <- accepts
  test     <- rejects
  raw_test <- rejects
  rm(list = c('accepts', 'rejects'))
  
  # drop target levels
  train[[target]] <- droplevels(train[[target]])
  
  # modeling task
  ri_tsk <- makeClassifTask(data = train, target = target, positive = target)
  
  # set up learners
  if (weak_learner == 'classif.glmnet') {
    weak_learner <- makeLearner('classif.glmnet', predict.type = 'prob', alpha = 1)
  }else{
    weak_learner <- makeLearner(weak_learner, predict.type = 'prob')
  }
  strong_learner  <- makeLearner(strong_learner,  predict.type = 'prob')
  
  # evaluation params
  bm_metric   <- mlr::auc
  bm_min_iter <- 10^2
  bm_max_iter <- 10^4
  bm_epsilon  <- 10^4
  
  
  ####### FILTERING REJECTS
  
  # filter if parameters are set
  if ((filtering_beta[1] > 0) | (filtering_beta[2] < 1)) {
    
    # filter rejects
    filter_idx <- filteringStage(accepts   = train,
                                 rejects   = test,
                                 beta      = filtering_beta,
                                 num_trees = 100)
    
    # drop filtered cases
    test     <- test[filter_idx,     ]
    raw_test <- raw_test[filter_idx, ]
  }
  
  
  ####### TRAINING STAGE
  
  # train if early stop is enabled
  if (early_stop == T) {
    
    # train and predict with strong learner
    strong_model  <- mlr::train(learner = strong_learner, task = ri_tsk)
    strong_preds  <- predict(strong_model, newdata = test)
    strong_pr_id  <- rownames(strong_preds$data)
    strong_preds  <- strong_preds$data$prob.BAD
    
    # produce prior for rejects
    rejects_preds <- predict(strong_model, newdata = holdout_rejects)
    
    # evaluate strong learner
    perf_vector[iter] <- bayesianMetric(metric         = bm_metric,
                                        model          = strong_model,
                                        accepts        = holdout_accepts,
                                        rejects        = holdout_rejects,
                                        rejects_prior  = rejects_preds$data$prob.BAD,
                                        min_iterations = bm_min_iter,
                                        max_iterations = bm_max_iter,
                                        epsilon        = bm_epsilon,
                                        seed           = 1)
  }
  
  
  ####### ASSIGNING LABELS: ITERATION 1
  
  # train and predict with weak learner
  model <- mlr::train(learner = weak_learner, task = ri_tsk)
  preds <- predict(model, newdata = test)
  pr_id <- rownames(preds$data)
  preds <- preds$data$prob.BAD
  
  # sampling rejects
  sampling_idx <- sample(nrow(test))[1:round(sampling_percent*nrow(test))]
  if (sampling_percent == 1) {sampling_idx = 1:nrow(test)}
  preds <- preds[sampling_idx]
  pr_id <- pr_id[sampling_idx]
  
  # set thresholds 
  con_b <- quantile(preds, 1 - per_b)
  con_g <- 1 - quantile(preds, per_g)
  
  # find confident predictions
  confident_preds        <- preds[preds >= con_b | preds <= (1 - con_g)]
  names(confident_preds) <- pr_id[preds >= con_b | preds <= (1 - con_g)]
  confidents_labs        <- confident_preds
  confidents_labs[confident_preds >= con_b]       <- 'BAD'
  confidents_labs[confident_preds <= (1 - con_g)] <- 'GOOD'
  confident_preds <- confidents_labs
  
  # save confident predictions
  confident_labels <- confident_preds
  
  # information
  if (silent == F) {
    print(paste0('-- iteration ', iter, ': appended ', 
                 length(confident_preds), ' cases (' ,
                 sum(confident_preds == 'BAD'),  ' BAD + ', 
                 sum(confident_preds == 'GOOD'), ' GOOD)'))
  }
  
  
  ####### ASSIGNING LABELS: NEXT ITERATIONS
  
  # self-learning loop
  while ((length(confident_preds) > 0) & nrow(test) > 0 & iter < max_iterations) {
    
    ##### LABELING REJECTS
    
    # extract new data
    new_train <- test[names(confident_preds), ]
    test      <- rbind((test[sampling_idx, ])[preds < con_b & preds > (1 - con_g), ],
                       test[-sampling_idx, ])
    
    # append selected labeled rejects
    new_train[[target]] <- confident_preds
    train               <- rbind(train, new_train)
    
    # check if test is not empty
    if (nrow(test) == 0) {
      if (silent == F) {
        print('-- early stopping: no more confident predictions')
      }
      break
    }
    
    # modeling task
    ri_tsk <- makeClassifTask(data = train, target = target, positive = target)
    
    # train and predict with weak learner
    model <- mlr::train(learner = weak_learner, task = ri_tsk)
    preds <- predict(model, newdata = test)
    pr_id <- rownames(preds$data)
    preds <- preds$data$prob.BAD
    
    
    ##### TRAINING AND EARLY STOPPING
    
    # train if early stop is enabled
    if (early_stop == T) {
      
      # train and predict with strong learner
      strong_model <- mlr::train(learner = strong_learner, task = ri_tsk)
      strong_preds <- predict(strong_model, newdata = test)
      strong_pr_id <- rownames(strong_preds$data)
      strong_preds <- strong_preds$data$prob.BAD
      
      # evaluate strong learner
      perf_vector[iter + 1] <- bayesianMetric(metric         = bm_metric,
                                              model          = strong_model,
                                              accepts        = holdout_accepts,
                                              rejects        = holdout_rejects,
                                              rejects_prior  = rejects_preds$data$prob.BAD,
                                              min_iterations = bm_min_iter,
                                              max_iterations = bm_max_iter,
                                              epsilon        = bm_epsilon,
                                              seed           = 1)
      
      # check early stopping criteria
      if (perf_vector[iter + 1] <= perf_vector[iter]) {
        if (silent == F) {
          print('-- early stopping: performance does not improve')
        }
        break
      }
    }
    
    
    ##### APPENDING SELECTED LABELED REJECTS
    
    # sampling rejects
    sampling_idx <- sample(nrow(test))[1:round(sampling_percent*nrow(test))]
    if (sampling_percent == 1) {sampling_idx = 1:nrow(test)}
    preds <- preds[sampling_idx]
    pr_id <- pr_id[sampling_idx]
    
    # find confident predictions
    # confident_preds        <- preds[preds >= con_b | preds <= (1 - con_g)]
    # names(confident_preds) <- pr_id[preds >= con_b | preds <= (1 - con_g)]
    # confidents_labs        <- confident_preds
    # confidents_labs[confident_preds >= con_b]       <- 'BAD'
    # confidents_labs[confident_preds <= (1 - con_g)] <- 'GOOD'
    # confident_preds <- confidents_labs
    confident_good <- preds[preds <= (1 - con_g)]
    names(confident_good) <- pr_id[preds <= (1 - con_g)]
    if (length(confident_good) > round(per_g*length(preds))) {
      confident_good <- confident_good[sample(length(confident_good))[1:round(per_g*length(preds))]]
    }
    confident_bad <- preds[preds >= (con_b)]
    names(confident_bad) <- pr_id[preds >= (con_b)]
    if (length(confident_bad) > round(per_b*length(preds))) {
      confident_bad <- confident_bad[sample(length(confident_bad))[1:round(per_b*length(preds))]]
    }
    confident_preds <- c(confident_good, confident_bad)
    confidents_labs <- confident_preds
    confidents_labs[confident_preds <= (1 - con_g)] <- 'GOOD'
    confidents_labs[confident_preds >= con_b]       <- 'BAD'
    confident_preds <- confidents_labs
    
    # save confident predictions
    confident_labels <- c(confident_labels, confident_preds)
    
    # information
    if (silent == F) {
      print(paste0('-- iteration ', iter + 1, ': appended ', 
                   length(confident_preds), ' cases (' ,
                   sum(confident_preds == 'BAD'),  ' BAD + ', 
                   sum(confident_preds == 'GOOD'), ' GOOD)'))
    }
    iter <- iter + 1
  }
  
  
  ####### EXTRACT PREDICTED LABELS
  
  # extract predicted labels
  raw_test[names(confident_labels), target] <- confident_labels
  preds <- raw_test[[target]]
  names(preds) <- rownames(raw_test)
  
  # information
  if (silent == F) {
    print(paste0('-- finished'))
  }
  
  # return predictions
  preds <- as.factor(preds)
  return(preds)
}
