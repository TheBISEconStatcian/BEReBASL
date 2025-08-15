#' @title Filtering stage of Bias-aware self-learning
#' 
#' @description Filters rejected cases that are most and least similar to accepts.
#' 
#' @details 
#' This function filters rejects as a part of the bias-aware self-learning framework.
#' 
#' Bias-aware self-learning (BASL) is a reject inference framework that mitigates the effect 
#' of sampling bias on model training. BASL is based on semi-supervised self-learning framework
#' and introduces extensions to adapt it to a setting with sampling bias. 
#' 
#' This function implements a filtering stage that is used to filter rejects before 
#' assigning them with labels. The filtering is performed with isolation forest, which is
#' a novelty detection algorithm used to assess the similarity between accepts and rejects.
#' 
#' @param accepts (data.frame): data set containing accepted applications
#' @param rejects (data.frame): data set containing rejected applications
#' @param target (character): name of the target variable
#' @param beta (float): percentiles of rejects to be filtered
#' @param num_trees (int): number of trees in isolation forest
#' 
filteringStage <- function(accepts, 
                           rejects, 
                           target    = 'BAD', 
                           beta      = c(0, 1),
                           num_trees = 100) {
  
  # library
  #devtools::install_github("Zelazny7/isofor")
  
  # remove target
  accepts[target] <- NULL
  
  # fit isolation forest
  mod <- iForest(X = accepts, nt = num_trees)
  
  # predict probs
  p_anom <- predict(mod, rejects[, !(colnames(rejects) %in% target)])
  
  # filter rejects
  tmp_rejects <- ((p_anom >= quantile(p_anom, beta[1])) & 
                     (p_anom <= quantile(p_anom, beta[2])))
  
  return(tmp_rejects)
}
