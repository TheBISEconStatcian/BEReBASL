###################################
#                                 
#             SETTINGS            
#                                 
###################################

# clearing the memory
rm(list = ls())

# installing pacman
if (require(pacman) == F) install.packages('pacman')
library(pacman)

# libraries
p_load(mvtnorm, clusterGeneration, mclust, psych, ggplot2, gridExtra, stargazer, mlr, 
       glmnet, tidyr, gdata, matrixcalc, AUC, isofor)

# working directory
cd <- dirname(rstudioapi::getActiveDocumentContext()$path)
setwd(dirname(cd))

# setting inner folders
code.folder <- 'codes'
func.folder <- 'functions'
data.folder <- 'data'
resu.folder <- 'results'

# loading functions
source(file.path(code.folder, 'code_00_helper_functions.R'))

# notations
options(scipen = 10)



###################################
#                                 
#             PARAMETERS          
#                                 
###################################

# data generation
num_feats    <- 2
num_noise    <- 0
bad_ratio    <- 0.7
mean_dif     <- c(2, 1)
var_dif      <- 0.5
covars       <- list(matrix(c(1,  0.2,  0.2, 1), nrow = 2), 
                     matrix(c(1, -0.2, -0.2, 1), nrow = 2))
noise_var    <- 0
mixture      <- F
mix_mean_dif <- 5
mix_var_dif  <- 0

# acceptance loop
init_sample    <- 100
sample_size    <- 100
holdout_sample <- 3000
num_gens       <- 300
top_percent    <- 0.2

# scoring model
algo <- 'classif.logreg'
lrn  <- makeLearner(algo, predict.type = 'prob')



###################################
#                                 
#         INITIALIZATION
#                                 
###################################

####### PREPARATION

# placeholders
results <- list(data = list(), stats = list(), models = list(), boundaries = list())

# random seed
seed <- 77
set.seed(seed)

# timer
t.start <- proc.time()


####### INITIAL POPULATION

# generate data
res <- genCreditData(
  #################################### DIMENSIONALITY
  n                = init_sample,  # - sample size
  bad_ratio        = bad_ratio,    # - BAD ratio (if all D = 0)
  k_con            = num_feats,    # - no. continuous features
  k_cat            = 0,            # - no. categorical features
  k_bin            = 0,            # - no. binary features
  k_noise          = num_noise,    # - no. white-noise features
  #################################### CONTINUOUS FEATURES
  con_nonlinear    = 0.0,       # - share of nonlinear transformations
  con_mean_bad_dif = mean_dif,  # - mean difference between classes
  con_var_bad_dif  = var_dif,   # - share of var/covar difference between classes
  con_noise_var    = noise_var, # - variance of noise
  covars           = covars,    # - variance-covariance matrices
  #################################### MIXTURE OF GAUSSIANS
  mixture      = mixture,       # - mixture of two Gaussians
  mix_mean_dif = mix_mean_dif,  # - mean difference between components
  mix_var_dif  = mix_var_dif,   # - share of var/covar difference between components
  #################################### OTHER PARAMETERS
  seed             = seed,   # - random seed
  verbose          = F,      # - displaying feedback
  encode_factors   = F)      # - encoding of categorical features

# extract initial population
init_population <- res$data

# accept applicants using a business rule on X1
accepts_ind <- which(init_population$X1 >= quantile(init_population$X1, 1 - top_percent))
current_accepts <- init_population[ accepts_ind, ]
current_rejects <- init_population[-accepts_ind, ]

# check if both classes are present
if (sum(current_accepts$BAD == 'BAD') < 4) {
  accepts_ind <- c(which(init_population$X1 >= quantile(init_population$X1, 1 - top_percent))[1:(length(accepts_ind)-4)], 1:4)
  current_accepts <- init_population[ accepts_ind, ]
  current_rejects <- init_population[-accepts_ind, ]
}


####### HOLDOUT POPULATION

# generate holdout data
hold_population <- genCreditData(n = holdout_sample, replicate = res, seed = -seed)$data



###################################
#                                 
#          ACCEPTANCE LOOP
#                                 
###################################

# placeholders
stats <- data.frame(generation = 1:num_gens, sample_size       = NA, accept_ratio       = NA, 
                    bad_ratio_accepts = NA,  bad_ratio_rejects = NA, bad_ratio_unbiased = NA,
                    auc_accepts       = NA,  auc_unbiased      = NA, auc_inference      = NA)
models <- list(unbiased = list(), accepts = list(), inference = list())
labeled_rejects <- NULL

# acceptance loop
for (g in 1:num_gens) {
  
  
  ##### INFORMATION
  
  # print generation number
  if (g %% 10 == 0) {
    print(paste0('-- iteration ', g, '/', num_gens, ': ', nrow(current_accepts), ' accepts, ',
                 nrow(current_rejects), ' rejects...'))
  }
  
  # save data stats
  stats$sample_size[g]        <- nrow(current_accepts) + nrow(current_rejects)
  stats$accept_ratio[g]       <- nrow(current_accepts) / stats$sample_size[g]
  stats$bad_ratio_accepts[g]  <- (table(current_accepts$BAD) / nrow(current_accepts))['BAD']
  stats$bad_ratio_rejects[g]  <- (table(current_rejects$BAD) / nrow(current_rejects))['BAD']
  stats$bad_ratio_unbiased[g] <- stats$bad_ratio_accepts[g] * stats$accept_ratio[g] +
    stats$bad_ratio_rejects[g] * (1 - stats$accept_ratio[g])
  
  
  ##### DATA GENERATION
  
  # generate new applicants
  new_applicants <- genCreditData(n = sample_size, replicate = res, seed = seed + g)$data
  
  
  ##### ACCEPTS-BASED SCORECARD
  
  # train classifier on accepts
  tsk <- makeClassifTask(data = current_accepts, target = 'BAD', positive = 'BAD')
  clf <- mlr::train(lrn, tsk)
  
  # score new applicants and holdout cases
  preds <- predict(clf, newdata = new_applicants)
  holdout_preds_accepts  <- predict(clf, newdata = hold_population)
  stats$auc_accepts[g]   <- mlr::performance(holdout_preds_accepts, mlr::auc)

  
  ##### ORACLE SCORECARD
  
  # train debiased classifier on unbiased sample
  tsk_unbiased <- makeClassifTask(data = rbind(current_accepts, current_rejects), target = 'BAD',
                                  positive = 'BAD')
  clf_unbiased <- mlr::train(lrn, tsk_unbiased)
  
  # score holdout cases
  holdout_preds_unbiased  <- predict(clf_unbiased, newdata = hold_population)
  stats$auc_unbiased[g]   <- mlr::performance(holdout_preds_unbiased, mlr::auc)

  
  ##### CORRECTED SCORECARD
  
  # BASL parameters
  filtering_beta   <- c(0.01, 0.99)
  holdout_percent  <- 0.1
  labeling_percent <- 0.1
  sampling_percent <- 0.8
  multiplier       <- 2
  max_iterations   <- 5
  early_stop       <- T
  
  # label rejected cases
  rej_labels <- biasAwareSelfLearning(accepts          = current_accepts, 
                                      rejects          = current_rejects,
                                      target           = 'BAD', 
                                      filtering_beta   = filtering_beta,
                                      weak_learner     = 'classif.logreg',
                                      strong_learner   = 'classif.logreg',
                                      holdout_percent  = holdout_percent,
                                      labeling_percent = labeling_percent, 
                                      sampling_percent = sampling_percent,
                                      multiplier       = multiplier, 
                                      max_iterations   = max_iterations,
                                      early_stop       = early_stop,
                                      silent           = T)
  rej_labels <- rej_labels[!is.na(rej_labels)]
  
  # append labeled cases
  tmp_rejects     <- current_rejects[names(rej_labels), ]
  tmp_rejects$BAD <- rej_labels

  # train classifier on expanded data
  tsk_ri <- makeClassifTask(data = rbind(current_accepts, tmp_rejects), 
                            target = 'BAD', 
                            positive = 'BAD')
  clf_ri <- mlr::train(lrn, tsk_ri)

  # score new applicats and holdout cases
  preds_ri <- predict(clf_ri, newdata = new_applicants)
  holdout_preds_inference  <- predict(clf_ri, newdata = hold_population)
  stats$auc_inference[g]   <- mlr::performance(holdout_preds_inference, mlr::auc)

  
  ##### SAVING MODELS & DECISION BOUNDARIES
  
  # for selected generations
  if ((g == 1) | (g %% 10 == 0) | (g == num_gens - 1)) {
    
    # save models
    models$accepts[[g]]   <- clf$learner.model
    models$inference[[g]] <- clf_ri$learner.model
    models$unbiased[[g]]  <- clf_unbiased$learner.model
    
    # find decision boundaries
    tmp_grid <- saveDecisionBoundary(hold_population, 
                                     model_list = list(m1 = clf_unbiased, m2 = clf, m3 = clf_ri))
    tmp_grid$iteration <- g
    
    # save grids
    if (g == 1) {
      grid <- tmp_grid
    }else{
      grid <- rbind(grid, tmp_grid)
    }
  }
  
  
  ##### ACCEPTING APPLICANTS
  
  # skip for the last generation
  if (g < num_gens) {
    
    # accept top applicants
    accepts_ind <- which(preds$data$prob.GOOD >= quantile(preds$data$prob.GOOD, 1 - top_percent))
    new_accepts <- new_applicants[accepts_ind, ]
    if (length(accepts_ind) > round(top_percent*nrow(new_applicants))) {
      new_accepts <- new_accepts[sample(nrow(new_accepts))[1:round(top_percent*nrow(new_applicants))], ]
    }
    
    # select rejected applicants
    rejects_ind <- which(!(rownames(new_applicants) %in% rownames(new_accepts)))
    new_rejects <- new_applicants[rejects_ind, ]
    
    # append accepts and rejects
    current_accepts <- rbind(current_accepts, new_accepts)
    current_rejects <- rbind(current_rejects, new_rejects)
    
    # save labeled rejects
    if ((g == 1) | (g %% 10 == 0) | (g == num_gens - 1)) {
      tmp_rejects$generation <- g
      labeled_rejects <- rbind(labeled_rejects, tmp_rejects)
    }
  }
}



###################################
#
#        SCORECARD SELECTION
#
###################################

###### PARAMETERS

# CV parameters
cv_folds <- 4

# acceptance
acceptance <- 0.2

# metric parameters
bm_max_iterations <- 10^4
bm_epsilon        <- 10^-6

# random seed
seed <- 1
set.seed(seed)


###### PREPARATIONS

# placeholders
eval_models  <- NULL
eval_results <- NULL

# subset rejects
rejects <- current_rejects[sample(nrow(current_rejects))[1:(nrow(current_accepts) / cv_folds *
                                                              (1 - acceptance) / acceptance)], ]
rejects$BAD <- NA


###### TRAINING MODELS

# parameter set
discrete_ps <- makeParamSet(
  makeDiscreteParam('nrounds',   values = seq(10, 100, by = 5)),
  makeDiscreteParam('max_depth', values = c(1, 2, 3)),
  makeDiscreteParam('alpha',     values = 0),
  makeDiscreteParam('lambda',    values = 0),
  makeDiscreteParam('nthread',   values = 2),
  makeDiscreteParam('objective', values = 'binary:logistic')
)

# task settings
lrn <- makeLearner('classif.xgboost', predict.type = 'prob')
rdes <- makeResampleDesc('CV', iters = cv_folds, stratify = T)
ctrl <- makeTuneControlGrid()
tsk  <- makeClassifTask(data = current_accepts, target = 'BAD', positive = 'BAD')

# initialize empty list to store results
stored.models <- list()
lrn.saver <- makeSaveWrapper(lrn)

# perform CV on accepts
set.seed(seed)
res <- tuneParams(learner = lrn.saver, task = tsk, resampling = rdes,
                  par.set = discrete_ps, control = ctrl, measures = mlr::auc,
                  show.info = F)


###### SCORE REJECTS

# train model for scoring rejects
lrn <- makeLearner('classif.logreg', predict.type = 'prob')
tsk <- makeClassifTask(data = current_accepts, target = 'BAD', positive = 'BAD')
clf <- mlr::train(lrn, tsk)

# score rejects
rej_scores <- predict(clf, newdata = current_rejects)


###### EVALUATING MODELS

# placeholders
vals <- data.frame(model = 1:length(stored.models),
                   auc_accepts  = NA,
                   auc_holdout  = NA,
                   auc_rej_bad  = NA,
                   auc_rej_good = NA,
                   auc_average  = NA,
                   auc_weighted = NA,
                   auc_bayesian = NA)

# evaluation loop
fold <- 0
for (i in 1:length(stored.models)) {

  # print model number
  if (i %% 10 == 0) {
    print(paste0('- model ', i, '/', length(stored.models), '...'))
  }

  # fold index
  fold <- fold + 1
  if (fold > cv_folds) {fold <- 1}
  valid_idx <- res$resampling$test.inds[[fold]]

  # accepts
  preds <- predict(stored.models[[i]], newdata = current_accepts[valid_idx, ])
  vals$auc_accepts[i] <- mlr::performance(preds, mlr::auc)

  # holdout
  preds <- predict(stored.models[[i]], newdata = hold_population)
  vals$auc_holdout[i] <- mlr::performance(preds, mlr::auc)

  # AUC assuming rejects are good or bad
  preds_acc      <- predict(stored.models[[i]], newdata = current_accepts[valid_idx, ])
  preds_rej_bad  <- predict(stored.models[[i]], newdata = rejects)
  preds_rej_good <- preds_rej_bad
  preds_rej_bad$data$truth  <- 'BAD'
  preds_rej_good$data$truth <- 'GOOD'
  preds_rej_bad$data  <- rbind(preds_acc$data, preds_rej_bad$data)
  preds_rej_good$data <- rbind(preds_acc$data, preds_rej_good$data)
  vals$auc_rej_bad[i]  <- mlr::performance(preds_rej_bad,  mlr::auc)
  vals$auc_rej_good[i] <- mlr::performance(preds_rej_good, mlr::auc)

  # average AUC
  vals$auc_average[i] <- vals$auc_rej_bad[i]*0.5 + vals$auc_rej_bad[i]*0.5

  # weighted AUC
  w_bad <- sum(current_rejects$BAD == 'BAD') / nrow(current_rejects)
  vals$auc_weighted[i] <- vals$auc_rej_bad[i]*(w_bad) + vals$auc_rej_good[i]*(1 - w_bad)

  # Bayesian AUC
  vals$auc_bayesian[i] <- bayesianMetric(metric         = mlr::auc,
                                         model          = stored.models[[i]],
                                         accepts        = current_accepts[valid_idx, ],
                                         rejects        = rejects,
                                         rejects_prior  = rej_scores$data$prob.BAD,
                                         max_iterations = bm_max_iterations,
                                         epsilon        = bm_epsilon,
                                         seed           = 1)
}

# average over folds
vals <- data.frame(apply(vals[, c('auc_accepts', 'auc_holdout',
                                  'auc_average', 'auc_weighted',
                                  'auc_bayesian')], 2,
                         function(x) colMeans(matrix(x, cv_folds))))

# save results
eval_models  <- c(eval_models, stored.models)
eval_results <- rbind(eval_results, vals)



###################################
#
#          SAVING FILES
#
###################################

# time to run simulation
t.total <- round((((proc.time() - t.start) / 60)[3]), digits = 2)
print(paste0('Time to run simulation: ', t.total, ' min'))

# save data and stats
results$data <- list(accepts = current_accepts,
                     rejects = current_rejects,
                     labeled = labeled_rejects,
                     holdout = hold_population)
results$models     <- models
results$boundaries <- grid
results$params     <- res$params
results$stats      <- stats
results$evaluation <- list(models = eval_models, results = eval_results)

# clear memory
rm(list = c('current_accepts', 'new_accepts', 'current_rejects', 'new_rejects',
            'tmp_rejects', 'labeled_rejects', 'init_population', 'hold_population',
            'new_applicants', 'stats', 'models', 'grid', 'tmp_grid', 'lrn', 't.start',
            'tsk', 'tsk_ri', 'tsk_unbiased', 'preds', 'preds_ri', 'res', 'cd', 't.total',
            'clf', 'clf_ri', 'clf_unbiased', 'accepts_ind', 'rejects_ind', 'vals',
            'holdout_preds_accepts', 'holdout_preds_unbiased', 'holdout_preds_inference',
            'eval_models', 'eval_results', 'stored.models', 'rejects', 'rej_labels'))

# export environment
save.image(file = file.path(data.folder, paste0('simulation_image.RData')))
