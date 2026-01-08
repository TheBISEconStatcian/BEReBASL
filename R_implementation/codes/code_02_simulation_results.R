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
       glmnet, tidyr, gdata, matrixcalc, AUC, beepr)

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
#     IMPORT SIMULATION RESULTS
#                                 
###################################

# load environment
load(file.path(data.folder, paste0('simulation_image.RData')), verbose = F)

# minimal generation
min_gen <- 5

# unfold results
current_accepts <- results$data$accepts
current_rejects <- results$data$rejects
labeled_rejects <- results$data$labeled
hold_population <- results$data$holdout
stats           <- results$stats
models          <- results$models
grid            <- results$boundaries
evals           <- results$evaluation$results
rm(list = 'results')



###################################
#                                 
#        DATA DISTRIBUTION
#                                 
###################################

########## FEATURE DENSITIES

# preparations
hold_population$Sample <- 'Population'
current_accepts$Sample <- 'Accepts'
current_rejects$Sample <- 'Rejects'
data <- rbind(hold_population, current_accepts, current_rejects)
data$Sample <- factor(data$Sample, levels = c('Accepts', 'Rejects', 'Population'))

# density plot
d1 <- ggplot(data, aes(X1, fill = Sample, color = Sample)) + geom_density(alpha = 0.3, size = 1) +
  labs(y = 'Density', x = bquote('x'[1]), title = bquote('(a) Sampling Bias on x'[1])) +
  theme(plot.title = element_text(hjust = 0.5), legend.position = c(0.890, 0.830),
        legend.background = element_rect(linetype = 'solid', colour = 'black')) +
  scale_color_manual(values = c('#00BA38', 'orange', '#619CFF')) +
  scale_fill_manual(values = c('#00BA38', 'orange', '#619CFF'))
d2 <- ggplot(data, aes(X2, fill = Sample, color = Sample)) + geom_density(alpha = 0.3, size = 1) +
  labs(y = 'Density', x = bquote('x'[2]), title = bquote('(b) Sampling Bias on x'[2])) +
  theme(plot.title = element_text(hjust = 0.5), legend.position = c(0.890, 0.830),
        legend.background = element_rect(linetype = 'solid', colour = 'black')) +
  scale_color_manual(values = c('#00BA38', 'orange', '#619CFF')) +
  scale_fill_manual(values = c('#00BA38', 'orange', '#619CFF'))
grid.arrange(d1, d2, nrow = 1)
dev.copy2pdf(file = file.path(resu.folder, 'sim_feature_densities.pdf'), width = 12, height = 8)
dev.off()

# clean up
rm(list = 'data')
hold_population$Sample <- NULL
current_accepts$Sample <- NULL
current_rejects$Sample <- NULL


########## TARGET DENSITY

# preparations
data <- rbind(hold_population, current_accepts, current_rejects)

# density plot
par(mfrow = c(1, 2))
cdplot(BAD ~ X1, data = data, main = bquote('Target vs x'[1]), ylab = 'Target Density',
       xlim = c(quantile(data$X1, 0.01), quantile(data$X1, 0.99)), xlab = bquote('x'[1]))
abline(v = quantile(current_accepts$X1, 0.01), col = 'red', lty = 'dashed')
cdplot(BAD ~ X2, data = data, main = bquote('Target vs x'[2]), ylab = '',
       xlim = c(quantile(data$X2, 0.01), quantile(data$X2, 0.99)), xlab = bquote('x'[2]))
abline(v = quantile(current_accepts$X2, 0.01), col = 'red', lty = 'dashed')
dev.copy2pdf(file = file.path(resu.folder, 'sim_target_densities.pdf'), width = 12, height = 8)
dev.off()

# clean up
rm(list = 'data')


########## CORRELATION PLOTS

# accepts
pairs.panels(current_accepts[, c(1:2, which(colnames(current_accepts) == 'BAD'))], density = F, 
             main = '(a) Accepts', hist.col = 'grey', show.points = F, ellipses = T, rug = F)
dev.copy2pdf(file = file.path(resu.folder,'sim_corplot_accepts.pdf'), width = 10, height = 6)
dev.off()

# rejects
pairs.panels(current_rejects[, c(1:2, which(colnames(current_rejects) == 'BAD'))], density = F,
             main = '(b) Rejects', hist.col = 'grey', show.points = F, ellipses = T, rug = F)
dev.copy2pdf(file = file.path(resu.folder, 'sim_corplot_rejects.pdf'), width = 10, height = 6)
dev.off()

# holdout
pairs.panels(hold_population[, c(1:2, which(colnames(hold_population) == 'BAD'))], density = F,
             main = '(c) Holdout', hist.col = 'grey', show.points = F, ellipses = T, rug = F)
dev.copy2pdf(file = file.path(resu.folder,'sim_corplot_holdout.pdf'), width = 10, height = 6)
dev.off()



###################################
#                                 
#        IMPACT ON TRAINING       
#                                 
###################################

########## DECISION BOUNDARIES

# preparations
g <- 300
idx_a <- init_sample*top_percent + (sample_size*top_percent*g)
idx_r <- init_sample*(1-top_percent) + (sample_size*(1-top_percent)*(g))
cur_tmp_accepts <- current_accepts[1:min(idx_a, nrow(current_accepts)), ]
cur_tmp_rejects <- current_rejects[1:min(idx_r, nrow(current_rejects)), ]
cur_tmp_accepts$sample <- 'Accepts'
cur_tmp_rejects$sample <- 'Rejects'
datas <- rbind(cur_tmp_accepts, cur_tmp_rejects)
tmp_grid <- grid[grid$iteration == g, ]
oracle_grid <- grid[grid$iteration == num_gens, ]

# subset data to reduce plot size
set.seed(1)
datas_bad      <- datas[datas$BAD == 'BAD',  ][sample(nrow(datas[datas$BAD == 'BAD',  ])/4, replace = F), ]
datas_good     <- datas[datas$BAD == 'GOOD', ][sample(nrow(datas[datas$BAD == 'GOOD', ])/4, replace = F), ]
datas          <- rbind(datas_bad, datas_good)
subset_rejects <- labeled_rejects[labeled_rejects$generation == num_gens - 1, ]
subset_rejects <- subset_rejects[sample(nrow(subset_rejects)/4, replace = F), ]

# scatterplot
ggplot(data = datas, aes(x = X1, y = X2, color = sample)) + xlim(-3, 5) + ylim(-3, 3) +
  theme(axis.title.y = element_text(angle = 0, hjust = 0.5, vjust = 0.5)) + 
  geom_point(size = 1.5) + labs(y = bquote('x'[2]), x = bquote('x'[1]), color = 'Scoring Model') +
  geom_point(data = subset_rejects, aes(x = X1, y = X2, color = 'Labeled Rejects'), size = 1.5, shape = 15) +
  scale_color_manual(name = 'Sample', values = c('#00BA38', 'black', '#F8766D')) +
  geom_contour(data = oracle_grid, aes(z = prob1), color =  'blue',      size = 1.5, breaks = c(0, 0.5)) +
  geom_contour(data = tmp_grid,    aes(z = prob2), color =  'darkgreen', size = 1.5, breaks = c(0, 0.5)) +
  geom_contour(data = tmp_grid,    aes(z = prob3), color =  'orange',    size = 1.5, breaks = c(0, 0.5)) +
  geom_vline(aes(xintercept = 0, linetype = 'Oracle'),         size = 0, color = NA) +
  geom_vline(aes(xintercept = 0, linetype = 'Accepts'),        size = 0, color = NA) +
  geom_vline(aes(xintercept = 0, linetype = 'Accepts + BASL'), size = 0, color = NA) +
  scale_linetype_manual(name = 'Scoring Model', values = c(1, 1, 1),
                        guide = guide_legend(override.aes = list(color = c('darkgreen', 'orange', 'blue'), size = 1.5)))
dev.copy2pdf(file = file.path(resu.folder, 'sim_decision_boundaries.pdf'), width = 11*0.8, height = 6*0.8)
dev.off()


########## GAINS FROM BIAS-AWARE SELF-LEARNING

# preparations
stats_basl            <- stats[stats$generation >= min_gen, ]
stats_basl$auc_gap    <- stats_basl$auc_unbiased - stats_basl$auc_accepts
stats_basl$auc_gap_ri <- stats_basl$auc_unbiased - stats_basl$auc_inference
perf_basl <- gather(stats_basl, training, performance, auc_accepts:auc_inference, factor_key = T)

# performance plot
s1 <- ggplot(data = perf_basl, aes(x = generation, y = performance, col = training)) + geom_line(size = 1.5) +
  theme(plot.title = element_text(hjust = 0.5), legend.position = c(0.870, 0.140),
        legend.background = element_rect(linetype = 'solid', colour = 'black')) +
  labs(y = 'AUC on Holdout Sample', x = 'Acceptance Loop Iteration',
       color = 'Scroing Model', title = '(a) Performance') +
  scale_color_manual(labels = c('Accepts', 'Oracle', 'Accepts + BASL'),
                     values = c('#00BA38', '#619CFF', 'orange'))
gap <- gather(stats_basl, training, performance, auc_gap:auc_gap_ri, factor_key = T)
s2 <- ggplot(data = stats_basl, aes(x = generation, y = auc_gap)) + geom_line(color = '#00BA38', size = 1.5) +
  theme(plot.title = element_text(hjust = 0.5)) + 
  labs(y = 'AUC Gap', x = 'Acceptance Loop Iteration', title = '(b) Performance Gap') +
  geom_hline(yintercept = 0, linetype = 'ff', color = '#619CFF', size = 0.5) +
  geom_line(data = stats_basl, aes(x = generation, y = auc_gap_ri), color = 'orange', size = 1.5)
grid.arrange(s1, s2, nrow = 1, widths = c(5.5, 4.5))
dev.copy2pdf(file = file.path(resu.folder, 'sim_gains_from_basl.pdf'), width = 12, height = 5)
dev.off()

# print results
loss_due_to_bias <- stats_basl$auc_unbiased - stats_basl$auc_accepts
gain_basl <- stats_basl$auc_inference - stats_basl$auc_accepts
print(paste0('Average loss due to bias: ', round(mean(loss_due_to_bias), 4)))
print(paste0('Performance gains from BASL-based bias correction: ', 
             round(mean(gain_basl), 4)))
print(paste0('Performance gains from BASL-based bias correction: ', 
             100*round(mean(gain_basl/loss_due_to_bias), 4), '%'))

# performance plot
stats_loss <- stats[stats$generation >= min_gen, ]
perf <- gather(stats_loss, training, performance, auc_accepts:auc_unbiased, factor_key = T)
p1 <- ggplot(data = perf, aes(x = generation, y = performance, col = training)) + geom_line(size = 1.5) +
  theme(plot.title = element_text(hjust = 0.5), legend.position = c(0.870, 0.140),
        legend.background = element_rect(linetype = 'solid', colour = 'black')) +
  labs(y = 'AUC on Holdout Sample', x = 'Acceptance Loop Iteration',
       color = 'Scoring Model', title = '(b) Impact on Training') +
  scale_color_manual(labels = c('Accepts', 'Oracle'),
                     values = c('#00BA38', '#619CFF'))
print(p1)



###################################
#                                 
#       IMPACT ON EVALUATION     
#                                 
###################################

###### GAINS FROM BAYESIAN EVALUATION

# preparations 1
evals$method <- 'Others'
evals$method[which.max(evals$auc_holdout)] <- 'Best on Holdout'
evals$method[which.max(evals$auc_accepts)] <- 'Best on Accepts'
if (evals$method[which.max(evals$auc_holdout)] == evals$method[which.max(evals$auc_accepts)]) {
  evals_tmp <- evals[which.max(evals$auc_holdout), ]
  evals_tmp$method <- 'Best on Holdout'
  evals <- rbind(evals, evals_tmp)
}

# scatterplot 1
e1 <- ggplot(evals, aes(x = auc_accepts, y = auc_holdout, color = method)) +
  theme(plot.title = element_text(hjust = 0.5), legend.position = c(0.850, 0.150),
        legend.background = element_rect(linetype = 'solid', colour = 'black'),
        plot.margin = margin(5.5, 5.5+5, 5.5, 5.5+5)) +
  geom_point(size = 5, shape = 16) +
  labs(y = 'AUC on Holdout Sample', x = 'AUC on Accepts',
       color = 'Scoring Model', title = '(a) Evaluation on Accepts') +
  geom_hline(aes(yintercept = evals[evals$method == 'Best on Holdout', 'auc_holdout']),
             linetype = 2, size = 1, color = '#619CFF') +
  geom_hline(aes(yintercept = evals[evals$method == 'Best on Accepts', 'auc_holdout']),
             linetype = 2, size = 1, color = '#00BA38') +
  scale_color_manual(values = c('#00BA38', '#619CFF', 'gray20'))

# preparations 2
evals2 <- evals
evals2$method <- 'Others'
evals2$method[which.max(evals2$auc_holdout)]  <- 'Best on Holdout'
evals2$method[which.max(evals2$auc_bayesian)] <- 'Best Bayesian AUC'
if (evals2$method[which.max(evals2$auc_holdout)] == evals2$method[which.max(evals2$auc_bayesian)]) {
  evals2_tmp <- evals2[which.max(evals2$auc_holdout), ]
  evals2_tmp$method <- 'Best on Holdout'
  evals2 <- rbind(evals2, evals2_tmp)
}

# scatterplot 2
e2 <- ggplot(evals2, aes(x = auc_bayesian, y = auc_holdout, color = method)) +
  theme(plot.title = element_text(hjust = 0.5), legend.position = c(0.850, 0.150),
        legend.background = element_rect(linetype = 'solid', colour = 'black'),
        plot.margin = margin(5.5, 5.5+5, 5.5, 5.5+5)) +
  geom_point(size = 5, shape = 16) +
  labs(y = 'AUC on Holdout Sample', x = ' AUC',
       color = 'Scoring Model', title = '(b) Evaluation with Bayesian AUC') +
  geom_hline(aes(yintercept = evals[evals$method == 'Best on Holdout', 'auc_holdout']),
             linetype = 2, size = 1, color = '#619CFF') +
  geom_hline(aes(yintercept = evals2[evals2$method == 'Best Bayesian AUC', 'auc_holdout']),
             linetype = 2, size = 1, color = 'orange') +
  scale_color_manual(values = c('orange', '#619CFF', 'gray20'))

# export plot
grid.arrange(e1, e2, nrow = 1)
dev.copy(pdf, file.path(resu.folder, 'sim_gains_from_bayesian.pdf'), width = 12, height = 5)
dev.off()

# print results
r_accepts  <- cor(evals$auc_accepts,  evals$auc_holdout, method = 'spearman')
r_bayesian <- cor(evals$auc_bayesian, evals$auc_holdout, method = 'spearman')
gain_bayesian <- evals2$auc_holdout[evals2$method == 'Best Bayesian AUC'] - 
  evals$auc_holdout[evals$method == 'Best on Accepts']
print(paste0('Correlation between accepts-based AUC and holdout AUC: ', 
             round(r_accepts, 4)))
print(paste0('Correlation between Bayesian AUC and holdout AUC: ', 
             round(r_bayesian, 4)))
print(paste0('Performance gains from Bayesian model selection: ', 
             round(gain_bayesian, 4)))



###################################
#                                 
#          OVERALL GAINS 
#                                 
###################################

# helper functions for colors
branded_colors <- list(
  'orange' = '#FFA500',
  'green'  = '#00BA38',
  'blue'   = '#619CFF',
  'gray'   = '#333333'
)
branded_pal <- function(primary = 'blue', other = 'grey', direction = 1) {
  stopifnot(primary %in% names(branded_colors))
  function(n) {
    if (n > 4) warning('Branded Color Palette only has 4 colors.')
    if (n == 2) {
      other <- if (!other %in% names(branded_colors)) {
        other
      }else{
        branded_colors[other]
      }
      color_list <- c(other, branded_colors[primary])
    }else{
      color_list <- branded_colors[1:n]
    }
    color_list <- unname(unlist(color_list))
    if (direction >= 0) color_list else rev(color_list)
  }
}
scale_color_branded <- function(primary = 'blue', other = 'grey', direction = 1, ...) {
  ggplot2::discrete_scale('colour', 'branded', branded_pal(primary, other, direction), ...)
}

# plot with losses and gains
hold_population$Sample <- 'Population'
current_accepts$Sample <- 'Accepts'
current_rejects$Sample <- 'Rejects'
data <- rbind(hold_population, current_accepts, current_rejects)
data$Sample <- factor(data$Sample, levels = c('Accepts', 'Rejects', 'Population'))
d1 <- ggplot(data, aes(X1, fill = Sample, color = Sample)) + geom_density(alpha = 0.3, size = 1) +
  labs(y = 'Density', x = bquote('x'[1]), title = bquote('(a) Sampling Bias on x'[1])) +
  theme(plot.title = element_text(hjust = 0.5), legend.position = c(0.870, 0.840),
        legend.background = element_rect(linetype = 'solid', colour = 'black'),
        plot.margin = margin(5.5, 5.5+8, 5.5, 5.5)) +
  scale_color_manual(values = c('#00BA38', 'orange', '#619CFF')) +
  scale_fill_manual(values = c('#00BA38', 'orange', '#619CFF'))
s4 <- ggplot(data = perf_basl, aes(x = generation, y = performance, col = training)) + geom_line(size = 1.5) +
  theme(plot.title = element_text(hjust = 0.5), legend.position = c(0.830, 0.160),
        legend.background = element_rect(linetype = 'solid', colour = 'black'),
        plot.margin = margin(5.5, 5.5+5, 5.5, 5.5+8)) +
  labs(y = 'AUC on Holdout Sample', x = 'Acceptance Loop Iteration',
       color = 'Scroing Model', title = '(b) Impact on Training') +
  scale_color_manual(labels = c('Accepts', 'Oracle', 'Accepts + BASL'),
                     values = c('#00BA38', '#619CFF', 'orange'))
e4 <- ggplot(evals2, aes(x = auc_accepts, y = auc_holdout, color = method)) +
  theme(plot.title = element_text(hjust = 0.5), legend.position = c(0.810, 0.160),
        legend.background = element_rect(linetype = 'solid', colour = 'black'),
        plot.margin = margin(5.5, 5.5+5, 5.5, 5.5+5)) +
  geom_point(size = 4, shape = 16) +
  labs(y = 'AUC on Holdout Sample', x = 'AUC on Accepts',
       color = 'Scoring Model', title = '(c) Impact on Evaluation') +
  geom_hline(aes(yintercept = evals2[evals2$method == 'Oracle', 'auc_holdout']),
             linetype = 2, size = 1, color = '#619CFF') +
  geom_hline(aes(yintercept = evals2[evals2$method == 'Best Bayesian AUC', 'auc_holdout']),
             linetype = 2, size = 1, color = 'orange') +
  geom_hline(aes(yintercept = evals2[evals2$method == 'Best on Accepts', 'auc_holdout']),
             linetype = 2, size = 1, color = '#00BA38') + 
  scale_color_branded(breaks = c('Best on Accepts', 'Oracle', 'Best Bayesian AUC'))

# export plot
grid.arrange(d1, s4, e4, nrow = 1)
dev.copy(pdf, file.path(resu.folder, 'sim_loss_and_gains.pdf'), width = 20.5*0.75, height = 6*0.75)
dev.off()
