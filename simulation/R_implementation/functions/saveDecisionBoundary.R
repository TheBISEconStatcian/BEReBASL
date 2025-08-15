#' @title Save decision boundary
#' 
#' @description Saves decision boundary of a classification model
#' 
#' @param data (data.frame): data set containing the evaluation sample
#' @param model_list (list): list with classification models in mlr format
#' @param features (character): vector of two features in which decision boundaries are constructed
#' @param grid_length (character): length of the value grid (number of bins) for the specified features
#' @param plot (logical): whether to plot the resulting decision boundaries
#' 
#' @return data.frame with decision boundary coordinates
#' 
saveDecisionBoundary <- function(data, 
                                 model_list, 
                                 features = c('X1', 'X2'), 
                                 grid_length = 100, 
                                 plot = F) {
  
  # create plot grid
  x1 <- seq(min(data[[features[1]]]), max(data[[features[2]]]), length = grid_length)
  x2 <- seq(min(data[[features[1]]]), max(data[[features[2]]]), length = grid_length)
  grid  <- expand.grid(X1 = x1, X2 = x2, BAD = NA)
  
  # impute means for other V features
  k_con <- sum(grepl('X', colnames(data)))
  if (k_con > 2) {
    for (x in 3:k_con) {
      grid[[paste0('X', x)]] <- mean(data[[paste0('X', x)]])
    }
  }
  
  # impute means for other N features
  k_nos <- sum(grepl('N', colnames(data)))
  if (k_nos > 0) {
    for (x in 1:k_nos) {
      grid[[paste0('N', v)]] <- mean(data[[paste0('N', x)]])
    }
  }
  
  # save predictions
  for (obj in 1:length(model_list)) {
    preds <- predict(model_list[[obj]], newdata = grid)
    grid[[paste0('response', obj)]] <- preds$data$response
    grid[[paste0('prob',     obj)]] <- preds$data$prob.BAD
  }
  
  # plot diagrams
  if (plot == T) {
    
    # color params
    twoClassColor <- c('#F8766D', '#00BA38')
    names(twoClassColor) <- c('BAD', 'GOOD')
    
    # shape params
    twoClassShape <- c(16, 17)
    names(twoClassShape) <- c('BAD', 'GOOD')
    
    # region plot
    p1 <- ggplot(data = hold_population, aes(x = X1, y = X2, color = BAD)) +
      geom_tile(data = grid, aes(fill = response1, color = response1)) +
      scale_fill_manual(name  = 'Target', values = twoClassColor) +
      scale_color_manual(name = 'Target', values = twoClassColor) +
      scale_x_continuous(expand = c(0, 0)) +
      scale_y_continuous(expand = c(0, 0))
    
    # boundary plot
    p2 <- ggplot(data = hold_population, aes(x = X1, y = X2, color = BAD, pch = BAD)) +
      geom_point(size = 4) + xlab(features[1]) + ylab(features[2]) +
      scale_color_manual(name = 'Target', values = twoClassColor) +
      scale_shape_manual(name = 'Target', values = twoClassShape) +
      geom_contour(data = grid, aes(z = prob1), color = 'black', breaks = c(0, 0.5)) +
      geom_contour(data = grid, aes(z = prob2), color = 'red',   breaks = c(0, 0.5)) +
      geom_vline(aes(xintercept = 0, linetype = 'Unbiased'), size = 0, color = 'black') +
      geom_vline(aes(xintercept = 0, linetype = 'Accepts'),  size = 0, color = 'red') +
      scale_linetype_manual(name = 'Scorecard', values = c(1, 1), 
        guide = guide_legend(override.aes = list(color = c('red', 'black'), size = 1.5)))

    grid.arrange(p1, p2, nrow = 1)
  }
  
  # return saved grid
  return(grid)
}
