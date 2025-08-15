#' @title Factor encoding
#' 
#' @description Encodes factor features in a data.frame using dummy or label encoding.
#' 
#' @param data (data.frame): data set with factor features
#' @param factors (character): vector with names of factor features
#' @parm target (character): name of the target variable
#' @param type (character): one of c('dummy', 'label')
#' 
#' @return data.frame with encoded factors
#' 
encodeFactors <- function(data, 
                          factors = NULL, 
                          target  = NULL, 
                          type    = 'dummy') {
  
  # identify factors
  if (is.null(factors)) {
    factors <- colnames(data)[sapply(data, is.factor)]
  }
    
  # ignore target
  if (!is.null(target)) {
    factors <- factors[!factors %in% 'BAD']
  }
  
  # label encoding
  if (type == 'label') {
    for (var in factors) {
      levels(data[[var]]) <- 1:length(levels(data[[var]]))
      data[[var]] <- as.numeric(data[[var]])
    }
  }
  
  # dummy ebcoding
  if (type == 'dummy') {
    data <- createDummyFeatures(data, cols = factors)
  }
  
  # return data
  return(data)
}
