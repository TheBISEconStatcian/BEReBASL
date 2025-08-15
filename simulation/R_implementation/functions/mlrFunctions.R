###################################
#                                 
#    MLR MODEL EXPORT UTILITIES        
#                                 
###################################

# The funtions are used to automatically save models trained within the mlr resampling.

##### FUNCTION 1
makeSaveWrapper = function(learner) {
  mlr:::makeBaseWrapper(
    id = paste0(learner$id, 'save', sep = '.'),
    type = learner$type,
    next.learner = learner,
    par.set = makeParamSet(),
    par.vals = list(),
    learner.subclass = 'SaveWrapper',
    model.subclass = 'SaveModel')
}

##### FUNCTION 2
trainLearner.SaveWrapper = function(.learner, .task, .subset, ...) {
  m = train(.learner$next.learner, task = .task, subset = .subset)
  stored.models <<- c(stored.models, list(m))
  mlr:::makeChainModel(next.model = m, cl = 'SaveModel')
}

##### FUNCTION 3
predictLearner.SaveWrapper = function(.learner, .model, .newdata, ...) {
  NextMethod(.newdata = .newdata)
}
