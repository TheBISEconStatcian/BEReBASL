###################################
#                                 
#    LOADING HELPER FUNCTIONS
#                                 
###################################

# Loads all helper functions from the specified folder.

# setting working directory
load_functions <- function(path) {
  
  # getting file list
  file.list <- list.files(path)
  
  # processing functions
  for(i in 1:length(file.list)) {
    
    # load function
    source(file.path(path, file.list[i]))
    print(file.path('Loading ', file.list[i]))
  }
}
 
# import functions
load_functions(func.folder)
