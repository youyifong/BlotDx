get.dat = function(HSV, sample_set, model, crop) {
  
  if (file.exists(glue('Class_Label/pred/{sample_set}/HSV{HSV}_Final_2classes_{crop}_{model}_fold0.csv'))) {
    
    # pool cross validated results  
    res_list = lapply(0:4, function(fold) {
      read.csv(glue('Class_Label/pred/{sample_set}/HSV{HSV}_Final_2classes_{crop}_{model}_fold{fold}.csv'), header=T)
    })
    res_all = do.call(rbind, res_list)
    
  } else {
    
    # not-cross validation, but there may be more than 1 sample set
    if (grepl(",", sample_set)) {
      sample_sets = strsplit(sample_set, ',')[[1]]
    } else {
      sample_sets = c(sample_set)
    }
    
    res_all = do.call(
      rbind, 
      lapply(sample_sets, function (s) {
        read.csv(glue('Class_Label/pred/{s}/HSV{HSV}_Final_2classes_{crop}_{model}.csv'), header=T)
      })
    )
    
  }
}
