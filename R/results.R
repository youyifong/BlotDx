################################################################################
# perform statistical tests on prediction results

library(kyotil)
library(exact2x2)

header=TRUE

for (i in 1:2) {
  
  sapply(1:4, function(j) {
    if (j==1) {
      dat1=read.csv('Class_Label/HSV'%.%i%.%'_alltest_SEG_sS1_strips_v4_pretrained.csv', header=header)
      dat2=read.csv('Class_Label/HSV'%.%i%.%'_alltest_SEG_sS1_strips_v4_nopretrained.csv', header=header)
    } else if (j==2) {
      dat1=read.csv('Class_Label/HSV'%.%i%.%'_alltest_SEG_sS1_strips_v6_pretrained.csv', header=header)
      dat2=read.csv('Class_Label/HSV'%.%i%.%'_alltest_SEG_sS1_strips_v6_nopretrained.csv', header=header)
    } else if (j==3) {
      dat1=read.csv('Class_Label/HSV'%.%i%.%'_alltest_DET_dS_strips_pretrained.csv', header=header)
      dat2=read.csv('Class_Label/HSV'%.%i%.%'_alltest_DET_dS_strips_nopretrained.csv', header=header)
    } else if (j==4) {
      dat1=read.csv('Class_Label/HSV'%.%i%.%'_alltest_3methods_pretrained_ensemble.csv', header=header)
      dat2=read.csv('Class_Label/HSV'%.%i%.%'_alltest_3methods_nopretrained_ensemble.csv', header=header)
    }
    
    # compare correct responses from two methods
    
    # second column is gt
    stopifnot(dat1[,2]==dat1[,2])
    
    tab=table(dat1[,1]==dat1[,2], dat2[,1]==dat2[,2])
    print(tab)
    
    # tab=matrix(c(1,0,2,54),2,2)
    

    # Exact McNemar
    exact2x2::mcnemar.exact(tab)$p.value
      
    # McNemar
    # mcnemar.test(tab)$p.value
  
    # Binomial
    # prop.test(c(sum(dat1[,1]==dat1[,2]), sum(dat2[,1]==dat2[,2])), c(nrow(dat1), nrow(dat2)))
    
  })

}