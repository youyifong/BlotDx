library(kyotil)
library(exact2x2)
library(glue)
library(gt)
library(officer)
library(flextable)


cropVersions = c("SEG_sS1_strips_v4", "SEG_sS1_strips_v6", "DET_dS_strips", "ensemble")

tabs = lapply (1:2, function(HSV) {
  
  res = mysapply(cropVersions, function(j) {
    # myprint(HSV, j)
    
    folder = 'Class_Label/pred/201608-201702_Tranche1/alltest/'
    dat1=read.csv(glue('{folder}HSV{HSV}_Final_2classes_{j}_pretrained.csv'), header=T)
    dat2=read.csv(glue('{folder}HSV{HSV}_Final_2classes_{j}_nopretrained.csv'), header=T)
    
    # add results from tranche 2
    folder = 'Class_Label/pred/201608-201702_Tranche2/'
    dat1a=read.csv(glue('{folder}HSV{HSV}_Final_2classes_{j}_pretrained.csv'), header=T)
    dat2a=read.csv(glue('{folder}HSV{HSV}_Final_2classes_{j}_nopretrained.csv'), header=T)
    
    dat1=rbind(dat1, dat1a)
    dat2=rbind(dat2, dat2a)
    
    # second column is gt
    stopifnot(dat1[,2]==dat2[,2])
    
    # compare correct responses from two methods
    tab=table(dat1[,1]==dat1[,2], dat2[,1]==dat2[,2])

    # Exact McNemar test
    pval = exact2x2::mcnemar.exact(tab)$p.value
    
    # McNemar test
    # mcnemar.test(tab)$p.value
    
    # Binomial test
    # prop.test(c(sum(dat1[,1]==dat1[,2]), sum(dat2[,1]==dat2[,2])), c(nrow(dat1), nrow(dat2)))
    
    # return correct counts and p val    
    c(sum(tab["TRUE",]), sum(tab[,"TRUE"]), pval)
    
  })
  
  res
  
})

# add rownames as column
tab = data.frame(c("SEG1", "SEG2", "DET", "ENSBL"), cbind(tabs[[1]], tabs[[2]]))
tab[,'X3'] = formatPvalues(tab[,'X3'])
tab[,'X6'] = formatPvalues(tab[,'X6'])

table2 = tab
colnames(table2)[1]="crop"

# add single strip comparison

tabs = lapply (1:2, function(HSV) {
  
  j="SEG_sS1_strips_v4"
  
  folder = 'Class_Label/pred/201608-201702_Tranche1/test/'
  dat1sS=read.csv(glue('{folder}HSV{HSV}_Final_2classes_{j}_HSV{HSV}_pretrained.csv'), header=T)
  dat2sS=read.csv(glue('{folder}HSV{HSV}_Final_2classes_{j}_HSV{HSV}_nopretrained.csv'), header=T)
  
  # add results from validation
  folder = 'Class_Label/pred/201608-201702_Tranche1/validation/'
  dat1a=read.csv(glue('{folder}HSV{HSV}_Final_2classes_{j}_HSV{HSV}_pretrained.csv'), header=T)
  dat2a=read.csv(glue('{folder}HSV{HSV}_Final_2classes_{j}_HSV{HSV}_nopretrained.csv'), header=T)
  
  # add results from tranche 2
  folder = 'Class_Label/pred/201608-201702_Tranche2/'
  dat1b=read.csv(glue('{folder}HSV{HSV}_Final_2classes_{j}_HSV{HSV}_pretrained.csv'), header=T)
  dat2b=read.csv(glue('{folder}HSV{HSV}_Final_2classes_{j}_HSV{HSV}_nopretrained.csv'), header=T)
  
  dat1sS=rbind(dat1sS, dat1a, dat1b)
  dat2sS=rbind(dat2sS, dat2a, dat2b)
  
  # second column is gt
  stopifnot(dat1sS[,2]==dat2sS[,2])
  
  # compare correct responses from with TL and without TL
  tab=table(dat1sS[,1]==dat1sS[,2], dat2sS[,1]==dat2sS[,2])
  
  # Exact McNemar test
  pval = exact2x2::mcnemar.exact(tab)$p.value
  
  # return correct counts and p val    
  ret = matrix(c(sum(tab["TRUE",]), sum(tab[,"TRUE"]), pval), nrow=1)
  
  
  
  # double strip prediction results
  folder = 'Class_Label/pred/201608-201702_Tranche1/alltest/'
  dat1dS=read.csv(glue('{folder}HSV{HSV}_Final_2classes_{j}_pretrained.csv'), header=T)
  dat2dS=read.csv(glue('{folder}HSV{HSV}_Final_2classes_{j}_nopretrained.csv'), header=T)
  
  # add results from tranche 2
  folder = 'Class_Label/pred/201608-201702_Tranche2/'
  dat1a=read.csv(glue('{folder}HSV{HSV}_Final_2classes_{j}_pretrained.csv'), header=T)
  dat2a=read.csv(glue('{folder}HSV{HSV}_Final_2classes_{j}_nopretrained.csv'), header=T)
  
  dat1dS=rbind(dat1dS, dat1a)
  dat2dS=rbind(dat2dS, dat2a)
  
  
  # compare correct responses between sS and dS
  tab1=table(dat1sS[,1]==dat1sS[,2], dat1dS[,1]==dat1dS[,2])
  tab2=table(dat2sS[,1]==dat2sS[,2], dat2dS[,1]==dat2dS[,2])
  
  # Exact McNemar test
  pvals = c(exact2x2::mcnemar.exact(tab1)$p.value, exact2x2::mcnemar.exact(tab2)$p.value)
  
  rbind(ret, c(pvals, NA))  
  
})

# add rownames as column
tab = data.frame(c("sS"," "), cbind(tabs[[1]], tabs[[2]]))
tab[,'X3'] = formatPvalues(tab[,'X3'])
tab[,'X6'] = formatPvalues(tab[,'X6'])
colnames(tab)[1]="crop"


# final table
table=rbind(table2, tab[1,,drop=F])
# turn all columns into chr
table[] <- lapply(table, as.character)
# add tab[2,]
table = rbind(table, 
      c("P-value^", unlist(formatPvalues(tab[2,2:3])), " ", unlist(formatPvalues(tab[2,5:6])), " ")
)



{
  ft <- flextable(table)
  
  # remove column names
  ft <- delete_part(x = ft, part = "header")
  
  ft <- add_header_row(
    ft,
    values = c("", rep(c('With TL', 'No TL', 'P-value*'), 2))
  )
  ft <- add_header_row(
    ft,
    values = c("", "HSV 1", "HSV 2"),
    colwidths = c(1, 3, 3)
  )
  ft <- align(ft, align = "center", part = "header")
  
  ft <- font(ft, fontname = "Times New Roman", part = "all")
  ft <- fontsize(ft, size = 12, part = "all")
  
  doc <- read_docx()
  doc <- body_add_par(doc, "", style = "Normal")
  doc <- body_add_flextable(doc, ft)
  print(doc, target = "R/output/Table2.docx")
}


