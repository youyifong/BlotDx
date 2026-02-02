library(kyotil)
library(glue)
library(gt)
library(officer)
library(flextable)

source("R/helper.R")

# library(argparse)
# parser <- ArgumentParser()
# parser$add_argument("--model", required=TRUE)
# parser$add_argument("--sample_set", required=TRUE)
# args <- parser$parse_args()
# str(args)
# model = args$model
# sample_set = args$sample_set

sample_set = '201608-201702_Tranche1and2'
model = "201608-201702_Tranche1and2_pretrained"

by_subgroup = 'sex' # sex or WA


dir.create("R/output", showWarnings = FALSE, recursive=TRUE)

crops =        c("SEG_sS1_strips_v4", "SEG_sS1_strips_v6", "DET_dS_strips", "ensemble")
names(crops) = c("SEG1", "SEG2", "DET", "ENSBL")

dat.demo = read.csv("Class_Label/gt/201608-201702_Tranche1and2_accessions_with_partialdemog.csv")
dat.demo$WA = ifelse(dat.demo$ord_loctype_clean=="Outpatient - Mayo", 'non-WA', 'WA')
# add diagnostics
dat_acc_1 = read.csv("Class_Label/gt/201608-201702_Tranche1and2_patient_samples.csv")
dat.demo$FinalHSV1 = dat_acc_1$FinalHSV1[match(dat.demo$StripID, dat_acc_1$strip_id)]
dat.demo$FinalHSV2 = dat_acc_1$FinalHSV2[match(dat.demo$StripID, dat_acc_1$strip_id)]

mytable(dat.demo$sex)

renton.demo = read.csv("Class_Label/gt/rentonaccessions_with_partialdemog.csv")
renton.demo$WA = ifelse(renton.demo$ord_loctype_clean=="Outpatient - Mayo", 'non-WA', 'WA')

tab2x2a=mytable(renton.demo$WA, renton.demo$sex)[,1:2]
tab2x2a
fisher.test(tab2x2a)

tab2x2b=mytable(dat.demo$WA, dat.demo$sex)[,1:2]
tab2x2b
fisher.test(tab2x2b)

tab2x2c=rbind(tab2x2a[2,], tab2x2b[2,])
rownames(tab2x2c) = c("Renton", "Primary")
tab2x2c
fisher.test(tab2x2c)


subgroups = unique(dat.demo[[by_subgroup]])
# remove U for sex
if (by_subgroup=='sex') subgroups = c('F','M')

# mytable(dat.demo$state_impute, dat.demo$ord_loctype_clean)


pvals = sapply (1:2, function(HSV) {
  pvals = c()
  
  # test whether distribution of HSV cases differ by subgroup
  tab = mytable(dat.demo[["FinalHSV"%.%HSV]], dat.demo[[by_subgroup]])
  print(tab)
  pvals = c(pvals, fisher.test(tab[,1:2])$p.value)
  
  res_all = get.dat(HSV, sample_set, model, crop='ensemble')
  res_all[[by_subgroup]] = dat.demo[[by_subgroup]][match(res_all$StripID, dat.demo$StripID)]
  
  # test whether false neg rate differ by subgroup
  tab = with(subset(res_all,GroundTruth==1), table(FinalPredicted, get(by_subgroup)))
  print(tab)
  pvals = c(pvals, fisher.test(tab[,1:2])$p.value)
  
  # test whether false POS rate differ by subgroup
  tab = with(subset(res_all,GroundTruth==0), table(FinalPredicted, get(by_subgroup)))
  print(tab)
  pvals = c(pvals, fisher.test(tab[,1:2])$p.value)
  
  pvals
})
{
pvals= formatPvalues(pvals)
pvals = data.frame(pvals)
rownames(pvals)=c("same prevalence between ", "same recall between ", "same precision between ")%.%by_subgroup
colnames(pvals)=c("HSV-1", "HSV-2")
# add row names as column
pvals = cbind("Hypothesis"=rownames(pvals), pvals)
pvals
}

# classification performance metrics table
{
  tabs=list()
  iter=1
  for (HSV in 1:2) {
  for (subgroup in subgroups) {
      
    res = mysapply (crops, function(crop) {
      res_all = get.dat(HSV, sample_set, model, crop)
      
      res_all[[by_subgroup]] = dat.demo[[by_subgroup]][match(res_all$StripID, dat.demo$StripID)]
      # if (crop=="ensemble" & .sex=='F') print(table(res_all$GroundTruth, res_all$FinalPredicted, res_all[[by_subgroup]]))
      
      res_all = res_all[res_all[[by_subgroup]]==subgroup & !is.na(res_all[[by_subgroup]]), ]
      
      # compare FinalPredicted to GroundTruth to compute ROCAUC, PR_AUC, ACC, F1, Recall, Precision, Specificity
      unlist(metrics_from_preds(res_all$FinalPredicted, res_all$GroundTruth))
    })
    
    tab = as.data.frame(formatDouble(res, 3))
    # add row names as column
    tab = cbind(" "=rownames(tab), tab)
    tabs[[iter]]=tab
    iter=iter+1
    
  }
  }
  
  metrics = rbind(
    HSV1a=c("HSV1, "%.%subgroups[1], rep("",ncol(tab)-1)),
    tabs[[1]], 
    HSV1b=c("HSV1, "%.%subgroups[2], rep("",ncol(tab)-1)), 
    tabs[[2]],
    HSV2a=c("HSV2, "%.%subgroups[1], rep("",ncol(tab)-1)),
    tabs[[3]], 
    HSV2b=c("HSV2, "%.%subgroups[2], rep("",ncol(tab)-1)), 
    tabs[[4]]
  )
  metrics
}



# Function to apply your specific styling quickly
{
  style_my_table <- function(data) {
    flextable(data) %>%
      font(fontname = "Times New Roman", part = "all") %>%
      fontsize(size = 12, part = "all") %>%
      align(j = 1, align = "left", part = "all") %>%     # First col left
      align(j = -1, align = "center", part = "all") %>%  # Others center
      bold(part = "header") %>%
      autofit()
  }
  
  doc <- read_docx("R/table_template.docx")
  
  # Add first table
  doc <- body_add_flextable(doc, value = style_my_table(metrics))
  doc <- body_add_par(doc, "", style = "Normal")
  
  # Add second table
  doc <- body_add_flextable(doc, value = style_my_table(pvals))
  
  print(doc, target = glue("R/output/{sample_set}_{model}_performance_by_{by_subgroup}.docx"))
}


