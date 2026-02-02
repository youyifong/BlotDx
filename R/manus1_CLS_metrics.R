library(kyotil)
library(exact2x2)
library(glue)
library(gt)
library(officer)
library(flextable)

source("R/helper.R")

library(argparse)
parser <- ArgumentParser()
parser$add_argument("--model", required=TRUE)
parser$add_argument("--sample_set", required=TRUE)
args <- parser$parse_args()

str(args)

model = args$model
# model = "201608-201702_Tr12_pretrained"
# model = "pretrained"

sample_set = args$sample_set
# sample_set = "201608-201702_Tranche2,alltest"
# sample_set = '201608-201702_Tranche1and2'

dir.create("R/output", showWarnings = FALSE, recursive=TRUE)

crops =        c("SEG_sS1_strips_v4", "SEG_sS1_strips_v6", "DET_dS_strips", "ensemble")
names(crops) = c("SEG1", "SEG2", "DET", "ENSBL")



  
# classification performance metrics table
{
tabs=list()
pvals=list()
for (HSV in 1:2) {
  
  res = mysapply (crops, function(crop) {
    res_all = get.dat(HSV, sample_set, model, crop)
    
    # # checking on an issue
    # ptids_file = read.csv("Class_Label/gt/Tranche1and2_927.csv")
    # # file the missing ptid
    # ptids_file$StripID[!ptids_file$StripID %in% res_all$StripID]
    # 
    # 
    # gt = read.csv('Class_Label/gt/sS_labels_201608-201702.csv')
    # 
    # '2016.10.03_CL_04_225' %in% gt$strip_id
    
    # compare FinalPredicted to GroundTruth to compute ROCAUC, PR_AUC, ACC, F1, Recall, Precision, Specificity
    unlist(metrics_from_preds(res_all$FinalPredicted, res_all$GroundTruth))
  })
  
  tab = as.data.frame(formatDouble(res, 3))
  tab = cbind(" "=rownames(tab), tab) # add row names as column
  tabs[[HSV]]=tab
  
  res = lapply (crops, function(crop) {
    res_all = get.dat(HSV, sample_set, model, crop)
  })
  res.all.crops = do.call(cbind, res)
  
  pvals.df = matrix(NA, len(crops), len(crops)) 
  for(iter in 2:len(crops)) {
    for (j in 1:(iter-1)) {
      pvals.df[iter,j] = exact2x2::mcnemar.exact(table(
        res.all.crops[[names(crops)[iter]%.%".FinalPredicted"]], 
        res.all.crops[[names(crops)[j]%.%".FinalPredicted"]]
        ))$p.value
    }
  }
  pvals.df = as.data.frame(formatPvalues(pvals.df))
  pvals.df = cbind(names(crops), pvals.df)
  colnames(pvals.df) = c(" ", names(crops))
  pvals[[HSV]] = pvals.df
  
}

metrics = rbind(HSV1=c("HSV1", rep("",ncol(tab)-1)),
            tabs[[1]], 
            HSV2=c("HSV2", rep("",ncol(tab)-1)), 
            tabs[[2]])

}



# confusion matrices 
confmat.ls = lapply (crops, function(crop) {
  # sink(glue("R/output/{sample_set}_{model}_misclassified.txt"))     # Start capturing output
  res = lapply (1:2, function(HSV) {
    
    res_all = get.dat(HSV, sample_set, model, crop)
    cm <- table(True = res_all$GroundTruth, Predicted = res_all$FinalPredicted)
    cm
    
    # cat(glue("HSV{HSV} POSITIVE misclassified as negative:"),"\n\n")
    # tmp = subset(res_all, GroundTruth==1 & FinalPredicted==0, StripID, drop=F)
    # print(tmp[order(tmp[[1]]),,drop=F])    
    # cat(glue("HSV{HSV} negative misclassified as POSITIVE:"),"\n\n")
    # tmp = subset(res_all, GroundTruth==0 & FinalPredicted==1, StripID, drop=F)
    # print(tmp[order(tmp[[1]]),,drop=F])
  })
  # sink()

  tab = as.data.frame(cbind(res[[1]], res[[2]]))
  # modify column names b/c gt does not allow duplicate col names
  colnames(tab) = c("HSV1_neg","HSV1_POS","HSV2_neg","HSV2_POS")
  # add row names as a column
  confmat = cbind("True:Pred"=c("Negative", "Positive"), tab)
  confmat
})



# Write results to Word tables

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

# Add confusion matrices
for (confmat in confmat.ls) {
  doc <- body_add_flextable(doc, value = style_my_table(confmat))
}

# Add pvals
doc <- body_add_par(doc, "", style = "Normal")
doc <- body_add_flextable(doc, value = style_my_table(pvals[[1]]))
doc <- body_add_flextable(doc, value = style_my_table(pvals[[2]]))

print(doc, target = glue("R/output/{sub('/','',sample_set)}_{model}_performance.docx"))
}
