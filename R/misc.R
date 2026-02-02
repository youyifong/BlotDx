################################################################################
# combine list of samples and accession 

library(kyotil)
primary <- read.csv("Class_Label/pred/201608-201702_Tranche1and2/HSV1_Final_2classes_ensemble_201608-201702_Tranche1and2_pretrained.csv", header = TRUE)
dat_acc = read.csv("Class_Label/gt/sS_labels_201608-201702.csv", header = TRUE)
# subset to primary dataset
dat_acc_1 = dat_acc[dat_acc$strip_id %in% primary$StripID,]
dat_acc_1 = subset(dat_acc_1, !is.na(Accession.number))
mywrite.csv(dat_acc_1, "Class_Label/gt/201608-201702_Tranche1and2_patient_samples")

# I gave Pavitra 201608-201702_Tranche1and2_patient_samples.csv 
# she made 201608-201702_Tranche1and2_accessions_with_partialdemog.doc 
# Note that 2016.08.04_CL_02_255 and 2016.08.04_CL_02_253 have the same accession for different time points from the same person

demo = read.csv("Class_Label/gt/201608-201702_Tranche1and2_accessions_with_partialdemog.csv")


# following chunk is out of date
library(kyotil)
dat_label <- read.csv("Class_Label/gt/sS_labels_201907-202505.csv", header = TRUE)
dat_acc = read.csv("Class_Label/gt/Renton1_184.csv", header = TRUE)
dat_acc$Accession = dat_label$Accession.number[match(dat_acc$StripID, dat_label$strip_id)]
dat_acc$FinalHSV1 = dat_label$FinalHSV1[match(dat_acc$StripID, dat_label$strip_id)]
dat_acc$FinalHSV2 = dat_label$FinalHSV2[match(dat_acc$StripID, dat_label$strip_id)]
mywrite.csv(dat_acc, "Class_Label/gt/Renton1_184_.csv")



################################################################################
# compare the gt label before and after 2026.01.26 fix

dat = read.csv("Class_Label/gt/Tranche1and2_927.csv")

newdat= read.csv("Class_Label/gt/sS_labels_201608-201702.csv")

dat$FinalHSV1new = newdat$FinalHSV1[match(dat$StripID, newdat$strip_id)]
dat$FinalHSV2new = newdat$FinalHSV2[match(dat$StripID, newdat$strip_id)]

mytable(dat$FinalHSV1new, dat$FinalHSV1)


# list strips that are POS or neg

img_path <- "Image/blots/201608-201702_Tranche1/DET_dS_strips"
file_list <- list.files(path = img_path, pattern = "\\.png$", full.names = TRUE, recursive = FALSE)
file_names <- basename(file_list)
file_stems <- tools::file_path_sans_ext(file_names)
tranche1 = file_stems

img_path <- "Image/blots/201608-201702_Tranche2/DET_dS_strips"
file_list <- list.files(path = img_path, pattern = "\\.png$", full.names = TRUE, recursive = FALSE)
file_names <- basename(file_list)
file_stems <- tools::file_path_sans_ext(file_names)
tranche2 = file_stems

mytable(dat$FinalHSV1new[dat$StripID %in% tranche1], dat$FinalHSV1[dat$StripID %in% tranche1])
mytable(dat$FinalHSV1new[dat$StripID %in% tranche2], dat$FinalHSV1[dat$StripID %in% tranche2])




################################################################################
# tabulate the class_id and class columns in the DS_detection_class.csv file

library(kyotil)

dat=read.csv('Image_Annotation\\CL_JHCL_sS_SEG_json_maskIDclassID.csv')

names(dat)

mytable(dat$file_name)

dat=subset(dat, !contain(file_name, 'JHCL'))

# get a mapping between class_id and class from dat
tab = mytable(dat$class_id, dat$class)
# for each row, find the name of the column with the maximum value
label = colnames(tab) [apply(tab, 1, function(x) which.max(x))]

tab = as.matrix(mytable(dat$class_id), ncol=1)
colnames(tab) = c('count')
tab = cbind(label, tab)
print(tab, quote=F)



################################################################################
# distribution of FinalHSVboth

dat_label <- read.csv("Class_Label\\gt\\sS_labels.csv", header = TRUE)
mytable(dat_label$FinalHSVboth)


################################################################################
# figure out the distribution of INDET

library(readxl)
library(dplyr)

dat1=read_excel("Class_Label/gt/HSWB Results 201608-201702 fixed20260126.xlsx", sheet = "HSWB")
# change column 'Accession number' to Accession
dat1 = rename(dat1, Accession = `Accession number`)

n=sum(mytable(dat1$`Final HSV-1`)[c("negative","POSITIVE")])
N=len(!is.na(dat1$Accession))-1
print(glue("{n}/{N}={n/N}") )

dat2=read_excel("Class_Label/gt/HSWB Results 201608-201702 fixed20260126.xlsx", sheet = "HSWB Ads")
# change column 'Accession number' to Accession
dat2 = rename(dat2, Accession = `Accession number`)

# filter dat2 by Accession number from dat1
dat2 = subset(dat2, Accession %in% dat1$Accession)
# tabluate dat2$"Final HSV-1"
tab = mytable(dat2$`Final HSV-1`); tab
# tabulate dat2$"Final HSV-2"
tab = mytable(dat2$`Final HSV-2`); tab
# cross tabulate dat2$"Final HSV-1" and dat2$"Final HSV-2"
tab = mytable(dat2$`Final HSV-1`, dat2$`Final HSV-2`); tab


################################################################################
# binomial confidence interval

library(Hmisc)
library(kyotil)
library(glue)

# accuracy confint
x=176; n=184
res=formatDouble(binconf(x,n)*100, 1); glue("{res[,1]}% (95% CI: {res[,2]}%-{res[,3]}%)")

# 95.7% (95% CI: 91.7%-97.8%)


# compare accuracy between primary and external

# HSV1
fisher.test(matrix(c(11, 915, 5, 180),2))$p.val
# 0.165

# HSV2
fisher.test(matrix(c(10, 916, 7, 177),2))$p.val
# 0.014

################################################################################
# compare different classifier performance in the external dataset

library(kyotil)
library(glue)
library(exact2x2)

# 6a

a.hsv1.seg1=read.csv("Class_Label/pred/201907-202505new/HSV1_Final_2classes_SEG_sS1_strips_v4_201608-201702_Tranche1and2new_pretrained.csv")
a.hsv1.seg2=read.csv("Class_Label/pred/201907-202505new/HSV1_Final_2classes_SEG_sS1_strips_v6_201608-201702_Tranche1and2new_pretrained.csv")
a.hsv1.det     =read.csv("Class_Label/pred/201907-202505new/HSV1_Final_2classes_DET_dS_strips_201608-201702_Tranche1and2new_pretrained.csv")
a.hsv1.en      =read.csv("Class_Label/pred/201907-202505new/HSV1_Final_2classes_ensemble_201608-201702_Tranche1and2new_pretrained.csv")

exact2x2::mcnemar.exact(table(a.hsv1.seg1$FinalPredicted, a.hsv1.seg2$FinalPredicted))$p.value
exact2x2::mcnemar.exact(table(a.hsv1.seg1$FinalPredicted, a.hsv1.det$FinalPredicted))$p.value
exact2x2::mcnemar.exact(table(a.hsv1.seg2$FinalPredicted, a.hsv1.det$FinalPredicted))$p.value

a.hsv2.seg1=read.csv("Class_Label/pred/201907-202505new/HSV2_Final_2classes_SEG_sS1_strips_v4_201608-201702_Tranche1and2new_pretrained.csv")
a.hsv2.seg2=read.csv("Class_Label/pred/201907-202505new/HSV2_Final_2classes_SEG_sS1_strips_v6_201608-201702_Tranche1and2new_pretrained.csv")
a.hsv2.det     =read.csv("Class_Label/pred/201907-202505new/HSV2_Final_2classes_DET_dS_strips_201608-201702_Tranche1and2new_pretrained.csv")
a.hsv2.en      =read.csv("Class_Label/pred/201907-202505new/HSV2_Final_2classes_ensemble_201608-201702_Tranche1and2new_pretrained.csv")

exact2x2::mcnemar.exact(table(a.hsv2.seg1$FinalPredicted, a.hsv2.seg2$FinalPredicted))$p.value
exact2x2::mcnemar.exact(table(a.hsv2.seg1$FinalPredicted, a.hsv2.det$FinalPredicted))$p.value
exact2x2::mcnemar.exact(table(a.hsv2.seg2$FinalPredicted, a.hsv2.det$FinalPredicted))$p.value


# 6b

b.hsv1.seg1=read.csv("Class_Label/pred/201907-202505new_bw/HSV1_Final_2classes_SEG_sS1_strips_v4_201608-201702_Tranche1and2new_bw_pretrained.csv")
b.hsv1.seg2=read.csv("Class_Label/pred/201907-202505new_bw/HSV1_Final_2classes_SEG_sS1_strips_v6_201608-201702_Tranche1and2new_bw_pretrained.csv")
b.hsv1.det     =read.csv("Class_Label/pred/201907-202505new_bw/HSV1_Final_2classes_DET_dS_strips_201608-201702_Tranche1and2new_bw_pretrained.csv")
b.hsv1.en           =read.csv("Class_Label/pred/201907-202505new_bw/HSV1_Final_2classes_ensemble_201608-201702_Tranche1and2new_bw_pretrained.csv")

exact2x2::mcnemar.exact(table(b.hsv1.seg1$FinalPredicted, b.hsv1.seg2$FinalPredicted))$p.value
exact2x2::mcnemar.exact(table(b.hsv1.seg1$FinalPredicted, b.hsv1.det$FinalPredicted))$p.value
exact2x2::mcnemar.exact(table(b.hsv1.seg2$FinalPredicted, b.hsv1.det$FinalPredicted))$p.value

exact2x2::mcnemar.exact(table(b.hsv1.en$FinalPredicted, b.hsv1.seg1$FinalPredicted))$p.value
exact2x2::mcnemar.exact(table(b.hsv1.en$FinalPredicted, b.hsv1.seg2$FinalPredicted))$p.value
exact2x2::mcnemar.exact(table(b.hsv1.en$FinalPredicted, b.hsv1.det$FinalPredicted))$p.value

b.hsv2.seg1=read.csv("Class_Label/pred/201907-202505new_bw/HSV2_Final_2classes_SEG_sS1_strips_v4_201608-201702_Tranche1and2new_bw_pretrained.csv")
b.hsv2.seg2=read.csv("Class_Label/pred/201907-202505new_bw/HSV2_Final_2classes_SEG_sS1_strips_v6_201608-201702_Tranche1and2new_bw_pretrained.csv")
b.hsv2.det     =read.csv("Class_Label/pred/201907-202505new_bw/HSV2_Final_2classes_DET_dS_strips_201608-201702_Tranche1and2new_bw_pretrained.csv")
b.hsv2.en           =read.csv("Class_Label/pred/201907-202505new_bw/HSV2_Final_2classes_ensemble_201608-201702_Tranche1and2new_bw_pretrained.csv")

exact2x2::mcnemar.exact(table(b.hsv2.seg1$FinalPredicted, b.hsv2.seg2$FinalPredicted))$p.value
exact2x2::mcnemar.exact(table(b.hsv2.seg1$FinalPredicted, b.hsv2.det$FinalPredicted))$p.value
exact2x2::mcnemar.exact(table(b.hsv2.seg2$FinalPredicted, b.hsv2.det$FinalPredicted))$p.value



# 6c

c.hsv1.seg1=read.csv("Class_Label/pred/201907-202505new/HSV1_Final_2classes_SEG_sS1_strips_v4_201608-201702_Tranche1and2new_focal_pretrained.csv")
c.hsv1.seg2=read.csv("Class_Label/pred/201907-202505new/HSV1_Final_2classes_SEG_sS1_strips_v6_201608-201702_Tranche1and2new_focal_pretrained.csv")
c.hsv1.det     =read.csv("Class_Label/pred/201907-202505new/HSV1_Final_2classes_DET_dS_strips_201608-201702_Tranche1and2new_focal_pretrained.csv")
c.hsv1.en      =read.csv("Class_Label/pred/201907-202505new/HSV1_Final_2classes_ensemble_201608-201702_Tranche1and2new_focal_pretrained.csv")

exact2x2::mcnemar.exact(table(c.hsv1.seg1$FinalPredicted, c.hsv1.seg2$FinalPredicted))$p.value
exact2x2::mcnemar.exact(table(c.hsv1.seg1$FinalPredicted, c.hsv1.det$FinalPredicted))$p.value
exact2x2::mcnemar.exact(table(c.hsv1.seg2$FinalPredicted, c.hsv1.det$FinalPredicted))$p.value

c.hsv2.seg1=read.csv("Class_Label/pred/201907-202505new/HSV2_Final_2classes_SEG_sS1_strips_v4_201608-201702_Tranche1and2new_focal_pretrained.csv")
c.hsv2.seg2=read.csv("Class_Label/pred/201907-202505new/HSV2_Final_2classes_SEG_sS1_strips_v6_201608-201702_Tranche1and2new_focal_pretrained.csv")
c.hsv2.det     =read.csv("Class_Label/pred/201907-202505new/HSV2_Final_2classes_DET_dS_strips_201608-201702_Tranche1and2new_focal_pretrained.csv")
c.hsv2.en           =read.csv("Class_Label/pred/201907-202505new/HSV2_Final_2classes_ensemble_201608-201702_Tranche1and2new_focal_pretrained.csv")

exact2x2::mcnemar.exact(table(c.hsv2.seg1$FinalPredicted, c.hsv2.seg2$FinalPredicted))$p.value
exact2x2::mcnemar.exact(table(c.hsv2.seg1$FinalPredicted, c.hsv2.det$FinalPredicted))$p.value
exact2x2::mcnemar.exact(table(c.hsv2.seg2$FinalPredicted, c.hsv2.det$FinalPredicted))$p.value

exact2x2::mcnemar.exact(table(c.hsv2.en$FinalPredicted, c.hsv2.seg1$FinalPredicted))$p.value
exact2x2::mcnemar.exact(table(c.hsv2.en$FinalPredicted, c.hsv2.seg2$FinalPredicted))$p.value
exact2x2::mcnemar.exact(table(c.hsv2.en$FinalPredicted, c.hsv2.det$FinalPredicted))$p.value


# compare across a,b,c
hsv2.pos=a.hsv2.seg1$GroundTruth==1

exact2x2::mcnemar.exact(table(a.hsv1.seg1$FinalPredicted, b.hsv1.seg1$FinalPredicted))$p.value
exact2x2::mcnemar.exact(table(a.hsv2.seg1$FinalPredicted, b.hsv2.seg1$FinalPredicted))$p.value

exact2x2::mcnemar.exact(table(b.hsv2.seg1$FinalPredicted, c.hsv2.seg1$FinalPredicted))$p.value
exact2x2::mcnemar.exact(table(b.hsv2.seg2$FinalPredicted, c.hsv2.seg2$FinalPredicted))$p.value
exact2x2::mcnemar.exact(table(b.hsv2.det$FinalPredicted, c.hsv2.det$FinalPredicted))$p.value

exact2x2::mcnemar.exact(table(a.hsv1.en$FinalPredicted, b.hsv1.en$FinalPredicted))$p.value
exact2x2::mcnemar.exact(table(a.hsv1.en$FinalPredicted, c.hsv1.en$FinalPredicted))$p.value

exact2x2::mcnemar.exact(table(a.hsv2.en$FinalPredicted, b.hsv2.en$FinalPredicted))$p.value
exact2x2::mcnemar.exact(table(a.hsv2.en$FinalPredicted, c.hsv2.en$FinalPredicted))$p.value


# primary dataset, 5-fold cv

# HSV-1
cv.hsv1.seg1 = do.call(rbind, lapply(0:4, function(fold) read.csv(glue('Class_Label/pred/201608-201702_Tranche1and2/HSV1_SEG_sS1_strips_v4_fold{fold}_pretrained.csv'), header=T)))
cv.hsv1.seg2 = do.call(rbind, lapply(0:4, function(fold) read.csv(glue('Class_Label/pred/201608-201702_Tranche1and2/HSV1_SEG_sS1_strips_v6_fold{fold}_pretrained.csv'), header=T)))
cv.hsv1.det  = do.call(rbind, lapply(0:4, function(fold) read.csv(glue('Class_Label/pred/201608-201702_Tranche1and2/HSV1_DET_dS_strips_fold{fold}_pretrained.csv'), header=T)))
cv.hsv1.en   = do.call(rbind, lapply(0:4, function(fold) read.csv(glue('Class_Label/pred/201608-201702_Tranche1and2/HSV1_ensemble_fold{fold}_pretrained.csv'), header=T)))

exact2x2::mcnemar.exact(table(cv.hsv1.seg1$FinalPredicted, cv.hsv1.seg2$FinalPredicted))$p.value
exact2x2::mcnemar.exact(table(cv.hsv1.seg1$FinalPredicted, cv.hsv1.det$FinalPredicted))$p.value
exact2x2::mcnemar.exact(table(cv.hsv1.seg2$FinalPredicted, cv.hsv1.det$FinalPredicted))$p.value

exact2x2::mcnemar.exact(table(cv.hsv1.en$FinalPredicted, cv.hsv1.seg1$FinalPredicted))$p.value
exact2x2::mcnemar.exact(table(cv.hsv1.en$FinalPredicted, cv.hsv1.seg2$FinalPredicted))$p.value
exact2x2::mcnemar.exact(table(cv.hsv1.en$FinalPredicted, cv.hsv1.det$FinalPredicted))$p.value

# HSV-2
cv.hsv2.seg1 = do.call(rbind, lapply(0:4, function(fold) read.csv(glue('Class_Label/pred/201608-201702_Tranche1and2/HSV2_SEG_sS1_strips_v4_fold{fold}_pretrained.csv'), header=T)))
cv.hsv2.seg2 = do.call(rbind, lapply(0:4, function(fold) read.csv(glue('Class_Label/pred/201608-201702_Tranche1and2/HSV2_SEG_sS1_strips_v6_fold{fold}_pretrained.csv'), header=T)))
cv.hsv2.det  = do.call(rbind, lapply(0:4, function(fold) read.csv(glue('Class_Label/pred/201608-201702_Tranche1and2/HSV2_DET_dS_strips_fold{fold}_pretrained.csv'), header=T)))
cv.hsv2.en   = do.call(rbind, lapply(0:4, function(fold) read.csv(glue('Class_Label/pred/201608-201702_Tranche1and2/HSV2_ensemble_fold{fold}_pretrained.csv'), header=T)))

exact2x2::mcnemar.exact(table(cv.hsv2.seg1$FinalPredicted, cv.hsv2.seg2$FinalPredicted))$p.value
exact2x2::mcnemar.exact(table(cv.hsv2.seg1$FinalPredicted, cv.hsv2.det$FinalPredicted))$p.value
exact2x2::mcnemar.exact(table(cv.hsv2.seg2$FinalPredicted, cv.hsv2.det$FinalPredicted))$p.value

exact2x2::mcnemar.exact(table(cv.hsv2.en$FinalPredicted, cv.hsv2.seg1$FinalPredicted))$p.value
exact2x2::mcnemar.exact(table(cv.hsv2.en$FinalPredicted, cv.hsv2.seg2$FinalPredicted))$p.value
exact2x2::mcnemar.exact(table(cv.hsv2.en$FinalPredicted, cv.hsv2.det$FinalPredicted))$p.value
