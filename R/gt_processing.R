################################################################################
# Compute majority label from the three HSV columns
# run from repo root directory: Rscript R/gt_processing.R on volta with ml R/4.2

library(kyotil)
library(readxl)
library(glue)

# read the first sheet in CL and JHCL HSWB Results - Starting 2016.xlsx

# dat=read_xlsx('Class_Label/gt/HSWB Results 201907-202505 checked20260126.xlsx', sheet=1)
# save_as = 'sS_labels_201907-202505.csv'

dat=read_xlsx('Class_Label/gt/HSWB Results 201608-201702 fixed20260126.xlsx', sheet=1)
save_as = 'sS_labels_201608-201702.csv'

names(dat)

# for column 5 to 10, verify that the values are in the code book
for (i in 5:12) {
  print(names(dat)[i])
  print(mytable(dat[[i]]))
}

# correct typos: POSITVE should be POSITIVE      
for (i in 5:12) {
  dat[[i]] = gsub('POSITVE', 'POSITIVE', dat[[i]])
}

# for column 5 to 10, verify that the values are in the code book
for (i in 5:12) {
  print(names(dat)[i])
  print(mytable(dat[[i]]))
}

# rename column Final HSV-1 to FinalHSV1
names(dat)[names(dat) == 'Final HSV-1'] = 'FinalHSV1'
# rename column Final HSV-2 to FinalHSV2
names(dat)[names(dat) == 'Final HSV-2'] = 'FinalHSV2'


# for each row, compute the majority label among the three HSV-1 columns 
dat$MajorityHSV1 = apply(dat[, c("1st HSV-1", "2nd HSV-1", "3rd HSV-1")], 1, function(x) {
  tab = mytable(x)
  names(tab) [which.max(tab)]
})

# for each row, compute the majority label among the three HSV-2 columns
dat$MajorityHSV2 = apply(dat[, c("1st HSV-2", "2nd HSV-2", "3rd HSV-2")], 1, function(x) {
  tab = mytable(x)
  names(tab) [which.max(tab)]
})

# insert a new column as a first column named img_file
dat = cbind(img_file=
              paste(format(as.Date(dat$`Run date`, format='%m/%d/%Y'), '%Y.%m.%d'), 
                    gsub('/', '', dat$`Blot tech`), 
                    sprintf('%02d', dat$`Blot page`), sep='_')
            , dat)

# add a new column named mask_id that starts at 255 and decreases by 2 with each row until it hits a new value of img_file
mask_id = 255
for (i in 1:nrow(dat)) {
  if (i == 1 || dat$img_file[i] != dat$img_file[i-1]) {
    mask_id = 255
  }
  dat$mask_id[i] = mask_id
  mask_id = mask_id - 2
}

# add a new column named strip_id that combines img_file and mask_id
dat$strip_id = paste(dat$img_file, sprintf('%03d', dat$mask_id), sep='_')


# correct a typo in FinalHSV1 and FinalHSV2: R4O should be R40
dat$FinalHSV1[dat$FinalHSV1 == "R4O"] <- "R40"
dat$FinalHSV2[dat$FinalHSV2 == "R4O"] <- "R40"

# add a new column based on FinalHSV1 and FinalHSV2. If FinalHSV1 is not negative or POSITIVE, then it is FinalHSV1. If FinalHSV1 is negative or POSITIVE, then it is FinalHSV1+FinalHSV2.
dat$FinalHSVboth <- dat$FinalHSV1
kp = dat$FinalHSV1 == "negative" | dat$FinalHSV1 == "POSITIVE"
dat$FinalHSVboth[kp] <- paste(substr(dat$FinalHSV1, 1, 3), substr(dat$FinalHSV2, 1, 3), sep = "_")[kp]
mytable(dat$FinalHSVboth)

# combine R40 and RPT as R40RPT
dat$FinalHSVboth[dat$FinalHSVboth == "R40"] <- "R40RPT"
dat$FinalHSVboth[dat$FinalHSVboth == "RPT"] <- "R40RPT"


dat[["Run date"]] <- format(dat[["Run date"]], "%-m/%-d/%Y")


str(dat)

# if the columns changed in the data file, this chunk has to change as well
# add rows for strips in the ablation study. 
new_rows <- read.table(text = "
2016.09.01_CZ_01	9/1/2016	1	CZ	T42509	POSITIVE	negative	POSITIVE	negative	POSITIVE	negative	POSITIVE	negative	NA	NA	NA	POSITIVE	negative	209	2016.09.01_CZ_01_209_white	POS_neg
2016.09.01_CZ_01	9/1/2016	1	CZ	T42509	POSITIVE	negative	POSITIVE	negative	POSITIVE	negative	POSITIVE	negative	NA	NA	NA	POSITIVE	negative	209	2016.09.01_CZ_01_209_white_mask1	POS_neg
2016.09.01_CZ_01	9/1/2016	1	CZ	T42509	POSITIVE	negative	POSITIVE	negative	POSITIVE	negative	POSITIVE	negative	NA	NA	NA	POSITIVE	negative	209	2016.09.01_CZ_01_209_white_mask2	POS_neg
2016.09.01_CZ_01	9/1/2016	1	CZ	T42509	POSITIVE	negative	POSITIVE	negative	POSITIVE	negative	POSITIVE	negative	NA	NA	NA	POSITIVE	negative	209	2016.09.01_CZ_01_209_white_mask3	POS_neg
2016.09.01_CZ_01	9/1/2016	1	CZ	T42509	POSITIVE	negative	POSITIVE	negative	POSITIVE	negative	POSITIVE	negative	NA	NA	NA	POSITIVE	negative	209	2016.09.01_CZ_01_209_white_mask4	POS_neg
2016.09.01_CZ_01	9/1/2016	1	CZ	T42509	POSITIVE	negative	POSITIVE	negative	POSITIVE	negative	POSITIVE	negative	NA	NA	NA	POSITIVE	negative	209	2016.09.01_CZ_01_209_white_mask5	POS_neg
", sep = "\t", stringsAsFactors = FALSE, header = FALSE)
colnames(new_rows) <- colnames(dat)
dat <- rbind(dat, new_rows)


library(readr)
# no quotes unless needed. no row names
write_csv(dat, glue('Class_Label/gt/{save_as}'))


