################################################################################
# Compute majority label from the three HSV columns

library(kyotil)
library(readxl)

# read the first sheet in CL and JHCL HSWB Results - Starting 2016.xlsx

dat=read_xlsx('Image_Annotation\\HSWB Results - Starting 2016 - subset.xlsx', sheet=1)

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

# save the data frame to a file
write.csv(dat, 'Image\\sS_labels.csv', row.names=FALSE)


