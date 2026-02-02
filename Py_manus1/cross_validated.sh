#!/bin/bash

echo $gpuid

export trainingSet="201608-201702_Tranche1and2_bw"
export gt="Class_Label/gt/sS_labels_201608-201702.csv"
export normalize=1
export focal_loss=0
export sharpening=0
export modelprior=pretrained # nopretrained, pretrained
export modelSuffix="${trainingSet}_$([[ $focal_loss == 1 ]] && echo "focal_")$([[ $sharpening == 1 ]] && echo "sharpen_")$([[ $normalize == 0 ]] && echo "nonorm_")${modelprior}"

export epochs=120

for HSV_type in 1 2
do
for fold in 0 1 2 3 4
do
for cropVersion in SEG_sS1_strips_v4  SEG_sS1_strips_v6  DET_dS_strips
do
  export train_img_dir="Image/blots/${trainingSet}/${cropVersion}/cv${fold}/train"

  python Py_manus1/tv_CLS_train.py --HSV $HSV_type --num_classes 2 --diagnostic_type Final --gpu_id $gpuid --n_epochs $epochs \
   --train_img_dir ${train_img_dir} --val_img_dir ${train_img_dir} \
   --label_file $gt \
   --pretrained_model $([ "$modelprior" == "nopretrained" ] && echo "None" || echo "IMAGENET1K_V2") \
   --focal_loss $focal_loss --sharpening $sharpening --normalize $normalize

  cp $(find working_model${gpuid}/* -type f -printf "%T@ %p\n" | sort -n | cut -d' ' -f 2- | tail -n 1)  Model/CLS_HSV${HSV_type}_Final_2classes_${cropVersion}_${modelSuffix}_fold${fold}_seed${gpuid}.pth

done
done
done
