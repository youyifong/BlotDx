# Manuscript 1 BlotDx

## Setup
```bash setup 

ml R/4.2.0-foss-2021b
ml Python/3.9.6-GCCcore-11.2.0
ml IPython/7.26.0-GCCcore-11.2.0
ml cuDNN/8.9.7.29-CUDA-12.3.0
source ~/envs/tv013/bin/activate
export CUBLAS_WORKSPACE_CONFIG=:4096:8 # for GPU computation reproducibility
cd /fh/fast/fong_y/HSVW/HSVWesternDiagnosticMethods 

```

The first three lines Load cuDNN and Python modules on Volta (the specific Python module is an older version and not available on the Gizmo nodes).
The fourth line loads python virtual environment. Alternatively, for tv014, ml cuDNN/8.4.1.50-CUDA-11.7.0 (works with torchvision 0.14.1+cu117)
CUBLAS_WORKSPACE_CONFIG is needed because of a RuntimeError: Deterministic behavior was enabled, but this operation is not deterministic because it uses CuBLAS and you have CUDA >= 10.2. https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
The working dir is the root of the repo.

```bash jupyter notebook
# Run Jupyter notebook. Go to the url with the hostname.
nohup python3 -m jupyter notebook --ip=$(hostname) --no-browser &

# Convert notebook to HTML without the code.
jupyter nbconvert --to html --no-input your_notebook.ipynb
```

## Stage 1: cropping blots


2016.08.30_JHCL_01 (Tranche 2) needed to be manually sharpened in order to crop all the blots from the image.

### Data processing

All images are resized to 1936x1288. Run code from preprocessing.py interactively.

Use VIA to draw gt masks detection/segmentation, and then run the following to generate mask images.

```bash VIAprojectJSON_to_masks

python Py_pre_post_processing/VIAprojectJSON_to_masks.py --via_json_path Image_Annotation/201907-202505_dS_DET.json --mask_output_dir Mask/201907-202505/gt_dS_DET/

python Py_pre_post_processing/VIAprojectJSON_to_masks.py --via_json_path Image_Annotation/201907-202505_sS_SEG.json --mask_output_dir Mask/201907-202505/gt_sS_SEG/

python Py_pre_post_processing/VIAprojectJSON_to_masks.py --via_json_path Image_Annotation/MJ_sS_SEG.json --mask_output_dir Mask/201608-201702_Tranche1/gt_sS_SEG/

```

### dS detection

#### Detection only

**Train**

```bash train 201608-201702_Tranche1

# model training 
nohup python Py_manus1/tv_DET_train.py --num_classes 2 --dir "trainDETds_CL" --pretrained_model coco --gpu_id 0 --n_epochs 1000 &
lt trainDETds_CL/tvmodels0
# saving the model
cp trainDETds_CL/tvmodels0/fasterrcnn_trained_model_2024_10_18_09_01_27_1000.pth Model/DET_dS.pth

```

```bash train 201608-201702_Tranche1 for revision

# with random sharpening
export gpuid=0

nohup python Py_manus1/tv_DET_train.py --num_classes 2 --train_img_dir "Image/blots/201608-201702_Tranche1/CL" --mask_dir "Image/sheets/201608-201702_Tranche1/gt_dS_DET"  --pretrained_model coco --gpu_id $gpuid --n_epochs 1000 &

# save the latest model in working_model${gpuid} to Model/
cp $(find working_model${gpuid}/* -type f -printf "%T@ %p\n" | sort -n | cut -d' ' -f 2- | tail -n 1)    Model/DET_dS_2.pth

```

```bash train 201907-202505, not used

export gpuid=0

# fine tune original model with 2 external validation dataset images
python Py_manus1/tv_DET_train.py --num_classes 2 --train_img_dir "Image/sheets/201907-202505_train" --mask_dir "Mask/201907-202505/gt_dS_DET"  --pretrained_model "Model/DET_dS.pth" --gpu_id $gpuid --n_epochs 200 
# save the latest model in working_model${gpuid} to Model/
cp $(find working_model${gpuid}/* -type f -printf "%T@ %p\n" | sort -n | cut -d' ' -f 2- | tail -n 1)    Model/DET_dS_Renton_1.pth

```


**Predict**

```bash predict 201608-201702

# making prediction on validation set
python Py_manus1/tv_DET_predict.py --num_classes 2 --dir "Image/validation" --param_set pred_DET_dS --the_model Model/DET_dS.pth --box_score_threshold 0.5 --box_size_threshold_ 1000

python Py_manus1/tv_DET_predict.py --num_classes 2 --dir "Image/validation" --param_set pred_DET_dS_default --the_model Model/DET_dS.pth --box_score_threshold 0.5 

# making prediction on test set
python Py_manus1/tv_DET_predict.py --num_classes 2 --dir "Image/CZ/more" --param_set pred_DET_dS --the_model Model/DET_dS.pth --box_score_threshold 0.5 

# making prediction on training set
python Py_manus1/tv_DET_predict.py --num_classes 2 --dir "Image/CL" --param_set pred_DET_dS --the_model Model/DET_dS.pth --box_score_threshold 0.5 

# making prediction on Tranche 2
python Py_manus1/tv_DET_predict.py --num_classes 2 --dir Image/sheets/201608-201702_Tranche2 --param_set Mask/201608-201702_Tranche2/DET_dS_strips --the_model Model/DET_dS.pth --box_score_threshold 0.3 

# model trained with 21 CL (original) with sharpening augmentation, misses none
python Py_manus1/tv_DET_predict.py --input Image/blots/201608-201702_Tranche1/CL --param_set set4 --the_model "Model/DET_dS_2.pth" --box_size_threshold_L 20000 --box_size_threshold_U 100000 --rpn_nms_threshold 0.7 --box_score_threshold 0.01  

# model trained with 21 CL (original) with sharpening augmentation, misses none
python Py_manus1/tv_DET_predict.py --input Image/sheets/201608-201702_Tranche1 --param_set set4 --the_model "Model/DET_dS_2.pth" --box_size_threshold_L 20000 --box_size_threshold_U 100000 --rpn_nms_threshold 0.7 --box_score_threshold 0.01  

# model trained with 21 CL (original) with sharpening augmentation, misse none
python Py_manus1/tv_DET_predict.py --input Image/sheets/201608-201702_Tranche2 --param_set set4 --the_model "Model/DET_dS_2.pth" --box_size_threshold_L 21500 --box_size_threshold_U 100000 --rpn_nms_threshold 0.7 --box_score_threshold 0.01  

python Py_manus1/tv_DET_predict.py --input Image/sheets/tmp --param_set set4 --the_model "Model/DET_dS_2.pth" --box_size_threshold_L 21500 --box_size_threshold_U 100000 --rpn_nms_threshold 0.7 --box_score_threshold 0.01  


```

```bash predict 201907-202505

# model trained with 21 CL (original), misses 2
python Py_manus1/tv_DET_predict.py --input Image/sheets/201907-202505 --param_set set4 --the_model "Model/DET_dS.pth"  --rpn_nms_threshold 0.7 --box_score_threshold 0.01  

# model trained with 21 CL (original) with sharpening augmentation, misses none
python Py_manus1/tv_DET_predict.py --input Image/sheets/201907-202505 --param_set set4 --the_model "Model/DET_dS_2.pth" --box_size_threshold_L 5000 --box_size_threshold_U 100000 --rpn_nms_threshold 0.7 --box_score_threshold 0.01  

# model trained with 2 images from external validation set, miss none, boxes are a little bigger than those made by pred_DET_dS
python Py_manus1/tv_DET_predict.py --input Image/sheets/201907-202505 --param_set set1 --the_model Model/DET_dS_Renton_1.pth  --rpn_nms_threshold 0.4 --box_score_threshold 0.1

```

#### Detection + classification

```bash
nohup python Py_manus1/tv_DET_train.py --num_classes 11 --dir "trainDETds_CL" --pretrained_model coco --gpu_id 0 --n_epochs 1000 &

lt trainDETds_CL/tvmodels0

cp trainDETds_CL/tvmodels0/fasterrcnn_trained_model_2024_10_20_11_20_39_1000.pth Model/DETCLS_dS.pth

python Py_manus1/tv_DET_predict.py --num_classes 11 --dir "validation" --param_set Py_manus1/predDETdSclass_val --the_model Model/DETdSclass_CL.pth --box_score_threshold 0.5 --box_size_threshold 1000

python Py_manus1/tv_DET_predict.py --num_classes 11 --dir "test" --param_set Py_manus1/predDETdSclass_test --the_model Model/DETdSclass_CL.pth --box_score_threshold 0.5 --box_size_threshold 1000
```

Performance is not good.

### sS detection

trainSEGsS_CL_MJ contains masks, which are used to derive boxes in data loader.

```bash
nohup python Py_manus1/tv_DET_train.py --num_classes 2 --dir "trainSEGsS_CL_MJ" --pretrained_model coco --gpu_id 0 --n_epochs 1000 &

lt trainSEGsS_CL_MJ/tvmodels0

cp trainSEGsS_CL_MJ/tvmodels0/fasterrcnn_trained_model_2024_10_19_11_30_58_1000.pth Model/DETsS_CL_MJ.pth

python Py_manus1/tv_DET_predict.py --num_classes 2 --dir "validation" --param_set Py_manus1/predDETsS_val --the_model Model/DETsS_CL_MJ.pth

python Py_manus1/tv_DET_predict.py --num_classes 2 --dir "CZ/more" --param_set Py_manus1/predDETsS_test --the_model Model/DETsS_CL_MJ.pth
```

All single strips are identified.

### sS segmentation

#### sS

Permute color, use tsp.imread. Miss a few strips.

```bash
nohup python Py_manus1/tv_SEG_train.py --train_img_dir "Image/train_SEG" --mask_dir "Image_Masks_sS_SEG" --data_aug_ctrl 1 --pretrained_model coco --gpu_id 0 --n_epochs 1000 &

lt working_model

cp working_model/maskrcnn_trained_model_2024_10_28_10_56_43_1000.pth Model/SEG_sS.pth

python Py_manus1/tv_SEG_predict.py --input "Image/validation" --param_set "pred_SEG_sS" --the_model "Model/SEG_sS.pth"
```

#### sS1

Not permute color, use cv2.imread(..., -1). 

**Training**

```bash train 201608-201702_Tranche1

nohup python Py_manus1/tv_SEG_train.py --train_img_dir "Image/train_SEG" --mask_dir "Image_Masks_sS_SEG" --data_aug_ctrl 0 --pretrained_model coco --gpu_id 0 --n_epochs 1000 &

lt working_model

cp working_model/maskrcnn_trained_model_2024_10_28_13_20_53_1000.pth Model/SEG_sS1.pth

```

```bash train all of 201608-201702_Tranche1, a total of 32 files, with data_aug_ctrl=1

export gpuid=0

python Py_manus1/tv_SEG_train.py --train_img_dir "Image/sheets/201608-201702_Tranche1" --mask_dir "Mask/201608-201702_Tranche1/gt_sS_SEG" --data_aug_ctrl 1 --pretrained_model coco --gpu_id $gpuid --n_epochs 200 

# save the latest model in working_model${gpuid} to Model/
cp  $(find working_model${gpuid}/* -type f -printf "%T@ %p\n" | sort -n | cut -d' ' -f 2- | tail -n 1)  Model/SEG_sS1_2.pth

```

```bash train all of 201608-201702_Tranche1, a total of 32 files, with data_aug_ctrl=0

export gpuid=0

python Py_manus1/tv_SEG_train.py --train_img_dir "Image/sheets/201608-201702_Tranche1" --mask_dir "Mask/201608-201702_Tranche1/gt_sS_SEG" --data_aug_ctrl 0 --pretrained_model coco --gpu_id $gpuid --n_epochs 200 

# save the latest model in working_model${gpuid} to Model/
cp  $(find working_model${gpuid}/* -type f -printf "%T@ %p\n" | sort -n | cut -d' ' -f 2- | tail -n 1)  Model/SEG_sS1_3.pth

```

```bash train 201608-201702_Tranche1_train2, 21 CL + 5 MJ, with data_aug_ctrl=[False, True], Results are not reproducible because of Mask R-CNN non-determinism.

nohup python Py_manus1/tv_SEG_train.py --train_img_dir "Image/sheets/201608-201702_Tranche1_train2" --mask_dir "gt_sS_SEG" --pretrained_model coco --gpu_id 0 --n_epochs 1000 &

cp  $(find working_model0/* -type f -printf "%T@ %p\n" | sort -n | cut -d' ' -f 2- | tail -n 1)  Model/SEG_sS1_4.pth

sha256sum SEG_sS1_4.pth EG_sS1_9.pth 

```


**Prediction**

```bash predict 201608-201702
python Py_manus1/tv_SEG_predict.py --input "Image/validation" --param_set "Mask/pred_SEG_sS1" --the_model "Model/SEG_sS1.pth"

python Py_manus1/tv_SEG_predict.py --input "Image/CL" --param_set "Mask/pred_SEG_sS1" --the_model "Model/SEG_sS1.pth"  --rpn_nms_threshold 0.7 --box_score_threshold 0.1

# making prediction on 201608-201702_Tranche2
python Py_manus1/tv_SEG_predict.py --input Image/sheets/201608-201702_Tranche2 --param_set Mask/201608-201702_Tranche2/pred_SEG_sS1 --the_model "Model/SEG_sS1.pth"  --rpn_nms_threshold 0.7 --box_score_threshold 0.1

# making prediction on 201608-201702_Tranche1/CL using model trained with 21 CL + 4 MJ, misses only 1: 2016.10.03_CL_02, replace with sharpened version
python Py_manus1/tv_SEG_predict.py --input Image/sheets/201608-201702_Tranche1/CL --param_set set3 --the_model "Model/SEG_sS1_4.pth" --box_size_threshold_L 3000 --box_size_threshold_U 30000 --rpn_nms_threshold 0.7 --box_score_threshold 0.01  --mask_threshold 0.4  --mask_iou_supp_threshold 0.2

# making prediction on 201608-201702_Tranche1 using model trained with 21 CL + 4 MJ, misses only 1: 2016.10.03_CL_02, replace with sharpened version
python Py_manus1/tv_SEG_predict.py --input Image/sheets/201608-201702_Tranche1 --param_set set3 --the_model "Model/SEG_sS1_4.pth" --box_size_threshold_L 3000 --box_size_threshold_U 30000 --rpn_nms_threshold 0.7 --box_score_threshold 0.01  --mask_threshold 0.4  --mask_iou_supp_threshold 0.2

# making prediction on 201608-201702_Tranche2 using model trained with 21 CL + 4 MJ, 
python Py_manus1/tv_SEG_predict.py --input Image/sheets/201608-201702_Tranche2 --param_set set3 --the_model "Model/SEG_sS1_4.pth" --box_size_threshold_L 3000 --box_size_threshold_U 30000 --rpn_nms_threshold 0.7 --box_score_threshold 0.01  --mask_threshold 0.4  --mask_iou_supp_threshold 0.2


# prediction using model trained with 21 CL + 4 MJ
python Py_manus1/tv_SEG_predict.py --input Image/sheets/201608-201702_Tranche1/CL --param_set set3 --the_model "Model/SEG_sS1_5.pth" --box_size_threshold_L 3000 --box_size_threshold_U 30000 --rpn_nms_threshold 0.7 --box_score_threshold 0.01  --mask_threshold 0.4  --mask_iou_supp_threshold 0.2

# prediction using model trained with 21 CL + 4 MJ
python Py_manus1/tv_SEG_predict.py --input Image/sheets/201608-201702_Tranche1/CL --param_set set3 --the_model "Model/SEG_sS1_6.pth" --box_size_threshold_L 3000 --box_size_threshold_U 30000 --rpn_nms_threshold 0.7 --box_score_threshold 0.01  --mask_threshold 0.4  --mask_iou_supp_threshold 0.2

# prediction using model trained with 21 CL + 4 MJ, miss many
python Py_manus1/tv_SEG_predict.py --input Image/sheets/201608-201702_Tranche1/CL --param_set set3 --the_model "Model/SEG_sS1_7.pth" --box_size_threshold_L 3000 --box_size_threshold_U 30000 --rpn_nms_threshold 0.7 --box_score_threshold 0.01  --mask_threshold 0.4  --mask_iou_supp_threshold 0.2

```

```bash predict 201907-202505

# 201907-202505, prediction using model trained with 21 CL + 1 MJ (original), miss 2
python Py_manus1/tv_SEG_predict.py --input Image/sheets/201907-202505 --param_set set3 --the_model "Model/SEG_sS1.pth" --box_size_threshold_L 3000 --box_size_threshold_U 30000 --rpn_nms_threshold 0.7 --box_score_threshold 0.01  --mask_threshold 0.4  --mask_iou_supp_threshold 0.2

# 201907-202505, prediction using model trained with 21 CL + 4 MJ, perfect
python Py_manus1/tv_SEG_predict.py --input Image/sheets/201907-202505 --param_set set3 --the_model "Model/SEG_sS1_4.pth" --box_size_threshold_L 3000 --box_size_threshold_U 30000 --rpn_nms_threshold 0.7 --box_score_threshold 0.01  --mask_threshold 0.4  --mask_iou_supp_threshold 0.2

# 201907-202505, prediction using model trained with 21 CL + 4 MJ, misses 4
python Py_manus1/tv_SEG_predict.py --input Image/sheets/201907-202505 --param_set set3 --the_model "Model/SEG_sS1_5.pth" --box_size_threshold_L 3000 --box_size_threshold_U 30000 --rpn_nms_threshold 0.7 --box_score_threshold 0.01  --mask_threshold 0.4  --mask_iou_supp_threshold 0.2

# 201907-202505, prediction using model trained with 21 CL + 4 MJ, misses many
python Py_manus1/tv_SEG_predict.py --input Image/sheets/201907-202505 --param_set set3 --the_model "Model/SEG_sS1_6.pth" --box_size_threshold_L 3000 --box_size_threshold_U 30000 --rpn_nms_threshold 0.7 --box_score_threshold 0.01  --mask_threshold 0.4  --mask_iou_supp_threshold 0.2

# 201907-202505, prediction using model trained with 21 CL + 4 MJ, misses many 
python Py_manus1/tv_SEG_predict.py --input Image/sheets/201907-202505 --param_set set3 --the_model "Model/SEG_sS1_7.pth" --box_size_threshold_L 3000 --box_size_threshold_U 30000 --rpn_nms_threshold 0.7 --box_score_threshold 0.01  --mask_threshold 0.4  --mask_iou_supp_threshold 0.2

```

Details
- rpn_nms_threshold: larger, more boxes
- mask_threshold 0.4 may be included to include more of the strip, but this changes the mask
- implement a custom NMS to filter out redundant instances (not boxes) with mask_iou_supp_threshold

#### sS2

Permute color, use cv2.imread(..., -1). Miss a few strips.

```bash
nohup python Py_manus1/tv_SEG_train.py --train_img_dir "Image/train_SEG" --mask_dir "Image_Masks_sS_SEG" --data_aug_ctrl 1 --pretrained_model coco --gpu_id 0 --n_epochs 1000 &

lt working_model

cp working_model/maskrcnn_trained_model_2024_10_28_16_20_07_1000.pth Model/SEG_sS2.pth

python Py_manus1/tv_SEG_predict.py --input "Image/validation" --the_model "Model/SEG_sS2.pth"  --param_set "Mask/pred_SEG_sS2"
```

## Stage 2: classification

### Cropping blots and processing gt labels

Run save_cropped_strips.py to generate strip images. Move the strip images to destination and rename them if needed, 

```bash 201608-201702_Tranche1 CL
python Py_pre_post_processing/save_cropped_strips.py --img_dir Image/sheets/201608-201702_Tranche1/CL --mask_dir DET_dS_2_set4 
python Py_pre_post_processing/save_cropped_strips.py --img_dir Image/sheets/201608-201702_Tranche1/CL --mask_dir SEG_sS1_4_set3 --crop_version 4 
python Py_pre_post_processing/save_cropped_strips.py --img_dir Image/sheets/201608-201702_Tranche1/CL --mask_dir SEG_sS1_4_set3 --crop_version 6 
```

```bash 201608-201702_Tranche1, e.g., mv 201608-201702_Tranche1/DET_dS_2_set4 to 201608-201702_Tranche1new/DET_dS_strips
python Py_pre_post_processing/save_cropped_strips.py --img_dir Image/sheets/201608-201702_Tranche1 --mask_dir DET_dS_2_set4 
python Py_pre_post_processing/save_cropped_strips.py --img_dir Image/sheets/201608-201702_Tranche1 --mask_dir SEG_sS1_4_set3 --crop_version 4 
python Py_pre_post_processing/save_cropped_strips.py --img_dir Image/sheets/201608-201702_Tranche1 --mask_dir SEG_sS1_4_set3 --crop_version 6 
```

```bash 201608-201702_Tranche2, e.g., mv 201608-201702_Tranche2/DET_dS_2_set4 to 201608-201702_Tranche2new/DET_dS_strips
python Py_pre_post_processing/save_cropped_strips.py --img_dir Image/sheets/201608-201702_Tranche2 --mask_dir DET_dS_2_set4 
python Py_pre_post_processing/save_cropped_strips.py --img_dir Image/sheets/201608-201702_Tranche2 --mask_dir SEG_sS1_4_set3 --crop_version 4 
python Py_pre_post_processing/save_cropped_strips.py --img_dir Image/sheets/201608-201702_Tranche2 --mask_dir SEG_sS1_4_set3 --crop_version 6 
```

```bash 201608-201702_Tranche2
python Py_common/pre_post_processing/save_cropped_strips.py --img_dir Image/sheets/201608-201702_Tranche2 --mask_dir Mask/201608-201702_Tranche2/pred_DET_dS 
python Py_common/pre_post_processing/save_cropped_strips.py --img_dir Image/sheets/201608-201702_Tranche2 --mask_dir Mask/201608-201702_Tranche2/pred_SEG_sS1 --crop_version 4
python Py_common/pre_post_processing/save_cropped_strips.py --img_dir Image/sheets/201608-201702_Tranche2 --mask_dir Mask/201608-201702_Tranche2/pred_SEG_sS1 --crop_version 6
```

```bash 201907-202505
python Py_pre_post_processing/save_cropped_strips.py --img_dir Image/sheets/201907-202505 --mask_dir DET_dS_2_set4 
python Py_pre_post_processing/save_cropped_strips.py --img_dir Image/sheets/201907-202505 --mask_dir SEG_sS1_4_set3 --crop_version 4
python Py_pre_post_processing/save_cropped_strips.py --img_dir Image/sheets/201907-202505 --mask_dir SEG_sS1_4_set3 --crop_version 6
```


**Ground truth classification labels**

- Modify Image_Annotation/"HSWB Results - Starting 2016 - subset.xlsx" to add new test data. Note that controls do not have rows in "HSWB Results - Starting 2016.xlsx: and so they need to be added to the ground truth class label file in the right order.

- 2026.01.26 We realized there were errors in the spreadsheet. We discovered two types of data entry problems: Page numbers offset -  three images affected, 2017.02.23_CL, 03, 04 and 05, covering about 30 samples in the primary set; Run date drag – affecting a couple of samples in the primary set. The corrected file is named HSWB Results 201608-201702 fixed20260126.xlsx. The external validation data file had no issue, and is renamed HSWB Results 201907-202505


Run R/gt_processing.R to generate Image/sS_labels.csv. See the R script for instructions.

**Turn blots black and white**

```bash turn rgb to bw images

python Py_pre_post_processing/color2bw_lum_hist.py --input Image/blots/201608-201702_Tranche1and2new/DET_dS_strips --output Image/blots/201608-201702_Tranche1and2new_bw/DET_dS_strips 

python Py_pre_post_processing/color2bw_lum_hist.py --input Image/blots/201608-201702_Tranche1and2new/SEG_sS1_strips_v4 --output Image/blots/201608-201702_Tranche1and2new_bw/SEG_sS1_strips_v4 

python Py_pre_post_processing/color2bw_lum_hist.py --input Image/blots/201608-201702_Tranche1and2new/SEG_sS1_strips_v6 --output Image/blots/201608-201702_Tranche1and2new_bw/SEG_sS1_strips_v6 --tiff



python Py_pre_post_processing/color2bw_lum_hist.py --input Image/blots/201608-201702_Tranche1and2/DET_dS_strips --output Image/blots/201608-201702_Tranche1and2_bw/DET_dS_strips 

python Py_pre_post_processing/color2bw_lum_hist.py --input Image/blots/201608-201702_Tranche1and2/SEG_sS1_strips_v4 --output Image/blots/201608-201702_Tranche1and2_bw/SEG_sS1_strips_v4 

python Py_pre_post_processing/color2bw_lum_hist.py --input Image/blots/201608-201702_Tranche1and2/SEG_sS1_strips_v6 --output Image/blots/201608-201702_Tranche1and2_bw/SEG_sS1_strips_v6 --tiff 



python Py_pre_post_processing/color2bw_lum_hist.py --input Image/blots/201907-202505new/DET_dS_strips --output Image/blots/201907-202505new_bw/DET_dS_strips

python Py_pre_post_processing/color2bw_lum_hist.py --input Image/blots/201907-202505new/SEG_sS1_strips_v4 --output Image/blots/201907-202505new_bw/SEG_sS1_strips_v4

python Py_pre_post_processing/color2bw_lum_hist.py --input Image/blots/201907-202505new/SEG_sS1_strips_v6 --output Image/blots/201907-202505new_bw/SEG_sS1_strips_v6 --tiff

```

### Train and predict 

```bash train

# training can be run on several gpus simultaneously. There are 3 GPUs on Volta. GPUID 3 uses the same GPU as GPUID 0, but rng is seeded differently.
export gpuid=0 # 0,1,2, 3,4,5, ... 

export trainingSet="201608-201702_Tranche1and2new"
export gt="Class_Label/gt/sS_labels_201608-201702.csv"
export normalize=1
export sharpening=0
export focal_loss=1
export modelprior=pretrained # nopretrained, pretrained
export modelSuffix="${trainingSet}_$([[ $focal_loss == 1 ]] && echo "focal_")$([[ $sharpening == 1 ]] && echo "sharpen_")$([[ $normalize == 0 ]] && echo "nonorm_")${modelprior}"
export epochs=120

for HSV in 1 2
do        
for cropVersion in DET_dS_strips  SEG_sS1_strips_v4  SEG_sS1_strips_v6
# for cropVersion in gt_strips_v4 gt_strips_v6 gt_dS_DET_strips 
# for cropVersion in SEG_sS1_strips_v4_HSV$HSV
do    
    python Py_manus1/tv_CLS_train.py --HSV $HSV --num_classes 2 --diagnostic_type Final --gpu_id $gpuid --n_epochs $epochs  --train_img_dir Image/blots/${trainingSet}/${cropVersion} --val_img_dir Image/blots/${trainingSet}/${cropVersion} --label_file $gt --pretrained_model $([ "$modelprior" == "nopretrained" ] && echo "None" || echo "IMAGENET1K_V2") --focal_loss $focal_loss --sharpening $sharpening --normalize $normalize
    
    # save the latest model in working_model${gpuid} to Model/
    cp $(find working_model${gpuid}/* -type f -printf "%T@ %p\n" | sort -n | cut -d' ' -f 2- | tail -n 1)    Model/CLS_HSV${HSV}_Final_2classes_${cropVersion}_${modelSuffix}_seed${gpuid}.pth               
done
done

```

```bash predict, make csv and html files
# uses 1 GPU only

export trainingSet="201608-201702_Tranche1and2new"
export test_set="201907-202505new" # under Image/blots/
export test_gt="Class_Label/gt/sS_labels_201907-202505.csv" # sS_labels_201608-201702.csv sS_labels_201907-202505.csv
export sharpening=0
export focal_loss=1
export normalize=1
export modelprior=pretrained # nopretrained, pretrained
export modelSuffix="${trainingSet}_$([[ $focal_loss == 1 ]] && echo "focal_")$([[ $sharpening == 1 ]] && echo "sharpen_")$([[ $normalize == 0 ]] && echo "nonorm_")${modelprior}"

for HSV in 1 2
do
    # for each cropVersion, predict with models seed0, seed1, seed2, and do majority vote
    for cropVersion in SEG_sS1_strips_v4  SEG_sS1_strips_v6  DET_dS_strips
    # for cropVersion in SEG_sS1_strips_v4_HSV$HSV # for sS input study
#    for cropVersion in gt_strips_v4 gt_strips_v6 gt_dS_DET_strips
    do    
        python Py_manus1/tv_CLS_predict.py --HSV $HSV --diagnostic_type Final \
                --normalize ${normalize} \
                --test_img_dir "Image/blots/${test_set}/${cropVersion}" \
                --label_file ${test_gt} \
                --save_to Class_Label/pred/${test_set}\
                --the_model \
Model/CLS_HSV${HSV}_Final_2classes_${cropVersion}_${modelSuffix}_seed0.pth,\
Model/CLS_HSV${HSV}_Final_2classes_${cropVersion}_${modelSuffix}_seed1.pth,\
Model/CLS_HSV${HSV}_Final_2classes_${cropVersion}_${modelSuffix}_seed2.pth
    done    

    # ensemble results from 3 cropVersions 
    python Py_manus1/tv_CLS_en.py --HSV ${HSV} --sample_set ${test_set} --modelSuffix ${modelSuffix}
done

```

```bash R/Rhino, make Word tables (Volta does not have the system libraries required to write docx files)

ml fhR/4.4.0-foss-2023b

## performance in 201907-202505new

Rscript R/manus1_CLS_metrics.R  --model 201608-201702_Tranche1and2new_pretrained  --sample_set 201907-202505new
# True:Pred	-	+	-	+
# Negative	98	0	143	1
# Positive	5	82	6	34
Rscript R/manus1_CLS_metrics.R  --model 201608-201702_Tranche1and2new_bw_pretrained  --sample_set 201907-202505new_bw
# True:Pred	-	+	-	+
# Negative	98	0	143	1
# Positive	3	84	5	35
Rscript R/manus1_CLS_metrics.R  --model 201608-201702_Tranche1and2new_focal_pretrained  --sample_set 201907-202505new
# True:Pred	-	+	-	+
# Negative	97	1	143	1
# Positive	7	80	7	33


## performance in 201608-201702_Tranche1and2

Rscript R/manus1_CLS_metrics.R  --model 201608-201702_Tranche1and2_pretrained  --sample_set 201608-201702_Tranche1and2
# True:Pred	-	+	-	+
# Negative	430	8	722	3
# Positive	10	479	10	192

Rscript R/manus1_CLS_metrics.R  --model 201608-201702_Tranche1and2_bw_pretrained  --sample_set 201608-201702_Tranche1and2_bw
# True:Pred	-	+	-	+
# Negative	436	2	720	4
# Positive	5	483	10	192



Rscript R/manus1_CLS_metrics.R  --model 201608-201702_Tranche1and2new_focal_sharpen_pretrained  --sample_set 201907-202505new
#   True:Pred HSV1_neg HSV1_POS HSV2_neg HSV2_POS
# 0  Negative       98        1      143        2
# 1  Positive        5       81        7       32



Rscript R/manus1_CLS_metrics.R  --model 201608-201702_Tranche1and2new_bw_nonorm_pretrained  --sample_set 201907-202505new_bw
#   True:Pred HSV1_neg HSV1_POS HSV2_neg HSV2_POS
# 0  Negative       96        2      129       15
# 1  Positive        1       86        5       35


Rscript R/manus1_CLS_metrics.R  --model 201608-201702_Tranche1newCL_pretrained  --sample_set 201907-202505new
#   True:Pred HSV1_neg HSV1_POS HSV2_neg HSV2_POS
# 0  Negative       98        0      143        1
# 1  Positive        7       80        9       31
Rscript R/manus1_CLS_metrics.R  --model 201608-201702_Tranche1newCL_focal_sharpen_pretrained  --sample_set 201907-202505new
#   True:Pred HSV1_neg HSV1_POS HSV2_neg HSV2_POS
# 0  Negative       95        3      141        3
# 1  Positive       14       73        3       37
Rscript R/manus1_CLS_metrics.R  --model 201608-201702_Tr1trainnew_focal2_sharpen_pretrained  --sample_set 201907-202505new
#  True:Pred HSV1_neg HSV1_POS HSV2_neg HSV2_POS
#0  Negative       98        0      141        3
#1  Positive        7       80        3       37


Rscript R/manus1_CLS_metrics.R  --model 201608-201702_Tr1new_pretrained  --sample_set 201907-202505new
#  True:Pred HSV1_neg HSV1_POS HSV2_neg HSV2_POS
#0  Negative       98        0      143        1
#1  Positive        7       80        5       35


Rscript R/manus1_CLS_metrics.R  --model 201608-201702_Tr1train_focal2_sharpen_pretrained  --sample_set 201907-202505new
#  True:Pred HSV1_neg HSV1_POS HSV2_neg HSV2_POS
#0  Negative       98        0      144        0
#1  Positive       12       75       13       27

Rscript R/manus1_CLS_metrics.R  --model 201608-201702_Tr1train_focal_sharpen_pretrained  --sample_set 201907-202505new
#  True:Pred HSV1_neg HSV1_POS HSV2_neg HSV2_POS
#0  Negative       97        1      143        1
#1  Positive       14       73        9       31

Rscript R/manus1_CLS_metrics.R  --model 201608-201702_Tr1train_focal_pretrained  --sample_set 201907-202505new
#  True:Pred HSV1_neg HSV1_POS HSV2_neg HSV2_POS
#0  Negative       96        2      142        2
#1  Positive       19       68       11       29

Rscript R/manus1_CLS_metrics.R  --model 201608-201702_Tr1train_sharpen_pretrained  --sample_set 201907-202505new
#  True:Pred HSV1_neg HSV1_POS HSV2_neg HSV2_POS
#0  Negative       98        0      143        1
#1  Positive       10       77        9       31

Rscript R/manus1_CLS_metrics.R  --model pretrained  --sample_set 201907-202505new
#  True:Pred HSV1_neg HSV1_POS HSV2_neg HSV2_POS
#0  Negative       98        0      143        1
#1  Positive        8       79       10       30



Rscript R/manus1_CLS_metrics.R  --model 201608-201702_Tr1train_focal2_sharpen_pretrained  --sample_set 201907-202505
#  True:Pred HSV1_neg HSV1_POS HSV2_neg HSV2_POS
#0  Negative       97        1      143        1
#1  Positive       10       77       13       27

Rscript R/manus1_CLS_metrics.R  --model 201608-201702_Tr1train_focal_sharpen_pretrained  --sample_set 201907-202505
#  True:Pred HSV1_neg HSV1_POS HSV2_neg HSV2_POS
#0  Negative       98        0      142        2
#1  Positive       16       71        6       34

Rscript R/manus1_CLS_metrics.R  --model 201608-201702_Tr1train_sharpen_pretrained  --sample_set 201907-202505
#  True:Pred HSV1_neg HSV1_POS HSV2_neg HSV2_POS
#0  Negative       98        0      143        1
#1  Positive        7       80        7       33

Rscript R/manus1_CLS_metrics.R  --model pretrained  --sample_set 201907-202505
#  True:Pred HSV1_neg HSV1_POS HSV2_neg HSV2_POS
#0  Negative       98        0      143        1
#1  Positive        8       79       10       30

Rscript R/manus1_CLS_metrics.R  --model 201608-201702_Tr1train_focal_pretrained  --sample_set 201907-202505
#  True:Pred HSV1_neg HSV1_POS HSV2_neg HSV2_POS
#0  Negative       95        3      142        2
#1  Positive       22       65       13       27



Rscript R/manus1_CLS_metrics.R  --model pretrained  --sample_set 201608-201702_Tranche2,201608-201702_Tranche1/alltest
#  True:Pred HSV1_neg HSV1_POS HSV2_neg HSV2_POS
#0  Negative      290        6      458        7
#1  Positive       24      292       10      137


```

```bash predict with one model one seed, used less often

python Py_manus1/tv_CLS_predict.py --HSV $HSV --diagnostic_type Final --test_img_dir $test_img_dir  --the_model Model/CLS_HSV${HSV}_Final_2classes_$cropVersion${modelprior}_seed${gpuid}.pth 

```

```bash feature_extraction
for seed in 0 1 2
do
    python Py_manus1/tv_CLS_predict_CAM_explore.py --HSV 1 --model CLS_HSV1_Final_2classes_SEG_sS1_strips_v4_nopretrained_seed${seed}.pth
    python Py_manus1/tv_CLS_predict_CAM_explore.py --HSV 2 --model CLS_HSV2_Final_2classes_SEG_sS1_strips_v4_nopretrained_seed${seed}.pth
done
```

### 5-CV train and predict

In three terminals with export gpuid=0 1 2, respectively, run the following commands to train:

```bash 5-cv train

nohup Py_manus1/cross_validated.sh  >nohup${gpuid}.out 2>&1 &

```

The ensemble step in the following creates a single file with predictions from all folds, which can be used to get the list of samples in the study because the list is a subset of all the blots whose final predictions are POS or neg. An html page containing all misclassified strips is also created.

```bash prediction for all folds
# uses only 1 GPU

export test_set="201608-201702_Tranche1and2_bw" # under Image/blots/
export test_gt="Class_Label/gt/sS_labels_201608-201702.csv" 
export sharpening=0
export focal_loss=0
export normalize=1
export modelprior=pretrained # nopretrained, pretrained
export modelSuffix="${test_set}_$([[ $focal_loss == 1 ]] && echo "focal_")$([[ $sharpening == 1 ]] && echo "sharpen_")$([[ $normalize == 0 ]] && echo "nonorm_")${modelprior}"

for HSV in 1 2
do
    for fold in 0 1 2 3 4
    do
    for cropVersion in SEG_sS1_strips_v4 SEG_sS1_strips_v6 DET_dS_strips
    do
        python Py_manus1/tv_CLS_predict.py --HSV ${HSV} --diagnostic_type Final \
            --normalize ${normalize} \
            --test_img_dir "Image/blots/${test_set}/${cropVersion}/cv${fold}/test" \
            --label_file ${test_gt} \
            --save_to Class_Label/pred/${test_set} \
            --the_model \
Model/CLS_HSV${HSV}_Final_2classes_${cropVersion}_${modelSuffix}_fold${fold}_seed0.pth,\
Model/CLS_HSV${HSV}_Final_2classes_${cropVersion}_${modelSuffix}_fold${fold}_seed1.pth,\
Model/CLS_HSV${HSV}_Final_2classes_${cropVersion}_${modelSuffix}_fold${fold}_seed2.pth
    done
    done

    python Py_manus1/tv_CLS_en.py --HSV ${HSV} --sample_set ${test_set} --modelSuffix ${modelSuffix} --cv
done

```



