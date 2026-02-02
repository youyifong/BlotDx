#!/usr/bin/env python
# coding: utf-8
#This script explore CAM map 

import sys, os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn.functional as F # noqa
import numpy as np
import pandas as pd
import argparse

# Add the repo root to the library path. No need to do this if in Ipython
in_ipython = 'get_ipython' in globals()
if not in_ipython:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Py_common.tv_utils import fix_all_seeds_torch
from Py_common.tv_Dataset_strips import ValDataset_strips
from Py_manus1.CBAM import ResNet50_CBAM, predict, find_item,tensor_to_rgb_image, CAM_Manual
from Py_manus1.CBAM import plot_unique_gradient_histgram, plot_gradient_heatmap, select_info, order_informations, plot_function
from Py_manus1.CBAM import LabelProcessor, get_prediction_df, compute_performance, AllImageInfo, get_sorted_score_idxes


#Run sbatch:
# source ~/.bashrc
# conda activate hsvw
# python3 -u TV/tv_CLS_predict_CAM_explore.py --HSV 1 --selected_sp 2016.09.22_CZ_03_213 2016.09.22_CZ_03_215 2016.09.22_CZ_03_211 --sort_by abs_activation
# python3 -u TV/tv_CLS_predict_CAM_explore.py --HSV 1 --selected_sp 2016.09.22_CZ_03_213 2016.09.22_CZ_03_215 2016.09.22_CZ_03_211 --sort_by abs_gradients


##############################################################################################################################
##########################################      Set arguments                       ##########################################
##############################################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--HSV', default='1', type=int, help='1 or 2')
parser.add_argument('--num_classes', default=2, type=int, help='Number of classes. Default: %(default)s')
parser.add_argument('--diagnostic_type', default='Final', type=str, help='Final, Majority, ...')

parser.add_argument('--input_data_name', default='SEG_sS1_strips_v4', type=str, help='the input data folder suffix (e.g., SEG_sS1_strips_v6, SEG_sS1_strips_v4, DET_dS_strips)')
#parser.add_argument('--train_img_dir', default='Image/CL_SEG_sS1_strips_v4', type=str, help='folder directory containing training image and mask files. There can be unused mask files.')
#parser.add_argument('--val_img_dir', default='Image/validation_SEG_sS1_strips_v4', type=str, help='folder directory containing training image and mask files. There can be unused mask files.')
#parser.add_argument('--test_img_dir', default='Image/test_SEG_sS1_strips_v4', type=str, help='folder directory containing training image and mask files. There can be unused mask files.')
parser.add_argument('--label_file', default='Class_Label/gt/sS_labels.csv', type=str, help='folder directory containing labels file')
parser.add_argument('--mask_dir', default='None', type=str, help='if None, imgs are already cropped; else, folder directory containing training image and mask files. There can be unused mask files.')
parser.add_argument('--batch_size', default=24, type=int, help='batch size. Default: %(default)s')
parser.add_argument('--gpu_id', default=0, type=int, help='which gpu to use. Default: %(default)s')
parser.add_argument('--model', default='CLS_HSV1_Final_2classes_SEG_sS1_strips_v4_pretrained_seed0.pth', type=str, help='pretrained model (e.g, Resnet50, Resnet50_withPretrainedWeight (These two from previous code), New: CBAM_Resnet50_withPretrainedWeight)')
parser.add_argument('--out_location', default='Feature_Heatmap/', type=str, help='output directory')
parser.add_argument('--sort_by', default='abs_activation', type=str, help='sort the individual maps by: e.g., abs_gradients, abs_activation, gradients, activation')
parser.add_argument('--selected_sp', default=['2016.09.22_CZ_03_217','2016.09.01_CZ_01_251','2016.09.22_CZ_02_239','2016.09.22_CZ_03_215', '2016.09.22_CZ_03_213', '2016.09.22_CZ_03_211', '2016.09.01_CZ_03_249', '2016.10.31_CZ_01_231'] ,type=str, nargs='*', help='list of sample IDs for the individual plots, plot for all test ids if None')
args = parser.parse_known_args()[0]
print(args)



##############################################################################################################################
##########################################      #Select input data                  ##########################################
##############################################################################################################################
#Test data path
test_img_dir = 'Image/test_' + args.input_data_name

#Vis method
vis_method = 'CAM'

###################################################################################################################
##########################################      #Model Weights                      ##########################################
##############################################################################################################################
# model_dict_hsv1 = {'Resnet50_withPretrainedWeight': 'CLS_HSV1_Final_2classes_SEG_sS1_strips_v4_pretrained_LucasSeed0.pth',
#                    'CBAM_Resnet50_withPretrainedWeight': 'cls_2025_02_20_14_18_19_100.pth'}
# model_dict_hsv2 = {'Resnet50_withPretrainedWeight': 'CLS_HSV2_Final_2classes_SEG_sS1_strips_v4_pretrained_LucasSeed0.pth',
#                    'CBAM_Resnet50_withPretrainedWeight': 'cls_2025_02_20_15_14_27_100.pth'}
#
# if args.HSV == 1:
#     model_name = model_dict_hsv1[args.pretrained_model]
# elif args.HSV == 2:
#     model_name = model_dict_hsv2[args.pretrained_model]

model_name = args.model

model_path = os.path.join('Model/', model_name)
# need to set visibility before defining device
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
### Check whether gpu is available
if torch.cuda.is_available():
    gpu = True
    device = torch.device('cuda')  # this will use the visible gpu
else:
    gpu = False
    device = torch.device('cpu')

# set seeds
fix_all_seeds_torch(args.gpu_id)

### Set Directory
save_path = os.path.join(args.out_location, args.model)
if not os.path.exists(save_path):
    os.makedirs(save_path)



########################################
########     Load test data     ########
########################################
test_ds = ValDataset_strips(img_dir= test_img_dir,
                         label_file=args.label_file,
                         HSV=args.HSV,
                         diagnostic_type=args.diagnostic_type,
                         num_classes=args.num_classes,
                         mask_dir=None if args.mask_dir == 'None' else args.mask_dir,
                         nchan = None # assuming we will supply strip images and mask_dir is None
                )
print(f"Number of samples: {len(test_ds)}")
test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
n_batches_test = len(test_loader)


########################################
########    Construct Model     ########
########################################
nchan=test_ds.nchan
print(f"nchan: {nchan}")

# Load model

# Lucas, could you review this block?
# if args.model contain CBAM, use cbam model
if 'CBAM' in args.model:
    model = ResNet50_CBAM(nchan = nchan, num_classes = args.num_classes, pretrained_model = "Resnet50_withPretrainedWeight", use_cbam_class = True, reduction_ratio = 1, kernel_cbam = 3, freeze = False)
    cbam_flag = True
else:
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.conv1 = nn.Conv2d(nchan, 64, kernel_size=7, stride=2, padding=3, bias=False) # Train1
    model.fc = nn.Linear(model.fc.in_features, args.num_classes)
    cbam_flag = False

# if args.pretrained_model == 'Resnet50':
#     model = models.resnet50()
#     model.conv1 = nn.Conv2d(nchan, 64, kernel_size=7, stride=2, padding=3, bias=False) # Train1
#     model.fc = nn.Linear(model.fc.in_features, args.num_classes)
#     cbam_flag = False
# elif args.pretrained_model == "Resnet50_withPretrainedWeight":
#     model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
#     model.conv1 = nn.Conv2d(nchan, 64, kernel_size=7, stride=2, padding=3, bias=False) # Train1
#     model.fc = nn.Linear(model.fc.in_features, args.num_classes)
#     cbam_flag = False
# elif args.pretrained_model == "CBAM_Resnet50_withPretrainedWeight":
#     model = ResNet50_CBAM(nchan = nchan, num_classes = args.num_classes, pretrained_model = "Resnet50_withPretrainedWeight", use_cbam_class = True, reduction_ratio = 1, kernel_cbam = 3, freeze = False)
#     cbam_flag = True
    
# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()

# Load the state dictionary into the model
# weights_only leads to an error when I run this line: TypeError: 'weights_only' is an invalid keyword argument for Unpickler()
model.load_state_dict(torch.load(model_path))#, weights_only = False))
model.to(device)
print('Model Loaded')



########################################
########    Predict             ########
########################################
loss_test, accuracy_test, correct_test, auc_test , all_ids_test, all_scores_test, all_labels_test = predict(test_loader,model, device, criterion)

#Load label file
label_df = pd.read_csv(args.label_file)
label_processor = LabelProcessor(label_df, args.diagnostic_type)
label_df = label_processor.process_labels()

#Pred
pred_df = get_prediction_df(all_ids_test, all_scores_test, all_labels_test, label_df, pred_thres = 0.5)

############################################################################################################
# Get All CAM, HEATMAP, GRADIENT, ACTIVATION, FEATURE MAP for all test IDs
############################################################################################################
cam_list = []
heatmap_list = []
gradient_list = []
scores_list = []
feature_map_list = []
pred_class_list = []

for strip_id in all_ids_test:
    img, label, sp_id = find_item(test_ds, strip_id)
    img = img.to(device)
    img_tensor = img.unsqueeze(0).to(device)

    cur_pred_df = pred_df.loc[pred_df['strip_id'] == strip_id]
    cur_pred_class = cur_pred_df['PRED_CLASS'].item()

    # Initialize
    cam_ranka = CAM_Manual(model=model, cam_method=vis_method, opposite_class=False, CBAM_FLAG=False)

    # Get cam: avg/sum overall all activation map; heatmap: resized cam, gradients gradients (alphas), and raw_scores (grad*feature map), feature_maps are the feature maps from CNN
    cam, heatmap, gradients, raw_scores, feature_maps = cam_ranka(img_tensor)

    cam_list.append(cam.unsqueeze(0).cpu().numpy())
    heatmap_list.append(np.expand_dims(heatmap, axis=0))
    gradient_list.append(gradients.cpu().numpy())
    scores_list.append(raw_scores.cpu().numpy())
    feature_map_list.append(feature_maps.cpu().numpy())
    pred_class_list.append(cur_pred_class)

cam_all = np.vstack(cam_list)
heatmap_all = np.vstack(heatmap_list)
gradient_all = np.vstack(gradient_list)
score_all = np.vstack(scores_list)
feature_map_all = np.vstack(feature_map_list)
pred_class_all = np.vstack(pred_class_list)

print("Gradient:", gradient_all.shape)
print("Scores:", score_all.shape)
print("CAM:", cam_all.shape)
print("Heatmap (resized CAM):", heatmap_all.shape)
print("Feature maps:", feature_map_all.shape)
print("Predicted Class:", pred_class_all.shape)
# Create image object for all test image
image_obj = AllImageInfo(gradient_all, score_all, cam_all, heatmap_all, feature_map_all, pred_class_all)

# sum over axis 2 and 3 of score_all to get the final score
out = score_all.sum(axis=(2, 3))
# transpose to make columns correspond to strip ids
out = np.transpose(out)
# save out to a csv file and use all_ids_test as column names
np.savetxt(os.path.join(save_path, 'score_by_feature_testimage.csv'), out, delimiter=',', header=','.join(all_ids_test),
           comments="")



############################################################################################################
# Visualization of gradients
############################################################################################################

#A. Top 10 Gradients for all strip id
file_path = os.path.join(save_path, "gradient_all_heatmap.png")
plot_gradient_heatmap(gradient_all, 10, pred_class_all, file_path, args.HSV, vis_method, sort_by_grad = True, sort_by_class = True)

#B. Histograms of Unique Gradient for strip id
grad_unique, idx_unique = np.unique(gradient_all, axis=0, return_index=True)
grad_class = pred_class_all[idx_unique].flatten().tolist()

file_path = os.path.join(save_path, "gradient_unique_hist.png")
plot_unique_gradient_histgram(grad_unique, grad_class, args.HSV, vis_method, file_path)

#C.TOP 10 (By Abs Gradeints) features for CLASS1 and CLASS0
top_f = 10
top_f_dict = {}
for i in range(2):
    top_f_idxes = np.argsort(abs(grad_unique), axis=1)[i][::-1].tolist()[0:top_f]
    top_f_dict[grad_class[i]] = top_f_idxes
print('Top 10 Feature indexes:',top_f_dict)

file_path = os.path.join(save_path, "TOP_Gradient_FeaturesIndex_byClass.txt")
with open(file_path, 'w') as f:
    f.write(str(top_f_dict))



scores_df = pd.read_csv(os.path.join(save_path, 'score_by_feature_testimage.csv'))
scores_overall = np.array(scores_df.abs().mean(axis = 1)) #average over all samples

feature_idxes_sorted = get_sorted_score_idxes(scores_overall, use_abs = True)
top_k = 7
top_oa_idxes = feature_idxes_sorted[0:top_k]
top_oa_scores = scores_overall[top_oa_idxes].round(3)
print('TOP feature overall:', top_oa_idxes)
print('TOP score overall:',top_oa_scores)


save_path2 = os.path.join(save_path, "per_feature_heatmaps")
if not os.path.exists(save_path2):
    os.makedirs(save_path2)
plot_function(all_ids_test,test_ds,image_obj,scores_df,top_oa_idxes,top_oa_scores,save_path2, device, all_ids_test, scores_overall, vis_method)
