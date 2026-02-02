#!/usr/bin/env python
# coding: utf-8
# This script does prediction on test data and heatmaps
import os, time, datetime
import torch
import torch.nn as nn
from sklearn.metrics import roc_curve, auc
from torch.utils.data import DataLoader
from torchvision import models
import sys
import torch.nn.functional as F # noqa
import torchvision
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pandas as pd
from pytorch_grad_cam.utils.image import deprocess_image 
import re
import argparse

# Add the repo root to the library path. No need to do this if in Ipython
# in_ipython = 'get_ipython' in globals()
# if not in_ipython:
try:
    repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
except NameError:
    repo_dir = os.getcwd()

print(repo_dir)
sys.path.append(repo_dir)
    
        
from Py_common.tv_utils import fix_all_seeds_torch
from Py_common.tv_Dataset_strips import TrainDataset_strips, ValDataset_strips
from Py_manus1.CBAM import  ResNet50_CBAM, predict, find_item, CAM_VIS, show_cam_on_image,tensor_to_rgb_image, plot_cam_heatmap_V2
from Py_manus1.CBAM import LabelProcessor, get_prediction_df, compute_performance
from Py_manus1.CBAM import get_6chan_image




#Run sbatch:
# ml Python/3.9.6-GCCcore-11.2.0
# ml IPython/7.26.0-GCCcore-11.2.0
# ml cuDNN/8.9.7.29-CUDA-12.3.0
# export CUBLAS_WORKSPACE_CONFIG=:4096:8
# source hsvw3/bin/activate
# python3 -u TV/tv_CLS_predict_CAM.py --model_name CLS_HSV1_Final_2classes_DET_dS_strips_nopretrained_seed0.pth

##############################################################################################################################
##########################################      Set arguments                       ##########################################
##############################################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--num_classes', default=2, type=int, help='Number of classes. Default: %(default)s')
parser.add_argument('--diagnostic_type', default='Final', type=str, help='Final, Majority, ...')
parser.add_argument('--label_file', default='Class_Label/gt/sS_labels.csv', type=str, help='folder directory containing labels file')
parser.add_argument('--mask_dir', default='None', type=str, help='if None, imgs are already cropped; else, folder directory containing training image and mask files. There can be unused mask files.')
parser.add_argument('--batch_size', default=24, type=int, help='batch size. Default: %(default)s')
parser.add_argument('--out_location', default='Feature_Heatmap', type=str, help='output directory')
parser.add_argument('--model_name',  default='CLS_HSV1_Final_2classes_SEG_sS1_strips_v4_pretrainedGAIN_LucasSeed0.pth', type=str, help='model name')

args = parser.parse_known_args()[0]
print(args)


##############################################################################################################################
##########    #Model and test img  path (Using args.)                               ##########################################
##############################################################################################################################
HSV = int(re.search(r'HSV(\d+)', args.model_name).group(1))
input_data_name = re.search(r'2classes_(.*?)_(?:no)?pretrained', args.model_name).group(1)
gpu_id = int(re.search(r'seed(\d+)', args.model_name, re.IGNORECASE).group(1))
model_path = os.path.join('Models/', args.model_name)
test_img_dir = 'Image/test_' + input_data_name



########################################
########     Set GPU            ########
########################################
# need to set visibility before defining device
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
### Check whether gpu is available
if torch.cuda.is_available():
    gpu = True
    device = torch.device('cuda')  # this will use the visible gpu
else:
    gpu = False
    device = torch.device('cpu')

# set seeds
fix_all_seeds_torch(gpu_id)

########################################
########     Set Directory      ########
########################################
outdir = os.path.join(args.out_location, args.model_name)
save_path = os.path.join(outdir, "overall")
if not os.path.exists(save_path):
    os.makedirs(save_path)

########################################
########     Load test data     ########
########################################
test_ds = ValDataset_strips(img_dir= test_img_dir,
                         label_file=args.label_file,
                         HSV=HSV,
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
nchan = test_ds.nchan
print(f"nchan: {nchan}")

# Load model
if 'CBAM' in args.model_name:
    model = ResNet50_CBAM(nchan = nchan, num_classes = args.num_classes, pretrained_model = "Resnet50_withPretrainedWeight", use_cbam_class = True, reduction_ratio = 1, kernel_cbam = 3, freeze = False)
    cbam_flag = True
elif 'GAIN' in args.model_name:
    gain_alpha = 1 #recommended in the paper
    model = GAIN(nchan = nchan,  num_classes = args.num_classes, hook_place = -2)
    cbam_flag = False
elif 'nopretrained' in args.model_name:
    model = models.resnet50()
    model.conv1 = nn.Conv2d(nchan, 64, kernel_size=7, stride=2, padding=3, bias=False) # Train1
    model.fc = nn.Linear(model.fc.in_features, args.num_classes)
    cbam_flag = False
else:
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.conv1 = nn.Conv2d(nchan, 64, kernel_size=7, stride=2, padding=3, bias=False) # Train1
    model.fc = nn.Linear(model.fc.in_features, args.num_classes)
    cbam_flag = False
    
# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()

# Load the state dictionary into the model
model.load_state_dict(torch.load(model_path))
model.to(device)
print('Model Loaded')



########################################
########    Predict             ########
########################################
loss_test, accuracy_test, correct_test, auc_test , all_ids_test, all_scores_test, all_labels_test = predict(test_loader,model, device, criterion, args.model_name)

#Load label file
label_df = pd.read_csv(args.label_file)
label_processor = LabelProcessor(label_df, args.diagnostic_type)
label_df = label_processor.process_labels()

#Pred df
pred_df = get_prediction_df(all_ids_test, all_scores_test, all_labels_test, label_df, pred_thres = 0.5)
pred_df.to_csv(outdir + '/pred.csv', index = True)

#Performance
pref_df = compute_performance(list(pred_df[args.diagnostic_type + 'HSV' + str(HSV)]),pred_df['PRED_PROB'],pred_df['PRED_CLASS'], input_data_name)
print(pref_df)
pref_df.to_csv(outdir + '/pref.csv', index = True)



########################################
########    Plot                ########
########################################
print('Plot Important Area...')
group_names = ['(iii) HSV-1 + , HSV-2 +', 
               '(i) HSV-1 + , HSV-2 -' , 
               '(ii) HSV-1 – , HSV-2 +',
               '(iv) HSV-1 – , HSV-2 -']
opposite_class  = False
for grp in group_names:

    #Get IDs
    selected_ids = list(pred_df.loc[pred_df['GROUP'] == grp, 'strip_id'])
    print('Plot for ' + grp + ": " + str(len(selected_ids)))
    
    #Create output folder
    if opposite_class == False:
        save_path2 = os.path.join(save_path, grp)
    else:
        save_path2 = os.path.join(save_path, 'opposite_yc' ,grp)
        
    if not os.path.exists(save_path2):
        os.makedirs(save_path2)
    save_path_mis = os.path.join(save_path2, 'Misclassified')

    #Plot
    for strip_id in selected_ids:
        img, label, sp_id  = find_item(test_ds, strip_id)
        img = img.to(device)
        
        #Get RGB img
        rgb_img = tensor_to_rgb_image(img) #Get RGB img for plot
        if nchan == 6:
            rgb_img = get_6chan_image(rgb_img[:,:,:3],rgb_img[:,:,3:])


        #Get predicted label and actual label
        cur_pred_df = pred_df.loc[pred_df['strip_id'] == strip_id]
        cur_pred_class = cur_pred_df['PRED_CLASS'].item()
        cur_pred_prob  = cur_pred_df['PRED_PROB'].item()
            
        ############################################################################################################
        #Plot for list of methods
        ############################################################################################################
        vis_methods_list = ['CAM','GradCAM', 'GradCAM++']
        cmap = 'RdYlBu_r'
        vis_list = []
        for vis_method in vis_methods_list:
            #Initialize 
            cam_vis = CAM_VIS(model, args.model_name, vis_method, device, opposite_class, cbam_flag)
            heatmap, gb = cam_vis(img)
            
            #Overlay heatmap to img
            vis = show_cam_on_image(rgb_img, heatmap, cmap = cmap, 
                                    use_rgb=False, 
                                    use_custom_cmap = True, 
                                    image_weight = 0.5, 
                                    cam_method = vis_method, 
                                    cam_direct = False)

            vis_list.append(vis)

        ############################################################################################################
        # Plot the results
        ############################################################################################################
        plot_name_list = ['Image'] + vis_methods_list
        plot_vis_list = [rgb_img] + vis_list

        if cur_pred_class != label:
            if not os.path.exists(save_path_mis):
                os.makedirs(save_path_mis)
            final_path = save_path_mis
        else:
            final_path = save_path2
        
        #map_title =  f'True Class: {cur_pred_class}, Predicted Probability: {cur_pred_prob:.2f}'
        map_title = grp
        plot_cam_heatmap_V2(plot_name_list, plot_vis_list, sp_id, label,
                            cur_pred_class, cur_pred_prob, 
                            final_path, cmap, map_title)