import torch
import torch.nn as nn
from torchvision import models
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix,fbeta_score,average_precision_score
from sklearn import metrics
import matplotlib as mpl

#https://github.com/jacobgil/pytorch-grad-cam
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, BinaryClassifierOutputTarget
#from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import deprocess_image, preprocess_image


class ResNet50_CBAM(nn.Module):
    def __init__(self, 
                 nchan = 3, 
                 num_classes = 2, 
                 pretrained_model = 'Resnet50_withPretrainedWeight', 
                 use_cbam_class = False,
                 reduction_ratio = 1,
                 kernel_cbam = 3, 
                 freeze = False):
        super(ResNet50_CBAM, self).__init__()
        
        #Get Parameters
        self.use_cbam_class = use_cbam_class

        # Load a pretrained model
        if pretrained_model == 'Resnet50':
            m = models.resnet50()
        elif pretrained_model == "Resnet50_withPretrainedWeight":
            m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        #Get feature dim for fc
        fc_dim = m.fc.in_features

        #Modify the first convolutional layer to accept nchan input channels
        m.conv1 = nn.Conv2d(nchan, 64, kernel_size=7, stride=2, padding=3, bias=False) # Train1
        
        #Remove classifaction layer from the original pretrained model
        self.features = nn.Sequential(*list(m.children())[:-2])

        #define self.cbam
        if self.use_cbam_class:
            self.cbam = CBAM(n_channels_in = fc_dim, reduction_ratio = reduction_ratio, kernel_size = kernel_cbam)

        #classification layer
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(fc_dim, num_classes)

        #Freeze the layers before the last layer
        if freeze:
            for name, param in self.m.named_parameters():
                if "fc" not in name:
                    param.requires_grad = False


    def forward(self, x):
        out = self.features(x)

        if self.use_cbam_class:
            out = out  + self.cbam(out)
        
        
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out 

class CBAM(nn.Module):

    def __init__(self, n_channels_in = 3, reduction_ratio = 2, kernel_size = 3):
        super(CBAM, self).__init__()
        self.n_channels_in = n_channels_in
        self.reduction_ratio = reduction_ratio
        self.kernel_size = kernel_size

        self.channel_attention = ChannelAttention(n_channels_in, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, f):
        chan_att = self.channel_attention(f)
        # print(chan_att.size())
        fp = chan_att * f
        # print(fp.size())
        spat_att = self.spatial_attention(fp)
        # print(spat_att.size())
        fpp = spat_att * fp
        # print(fpp.size())
        return fpp


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size):
        super(SpatialAttention, self).__init__()
        self.kernel_size = kernel_size

        assert kernel_size % 2 == 1, "Odd kernel size required"
        self.conv = nn.Conv2d(in_channels = 2, out_channels = 1, kernel_size = kernel_size, padding= int((kernel_size-1)/2))
        # batchnorm

    def forward(self, x):
        max_pool = self.agg_channel(x, "max")
        avg_pool = self.agg_channel(x, "avg")
        pool = torch.cat([max_pool, avg_pool], dim = 1)
        conv = self.conv(pool)
        # batchnorm ????????????????????????????????????????????
        conv = conv.repeat(1,x.size()[1],1,1)
        att = torch.sigmoid(conv)        
        return att

    def agg_channel(self, x, pool = "max"):
        b,c,h,w = x.size()
        x = x.view(b, c, h*w)
        x = x.permute(0,2,1)
        if pool == "max":
            x = F.max_pool1d(x,c)
        elif pool == "avg":
            x = F.avg_pool1d(x,c)
        x = x.permute(0,2,1)
        x = x.view(b,1,h,w)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, n_channels_in, reduction_ratio):
        super(ChannelAttention, self).__init__()
        self.n_channels_in = n_channels_in
        self.reduction_ratio = reduction_ratio
        self.middle_layer_size = int(self.n_channels_in/ float(self.reduction_ratio))

        self.bottleneck = nn.Sequential(
            nn.Linear(self.n_channels_in, self.middle_layer_size),
            nn.ReLU(),
            nn.Linear(self.middle_layer_size, self.n_channels_in)
        )


    def forward(self, x):
        kernel = (x.size()[2], x.size()[3])
        avg_pool = F.avg_pool2d(x, kernel )
        max_pool = F.max_pool2d(x, kernel)

        
        avg_pool = avg_pool.view(avg_pool.size()[0], -1)
        max_pool = max_pool.view(max_pool.size()[0], -1)
        

        avg_pool_bck = self.bottleneck(avg_pool)
        max_pool_bck = self.bottleneck(max_pool)

        pool_sum = avg_pool_bck + max_pool_bck

        sig_pool = torch.sigmoid(pool_sum)
        sig_pool = sig_pool.unsqueeze(2).unsqueeze(3)

        out = sig_pool.repeat(1,1,kernel[0], kernel[1])
        return out


###Prediction Ultility
def predict(data_loader, model, device, criterion):
    model.eval()
    running_loss = 0.0
    all_scores=[] 
    all_labels=[] 
    all_ids = []
    all_pred_labels= []

    with torch.no_grad():
        for images, labels, strip_ids in data_loader:
            images, labels = images.to(device), labels.to(device)


            #Predict
            outputs = model(images)
            _, predicted = torch.max(outputs, 1) #predict class
            probabilities = F.softmax(outputs, dim=1) #predict prob

            #Compute Loss
            loss = criterion(outputs, labels)
            running_loss += loss.item()

           
            all_scores.extend(probabilities[:,1].cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_ids.extend(strip_ids)
            all_pred_labels.extend(predicted.cpu().tolist())

    #compute epoch loss
    epoch_loss = running_loss / len(data_loader)

    #compute epoch accuracy
    correct = (torch.tensor(all_pred_labels) == torch.tensor(all_labels)).sum().item()
    total =len(all_labels)
    corrected_frac = f"{correct}/{total}"
    accuracy = 100 * correct / total

    # compute AUC between scores and labels
    fpr, tpr, thresholds = roc_curve(all_labels, all_scores, pos_label=1)
    auc_score = auc(fpr, tpr)

    return epoch_loss, accuracy, corrected_frac, auc_score, all_ids, all_scores, all_labels




#HEATMAP Utility
def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      n_chan: int = 3,
                      use_rgb: bool = False,
                      use_custom_cmap: bool = False,
                      cmap: str = 'RdYlBu_r',
                      colormap: int = cv2.COLORMAP_JET,
                      image_weight: float = 0.5,
                      cam_method: str = 'GradCAM') -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.

    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :param image_weight: The final result is image_weight * img + (1-image_weight) * mask.
    :returns: The default image with the cam overlay.
    """

    if cam_method != 'Guided GradCAM':
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap) #this coverts to BGR images
        if use_custom_cmap:
            heatmap = apply_custom_colormap(mask, cmap=cmap) 
        if use_rgb:
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap = np.float32(heatmap) / 255
    else:
        heatmap = mask

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    if image_weight < 0 or image_weight > 1:
        raise Exception(
            f"image_weight should be in the range [0, 1].\
                Got: {image_weight}")

    #This is updated from the original function: Apply the heatmap to each channel separately
    cam = np.zeros_like(img)
    for i in range(n_chan):
        cam[:, :, i] = (1 - image_weight) * heatmap[:, :, i % 3] + image_weight * img[:, :, i]
    #cam = (1 - image_weight) * heatmap + image_weight * img This is from original function
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)
        

def apply_custom_colormap(mask, cmap="RdYlBu_r"):
    
    # Scale to [0,255]
    mask = np.uint8(255 * mask)  

    # get colormap
    colormap = plt.get_cmap(cmap)
    heatmap = colormap(mask / 255.0)[:, :, :3]  # Get RGB channels

    # Scale to [0,255]
    heatmap = (heatmap * 255).astype(np.uint8)  
    
    return heatmap

def get_6chan_image(img1, img2):
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    
    img1 -= img1.min()
    img1 /= img1.max() + 1e-8  # prevent division by zero
    
    img2 -= img2.min()
    img2 /= img2.max() + 1e-8

    # overlay
    alpha = 0.5
    overlay = alpha * img1 + (1 - alpha) * img2
    overlay = np.clip(overlay, 0, 1)
    # Plot
    # plt.imshow(overlay)
    # plt.axis('off')
    # plt.show()
    return overlay

# def show_cam_on_image_v2(img: np.ndarray,
#                       mask: np.ndarray,
#                       n_chan: int = 3,
#                       cmap: str = 'jet',
#                       image_weight: float = 0.5,
#                       cam_method: str = 'GradCAM') -> np.ndarray:
#     """ This function overlays the cam mask on the image as an heatmap.
#     By default the heatmap is in BGR format.

#     :param img: The base image in RGB or BGR format.
#     :param mask: The cam mask.
#     :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
#     :param colormap: The OpenCV colormap to be used.
#     :param image_weight: The final result is image_weight * img + (1-image_weight) * mask.
#     :returns: The default image with the cam overlay.
#     """

#     if cam_method != 'Guided GradCAM':
#         heatmap = apply_custom_colormap(mask, cmap=cmap)  
#         heatmap = np.float32(heatmap) / 255
#     else:
#         heatmap = mask

#     #This is updated from the original function: Apply the heatmap to each channel separately
#     cam = np.zeros_like(img)
#     for i in range(n_chan):
#         cam[:, :, i] = (1 - image_weight) * heatmap[:, :, i % 3] + image_weight * img[:, :, i]
#     #cam = (1 - image_weight) * heatmap + image_weight * img This is from originnal function
#     cam = cam / np.max(cam)
#     return np.uint8(255 * cam), cmap

#Manual implementation
#Modified from https://medium.com/@stepanulyanin/implementing-grad-cam-in-pytorch-ea0937c31e82
class CAM_RESNET50_manual(nn.Module):
    def __init__(self, trained_model, CBAM_FLAG = False):
        super(CAM_RESNET50_manual, self).__init__()
        
        # get the pretrained model
        self.m = trained_model
        self.CBAM_FLAG = CBAM_FLAG

        if CBAM_FLAG == False:
            # get feature extraction layers
            self.features_conv = nn.Sequential(*list(self.m.children()))[:-2]
        elif CBAM_FLAG == True:
            # get feature extraction layers
            self.features_conv = self.m.features  

        # get the max pool 
        self.max_pool = nn.Sequential(*list(self.m.children()))[-2]
        
        # get the classifier 
        self.classifier = nn.Sequential(*list(self.m.children()))[-1]
        
        # placeholder for the gradients
        self.gradients = None
    
    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad
        
    def forward(self, x):
        x = self.features_conv(x)
        
        if self.CBAM_FLAG == True:
            x = x + self.m.cbam(x)  
            
        x = self.max_pool(x)
        # register the hook for classification layer
        x = x.view((1, -1))
        h = x.register_hook(self.activations_hook)
        x = self.classifier(x)
        
        return x
    
    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    # method for the feature extraction
    def get_feature_maps(self, x):
        return self.features_conv(x)


class GRADCAM_RESNET50_manual(nn.Module):
    def __init__(self, trained_model, CBAM_FLAG = False, hook_place = -2):
        # hook_place: -2 means the next layer is -2, which is max_pool
        # hook_place: -3 means the next layer is -3, which is layer4

        super(GRADCAM_RESNET50_manual, self).__init__()
        
        # get the pretrained model
        self.m = trained_model
        self.CBAM_FLAG = CBAM_FLAG
        self.hook_place = hook_place

        # get feature extraction layers
        if CBAM_FLAG == False:
            self.features_conv = nn.Sequential(*list(self.m.children()))[:hook_place]
        elif CBAM_FLAG == True:
            self.features_conv = self.m.features

        # get layer 1-4
        self.layer1 = nn.Sequential(*list(self.m.children()))[-6]
        self.layer2 = nn.Sequential(*list(self.m.children()))[-5]
        self.layer3 = nn.Sequential(*list(self.m.children()))[-4]
        self.layer4 = nn.Sequential(*list(self.m.children()))[-3]

        # get the max pool
        self.max_pool = nn.Sequential(*list(self.m.children()))[-2]
        
        # get the classifier
        self.classifier = nn.Sequential(*list(self.m.children()))[-1]

        
        # placeholder for the gradients
        self.gradients = None
    
    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad
        
    def forward(self, x):

        if self.CBAM_FLAG == True:
            x = self.features_conv(x)
            x = x + self.m.cbam(x)
            x.register_hook(self.activations_hook)  # register the hook for the last CNN layer
            x = self.max_pool(x)
            x = x.view((1, -1))
            x = self.classifier(x)

        else:
            x = self.features_conv(x)
            x.register_hook(self.activations_hook) # register the hook for the last CNN layer
            if self.hook_place == -6:
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)
            elif self.hook_place == -5:
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)
            elif self.hook_place == -4:
                x = self.layer3(x)
                x = self.layer4(x)
            elif self.hook_place == -3:
                x = self.layer4(x)
            elif self.hook_place == -2:
                pass
            x = self.max_pool(x)
            x = x.view((1, -1))
            x = self.classifier(x)


        return x
    
    def get_activations_gradient(self):
        return self.gradients
    
    def get_feature_maps(self, x):
        return self.features_conv(x)


class CAM_Manual(nn.Module):
    def __init__(self, model, cam_method, opposite_class = False, CBAM_FLAG = False):
        super(CAM_Manual,self).__init__()    
        self.cam_method = cam_method
        self.opposite_class = opposite_class 

        if cam_method == 'CAM':
            self.net = CAM_RESNET50_manual(model, CBAM_FLAG)
        elif cam_method == 'GradCAM Manual':
            self.net = GRADCAM_RESNET50_manual(model, CBAM_FLAG)
            
    def forward(self, input_tensor):
        #Predict
        self.net.eval()
        pred = self.net(input_tensor)

        if self.opposite_class == False:
            pred_class = pred.argmax().item()
        else:
            pred_class = pred.argmin().item()
        
        # get the gradient of pred_y wrt the feature
        pred[0,pred_class].backward()
    
        #pull the gradients out of the model
        gradients = self.net.get_activations_gradient()
        
        if self.cam_method == 'CAM':
            gradients = gradients.squeeze() #[2048) #2nd way: gradients =  self.net.classifier.weight.data[pred_class,] #[2048)
        elif self.cam_method == 'GradCAM Manual':
            gradients = torch.mean(gradients, dim=[0, 2, 3]) #[2048] # pool the gradients across the channels
        
        # get the features maps
        A_k = self.net.get_feature_maps(input_tensor).detach() #[[1, 2048, 14, 2])

        #Get activation weight the channels by corresponding gradients
        activations = torch.zeros(A_k.shape, dtype=torch.float32)
        for i in range(A_k.shape[1]):
            activations[:, i, :, :] = A_k[:, i, :, :]*gradients[i]
        
        
        if self.cam_method == 'CAM':
            cam = activations.sum(dim=1).squeeze()  #[14,2] #Sum of all chan
        elif self.cam_method == 'GradCAM Manual':
            cam = activations.mean(dim=1).squeeze() #[14,2] # average of all chan
            cam = F.relu(cam).cpu() 
            
        #Norm
        cam_norm = cam - cam.min()
        cam_norm = cam_norm/cam_norm.max()#[14,2]
        
        #Resize to image size
        heatmap = cv2.resize(cam_norm.cpu().numpy(), (input_tensor.shape[3], input_tensor.shape[2]))
        
        return cam_norm, heatmap, gradients, activations, A_k

        
class CAM_VIS(nn.Module):
    def __init__(self, model, pretrained_model_name, cam_method, device, opposite_class = False, CBAM_FLAG = False):
        super(CAM_VIS, self).__init__()
        self.model = model
        self.cam_method = cam_method
        self.device = device
        self.opposite_class = opposite_class
        
        if 'CBAM' in pretrained_model_name:
            target_layers = [model.features[-1][-1]]
        else:
            target_layers = [model.layer4[-1]]
            
        if cam_method == 'GradCAM' or cam_method == 'Guided GradCAM':
            self.cam = GradCAM(model=model, target_layers=target_layers)
        elif cam_method == 'HiResCAM':
            self.cam = HiResCAM(model=model, target_layers=target_layers)
        elif cam_method == 'ScoreCAM':
            self.cam = ScoreCAM(model=model, target_layers=target_layers)  
        elif cam_method == 'GradCAM++':
            self.cam = GradCAMPlusPlus(model=model, target_layers=target_layers)    
        elif cam_method == 'AblationCAM':
            self.cam = AblationCAM(model=model, target_layers=target_layers)
        elif cam_method == 'XGradCAM':
            self.cam = XGradCAM(model=model, target_layers=target_layers)  
        elif cam_method == 'EigenCAM':
            self.cam = EigenCAM(model=model, target_layers=target_layers)
        elif cam_method == 'FullGrad':
            self.cam = FullGrad(model=model, target_layers=target_layers)
        elif cam_method == 'CAM' or cam_method == 'GradCAM Manual':
            self.cam = CAM_Manual(model=model, cam_method = cam_method, CBAM_FLAG = CBAM_FLAG)

    def forward(self, tensor_img):
        
        #Change Input size
        input_tensor = tensor_img.unsqueeze(0).to(self.device)  

        if self.cam_method == 'CAM' or self.cam_method  == 'GradCAM Manual':
            cam, heatmap, gradients, activations, feature_maps = self.cam(input_tensor)
        else:
            #Get pred class
            pred = self.model(input_tensor)

            if self.opposite_class == False:
                pred_class = pred.argmax().item()
            else:
                pred_class = pred.argmin().item()
                
            #Convert for CAM categories
            targets = [ClassifierOutputTarget(pred_class)]
            # Generate the CAM
            #aug_smooth=True: This applies test-time augmentation, which involves performing multiple transformations on the input image (like horizontal flips and scaling) and averaging the results. This helps to better center the CAM around the objects in the image1.
            #eigen_smooth=True: This uses the first principal component of the activations weighted by their gradients. It effectively removes a lot of noise from the CAM, resulting in cleaner visual explanations
            grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets, aug_smooth=False, eigen_smooth=False)
            heatmap = grayscale_cam[0, :]  # Get the CAM for the first image in the batch
    
            if self.cam_method == 'Guided GradCAM':
                gb_model = GuidedBackpropReLUModel(model=self.model, device=self.device)
                gb = gb_model(input_tensor, target_category=pred_class) #this returns the gradient wrt to the input image
                cam_mask = cv2.merge([heatmap] * gb.shape[2])
                heatmap = cam_mask * gb #deprocess_image(cam_mask * gb)

        return heatmap



def tensor_to_rgb_image(tensor_img):
    
    if tensor_img.is_cuda:
        tensor_img = tensor_img.cpu()
    
    # Convert the tensor to numpy array
    np_array = tensor_img.numpy()
    
    # Normalize the values to be in range [0, 1]
    np_array = (np_array - np.min(np_array)) / (np.max(np_array) - np.min(np_array))
    
    # Transpose the array to match the shape (height, width, channels)
    np_array = np.transpose(np_array, (1, 2, 0))
    
    return np_array



def plot_cam_heatmap_V2(plot_name_list, plot_vis_list, sp_id, true_label,pred_class, pred_prob, plot_dir, cmap):
    plt.figure(figsize=(10, 5))
    rows = 1
    cols = (len(plot_vis_list) + rows - 1) // rows 
    
    #Norm becuase the first image has diff scale
    all_min = min([img.min() for img in plot_vis_list])
    all_max = max([img.max() for img in plot_vis_list])

    im_handles = []
    for i in range(len(plot_vis_list)):
        plt.subplot(rows, cols, i + 1)
        im = plt.imshow(plot_vis_list[i], cmap=cmap, vmin=all_min, vmax=all_max)
        im_handles.append(im)
        plt.title(plot_name_list[i], fontsize = 8)
        plt.axis('off')

    # Adjust space to avoid overlap with colorbar
    plt.subplots_adjust(bottom=0.20)

    # Add common horizontal colorbar at the bottom
    cbar_ax = plt.gcf().add_axes([0.25, 0.12, 0.5, 0.025])  # [left, bottom, width, height]
    #plt.colorbar(im_handles[0], cax=cbar_ax, orientation='horizontal')
    cbar = plt.colorbar(im_handles[0], cax=cbar_ax, orientation='horizontal')
    cbar.set_ticks(np.linspace(0, 255, 6))
    cbar.set_ticklabels([f"{t:.1f}" for t in np.linspace(0, 1, 6)])  # Format labels from 0 to 1

    map_title =  'True Class: ' + str(true_label) + ', Predicted Probability: ' + f'{pred_prob:.2f}'
    #map_title =  sp_id + '\n Predicted Probability: ' + f'{pred_prob:.2f}' +  ', True Class: ' + str(true_label)
    plt.suptitle(map_title, fontsize=10)  
    #plt.show()
    
    outname = "ACTUAL" + f'{true_label}' + "_PRED" + f'{pred_class}' + "_PROB" + f'{pred_prob:.2f}'
    file_path = os.path.join(plot_dir, f"{sp_id}_{outname}.png")
    plt.savefig(file_path, dpi=300, bbox_inches='tight', pad_inches=0.2, facecolor='white')
    

    plt.close()

def normalize_to_minus_one_to_one(arr):
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    
    # Avoid division by zero if all values are the same
    if arr_max == arr_min:
        return np.zeros_like(arr)  # Return all zeros if there's no variation

    normalized_arr = 2 * (arr - arr_min) / (arr_max - arr_min) - 1
    return normalized_arr
    
def plot_individual_maps(plot_array, scores, scores_overall, feature_idxes, rgb_image, top_k, full_cam, file_path, map_title, 
                         vis_method = 'CAM', pos_first = True, cmap = 'RdYlBu_r', fig_size = (20,12)):
    
    #Norm for acroos all maps
    #plot_array = min_max_normalize(plot_array)
    plot_array = plot_array - plot_array.min() 
    plot_array = plot_array/plot_array.max()
    
    vis_list = []
    f_idx_list = []
    score_list = []
    score_overall_list = []
    for i in range(top_k):
        cur_score = scores[i]
        cur_score_overall = scores_overall[i]
        
        #resize
        heatmap = plot_array[i,]
        heatmap = cv2.resize(heatmap, (rgb_image.shape[1], rgb_image.shape[0]))
        
        #Overlay heatmap to img
        vis = show_cam_on_image(rgb_image, heatmap, cmap = cmap, use_rgb=False, use_custom_cmap = True, image_weight = 0.5, cam_method = vis_method)

        vis_list.append(vis)
        f_idx_list.append(feature_idxes[i])
        score_list.append(cur_score)
        score_overall_list.append(cur_score_overall)

    #Full cam
    full_cam_img = normalize_to_minus_one_to_one(full_cam)
    full_cam_img = show_cam_on_image(rgb_image, full_cam, cmap = cmap, use_rgb=False, use_custom_cmap = True, image_weight = 0.5, cam_method = vis_method)

    # Plot the heatmaps
    vis_list = [rgb_image, full_cam_img] + vis_list
    f_idx_list = ['-1', '-1'] + f_idx_list
    score_list = ['-1', '-1'] + score_list
    score_overall_list = ['-1', '-1'] + score_overall_list

    #PLOT
    plt.figure(figsize=fig_size)
    rows = 1
    cols = (len(vis_list) + rows - 1) // rows 
    # Create a list to store all axes
    axes = []
    for i in range(len(vis_list)):
        ax = plt.subplot(rows, cols, i + 1)
        im = ax.imshow(vis_list[i],cmap=cmap)
        axes.append(ax)

        if i == 0:
            plt.title('Image \n',fontsize=15)
        elif i == 1:
            plt.title('CAM \n',fontsize=15)
        else:
            plt.title(f"F{f_idx_list[i]} \nOA:{score_overall_list[i]:.2f} \nSP:{score_list[i]:.2f} ", fontsize=15)
        plt.axis('off')

    #Normalize colormap since the first image use differnt scale 
    norm = mpl.colors.Normalize(vmin=0, vmax=255)  # 
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Empty array, as we only need the colormap
    
    # Add a shared color bar at the bottom
    cbar = plt.colorbar(sm, ax=axes, orientation='horizontal', fraction=0.046, pad=0.04, extend='neither')
    tick_positions = np.linspace(0, 255, 6)
    tick_labels = np.linspace(-1, 1, 6)
    cbar.set_ticks(tick_positions)
    cbar.set_ticklabels([f"{t:.1f}" for t in tick_labels])  # Format labels from -1 to 1


    plt.suptitle(map_title, fontsize=17) 
    plt.savefig(file_path, dpi=300, bbox_inches='tight', pad_inches=0)
    #plt.show()
    plt.close()
    
def plot_gradient_heatmap(plot_m, num_f, pred_class, file_path, HSV, vis_method, sort_by_grad = True, sort_by_class = True):
    
    #sort by predicted class
    if sort_by_class:
        idx_sorted = np.argsort(pred_class, axis=0).flatten()
        plot_m = plot_m[idx_sorted,:]

    #sort by gradient from high to low
    if sort_by_grad:
        plot_m = np.sort(plot_m, axis=1)[:, ::-1] 

    #plot
    plt.figure(figsize=(20, 10))
    plt.imshow(plot_m[:,0:num_f], aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title("Heatmap of the Gradient (alpha_ks) - HSV" + str(HSV) + " Model" + "\nMethod: " + vis_method)
    plt.xlabel('Features')
    plt.ylabel('strip_id')
    plt.savefig(file_path, dpi=300, bbox_inches='tight', pad_inches=0)
    #plt.show()
    plt.close()


def plot_unique_gradient_histgram(unique_gradients, gradient_class, HSV, vis_method,file_path):
    color_dict = {0: 'darkblue', 1: 'darkred'}
    plt.figure(figsize=(12, 6))
    plt.hist(unique_gradients[0], bins=50, alpha=0.5, label="Predicted Class " + str(gradient_class[0]),  color= color_dict[gradient_class[0]], edgecolor='black') 
    plt.hist(unique_gradients[1], bins=50, alpha=0.5, label="Predicted Class " + str(gradient_class[1]) , color= color_dict[gradient_class[1]], edgecolor='black') 
    
    plt.xlabel("Gradient")
    plt.ylabel("Frequency")
    plt.title("Histogram of Unique Gradients - HSV" + str(HSV) + " Model" + "\nMethod: " + vis_method)
    plt.xlim(-0.35, 0.35)  
    plt.ylim(0, 350)  
    plt.legend()
    #plt.show()
    plt.savefig(file_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()


def order_informations(grad, act_maps, f_maps, f_map_avg, f_and_grad, sorted_by = 'abs_gradients'):

    #sort by abs scores
    if sorted_by == 'abs_gradients':
        f_idx = get_sorted_feature_idx_abs(grad)
    elif sorted_by == 'abs_activation': #equal to product of grad and avg featue
        f_idx = get_sorted_feature_idx_abs(f_and_grad)
    elif sorted_by == 'gradients': #equal to product of grad and avg featue
        f_idx = get_sorted_feature_idx(grad)
    elif sorted_by == 'activation': #equal to product of grad and avg featue
        f_idx = get_sorted_feature_idx(f_and_grad)

    #Updated act_maps and grad
    grad = grad[f_idx,]
    act_maps = act_maps[f_idx,:,:]
    f_maps = f_maps[f_idx,:,:]
    f_map_avg = f_map_avg[f_idx,]
    f_and_grad = f_and_grad[f_idx,]

    return grad, act_maps , f_maps, f_map_avg, f_and_grad, f_idx

def get_sorted_feature_idx_abs(scores):
    f_idx_sorted = np.argsort(abs(scores), axis=0)[::-1]
    return f_idx_sorted.tolist()

def get_sorted_feature_idx(scores):
    f_idx_sorted = np.argsort(scores, axis=0)[::-1]
    return f_idx_sorted.tolist()

def get_topk_rankedby_overall_and_specific(grad, act_maps, f_maps, f_map_avg, f_and_grad, overall_f_and_grad, top_k_overall = 9, top_k_sp = 9):

    #overall sorting
    f_idx_overall = get_sorted_feature_idx_abs(overall_f_and_grad)
    
    #overall top 
    f_idx_overall_top = f_idx_overall[0:top_k_overall]
    
    #specific sorting
    f_idx_spec = get_sorted_feature_idx_abs(f_and_grad)

    #Specific top
    f_idx_spec_top = [f for f in f_idx_spec if f not in f_idx_overall_top][0:top_k_sp] #Get the top 10 not in f_indx1 (overall top10)

    #top overall and top spec
    f_idx_top = f_idx_overall_top + f_idx_spec_top

    #Other idx not in top
    f_idx_other = [f for f in f_idx_overall if f not in f_idx_top]

    #all idx ordered
    f_idx = f_idx_top + f_idx_other
    
    #Updated act_maps and grad
    grad = grad[f_idx,]
    act_maps = act_maps[f_idx,:,:]
    f_maps = f_maps[f_idx,:,:]
    f_map_avg = f_map_avg[f_idx,]
    f_and_grad = f_and_grad[f_idx,]
    overall_f_and_grad = overall_f_and_grad[f_idx,]

    return grad, act_maps , f_maps, f_map_avg, f_and_grad, overall_f_and_grad, f_idx_overall_top, f_idx_spec_top, f_idx
    
def select_info(grad, act_maps, f_maps, f_map_avg, f_and_grad, activation_type = "ALL"):
    
    if activation_type != "ALL":
        act_sum = np.sum(act_maps, axis=(1,2))  #Sum of each activation map
        if activation_type == "POS":
            keep_idx = np.where(act_sum > 0)[0].tolist()
        elif activation_type == "NEG":
            keep_idx = np.where(act_sum < 0)[0].tolist()
        elif activation_type == "EXC0":
            keep_idx = np.where(act_sum != 0)[0].tolist()
            
        #Updated act_maps and grad
        grad = grad[keep_idx,]
        act_maps = act_maps[keep_idx,:,:]
        f_maps = f_maps[keep_idx,:,:]
        f_map_avg = f_map_avg[keep_idx,]
        f_and_grad = f_and_grad[keep_idx,]
    
    return grad, act_maps, f_maps, f_map_avg, f_and_grad


def min_max_normalize(matrix):
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    normalized_matrix = (matrix - min_val) / (max_val - min_val)
    return normalized_matrix

def get_sorted_idxes(lst):
    pos = sorted((x for x in enumerate(lst) if x[1] >= 0), key=lambda x: -x[1])
    neg = sorted((x for x in enumerate(lst) if x[1] < 0), key=lambda x: x[1])
    sorted_indices = [x[0] for x in pos + neg]
    return sorted_indices
    
    
def sort_list_by_idxes(lst, idxes):
    return [lst[i] for i in idxes]



class LabelProcessor:
    def __init__(self, df, diagnostic_type):
        self.df = df.copy()
        self.label_col1 = diagnostic_type + 'HSV1'
        self.label_col2 = diagnostic_type + 'HSV2'

    def convert_label(self, value):
        return 1 if value == 'POSITIVE' else 0
        
    def process_labels(self):

        #Selecte columns
        self.df =  self.df[['strip_id', self.label_col1 , self.label_col2]]

        # Convert postive to 1, others to 0
        self.df[self.label_col1] = self.df[self.label_col1].apply(self.convert_label)
        self.df[self.label_col2] = self.df[self.label_col2].apply(self.convert_label)

        #Add group labels
        self.df['GROUP'] = pd.NA
        self.df.loc[(self.df[self.label_col1] == 0 )& (self.df[self.label_col2] == 0), 'GROUP'] = 'HSV1NEG_HSV2NEG' 
        self.df.loc[(self.df[self.label_col1] == 0 )& (self.df[self.label_col2] == 1), 'GROUP'] = 'HSV1NEG_HSV2POS' 
        self.df.loc[(self.df[self.label_col1] == 1 )& (self.df[self.label_col2] == 0), 'GROUP'] = 'HSV1POS_HSV2NEG' 
        self.df.loc[(self.df[self.label_col1] == 1 )& (self.df[self.label_col2] == 1), 'GROUP'] = 'HSV1POS_HSV2POS'
        
        return self.df


def get_prediction_df(selected_ids, pred_prob, y_true, label_data, pred_thres = 0.5):

    #Get prediction df
    prediction_df = pd.DataFrame({"strip_id": selected_ids, "PRED_PROB": pred_prob , "Label": y_true})
    prediction_df['PRED_CLASS'] = 0
    cond= prediction_df['PRED_PROB'] >= pred_thres
    prediction_df.loc[cond,'PRED_CLASS'] = 1
    
    #Combine label
    prediction_df = prediction_df.merge(label_data, on = ['strip_id'], how = "left")

    return prediction_df
    

def compute_performance(y_true,y_pred_prob,y_pred_class, cohort_name):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_class).ravel() #CM

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred_prob, pos_label=1)
    
    # Average precision score = PR-AUC
    PR_AUC = round(average_precision_score(y_true, y_pred_prob),2)

    AUC = round(metrics.auc(fpr, tpr),2)
    ACC = round(accuracy_score(y_true, y_pred_class),2)
    F1 = round(f1_score(y_true, y_pred_class),2)
    Recall = round(recall_score(y_true, y_pred_class),2)
    Precision = round(precision_score(y_true, y_pred_class),2)
    Specificity = round(tn / (tn + fp),2)
    perf_tb = pd.DataFrame({"ROCAUC": AUC, 
                            "PR_AUC":PR_AUC,
                            "ACC": ACC,
                            "F1": F1,
                            "Recall": Recall,
                            "Precision":Precision,
                            "Specificity":Specificity},index = [cohort_name])
    
    return perf_tb

from torch.utils.data import Dataset
class AllImageInfo(Dataset):
    def __init__(self, gradients, scores, cams, resized_cams, feature_maps, pred_classes):
        
        self.gradients = gradients
        self.feature_maps = feature_maps
        self.scores = scores
        self.cams = cams
        self.resized_cams = resized_cams
        self.pred_classes = pred_classes


    def __len__(self):
        return len(self.gradient_all)

    def __getitem__(self, idx):
        grad = self.gradients[idx,]            #[2048,]        #This is the gradient wrt each feature
        sc = self.scores[idx,]                 #(2048, 14, 2)  #This is grad*feature_map
        cam = self.cams[idx,]                  #[14,2]         $this is equal to normed sc.sum(axis = 0) 
        cam_resized = self.resized_cams[idx,]  #[420,46]       #this is resized_cam for plot
        fm = self.feature_maps[idx,]           #(2048, 14, 2)  #this is the feature maps
        pc = self.pred_classes[idx,].item()                    #this is the predicted class
        
        return grad, sc, cam, cam_resized, fm, pc


def get_sorted_score_idxes(in_scores, use_abs = True):

    if use_abs == True:
        sorted_idxes = get_sorted_feature_idx_abs(in_scores) 
    else:
        sorted_idxes = get_sorted_feature_idx(in_scores) 

    return sorted_idxes


def normalize_to_minus_one_to_one(arr):
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    
    # Avoid division by zero if all values are the same
    if arr_max == arr_min:
        return np.zeros_like(arr)  # Return all zeros if there's no variation

    normalized_arr = 2 * (arr - arr_min) / (arr_max - arr_min) - 1
    return normalized_arr

def symlog_transform(matrix):
    return np.sign(matrix) * np.log1p(np.abs(matrix))  # Keeps sign while applying log

# Function to find the item
def find_item(dataset, target):
    for item in dataset:
        if item[2] == target:
            return item
    return None

def plot_function(selected_ids, test_ds, image_obj, scores_df, top_oa_idxes, top_oa_scores, save_loc, device, all_ids_test, scores_overall, vis_method):
    for selected_sp in selected_ids:
        # get image
        img, label, sp_id = find_item(test_ds, selected_sp)
        img = img.to(device)

        # Get RGB img
        rgb_img = tensor_to_rgb_image(img)  # Get RGB img for plot

        # Get patient indx
        pt_idx = all_ids_test.index(selected_sp)

        # Get all info
        cur_grad, cur_score_map, cur_cam_norm_ori, cur_cam_resized, cur_featuremaps, cur_pred_class = image_obj[
            pt_idx]  # NOTE: cur_cam_norm =  normed version of cur_score_map.sum(axis = 0)

        # Get CAM before normalization
        cur_cam = cur_score_map.sum(axis=0)

        # Get current image specific scores for each feature
        cur_score_sum = np.array(scores_df[selected_sp])  # this is equal to cur_score_map.sum(axis = (1,2))

        # Print overall
        #         print('TOP feature overall:', top_oa_idxes)
        #         print('TOP score overall:',top_oa_scores)

        # Get top K specific feature
        cur_idxes_sorted = get_sorted_score_idxes(cur_score_sum, use_abs=True)
        top_k = 7
        top_sp_idxes = cur_idxes_sorted[0:top_k]
        top_sp_scores = cur_score_sum[top_sp_idxes].round(2)
        #         print('TOP feature specific:', top_sp_idxes)
        #         print('TOP score specific:',top_sp_scores)

        # Get Top specific, but not in overall
        top_sp_idxes2 = [f for f in cur_idxes_sorted if f not in top_oa_idxes][0:top_k]
        top_sp_scores2 = cur_score_sum[top_sp_idxes2].round(2)
        #         print('TOP feature specific:2', top_sp_idxes2)
        #         print('TOP score specific2:',top_sp_scores2)

        rgb_image = rgb_img
        fig_size = (30, 16)
        cmap = 'RdYlBu_r'

        # All
        cur_cam2 = cur_cam / cur_cam.max()

        const = cur_score_map.max()
        cur_score_map2 = [x / const for x in cur_score_map]

        # All
        combs = np.append([cur_cam2], cur_score_map2, axis=0)

        # Norm for acroos all maps
        combs = combs - combs.min()
        combs = combs / combs.max()

        # select
        scores = list(cur_score_sum[top_oa_idxes + top_sp_idxes2])
        selected_scores_overall = list(scores_overall[top_oa_idxes + top_sp_idxes2])

        # Modify index
        top_oa_idxes_modified = [x + 1 for x in top_oa_idxes]
        top_sp_idxes_modified = [x + 1 for x in top_sp_idxes2]
        feature_idxes = [0] + top_oa_idxes_modified + top_sp_idxes_modified
        plot_array = combs[feature_idxes]

        vis_list = []
        f_idx_list = []
        score_list = []
        score_overall_list = []
        for i in range(len(feature_idxes)):

            # resize
            if i > 0:
                cur_score = scores[i - 1]
                cur_score_overall = selected_scores_overall[i - 1]
            else:
                cur_score = None
                cur_score_overall = None
            heatmap = plot_array[i,]
            heatmap = cv2.resize(heatmap, (rgb_image.shape[1], rgb_image.shape[0]))

            # Overlay heatmap to img
            vis = show_cam_on_image(rgb_image, heatmap, cmap=cmap, use_rgb=False, use_custom_cmap=True,
                                    image_weight=0.5, cam_method=vis_method)

            vis_list.append(vis)
            f_idx_list.append(feature_idxes[i])
            score_list.append(cur_score)
            score_overall_list.append(cur_score_overall)

        # Plot the heatmaps
        vis_list = [rgb_image] + vis_list
        f_idx_list = ['-1'] + f_idx_list
        score_list = ['-1'] + score_list
        score_overall_list = ['-1'] + score_overall_list

        fig, axes = plt.subplots(1, len(vis_list), figsize=fig_size)

        for i, ax in enumerate(axes):
            im = ax.imshow(vis_list[i], cmap=cmap)
            ax.set_xticks([])  # Remove x ticks
            ax.set_yticks([])  # Remove y ticks

            if i == 0:
                ax.set_title('Image \n', fontsize=15)
            elif i == 1:
                ax.set_title('CAM \n', fontsize=15)
            else:
                ax.set_title(
                    f"F{int(f_idx_list[i]) - 1} \nOA:{float(score_overall_list[i]):.2f} \nSP:{float(score_list[i]):.2f} ",
                    fontsize=15)

        # Add a vertical dashed line
        fig.subplots_adjust(wspace=0.1)

        # Get the x position of the 11th and 12th subplot
        x_left = axes[int(len(vis_list) / 2)].get_position().x1
        x_right = axes[int(len(vis_list) / 2 + 1)].get_position().x0
        x_middle = (x_left + x_right) / 2  # Middle of the white space

        # Correct way to add a vertical dashed line
        line = mpl.lines.Line2D([x_middle, x_middle], [0.18, 0.9], transform=fig.transFigure,
                                color='black', linestyle='dashed', linewidth=2)
        fig.lines.append(line)  # Add the line to the figure

        # Normalize colormap since the first image use differnt scale
        norm = mpl.colors.Normalize(vmin=0, vmax=255)  #
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # Empty array, as we only need the colormap

        # Add a shared color bar at the bottom
        cbar = plt.colorbar(sm, ax=axes, orientation='horizontal', fraction=0.046, pad=0.04, extend='neither')
        tick_positions = np.linspace(0, 255, 6)
        tick_labels = np.linspace(-1, 1, 6)
        cbar.set_ticks(tick_positions)
        cbar.set_ticklabels([f"{t:.1f}" for t in tick_labels])  # Format labels from -1 to 1

        # save_path4 = os.path.join(save_loc, 'Top9OAandTOP9SP')
        save_path4 = save_loc
        if not os.path.exists(save_path4):
            os.makedirs(save_path4)

        if cur_pred_class != label:
            save_path_mis = os.path.join(save_path4, 'Misclassified')

            if not os.path.exists(save_path_mis):
                os.makedirs(save_path_mis)
            final_path = save_path_mis
        else:
            final_path = save_path4

        plt_title = selected_sp + '\n Predicted Class ' + str(cur_pred_class) + ' True Class: ' + str(label)
        map_title = 'Score Map\n' + plt_title
        plt.suptitle(map_title, fontsize=17)

        plt.savefig(os.path.join(final_path, selected_sp + "_Score_Map.png"), dpi=300, bbox_inches='tight',
                    pad_inches=0, facecolor='white')
        plt.show()
        plt.close()
