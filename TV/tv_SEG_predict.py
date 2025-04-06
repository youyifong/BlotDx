import argparse, os, sys
import numpy as np
# import cv2 # needed for polylines

import torch, torchvision
from torchvision.models.detection import faster_rcnn
from torchvision.models.detection import mask_rcnn


from tsp import imread, imsave
from tsp.AP import masks_to_outlines
# from tsp.AP import csi, average_dice

# don't run this line in ipython. It allows us to run python TV/tv_CLS_train.py from the root directory on Linux
in_ipython = 'get_ipython' in globals()
if not in_ipython:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TV.tv_Dataset_blot import TestDataset_blot
from TV.tv_utils import crop_with_overlap


verbose = False

parser = argparse.ArgumentParser()

parser.add_argument('--the_model', required=False, default='Model/SEG_sS.pth', type=str, help='pretrained model to use for prediction')
parser.add_argument('--dir', default="Image/validation", type=str, help='folder directory containing test images')
parser.add_argument('--num_classes', default=2, type=int, help='number of classes, 2 for foreground/background, 11 for HSV diagnostic')
parser.add_argument('--save_dir', default="TV/predSEGsS_val", type=str, help='folder directory containing prediction results')

parser.add_argument('--box_score_threshold', default=0.5, type=float, help='minimum score threshold, confidence score or each prediction. Default: %(default)s')
parser.add_argument('--mask_threshold', default=0.5, type=float, help='mask threshold, the predicted masks for each instance, in 0-1 range. In order to obtain the final segmentation masks, the soft masks can be thresholded, generally with a value of 0.5 (mask >= 0.5). Default: %(default)s')
parser.add_argument('--box_size_threshold', default=1000, type=float, help='minimum box size threshold. 1000 for pair of strips. Default: %(default)s')
parser.add_argument('--rpn_nms_threshold', default=0.7, type=float, help='NMS threshold. Default: %(default)s')
parser.add_argument('--box_detections_per_img', default=100, type=int, help='maximum number of detections per image, for all classes. Default: %(default)s')
parser.add_argument('--gpu_id', default=1, type=int, help='which gpu to use. Default: %(default)s')

args = parser.parse_known_args()[0]
# print(args)

os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu_id)
### this has to be done after visible device is set
if torch.cuda.is_available() :
    gpu = True
    device = torch.device('cuda')
else :
    gpu = False
    device = torch.device('cpu')

os.makedirs(args.save_dir, exist_ok=True)


test_ds = TestDataset_blot(root=args.dir)
# test_ds.imgs
# test_ds[0]


box_detections_per_img = args.box_detections_per_img # default is 100, but 539 is used in a reference


def get_model():
    num_classes = args.num_classes
    
    model1 = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(
            rpn_nms_thresh=args.rpn_nms_threshold, # RPN_NMS_THRESHOLD (for inference)
            box_detections_per_img=args.box_detections_per_img # DETECTION_MAX_INSTANCE            
    )

    # get the number of impute features for the classifier
    in_features = model1.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model1.roi_heads.box_predictor = faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    
    # get the number of input features for the mask classifier
    in_features_mask = model1.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # replace the mask predictor with a new one
    model1.roi_heads.mask_predictor = mask_rcnn.MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes) # a value is changed from 91 to 2
    
    return model1



model = get_model() # get mask r-cnn
model.to(device)
model.load_state_dict(torch.load(args.the_model, map_location=device))

model.eval()
AP_arr=[]
ARI_arr=[]
DICE_arr=[]

OVERLAP = 80
THRESHOLD = 2
# 1936x1288
AUTOSIZE_MAX_SIZE=1936 # 5x1 .35
# AUTOSIZE_MAX_SIZE=300 # 4x1, .36
# AUTOSIZE_MAX_SIZE=500 # 3x1
# AUTOSIZE_MAX_SIZE=1000 # 2x1
# AUTOSIZE_MAX_SIZE=2000 # 1x1, .20


def mask_iou(mask1, mask2):
    mask1 = mask1[0]
    mask2 = mask2[0]
    intersection = (mask1 & mask2).sum()
    union = (mask1 | mask2).sum()
    return intersection.float() / union.float()

# Assume masks is a binary tensor [N, 1, H, W] and scores [N]
def filter_overlapping_masks(masks_to_filter, scores, iou_threshold=0.5):
    res = []
    indices = scores.argsort(descending=True)

    while indices.numel() > 0:
        current_idx = indices[0].item()
        res.append(current_idx)

        current_mask = masks_to_filter[current_idx]
        remaining_masks = masks_to_filter[indices[1:]]

        # Compute IoU with remaining masks
        IoUs = torch.tensor([mask_iou(current_mask, mask) for mask in remaining_masks]) # noqa

        # Select only masks with IoU <= threshold
        indices = indices[1:][IoUs <= iou_threshold]

    return torch.tensor(res)

# noinspection PyTypeChecker
for idx, sample in enumerate(test_ds):
    # sample = next(iter(test_ds))
    
    img = sample['image']
    image_id = sample['image_id']
    img_path = sample['img_path']
    print(f"Processing {img_path}")
    
    shape=img.shape
    if verbose: print(f"image shape: {shape}")
    nrows, ncols = int(np.ceil(shape[-2] / AUTOSIZE_MAX_SIZE)), int(np.ceil(shape[-1] / AUTOSIZE_MAX_SIZE))
    if verbose: print(f"nrow: {nrows}, ncol: {ncols}")
    crops = crop_with_overlap(img, OVERLAP, nrows, ncols)
    boxes_ls = []

    # there is actually only one row, one column
    crop = crops[0]    
    if verbose: print(f"crop shape: {crop.shape}")
    with torch.no_grad():
        result1 = model([crop.to(device)])[0] # result1 is a dict: 'boxes', 'labels', 'scores', 'masks'
    # print (result1['boxes'].shape) # N x 4
    # print (result1['masks'].shape) # N x 1 x H x W
    # print (result1['scores'].shape) # N

    # remove redundant boxes not instances
    # keep_indices = torchvision.ops.nms(result1['boxes'], result1['scores'], iou_threshold=0.3)

    # remove redundant masks
    # noinspection PyTypeChecker
    keep_indices = filter_overlapping_masks(result1['masks'] > args.mask_threshold,
                                            result1['scores'],
                                            iou_threshold=0.5)

    result_masks = result1['masks'][keep_indices].cpu().numpy()
    result_boxes = result1['boxes'][keep_indices].cpu().numpy()
    result_scores = result1['scores'][keep_indices].cpu().numpy()
    if verbose: print(f"crop number of instances: {result_boxes.shape[0]}")
    if result_boxes.shape[0] == 0:
        print("no boxes found")
        raise Exception("no boxes found")


    # make a 2D array masks
    height_test, width_test = result_masks[0].shape[1:]
    masks = np.zeros((height_test, width_test), dtype='int16')
    boxes = np.zeros((height_test, width_test), dtype='int16')

    pts_ls=[]
    for val, ind_mask_map in enumerate(result_masks):
        if result_scores[val] < args.box_score_threshold:
            continue

        box = result_boxes[val]

        size = abs((box[0] - box[2]) * (box[1] - box[3]))

        if size < args.box_size_threshold:
            continue

        # pixels inside a mask is assigned 255-val
        binary_mask = ind_mask_map[0,:,:] > args.mask_threshold
        # if two masks are overlapped, remove the overlapped 
        # binary_mask = remove_overlapping_pixels(binary_mask, previous_masks) 
        masks[np.where(binary_mask)] = 255-val

        # Extract polygon points
        x_points = [box[0], box[0], box[2], box[2]]
        y_points = [box[1], box[3], box[3], box[1]]
        pts=np.column_stack((x_points, y_points))
        pts_ls.append (pts.astype(np.int32))
    
    
    # file name
    saveto = args.save_dir + '/' + os.path.basename(img_path).replace(".png", '')
    
    # Save masks
    imsave(saveto+'_tv_masks.png', masks)
    
    # Save masks outline
    # mask2outline(saveto+'_tv_masks.png')
    
    # Save an output image with masks and boxes overlaid on top of image
    overlay_img = imread(img_path) # read as is
    overlay_img = overlay_img.copy() # make it writable, not sure why, but this seems to be necessary for polylines to work
    # add mask outline in green
    outline = masks_to_outlines(masks)
    overlay_img[outline,0]=0
    overlay_img[outline,1]=255
    overlay_img[outline,2]=0
    # add box outline in red
    # cv2.polylines(overlay_img, pts_ls, isClosed=True, color=(255, 0, 0), thickness=1)

    imsave(saveto+'_tv_output.png', overlay_img)

