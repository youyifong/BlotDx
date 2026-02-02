import argparse, os, cv2
import numpy as np
from pathlib import Path
 

import torch, torchvision
from torchvision.models.detection import faster_rcnn

from tsp import imread, imsave
# from tsp.AP import csi, average_dice, masks_to_outlines
# noinspection PyTypeChecker
# from tsp.AP import mask2outline


# don't run this line in ipython. It allows us to run python TV/tv_CLS_train.py from the root directory on Linux
import sys
in_ipython = 'get_ipython' in globals()
if not in_ipython:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Py_common.tv_Dataset_sheets import TestDataset_sheets
from Py_common.tv_utils import crop_with_overlap

from PIL import Image, ImageDraw

parser = argparse.ArgumentParser()

parser.add_argument('--input', default="Image/validation", type=str, help='folder directory containing test images')
parser.add_argument('--num_classes', default=2, type=int, help='number of classes, 2 for foreground/background, 11 for HSV diagnostic')
parser.add_argument('--the_model', required=False, default='Model/DET_dS.pth', type=str, help='pretrained model to use for prediction')
parser.add_argument('--param_set', default="set1", type=str, help='used as a suffix after model name to indicate place to save prediction results, e.g. Image/sheets/201907-202505w/DET_dS_set1')

parser.add_argument('--box_score_threshold', default=0.5, type=float, help='minimum score threshold, 0.8 for pair of strips. Default: %(default)s')
parser.add_argument('--rpn_nms_threshold', default=0.7, type=float, help='NMS threshold. Default: %(default)s')
parser.add_argument('--box_size_threshold_L', default=5000, type=float, help='minimum box size threshold. 1000 for pair of strips. Default: %(default)s')
parser.add_argument('--box_size_threshold_U', default=100000, type=float, help='maximum box size threshold. 1000 for pair of strips. Default: %(default)s')
parser.add_argument('--box_detections_per_img', default=100, type=int, help='maximum number of detections per image, for all classes. Default: %(default)s')

parser.add_argument('--gt_masks_dir', default="", type=str, help='folder directory containing gt masks')
parser.add_argument('--verbose', default=1, type=int, help='verbose')

args = parser.parse_known_args()[0]
print(args)

verbose = args.verbose



os.environ["CUDA_VISIBLE_DEVICES"]="0"
### this has to be done after visible device is set
if torch.cuda.is_available() :
    gpu = True
    device = torch.device('cuda')
else :
    gpu = False
    device = torch.device('cpu')


save_dir = os.path.join(args.input, Path(args.the_model).stem + "_" + args.param_set)
os.makedirs(save_dir, exist_ok=True)

test_ds = TestDataset_sheets(root=args.input)
# test_ds.imgs
# test_ds[0]

box_detections_per_img = args.box_detections_per_img # default is 100, but 539 is used in a reference

def get_model():
    num_classes = args.num_classes
    
    the_model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
            rpn_nms_thresh=args.rpn_nms_threshold, # RPN_NMS_THRESHOLD (for inference)
            box_detections_per_img=args.box_detections_per_img # DETECTION_MAX_INSTANCE            
    )

    # get the number of input features for the classifier
    in_features = the_model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    the_model.roi_heads.box_predictor = faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    
    
    return the_model

model = get_model() # get mask r-cnn
model.to(device)


### Load pre-trained model
model.load_state_dict(torch.load(args.the_model, map_location=device))
# print(model.state_dict())

# 2025/12/19 to remove overlapping boxes
model.roi_heads.nms_thresh = 0.25


### Prediction
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

def get_color(value):
    # Dictionary mapping values from 1 to 7 to distinct BGR colors
    color_map = {
        1: (0, 0, 255),     # +/-, Red (BGR format)
        2: (0, 255, 0),     # -/+, Green
        3: (255, 0, 0),     # +/+, Blue
        4: (128, 128, 128), # -/-, Gray
        5: (0, 255, 255),   # RPT, Yellow
        6: (0, 165, 255),   # R40, Orange
        7: (255, 255, 0)    # rest, IND-related, Cyan
    }

    # Ensure the value is between 1 and 10
    if value < 1 or value > 10:
        raise ValueError("Value should be between 1 and 10.")
    
    # Return a distinct BGR color for values 1 to 7, and use the same color as 7 for 8 to 10
    return color_map.get(value, color_map[7])



# noinspection PyTypeChecker
for idx, sample in enumerate(test_ds):
    # sample = next(iter(test_ds))
    
    img = sample['image']
    image_id = sample['image_id']
    img_path = sample['img_path']
    if verbose: print(f"Processing {img_path}")

    shape=img.shape
    # if verbose: print(f"image shape: {shape}")
    nrows, ncols = int(np.ceil(shape[-2] / AUTOSIZE_MAX_SIZE)), int(np.ceil(shape[-1] / AUTOSIZE_MAX_SIZE))
    # if verbose: print(f"nrow: {nrows}, ncol: {ncols}")
    crops = crop_with_overlap(img, OVERLAP, nrows, ncols)
    boxes_ls = []

    # there is actually only one row, one column
    crop = crops[0]    
    # if verbose: print(f"crop shape: {crop.shape}")
    with torch.no_grad():
        result1 = model([crop.to(device)])[0] # result1 is a dict: 'boxes', 'labels', 'scores', 'masks'

    # result_masks = result1['masks'].cpu().numpy()
    result_boxes = result1['boxes'].cpu().numpy()
    result_scores = result1['scores'].cpu().numpy()
    result_labels = result1['labels'].cpu().numpy()
    if verbose: print(f"number of boxes: {result_boxes.shape[0]}")
    if result_boxes.shape[0] == 0:
        # throw error
        raise ValueError("No instances detected in the image.")
                            
    # save masks
    mask_image = Image.new('L', (1936, 1288), 0)    
    draw = ImageDraw.Draw(mask_image)

    overlay_img = imread(img_path)
    overlay_img = overlay_img.copy() # make it writable, not sure why, but this seems to be necessary for polylines to work

    mask_id=255
    i=0
    for box in result_boxes:
        # Extract polygon points
        x_points = [box[0], box[0], box[2], box[2]]
        y_points = [box[1], box[3], box[3], box[1]]
        points = list(zip(x_points, y_points))
        
        size = abs((box[0]-box[2]) * (box[1]-box[3]))

        asp_ratio = abs((box[1]-box[3]) / (box[0]-box[2]))
    
        if result_scores[i]>args.box_score_threshold and args.box_size_threshold_L < size < args.box_size_threshold_U and 4 <= asp_ratio <= 12:
        # if True:

            draw.polygon(points, outline=mask_id, fill=mask_id) 

            # for overlay_img

            # Extract polygon points
            points = np.array(
                [[box[0], box[1]],
                        [box[0], box[3]],
                        [box[2], box[3]],
                        [box[2], box[1]]], np.int32)
            
            # Reshape points for polylines function
            points = points.reshape((-1, 1, 2))
            
            # Draw the polygon (thickness = 2, color = white)
            cv2.polylines(overlay_img, [points], isClosed=True, color=get_color(result_labels[i]), thickness=2)

            mask_id=mask_id-1        
        i=i+1        
    if verbose: print(f"number of boxes passing: {255-mask_id}")
    
    saveto = save_dir + '/' + os.path.splitext(os.path.basename(img_path))[0] + '_tv_masks.png'
    mask_image.save(saveto)
    
    imsave(saveto.replace('_tv_masks.png','_tv_output.png'), overlay_img)

