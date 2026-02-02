import argparse, os, time, datetime, random
import numpy as np
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# don't run this line in ipython. It allows us to run python TV/tv_CLS_train.py from the root directory on Linux
import sys
in_ipython = 'get_ipython' in globals()
if not in_ipython:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Py_common.tv_utils import fix_all_seeds_torch
from Py_common.tv_Dataset_sheets import TrainDataset_sheets


### Set arguments
parser = argparse.ArgumentParser()

parser.add_argument('--train_img_dir', default='Image/sheets/201907-202505_train', type=str, help='folder directory containing training images')
parser.add_argument('--mask_dir', default='Mask/201907-202505w/gt_dS_DET', type=str, help='folder directory containing mask images')
parser.add_argument('--label_file', default='Class_Label/gt/sS_labels.csv', type=str, help='folder directory containing labels file')

parser.add_argument('--num_classes', default=2, type=int, help='number of classes, 2 for foreground/background, 11 for HSV diagnostic')
parser.add_argument('--pretrained_model', required=False, default='Model/DET_dS.pth', type=str, help='pretrained model to use for starting training') 
parser.add_argument('--batch_size', default=4, type=int, help='batch size. Default: %(default)s')
parser.add_argument('--n_epochs',default=200, type=int, help='number of epochs. Default: %(default)s')
parser.add_argument('--min_size', default=800, type=int, help='minimum size of gt box to be considered for training. Default: %(default)s')
parser.add_argument('--box_detections_per_img', default=100, type=int, help='maximum number of detections per image, for all classes. Default: %(default)s')
parser.add_argument('--gpu_id', default=0, type=int, help='which gpu to use. Default: %(default)s')
args = parser.parse_known_args()[0]
print(args)


# need to set visibility before defining device
os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu_id)
### Check whether gpu is available
if torch.cuda.is_available() :
    gpu = True
    device = torch.device('cuda') # this will use the visible gpu
else :
    gpu = False
    device = torch.device('cpu')
#device = torch.device('cpu') # try this when cuda is out of memory


# set seeds
fix_all_seeds_torch(args.gpu_id)


# # get image dimension from a mask file
# img = imread(glob.glob(args.dir + "/*_mask.png")[0]) 
# # set min_size
# min_size = min(img.shape)

min_size=args.min_size

print("min_size: "+str(min_size))

### Set Directory
save_path = 'working_model'+str(args.gpu_id)
if not os.path.isdir(save_path):
    os.makedirs(save_path)


### Define train and test dataset
train_ds = TrainDataset_sheets(img_dir=args.train_img_dir,
                            mask_dir=args.mask_dir,
                            label_file=args.label_file,
                            data_aug_ctrl=[False, True] # permute_B_R, sharpening
                            )
# train_ds[0]

# to preserve dataloader reproducibility https://pytorch.org/docs/stable/notes/randomness.html
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
g = torch.Generator()
g.manual_seed(0)

# Define Dataloader
batch_size = args.batch_size
if gpu:
    train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=4, shuffle=False, worker_init_fn=seed_worker, generator=g, collate_fn=lambda x: tuple(zip(*x))) # on linux
    n_batches = len(train_dl)
else:
    train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=0, shuffle=False, worker_init_fn=seed_worker, generator=g, collate_fn=lambda x: tuple(zip(*x))) # on local
    n_batches = len(train_dl)

# images, targets = next(iter(train_dl))


# initial weight for training
# if args.pretrained_model is not coco but a previously trained model, it is loaded after model is created
if args.pretrained_model == 'coco':
    initial_weight = torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
else:
    initial_weight = None


def get_model():
    num_classes = args.num_classes
    
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
            weights=initial_weight,
            min_size = min_size, # IMAGE_MIN_DIM
            max_size = 10000, # IMAGE_MAX_DIM, set to a large number so that it has no impact

            box_score_thresh=0.7, # DETECTION_MIN_CONFIDENCE
            rpn_nms_thresh=0.9, # RPN_NMS_THRESHOLD
            
            # set all ROIS upper bound to args.box_detections_per_img
            rpn_pre_nms_top_n_train=args.box_detections_per_img, # RPN_NMS_ROIS_TRAINING
            rpn_pre_nms_top_n_test=args.box_detections_per_img, # RPN_NMS_ROIS_INFERENCE
            rpn_post_nms_top_n_train=args.box_detections_per_img, # RPN_NMS_ROIS_TRAINING
            rpn_post_nms_top_n_test=args.box_detections_per_img, # RPN_NMS_ROIS_INFERENCE
            rpn_batch_size_per_image=args.box_detections_per_img, # RPN_TRAIN_ANCHORS_PER_IMAGE
            box_batch_size_per_image=args.box_detections_per_img, # TRAIN_ROIS_PER_IMAGE
            box_detections_per_img=args.box_detections_per_img # DETECTION_MAX_INSTANCE            
    )
    
    # get the number of inpute features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

model = get_model() # get mask r-cnn
model.to(device)
#model.state_dict()

# Load pre-trained model 
if args.pretrained_model != 'coco' and args.pretrained_model != "None":
    model.load_state_dict(torch.load(args.pretrained_model, map_location=device))
    print("loading pretrained model")


### Declare which parameters are trained or not trained (freeze)
# Print parameters in mask r-cnn model 
#for name, param in model.named_parameters():
#    print("Name: ", name, "Requires_Grad:", param.requires_grad)


# If requires_grad = false, you are freezing the part of the model as no changes happen to its parameters. 
# All layers have the parameters modified during training as requires_grad is set to true.
for param in model.parameters(): 
    param.requires_grad = True


### Training stage (training loop) start from here!
model.train()
num_epochs = args.n_epochs
momentum = 0.9
learning_rate = 0.001
weight_decay = 0.0005
use_scheduler = False # scheduler
d = datetime.datetime.now()

wt_loss_classifier = .5 if args.num_classes==2 else 1


params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

for epoch in range(1, num_epochs+1):
    time_start = time.time()
    loss_accum = 0.0 # sum of total losses
    
    for batch_idx, (images, targets) in enumerate(train_dl, 1): # images, targets = next(iter(train_dl))
        # predict
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets] # k:key, v:value
        
        loss_dict = model(images, targets)
        # loss = sum(loss for loss in loss_dict.values()) # sum of losses
        loss = wt_loss_classifier * loss_dict['loss_classifier'] +loss_dict['loss_box_reg'] +loss_dict['loss_objectness'] +loss_dict['loss_rpn_box_reg'] 
        
        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # logging
        loss_accum += loss.item()
                
        if batch_idx % 30 == 0:
            print(f"Batch {batch_idx}/{n_batches} train loss: {loss.item():5.3f}")
    
    if use_scheduler:
        lr_scheduler.step()
    
    # Train losses
    train_loss = loss_accum / n_batches
    
    elapsed = time.time() - time_start
    
    # Print loss
    # if epoch==1 or epoch==5 or epoch%10==0:
    prefix = f"[Epoch {epoch}/{num_epochs}]"
    print(f"{prefix} Train loss: {train_loss:5.3f}, [{elapsed:.0f} secs]")

    # Save the trained parameters every xx epochs
    if epoch%20 == 0:
        torch.save(model.state_dict(), os.path.join(save_path, 'fasterrcnn_trained_model' + d.strftime("_%Y_%m_%d_%H_%M_%S") + "_"+ str(epoch) + '.pth'))
    

