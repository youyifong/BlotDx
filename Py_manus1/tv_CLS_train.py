import os, time, datetime

# import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc
from torch.utils.data import DataLoader
from torchvision import models
import sys
import torch.nn.functional as F # noqa

# don't run this line in ipython. It allows us to run python TV/tv_CLS_train.py from the root directory on Linux
in_ipython = 'get_ipython' in globals()
if not in_ipython:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Py_common.tv_utils import fix_all_seeds_torch
from Py_common.tv_Dataset_strips import TrainDataset_strips, ValDataset_strips

### Set arguments
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--nchan', default='3', type=int, help='1 or 2')
parser.add_argument('--HSV', default='1', type=int, help='1 or 2')
parser.add_argument('--num_classes', default=2, type=int, help='Number of classes. Default: %(default)s')
parser.add_argument('--diagnostic_type', default='Final', type=str, help='Final, Majority, ...')

parser.add_argument('--train_img_dir', default='Image/CL_SEG_sS1_strips_v4', type=str, help='folder directory containing training image and mask files. There can be unused mask files.')
parser.add_argument('--val_img_dir', default='Image/validation_SEG_sS1_strips_v4', type=str, help='folder directory containing training image and mask files. There can be unused mask files.')
parser.add_argument('--label_file', default='Class_Label/gt/sS_labels.csv', type=str, help='folder directory containing labels file')
parser.add_argument('--mask_dir', default='None', type=str, help='if None, imgs are already cropped; else, folder directory containing training image and mask files. There can be unused mask files.')

parser.add_argument('--focal_loss', default='0', type=int, help='0 or 1')
parser.add_argument('--sharpening', default='0', type=int, help='0 or 1')
parser.add_argument('--normalize', default='1', type=int, help='0 or 1')

parser.add_argument('--batch_size', default=24, type=int, help='batch size. Default: %(default)s')
parser.add_argument('--n_epochs',default=100, type=int, help='number of epochs. Default: %(default)s')
parser.add_argument('--gpu_id', default=0, type=int, help='which gpu to use. Default: %(default)s')
parser.add_argument('--pretrained_model', default='IMAGENET1K_V2', type=str, help='pretrained model to use for starting training')
args = parser.parse_known_args()[0]
print(args)


# need to set visibility before defining device
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id % 3) # volta has only 3 gpus
### Check whether gpu is available
if torch.cuda.is_available():
    gpu = True
    device = torch.device('cuda')  # this will use the visible gpu
else:
    gpu = False
    device = torch.device('cpu')
# device = torch.device('cpu') # try this when cuda is out of memory

fix_all_seeds_torch(args.gpu_id)

### Set Directory
save_path = 'working_model'+str(args.gpu_id)
if not os.path.isdir(save_path):
    os.makedirs(save_path)


### Define train and validation dataset
train_ds = TrainDataset_strips(img_dir=args.train_img_dir,
                           label_file=args.label_file,
                           HSV=args.HSV,
                           diagnostic_type=args.diagnostic_type,
                           num_classes=args.num_classes,
                           mask_dir=None if args.mask_dir=='None' else args.mask_dir,
                           nchan=None, # assuming we will supply strip images and mask_dir is None
                           data_aug_ctrl=[False, args.sharpening==1], # permute_B_R, sharpening
                           reorder=True,
                           normalize=args.normalize==1
                           )
print(f"Number of training samples: {len(train_ds)}")
# train_ds[0]

train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False)
n_batches = len(train_loader)
# images, labels = next(iter(train_loader))


val_ds = ValDataset_strips  (img_dir=args.val_img_dir,
                         label_file=args.label_file,
                         HSV=args.HSV,
                         diagnostic_type=args.diagnostic_type,
                         num_classes=args.num_classes,
                         mask_dir=None if args.mask_dir == 'None' else args.mask_dir,
                         nchan = None, # assuming we will supply strip images and mask_dir is None
                         normalize=args.normalize==1
                    )
print(f"Number of validation samples: {len(val_ds)}")
# val_ds[0]

val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
n_batches_val = len(val_loader)
# images, labels, strip_ids = next(iter(val_loader))


# Load a pretrained ResNet50 model
if args.pretrained_model == 'None':
    model = models.resnet50()
else:
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

# Modify the first convolutional layer to accept nchan input channels
# nchan = train_ds[0][0].shape[0] # this triggers random number generator
nchan=train_ds.nchan
print(f"nchan: {nchan}")

model.conv1 = nn.Conv2d(nchan, 64, kernel_size=7, stride=2, padding=3, bias=False) # Train1
# try different kernel_size, stride, padding
# model.conv1 = nn.Conv2d(nchan, 64, kernel_size=3, stride=1, padding=1, bias=False) # Train2

# Modify the fully connected layer to match the number of classes
model.fc = nn.Linear(model.fc.in_features, args.num_classes)

# Move the model to the device (GPU or CPU)
model = model.to(device)

# save the model before training
# for this to work, args.train_img_dir can only have one /
# commented out b/c in general we don't need to save this model
# torch.save(model.state_dict(), f"Model/CLS_HSV{args.HSV}_{args.diagnostic_type}_{args.num_classes}classes_{args.train_img_dir.split('_', 1)[1]}_before_training.pth")



### Define loss function and optimizer


# Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        ce = F.cross_entropy(
            logits, targets, weight=self.alpha, reduction='none'
        )
        pt = torch.exp(-ce)
        loss = (1 - pt) ** self.gamma * ce

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

# set class weights
class_counts = torch.zeros(args.num_classes, dtype=torch.long)
for _, (images, labels, strip_ids) in enumerate(train_loader, 1):
    for c in range(args.num_classes):
        class_counts[c] += (labels == c).sum()
alpha = 1.0 / torch.sqrt(class_counts)
alpha = alpha / alpha.sum()   # optional normalization
print("Class counts:", class_counts.tolist(), "   alpha:", alpha)

if args.focal_loss==1:
    criterion = FocalLoss(
        gamma=2.0,
        alpha=alpha.to(device),
        reduction='mean'
    )
else:
    criterion = nn.CrossEntropyLoss()


### Training stage (training loop) start from here!
num_epochs = args.n_epochs
momentum = 0.9
learning_rate = 0.001
weight_decay = 0.0005
use_scheduler = False # scheduler
d = datetime.datetime.now()

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)


for epoch in range(1, num_epochs+1):
    time_start = time.time()

    model.train()  # needed if we switch to eval within each epoch

    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels, strip_ids) in enumerate(train_loader, 1):
        # images, labels = next(iter(train_loader))
        # noinspection PyUnresolvedReferences
        images, labels = images.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Calculate statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if batch_idx % 30 == 0:
            print(f"Batch {batch_idx}/{n_batches} Batch train loss: {loss.item():5.3f}")

    if use_scheduler:
        lr_scheduler.step()

    # Train losses
    train_loss = running_loss / n_batches
    accuracy = 100 * correct / total


    # Validation losses
    model.eval()
    # running_loss = 0.0
    correct_val = 0
    total_val = 0
    all_scores=[]
    all_labels=[]

    with torch.no_grad():
        for images, labels, strip_ids in val_loader:
            images, labels = images.to(device), labels.to(device)

            total_val += labels.size(0)

            outputs = model(images)

            loss = criterion(outputs, labels)
            # running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct_val += (predicted == labels).sum().item()

            probabilities = F.softmax(outputs, dim=1)
            all_scores.extend(probabilities[:,1].cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    # epoch_loss = running_loss / len(val_loader)
    accuracy_val = 100 * correct_val / total_val

    # compute AUC between scores and labels
    fpr, tpr, thresholds = roc_curve(all_labels, all_scores, pos_label=1)

    # # separate summary of all_scores for all_labels 1 and 0
    # np.quantile(np.array(all_scores)[np.array(all_labels)==1], [0.1, 0.25, 0.5, 0.75, 0.9])
    # np.quantile(np.array(all_scores)[np.array(all_labels)==0], [0.1, 0.25, 0.5, 0.75, 0.9])

    elapsed = time.time() - time_start
    prefix = f"[Epoch {epoch}/{num_epochs}]"
    print(f"{prefix} Train Accuracy: {correct}/{total}, Val Accuracy: {correct_val}/{total_val}, Val AUC: {auc(fpr, tpr):.4f}, [{elapsed:.0f} secs]")

    # Save the trained parameters every xx epochs
    if epoch%100 == 0:
        torch.save(model.state_dict(), os.path.join(save_path, 'cls' + d.strftime("_%Y_%m_%d_%H_%M_%S") + "_"+ str(epoch) + '.pth'))


