import os, csv
from pathlib import Path
 
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc

import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F # noqa
from torch.utils.data import DataLoader

# don't run this line in ipython. It allows us to run python TV/tv_CLS_train.py from the root directory on Linux
import sys
in_ipython = 'get_ipython' in globals()
if not in_ipython:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Py_common.tv_utils import fix_all_seeds_torch
from Py_common.tv_Dataset_strips import ValDataset_strips # TestDataset_strips

### Set arguments
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--HSV', default='1', type=int, help='1 or 2')
parser.add_argument('--num_classes', default=2, type=int, help='Number of classes. Default: %(default)s')
parser.add_argument('--diagnostic_type', default='Final', type=str, help='Final, Majority, ...')
parser.add_argument('--the_model', required=False, default='Model/CLS_HSV1_Final_2classes_dS_3chan_DET_dS_strips_seed0.pth,Model/CLS_HSV1_Final_2classes_dS_3chan_DET_dS_strips_seed1.pth,Model/CLS_HSV1_Final_2classes_dS_3chan_DET_dS_strips_seed2.pth', type=str, help='pretrained model to use for prediction')

parser.add_argument('--test_img_dir', default='Image/validation_DET_dS_strips', type=str, help='folder directory containing training test images')
parser.add_argument('--mask_dir', default='None', type=str, help='an alternative way to provide test images. folder directory containing mask files.')
parser.add_argument('--label_file', default='Class_Label/gt/sS_labels.csv', type=str, help='folder directory containing labels file')

parser.add_argument('--normalize', default='1', type=int, help='0 or 1')

parser.add_argument('--batch_size', default=24, type=int, help='batch size. Default: %(default)s')
parser.add_argument('--gpu_id', default=0, type=int, help='which gpu to use. Default: %(default)s')
parser.add_argument('--save_to', default="", help='file name for saving results')

args = parser.parse_known_args()[0]
print(args)
print("")

html_header="""<!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Image Viewer</title>
        <style>
            .image-row { 
                display: flex; 
                flex-wrap: wrap;       /* Allows images to wrap to next line if many */
                flex-direction: row; 
                gap: 5px; 
                font-family: sans-serif; 
            }
            .image-item {
                flex: 0 0 auto;        /* Prevents stretching */
                width: 50px;           /* Adjusted from 50px to be slightly more visible */
            }
            img { 
                display: block; 
                width: 100%; 
                height: auto; 
                border: 1px solid #ccc; 
            }
        </style>
    </head>
    <body>        
    """

os.makedirs(args.save_to, exist_ok=True)


# need to set visibility before defining device
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
### Check whether gpu is available
if torch.cuda.is_available():
    gpu = True
    device = torch.device('cuda')  # this will use the visible gpu
else:
    gpu = False
    device = torch.device('cpu')
# device = torch.device('cpu') # try this when cuda is out of memory

# set seeds
fix_all_seeds_torch(args.gpu_id)


test_ds = ValDataset_strips (img_dir=args.test_img_dir,
                         label_file=args.label_file,
                         HSV=args.HSV,
                         diagnostic_type=args.diagnostic_type,
                         num_classes=args.num_classes,
                         mask_dir=None if args.mask_dir == 'None' else args.mask_dir,
                         nchan = None, # assuming we will supply strip images and mask_dir is None
                         normalize=args.normalize
                )
print(f"Number of samples: {len(test_ds)}")

nchan = test_ds[0][0].shape[0]

# turn args.the_model into a list
args.the_model = args.the_model.split(',')
all_prob_ls = []
all_predicted_ls = []
all_labels = [] # defined to remove a warning about variable being out of scope after the loop
all_strip_ids = [] # defined to remove a warning about variable being out of scope after the loop

for the_model in args.the_model:

    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    n_batches = len(test_loader)

    model = models.resnet50()

    model.conv1 = nn.Conv2d(nchan, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # model.conv1 = nn.Conv2d(nchan, 64, kernel_size=3, stride=1, padding=1, bias=False) # change default kernel size
    # Modify the fully connected layer to match the number of classes
    model.fc = nn.Linear(model.fc.in_features, args.num_classes)

    # Move the model to the device (GPU or CPU)
    model = model.to(device)
    # the model
    model.load_state_dict(torch.load(the_model, map_location=device))
    # print(model.state_dict())


    # Validation losses
    model.eval()

    all_strip_ids = []
    all_labels = []
    all_prob = []
    all_predicted = []
    misclassified_strips = pd.DataFrame(columns=['strip_id', 'label', 'predicted', 'probability'])

    with torch.no_grad():
        for images, labels, strip_ids in test_loader:
            # images = next(iter(test_loader))
            images, labels = images.to(device), labels.to(device)

            images = images.to(device)
            outputs = model(images)

            _, predicted = torch.max(outputs, 1)
            all_predicted.extend(predicted.cpu().tolist())
            # correct_val += (predicted == labels).sum().item()

            probabilities = F.softmax(outputs, dim=1)
            all_prob.extend(probabilities[:, 1].cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_strip_ids.extend(strip_ids)

            for i in range(len(labels)):
                if predicted[i] != labels[i]:
                    misclassified_strips.loc[len(misclassified_strips)] = [
                        strip_ids[i],
                        labels[i].item(),
                        predicted[i].item(),
                        probabilities[i, 1].item()
                    ]

    print(f"Model: {the_model}")

    fpr, tpr, thresholds = roc_curve(all_labels, all_prob, pos_label=1)
    correct = sum(1 for x, y in zip(all_predicted, all_labels) if x == y)
    print(f"Accuracy: {correct}/{len(all_predicted)}, AUC: {auc(fpr, tpr):.4f}")

    # print(f"Misclassified strips:\n {misclassified_strips}")

    all_prob_ls.append(all_prob)
    all_predicted_ls.append(all_predicted)

    # save all_prob to a file
    full_path = os.path.join(args.save_to, Path(the_model).stem.removeprefix("CLS_")+".csv")
    header = ["PredictedProb", "GroundTruth", "StripID"]
    # make sure the directory exists
    with open(full_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for row in zip(all_prob, all_labels, all_strip_ids):
            writer.writerow(row)

    print("")


if len(args.the_model) > 1:

    print("Working on ensemble results")

    # find the most common prediction
    from collections import Counter
    all_predicted_ar = np.array(all_predicted_ls).T
    final_predicted = []
    for i in range(len(all_predicted_ar)):
        final_predicted.append(Counter(all_predicted_ar[i]).most_common(1)[0][0])
    final_predicted = np.array(final_predicted)

    # save predictions to CSV

    # take the common part from args.the_model, remove CLS_ and _seed
    filename = os.path.commonprefix([Path(t).stem.removeprefix("CLS_") for t in args.the_model])
    filename = filename.removesuffix("_seed") 
    full_path = os.path.join(args.save_to, filename + ".csv")
    header = ["FinalPredicted", "GroundTruth", "StripID"]
    with open(full_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for row in zip(final_predicted, all_labels, all_strip_ids):
            writer.writerow(row)

    # write misclassified sample ids to files

    misclassified = [z for x, y, z in zip(final_predicted, all_labels, all_strip_ids) if x != y]
    print(f"Accuracy: {len(all_labels)-len(misclassified)}/{len(all_labels)}\n")

    en_false_pos = [z for x, y, z in zip(final_predicted, all_labels, all_strip_ids) if x == 1 and y == 0]
    en_false_neg = [z for x, y, z in zip(final_predicted, all_labels, all_strip_ids) if x == 0 and y == 1]
    en_true_pos = [z for x, y, z in zip(final_predicted, all_labels, all_strip_ids) if x == 1 and y == 1]
    en_true_neg = [z for x, y, z in zip(final_predicted, all_labels, all_strip_ids) if x == 0 and y == 0]

    img_base_path = "../../../" + args.test_img_dir

    html_filename = os.path.join(args.save_to, f"{filename}_miss.html")
    with open(html_filename, "w") as f:
        # Write the Header and Styles
        f.write(html_header)
        f.write(f'{filename} <br><br><br>')

        f.write(f'False positives:<br> {"<br> ".join(map(str, en_false_pos))} \n <div class="image-row">')
        for item in en_false_pos:
            # Assuming item is the filename without .png, e.g., '2021.10.28_ASCL_01_235'
            f.write(f'        <div class="image-item">\n')
            f.write(f'            <img src="{img_base_path}/{item}.png" title="{item}">\n')
            f.write(f'        </div>\n')
        f.write("</div><br><br><br>")

        f.write(f'False negatives:<br> {"<br> ".join(map(str, en_false_neg))} \n <div class="image-row">')
        for item in en_false_neg:
            # Assuming item is the filename without .png, e.g., '2021.10.28_ASCL_01_235'
            f.write(f'        <div class="image-item">\n')
            f.write(f'            <img src="{img_base_path}/{item}.png" title="{item}">\n')
            f.write(f'        </div>\n')
        f.write("</div>")

        f.write("</body></html>")


    html_filename = os.path.join(args.save_to, f"{filename}_tp.html")
    with open(html_filename, "w") as f:
        # Write the Header and Styles
        f.write(html_header)

        f.write('True positives \n <div class="image-row">')
        for item in en_true_pos:
            # Assuming item is the filename without .png, e.g., '2021.10.28_ASCL_01_235'
            f.write(f'        <div class="image-item">\n')
            f.write(f'            <img src="{img_base_path}/{item}.png" title="{item}">\n')
            f.write(f'        </div>\n')
        f.write("</div>")

        f.write("</body></html>")


    html_filename = os.path.join(args.save_to, f"{filename}_tn.html")
    with open(html_filename, "w") as f:
        # Write the Header and Styles
        f.write(html_header)

        f.write('True negatives \n <div class="image-row">')
        for item in en_true_neg:
            # Assuming item is the filename without .png, e.g., '2021.10.28_ASCL_01_235'
            f.write(f'        <div class="image-item">\n')
            f.write(f'            <img src="{img_base_path}/{item}.png" title="{item}">\n')
            f.write(f'        </div>\n')
        f.write("</div>")

        f.write("</body></html>")

    # with open(os.path.join(args.save_to, f"{filename}_false_pos.txt"), "w") as f:
    #     for item in en_false_pos:
    #         f.write(f"{item}\n")

    


