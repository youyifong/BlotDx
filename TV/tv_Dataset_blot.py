import os, glob, torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

# from tsp import imread
import cv2
from TV.tv_utils import random_rotate_and_resize, normalize_img




### Dataset and DataLoader, one blot at a time
class TrainDataset_blot(Dataset):
    def __init__(self, img_dir, mask_dir, label_file, data_aug_ctrl=True, num_classes=2):

        # Load image and mask files, and sort them
        self.img_files = sorted(glob.glob(os.path.join(img_dir, '*.png')))
        self.num_classes = num_classes
        self.data_aug_ctrl = data_aug_ctrl
        if self.num_classes>2:
            self.class_df = pd.read_csv(os.path.join(label_file))

        self.mask_files=[]
        all_mask_files = sorted(glob.glob(os.path.join(mask_dir, '*_mask*.png')))
        for img_file in self.img_files:
            img_name = os.path.basename(img_file).replace("_img.png", ".png").replace(".png", "")
            mask_file = [f for f in all_mask_files if img_name in f]
            if len(mask_file) != 1:
                raise Exception(f"Error: {len(mask_file)} mask files found for {img_name}")
            else:
                self.mask_files.append(mask_file[0])

    def __getitem__(self, idx):
        # it is a key to performance that cv2.imread is used instead of tsp.imread b/c it leads to better contrast
        img = cv2.imread(self.img_files[idx], -1)
        img = normalize_img(img)

        mask = cv2.imread(self.mask_files[idx], -1)
        
        # make sure the first dimension is channel
        if img.shape[2] == min(img.shape):
            img = np.transpose(img, (2, 0, 1))

        # Transformation
        # leave xy, patch size, unspecified
        img_trans, mask_trans = random_rotate_and_resize (X=[img], Y=[mask],
                                                          scale_range=1, do_flip=True, do_rotate=True, permute_B_R=self.data_aug_ctrl)
        # if the patch does not have any gt mask, redo transformation
        while len(np.unique(mask_trans)) == 1: 
            img_trans, mask_trans = random_rotate_and_resize (X=[img], Y=[mask],
                                                          scale_range=1, do_flip=True, do_rotate=True, permute_B_R=self.data_aug_ctrl)

        obj_ids = np.unique(mask_trans) # get list of gt masks, e.g. [0,1,2,3,...]
        obj_ids = obj_ids[1:] # remove background 0
        num_objs = len(obj_ids)

        # Split a mask map into multiple binary mask map
        masks = mask_trans == obj_ids[:, None, None] # masks is an array of shape num_objs x height x width
        
        # Get labels
        if self.num_classes>2:
            # subset class_df to the current image
            subset_class_df = self.class_df[self.class_df['file_name']==os.path.basename(self.img_files[idx]).replace("_img.png",".png")]
            # extract class label based on mask_id
            # first, make a data frame of mask_id
            id1_df = pd.DataFrame(obj_ids, columns=['mask_id'])        
            # second, left-merge mask_id df and class_df
            mapped_df = id1_df.merge(subset_class_df, on='mask_id', how='left')        
            # last, get class_id
            labels = mapped_df['class_id'].tolist()
        else:
            labels = torch.ones((num_objs,), dtype=torch.int64) # all 1            
            
        # Get bounding box coordinates for each mask
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i]) # noqa
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            boxes.append([xmin, ymin, xmax, ymax])
        
        
        # Convert everything into a torch.Tensor
        img = torch.as_tensor(img_trans, dtype=torch.float32) # for image
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64) 
        masks = torch.as_tensor(masks, dtype=torch.uint8) # dtpye needs to be changed to uint16 or uint32
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) # calculating height*width for bounding boxes
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64) # suppose all instances are not crowd; if instances are crowded in an image, 1
        
        # Remove too small box (too small gt box makes an error in training)
        keep_box_idx = torch.where(area > 10) # args.min_box_size
        boxes = boxes[keep_box_idx]
        labels = labels[keep_box_idx]
        masks = masks[keep_box_idx]
        image_id = image_id
        area = area[keep_box_idx]
        iscrowd = iscrowd[keep_box_idx]
        
        # Required target for the Mask R-CNN
        target = {
                'boxes': boxes,
                'labels': labels,
                'masks': masks,
                'image_id': image_id,
                'area': area,
                'iscrowd': iscrowd
                }
        
        return img, target
    
    def __len__(self):
        return len(self.img_files)




### Dataset and DataLoader (prediction)
class TestDataset_blot(Dataset):
    def __init__(self, root):
        self.img_files = glob.glob(os.path.join(root, '*.png'))


    def __getitem__(self, idx):
        img_path = self.img_files[idx]

        img = cv2.imread(img_path, -1) # read as is
        
        # make sure the first dimension is channel
        if img.shape[2] == min(img.shape):
            img = np.transpose(img, (2, 0, 1))
        
        img = normalize_img(img) # normalize image

        # Convert image into tensor
        img = torch.as_tensor(img, dtype=torch.float32) # for image
        
        return {'image': img, 'image_id': idx, 'img_path': img_path}
    
    def __len__(self):
        return len(self.img_files)


