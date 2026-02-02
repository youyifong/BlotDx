import os, glob, torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path

# from tsp import imread
import cv2
from Py_common.tv_utils import random_rotate_and_resize_sheet, normalize_img




### Dataset and DataLoader, one photo at a time, for segmentation and detection use
class TrainDataset_sheets(Dataset):
    def __init__(self, img_dir, mask_dir, label_file, data_aug_ctrl=None, num_classes=2):

        # Load image and mask files, and sort them
        self.img_files = sorted(
            str(f) for f in Path(img_dir).iterdir()
            if f.suffix.lower() in ('.png', '.jpg', '.jpeg')
        )

        # print(f"Total {len(self.img_files)} image files found for training.")

        if data_aug_ctrl is None:
            data_aug_ctrl = [False, False] # permute_B_R, sharpening
        self.data_aug_ctrl = data_aug_ctrl

        self.num_classes = num_classes
        if self.num_classes>2:
            self.class_df = pd.read_csv(os.path.join(label_file))

        self.mask_files=[]
        all_mask_files = sorted(
            str(f) for f in Path(mask_dir).iterdir()
            if f.suffix.lower() in ('.png', '.jpg', '.jpeg') and "_mask" in f.name
        )

        img_files_to_remove = []
        for img_file in self.img_files:
            img_name = Path(img_file).stem.replace("_img", "")
            mask_file = [f for f in all_mask_files if img_name in f]
            if len(mask_file) == 0:
                # raise Exception(f"Error: {len(mask_file)} mask files found for {img_name}")

                # change to only skip the file instead of raising an error
                # print (f"Error: {len(mask_file)} mask files found for {img_name}. Skipping this file.")
                # remove the corresponding image file as well
                img_files_to_remove.append(img_file)
                continue
            elif len(mask_file) > 1:
                raise Exception(f"Error: {len(mask_file)} mask files found for {img_name}")
            else:
                # print(f"Found mask file {mask_file} for {img_name}")
                self.mask_files.append(mask_file[0])

        # remove image files without corresponding mask files
        self.img_files = [
            f for f in self.img_files
            if f not in img_files_to_remove
        ]
        print(f"Total {len(self.img_files)} image and {len(self.mask_files)} mask found for training.")
        # print(self.img_files)
        # print(self.mask_files)

    def __getitem__(self, idx):
        # it is a key to performance that cv2.imread is used instead of tsp.imread b/c it leads to better contrast
        img = cv2.imread(self.img_files[idx], -1)
        # print(self.img_files[idx], img.shape) # check if image has an alpha channel
        img = normalize_img(img)

        mask = cv2.imread(self.mask_files[idx], -1)
        
        # make sure the first dimension is channel
        if img.shape[2] == min(img.shape):
            img = np.transpose(img, (2, 0, 1))

        # Transformation
        # leave xy, patch size, unspecified
        img_trans, mask_trans = random_rotate_and_resize_sheet (X=[img], Y=[mask],
                                                          scale_range=1, do_flip=True, do_rotate=True,
                                                          permute_B_R=self.data_aug_ctrl[0],
                                                          sharpening=self.data_aug_ctrl[1])
        # if the patch does not have any gt mask, redo transformation
        while len(np.unique(mask_trans)) == 1: 
            img_trans, mask_trans = random_rotate_and_resize_sheet (X=[img], Y=[mask],
                                                          scale_range=1, do_flip=True, do_rotate=True,
                                                          permute_B_R=self.data_aug_ctrl[0],
                                                          sharpening=self.data_aug_ctrl[1])

        import hashlib
        h = hashlib.sha256(img_trans.tobytes()).hexdigest()
        print(h)

        obj_ids = np.unique(mask_trans) # get list of gt masks, e.g. [0,1,2,3,...]
        obj_ids = obj_ids[1:] # remove background 0
        num_objs = len(obj_ids)

        # Split a mask map into multiple binary mask map
        masks = mask_trans == obj_ids[:, None, None] # masks is an array of shape num_objs x height x width
        
        # Get labels
        if self.num_classes>2:
            # subset class_df to the current image
            fname1 = os.path.basename(self.img_files[idx])
            fname_no_ext = os.path.splitext(fname1)[0]   # removes extension
            clean_name = fname_no_ext.replace("_img", "")  # removes _img
            subset_class_df = self.class_df[self.class_df['file_name'] == clean_name]
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
class TestDataset_sheets(Dataset):
    def __init__(self, root):
        self.img_files = sorted(
            str(f) for f in Path(root).iterdir()
            if f.suffix.lower() in ('.png', '.jpg', '.jpeg')
        )


    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        # print(img_path)
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


