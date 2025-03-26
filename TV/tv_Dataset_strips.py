import os, glob, torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

from tsp import imread
from TV.pre_post_processing.crop_strips import crop_strips
from TV.tv_utils import random_rotate_and_resize_strip, normalize_img


### Dataset and DataLoader. Prepares a pair of strips at a time as input, either HSV-1 or HSV-2 status as outcome
class TrainDataset_strips(Dataset): # noqa
    def __init__(self, img_dir, label_file, HSV, diagnostic_type, num_classes, mask_dir=None, nchan=None, data_augmentation=True, reorder=True): # noqa
        """
        load all pairs of strips from all images

        Parameters:
            img_dir: directory where training images files are stored
            label_file: path to the label file. There can be unused labels in the file
            HSV: 1 or 2 for HSV-1 or HSV-2 status as label
            diagnostic_type: Final, Majority
            num_classes: .e.g. 2
            mask_dir: directory where masks files are stored. There can be unused mask files in the dir
            nchan: 6 channels if HSV1 and 2 strips are stacked together and 3 if they are side by side
        """

        self.data_augmentation = data_augmentation
        # if mask_dir is None, nchan has to be None as well
        assert (mask_dir is None) == (nchan is None)

        img_files = sorted(glob.glob(os.path.join(img_dir, '*')))
        labels_df = pd.read_csv(label_file)

        # diagnostic_type needs to be in ['Final', 'Majority']
        assert diagnostic_type in ['Final', 'Majority']
        assert HSV in [1, 2]

        all_strips=[]
        all_labels=[]
        all_ids=[]

        if mask_dir is None:
            # read dS images from hard drive
            for img_file in img_files:
                # each img file has 1 pair of strips
                img = imread(img_file) # (height, width, 3) for png and (6, height, width) for tiff
                img = normalize_img(img) # work with either channel-first or channel-last
                if img.shape[2] == np.min(img.shape):
                    img = np.transpose(img, (2, 0, 1)) # (3, height, width)
                self.nchan = img.shape[0]
                all_strips.append(img)

                # get labels by strip_id
                img_name = os.path.splitext(os.path.basename(img_file))[0]
                subset_labels = labels_df[labels_df['strip_id'] == img_name]
                # assert subset_labels is not empty
                if len(subset_labels) == 0:
                    raise Exception(f"Error: no labels found for {img_name}")
                all_labels.extend(subset_labels[diagnostic_type + 'HSV' + str(HSV)])
                all_ids.extend(subset_labels['strip_id'])

        else:
            # generate dS images on the fly
            self.nchan = nchan
            mask_files = sorted(glob.glob(os.path.join(mask_dir, '*_mask.png')))
            for img_file in img_files:
                # each img file has up to 24 pairs of strips
                print(f"Processing {img_file}")

                img_name = os.path.basename(img_file).replace("_img.png", ".png").replace(".png", "")

                # find the mask file
                mask_file = [f for f in mask_files if img_name in f]
                if len(mask_file) != 1:
                    raise Exception(f"Error: {len(mask_file)} mask files found for {img_name}")
                masks = imread(mask_file[0])  # (1288, 1936)

                # get strip images
                img = imread(img_file)  # (height, width, 3)
                # img = img[:, :, [2, 1, 0]] # switch B and R channels
                img = normalize_img(img)  # (height, width, 3)
                res = crop_strips(img, masks, nchan, strip_width=23,
                                  strip_height=420)  # (nchan, strip_height, strip_width)
                all_strips.extend(res)

                # get labels by img_file
                subset_labels = labels_df[labels_df['img_file'] == img_name]
                # assert that mask_id is unique and in descending order
                assert all(-2 == subset_labels['mask_id'].iloc[i] - subset_labels['mask_id'].iloc[i - 1] for i in
                           range(1, len(subset_labels)))
                all_labels.extend(subset_labels[diagnostic_type + 'HSV' + str(HSV)])
                all_ids.extend(subset_labels['strip_id'])


        assert len(all_strips) == len(all_labels)


        if num_classes==2:
            # discard strips that are not POS or neg

            if not reorder:
                # discard strips whose labels are not POSITIVE or negative
                self.strips = [all_strips[i] for i in range(len(all_labels)) if
                               all_labels[i] in ['POSITIVE', 'negative']]  # noqa
                self.strip_ids = [all_ids[i] for i in range(len(all_labels)) if
                                  all_labels[i] in ['POSITIVE', 'negative']] # noqa
                self.labels = [all_labels[i] for i in range(len(all_labels)) if
                               all_labels[i] in ['POSITIVE', 'negative']]  # noqa

                # convert labels to 1 and 0
                self.labels = [1 if label == 'POSITIVE' else 0 for label in self.labels]
                print(f"Number of POSITIVE and negative strips: {sum(self.labels)} : {len(self.labels)-sum(self.labels)}")

            else:
                # reorder strips to mix POS and neg strips evenly

                # split all_strips into two lists, one for when the corresponding label in all_labels is POSITIVE, the other for when the label is negative
                strips_POS = [all_strips[i] for i in range(len(all_labels)) if all_labels[i] == 'POSITIVE']  # noqa
                strips_neg = [all_strips[i] for i in range(len(all_labels)) if all_labels[i] == 'negative']  # noqa
                print(f"Number of POSITIVE and negative strips: {len(strips_POS)} : {len(strips_neg)}")

                ids_POS = [all_ids[i] for i in range(len(all_labels)) if all_labels[i] == 'POSITIVE']  # noqa
                ids_neg = [all_ids[i] for i in range(len(all_labels)) if all_labels[i] == 'negative']  # noqa

                if len(strips_POS) > len(strips_neg):
                    strips_large = strips_POS
                    strips_small = strips_neg
                    ids_large = ids_POS
                    ids_small = ids_neg
                    label_large = 1 # 'POSITIVE'
                    label_small = 0 # 'negative'
                else:
                    strips_large = strips_neg
                    strips_small = strips_POS
                    ids_large = ids_neg
                    ids_small = ids_POS
                    label_large = 0 # 'negative'
                    label_small = 1 # 'POSITIVE'

                order = insert_evenly(len(strips_small), len(strips_large))

                strips=[]
                labels=[]
                ids=[]

                for i in range(len(order)):
                    if order[i] == "s":
                        strips.append(strips_small.pop(0))
                        labels.append(label_small)
                        ids.append(ids_small.pop(0))
                    else:
                        strips.append(strips_large.pop(0))
                        labels.append(label_large)
                        ids.append(ids_large.pop(0))

                self.strips = strips
                self.labels = labels
                self.strip_ids = ids

    def __getitem__(self, idx):

        img = self.strips[idx]
        label = self.labels[idx]
        strip_id = self.strip_ids[idx]

        if self.data_augmentation:
            # Transformation and convert into a torch.Tensor
            img = torch.as_tensor(random_rotate_and_resize_strip(X=[img],
                                                                 scale_range=0.2,
                                                                 do_flip=self.nchan==6, # bad for side by side strips
                                                                 do_rotate=True,
                                                                 permute_B_R=True), dtype=torch.float32)
        else:
            img = torch.as_tensor(img, dtype=torch.float32)

        return img, label, strip_id

    def __len__(self):
        return len(self.strips)


# same as Train but not reordering the strips in the list and no data augmentation
class ValDataset_strips(TrainDataset_strips): # noqa
    def __init__(self, img_dir, label_file, HSV, diagnostic_type, num_classes, mask_dir, nchan): # noqa
        super().__init__(img_dir, label_file, HSV, diagnostic_type, num_classes, mask_dir, nchan,
                         data_augmentation=False, reorder=False)





### Different from train dataset in that there are no labels
class TestDataset_strips(Dataset):  # noqa
    def __init__(self, img_dir, mask_dir, nchan):  # noqa

        img_files = sorted(glob.glob(os.path.join(img_dir, '*.png')))
        mask_files = sorted(glob.glob(os.path.join(mask_dir, '*_mask.png')))

        all_strips=[]
        for img_file in img_files:
            print(f"Processing {img_file}")

            img_name = os.path.basename(img_file).replace("_img.png", ".png").replace(".png", "")

            # find the corresponding mask file
            mask_file = [f for f in mask_files if img_name in f]
            if len(mask_file) != 1:
                raise Exception(f"Error: {len(mask_file)} mask files found for {img_name}")
            masks = imread(mask_file[0])

            # get strip images
            img = imread(img_file) # (height, width, 3)
            # img = img[:, :, [2, 1, 0]] # switch B and R channels
            img = normalize_img(img)  # (height, width, 3)
            all_strips.extend(crop_strips(img, masks, nchan, strip_width=23, strip_height=420)) # (6, 420, 23)

        self.strips = all_strips


    def __getitem__(self, idx):
        # Convert image into tensor
        img = torch.as_tensor(self.strips[idx], dtype=torch.float32) # for image

        return img
    
    def __len__(self):
        return len(self.strips)




#  insert 1, ..., n_small into 1, ..., n_large so that the number from the first series are spaced as evenly as possible.
def insert_evenly(n_small, n_large):
    assert n_small <= n_large

    # Calculate the positions to insert n_small numbers into large_sequence
    if n_small == n_large:
        step = 1
    else:
        step = n_large / (n_small + 1)

    result = []
    # Index of current element from n_small to insert
    small_idx = 1
    for i in range(n_large):
        result.append("l")
        # Calculate the target position for insertion
        if i == int(small_idx * step) - 1 and small_idx <= n_small:
            result.append("s")
            small_idx += 1

    assert len(result) == n_large + n_small

    return result

