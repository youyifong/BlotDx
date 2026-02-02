import os, glob, sys
import numpy as np

from tsp import imread, imsave

in_ipython = 'get_ipython' in globals()
if not in_ipython:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from crop_strips import crop_strips, reorder_strip_id, crop_strips_dS_box

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--mask_dir', default="", type=str, help='DET_dS_2_set4')
parser.add_argument('--img_dir', default="", type=str, help='e.g., Image/sheets/201907_202505')
parser.add_argument('--crop_version', default=4, type=int, help='only applicable for SEG_sS, 4 or 6. Default: %(default)s')

args = parser.parse_known_args()[0]
print(args)


crop_version = args.crop_version

mask_dir = os.path.join(args.img_dir, args.mask_dir)
img_dir = args.img_dir

mask_files = sorted(glob.glob(os.path.join(mask_dir, '*_mask*.png')))

# replace Mask/ by Image/ from mask_dir
saveto = mask_dir.replace("Image/sheets/", "Image/blots/")
if 'sS' in mask_dir:
    saveto += "_v" + str(crop_version)
os.makedirs(saveto, exist_ok=True)
print(f"Saving cropped strips to {saveto}")
# remove all files in the directory since png files time stamp is not reliable
# for filename in os.listdir(saveto): os.remove(os.path.join(saveto, filename))

# create folders for HSV1 strips and HSV2 strips
if crop_version == 4 and 'sS' in mask_dir:
    os.makedirs(saveto + "_HSV1", exist_ok=True)
    os.makedirs(saveto + "_HSV2", exist_ok=True)
    # remove all files in the directory since png files time stamp is not reliable
    # for filename in os.listdir(saveto + "_HSV1"): os.remove(os.path.join(saveto + "_HSV1", filename))
    # for filename in os.listdir(saveto + "_HSV2"): os.remove(os.path.join(saveto + "_HSV2", filename))

img_files = sorted(glob.glob(os.path.join(img_dir, '*.png')))

for img_file in img_files:

    print(f"Processing {img_file}")

    img_name = os.path.basename(img_file).replace("_img.png", ".png").replace(".png", "")

    # find the mask file
    mask_file = [f for f in mask_files if img_name in f]
    if len(mask_file) != 1:
        raise Exception(f"Error: {len(mask_file)} mask files found for {img_name}")
    masks = imread(mask_file[0])  # (1288, 1936)
    masks = reorder_strip_id (masks)

    # get strip images
    img = imread(img_file)  # (height, width, 3)

    if 'DET' in mask_dir:
        strips = crop_strips_dS_box (img, masks, strip_width=23, strip_height=420)
    else:
        strips = crop_strips(img, masks, crop_version, strip_width=23, strip_height=420)  # (3, 420, 46) or (6, 420, 23)

    # save each strip as an image file
    # the convention in windows viewing is channel-last for png files, and channel-first for tiff files
    if strips is not None:
        for i, strip in enumerate(strips):
            if crop_version == 6 and 'sS' in mask_dir:
                strip_path = os.path.join(saveto, img_name + '_' + str(255-i*2) + '.tiff')
            else:
                strip = np.transpose(strip, (1, 2, 0))
                strip_path = os.path.join(saveto, img_name + '_' + str(255-i*2) + '.png')
            imsave(strip_path, strip)

            # if crop_version is 4, split the strip into a left and a right part equally and save to two files with HSV1 and HSV2 in the file names
            if crop_version == 4 and 'sS' in mask_dir:
                strip_path1 = os.path.join(saveto + "_HSV1", img_name + '_' + str(255-i*2) + '.png')
                strip_path2 = os.path.join(saveto + "_HSV2", img_name + '_' + str(255-i*2) + '.png')
                imsave(strip_path1, strip[:, 0:23, :])
                imsave(strip_path2, strip[:, 23:46, :])
