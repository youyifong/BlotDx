import os, glob
import numpy as np

from tsp import imread, imsave
from pre_post_processing.crop_strips import crop_strips, reorder_strip_id, crop_strips_dS_box


# choose 1 from the following
# mask_dir='Masks/gt_sS_SEG' # gt masks
mask_dir='Masks/gt_dS_DET' # gt masks
# mask_dir='Masks/pred_SEG_sS1' # predicted masks
# mask_dir='Masks/pred_DET_dS' # predicted masks

# choose 1 from the following
# img_dirs=['Image/CL', 'Image/validation', 'Image/CZ']
img_dirs=['Image/validation']

crop_version = 6


mask_files = sorted(glob.glob(os.path.join(mask_dir, '*_mask*.png')))

for img_dir in img_dirs:

    # remove Masks/ from mask_dir
    # remove pred_ from mask_dir
    saveto = (img_dir + '_' +
              mask_dir.replace("pred_","").replace("Masks/", "").replace("gt_sS_SEG", "gt") +
              "_strips")
    if 'sS' in mask_dir:
        saveto += "_v" + str(crop_version)

    os.makedirs(saveto, exist_ok=True)  # mkdir
    # remove all files in the directory since png files time stamp is not reliable
    for filename in os.listdir(saveto):
        file_path = os.path.join(saveto, filename)
        os.remove(file_path)

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
        for i, strip in enumerate(strips):
            if crop_version == 6 and 'sS' in mask_dir:
                strip_path = os.path.join(saveto, img_name + '_' + str(255-i*2) + '.tiff')
            else:
                strip = np.transpose(strip, (1, 2, 0))
                strip_path = os.path.join(saveto, img_name + '_' + str(255-i*2) + '.png')

            imsave(strip_path, strip)
