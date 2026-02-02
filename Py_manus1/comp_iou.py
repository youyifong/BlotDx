from tsp.AP import compute_iou, dice
from tsp import imread


# compute iou, dice
def f (pred_path, gt_path):
    # pred_path = 'Mask/pred_DET_dS/2016.09.22_CZ_01_tv_masks.png'
    # gt_path = 'Mask/gt_dS_DET/2016.09.22_CZ_01_mask.png'
    y_pred = imread(pred_path)
    labels = imread(gt_path)
    iou = compute_iou(mask_true=labels, mask_pred=y_pred)  # compute iou
    return [round(iou[iou > 0].mean(),2), round(dice(y_pred, labels),2)]


# DET
print(f('Mask/pred_DET_dS/2016.09.22_CZ_01_tv_masks.png',   'Mask/gt_dS_DET/2016.09.22_CZ_01_mask.png'))
print(f('Mask/pred_DET_dS/2016.10.04_JHCL_01_tv_masks.png', 'Mask/gt_dS_DET/2016.10.04_JHCL_01_mask.png'))
print(f('Mask/pred_DET_dS/2016.10.04_JHCL_02_tv_masks.png', 'Mask/gt_dS_DET/2016.10.04_JHCL_02_mask.png'))
print(f('Mask/pred_DET_dS/2016.10.04_JHCL_03_tv_masks.png', 'Mask/gt_dS_DET/2016.10.04_JHCL_03_mask.png'))
# [0.88, 0.93]
# [0.87, 0.93]
# [0.87, 0.93]
# [0.86, 0.92]


# SEG
print(f('Mask/pred_SEG_sS1/2016.09.22_CZ_01_tv_masks.png',   'Mask/gt_sS_SEG/2016.09.22_CZ_01_mask.png'))
print(f('Mask/pred_SEG_sS1/2016.10.04_JHCL_01_tv_masks.png', 'Mask/gt_sS_SEG/2016.10.04_JHCL_01_mask.png'))
print(f('Mask/pred_SEG_sS1/2016.10.04_JHCL_02_tv_masks.png', 'Mask/gt_sS_SEG/2016.10.04_JHCL_02_mask.png'))
print(f('Mask/pred_SEG_sS1/2016.10.04_JHCL_03_tv_masks.png', 'Mask/gt_sS_SEG/2016.10.04_JHCL_03_mask.png'))
# [0.83, 0.91]
# [0.84, 0.91]
# [0.84, 0.92]
# [0.84, 0.91]
