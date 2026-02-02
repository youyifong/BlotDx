import cv2, math, glob
import numpy as np
from PIL import Image, ImageDraw

from fit_edge import fit_edge



def fit_quad_to_masks_linefit(fname):
    """
    Fits a quadrilateral to the binary masks.

    fit a straight to the left edge
    fit a straight line to the right edge
    fit a straight line to the top edge, but constrain the y coordinate difference to be <=1
    same for the bottom

    Parameters:
        fname: name of a file that contains binary masks where the object is white (255) and background is black (0).

    Returns:
        quadrilateral (list): List of four corner points representing the quadrilateral.
    """
    
    masks = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
    # np.histogram(masks, bins=np.append(np.unique(masks), np.inf))

    image_height, image_width = masks.shape        
    fillid=255    
    masks_image = Image.new('L', (image_width, image_height), 0) # Create a blank image with the same dimensions
    draw = ImageDraw.Draw(masks_image)
    
    indices = np.unique(masks)[1:]
    for i in indices: 
    # for i in [x+5*1 for x in [1,2,3,4,5]]: 
        
        points1 = np.argwhere(masks == i)
        # Reshape the array to (n, 1, 2); then revert the order of coordinates from y,x to x,y
        # points = np.array(points1).reshape(-1, 1, 2)[:, :, ::-1]
        
        plot=False # for debug
        left_edge = fit_edge(points, mode='left', plot=plot)
        right_edge = fit_edge(points, mode='right', plot=plot)
        top_edge = fit_edge(points, mode='top', plot=plot)
        bottom_edge = fit_edge(points, mode='bottom', plot=plot)
        [left_edge, top_edge, right_edge, bottom_edge]      
        
        quadrilateral = find_all_intersections([left_edge, top_edge, right_edge, bottom_edge]); quadrilateral
        draw.polygon(quadrilateral, outline=fillid, fill=fillid) 
        fillid=fillid-1

    masks_image.save(fname.replace('.png', "_linefit.png"))
        


    
files = glob.glob('cellpose_train_CL_MJ_pred/*_cp_masks.png')
for f in files:
    fit_quad_to_masks_linefit(f)
    

