import cv2
import numpy as np
from PIL import Image, ImageDraw

def fit_quad_to_masks_rect(fname):
    """
    Fits a quadrilateral to the binary masks. Try several methods, but mostly the one that works is a flexible rectangle
    
    Several methods suggested by ChatGPT. Did not work very well

    Parameters:
        fname: name of a file that contains binary masks where the object is white (255) and background is black (0).

    Returns:
        quadrilateral (list): List of four corner points representing the quadrilateral.
    """
    masks = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
    # np.histogram(masks, bins=np.append(np.unique(masks), np.inf))
    
    # hard code dimension
    image_height, image_width = masks.shape
        
    fillid=255
    
    # Create a blank image with the same dimensions
    masks_image = Image.new('L', (image_width, image_height), 0)
    #print(np.histogram(masks_image, bins=np.append(np.unique(masks_image), np.inf)))

    draw = ImageDraw.Draw(masks_image)
    
    indices = np.unique(masks)[1:]
    for i in indices: 
        
        points = np.argwhere(masks == i)
    
        # Reshape the array to (n, 1, 2); then revert the order from y,x to x,y
        contour = np.array(points).reshape(-1, 1, 2)[:, :, ::-1]        

        # Use cv2.approxPolyDP to approximate the contour to a quadrilateral
        epsilon = 0.001 * cv2.arcLength(contour, True)  # Tolerance for approximation
        approx = cv2.approxPolyDP(contour, epsilon, True)
        approx
        
        # If the approximation has more than 4 points, we may reduce it using convex hull
        if len(approx) > 4:
            approx = cv2.convexHull(approx)
            
        quadrilateral = [tuple(point[0].tolist()) for point in approx]  # Extract coordinates
    
        # Check if we have a quadrilateral is a list of tuples
        if len(approx) == 4:
            quadrilateral = [tuple(point[0].tolist()) for point in approx]  # Extract coordinates
        # elif len(approx)==2:# does not workwell
        #     quadrilateral = [tuple([approx[0,0,0], approx[0,0,1]]),
        #                      tuple([approx[0,0,0], approx[1,0,1]]),
        #                      tuple([approx[1,0,0], approx[1,0,1]]),
        #                      tuple([approx[1,0,0], approx[0,0,1]])]
        else:
            # If not exactly 4 points, we need to approximate again to get 4 points.
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            quadrilateral = [tuple(point) for point in box]
    
        # Draw the polygon
        quadrilateral
        draw.polygon(quadrilateral, outline=fillid, fill=fillid) 
        fillid=fillid-1
    
    masks_image.save(fname.replace('.png', "_rect.png"))
    


fname='cellpose_masks_test_images_before_contrast/2016.08.08_CL_04_cp_masks.png'
fit_quad_to_masks_rect(fname)
    
fname='cellpose_masks_test_images_before_contrast/2016.08.08_CL_03_cp_masks.png'
fit_quad_to_masks_rect(fname)
    
fname='cellpose_masks_test_images_before_contrast/2016.08.08_CL_02_cp_masks.png'
fit_quad_to_masks_rect(fname)
