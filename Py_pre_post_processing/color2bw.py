from PIL import Image
import numpy as np
from scipy.optimize import minimize

def color2bw (img_file, mask_file, initial_w = [1/3, 1/3, 1/3], bright_strip=False):
    
    image = Image.open(img_file)
    mask = Image.open(mask_file)
    image_np = np.array(image)
    mask_np = np.array(mask)

    # Define the function to be maximized (negated for minimization)
    def diff(w):
        # Extract pixel values where mask is 0 (background) and mask is not 0 (foreground)
        bg_pixels = image_np[mask_np == 0]
        st_pixels = image_np[mask_np != 0]
        
        if bg_pixels.size == 0 or st_pixels.size == 0:
            # If either group is empty, return a large negative number to avoid None issues
            return -np.inf
    
        # Apply weighted combination of the RGB channels for background and foreground
        bg = w[0] * bg_pixels[:, 0] + w[1] * bg_pixels[:, 1] + w[2] * bg_pixels[:, 2]
        st = w[0] * st_pixels[:, 0] + w[1] * st_pixels[:, 1] + w[2] * st_pixels[:, 2]
        
        if bright_strip:
            # strip values need to be high, thus min
            return np.median(bg) - np.median(st)
        else:
            return np.median(st) - np.median(bg)
            
    
    # Constraint: the sum of the weights should be 1
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        
    # Bounds: each weight should be between 0 and 1
    bounds = [(-2, 2), (-2, 2), (-2, 2)]
    
    # Perform the optimization
    result = minimize(diff, initial_w, method='SLSQP', bounds=bounds, constraints=constraints)
    
    # Optimized weights
    optimal_w = result.x
    
    # Print the result
    print("Optimized weights:", optimal_w)
    print("Minimized diff value:", result.fun)
        
    return optimal_w[0] * image_np[:, :, 0] + optimal_w[1] * image_np[:, :, 1] + optimal_w[2] * image_np[:, :, 2]
    


import glob, os
files = glob.glob('CL/*.png')
for f in files:
    tmp = os.path.normpath(f).split(os.sep)    
    # place of the mask file 
    m = tmp[0]+'/masks/' + tmp[1].replace('.png', '_mask.png')
    
    bw_image_np = color2bw (f, m, initial_w = [1/3, 1/3, 1/3])
    bw_image = Image.fromarray(bw_image_np.astype('uint8'))
    # bw_image.show()
    newfolder=tmp[0]+'_bw/'
    if not os.path.isdir(newfolder):
        os.mkdir(newfolder)
    bw_image.save(newfolder + tmp[1])


# for CZ, use predicted masks instead of gt masks
import glob, os
files = glob.glob('CZ/*.png')
for f in files:
    tmp = os.path.normpath(f).split(os.sep)    
    # place of the mask file 
    m = 'cellpose_train_CL_MJ_pred/' + tmp[1].replace('.png', '_cp_masks.png')
    
    bw_image_np = color2bw (f, m, initial_w = [1/3, 1/3, 1/3])
    bw_image = Image.fromarray(bw_image_np.astype('uint8'))
    # bw_image.show()
    newfolder=tmp[0]+'_bw/'
    if not os.path.isdir(newfolder):
        os.mkdir(newfolder)
    bw_image.save(newfolder + tmp[1])


