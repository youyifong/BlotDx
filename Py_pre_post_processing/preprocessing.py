import cv2, glob, os
import numpy as np


'''
Resize image
'''

# files = glob.glob('Image/sheets/201907_202505w_raw/*.JPG')
# newfolder='Image/sheets/201907_202505w/'
# if not os.path.isdir(newfolder):
#     os.mkdir(newfolder)
#
# for f in files:
#     print(f)
#     img = cv2.imread(f, -1)
#
#     # resiz
#     H = 1288; W = 1936
#     img = cv2.resize(img, (W, H))
#
#     dst = newfolder + os.path.basename(f)
#     # saving as JPG would lead to some quality loss, file size compression ratio is about 1/3
#     dst = dst.replace(".JPG", ".png")
#
#     cv2.imwrite(
#         dst,
#         img[:,:,0:3]) # skip alpha channel by using 0:3
    

'''
Sharpening
'''

files = glob.glob('Image/sheets/201608-201702_Tranche2/2016.08.30_JHCL_01.png'); newfolder='Image/sheets/201608-201702_Tranche2/' 
# files = glob.glob('Image/sheets/201608-201702_Tranche1/CL/2016.10.03_CL_02.png'); newfolder='Image/sheets/201608-201702_Tranche1/CL/' 
if not os.path.isdir(newfolder):
    os.mkdir(newfolder)

for f in files:
    print(f)
    img = cv2.imread(f, -1)    
    print(img.shape)

    # # # Basic sharpening
    # kernel = np.array([[0, -1,  0],
    #                 [-1,  5, -1],
    #                 [0, -1,  0]])
    # img = cv2.filter2D(img, -1, kernel)

    # Gaussian blur sharpening
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=3)
    # Unsharp mask: output = img + amount*(img - blur)
    amount = 1.5    # strength of sharpening
    img = cv2.addWeighted(img, 1 + amount, blur, -amount, 0)
    
    dst = newfolder + os.path.basename(f)
    # saving as JPG compresses several fold more than png 
    dst = dst.replace(".JPG", ".png")
    
    cv2.imwrite(
        dst, 
        img[:,:,0:3]) # skip alpha channel by using 0:3
    


'''
Make warmer image
'''

# files = glob.glob('Image/sheets/201907_202505/*.png')
# newfolder='Image/sheets/201907_202505_warmed/'
# if not os.path.isdir(newfolder):
#     os.mkdir(newfolder)

# for f in files:
#     print(f)
#     img = cv2.imread(f, -1)    

#     # Convert to float for safe math
#     warm = img.astype(np.float32)
#     # Increase Red channel, decrease Blue channel
#     warm[:,:,2] *= 1.08    # Red channel (BGR order: 0=B,1=G,2=R)
#     warm[:,:,0] *= 0.92    # Blue channel
#     # Clip to valid range
#     img = np.clip(warm, 0, 255).astype(np.uint8)

#     dst = newfolder + os.path.basename(f)
#     # saving as JPG compresses several fold more than png 
#     dst = dst.replace(".JPG", ".png")
    
#     cv2.imwrite(
#         dst, 
#         img[:,:,0:3]) # skip alpha channel by using 0:3
    

'''
remove alpha channel
'''

import cv2, glob, os
files = glob.glob('Image/sheets/tmp/*.png')
newfolder        ='Image/sheets/tmp/'
if not os.path.isdir(newfolder):
    os.mkdir(newfolder)

for f in files:
    print(f)
    img = cv2.imread(f, -1)

    if img.shape[2] < 4:
        print(f'Image {f} has no alpha channel, skipping.')
        continue

    dst = newfolder + os.path.basename(f)
    # # saving as JPG compresses several fold more than png
    # dst = dst.replace(".JPG", ".png")

    cv2.imwrite(
        dst,
        img[:,:,0:3]) # skip alpha channel by using 0:3
    

