import os, glob
import numpy as np
from tsp import imread, imsave
from Py_pre_post_processing.crop_strips import crop_strips, crop_strips_dS_box


'''
Set the right hand side of an image to a gray pixel value for testing purposes.
'''

img_file = 'Image/test_SEG_sS1_strips_v4/2016.09.01_CZ_01_209.png'

img = imread(img_file)  # (420, 46, 3)

# set the right hand side to a gray pixel value
img[:, 23:] = img[10, 7]

# save the image
imsave(img_file.replace(".png", "_white.png"), img)



'''
Mask some bands on left hand side of an image to a gray pixel value for testing purposes.
'''

img_file = 'Image/test_SEG_sS1_strips_v4/2016.09.01_CZ_01_209_white.png'

img = imread(img_file)  # (420, 46, 3)
img[80:100, :23] = img[10, 7]
imsave(img_file.replace(".png", "_mask1.png"), img)

img = imread(img_file)  # (420, 46, 3)
img[100:500, :23] = img[10, 7]
imsave(img_file.replace(".png", "_mask2.png"), img)

img = imread(img_file)  # (420, 46, 3)
img[0:80, :23] = img[10, 7]
img[100:500, :23] = img[10, 7]
imsave(img_file.replace(".png", "_mask3.png"), img)

img = imread(img_file)  # (420, 46, 3)
img[0:500, :23] = img[10, 7]
imsave(img_file.replace(".png", "_mask4.png"), img)

img = imread(img_file)  # (420, 46, 3)
img[25:500, :23] = img[10, 7]
imsave(img_file.replace(".png", "_mask5.png"), img)


'''
modify all files in Image/CL_SEG_sS1_strips_v4_HSV1_W48px by adding one pixel on the left side that copies the first pixel column and adding one pixel on the right side that copies the last pixel column.
'''

# img_dir = 'Image/CL_SEG_sS1_strips_v4_W48px/'
# img_dir = 'Image/alltest_SEG_sS1_strips_v4_W48px/'
img_dir = 'Image/validation_SEG_sS1_strips_v4_W48px/'
img_files = glob.glob(os.path.join(img_dir, '*.png'))
for img_file in img_files:
    img = imread(img_file)  # (420, 48, 3)
    left_col = img[:, 0:1, :]  # (420, 1, 3)
    right_col = img[:, -1:, :]  # (420, 1, 3)
    img_modified = np.concatenate((left_col, img, right_col), axis=1)  # (420, 50, 3)
    imsave(img_file.replace('.png', '_W48px.png'), img_modified)
    # print(f'Modified and saved: {img_file.replace(".png", "_W48px.png")}')



'''
make 201608-201702_Tranche2
'''

# pick 34 files from 201608-201702 that are not in 201608-201702_Tranche1
import glob, os
src_dir = '/fh/fast/fong_y/HSVW/HSVWesternDiagnosticMethods/Image/sheets/201608-201702/'
dst_dir = '/fh/fast/fong_y/HSVW/HSVWesternDiagnosticMethods/Image/sheets/201608-201702_Tranche2/'
all_files = glob.glob(os.path.join(src_dir, '*.png'))
tranche1_files = glob.glob(os.path.join('/fh/fast/fong_y/HSVW/HSVWesternDiagnosticMethods/Image/sheets/201608-201702_Tranche1/', '*.png'))
tranche1_file_names = [os.path.basename(f) for f in tranche1_files]

count = 0
for file in all_files:
    file_name = os.path.basename(file)
    if file_name not in tranche1_file_names:
        # print(file_name)
        os.system(f'cp {file} {dst_dir}')
        count += 1
        if count >= 34:
            break
print(f'Copied {count} files to {dst_dir}')


# some of these files are ADS, so need to get more into a temp folder 201608-201702_Tranche2a and manually check which ones are ADS
# note that could have checked hsv_catalogue.xlsx
import glob, os
src_dir = '/fh/fast/fong_y/HSVW/HSVWesternDiagnosticMethods/Image/sheets/201608-201702/'
dst_dir = '/fh/fast/fong_y/HSVW/HSVWesternDiagnosticMethods/Image/sheets/201608-201702_Tranche2a/'
t2_dir  = '/fh/fast/fong_y/HSVW/HSVWesternDiagnosticMethods/Image/sheets/201608-201702_Tranche2/'
all_files = glob.glob(os.path.join(src_dir, '*.png'))
tranche1_files = glob.glob(os.path.join('/fh/fast/fong_y/HSVW/HSVWesternDiagnosticMethods/Image/sheets/201608-201702_Tranche1/', '*.png'))
tranche1_file_names = [os.path.basename(f) for f in tranche1_files]
tranche2_files = glob.glob(os.path.join('/fh/fast/fong_y/HSVW/HSVWesternDiagnosticMethods/Image/sheets/201608-201702_Tranche2/', '*.png'))
tranche2_file_names = [os.path.basename(f) for f in tranche2_files]

count = 0
for file in all_files:
    file_name = os.path.basename(file)
    if file_name not in tranche1_file_names and file_name not in tranche2_file_names:
        # print(file_name)
        os.system(f'cp {file} {dst_dir}')
        count += 1
        if count >= 34:
            break
print(f'Copied {count} files to {dst_dir}')



'''
Resize MJ for experimentation
'''
import cv2
import glob, os

files = glob.glob('MJ/*.png')

for f in files:
    
    # create new folder to store scaled down images and masks
    tmp = os.path.normpath(f).split(os.sep)    
    newfolder=tmp[0]+'_half/'
    if not os.path.isdir(newfolder):
        os.mkdir(newfolder)
        
    newmaskfolder=tmp[0]+'_half/masks/'
    if not os.path.isdir(newmaskfolder):
        os.mkdir(newmaskfolder)
    
    # image
    image = cv2.imread(f, -1)    
    # Scale down by half
    scaled_image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
    # Save. skip alpha channel by using 0:3    
    cv2.imwrite(newfolder + tmp[1], scaled_image[:,:,0:3])
    
    # mask 
    m = tmp[0]+'/masks/' + tmp[1].replace('.png', '_mask.png')
    image = cv2.imread(m,-1)    
    # Scale down by half
    scaled_image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
    # Save or display the scaled image
    cv2.imwrite(newmaskfolder + tmp[1].replace('.png', '_mask.png'), scaled_image)
