# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 08:29:11 2024

@author: Youyi
"""

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

