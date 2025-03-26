# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 10:34:33 2024

@author: Youyi
"""


### Data check

import cv2, glob

files = glob.glob('../train_1/*_img.png')
for f in files:
    img=cv2.imread(f, -1)
    
    if img.shape[2]==4:
        print(f)
        # Save. skip alpha channel by using 0:3    
        cv2.imwrite(f, img[:,:,0:3])


# cv2.imread('2016.08.03_MJ_01.png', -1).shape
# cv2.imread('MJ/2016.08.03_MJ_01.png', -1).shape


# go through all directories under D:\DeepLearning\HSVWesternDiagnosticMethods\Image1. For each directory, remove files whose names are not in the strip_id column of sS_labels.csv
import os
import pandas as pd

# Load allowed filenames from the CSV
csv_path = r'D:\DeepLearning\HSVWesternDiagnosticMethods\Image1\sS_labels.csv'
df = pd.read_csv(csv_path)
allowed_names = set(df['strip_id'].astype(str))


# Base directory to process
base_dir = r'D:\DeepLearning\HSVWesternDiagnosticMethods\Image1'

# Go through all directories and subdirectories
for root, dirs, files in os.walk(base_dir):
    for file in files:
        file_name, _ = os.path.splitext(file)  # remove extension
        if file_name not in allowed_names:
            file_path = os.path.join(root, file)
            try:
                os.remove(file_path)
                print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
