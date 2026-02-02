"""
This script creates 5 cross-validation datasets from the original dataset.
"""

import os, glob
import numpy as np
import random

# dirs=['Image/blots/201608-201702_Tranche1and2/SEG_sS1_strips_v4', 'Image/blots/201608-201702_Tranche1and2/SEG_sS1_strips_v6', 'Image/blots/201608-201702_Tranche1and2/DET_dS_strips']

# dirs=['Image/blots/201608-201702_Tranche1and2new_bw/SEG_sS1_strips_v4',
#       'Image/blots/201608-201702_Tranche1and2new_bw/SEG_sS1_strips_v6',
#       'Image/blots/201608-201702_Tranche1and2new_bw/DET_dS_strips',]

dirs=[
      'Image/blots/201608-201702_Tranche1and2new_bw/DET_dS_strips']

for dir in dirs:
    # set random seed in the same way for all dirs
    seed = 0
    np.random.seed(seed)
    random.seed(seed)

    # read file names from dir
    files = sorted(glob.glob(os.path.join(dir, '*')))
    # randomly divide files into 5 groups
    n = len(files)
    print(n)
    # draw with replacement
    samples = np.random.choice(range(n), size=n, replace=False)
    files = [files[i] for i in samples]

    groups = np.array_split(files, 5)   
    print([len(g) for g in groups]) 

    # this is the wrong code that was used before. It creates 6 groups if the number of files is not a multiple of 5    
    # group_size = n // 5
    # groups = [files[i:i + group_size] for i in range(0, n, group_size)]
    # # add remaining files to the last group
    # groups[-1].extend(files[group_size * 5:])

    # make 5 directories
    for i in range(5):
        os.makedirs(os.path.join(dir, f'cv{i}'), exist_ok=True)
        # create a train and a test dir under each cv dir
        os.makedirs(os.path.join(dir, f'cv{i}/train'), exist_ok=True)
        os.makedirs(os.path.join(dir, f'cv{i}/test'), exist_ok=True)
        # copy files to the train and test dir
        for j in range(5):
            if i == j:
                for file in groups[j]:
                    os.system(f'cp {file} {os.path.join(dir, f"cv{i}/test")}')
            else:
                for file in groups[j]:
                    os.system(f'cp {file} {os.path.join(dir, f"cv{i}/train")}')


