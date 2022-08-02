# %%
import tensorflow as tf
import pandas as pd
import numpy as np 
import os
from PIL import Image 
import PIL 
import preProcessesImages
import matplotlib.image as mpimg
from tqdm import tqdm
import requests
import sys
from shutil import copyfile

# %%

SEED_VALUE = 1235                                                                           
np.random.seed(seed=SEED_VALUE)  

namesMaj = ['Square',  'Pentagon',  'Hexagon',  'Circle',  'Heptagon']
namesMin = ['Star','Square','Pentagon','Nonagon', 'Octagon']

cmplxName = ["c1", "c2", "c3", "c4", "c5"]

irName = ["b5", "b4", "b3", "b2", "b1"]
minSize = [5000,1500,750,250,125]

# %%
arr=None
if os.path.isdir('../../data/shapes/raw') == False:
    print("Please go to  https://data.mendeley.com/datasets/wzr2yv7r53/1 to download the shapes dataset")
else:
    arr = os.listdir('../../data/shapes/raw/')
# %%

#CREATE DATASETS
#For complexity level
for i in range(len(namesMaj)):
    #get majority and minority instances file names
    majClassInstances = list(filter(lambda name: namesMaj[i] in name, arr))
    print(len(majClassInstances))
    minClassInstances = list(filter(lambda name: namesMin[i] in name, arr))
    majTstIdx = np.random.choice(len(majClassInstances),5000, replace=False)
    minTstIdx = np.random.choice(len(minClassInstances),5000, replace=False)
    majTrnIdx = np.random.choice(np.setdiff1d(np.arange(len(majClassInstances)),majTstIdx), 5000, replace=False)
    minTrnIdx = np.setdiff1d(np.arange(len(minClassInstances)),minTstIdx)
    os.makedirs('../../data/shapes/testDatasets/'+cmplxName[i]+'/')
    for idx in majTstIdx:
        copyfile('../../data/shapes/raw/'+majClassInstances[idx], '../../data/shapes/testDatasets/'+cmplxName[i]+'/'+majClassInstances[idx])
    for idx in minTstIdx:
        copyfile('../../data/shapes/raw/'+minClassInstances[idx], '../../data/shapes/testDatasets/'+cmplxName[i]+'/'+minClassInstances[idx])
    #For imbalance level
    for j in range(len(minSize)):
        os.makedirs('../../data/shapes/trainDatasets/'+cmplxName[i]+'/'+irName[j]+'/')
        minTrnIdx = np.random.choice(minTrnIdx, minSize[j], replace=False)
        for idx in majTrnIdx:
            copyfile('../../data/shapes/raw/'+majClassInstances[idx], '../../data/shapes/trainDatasets/'+cmplxName[i]+'/'+irName[j]+'/'+majClassInstances[idx])
        for idx in minTrnIdx:
            copyfile('../../data/shapes/raw/'+minClassInstances[idx], '../../data/shapes/trainDatasets/'+cmplxName[i]+'/'+irName[j]+'/'+minClassInstances[idx])

