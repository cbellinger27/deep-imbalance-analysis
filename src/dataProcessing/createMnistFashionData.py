# %%
import tensorflow as tf
import pandas as pd
import numpy as np 
import os
from PIL import Image 
import PIL 
import preProcessesImages
import matplotlib.image as mpimg


(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
train_images, train_labels = preProcessesImages.processMnistFashionData(train_images, train_labels)
test_images, test_labels = preProcessesImages.processMnistFashionData(test_images, test_labels)

SEED_VALUE = 1235                                                                           
np.random.seed(seed=SEED_VALUE)  

names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
cmplxName = ["c1", "c2", "c3", "c4", "c5"]
cc = [[3,7],[8,2],[7,9],[6,4],[2,6]]

irName = ["b1", "b2", "b3", "b4", "b5"]
ir = [0.025, 0.05, 0.15, 0.3, 1]


# %%
#TRAINING SETS: This loop creates to the file structure for the trianing data and places the individual images in the file structure. The loop creates one
#directory for each complexity level named c1, c2,...,c5. In side each complexity level directory, the loop creates five additional directories, on for
#imbalance level named c1_b1, c1_b2,...,c5_b4, c5_b5. Finally, inside each of these, there is one directory for each class names class0 and class1.
#The majority class images are placed inside class0 and the majority class images are placed in side class1.

#for each complexity level
for cmplx_idx in range(len(cmplxName)):
    #for each imbalanace level
    for ir_idx in range(len(irName)):
        print("Complexity: " + cmplxName[cmplx_idx] + " IR: " + irName[ir_idx])
        #sample minority training data
        min_idx = np.where(train_labels==cc[cmplx_idx][1])[0]
        min_idx = np.random.choice(min_idx, np.round(5000*ir[ir_idx]).astype(int))
        #sample majority training data
        maj_idx = np.where(train_labels==cc[cmplx_idx][0])[0]
        maj_idx = np.random.choice(maj_idx, 5000)
        img_cntr = 0
        if not os.path.isdir("../../data/mnistFashion/trainDatasets/"+cmplxName[cmplx_idx]+"/"+cmplxName[cmplx_idx]+"_"+irName[ir_idx]+"/class1/"):
            os.makedirs("../../data/mnistFashion/trainDatasets/"+cmplxName[cmplx_idx]+"/"+cmplxName[cmplx_idx]+"_"+irName[ir_idx]+"/class1/")
        for idx in min_idx:
            mpimg.imsave("../../data/mnistFashion/trainDatasets/"+cmplxName[cmplx_idx]+"/"+cmplxName[cmplx_idx]+"_"+irName[ir_idx]+"/class1/image_"+str(img_cntr)+".png", train_images[idx])
            img_cntr+=1
        if not os.path.isdir("../../data/mnistFashion/trainDatasets/"+cmplxName[cmplx_idx]+"/"+cmplxName[cmplx_idx]+"_"+irName[ir_idx]+"/class0/"):
            os.makedirs("../../data/mnistFashion/trainDatasets/"+cmplxName[cmplx_idx]+"/"+cmplxName[cmplx_idx]+"_"+irName[ir_idx]+"/class0/")
        img_cntr = 0
        for idx in maj_idx:
            mpimg.imsave("../../data/mnistFashion/trainDatasets/"+cmplxName[cmplx_idx]+"/"+cmplxName[cmplx_idx]+"_"+irName[ir_idx]+"/class0/image_"+str(img_cntr)+".png", train_images[idx])
            img_cntr+=1

# %%
#TESTING SETS: This loop creates to the file structure for the test images. The models are evalauted on balanced datasets, therefore subdirectories are
#not created for the dividual imbalance levels. The loop creates one directory for each complexity level named c1, c2,...,c5. inside each of these, 
#there is one directory for each class names class0 and class1. The majority class images are placed inside class0 and the majority class images are 
# placed in side class1.

#for each complexity level
for cmplx_idx in range(len(cmplxName)):
    print("Complexity: " + cmplxName[cmplx_idx])
    #sample minority training data
    min_idx = np.where(test_labels==cc[cmplx_idx][1])[0]
    min_idx = np.random.choice(min_idx, 5000)
    #sample majority training data
    maj_idx = np.where(test_labels==cc[cmplx_idx][0])[0]
    maj_idx = np.random.choice(maj_idx, 5000)
    img_cntr = 0
    if not os.path.isdir("../../data/mnistFashion/testDatasets/"+cmplxName[cmplx_idx]+"/class1/"):
        os.makedirs("../../data/mnistFashion/testDatasets/"+cmplxName[cmplx_idx]+"/class1/")
    for idx in min_idx:
        mpimg.imsave("../../data/mnistFashion/testDatasets/"+cmplxName[cmplx_idx]+"/class1/image_"+str(img_cntr)+".png", test_images[idx])
        img_cntr+=1
    if not os.path.isdir("../../data/mnistFashion/testDatasets/"+cmplxName[cmplx_idx]+"/class0/"):
        os.makedirs("../../data/mnistFashion/testDatasets/"+cmplxName[cmplx_idx]+"/class0/")
    img_cntr = 0
    for idx in maj_idx:
        mpimg.imsave("../../data/mnistFashion/testDatasets/"+cmplxName[cmplx_idx]+"/class0/image_"+str(img_cntr)+".png", test_images[idx])
        img_cntr+=1

