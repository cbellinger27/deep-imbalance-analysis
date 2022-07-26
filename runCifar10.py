# %%
import numpy as np
import tensorflow as tf
import os
import PIL
import PIL.Image
from src.models import  mnistFashion_cifar10_cnn as cnn_model
from src import helpers
import datetime
from copy import copy
import pathlib
import sys
import pickle
import matplotlib.pyplot as plt

tf.random.set_seed(1234)
np.random.seed(seed=1235)


#RUNNER
DATASET_PATH_TRN = "~/data/cifar10/trainDatasets/"
DATASET_PATH_TST = "~/data/cifar10/testDatasets/"

if not os.path.isdir("../results/cifar10/"):
    os.makedirs("../results/cifar10/")

CMPLX = "c1"
IMB_LVs = ["1","2","3","4","5"]
MLD_DVs = [1,2,3,4,5]
DO_DP = [True, False]
DO_RLROP = [True, False] #ReduceLROnPlateau
DO_EARLY_STOP = [True, False]
HU = 10  
EP = 100   

RSLTS =[]
RSLTS_TRN = []                   
MLD_STRS =[] 
RSLTS_ALL = []

MLD_STRS_ALL = []                                                   
RSLTS_TRN_ALL = []                                                  


#for each imbalanced level in datasets with complexity C            
for imblv in IMB_LVs:                                               
    SEED_VALUE = 1235                                               
    tf.random.set_seed(SEED_VALUE)                                  
    np.random.seed(seed=SEED_VALUE)                                 
    mld_str = "cnnCifar10Model_Imblv_"+imblv                        
    trnDirs = DATASET_PATH_TRN+CMPLX+"/"+CMPLX+"_b"+imblv           
    tstDirs = DATASET_PATH_TST+CMPLX                                
    #initial the image generator                                    
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,validation_split=0.7)                     
    #intializet the test generator                                  
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)                                           
    #for each mode depth                                            
    for mldlv in MLD_DVs:                                           
        mld_str = mld_str+"_depth_"+str(mldlv)                      
        for dp in DO_DP:                                            
            mld_str = mld_str+"_withDp_"+str(dp)                    
            for rlrop in DO_RLROP: 
                mld_str = mld_str+"_withRLROP_"+str(rlrop)   
                for earlStp in DO_EARLY_STOP:
                    mld_str = mld_str+"_withErlStp_"+str(earlStp)                              
                    SEED_VALUE = 1235                                                 
                    RSLTS =[]
                    RSLTS_TRN = []                   
                    MLD_STRS =[]                             
                    for i in range(5):                                  
                        SEED_VALUE += 345                               
                        tf.random.set_seed(SEED_VALUE)                  
                        np.random.seed(seed=SEED_VALUE)                 
                        #instance the mode of depth d                   
                        inputDim =  (48, 48, 3)                        
                        outputDim = 2                                   
                        model = cnn_model.get_modelCnnSmall(inputDim, outputDim, hidden=HU, depth=mldlv, useDp=dp)            
                        # image loaders                                
                        train_generator = train_datagen.flow_from_directory(                                                     
                            trnDirs,                                    
                            subset='training',                          
                            shuffle=True,                               
                            target_size=(48, 48),                       
                            batch_size=32,                              
                            class_mode='categorical')                     
                        val_generator = train_datagen.flow_from_directory(                                                       
                            trnDirs,                                    
                            subset='validation',                        
                            shuffle=True,                               
                            target_size=(48, 48),                       
                            batch_size=32,                              
                            class_mode='categorical')                   
                        tst_generator = test_datagen.flow_from_directory(                                                        
                            tstDirs,                                    
                            target_size=(48, 48),                       
                            batch_size=32,                              
                            class_mode='categorical')                   
                        #training the model 
                        earlyStop = None
                        reduce_lr = None
                        cbs = []
                        if earlStp:
                            earlyStop = tf.keras.callbacks.EarlyStopping(monitor='val_prc',patience=10,mode='max',restore_best_weights=True) 
                            cbs.append(earlyStop)                    
                        if rlrop:
                            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_prc', factor=0.2, patience=5, min_lr=1e-5)  
                            cbs.append(reduce_lr)             
                        history = model.fit(                        
                            train_generator,                        
                            validation_data=val_generator,          
                            epochs=EP,
                            shuffle=True,
                            callbacks=cbs)                              
                        #test the model                                 
                        ev = model.evaluate(tst_generator)              
                        RSLTS.append(dict(zip(model.metrics_names, ev)))
                        RSLTS_TRN.append(history)                       
                        MLD_STRS.append(mld_str + "_iteration_"+ str(i))
                    trnRes, valRes = helpers.collectResults(RSLTS_TRN)
                    metrics = ['loss', 'gm', 'auc', 'prc', 'tp']
                    # save AUC, PRC, GM                                  
                    RSLTS_ALL.append(RSLTS)                             
                    RSLTS_TRN_ALL.append(RSLTS_TRN)                     
                    MLD_STRS_ALL.append(MLD_STRS)
    a_file = open('results/cifar10/cnnModel_Ep_DO_ERLSTP_True'+str(EP)+'_Hu'+str(HU)+'_Imblv_'+str(imblv)+"_cmplx_"+str(CMPLX)+'.pkl', 'wb')                       
    pickle.dump(RSLTS_ALL, a_file)
    a_file.close()
    print("Done imbalance leve: " + imblv)
    RSLTS =[]
    RSLTS_TRN = []                   
    MLD_STRS =[] 
    RSLTS_ALL = []
