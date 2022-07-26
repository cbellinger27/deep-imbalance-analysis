import numpy as np
import tensorflow as tf
import os
import pandas as pd



from src.models import backbone_mlp as mlp_model
from src import helpers
import datetime
from copy import copy
import pathlib
import sys
import pickle
import matplotlib.pyplot as plt

tf.random.set_seed(1234)
np.random.seed(seed=1235)

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

if not os.path.isdir("../results/backbone/"):
    os.makedirs("../results/backbone/")

#RUNNER
DATASET_PATH_TRN = "~/data/backbone/TraditionalBackbone_size"
DATASET_PATH_TST = "~/data/backbone/TraditionalBackbone_test/"

SIZE = "5"
for CMPLX in ["5"]:
    IMB_LVs = ["5","4","3","2","1"]
    MLD_DVs = [1, 2, 3, 4, 5]
    DO_DP = [True, False]
    DO_RLROP = [True, False] #ReduceLROnPlateau
    DO_EARLY_STOP = [True, False]
    HU = 50  
    EP = 200   
    
    EARLY_STOP=[True, False]
    
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
        mld_str = "mlpTraditionalBackBoneModel_Imblv_"+imblv                        
        trnDirs = DATASET_PATH_TRN+SIZE+"/size"+SIZE+"_complex"+CMPLX+"_imbalance"+imblv+".xlsx"           
        tstDirs = DATASET_PATH_TST+"test_complex"+CMPLX+".xlsx"        
        df_trn = pd.read_excel(trnDirs)
        df_trn.drop(['Unnamed: 0'], axis=1, inplace=True)      
        X_trn = df_trn["X"]     
        Y_trn = df_trn["Y"]        
        X_trn = X_trn.to_numpy().reshape(len(X_trn),1)
        Y_trn = Y_trn.to_numpy().reshape(len(Y_trn),1)
        shuf_idx = np.random.choice(len(Y_trn),len(Y_trn))
        Ytrn_Encoded = tf.keras.utils.to_categorical(Y_trn)
        X_trn = X_trn[shuf_idx]
        Y_trn = Y_trn[shuf_idx]
        Ytrn_Encoded = Ytrn_Encoded[shuf_idx]
        df_tst = pd.read_excel(tstDirs)
        df_tst.drop(['Unnamed: 0'], axis=1, inplace=True)
        X_tst = df_tst["X"]     
        Y_tst = df_tst["Y"]        
        X_tst = X_tst.to_numpy().reshape(len(X_tst),1)
        Y_tst = Y_tst.to_numpy().reshape(len(Y_tst),1)
        Ytst_Encoded = tf.keras.utils.to_categorical(Y_tst)
        for mldlv in MLD_DVs:                                           
            mld_str = mld_str+"_depth_"+str(mldlv)                      
            for dp in DO_DP:                                            
                mld_str = mld_str+"_withDp_"+str(dp)                    
                for rlrop in DO_RLROP: 
                    mld_str = mld_str+"_withRLROP_"+str(rlrop)   
                    for earlStp in EARLY_STOP:
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
                            inputDim = (1,)                          
                            outputDim = 2                                   
                            model = mlp_model.get_mlpModel(inputDim, outputDim, hidden=HU, depth=mldlv, useDp=dp)                  
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
                                X_trn, Ytrn_Encoded, batch_size=64,       
                                epochs=EP, shuffle=True,  validation_split=0.75,
                                callbacks=cbs)                              
                            #test the model                                 
                            ev = model.evaluate(X_tst, Ytst_Encoded)              
                            RSLTS.append(dict(zip(model.metrics_names, ev)))
                            RSLTS_TRN.append(history)                      
                            MLD_STRS.append(mld_str + "_iteration_"+ str(i))
                            trnRes, valRes = helpers.collectResults(RSLTS_TRN)
                        # save AUC, PRC, GM                                  
                        RSLTS_ALL.append(RSLTS)                             
                        RSLTS_TRN_ALL.append(RSLTS_TRN)                     
                        MLD_STRS_ALL.append(MLD_STRS)
        mld_str= 'results/backbone/mlpTraditionalBackBoneModelSize'+str(SIZE)+'_Ep_DO_ERLSTP_True'+str(EP)+'_Hu'+str(HU)+'_Imblv_'+str(imblv)+"_cmplx_"+str(CMPLX)+'.pkl'
        tmpLst = RSLTS_ALL
        a_file = open(mld_str, 'wb')                     
        pickle.dump(tmpLst, a_file)
        a_file.close()
        RSLTS =[]
        RSLTS_TRN = []                   
        MLD_STRS =[] 
        RSLTS_ALL = []
        print("Done imbalance leve: " + imblv)
