# %%
import numpy as np
import pickle
import matplotlib.pyplot as plt
from pandas.tseries import offsets

CMPLX = "c1"
CMPLX_INT = 1
IMB_LVs = ["5","4","3","2","1"]
MLD_DVs = [1, 2, 3, 4, 5]
DO_DP = [True, False]
DO_RLROP = [True, False] #ReduceLROnPlateau
DO_EARLY_STOP = [True, False]
# PATH = '../results/shapes/cnnShapesModel_'
PATH = '../results/mnistFashion/cnnModel_'
# PATH = '../results/cifar10/cnnModel_'
HU = 10  
EP = 100  

rsltStr = []
dataAll = []   

i = 0
for imblv in IMB_LVs: 
    print(i)                                                                                                    
    #for each mode depth  
    mld_str = PATH+'Ep_DO_ERLSTP_True'+str(EP)+'_Hu'+str(HU)+'_Imblv_'+str(imblv)+"_cmplx_"+str(CMPLX)+'.pkl'
    print(mld_str)     
    tmpLst = pickle.load( open(mld_str, "rb" ) )
    i+=1                    
    dataAll.append(tmpLst)


# %%
  #for each imbalanced level in datasets with complexity C            
AUC_Mean = []
GM_Mean = []
PRC_Mean = []
prec_Mean = []
rec_Mean = []  
AUC_Std = []  
GM_Std = []  
PRC_Std = []  
prec_Std = []  
rec_Std = [] 

metaData = []
for i in range(len(IMB_LVs)):
    k=0
    print("imbalance level " + str(IMB_LVs[i]))          
    print(k)                                                                                                                            
    for mldlv in MLD_DVs:                                                           
        for dp in DO_DP:                                                             
            for rlrop in DO_RLROP:
                for earlStp in DO_EARLY_STOP:
                    tmpAuc = []
                    tmpGm = []
                    tmpPrc = []
                    tmpPrec = []
                    tmpRec = []
                    metaData.append([CMPLX_INT, int(IMB_LVs[i]), mldlv, dp, rlrop, earlStp])
                    print("model depth " + str(mldlv) + " dropout " + str(dp) + " reduce learning rate " + str(rlrop) + " early stopping " + str(earlStp)) 
                    print(k)
                    for j in range(5):                                                           
                        tmpAuc.append(dataAll[i][k][j]['auc'])
                        tmpGm.append(dataAll[i][k][j]['gm'])
                        tmpPrc.append(dataAll[i][k][j]['prc'])
                        tmpPrec.append(dataAll[i][k][j]['precision'])
                        tmpRec.append(dataAll[i][k][j]['recall'])
                    AUC_Mean.append(np.mean(tmpAuc))
                    GM_Mean.append(np.mean(tmpGm))
                    PRC_Mean.append(np.mean(tmpPrc))
                    prec_Mean.append(np.mean(tmpPrec))
                    rec_Mean.append(np.mean(tmpRec))
                    AUC_Std.append(np.std(tmpAuc))
                    GM_Std.append(np.std(tmpGm))
                    PRC_Std.append(np.std(tmpPrc))
                    prec_Std.append(np.std(tmpPrec))
                    rec_Std.append(np.std(tmpRec))
                    k+=1

resultsNp = np.ndarray(shape=(200,0))
resultsNp = np.concatenate((resultsNp, np.array(AUC_Mean).reshape(200,1)),axis=1)
resultsNp = np.concatenate((resultsNp, np.array(AUC_Std).reshape(200,1)),axis=1)
resultsNp = np.concatenate((resultsNp, np.array(GM_Mean).reshape(200,1)),axis=1)
resultsNp = np.concatenate((resultsNp, np.array(GM_Std).reshape(200,1)),axis=1)
resultsNp = np.concatenate((resultsNp, np.array(PRC_Mean).reshape(200,1)),axis=1)
resultsNp = np.concatenate((resultsNp, np.array(PRC_Std).reshape(200,1)),axis=1)
resultsNp = np.concatenate((resultsNp, np.array(prec_Mean).reshape(200,1)),axis=1)
resultsNp = np.concatenate((resultsNp, np.array(prec_Std).reshape(200,1)),axis=1)
resultsNp = np.concatenate((resultsNp, np.array(rec_Mean).reshape(200,1)),axis=1)
resultsNp = np.concatenate((resultsNp, np.array(rec_Std).reshape(200,1)),axis=1)


# %%
metaData = np.vstack(metaData).astype(np.int64)
resultsNp = np.round(resultsNp,4)
hdr = "CMPLX,IMB_Level,Model_Depth,DP,RLROP,EarlyStop,AUC_Mean,AUC_Std,GM_Mean,GM_Std,PRC_Mean,PRC_Std,prec_Mean,prec_Std,rec_Mean,rec_Std"
np.savetxt(PATH+"Ep"+str(EP)+"_Hu"+str(HU)+"_Cmplx_"+CMPLX+".csv", np.concatenate((metaData,resultsNp),axis=1), header=hdr, delimiter=',',fmt="%i %i %i %i %i %i %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f")
np.savetxt(PATH+"MetaData_Ep"+str(EP)+"_Hu"+str(HU)+"_Cmplx_"+CMPLX+".csv", metaData, delimiter=',',fmt="%i")


# %%
