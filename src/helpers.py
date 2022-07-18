

def collectResults(trnResList):
    metrics = ['loss', 'gm', 'auc', 'prc', 'tp']
    trnSelctResList = []
    valSelctResList = []
    for _ in metrics:
        trnSelctResList.append([])
        valSelctResList.append([])
    for i in range(len(trnResList)):
        for j in range(len(metrics)):
            trnSelctResList[j].append(trnResList[i].history[metrics[j]])
            valSelctResList[j].append(trnResList[i].history["val_"+metrics[j]])
    return trnSelctResList, valSelctResList

# mean of irregular length list of lists
def listMean(lstResults):
    list_len = [len(i) for i in lstResults]
    mxLen = max(list_len)
    mnList = []
    stdList = []
    numList = len(lstResults)
    for i in range(mxLen):
        tmpVal = []
        for j in range(5):
            if len(lstResults[j]) > i:
                tmpVal.append(lstResults[j][i])
        mnList.append(np.mean(tmpVal))
        stdList.append(np.std(tmpVal))
    return mnList, stdList
