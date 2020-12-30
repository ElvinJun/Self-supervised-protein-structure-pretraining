def metrics(dic_real, dic_pred):
    tp = tn = fp = fn = 0
    for key in dic_real:
        if dic_real[key] >= 3 and dic_pred[key] >= 3:
            tp += 1
        elif dic_real[key] >= 3 and dic_pred[key] < 3:
            fn += 1
        elif dic_real[key] <= 3 and dic_pred[key] <= 3:
            tn += 1
        elif dic_real[key] < 3 and dic_pred[key] > 3:
            fp += 1
#     print(tp, tn, fp, fn)
#     正类召回 正类准确 负类召回 负类准确
#     print(tp / (tp+fn), tp/(tp+fp), tn/(tn+fp), tn/(tn+fn))

    #ACC
    return (tp + tn) / (tp + fn + tn + fp)
    

import numpy as np
PtsRep_triple = []
n = 256
tmp_total = [0.0] * n

#5-fold cross validation
for fold in range(0,5):
    numpy = np.load('D://linux_test_data/gfp_recall/npy/unirep/gfp_triple_' + str(fold) + '_3more.npy', allow_pickle=True)
    files = numpy[1]
    value = numpy[0]
    dic_real = {}
    dic_pred = {}
    tmp = []
    for i in range(len(files)):
        dic_pred[files[i]] = value[i]
        dic_real[files[i]] = value[i + 25517]
    real_sort = sorted(dic_real.items(), key=lambda x:x[1], reverse = True)
    pred_sort = sorted(dic_pred.items(), key=lambda x:x[1], reverse = True) 
    acc = metrics(dic_real, dic_pred)
#     print('Accuracy '+ str(fold) + ':', acc)
    
    #Recall statistics
    for real in real_sort[:n]:
        real = real[0]
        count = 0
        for v in pred_sort:
            if v[0] == real:
                tmp.append(count)
                break
            count += 1
    for i in range(n):
        tmp_total[i] += tmp[i] 
 
for i in range(len(tmp_total)):
    PtsRep_triple.append(tmp_total[i]/5)
print(PtsRep_triple)
