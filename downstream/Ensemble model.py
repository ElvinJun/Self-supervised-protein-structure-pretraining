import numpy as np
def filter(repre, fold):
    numpy = np.load('D://linux_test_data/gfp_recall/npy/' + repre + '/gfp_triple_' + str(fold) + '_3more.npy', allow_pickle=True)
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
    return real_sort, dic_pred

r = 0
number = 0
n = 5
tmp_total = [0.0] * 256
multi_seq = []
for i in range(n):    
    real_sort, pred_spsr = filter('spsr', i)
    real_sort, pred_tape = filter('Tape', i)
    real_sort, pred_unirep = filter('Unirep', i)

    dic_average = {}
    for key in pred_spsr:
        dic_average[key] = pred_spsr[key] * 0/2 + pred_tape[key] * 1/2 + pred_unirep[key] * 1/2
    pred_average = sorted(dic_average.items(), key=lambda x:x[1], reverse = True) 

    tmp_average = []
    for real in real_sort[:256]:
        real = real[0]
        count = 0
        for v in pred_average:
            if v[0] == real:
                tmp_average.append(count)
                break
            count += 1

    number += tmp_average[0]
    r += np.mean(tmp_average)
    print(np.mean(tmp_average))
    
    for i in range(256):
        tmp_total[i] += tmp_average[i] 
     
for i in range(len(tmp_total)):
    multi_seq.append(tmp_total[i]/5)
print(r/n, number/n)
print(multi_seq)
