import numpy as np
import os
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris,load_digits
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#hyper-parameters
topology = 'villin'
threshold = 1.08

target = []
path = 'D:\\linux_test_data\\stability_embed\\self_elmo_net_2_3_768_finetune'
count = 0
c = np.array(len(files) * [[0.0] * 512])
count = 0
for file in os.listdir(path):
    key = file[:-4]
    file1 = str.lower(file)
    t = file.split('.')[0]
    if file1 in files and str.lower(t[:len(topology)]) == str.lower(topology):
        a = np.load(os.path.join(path, file))
        a =np.array([np.mean(a, 0)])
        c[count] = a
        
        if t[:6] == 'villin' or str.lower(t[:4]) == 'pin1' or t[:6] == 'hYAP65':
            if t[:6] == 'villin':
                if float(dic_value[key]) > 1.08:
                    target.append(5)
                else:
                    target.append(0)
            elif str.lower(t[:4])== 'pin1':
                if float(dic_value[key]) > 1.17:
                    target.append(5)
                else:
                    target.append(0)
            elif t[:6] == 'hYAP65':
                if float(dic_value[key]) > 0.84:
                    target.append(5)
                else:
                    target.append(0)
                    
        else:
            if float(dic_value[key]) > threshold:
                target.append(5)
            else:
                target.append(0)

        if count % 1000 == 0:
            print(count)
        count += 1 
data = c[:count]
print(data.shape, count)
print(len(target))
#t-sne / plt
for i in range(36,37):
    print(i)
    X_tsne = TSNE(n_components=2,random_state=i).fit_transform(data)
    plt.figure(figsize=(10, 5))
    plt.rcParams['savefig.dpi'] = 200 #
    plt.rcParams['figure.dpi'] = 200 #
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], s = 5, alpha=3, c = target, cmap='Paired')
    plt.colorbar()
    plt.legend()
