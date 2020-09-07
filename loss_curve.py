#!/opt/anaconda3/bin/python

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt 
from scipy.stats import pearsonr, spearmanr
import matplotlib as mpl
mpl.use('Agg')

parser = argparse.ArgumentParser(description='Draw and save training and validating loss curves.')
parser.add_argument('-t', '--task', help='Task name, e.g. gfp/protherm/rocklin', required=True)
parser.add_argument('-i', '--index', help='Train index, e.g. 20040401', required=True)
parser.add_argument('-s', '--no_show', help='Do not show plot on the screen. Only save', action='store_true')
args = parser.parse_args()

task_name = args.task
train_index = args.index
outputs_path = '/home/caiyi/workspace/rgn/outputs/'
output_path = os.path.join(outputs_path, '%s__%s' % (task_name, train_index))
train_loss_path = os.path.join(output_path, 'train_loss.txt2')
valid_loss_path = os.path.join(output_path, 'valid_loss.txt')
figs_path = '/home/caiyi/outputs/fig/'
fig_path = os.path.join(figs_path, '%s_%s_loss_curve.png' % (task_name, train_index))
# fig_path = "log"

loss_train = []
loss_valid = []
pearson = []
spearman = []
file = open(train_loss_path, 'r')
file1 = open(valid_loss_path, 'r')
lines = file.readlines()
lines1 = file1.readlines()
file.close()
file1.close()
for line in lines:
	loss_train.append(float(line.split('age=')[1].strip()))
for line in lines1:
	line = line.split(' = ')
	if line[0] == 'val_loss_average': loss_valid.append(float(line[1]))
	#if line[0] == 'Pearson_number': pearson.append(float(line[1]))
	#if line[0] == 'Spearman_number': spearman.append(float(line[1]))

print(loss_train[1])
print(len(loss_valid))
# print(pearson[1], spearman[1])

num_train_loss_per_epoch = len(loss_train) // len(loss_valid)
average = []
loss_ = []
for i, loss_iter in enumerate(loss_train):
	average.append(loss_iter)
	if i % num_train_loss_per_epoch == 0 and i != 0:
		loss_.append(sum(average) / num_train_loss_per_epoch)
		average = []

fig, ax = plt.subplots(figsize=(10, 6)) 
# plt.rcParams['savefig.dpi'] = 500  # 图片像素
plt.rcParams['figure.dpi'] = 150  # 分辨率
ax.plot(np.arange(len(loss_)), loss_, label='train loss')
ax.plot(np.arange(len(loss_valid)), loss_valid, label='valid loss')
plt.xlabel('epochs', fontsize=15)
plt.ylabel('loss', fontsize=15)
plt.legend(loc=1, fontsize=15)

plt.savefig(fig_path)
if not args.no_show:
    os.system('display %s' % fig_path)

