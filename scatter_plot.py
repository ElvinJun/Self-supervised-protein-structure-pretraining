#!/opt/anaconda3/bin/python

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import matplotlib as mpl
mpl.use('Agg')

parser = argparse.ArgumentParser(description='Draw and save a scatter plot on a test set.')
parser.add_argument('-t', '--task', help='Task name, e.g. gfp/protherm/rocklin', required=True)
parser.add_argument('-i', '--index', help='Train index, e.g. 20040401', required=True)
parser.add_argument('-m', '--model', help='Model index, e.g. 969', required=True)
parser.add_argument('-s', '--no_show', help='Do not show plot on the screen. Only save', action='store_true')
parser.add_argument('-fa', '--factor', default=1)
parser.add_argument('-p', '--point_size', default=1)
args = parser.parse_args()

task_name = args.task
train_index = args.index
model_index = args.model
test_results_path = '/home/caiyi/test_results/'
test_result_filename = '%s_%s_%s_test.npy' % (task_name, train_index, model_index)
test_result_path = os.path.join(test_results_path, test_result_filename)
# FOR TESTING
# test_result_path = '/home/caiyi/outputs/rocklin__20071504-0/val/_124/pred_and_real.npy'

a = np.load(test_result_path, allow_pickle=True)
# a = [a]
pred = a[0][:len(a[0]) // 2] * float(args.factor)
real = a[0][len(a[0]) // 2:]
plt.rcParams['figure.dpi'] = 150
point_size = float(args.point_size)
plt.scatter(real, pred, s=point_size)
plt.scatter(real, real, s=point_size)
plt.xlabel('real value', fontsize=12)
plt.ylabel('predict value', fontsize=12)

fig_filename = '%s_%s_%s_scatter.png' % (task_name, train_index, model_index)
fig_path = os.path.join(test_results_path, 'fig', fig_filename)
plt.savefig(fig_path)
print(pearsonr(real, pred))
print(spearmanr(real, pred))
if not args.no_show:
    os.system('display %s' % fig_path)

