#!/opt/anaconda3/bin/python

import os
import numpy as np
import argparse
from scipy.stats import pearsonr, spearmanr
from sklearn import metrics
import torch
import torch.nn as nn

parser = argparse.ArgumentParser(description='Binary classification on a test set.')
parser.add_argument('-t', '--task', help='Task name, e.g. gfp/protherm/rocklin', required=True)
parser.add_argument('-i', '--index', help='Train index, e.g. 20040401', required=True)
parser.add_argument('-m', '--model', help='Model index, e.g. 969')
parser.add_argument('-th', '--threshold', help='Threshold for classification, usually 2.5 for gfp', required=True)
parser.add_argument('-f', '--file', help='Filename')

args = parser.parse_args()

task_name = args.task
train_index = args.index
model_index = args.model
filename = args.file
test_results_path = '/home/caiyi/test_results/'
all_filenames = os.listdir(test_results_path)
filenames = [name for name in all_filenames if name.startswith(f'{task_name}_{train_index}')]

# test_result_filename = '%s_%s_%s_test.npy' % (task_name, train_index, model_index)
test_result_paths = [os.path.join(test_results_path, filename) for filename in filenames]
if filename:
    test_result_paths = [filename]
threshold = float(args.threshold)


def classify(test_result_path):
    task_name, train_index, model_index, test_set = test_result_path.split('/')[-1].split('_')[:4]
    a = np.load(test_result_path, allow_pickle=True)
    pred = a[0][:len(a[0])//2]
    real = a[0][len(a[0])//2:]
    pred_bin = []
    real_bin = []
    for i in pred:
        if i >= threshold:
            pred_bin.append(1)
        else:
            pred_bin.append(0)
    for i in real:
        if i >= threshold:
            real_bin.append(1)
        else:
            real_bin.append(0)

    pearson_number = pearsonr(pred, real)
    print(test_result_path)
    print(pearson_number)
    pred_tensor = torch.tensor(pred)
    real_tensor = torch.tensor(real)
    mse = nn.MSELoss()(pred_tensor, real_tensor)
    print('MSE =', mse)

    TP = FP = TN = FN = 0
    for i in range(len(real)):
        if pred[i] > threshold and real[i] > threshold:
            TP += 1
        if pred[i] < threshold < real[i]:
            FN += 1
        if pred[i] > threshold > real[i]:
            FP += 1
        if pred[i] < threshold and real[i] < threshold:
            TN += 1

    print("TP=%d" % TP, "TN=%d" % TN, "FP=%d" % FP, "FN=%d" % FN) 
    TP += 1e-9; TN += 1e-9; FP += 1e-9; FN += 1e-9
    acc = (TP + TN) / (TP + TN + FP + FN)
    acc_t = TP / (TP + FP)
    acc_f = TN / (TN + FN)
    recall_t = TP / (TP + FN)
    recall_f = TN / (TN + FP)
    print("Acc = %.3f" % acc)
    print("Acc_T = %.3f  " % acc_t, "Acc_F = %.3f" % acc_f)
    print("Recall_T = %.3f  " % recall_t, "Recall_F = %.3f" % recall_f)

    fpr, tpr, th = metrics.roc_curve(real_bin, pred_bin)
    auc = metrics.auc(fpr, tpr)
    print("AUC = %.3f" % auc)
    print()

    with open(f'{test_results_path}/classify.txt', 'a') as output_file:
        output_file.write(f'{train_index}\t{task_name}\t{model_index}\t{mse:.3f}\t{acc:.3f}\t{auc:.3f}\t{acc_t:.3f}\t{acc_f:.3f}\t{recall_t:.3f}\t{recall_f:.3f}\t{test_set}\n')


if __name__ == '__main__':
    for path in test_result_paths:
        classify(path)

