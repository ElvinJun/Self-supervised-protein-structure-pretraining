#!/opt/anaconda3/bin/python

"""
train.py
Train a downstream MLP model or finetune a DeepPBS model
Arguments:
-p, --param: Path of parameter file
-o, --override: Override existing output files

TODO:
2. default config to a task, warning when disobey
4. See network.py TODO
5. model fusion
6. Read EricZhang's code to improve
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pathlib
import os
import shutil
import time
import numpy as np
import argparse
import json
import platform
from scipy.stats import pearsonr
from scipy.stats import spearmanr

from dataset import get_class_name
from network import network_dict
import deeppbs

test_set_p = False

# Constants for fine-tuning
dataset_dict = {'knn_60': 'DistanceWindow', 'knn_75': 'KNN2tor', 'image': 'Image2tor',
                'knn_75_batch': 'KNN2tor_batch'}
block_list_dict = {'knn_60': None, 'knn_75': ['MLP', 'LSTM', '2HidSig'], 'image': ['ConvOld', 'LSTM', '2HidSig'],
                   'knn_75_batch': ['MLP', 'BatchLSTM', 'BatchJoint']}
dims_dict = {'knn_60': None, 'knn_75': [75, 1024, 1024, 4], 'image': [5, 512, 512, 4],
             'knn_75_batch': [75, 1024, 1024, 4]}

# Load config file
default_param_file = '/home/joseph/KNN/down_pred/parameters.json'
parser = argparse.ArgumentParser(description='Train a downstream MLP model or finetune a DeepPBS model')
parser.add_argument('-p', '--param', help='Path of parameter file', default=default_param_file)
parser.add_argument('-o', '--override', help='Override existing output files', action="store_true")
parser.add_argument('-c', '--cv', help='Index of part in a cross validation', default=None)
parser.add_argument('-g', '--gpu', help='Index of GPU to use, overriding the parameter in config file. Only use this argument in a cross validation', default=None)

args = parser.parse_args()

override = args.override
param_file_path = args.param
param_file = open(param_file_path, 'r')
p = json.load(param_file)

# Set CUDA
if not args.gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = p['gpu']
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

device_num = p['device_num']
torch.cuda.set_device(device_num)
device = torch.device('cuda:%d' % device_num)

# Determine whether the run is fine-tuning or a downstream network training
finetune = False
if p['network'].startswith('ft_'):
    finetune = True

# Get paths of real value file, dataset list file and dataset folder
data_path = p[platform.system() + '_data_path']
output_path = p[platform.system() + '_output_path']
train_index = p['training_num']
task_name = p['task_name']
task_path = os.path.join(data_path, task_name)
dataset_names = p['dataset_names']
sub_dataset_names = p['sub_dataset_names']
main_dataset = dataset_names[0]
real_value_path = os.path.join(task_path, p['real_value_path'])
real_value_column = p['real_value_column']    # starts at 0
if not args.cv:
    train_set_list_path = os.path.join(task_path, main_dataset, p['train_set_list_path'])
    valid_set_list_path = os.path.join(task_path, main_dataset, p['valid_set_list_path'])
    print(train_set_list_path)
else:
    train_set_list_path = os.path.join(task_path, main_dataset, f'{p["train_set_list_path"]}_{args.cv}.txt')
    valid_set_list_path = os.path.join(task_path, main_dataset, f'{p["valid_set_list_path"]}_{args.cv}.txt')
train_set_paths = {dataset_names[i]: os.path.join(task_path, dataset_names[i], sub_dataset_names[i], p['train_set_paths'][i])
                   for i in range(len(dataset_names))}
valid_set_paths = {dataset_names[i]: os.path.join(task_path, dataset_names[i], sub_dataset_names[i], p['valid_set_paths'][i])
                   for i in range(len(dataset_names))}

# Load real value file
file = open(real_value_path, 'r')
lines = file.readlines()
file.close()
dic = {}
for line in lines:
    dic[line.split()[0]] = float(line.split()[real_value_column])

# Create (and load parameter of) a model and import dataset class
pretrained_model_path = os.path.join(p['pretrained_models_path'], p['loaded_pretrained_model'])
downstream_model_path = p['loaded_downstream_model']
if finetune:
    ft_dataset_name = p['network'][3:]
    block_list = block_list_dict[ft_dataset_name]
    dims = dims_dict[ft_dataset_name]
    exec('Dataset = deeppbs.%s' % dataset_dict[ft_dataset_name])
    model_blocks = {'local': block_list[0], 'global': block_list[1], 'predict': block_list[2]}
    # rocklin任务使用DeepPBSMean, gfp任务使用DeepPBS
    model = deeppbs.DeepPBSMean(blocks=model_blocks, dims=dims).to(device)   # 注意此处to(device)是否报错
    model.load_model(model_path=pretrained_model_path, blocks=['local', 'global'])
    if downstream_model_path:
        model.load_model(model_path=downstream_model_path, blocks=['predict'])
else:
    class_name, input_shape = get_class_name(args.param)
    print('Dataset:', class_name, 'Shape:', input_shape)
    exec('from dataset import %s as Dataset' % class_name)
    exec('from network import %s as MLP' % network_dict[p['network']])
    model = MLP(input_dim=768, stacked_dim=input_shape[0]).to(device)
    # TODO: change to load state_dict
    if downstream_model_path:
        model = torch.load('/home/caiyi/outputs/rocklin__20053101/404_Linear.pth', map_location='cuda:0')

# Create datasets and data loaders
train_dataset = Dataset(train_set_paths, list_path=train_set_list_path)
valid_dataset = Dataset(valid_set_paths, list_path=valid_set_list_path)
valid_set_size = len(valid_dataset)
train_set_size = len(train_dataset)
# DataLoader is a iterator, which generates the data of a batch accroding to batch_size
train_set_loader = DataLoader(dataset=train_dataset, shuffle=True)
valid_set_loader = DataLoader(dataset=valid_dataset, pin_memory=True)

# IF SHOW TEST SET RESULTS:
if test_set_p:
    test_set_list_path = os.path.join(task_path, main_dataset, p['test_set_list_path'])
    test_set_paths = {dataset_names[i]: os.path.join(task_path, dataset_names[i], sub_dataset_names[i], p['test_set_paths'][i])
                       for i in range(len(dataset_names))}
    test_dataset = Dataset(test_set_paths, list_path=test_set_list_path)
    test_set_size = len(test_dataset)
    test_set_loader = DataLoader(dataset=test_dataset, pin_memory=True)

# Load training parameters
auto_end = p['auto_end']
criterion = p['criterion_to_end']
num_performs = p['epoches_to_end']
th_converg = p['threshold_of_convergence']
target_epochs = p['target_epochs']
batch_size = p['batch_size']
learning_rate = p['learning_rate']
reg_param = p['regularization_param']   # 1e-4 * 1 / 3
exec('loss_function = %s()' % p['loss_function'])
exec('optimizer = torch.optim.%s(model.parameters(), lr=learning_rate, weight_decay=0.00001)' % p['optimizer'])

# Test whether CUDA is available
if torch.cuda.is_available():
    print('GPU available!!!')
    print('MainDevice=', device)

# Initiate training
total_iters = 0
epoch = 0
prev_performs = {'r': [], 'rho': [], 'val_loss': []}
# Create output folder
if not args.cv:
    train_name = f'{task_name}__{train_index}'
else:
    train_name = f'{task_name}__{train_index}-{args.cv}'
save_path = os.path.join(output_path, train_name)
if not override and pathlib.Path(save_path).exists():
    raise Exception('Train name already exists! Not allowing overriding!')
val_dir = os.path.join(save_path, 'val')
for folder in [save_path, val_dir]:
    pathlib.Path(folder).mkdir(parents=True, exist_ok=True)
# Copy config file to the output path to save it
shutil.copy(param_file_path, os.path.join(save_path, 'parameters.json'))
if args.cv:
    shutil.copy('/home/joseph/KNN/down_pred/cv.sh', os.path.join(save_path, 'cv.sh'))

print(model)

def validation(prev_performs, test_set=False):
    end_p = False
    valid_pred_values = np.zeros(valid_set_size)
    valid_real_values = np.zeros(valid_set_size)
    with torch.no_grad():
        # WHETHER TO SHOW TEST SET RESULTS
        if not test_set:
            valid_loss_file = open(os.path.join(save_path, 'valid_loss.txt'), 'a')
            dataloader = valid_set_loader
        else:
            valid_loss_file = open(os.path.join(save_path, 'test_loss.txt'), 'a')
            dataloader = test_set_loader
        valid_loss_file.write('epoch %d\n' % epoch)
        valid_losses = []
        valid_output_path = os.path.join(val_dir, '_%d' % epoch)
        pathlib.Path(valid_output_path).mkdir(parents=True, exist_ok=True)

        count = 0

        for valid_array, valid_filename in dataloader:
            valid_array = valid_array.to(device)
            valid_filename = valid_filename[0].split('.np')[0]
            valid_array = valid_array.float()   # 加结构以后报错，因此加此行

            valid_pred_value = model(valid_array[0])
            valid_pred_value = valid_pred_value.float()
            valid_real_value = torch.tensor(np.array(dic[valid_filename]), dtype=torch.float, device=device).float()
            valid_loss = loss_function(valid_pred_value, valid_real_value) # \
                # + reg_param * torch.sum(abs(w_val))

            valid_losses.append(float(valid_loss))
            # print(valid_pred_value.dtype, valid_pred_values.dtype, valid_pred_value.shape, valid_pred_values.shape)
            valid_pred_values[count] = valid_pred_value.data.cpu().numpy()
            valid_real_values[count] = dic[valid_filename]
            count += 1

        valid_output = np.concatenate((valid_pred_values, valid_real_values), 0).astype('float32')
        np.save(os.path.join(valid_output_path, 'pred_and_real'), valid_output)
        pearson = pearsonr(valid_pred_values, valid_real_values)
        spearman = spearmanr(valid_pred_values, valid_real_values)

        valid_loss_avg = sum(valid_losses) / valid_set_size
        valid_loss_file.write('val_loss_average = %f\n' % valid_loss_avg)
        valid_loss_file.write('Pearson_number = %f\n' % float(pearson[0]))
        valid_loss_file.write('Spearman_number = %f\n' % float(spearman[0]))
        valid_loss_file.close()
        if test_set:
            print('TEST SET RESULTS:')
        print('Pearson_number = %f' % float(pearson[0]))
        print('Spearman_number = %f' % float(spearman[0]))
        print('val_loss_average = %f' % valid_loss_avg)
        if not test_set:
            prev_performs['r'].append(pearson[0])
            prev_performs['rho'].append(spearman[0])
            prev_performs['val_loss'].append(-valid_loss_avg)
            performs = prev_performs[criterion]
            if len(performs) > num_performs:
                prev_perform = performs.pop(0)
                end_p = (prev_perform > max(performs) - th_converg)
                # print(prev_perform, performs, max(performs) - th_converg)
                # print("END=",end_p)
        return end_p


# Start training
while True:
    epoch_start_time = time.time()
    epoch_iter = 0
    losses = []
    print('epoch', epoch)
    j = 0

    # validation()

    for train_array, filename in train_set_loader:
        file_name = filename[0].split('.np')[0]
        train_array = train_array.to(device)
        train_array = train_array.float()   # 加结构以后报错，因此加此行
        predict_value = model(train_array[0])
        predict_value = predict_value.float()   # 转换为tensor
        real_value = torch.tensor(np.array(dic[file_name]), dtype=torch.float, device=device)
        train_loss = loss_function(predict_value, real_value) # + reg_param * torch.sum(abs(w))
        losses.append(float(train_loss))

        total_iters += batch_size
        epoch_iter += batch_size

        train_loss.backward()
        if (j + 1) % 8 == 0:
            optimizer.step()
            optimizer.zero_grad()

        j += 1

        if total_iters % 200 == 0:
            loss_average = sum(losses) / len(losses)

            print('iters', total_iters, '    mean_loss=', loss_average)
            with open(os.path.join(save_path, 'train_loss.txt'), 'a') as train_loss_file:
                train_loss_file.write('iter=%s  losses_average=%s\n' % (total_iters, loss_average))

        if total_iters % 1000 == 0:
            last_linear_path = os.path.join(save_path, 'last_Linear.pth')
            torch.save(model, last_linear_path)

        if total_iters % train_set_size == 0:
            save_linear_path = os.path.join(save_path, '%s_Linear.pth' % epoch)
            torch.save(model, save_linear_path)

            model.eval()
            model.is_training = False
            end_p = validation(prev_performs=prev_performs)
            if test_set_p:
                validation(prev_performs=None)
            model.train()

    print('End of epoch %d  \t Time Taken: %d sec' % (epoch, time.time() - epoch_start_time))
    epoch += 1
    if auto_end and end_p:
        break
    if not auto_end and epoch >= target_epochs:
        break

