"""
KeyError: '1N0J_1'
'real_value_path' of parameters.json in outputs dir only included train and valid set but didn't include test set.
TODO:
1. custom model and dataset
2. model fusion (average)
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import numpy as np
import json
import platform
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
from scipy.stats import spearmanr

from network import network_dict
from dataset import get_class_name
import deeppbs
import nn_modules

dataset_dict = {'knn_60': 'DistanceWindow', 'knn_75': 'KNN2tor', 'image': 'Image2tor',
                'knn_75_batch': 'KNN2tor_batch'}
block_list_dict = {'knn_60': None, 'knn_75': ['MLP', 'LSTM', '2HidSig'], 'image': ['ConvOld', 'LSTM', '2HidRelu'],
                   'knn_75_batch': ['MLP', 'BatchLSTM', 'BatchJoint']}
dims_dict = {'knn_60': None, 'knn_75': [75, 1024, 1024, 4], 'image': [5, 512, 512, 4],
             'knn_75_batch': [75, 1024, 1024, 4]}


def predict_test_set(task_name, train_index, models_index, gpu='0', att=False, cv=None, custom_test_set=None):
    param_file_path = '/home/caiyi/workspace/parameters.json'
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu

    # Load saved config file
    with open(param_file_path, 'r') as param_file:
        p0 = json.load(param_file)
    results_path = p0[platform.system() + '_output_path']
    test_outputs_path = p0[platform.system() + '_test_result_path']
    models_path = os.path.join(results_path, '%s__%s' % (task_name, train_index))
    saved_param_file_path = os.path.join(models_path, 'parameters.json')
    with open(saved_param_file_path, 'r') as saved_param_file:
        p = json.load(saved_param_file)

    # Set CUDA. Controversial
    device_num = p['device_num']
    device = torch.device('cuda:%d' % device_num)

    # Get paths of real value file, dataset list file and dataset folder
    data_path = p[platform.system() + '_data_path']
    task_path = os.path.join(data_path, task_name)
    dataset_names = p['dataset_names']
    if 'sub_dataset_names' in p:
        sub_dataset_names = p['sub_dataset_names']
    else:
        sub_dataset_names = ''
    main_dataset = dataset_names[0]
    real_value_path = os.path.join(task_path, p['real_value_path'])
    real_value_column = p['real_value_column']    # starts at 0
    if cv:
        test_set_list_path = os.path.join(task_path, main_dataset, f"{p['test_set_list_path']}_{cv}.txt")
    elif custom_test_set:
        test_set_list_path = custom_test_set
    else:
        test_set_list_path = os.path.join(task_path, main_dataset, p['test_set_list_path'])

    test_set_path = {dataset_names[i]: os.path.join(task_path, dataset_names[i], sub_dataset_names[i], p['test_set_paths'][i])
                     for i in range(len(dataset_names))}

    # Paths of the model to load and the test output file
    models_path = [os.path.join(models_path, str(idx) + '_Linear.pth') for idx in models_index]
    models_index = [str(idx) for idx in models_index]
    models_index_str = '-'.join(models_index) 
    # output_path = os.path.join(models_path, 'test_%d.npy' % model_index)
    output_path = os.path.join(test_outputs_path, '%s_%s_%s_test.npy' % (task_name, train_index, models_index_str))
    if custom_test_set:
        test_set_name = custom_test_set.split('/')[-1].split('.')[0]
        output_path = os.path.join(test_outputs_path, '%s_%s_%s_%s_test.npy' % (task_name, train_index, models_index_str, test_set_name))

    # Load real values
    with open(real_value_path, 'r') as real_value_file:
        lines = real_value_file.readlines()
    dic = {}
    for line in lines:
        dic[line.split()[0]] = float(line.split()[real_value_column])

    # import network and dataset class 
    if 'network' in p:
        network_name = p['network']
    else:
        network_name = 'our'
    if not p['network'].startswith('ft_'):
        class_name, input_shape = get_class_name(saved_param_file_path)
        print(class_name, input_shape)

        exec('from network import %s as MLP' % network_dict[p['network']])
        MLP1 = locals()['MLP']
        models = [MLP1(input_dim=input_shape[1], stacked_dim=input_shape[0]).to(device) for _ in range(len(models_path))]
        loaded_states = [torch.load(path, map_location='cuda:0').state_dict() for path in models_path]
        for i in range(len(models)):
            models[i].load_state_dict(loaded_states[i])
        # model = torch.load(model_path, map_location='cuda:0')
        exec('from dataset import %s as Dataset' % class_name)
        Dataset1 = locals()['Dataset']  # Why "name 'Dataset' is not defined"?
    else:
        ft_dataset_name = p['network'][3:]
        block_list = block_list_dict[ft_dataset_name]
        dims = dims_dict[ft_dataset_name]
        exec('Dataset = deeppbs.%s' % dataset_dict[ft_dataset_name])
        Dataset1 = locals()['Dataset']
        model_blocks = {'local': block_list[0], 'global': block_list[1], 'predict': block_list[2]}
        # DeepPBSMean for rocklin task, DeepPBS for gfp task
        models = [deeppbs.DeepPBS(blocks=model_blocks, dims=dims).to(device) for _ in range(len(models_path))]
        loaded_states = [torch.load(path, map_location='cuda:0').state_dict() for path in models_path]
        for i in range(len(models)):
            models[i].load_state_dict(loaded_states[i])

    test_dataset = Dataset1(test_set_path, list_path=test_set_list_path)
    test_set_size = len(test_dataset)
    data_loader = DataLoader(dataset=test_dataset, shuffle=False)

    pred_brightness_val = np.zeros(test_set_size)
    brightness_val_real = np.zeros(test_set_size)
    names = [None for _ in range(test_set_size)]

    print(f'Train index: {train_index}\nTask name: {task_name}')
    print(f'Dataset name: {dataset_names}')
    print(f'Network: {network_name}')
    if p['combining_method']:
        print(f'Combining method: {p["combining_method"]}')
    if p['description']:
        print(f'Description: {p["description"]}')
    print(f'Model index: {models_index}\n')
    print('Test result:')

    # 输入数据用模型预测
    with torch.no_grad():
        # model_dict = model.state_dict()
        for model in models:
            model.eval()
            model.is_training = False
        count = 0
        att_weight_sum = 0
        for train_arrays, filename in data_loader:
            val_file_name = filename[0].split('.np')[0]
            arrays = train_arrays.to(device)
            arrays = arrays.float()
            brightness_val_real[count] = dic[val_file_name]
            names[count] = val_file_name
            pred_sum = 0
            for model in models:
                pred = model(arrays[0])
                pred = pred.float()
                pred_sum += pred
                if att:
                    att_weight = model.attention_weight(arrays[0])
                    att_weight_sum += att_weight
            pred_avg = pred_sum / len(models)
            pred_brightness_val[count] = pred_avg.data.cpu().numpy()
            count += 1
        if att:
            att_weight_avg = att_weight_sum / (len(models) * count)
            att_weight_avg = att_weight_avg.data.cpu().numpy()

        test_output = np.concatenate((pred_brightness_val, brightness_val_real), 0).astype('float32')
        np.save(output_path, [test_output, names])
        if att:
            np.save(os.path.join(test_outputs_path, '%s_%s_%s_att_weight.npy' % (task_name, train_index, models_index_str)), att_weight_avg)

        print("Test set size =", count)
        pearson = pearsonr(pred_brightness_val, brightness_val_real)
        spearman = spearmanr(pred_brightness_val, brightness_val_real)

        print('Pearson = %f' % float(pearson[0]))
        print('Spearman = %f' % float(spearman[0]))

    return dataset_names, sub_dataset_names, network_name, p['description'], input_shape, pearson[0], spearman[0]

