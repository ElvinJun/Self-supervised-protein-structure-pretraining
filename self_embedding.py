import torch
import torch.nn as nn
# from npy_data_loader import DistanceWindow
from torch.utils.data import Dataset, DataLoader
import pathlib
import os
import time
import numpy as np
from torch.nn import functional as F
import math
import argparse
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from deeppbs import BatchLSTM
import rep_utils

# knn_path = '/home/caiyi/data/rocklin/ssm/ssm_knn_75/'
knn_path = '/home/caiyi/data/rocklin/knn_150/'
# model_path = '/home/caiyi/data/pretrained_models/200_Linear.pth'
model_path = '/home/caiyi/outputs/self__20082602/180_Linear.pth'
# output_path = '/home/caiyi/data/rocklin/knn_self_full/pt_062901/'
output_path = '/home/caiyi/data/rocklin/knn_self_512_full/pt_082602/'
dataset_name = 'knn_135_batch'

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('-g', '--device_num', type=int, default=7)
parser.add_argument('-i', '--input_dim', type=int, default=150)
args = parser.parse_args()

dic = {'A': 0, 'F': 1, 'C': 2, 'D': 3, 'N': 4, 'E': 5, 'Q': 6, 'G': 7, 'H': 8, 'L': 9, 'I': 10,
       'K': 11, 'M': 12, 'P': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19}
dataset_dict = {'knn_75': 'KNN75Dataset', 'knn_135': 'KNN135Dataset', 'knn_3': 'KNN3Dataset', 'knn_9': 'KNN9Dataset',
                'knn_45': 'KNN45Dataset', 'knn_90': 'KNN90Dataset', 'knn_150': 'KNN150Dataset', 'knn_135_batch': 'KnnBatch'}


class KNN3Dataset(Dataset):
    def __init__(self, distance_window_path):
        self.distance_window_path = distance_window_path
        self.file_list = os.listdir(distance_window_path)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        arrays = np.load(os.path.join(self.distance_window_path, filename))
        arrays = arrays[:, 0, -3:]
        filename = filename[:-4]
        return arrays, filename


class KNN75Dataset(Dataset):
    def __init__(self, distance_window_path):
        self.distance_window_path = distance_window_path
        self.file_list = os.listdir(distance_window_path)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        arrays = np.load(os.path.join(self.distance_window_path, filename))
        arrays = arrays.reshape(arrays.shape[0], 75)
        filename = filename[:-4]
        return arrays, filename


class KNN135Dataset(Dataset):
    def __init__(self, distance_window_path):
        self.distance_window_path = distance_window_path
        self.file_list = os.listdir(distance_window_path)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        arrays = np.load(os.path.join(self.distance_window_path, filename))
        arrays = arrays.reshape(arrays.shape[0], 135)
        filename = filename[:-4]
        return arrays, filename


class KNN9Dataset(Dataset):
    def __init__(self, distance_window_path):
        self.distance_window_path = distance_window_path
        self.file_list = os.listdir(distance_window_path)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        arrays = np.load(os.path.join(self.distance_window_path, filename))
        arrays = arrays[:, 0, :]
        arrays = arrays.reshape(arrays.shape[0], 9)
        filename = filename[:-4]
        return arrays, filename


class KNN45Dataset(Dataset):
    def __init__(self, distance_window_path):
        self.distance_window_path = distance_window_path
        self.file_list = os.listdir(distance_window_path)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        arrays = np.load(os.path.join(self.distance_window_path, filename))
        arrays = arrays[:, :5, :]
        arrays = arrays.reshape(arrays.shape[0], 45)
        filename = filename[:-4]
        return arrays, filename


class KNN90Dataset(Dataset):
    def __init__(self, distance_window_path):
        self.distance_window_path = distance_window_path
        self.file_list = os.listdir(distance_window_path)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        arrays = np.load(os.path.join(self.distance_window_path, filename))
        arrays = arrays[:, :10, :]
        arrays = arrays.reshape(arrays.shape[0], 90)
        filename = filename[:-4]
        return arrays, filename


class KnnBatch(Dataset):
    def __init__(self, path, aa_batchsize=60):
        self.input_path = path
        list_path = '/home/caiyi/data/rocklin/src/struct_ssm_test/set_full.txt'
        with open(list_path) as f:
            lines = f.read().split('\n')
            self.file_list = lines[:-1]
        length_file = '/home/caiyi/data/rocklin/src/len1.txt'
        length = {}
        with open(length_file) as f:
            lines = f.read().split('\n')
            for line in lines[:-1]:
                filename, l = line.split(':')
                length[filename] = int(l)
        len_seq_files = [[] for _ in range(2000)]
        lengths = []
        for filename in self.file_list:
            filename = filename[:-4]
            lengths.append(length[filename])
            len_seq_files[length[filename]].append(filename)

        # len_seq_files为按seq_length从小到大分类的文件名
        self.batchs = rep_utils.group_files(len_seq_files, lengths, aa_batchsize)

    def __len__(self):
        return len(self.batchs)

    def __getitem__(self, idx):
        filenames = self.batchs[idx]
        knn = []
        lengths = []
        for filename in filenames:
            pdb_name = filename.split('.')[0]
            print('SHAPE', np.load(os.path.join(self.input_path, f'{filename}.npy')).shape)
            knn.append(np.load(os.path.join(self.input_path, f'{filename}.npy')).reshape(-1, args.input_dim))
            lengths.append(np.shape(knn[-1])[0])
        knn = np.concatenate(knn)
        return knn, filenames, lengths


device_num = args.device_num
torch.cuda.set_device(device_num)
device = torch.device('cuda:%d' % device_num)
orign_arrays = {}

batch_size = 1
exec(f'KNNDataset = {dataset_dict[dataset_name]}')
full_dataset = KNNDataset(knn_path,)
# train_size = int(0.8 * len(full_dataset))
# test_size = len(full_dataset) - train_size
# train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

data_loader = DataLoader(dataset=full_dataset, shuffle=False, batch_size=batch_size)
# val_loader = DataLoader(dataset=test_dataset, pin_memory=True)

if torch.cuda.is_available():
    print('GPU available!!!')
    print('MainDevice=', device)


def swish_fn(x):
    """ Swish activation function """
    return x * torch.sigmoid(x)


class SelfSupervised(nn.Module):
    def __init__(self, input_dim=135, hidden_dim=256, feature_dim=256, output_dim=20):
        super().__init__()

        # ADD "// 2"
        self.input = nn.Linear(input_dim, hidden_dim // 2)
        self.ln1 = nn.LayerNorm(hidden_dim // 2)

        # ADD "// 2"
        self.hidden1 = nn.Linear(hidden_dim // 2, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

        self.lstm = BatchLSTM(feature_dim, hidden_dim * 2)
        self.ln3 = nn.LayerNorm(2 * hidden_dim)

        self.hidden = nn.Linear(hidden_dim * 2, hidden_dim)
        self.ln4 = nn.LayerNorm(hidden_dim)

        self.predict = nn.Linear(hidden_dim, len(dic))

        self.dropout = nn.Dropout(0.1)

    def forward(self, arrays, lengths):
        print('A',arrays.shape)
        hidden_states = swish_fn(self.ln1(self.input(arrays)))
        print('B',hidden_states.shape)
        hidden_states = swish_fn(self.ln2(self.hidden1(hidden_states)))
        print(hidden_states.shape)
        hidden_states = self.lstm(hidden_states, lengths)
        print('C',hidden_states.shape)
        hidden_states = self.ln3(hidden_states)
        return hidden_states


class SemilabelOld(nn.Module):
    def __init__(self, input_dim=75, hidden_dim=128, feature_dim=128, output_dim=20):
        super().__init__()

        self.input = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)

        self.lstm = nn.LSTM(feature_dim, hidden_dim, bidirectional=True)
        self.ln4 = nn.LayerNorm(2 * hidden_dim)

        self.predict = nn.Linear(2 * hidden_dim, len(dic))

    #         self.ln1 = nn.LayerNorm(hidden_dim)

    def forward(self, arrays):
        hidden_states = F.relu(self.ln1(self.input(arrays)))
        hidden_states, (hn, cn) = self.lstm(hidden_states.view(len(hidden_states), 1, -1))
        # hidden_states = F.relu(self.ln4(hidden_states.squeeze(1)))
        # hidden_states = self.predict(hidden_states)
        # output = F.softmax(hidden_states, dim=0)
        return hidden_states


if __name__ == "__main__":
    with torch.no_grad():
        model = torch.load(model_path, map_location=device)
        model.eval()
        model.is_training = False

    for arrays, file_name, lengths in data_loader:
        arrays = arrays.to(device).float()
        pred = model(arrays, lengths).float()

        pred_list = []
        last = 0
        for length in lengths:
            next_ = last + length
            pred_list.append(pred[last:next_].view(length, -1))
            last = next_
        for n in range(len(pred_list)):
            #print(n, file_name)
            output = pred_list[n].data.cpu().numpy()
            np.save(os.path.join(output_path, file_name[n][0]), output)

