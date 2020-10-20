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
from torch.autograd import Variable

knn_path = '/share/seqtonpy/rocklin/input/knn_150_pre'
model_path = '/home/joseph/KNN/outputs/self_20201016_4_sota_cm_knn150_49600/114_Linear.pth'
output_path = '/share/seqtonpy/rocklin/knn_self_512_full/self_20201016_4_sota_cm_knn150_49600_20201018_1_knn150_pre'
dataset_name = 'knn_onehot'

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('-g', '--device_num', type=int, default=6)
parser.add_argument('-i', '--input_dim', type=int, default=150)
args = parser.parse_args()

dic = {'A': 0, 'F': 1, 'C': 2, 'D': 3, 'N': 4, 'E': 5, 'Q': 6, 'G': 7, 'H': 8, 'L': 9, 'I': 10,
       'K': 11, 'M': 12, 'P': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19}
dataset_dict = {'knn_75': 'KNN75Dataset', 'knn_135': 'KNN135Dataset', 'knn_3': 'KNN3Dataset', 'knn_9': 'KNN9Dataset',
                'knn_45': 'KNN45Dataset', 'knn_90': 'KNN90Dataset', 'knn_150': 'KNN150Dataset', 'knn_180':
                'KNN180Dataset', 'knn_135_batch': 'KnnBatch', 'knn_onehot': 'Knnonehot'}


class Knnonehot(Dataset):
    """Extract distance window arrays"""
    def __init__(self, distance_window_path):
        self.distance_window_path = distance_window_path
        self.file_list = os.listdir(distance_window_path)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        arrays = np.load(os.path.join(self.distance_window_path, filename))
        arrays = arrays.reshape(arrays.shape[0], arrays.shape[1]*arrays.shape[2])
        filename = filename[:-4]
        # mix_arrays = np.concatenate((arrays[:-1], arrays[1:]), 1)
        # torsions = np.load(os.path.join(torsions_path, filename))
        return arrays, filename



device_num = args.device_num
torch.cuda.set_device(device_num)
device = torch.device('cuda:%d' % device_num)
orign_arrays = {}

batch_size = 1
exec(f'KNNDataset = {dataset_dict[dataset_name]}')
full_dataset = KNNDataset(knn_path,)
data_loader = DataLoader(dataset=full_dataset, shuffle=False, batch_size=batch_size)

if torch.cuda.is_available():
    print('GPU available!!!')
    print('MainDevice=', device)

def swish_fn(x):
    """ Swish activation function """
    return x * torch.sigmoid(x)

# class SelfSupervised(nn.Module):
#     def __init__(self, input_dim=150, hidden_dim=384, feature_dim=384, output_dim=20):
#         super().__init__()
#
#         # ADD "// 2"
#         self.input = nn.Linear(input_dim, hidden_dim // 2)
#         self.ln1 = nn.LayerNorm(hidden_dim // 2)
#
#         # ADD "// 2"
#         self.hidden1 = nn.Linear(hidden_dim // 2, hidden_dim)
#         self.ln2 = nn.LayerNorm(hidden_dim)
#
#         #self.lstm = BatchLSTM(feature_dim, hidden_dim * 2)
#         self.lstm = nn.LSTM(feature_dim, hidden_dim, bidirectional=True)
#         self.ln3 = nn.LayerNorm(2 * hidden_dim)
#
#         self.hidden = nn.Linear(hidden_dim * 2, hidden_dim)
#         self.ln4 = nn.LayerNorm(hidden_dim)
#
#         self.predict = nn.Linear(hidden_dim, len(dic))
#
#         self.dropout = nn.Dropout(0.1)
#
#     def forward(self, arrays):
#         print('A',arrays.shape)
#         hidden_states = swish_fn(self.ln1(self.input(arrays)))
#         print('B',hidden_states.shape)
#         hidden_states = swish_fn(self.ln2(self.hidden1(hidden_states)))
#
#         print(hidden_states.shape)
#         hidden_states = hidden_states.unsqueeze(1)
#         # hidden_states = self.lstm(hidden_states, lengths)
#         # hidden_states, _ = self.lstm(hidden_states.view(len(hidden_states), 1, -1))
#         hidden_states, _ = self.lstm(hidden_states)
#
#         print('C', hidden_states.shape)
#         hidden_states = hidden_states.squeeze(1)
#         # hidden_states = self.ln3(hidden_states.squeeze(1))
#         return hidden_states

# class Semilabel(nn.Module):
#     def __init__(self, input_dim=args.input_dim, hidden_dim=256, feature_dim=256, output_dim=20):
#         super().__init__()
#
#         self.input = nn.Linear(input_dim, hidden_dim//2)
#         self.ln1 = nn.LayerNorm(hidden_dim//2)
#
#         self.hidden1 = nn.Linear(hidden_dim//2, hidden_dim)
#         self.ln2 = nn.LayerNorm(hidden_dim)
#
#         self.lstm = nn.LSTM(feature_dim, hidden_dim, bidirectional=True)
#         self.ln3 = nn.LayerNorm(2 * hidden_dim)
#
#         # self.fc0 = nn.Linear(768, 1)
#
#         self.hidden = nn.Linear(2 * hidden_dim, hidden_dim)
#         self.ln4 = nn.LayerNorm(hidden_dim)
#
#         self.predict = nn.Linear(hidden_dim, len(dic))
#         self.dropout = nn.Dropout(0.1)
#
#     def forward(self, arrays):
#         hidden_states = swish_fn(self.ln1(self.input(arrays)))
#         hidden_states = swish_fn(self.ln2(self.hidden1(hidden_states)))
#         hidden_states, _ = self.lstm(hidden_states.view(len(hidden_states), 1, -1))
#         hidden_states = hidden_states.squeeze(1)
#         hidden_states = swish_fn(self.ln3(hidden_states))
#
#         # hidden_states_ = hidden_states.transpose(1, 0)
#         # hidden_states1 = hidden_states_.transpose(2, 1)
#         # m = nn.MaxPool1d(hidden_states1.shape[2])
#         # output = m(hidden_states1).reshape((hidden_states1.shape[0], 1, hidden_states1.shape[1]))
#         # similarity = torch.cosine_similarity(hidden_states_, output, dim=2).reshape(hidden_states_.shape[0],
#         #                                                                             hidden_states_.shape[1], 1)
#         # hidden_states = torch.mul(hidden_states_, similarity)
#         # hidden_states = hidden_states.squeeze(0)
#
#         # align = F.softmax(self.fc0(hidden_states), dim=0)
#         #
#         # hidden_states = torch.mul(hidden_states, align)
#
#         # hidden_states = swish_fn(self.ln4(self.hidden(hidden_states)))
#         # output = self.dropout(self.predict(hidden_states))
#         # output = F.softmax(hidden_states, dim=0)
#         # output = self.dropout(hidden_states)
#         return hidden_states

class Semilabel(nn.Module):
    def __init__(self, input_dim=args.input_dim, hidden_dim=384, feature_dim=384, output_dim=20):
        super().__init__()

        self.input = nn.Linear(input_dim, hidden_dim // 2)
        self.ln1 = nn.LayerNorm(hidden_dim // 2)

        self.hidden1 = nn.Linear(hidden_dim // 2, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

        self.lstm = nn.LSTM(feature_dim, hidden_dim, bidirectional=True)
        self.ln3 = nn.LayerNorm(2 * hidden_dim)

        self.hidden = nn.Linear(hidden_dim * 2, hidden_dim)
        self.ln4 = nn.LayerNorm(hidden_dim)

        self.predict = nn.Linear(hidden_dim, len(dic))
        self.dropout = nn.Dropout(0.1)

    def forward(self, arrays):
        hidden_states = swish_fn(self.ln1(self.input(arrays)))
        hidden_states = swish_fn(self.ln2(self.hidden1(hidden_states)))
        hidden_states = hidden_states.unsqueeze(1)
        hidden_states, _ = self.lstm(hidden_states)
        hidden_states = hidden_states.squeeze(1)
        # hidden_states = swish_fn(self.ln4(self.hidden(hidden_states)))
        # output = self.predict(hidden_states)
        # output = F.softmax(output, dim=-1)
        # output = self.dropout(output)

        return hidden_states


if __name__ == "__main__":
    with torch.no_grad():
        model = torch.load(model_path, map_location=device)
        model.eval()
        model.is_training = False

    for arrays, file_name in data_loader:
        arrays = arrays.to(device).float()
        pred = model(arrays[0]).float()

        pred_list = []
        last = 0
        #for length in lengths:
        #    next_ = last + length
        #    pred_list.append(pred[last:next_].view(length, -1))
        #    last = next_
        #for n in range(len(pred_list)):
            #print(n, file_name)
        #    output = pred_list[n].data.cpu().numpy()
        #    np.save(os.path.join(output_path, file_name[n][0]), output)
        np.save(os.path.join(output_path, file_name[0]), pred.data.cpu().numpy())
