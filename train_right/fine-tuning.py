import torch
import torch.nn as nn
import shutil
from torch.utils.data import Dataset, DataLoader
import pathlib
import os
import time
import numpy as np
from torch.nn import functional as F
import math
import argparse
import logging
import network
from logging import handlers
from scipy.stats import pearsonr
from scipy.stats import spearmanr

dic = {'A': 0, 'F': 1, 'C': 2, 'D': 3, 'N': 4, 'E': 5, 'Q': 6, 'G': 7, 'H': 8, 'L': 9, 'I': 10,
       'K': 11, 'M': 12, 'P': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19}

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--path_file', type=str, default='/home/elvin/SSPS/data/gfp/knn_150_1')
parser.add_argument('--seq_file', type=str, default='/home/elvin/SSPS/data/nr40/seq_fa/')
parser.add_argument('--mask_file', type=str, default='/home/elvin/SSPS/data/nr40/mask/')
parser.add_argument('--train_file', type=str, default='pretrain_sota.py')
parser.add_argument('--device_num', type=int, default=7)
parser.add_argument('--subset_name', type=str, default='20092001')
parser.add_argument('--input_dim', type=int, default=150)
parser.add_argument('--target_epochs', type=int, default=500)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--learning_rate', type=int, default=0.0002)
args = parser.parse_args()

device_num = args.device_num
torch.cuda.set_device(device_num)
device = torch.device('cuda:%d' % device_num)

#loss
# loss_function = nn.CrossEntropyLoss()
loss_function = nn.MSELoss()
subset_name = args.subset_name
total_iters = 0
train_name = 'self_%s' % subset_name
save_dir = './outputs/' + train_name
val_dir = os.path.join(save_dir, 'val')
for folder in [save_dir, val_dir]:
    pathlib.Path(folder).mkdir(parents=True, exist_ok=True)
shutil.copy(os.path.join(os.getcwd(), args.train_file),  save_dir)

class DistanceWindow(Dataset):
    """Extract distance window arrays"""
    def __init__(self, distance_window_path, file_path):
        self.distance_window_path = distance_window_path
        self.file_list = os.listdir(distance_window_path)
        self.file_list = []
        self.dic = {}
        files = open(file_path, 'r').readlines()
        values = open('/home/elvin/SSPS/data/gfp/gfp.txt', 'r').readlines()
        for f in files:
            self.file_list.append(f[:-1])
        for v in values:
            key, value = v.split()
            self.dic[key] = value[:-1]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        # seq = open(os.path.join(args.seq_file, filename.split('.')[0] + '.fasta')).readlines()[1]
        # seq_all = open(os.path.join(args.seq_file, filename.split('.')[0] + '.txt')).read()
        # mask = np.load(os.path.join(args.mask_file, filename.split('.')[0] + '.npy'))
        # for i in range(len(mask)):
        #     if mask[i] != 0:
        #         start = i
        #         break
        # for i in range(len(mask) - 1, -1, -1):
        #     if mask[i] != 0:
        #         end = i
        #         break
        # seq = seq_all[start: end+1]
        # sequence = torch.tensor([dic[aa] for aa in list(seq)])
        arrays = np.load(os.path.join(self.distance_window_path, filename))
        arrays = arrays.reshape(arrays.shape[0], arrays.shape[1]*arrays.shape[2])
        filename = filename.split('.')[0]
        value = float(self.dic[filename])
        # mix_arrays = np.concatenate((arrays[:-1], arrays[1:]), 1)
        # torsions = np.load(os.path.join(torsions_path, filename))
        return arrays, filename, value

train_dataset = DistanceWindow(
    distance_window_path=args.path_file, file_path='/home/elvin/SSPS/data/gfp/cv/set_train_0.txt'
)
test_dataset = DistanceWindow(
    distance_window_path=args.path_file, file_path='/home/elvin/SSPS/data/gfp/cv/set_valid_0.txt'
)
# train_size = int(0.95 * len(full_dataset))
# test_size = len(full_dataset) - train_size
# train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
train_size = len(train_dataset)
test_size = len(test_dataset)
data_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=args.batch_size)
val_loader = DataLoader(dataset=test_dataset, pin_memory=True)

if torch.cuda.is_available():
    print('GPU available!!!')
    print('MainDevice=', device)

def swish_fn(x):
    """ Swish activation function """
    return x * torch.sigmoid(x)

class Semilabel(nn.Module):
    def __init__(self, input_dim=args.input_dim, hidden_dim=256, feature_dim=256, output_dim=20, stacked_dim=223):
        super().__init__()
        self.input = nn.Linear(input_dim, hidden_dim // 2)
        self.ln1 = nn.LayerNorm(hidden_dim // 2)

        self.hidden1 = nn.Linear(hidden_dim // 2, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

        self.lstm = nn.LSTM(feature_dim, hidden_dim, bidirectional=True)
        self.ln3 = nn.LayerNorm(2 * hidden_dim)

        self.input = nn.Linear(input_dim, hidden_dim//2)
        self.ln1 = nn.LayerNorm(hidden_dim//2)

        self.hidden1 = nn.Linear(hidden_dim // 2, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

        self.lstm = nn.LSTM(feature_dim, hidden_dim, bidirectional=True)
        self.ln3 = nn.LayerNorm(2 * hidden_dim)

        self.fc1 = nn.Linear(hidden_dim*2, hidden_dim*2)
        self.ln = nn.LayerNorm(hidden_dim*2)
        self.fc2 = nn.Linear(stacked_dim, 1)
        self.output = nn.Linear(hidden_dim*2, 1)

    def forward(self, arrays):

        hidden_states = swish_fn(self.ln1(self.input(arrays)))
        hidden_states = swish_fn(self.ln2(self.hidden1(hidden_states)))
        hidden_states = hidden_states.unsqueeze(1)
        hidden_states, _ = self.lstm(hidden_states)
        arrays = hidden_states.squeeze(1)

        # align = F.softmax(self.fullc0(arrays), dim=0)
        # att_hidden = torch.mm(arrays.transpose(0, 1), align)
        # att_hidden1 = self.ln(torch.sigmoid(self.fullc1(att_hidden.transpose(0, 1))))
        # output = self.output(att_hidden1)

        hidden_gfp = self.ln(torch.relu(self.fc1(arrays)))
        hidden_gfp2 = torch.relu(self.fc2(hidden_gfp.transpose(0, 1)))
        output = self.output(hidden_gfp2.transpose(0, 1))

        return output


path_model = '/home/elvin/SSPS/train/outputs/self_20091601/199_Linear.pth'
model = Semilabel()
for p in model.parameters():
    nn.init.constant(p, 0)
#pretrain
mid_model = torch.load(path_model).state_dict()
save_model = mid_model
model_dict = model.state_dict()
state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}# dict_keys(['w', 'conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias'])
model_dict.update(state_dict)
model.load_state_dict(model_dict)
#pretrain
path_model_down = '/home/elvin/SSPS/down_pred/outputs_gfp/gfp__20091601_gfp-0/271_Linear.pth'
down_model = torch.load(path_model_down).state_dict()
save_model = down_model
model_dict = model.state_dict()
state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}# dict_keys(['w', 'conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias'])
# print(state_dict)
model_dict.update(state_dict)
# print(model_dict)

model.load_state_dict(model_dict)
# print(model.state_dict())
model = Semilabel().to(device)
optimizer = torch.optim.Adamax(model.parameters(), lr=args.learning_rate)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99, last_epoch=-1)

if __name__ == '__main__':
    for epoch in range(args.target_epochs):
        epoch_iter = 0
        print('epoch', epoch)
        losses = []
        j = 0
        # scheduler.step()
        for train_arrays, file_name, value in data_loader:
            train_arrays = torch.tensor(train_arrays, dtype=torch.float32)
            train_arrays = train_arrays.to(device)
            pred = model(train_arrays[0]).float()
            value = value.squeeze().cuda()
            total_loss = loss_function(value, pred)
            # total_loss = 0
            # l = len(seq)
            # total_len = 0
            # total_len += (l - 5) * 2 + (l - 4) * 2
            # total_loss += loss_function(pred[:l - 4], seq[4:]) * (l - 4)
            # total_loss += loss_function(pred[:l - 5], seq[5:]) * (l - 5)
            # total_loss += loss_function(pred[4:], seq[:l - 4]) * (l - 4)
            # total_loss += loss_function(pred[5:], seq[:l - 5]) * (l - 5)
            # total_loss /= total_len
            losses.append(float(total_loss))

            total_iters += args.batch_size
            epoch_iter += args.batch_size
            total_loss.backward()

            if (j + 1) % 1 == 0:
                optimizer.step()
                optimizer.zero_grad()
            j += 1
            # with open(save_dir + '/train_loss.txt', 'a') as writer:
            #     writer.write(str(file_name[0]) + '\n')
            #     writer.write(str(total_loss) + '\n')
            if total_iters % 200 == 0:
                loss_average = sum(losses) / len(losses)

                print('iters', total_iters, '    mean_loss=', loss_average)
                with open(save_dir + '/train_loss.txt', 'a') as writer:
                    writer.write('iter=%s  losses_average=%s\n' % (total_iters, loss_average))

            if total_iters % 1000 == 0:
                last_linear_filename = 'last_Linear.pth'
                last_linear_path = os.path.join(save_dir, last_linear_filename)
                torch.save(model, last_linear_path)

            if total_iters % train_size == 0:
                save_linear_filename = '%s_Linear.pth' % epoch
                save_linear_path = os.path.join(save_dir, save_linear_filename)
                torch.save(model, save_linear_path)
                model.eval()
                model.is_training = False

                with torch.no_grad():
                    writer_val = open(save_dir + '/val_loss.txt', 'a')
                    writer_val.write('epoch %d\n' % epoch)
                    val_losses = []
                    # val_output_folder = os.path.join(val_dir, '_%d' % epoch)
                    # pathlib.Path(val_output_folder).mkdir(parents=True, exist_ok=True)

                    for val_arrays, val_file_name, value_val in val_loader:
                        val_arrays = torch.tensor(val_arrays, dtype=torch.float32)
                        val_arrays = val_arrays.to(device)

                        pred_val = model(val_arrays[0]).float()
                        value_val = value_val.squeeze().cuda().float()
                        total_val_loss = loss_function(value_val, pred_val)
                        # seq_val = seq_val[1:] + seq_val[0]
                        # total_val_loss = 0
                        # total_val_len = 0
                        # l = len(seq_val)
                        # total_val_len += (l - 4) * 2 + (l - 5) * 2
                        # total_val_loss += loss_function(pred_val[:l - 4], seq_val[4:]) * (l - 4)
                        # total_val_loss += loss_function(pred_val[:l - 5], seq_val[5:]) * (l - 5)
                        # total_val_loss += loss_function(pred_val[4:], seq_val[:l - 4]) * (l - 4)
                        # total_val_loss += loss_function(pred_val[5:], seq_val[:l - 5]) * (l - 5)
                        # total_val_loss /= total_val_len
                        val_losses.append(float(total_val_loss))
                    val_loss_av = sum(val_losses) / len(val_losses)
                    writer_val.write('loss=%f\n' % (sum(losses) / len(losses)))
                    writer_val.write('val_loss_avera =%f\n' % val_loss_av)
                    writer_val.write('epoch %d, total_iters %d\n\n' % (epoch, total_iters))
                    writer_val.close()
                    print('val_loss_average=%f' % val_loss_av)
                    print('epoch %d, total_iters %d' % (epoch, total_iters))

                model.train()
