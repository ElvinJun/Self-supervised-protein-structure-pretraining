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
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import rep_utils
import deeppbs
from deeppbs import swish_fn

# DESCRIPTION: window_size=6; spherical

dic = {'A': 0, 'F': 1, 'C': 2, 'D': 3, 'N': 4, 'E': 5, 'Q': 6, 'G': 7, 'H': 8, 'L': 9, 'I': 10,
       'K': 11, 'M': 12, 'P': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19}

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--path_file', type=str, default='/home/caiyi/data/nr40/knn_150')
parser.add_argument('--seq_file', type=str, default='/home/caiyi/data/nr40/seq_fa/pdb.fasta')
parser.add_argument('--train_file', type=str, default='pretrain.py')
parser.add_argument('--device_num', type=int, default=2)
parser.add_argument('--subset_name', type=str, default='20082602')
parser.add_argument('--input_dim', type=int, default=150)
parser.add_argument('--target_epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--learning_rate', type=int, default=0.01)
args = parser.parse_args()
device_num = args.device_num
torch.cuda.set_device(device_num)
device = torch.device('cuda:%d' % device_num)

loss_function = nn.CrossEntropyLoss()
subset_name = args.subset_name
total_iters = 0
train_name = 'self__%s' % subset_name
save_dir = '/home/caiyi/outputs/' + train_name
length_file = '/home/caiyi/data/nr40/len.txt'
val_dir = os.path.join(save_dir, 'val')
for folder in [save_dir, val_dir]:
    pathlib.Path(folder).mkdir(parents=True, exist_ok=True)
shutil.copy(os.path.join(os.getcwd(), args.train_file),  save_dir)


full_dataset = deeppbs.KnnBatch(path=args.path_file, length_file=length_file, seq_file=args.seq_file)
train_size = int(0.95 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
data_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=args.batch_size)
val_loader = DataLoader(dataset=test_dataset, pin_memory=True)

if torch.cuda.is_available():
    print('GPU available!!!')
    print('MainDevice=', device)


class SelfSupervised(nn.Module):
    def __init__(self, input_dim=args.input_dim, hidden_dim=256, feature_dim=256, output_dim=20):
        super().__init__()

        self.input = nn.Linear(input_dim, hidden_dim // 2)
        self.ln1 = nn.LayerNorm(hidden_dim // 2)

        self.hidden1 = nn.Linear(hidden_dim // 2, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

        self.lstm = deeppbs.BatchLSTM(feature_dim, hidden_dim * 2)
        self.ln3 = nn.LayerNorm(2 * hidden_dim)

        self.hidden = nn.Linear(hidden_dim * 2, hidden_dim)
        self.ln4 = nn.LayerNorm(hidden_dim)

        self.predict = nn.Linear(hidden_dim, len(dic))

    def forward(self, arrays, lengths):
        hidden_states = swish_fn(self.ln1(self.input(arrays)))
        hidden_states = swish_fn(self.ln2(self.hidden1(hidden_states)))
        hidden_states = self.lstm(hidden_states, lengths)
        hidden_states = swish_fn(self.ln3(hidden_states))
        hidden_states = swish_fn(self.ln4(self.hidden(hidden_states)))
        output = self.predict(hidden_states)

        return output


model = SelfSupervised().to(device)
optimizer = torch.optim.Adamax(model.parameters(), lr=args.learning_rate)


if __name__ == '__main__':
    for epoch in range(args.target_epochs):
        epoch_iter = 0
        print('epoch', epoch)
        losses = []
        for train_arrays, seqs, file_names, lengths in data_loader:
            train_arrays = torch.tensor(train_arrays, dtype=torch.float32)[0]
            train_arrays = train_arrays.to(device)

            preds = model(train_arrays, lengths).float()
            pred_list = []
            last = 0
            for length in lengths:
                next_ = last + length
                pred_list.append(preds[last: next_].view(length, -1))
                last = next_

            total_loss = 0
            total_len = 0
            for n in range(len(file_names)):
                pred = pred_list[n]
                seq = seqs[n].squeeze().cuda()
                l = len(seq)
                total_len += (l - 1) * 2 + (l - 2) * 2 + (l - 3) * 2
                total_loss += loss_function(pred[:l-1], seq[1:]) * (l - 1)
                total_loss += loss_function(pred[:l-2], seq[2:]) * (l - 2)
                total_loss += loss_function(pred[:l-3], seq[3:]) * (l - 3)
                total_loss += loss_function(pred[1:], seq[:l-1]) * (l - 1)
                total_loss += loss_function(pred[2:], seq[:l-2]) * (l - 2)
                total_loss += loss_function(pred[3:], seq[:l-3]) * (l - 3)
            total_loss /= total_len
            losses.append(float(total_loss))
            total_iters += args.batch_size
            epoch_iter += args.batch_size
            total_loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            if total_iters % 32 == 0:
                loss_average = sum(losses) / len(losses)

                print('iters', total_iters, '    mean_loss=', loss_average)
                with open(save_dir + '/train_loss.txt', 'a') as writer:
                    writer.write('iter=%s  losses_average=%s\n' % (total_iters, loss_average))

            if total_iters % 1024 == 0:
                last_linear_filename = 'last_Linear.pth'
                last_linear_path = os.path.join(save_dir, last_linear_filename)
                torch.save(model, last_linear_path)

            if total_iters % train_size == 0 or epoch_iter >= train_size:
                save_linear_filename = '%s_Linear.pth' % epoch
                save_linear_path = os.path.join(save_dir, save_linear_filename)
                torch.save(model, save_linear_path)
                model.eval()
                model.is_training = False

                with torch.no_grad():
                    writer_val = open(save_dir + '/valid_loss.txt', 'a')
                    writer_val.write('epoch %d\n' % epoch)
                    val_losses = []

                    for val_arrays, val_seqs, val_file_names, val_lengths in val_loader:
                        val_arrays = torch.tensor(val_arrays[0], dtype=torch.float32)
                        val_arrays = val_arrays.to(device)

                        preds_val = model(val_arrays, val_lengths).float()
                        pred_list_val = []
                        last = 0
                        for length in val_lengths:
                            next_ = last + length
                            pred_list_val.append(preds_val[last: next_].view(length, -1))
                            last = next_

                        total_val_loss = 0
                        total_val_len = 0
                        for n in range(len(val_file_names)):
                            pred_val = pred_list_val[n]
                            seq_val = val_seqs[n].squeeze().cuda()
                            l = len(seq_val)
                            total_val_len += (l - 1) * 2 + (l - 2) * 2 + (l - 3) * 2
                            total_val_loss += loss_function(pred_val[:l-1], seq_val[1:]) * (l - 1)
                            total_val_loss += loss_function(pred_val[:l-2], seq_val[2:]) * (l - 2)
                            total_val_loss += loss_function(pred_val[:l-3], seq_val[3:]) * (l - 3)
                            total_val_loss += loss_function(pred_val[1:], seq_val[:l-1]) * (l - 1)
                            total_val_loss += loss_function(pred_val[2:], seq_val[:l-2]) * (l - 2)
                            total_val_loss += loss_function(pred_val[3:], seq_val[:l-3]) * (l - 3)
                        total_val_loss /= total_val_len
                        val_losses.append(float(total_val_loss))

                    val_loss_avg = sum(val_losses) / test_size
                    writer_val.write('val_loss_average = %f\n' % val_loss_avg)
                    writer_val.write('epoch %d, total_iters %d\n\n' % (epoch, total_iters))
                    writer_val.close()
                    print('val_loss_average=%f' % val_loss_avg)
                    print('epoch %d, total_iters %d' % (epoch, total_iters))

                model.train()

