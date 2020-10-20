# -*- coding: utf-8 -*
"""
FOR COMPATIBILITY OF predict_test_set.py
"""


import torch
import torch.nn as nn
import os
import numpy as np
from torch.utils.data import Dataset
import math
import torch.nn.functional as F
import rep_utils


'''
Datasets
'''
class KNN2tor_pept(Dataset):
    def __init__(self, file_list):
        self.input_path = './data/nr40/KNN'
        self.target_path = './data/nr40/tor'
        with open('./data/nr40/dataset_list/%s.txt' % file_list) as f:
            self.file_list = f.read().split('\n')

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        knn = np.load(os.path.join(self.input_path, '%s.npy' %
                                   filename)).reshape(-1, 15 * 4)
        knn_pept = np.concatenate([knn[:-1], knn[1:]], axis=1)
        sincos = np.load(os.path.join(self.target_path, '%s.npy' % filename))
        return knn_pept, sincos, filename


# FOR COMPATIBILITY. Older version of KNN2tor_pept
class DistanceWindow(Dataset):
    def __init__(self, path):
        self.input_path = path['knn_ori']
        self.file_list = os.listdir(path)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        arrays = np.load(os.path.join(self.input_path, filename)).reshape((-1, 60))
        mix_arrays = np.concatenate((arrays[:-1], arrays[1:]), 1)
        torsions = np.load(os.path.join(self.input_path, filename))
        return mix_arrays, filename


class KNN2tor(Dataset):
    # MODIFIED
    def __init__(self, path, list_path):
        # self.input_path = './data/nr40/KNN'
        # self.target_path = './data/nr40/tor'
        # with open('./data/nr40/dataset_list/%s.txt' % file_list) as f:
        #     self.file_list = f.read().split('\n')
        self.input_path = path['knn_75']
        if list_path == None:
            self.file_list = os.listdir(path)
        else:
            with open(list_path, 'r') as file:
                self.file_list = file.read().splitlines()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        # MODIFIED
        knn = np.load(os.path.join(self.input_path, filename)).reshape(-1, 15 * 5)
        # MODIFIED
        # sincos = np.load(os.path.join(self.target_path, '%s.npy' % filename))
        return knn, filename


class KNN2tor_batch(Dataset):
    def __init__(self, path, list_path, aa_batchsize=800):
        # self.input_path = './data/nr40/KNN'
        # self.target_path = './data/nr40/tor'
        self.input_path = path['knn_75_batch']
        if list_path == None:
            self.file_list = os.listdir(path)
        else:
            with open(file_list, 'r') as f:
                self.file_list = f.read().splitlines()

        length = {}
        with open('./data/nr40/len.txt') as f:
            lines = f.read().split('\n')
            for line in lines[:-1]:
                filename, l_ = line.split(':')
                length[filename] = int(l_)

        len_sep_files = [[] for _ in range(2000)]
        lengths = []
        for filename in self.file_list:
            lengths.append(length[filename])
            len_sep_files[length[filename]].append(filename)

        self.batchs = rep_utils.group_files(len_sep_files, lengths, aa_batchsize)

    def __len__(self):
        return len(self.batchs)

    def __getitem__(self, idx):
        filenames = self.batchs[idx]
        knn = []
        sincos = []
        lengths = []
        for filename in filenames:
            knn.append(np.load(os.path.join(self.input_path, '%s.npy' %
                                            filename)).reshape(-1, 15 * 5))
            sincos.append(np.load(os.path.join(
                self.target_path, '%s.npy' % filename)))
            lengths.append(np.shape(knn[-1])[0])

        knn = np.concatenate(knn)
        sincos = np.concatenate(sincos, 1)
        return knn, sincos, filenames, lengths


class KNNtest(Dataset):
    def __init__(self):
        self.input_path = './data/test/KNN'
        self.file_list = os.listdir(self.input_path)
        self.file_list = [filename[:-4] for filename in self.file_list]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        knn = np.load(os.path.join(self.input_path, '%s.npy' %
                                   filename)).reshape(-1, 15 * 5)
        return knn, filename


# MODIFIED
class Image2tor(Dataset):
    def __init__(self, path, list_path):
        # self.input_path = './data/nr40/comp_image_pept_r128'
        # self.input_path = './data/nr40/comp_image_ca_rescon'
        self.input_path = path['image_ori']
        # self.input_path = '/share/Data/processed/nr40/comp_image_ca_multiview'
        # self.target_path = './data/nr40/tor'
        if list_path == None:
            self.file_list = os.listdir(path)
        else:
            with open(list_path, 'r') as f:
                self.file_list = f.read().split('\n')

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        image = rep_utils.load_compressed_array(
            os.path.join(self.input_path, filename))
        image = np.swapaxes(image, 1, 3)
        # sincos = np.load(os.path.join(self.target_path, '%s.npy' % filename))
        return image, filename


class Image2tor_batch(Dataset):
    def __init__(self, file_list, aa_batchsize=800):
        # self.input_path = './data/nr40/comp_image_pept_r128'
        self.input_path = './data/nr40/comp_image_ca_rescon'
        # self.input_path = '/share/Data/processed/nr40/comp_image_ca_multiview'
        self.target_path = './data/nr40/tor'
        with open('./data/nr40/dataset_list/%s.txt' % file_list) as f:
            self.file_list = f.read().split('\n')

        length = {}
        with open('./data/nr40/len.txt') as f:
            lines = f.read().split('\n')
            for line in lines[:-1]:
                filename, l_ = line.split(':')
                length[filename] = int(l_)

        len_sep_files = [[] for _ in range(2000)]
        lengths = []
        for filename in self.file_list:
            lengths.append(length[filename])
            len_sep_files[length[filename]].append(filename)

        self.batchs = rep_utils.group_files(len_sep_files, lengths, aa_batchsize)

    def __len__(self):
        return len(self.batchs)

    def __getitem__(self, idx):
        filenames = self.batchs[idx]
        images = []
        sincos = []
        lengths = []
        for filename in filenames:
            image = rep_utils.load_compressed_array(
                os.path.join(self.input_path, '%s.npy' % filename))
            image = np.swapaxes(image, 1, 3)
            images.append(image)
            sincos.append(np.load(os.path.join(
                self.target_path, '%s.npy' % filename)))
            lengths.append(np.shape(images[-1])[0])

        images = np.concatenate(images)
        sincos = np.concatenate(sincos, 1)
        return images, sincos, filenames, lengths


class Image_test(Dataset):
    def __init__(self):
        # self.input_path = './data/test/comp_image_pept_r128'
        # self.input_path = './data/test/comp_image_ca_r128'
        self.input_path = './data/test/comp_image_ca_rescon'
        # self.input_path = '/share/Data/processed/test/comp_image_ca_multiview'
        self.file_list = os.listdir(self.input_path)
        self.file_list = [filename[:-4] for filename in self.file_list]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        image = rep_utils.load_compressed_array(
            os.path.join(self.input_path, '%s.npy' % filename))
        image = np.swapaxes(image, 1, 3)
        return image, filename


'''
nn_modules
'''
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


class LinearBN1D(nn.Module):
    def __init__(self, input_dim, output_dim, activation=nn.ReLU(inplace=True)):
        super().__init__()
        self.Linear = nn.Linear(input_dim, output_dim)
        self.BatchNorm1d = nn.BatchNorm1d(output_dim)
        self.activation = activation

    def forward(self, x):
        x = self.Linear(x)
        x = self.BatchNorm1d(x)
        x = self.activation(x)
        return x


class LinearLN(nn.Module):
    def __init__(self, input_dim, output_dim, activation=nn.ReLU(inplace=True)):
        super().__init__()
        self.Linear = nn.Linear(input_dim, output_dim)
        self.LayerNorm = nn.LayerNorm(output_dim)
        self.activation = activation

    def forward(self, x):
        x = self.Linear(x)
        x = self.LayerNorm(x)
        x = self.activation(x)
        return x


'''
LOCAL FEATURE BLOCKS
'''
class MLPLocalStruEmbedBlocks(nn.Module):
    def __init__(self, input_dim, basic_dim, layer_num=5, max_dim=2048):
        super().__init__()
        self.MLPLayers = nn.ModuleList([])
        self.MLPLayers.append(LinearBN1D(
            input_dim=input_dim, output_dim=basic_dim))

        large_layer = int((layer_num+1) // 2)
        layer_dims = np.ones(layer_num, dtype='int')
        for i in range(large_layer):
            layer_dims[i] = layer_dims[-i-1] = (2**i)*basic_dim

        for i in range(layer_num-1):
            self.MLPLayers.append(LinearBN1D(input_dim=min(layer_dims[i], max_dim),
                                             output_dim=min(layer_dims[i+1], max_dim)))

    def forward(self, x):
        for layer in self.MLPLayers:
            x = layer(x)
        return x


class Conv2dStaticSamePadding(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, image_size=128, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.stride = self.stride if len(self.stride) == 2 else [
            self.stride[0]] * 2

        assert image_size is not None
        ih, iw = image_size if type(image_size) == list else [
            image_size, image_size]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] +
                    (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] +
                    (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            self.static_padding = nn.ZeroPad2d(
                (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
        else:
            self.static_padding = Identity()

    def forward(self, x):
        x = self.static_padding(x)
        x = F.conv2d(x, self.weight, self.bias, self.stride,
                     self.padding, self.dilation, self.groups)
        return x


class Identity(nn.Module):
    def __init__(self, ):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


class ResConvLayer(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=3, stride=2):
        super().__init__()

        self._conv = Conv2dStaticSamePadding(
            in_channels=input_dim, out_channels=output_dim,
            kernel_size=kernel_size, stride=stride)
        self._bn = nn.BatchNorm2d(num_features=output_dim)

        self.activation = nn.ReLU(inplace=True)

    def forward(self, inp):
        input_shape = inp.size()
        x = inp
        x = self.activation(self._bn(self._conv(x)))

        if x.size() == input_shape:
            x = x + inp
        return x


# FOR COMPATIBILITY
class ResConvLayerOld(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=3, stride=2):
        super().__init__()

        self._conv = Conv2dStaticSamePadding(
            in_channels=input_dim, out_channels=output_dim,
            kernel_size=kernel_size, stride=stride, bias=False)
        self._bn = nn.BatchNorm2d(num_features=output_dim)

        self.activation = MemoryEfficientSwish()

    def forward(self, inp):
        input_shape = inp.size()
        x = inp 
        x = self.activation(self._bn(self._conv(x)))

        if x.size() == input_shape:
            x = x + inp 
        return x


class ConvLocalStruEmbedBlocks(nn.Module):
    def __init__(self, input_dim=5, output_dim=1024, basic_dim=64, block_num=4, blocks_repeat=3):
        super().__init__()
        self._conv_basic = Conv2dStaticSamePadding(
            in_channels=input_dim, out_channels=basic_dim, kernel_size=3, stride=2)
        self._bn0 = nn.BatchNorm2d(num_features=basic_dim)

        self._blocks = nn.ModuleList([])
        for _ in range(blocks_repeat - 1):
            self._blocks.append(ResConvLayer(
                input_dim=basic_dim, output_dim=basic_dim, stride=1))

        layer_dims = basic_dim * np.power(2, np.arange(block_num)).astype(int)
        for i in range(block_num - 1):
            self._blocks.append(ResConvLayer(
                input_dim=layer_dims[i], output_dim=layer_dims[i+1]))

            for _ in range(blocks_repeat - 1):
                self._blocks.append(ResConvLayer(
                    input_dim=layer_dims[i+1], output_dim=layer_dims[i+1], stride=1))

        self._conv_out = Conv2dStaticSamePadding(
            in_channels=layer_dims[-1], out_channels=output_dim//2, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=output_dim//2)

        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._max_pooling = nn.AdaptiveMaxPool2d(1)

        self.activation = nn.ReLU(inplace=True)

    def forward(self, inp):
        x = self.activation(self._bn0(self._conv_basic(inp)))

        for block in self._blocks:
            x = block(x)

        x = self.activation(self._bn1(self._conv_out(x)))

        x_mean = self._avg_pooling(x)
        x_max = self._max_pooling(x)
        x = torch.cat((x_mean, x_max), dim=1)
        return x


# FOR COMPATIBILITY
class ConvLocalStruEmbedBlocksOld(nn.Module):
    def __init__(self, input_dim=5, output_dim=512, basic_dim=64, block_num=4, blocks_repeat=3):
        super().__init__()
        self._conv_basic = Conv2dStaticSamePadding(
            in_channels=input_dim, out_channels=basic_dim, kernel_size=3, stride=2, bias=False)
        self._bn0 = nn.BatchNorm2d(num_features=basic_dim)

        self._blocks = nn.ModuleList([])
        for _ in range(blocks_repeat - 1):
            self._blocks.append(ResConvLayerOld(
                input_dim=basic_dim, output_dim=basic_dim, stride=1))

        layer_dims = basic_dim * np.power(2, np.arange(block_num)).astype(int)
        # layer_dims = basic_dim + basic_dim * np.arange(block_num)
        for i in range(block_num - 1):
            self._blocks.append(ResConvLayerOld(
                input_dim=layer_dims[i], output_dim=layer_dims[i+1]))

            for _ in range(blocks_repeat - 1):
                self._blocks.append(ResConvLayerOld(
                    input_dim=layer_dims[i+1], output_dim=layer_dims[i+1], stride=1))

        self._conv_out = Conv2dStaticSamePadding(
            in_channels=layer_dims[-1], out_channels=output_dim, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=output_dim)

        self._avg_pooling = nn.AdaptiveAvgPool2d(1)

        self.activation = MemoryEfficientSwish()

    def forward(self, inp):
        x = self.activation(self._bn0(self._conv_basic(inp)))

        for block in self._blocks:
            x = block(x)

        x = self.activation(self._bn1(self._conv_out(x)))

        x = self._avg_pooling(x)
        return x


'''
GLOBAL FEATURE BLOCKS
'''
class LSTMGlobalizingBlocks(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        hidden_dim = int(output_dim//2)
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, bidirectional=True)
        self._ln = nn.LayerNorm(output_dim)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x.view(x.size(0), 1, -1)
        x, _ = self.lstm(x)
        x = self.activation(self._ln(x))
        x = x.squeeze(1)
        return x


class BatchLSTMGlobalizingBlocks(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        hidden_dim = int(output_dim//2)
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, bidirectional=True, batch_first=True)
        self._ln = nn.LayerNorm(output_dim)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x, lengths):
        x_batch = []
        last = 0
        for length in lengths:
            next_ = last + length
            x_batch.append(x[last: next_].view(length, -1))
            last = next_

        x_batch = torch.nn.utils.rnn.pad_sequence(x_batch, batch_first=True)
        x_batch = torch.nn.utils.rnn.pack_padded_sequence(
            x_batch, lengths, batch_first=True)
        x_batch = self.lstm(x_batch)[0]
        x_batch, _ = torch.nn.utils.rnn.pad_packed_sequence(
            x_batch, batch_first=True)

        x_batch = self.activation(self._ln(x_batch))
        x = [x_batch[i, :lengths[i]] for i in range(len(lengths))]
        x = [torch.cat((x_[:-1], x_[1:]), 1) for x_ in x]
        x = torch.cat(x, 0)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=2000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        x = self.dropout(x)
        return x


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model):
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList(
            [nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        query, key, value = [layer(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for layer, x in zip(self.linear_layers, (query, key, value))]

        scores = torch.matmul(query, key.transpose(-2, -1)) \
            / math.sqrt(query.size(-1))
        attn = F.softmax(scores, dim=-1)
        x = torch.matmul(p_attn, value)

        x = x.transpose(1, 2).contiguous().view(
            batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)


class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        super().__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.feed_forward = FeedForward(
            d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.input_sublayer(
            x, lambda _x: self.attention.forward(_x, _x, _x))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)


class TransformerEncoders(nn.Module):
    def __init__(self, hidden=512, n_layers=12, attn_heads=8, dropout=0.1):
        super().__init__()
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])

    def forward(self, x):
        for transformer in self.transformer_blocks:
            x = transformer.forward(x)
        return x


class TransformerGlobalizingBlocks(nn.Module):
    def __init__(self, input_dim, num_head=8, num_layer=1):
        super().__init__()
        self.pos_encoder = PositionalEncoding(input_dim)
        self.transformer_encoders = TransformerEncoders(
            hidden=input_dim, n_layers=num_layer, attn_heads=num_head)

    def forward(self, x):
        x = x.view(x.size(0), 1, -1)
        x = self.pos_encoder(x)
        x = self.transformer_encoders(x)
        x = x.squeeze(1)
        return x


'''
PREDICT MODULES
'''
class JointPredictor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=512):
        super().__init__()
        self.buffering_layer = LinearLN(2*input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.cat((x[:-1], x[1:]), 1)
        x = self.buffering_layer(x)
        x = self.output(x)
        return x


class PrejointedPredictor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=512):
        super().__init__()
        self.buffering_layer = LinearLN(2*input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.buffering_layer(x)
        x = self.output(x)
        return x


class BatchJointPredictor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=512):
        super().__init__()
        self.buffering_layer = LinearLN(2*input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x_batch):
        x_batch = [torch.cat((x[:-1], x[1:]), 1) for x in x_batch]
        x = torch.cat(x_batch, 0)
        x = self.buffering_layer(x)
        x = self.output(x)
        return x


class SplitPredictor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, activation=Swish()):
        super().__init__()
        sub_output_dim = int(output_dim//2)
        self.sub_net1 = LinearLN(input_dim, hidden_dim, activation=activation)
        self.output1 = nn.Linear(hidden_dim, sub_output_dim)

        self.sub_net2 = LinearLN(input_dim, hidden_dim, activation=activation)
        self.output2 = nn.Linear(hidden_dim, sub_output_dim)

        self.activation = activation

    def forward(self, x): 
        x1 = self.sub_net1(x)
        x1 = self.output1(x1)

        x2 = self.sub_net2(x)
        x2 = self.output2(x2)

        x = torch.cat((x1, x2), 1)
        return x


class MLP2HiddenSigmoid(nn.Module):
    def __init__(self, input_dim, stacked_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(stacked_dim, 1)
        self.output = nn.Linear(input_dim, 1)
        self.ln = nn.LayerNorm(input_dim)

    def forward(self, arrays):
        hidden1 = self.ln(torch.sigmoid(self.fc1(arrays)))
        hidden2 = torch.sigmoid(self.fc2(hidden1.transpose(0, 1)))
        output = self.output(hidden2.transpose(0, 1))
        return output


class MLP2HiddenReLU(nn.Module):
    def __init__(self, input_dim, stacked_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(stacked_dim, 1)
        self.output = nn.Linear(input_dim, 1)
        self.ln = nn.LayerNorm(input_dim)

    def forward(self, arrays):
        hidden1 = self.ln(torch.relu(self.fc1(arrays)))
        hidden2 = torch.relu(self.fc2(hidden1.transpose(0, 1)))
        output = self.output(hidden2.transpose(0, 1))
        return output


# stacked_dim=1, the name is inproper
class MLP3HiddenReLU(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        hidden_dim = input_dim // 2
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.ln1 = nn.LayerNorm(input_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

    def forward(self, arrays):
        hidden1 = self.ln1(torch.relu(self.fc1(arrays)))
        hidden2 = self.ln2(torch.relu(self.fc2(hidden1)))
        output = self.fc3(hidden2)
        return output


def freeze_parameter(model):
    for param in model.parameters():
        param.requires_grad = False


def activate_parameter(model):
    for param in model.parameters():
        param.requires_grad = True


class DeepPBS(nn.Module):
    def __init__(self, blocks, dims):
        super().__init__()

        # knn_75: [75, 1024, 1024, 4]
        input_dim, local_dim, global_dim, output_dim = dims

        if blocks['local'] == 'Conv':
            self.parallel = True
            self.local_structure_embeding_block = ConvLocalStruEmbedBlocks(
                input_dim=input_dim, output_dim=local_dim).cuda()
            self.local_structure_embeding_blocks = nn.DataParallel(
                self.local_structure_embeding_block)
		# 'ConvOld': Model 35
        if blocks['local'] == 'ConvOld':
            self.parallel = True
            self.local_structure_embeding_block = ConvLocalStruEmbedBlocksOld(
                input_dim=input_dim, output_dim=local_dim).cuda()
            self.local_structure_embeding_blocks = nn.DataParallel(
                self.local_structure_embeding_block)
        elif blocks['local'] == 'MLP':
            self.parallel = False
            self.local_structure_embeding_blocks = MLPLocalStruEmbedBlocks(
                input_dim=input_dim, basic_dim=local_dim).cuda()
        elif blocks['local'] is None:
            self.parallel = False
            self.local_structure_embeding_blocks = Identity()

        if blocks['global'] == 'Transformer':
            self.local_structure_feature_globalizing_blocks = TransformerGlobalizingBlocks(
                input_dim=local_dim).cuda()
        elif blocks['global'] == 'LSTM':
            self.local_structure_feature_globalizing_blocks = LSTMGlobalizingBlocks(
                input_dim=local_dim, output_dim=global_dim).cuda()
        elif blocks['global'] == 'BatchLSTM':
            self.local_structure_feature_globalizing_blocks = BatchLSTMGlobalizingBlocks(
                input_dim=local_dim, output_dim=global_dim).cuda()
        elif blocks['global'] is None:
            self.local_structure_feature_globalizing_blocks = Identity()

        if blocks['predict'] == 'Joint':
            self.predicting_blocks = JointPredictor(
                input_dim=global_dim, output_dim=output_dim).cuda()
        elif blocks['predict'] == 'Prejoint':
            self.predicting_blocks = PrejointedPredictor(
                input_dim=global_dim, output_dim=output_dim).cuda()
        elif blocks['predict'] == 'BatchJoint':
            self.predicting_blocks = BatchJointedPredictor(
                input_dim=global_dim, output_dim=output_dim).cuda()
        elif blocks['predict'] == 'Split':
            self.predicting_blocks = SplitPredictor(
                input_dim=global_dim, output_dim=output_dim).cuda()
        elif blocks['predict'] == '2HidSig':
            self.predicting_blocks = MLP2HiddenSigmoid(input_dim=global_dim, stacked_dim=1).cuda()

    def forward(self, x, lens=None):
        x = self.local_structure_embeding_blocks(x)
        x = self.local_structure_feature_globalizing_blocks(x)
        x = self.predicting_blocks(x)
        return x

    def extract_feature(self, x, lens=None):
        local_feature = self.local_structure_embeding_blocks(x)
        global_feature = self.local_structure_feature_globalizing_blocks(local_feature)
        return local_feature, global_feature

    def save_model(self, save_name):
        checkpoint = {}
        if self.parallel:
            checkpoint['local'] = self.local_structure_embeding_block.state_dict()
        else:
            checkpoint['local'] = self.local_structure_embeding_blocks.state_dict()
        checkpoint['global'] = self.local_structure_feature_globalizing_blocks.state_dict()
        checkpoint['predict'] = self.predicting_blocks.state_dict()
        torch.save(checkpoint, '%s.pth.tar' % save_name)

    def load_model(self, model_path, blocks=['local', 'global', 'predict']):
        # checkpoint = torch.load(
        #     './outputs/%s/models/%s_model.pth.tar' % (train_name, model_name))
        checkpoint = torch.load(model_path)
        if 'local' in blocks:
            if self.parallel:
                self.local_structure_embeding_block.load_state_dict(
                    checkpoint['local'])
                self.local_structure_embeding_blocks = nn.DataParallel(
                    self.local_structure_embeding_block)
            else:
                self.local_structure_embeding_blocks.load_state_dict(
                    checkpoint['local'])
        if 'global' in blocks:
            self.local_structure_feature_globalizing_blocks.load_state_dict(
                checkpoint['global'])
        if 'predict' in blocks:
            self.predicting_blocks.load_state_dict(checkpoint['predict'])

    def freeze_blocks(self, blocks):
        if 'local' in blocks:
            freeze_parameter(self.local_structure_embeding_blocks)
        if 'global' in blocks:
            freeze_parameter(
                self.local_structure_feature_globalizing_blocks)
        if 'predict' in blocks:
            freeze_parameter(self.predicting_blocks)

    def activate_blocks(self, blocks):
        if 'local' in blocks:
            activate_parameter(self.local_structure_embeding_blocks)
        if 'global' in blocks:
            activate_parameter(
                self.local_structure_feature_globalizing_blocks)
        if 'predict' in blocks:
            activate_parameter(self.predicting_blocks)


class DeepPBSMean(DeepPBS):
    def forward(self, x, lens=None):
        x = self.local_structure_embeding_blocks(x)
        x = self.local_structure_feature_globalizing_blocks(x)
        x = torch.mean(x, axis=0)
        x = torch.unsqueeze(x, 0)
        x = self.predicting_blocks(x)
        return x


# FOR COMPATIBILITY
def swish_fn(x):
    """ Swish activation function """
    return x * torch.sigmoid(x)


# FOR COMPATIBILITY
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, feature_dim, output_dim):
        super().__init__()

        self.hidden1 = nn.Linear(input_dim, hidden_dim)
        self._bn1 = nn.BatchNorm1d(hidden_dim)

        self.hidden2 = nn.Linear(hidden_dim, 2 * hidden_dim)
        self._bn2 = nn.BatchNorm1d(2 * hidden_dim)

        self.hidden3 = nn.Linear(2 * hidden_dim, hidden_dim)
        self._bn3 = nn.BatchNorm1d(hidden_dim)

        self.extract_feature = nn.Linear(hidden_dim, feature_dim)
        self._bn4 = nn.BatchNorm1d(feature_dim)

        self.lstm = nn.LSTM(feature_dim, hidden_dim, bidirectional=True)
        self._bn5 = nn.BatchNorm1d(2 * hidden_dim)

        # self.sub_net1 = nn.Linear(2 * hidden_dim, hidden_dim)
        # self._bn_s1 = nn.BatchNorm1d(hidden_dim)
        # self.output1 = nn.Linear(hidden_dim, output_dim)
        #   
        # self.sub_net2 = nn.Linear(2 * hidden_dim, hidden_dim)
        # self._bn_s2 = nn.BatchNorm1d(hidden_dim)
        # self.output2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, arrays):
        hidden1 = swish_fn(self._bn1(self.hidden1(arrays)))
        hidden2 = swish_fn(self._bn2(self.hidden2(hidden1)))
        hidden3 = swish_fn(self._bn3(self.hidden3(hidden2)))
        features = swish_fn(self._bn4(self.extract_feature(hidden3)))

        hidden, (hn, cn) = self.lstm(features.view(len(features), 1, -1))
        hidden = swish_fn(hidden.squeeze(1))
        # hidden = swish_fn(self._bn5(hidden.squeeze(1)))
        hn = swish_fn(hn.squeeze(1))
        cn = swish_fn(cn.squeeze(1))
        #   
        # sub_hidden1 = swish_fn(self._bn_s1(self.sub_net1(hidden)))
        # # sub_hidden1 = F.dropout(sub_hidden1, p=0.5, training=self.training)
        # output1 = self.output1(sub_hidden1)
        #   
        # sub_hidden2 = swish_fn(self._bn_s2(self.sub_net2(hidden)))
        # # sub_hidden2 = F.dropout(sub_hidden2, p=0.5, training=self.training)
        # output2 = self.output1(sub_hidden2)
        #   
        # output = torch.cat([output1, output2], 1)
        return hidden, hn, cn

