"""
Downstream network modules
TODO:
1. use nn.Sequential() to rebuild and see what happens
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.weight_norm import weight_norm

network_dict = {'our': 'MLP', 'tape': 'ValuePredictionHead', 'relu': 'MLPReLU', 'relu_wn': 'MLPReLUWeightNorm',
                '3hid_relu': 'MLP3HiddenReLU', '3hid_relu_1': 'MLP3HiddenReLU1', '3hid_relu_2': 'MLP3HiddenReLU2',
                '4hid_relu': 'MLP4HiddenReLU', 'att_relu': 'MLPAttentionReLU', 'att_sig': 'MLPAttentionSigmoid',
                'fixed_sig': 'MLPFixedHiddenSigmoid', 'fixed_relu': 'MLPFixedHiddenReLU', '2fixed_sig':
                'MLP2FixedHiddenSigmoid', '2fixed_relu': 'MLP2FixedHiddenReLU', 'att_fixed_sig':
                'MLPAttentionFixedHiddenSigmoid', 'att_fixed_relu': 'MLPAttentionFixedHiddenReLU', 'att_sig_test1':
                'MLPAttentionSigmoidRocklinTest1', 'att_sig_test2': 'MLPAttentionSigmoidRocklinTest2', 'no_pretrain':
                'NoPretrain'}

# 4 Layers (2 hidden)
class MLP(nn.Module):
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


class MLPFixedHiddenSigmoid(nn.Module):
    def __init__(self, input_dim, stacked_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, arrays):
        hidden = self.ln(torch.sigmoid(self.fc1(arrays)))
        output = self.output(hidden)
        return output


class MLP2FixedHiddenSigmoid(nn.Module):
    def __init__(self, input_dim, stacked_dim, hidden_dim=512):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.output = nn.Linear(hidden_dim, 1)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim // 2)

    def forward(self, arrays):
        hidden1 = self.ln1(torch.sigmoid(self.fc1(arrays)))
        hidden2 = self.ln2(torch.sigmoid(self.fc2(hidden1)))
        output = self.output(hidden2)
        return output


class MLPFixedHiddenReLU(nn.Module):
    def __init__(self, input_dim, stacked_dim, hidden_dim=512):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(stacked_dim, 1)

        self.output = nn.Linear(hidden_dim, 1)
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, arrays):
        hidden1 = self.ln(torch.relu(self.fc1(arrays)))
        hidden2 = torch.sigmoid(self.fc2(hidden1.transpose(0, 1)))
        output = self.output(hidden2.transpose(0, 1))
        return output


class MLP2FixedHiddenReLU(nn.Module):
    def __init__(self, input_dim, stacked_dim, hidden_dim=512):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.output = nn.Linear(hidden_dim, 1)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim // 2)

    def forward(self, arrays):
        hidden1 = self.ln1(torch.relu(self.fc1(arrays)))
        hidden2 = self.ln2(torch.relu(self.fc2(hidden1)))
        output = self.output(hidden2)
        return output


class MLP3HiddenReLU(nn.Module):
    def __init__(self, input_dim, stacked_dim):
        super().__init__()
        hidden_dim = input_dim // 2
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, hidden_dim)
        self.fc3 = nn.Linear(stacked_dim, 1)
        self.output = nn.Linear(hidden_dim, 1)
        self.ln1 = nn.LayerNorm(input_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

    def forward(self, arrays):
        hidden1 = self.ln1(torch.relu(self.fc1(arrays)))
        hidden2 = self.ln2(torch.relu(self.fc2(hidden1)))
        hidden3 = torch.relu(self.fc3(hidden2.transpose(0, 1)))
        output = self.output(hidden3.transpose(0, 1)) 
        return output


class MLP3HiddenReLU1(nn.Module):
    def __init__(self, input_dim, stacked_dim):
        super().__init__()
        hidden_dim = input_dim // 2
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.output = nn.Linear(stacked_dim, 1)
        self.ln1 = nn.LayerNorm(input_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

    def forward(self, arrays):
        hidden1 = self.ln1(torch.relu(self.fc1(arrays)))
        hidden2 = self.ln2(torch.relu(self.fc2(hidden1)))
        hidden3 = torch.relu(self.fc3(hidden2))
        output = self.output(hidden3.transpose(0, 1))
        return output


class MLP3HiddenReLU2(nn.Module):
    def __init__(self, input_dim, stacked_dim):
        super().__init__()
        hidden_dim1 = input_dim // 2
        hidden_dim2 = hidden_dim1 // 2
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, 1)
        self.output = nn.Linear(stacked_dim, 1)
        self.ln1 = nn.LayerNorm(hidden_dim1)
        self.ln2 = nn.LayerNorm(hidden_dim2)

    def forward(self, arrays):
        hidden1 = self.ln1(torch.relu(self.fc1(arrays)))
        hidden2 = self.ln2(torch.relu(self.fc2(hidden1)))
        hidden3 = torch.relu(self.fc3(hidden2))
        output = self.output(hidden3.transpose(0, 1))
        return output


class MLP4HiddenReLU(nn.Module):
    def __init__(self, input_dim, stacked_dim):
        super().__init__()
        hidden_dim1 = input_dim // 2
        hidden_dim2 = hidden_dim1 // 2
        hidden_dim3 = hidden_dim2 // 2
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.fc4 = nn.Linear(hidden_dim3, 1)
        self.output = nn.Linear(stacked_dim, 1)
        self.ln1 = nn.LayerNorm(hidden_dim1)
        self.ln2 = nn.LayerNorm(hidden_dim2)
        self.ln3 = nn.LayerNorm(hidden_dim3)

    def forward(self, arrays):
        hidden1 = self.ln1(torch.relu(self.fc1(arrays)))
        hidden2 = self.ln2(torch.relu(self.fc2(hidden1)))
        hidden3 = self.ln3(torch.relu(self.fc3(hidden2)))
        hidden4 = torch.relu(self.fc4(hidden3))
        output = self.output(hidden4.transpose(0, 1))
        return output


# Network that TAPE used
class SimpleMLP(nn.Module):
    def __init__(self,
                 in_dim: int,
                 hid_dim: int,
                 out_dim: int,
                 dropout: float = 0.):
        super().__init__()
        self.main = nn.Sequential(
            weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=True),
            weight_norm(nn.Linear(hid_dim, out_dim), dim=None))

    def forward(self, x):
        return self.main(x)


class ValuePredictionHead(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.):
        super().__init__()
        self.value_prediction = SimpleMLP(hidden_size, 512, 1, dropout)

    def forward(self, pooled_output, targets=None):
        value_pred = self.value_prediction(pooled_output)
        outputs = (value_pred,)

        if targets is not None:
            loss_fct = nn.MSELoss()
            value_pred_loss = loss_fct(value_pred, targets)
            outputs = (value_pred_loss,) + outputs
        print(type(output[0]))
        return outputs[0], None   # (loss), value_prediction


class MLPReLU(nn.Module):
    def __init__(self, input_dim, stacked_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.ln = nn.LayerNorm(input_dim)
        self.fc2 = nn.Linear(stacked_dim, 1)
        self.output = nn.Linear(input_dim, 1)

    def forward(self, arrays):
        hidden1 = self.ln(torch.relu(self.fc1(arrays)))
        hidden2 = torch.relu(self.fc2(hidden1.transpose(0, 1)))
        output = self.output(hidden2.transpose(0, 1))
        return output


# Softmax函数参数已更改，测试时注意！
class MLPAttentionReLU(nn.Module):
    def __init__(self, input_dim, stacked_dim):
        super().__init__()
        self.fc0 = nn.Linear(input_dim, 1)
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.ln = nn.LayerNorm(input_dim)
        self.output = nn.Linear(input_dim, 1)

    def forward(self, arrays):
        # align = F.softmax(self.fc0(arrays))
        align = F.softmax(self.fc0(arrays), dim=0)
        hidden0 = torch.mm(arrays.transpose(0, 1), align)
        hidden1 = self.ln(torch.relu(self.fc1(hidden0.transpose(0, 1))))
        output = self.output(hidden1)
        return output

    def attention_weight(self, arrays):
        align = F.softmax(self.fc0(arrays), dim=0)
        return align


# Softmax函数参数已更改，测试时注意！
class MLPAttentionSigmoid(nn.Module):
    def __init__(self, input_dim, stacked_dim):
        super().__init__()
        self.fc0 = nn.Linear(input_dim, 1)
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.ln = nn.LayerNorm(input_dim)
        self.output = nn.Linear(input_dim, 1)

    def forward(self, arrays):
        # align = F.softmax(self.fc0(arrays))
        align = F.softmax(self.fc0(arrays), dim=0)
        hidden0 = torch.mm(arrays.transpose(0, 1), align)
        hidden1 = self.ln(torch.sigmoid(self.fc1(hidden0.transpose(0, 1))))
        output = self.output(hidden1)
        return output


class MLPAttentionSigmoidRocklinTest1(nn.Module):
    def __init__(self, input_dim, stacked_dim):
        super().__init__()
        self.fc0 = nn.Linear(input_dim, 1)
        self.fc1 = nn.Linear(input_dim, 256)
        self.output = nn.Linear(256, 1)
        self.ln = nn.LayerNorm(256)


    def forward(self, arrays):
        # align = F.softmax(self.fc0(arrays))
        align = F.softmax(self.fc0(arrays), dim=0)
        hidden0 = torch.mm(arrays.transpose(0, 1), align)
        hidden1 = self.ln(torch.sigmoid(self.fc1(hidden0.transpose(0, 1))))
        output = self.output(hidden1)
        return output


class MLPAttentionSigmoidRocklinTest2(nn.Module):
    def __init__(self, input_dim, stacked_dim):
        super().__init__()
        self.fc0 = nn.Linear(input_dim, 1)
        self.fc1 = nn.Linear(input_dim, 1)
        self.ln = nn.LayerNorm(1)


    def forward(self, arrays):
        # align = F.softmax(self.fc0(arrays))
        align = F.softmax(self.fc0(arrays), dim=0)
        hidden0 = torch.mm(arrays.transpose(0, 1), align)
        hidden1 = self.ln(torch.sigmoid(self.fc1(hidden0.transpose(0, 1))))
        return hidden1


class MLPAttentionFixedHiddenSigmoid(nn.Module):
    def __init__(self, input_dim, stacked_dim, hidden_dim=512):
        super().__init__()
        self.fc0 = nn.Linear(input_dim, 1)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, arrays):
        # align = F.softmax(self.fc0(arrays))
        align = F.softmax(self.fc0(arrays), dim=0)
        hidden0 = torch.mm(arrays.transpose(0, 1), align)
        hidden1 = self.ln(torch.sigmoid(self.fc1(hidden0.transpose(0, 1))))
        output = self.output(hidden1)
        return output


class MLPAttention2FixedHiddenSigmoid(nn.Module):
    def __init__(self, input_dim, stacked_dim, hidden_dim=512):
        super().__init__()
        self.fc0 = nn.Linear(input_dim, 1)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim // 2)
        self.output = nn.Linear(hidden_dim // 2, 1)

    def forward(self, arrays):
        # align = F.softmax(self.fc0(arrays))
        align = F.softmax(self.fc0(arrays), dim=0)
        hidden0 = torch.mm(arrays.transpose(0, 1), align)
        hidden1 = self.ln1(torch.sigmoid(self.fc1(hidden0.transpose(0, 1))))
        hidden2 = self.ln2(torch.sigmoid(self.fc2(hidden1)))
        output = self.output(hidden1)
        return output

class MLPAttention2FixedHiddenReLU(nn.Module):
    def __init__(self, input_dim, stacked_dim, hidden_dim=512):
        super().__init__()
        self.fc0 = nn.Linear(input_dim, 1)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim // 2)
        self.output = nn.Linear(hidden_dim // 2, 1)

    def forward(self, arrays):
        # align = F.softmax(self.fc0(arrays))
        align = F.softmax(self.fc0(arrays), dim=0)
        hidden0 = torch.mm(arrays.transpose(0, 1), align)
        hidden1 = self.ln1(torch.relu(self.fc1(hidden0.transpose(0, 1))))
        hidden2 = self.ln2(torch.relu(self.fc2(hidden1)))
        output = self.output(hidden1)
        return output

    def attention_weight(self, arrays):
        align = F.softmax(self.fc0(arrays), dim=0)
        return align


def swish_fn(x):
    """ Swish activation function """
    return x * torch.sigmoid(x)


class NoPretrain(nn.Module):
    def __init__(self, input_dim=135, hidden_dim=256, feature_dim=256, output_dim=20, stacked_dim=None):
        super().__init__()

        self.input = nn.Linear(input_dim, hidden_dim // 2)
        self.ln1 = nn.LayerNorm(hidden_dim // 2)

        self.hidden1 = nn.Linear(hidden_dim // 2, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

        self.lstm = nn.LSTM(feature_dim, hidden_dim, bidirectional=True)

        self.mlp = MLPAttentionSigmoid(hidden_dim * 2, None) 

    def forward(self, arrays):
        hidden_states = swish_fn(self.ln1(self.input(arrays)))
        hidden_states = swish_fn(self.ln2(self.hidden1(hidden_states)))
        hidden_states, _ = self.lstm(hidden_states.view(len(hidden_states), 1, -1))
        hidden_states = hidden_states.squeeze(1)
        output = self.mlp(hidden_states)

        return output


# 错误，待改
class MLPReLUWeightNorm(nn.Module):
    def __init__(self, hidden_dim, stacked_dim):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        # self.wn = weight_norm()
        self.fc2 = nn.Linear(stacked_dim, 1)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, arrays):
        hidden1 = torch.relu(weight_norm(self.fc1(arrays), dim=None))
        hidden2 = torch.relu(weight_norm(self.fc2(hidden1.transpose(0, 1)), dim=None))
        output = self.output(hidden2.transpose(0, 1))
        return output


# Deprecated. Only for compatibility
class SplitModel1(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden4 = nn.Linear(hidden_dim, hidden_dim)  # 隐藏层的线性输出
        self._bn6 = nn.LayerNorm(hidden_dim)
        self.hidden5 = nn.Linear(1, 1)  # 隐藏层的线性输出
        self.predict = nn.Linear(hidden_dim, 1)  # 输出层的线性输出

    def forward(self, arrays):
        hidden4 = self._bn6(F.sigmoid(self.hidden4(arrays)))
        hidden5 = F.sigmoid(self.hidden5(hidden4.transpose(0, 1)))
        output = self.predict(hidden5.transpose(0, 1)) 
        return output, hidden5


class SplitModel3(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden4 = nn.Linear(hidden_dim, hidden_dim)  # 隐藏层的线性输出
        self._bn6 = nn.LayerNorm(hidden_dim)
        self.hidden5 = nn.Linear(3, 1)  # 隐藏层的线性输出
        self.predict = nn.Linear(hidden_dim, 1)  # 输出层的线性输出

    def forward(self, arrays):
        hidden4 = self._bn6(F.sigmoid(self.hidden4(arrays)))
        hidden5 = F.sigmoid(self.hidden5(hidden4.transpose(0, 1)))
        output = self.predict(hidden5.transpose(0, 1)) 
        return output, hidden5


class SplitModel4(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden4 = nn.Linear(hidden_dim, hidden_dim)  # 隐藏层的线性输出
        self._bn6 = nn.LayerNorm(hidden_dim)
        self.hidden5 = nn.Linear(4, 1)  # 隐藏层的线性输出
        self.predict = nn.Linear(hidden_dim, 1)  # 输出层的线性输出

    def forward(self, arrays):
        hidden4 = self._bn6(F.sigmoid(self.hidden4(arrays)))
        hidden5 = F.sigmoid(self.hidden5(hidden4.transpose(0, 1)))
        output = self.predict(hidden5.transpose(0, 1)) 
        return output, hidden5


class SplitModel11(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden4 = nn.Linear(hidden_dim, hidden_dim)  # 隐藏层的线性输出
        self._bn6 = nn.LayerNorm(hidden_dim)
        self.hidden5 = nn.Linear(11, 1)  # 隐藏层的线性输出
        self.predict = nn.Linear(hidden_dim, 1)  # 输出层的线性输出

    def forward(self, arrays):
        hidden4 = self._bn6(F.sigmoid(self.hidden4(arrays)))
        hidden5 = F.sigmoid(self.hidden5(hidden4.transpose(0, 1)))
        output = self.predict(hidden5.transpose(0, 1)) 
        return output, hidden5


class SplitModel19(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden4 = nn.Linear(hidden_dim, hidden_dim)  # 隐藏层的线性输出
        self._bn6 = nn.LayerNorm(hidden_dim)
        self.hidden5 = nn.Linear(19, 1)  # 隐藏层的线性输出
        self.predict = nn.Linear(hidden_dim, 1)  # 输出层的线性输出

    def forward(self, arrays):
        hidden4 = self._bn6(F.sigmoid(self.hidden4(arrays)))
        hidden5 = F.sigmoid(self.hidden5(hidden4.transpose(0, 1)))
        output = self.predict(hidden5.transpose(0, 1)) 
        return output, hidden5

