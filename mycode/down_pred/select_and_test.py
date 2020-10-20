import os
import argparse
import json
import platform
from predict_test_set import predict_test_set

parser = argparse.ArgumentParser(description='Select a model with best performance on the valid set and test it on a test set.')
parser.add_argument('-t', '--task', help='Task name, e.g. gfp, protherm, rocklin', required=True)
parser.add_argument('-i', '--index', help='Train index, e.g. 20040401', required=True)
parser.add_argument('-c', '--criterion', help='Criterion used to select model, e.g. r, rho, val_loss', required=True)
parser.add_argument('-cv', '--cv', help='Index of part in a cross validation', default=None)
parser.add_argument('-g', '--gpu', help='GPU to use to run prediction on the test set, e.g. 0', default='6')
parser.add_argument('-n', '--number', help='Number of best models to use to test.', default=10)
parser.add_argument('-he', '--head', help='Only select models from first N epoches', default=None)
parser.add_argument('-a', '--attention', help='Whether to output the attention weights', action='store_true')
parser.add_argument('-l', '--test_set_list', help='Custom test set list path', default=None)
args = parser.parse_args()

task_name = args.task
train_index = args.index
criterion = args.criterion
n = args.number
att = args.attention
custom_test_set = args.test_set_list

param_file_path = '/home/joseph/KNN/down_pred/parameters.json'
with open(param_file_path, 'r') as param_file:
    p0 = json.load(param_file)

val_loss_filename = 'valid_loss.txt'
output_path = p0[platform.system() + '_output_path']
test_result_path = p0[platform.system() + '_test_result_path']
if args.cv:
    train_index = f'{train_index}-{args.cv}'
val_loss_path = os.path.join(output_path, '%s__%s' % (task_name, train_index), val_loss_filename)
val_loss_file = open(val_loss_path, 'r')
lines = val_loss_file.readlines()
val_loss = {}
r = {}
rho = {}
epoches = {}
line_list = []

for line in lines:
    line_list.append(line.split())
i = 0
while i < len(line_list):
    if line_list[i][0] == 'epoch':
        if line_list[i+1][0] == "val_loss_average" and line_list[i+2][0] == \
        'Pearson_number' and line_list[i+3][0] == 'Spearman_number':
            epoch = int(line_list[i][1])
            if epoch in epoches:
                raise Exception('Repeated epoch!')
            else:
                epoches[epoch] = True
            val_loss[float(line_list[i+1][2])] = epoch
            r[float(line_list[i+2][2])] = epoch
            rho[float(line_list[i+3][2])] = epoch
            i += 4
        else:
            raise Exception('Wrong file format!')

min_loss = sorted(val_loss.keys(), reverse=False)[:n]
max_r = sorted(r.keys(), reverse=True)[:n]
max_rho = sorted(rho.keys(), reverse=True)[:n]
print('max_r', [r[i] for i in max_r], max_r)
print('max_rho', [rho[i] for i in max_rho], max_rho)
print(f'min_loss {[val_loss[i] for i in min_loss]} {min_loss}\n')

if criterion == 'r':
    valid_performs = max_r
    models = [r[i] for i in valid_performs]
elif criterion == 'rho':
    valid_performs = max_rho
    models = [rho[i] for i in valid_performs]
    print(valid_performs, models)
elif criterion == 'val_loss':
    valid_performs = min_loss
    models = [val_loss[i] for i in valid_performs]
else:
    raise Exception('No such criterion!')

dataset_names, sub_dataset_names, network_name, description, shape, pearson, spearman = predict_test_set(
    task_name, train_index, models, args.gpu, att, args.cv, custom_test_set)

if args.cv:
    description += f'-{args.cv}'
with open(os.path.join(test_result_path, 'test_results.txt'), 'a') as sum_file:
    sum_file.write(f'{train_index}\t{task_name}\t{dataset_names}\t{shape}\t{criterion}\t'
                   f'{models}\t{valid_performs[0]:.4f}\t{pearson:.4f}\t{spearman:.4f}\t{sub_dataset_names}\t'
                   f'{network_name}\t{description}\t{custom_test_set}\n')