### 代码文件的说明

#### 预训练

pretrain.py：用于自监督预训练

self_embedding.py：用于自监督预训练的embedding

rep_utils.py：KNN和图像表征中使用到的函数和模块，被pretrain.py和self_embedding.py调用

#### 下游任务的训练

train.py：下游任务的训练。依赖于dataset.py和network.py，以及parameters.json

dataset.py：下游任务的不同表征的Dataset Class，使用其中的get_class_name()函数解析parameters.json中的参数，输出Dataset的Class name

network.py：下游任务的训练时使用的不同网络的模块，使用其中的`network_dict`将parameters.json中的参数转化为network的Class name

parameters.json：下游任务的文件路径、训练参数等。其中各个参数的解释见"parameters.json"章节

#### 下游任务的测试

select_and_test.py：选择最佳模型并将模型的信息提供给predict_test_set.py进行测试

predict_test_set.py：使用给定的模型，在给定的测试集上测试。本文件被select_and_test.py调用，不能直接运行。

classify.py：需要预先运行select_and_test.py。在使用\home\caiyi\test_result\中的npy文件，进行下游任务（仅适用于GFP任务）的分类。

classify_rocklin.py：rocklin任务的分类。由于此任务对于不同样本的分类阈值不同，因此不能使用classify.py，而应使用本文件。

#### 作图

loss_curve.py：损失曲线

scatter_plot.py：散点图

#### 其他

embedding.py：从PDB到KNN或图像的表征。

process_pdb.py：被embedding.py调用，不单独使用。

deeppbs.py：DeepPBS的Dataset和Network Module。使用DeepPBS进行预训练时被调用。不单独使用

### 文件存放位置

#### 代码

训练和测试需要用到的所有代码\home\caiyi\workspace\

#### 训练所需数据

训练和测试需要用到的所有数据位于\home\caiyi\data\

##### 各个训练任务

此文件夹下gfp, rocklin, protherm等文件夹分别代表GFP, 稳定性等不同任务的数据

##### 各个表征

对于每个任务的文件夹，如\home\caiyi\data\rocklin\中，不同文件夹代表不同的表征，如knn_self_512_full代表full size的自监督的512维的表征，tape代表平均后的TAPE表征，等等。

##### 同一表征的不同子集

对于每个表征的文件夹，如\home\caiyi\data\rocklin\knn_self_512_full\中，不同文件夹代表该表征的不同子集，如pt_082602表示8月26日训练的第2个预训练模型所得到的表征。

##### 原始数据

对于每个任务的文件夹，如\home\caiyi\data\rocklin\中的src文件夹，其中是真实值数据、蛋白质序列、PDB文件、数据集中的数据列表（也包括交叉验证用的数据集列表）等原始数据。

##### 预训练数据

预训练用到的NR40数据集位于\home\caiyi\data\nr40\。

#### 训练输出

预训练和下游任务训练的训练结果（训练/验证Loss，和每个epoch保存的模型）保存在\home\caiyi\output\文件夹

#### 测试结果

下游任务的所有测试结果保存在\home\caiyi\test_result文件夹，其中的\home\caiyi\test_result\test_result.txt文件中是测试的文本结果，其他npy文件是用于分类和画散点图的测试结果。

### 训练及测试

#### 预训练

```python
python pretrain.py --path_file /home/caiyi/data/nr40/knn_150 --seq_file /home/caiyi/data/nr40/seq_fa/pdb.fasta --subset_name 20082602 --device_num 0
```

#### Embedding

```python
python self_embedding.py
```

文件位置等参数调整直接在py文件中修改。

#### 下游任务的训练

```python
python train.py
```

train.py依赖于parameters.json，各种参数的调整在json文件中进行。json文件会被自动保存在输出文件夹。

#### 下游任务的训练：交叉验证

以五折交叉验证为例

```python
python train.py --cv 0 --gpu 0 &
python train.py --cv 1 --gpu 1 &
python train.py --cv 2 --gpu 2 &
python train.py --cv 3 --gpu 3 &
python train.py --cv 4 --gpu 4 &
```

可以将以上python代码写入shell脚本cv.sh中，然后运行

```shell
sh cv.sh
```

#### 下游任务的测试

```python
python select_and_test.py --task rocklin --index 20082602 --criterion rho
```

测试结果写入/home/caiyi/test_results/test_results.txt

#### 下游任务的测试：交叉验证

```python
python select_and_test.py --task rocklin --index 20080403-0 --criterion rho &
python select_and_test.py --task rocklin --index 20080403-1 --criterion rho &
python select_and_test.py --task rocklin --index 20080403-2 --criterion rho &
python select_and_test.py --task rocklin --index 20080403-3 --criterion rho &
python select_and_test.py --task rocklin --index 20080403-4 --criterion rho &
```

可以将以上python代码写入shell脚本batch_test.sh中，然后运行

```shell
sh cv.sh
```

#### 下游任务的分类



```python
python classify.py --task gfp --index 20072404 --model 
```



### parameters.json

#### 数据文件的载入

`task_name`：训练任务

`dataset_names`：使用的表征（是列表，因为可以结合多种表征），使用的表征需要在dataset.py中有被实现

`sub_dataset_names`：使用的表征子集（如`"pt_082602"`）

`combining_method`：当多种表征结合时，结合的方式（`"vstack"`或`"concat"`）

`network`：使用的网络，需要在network.py中有被实现

`real_value_path`：训练目标文件（真实值）的路径

`real_value_column`：真实值位于训练目标文件的哪一列（从0开始），对于rocklin任务，为`2`，对于gfp任务，为`1`

`train_set_paths`：训练输入的npy文件存于表征的文件夹（如\home\caiyi\data\rocklin\knn_self_512_full\）下的位置，如直接存于表征的文件夹，则为`""`

`valid_set_paths`, `test_set_paths`：同上

`train_set_list_path`：训练集的列表文件，格式为每一行为一个训练样本的文件名（包括扩展名）

`valid_set_list_path`, `test_set_list_path`：同上

`loaded_pretrained_model`, `loaded_downstream_model`：fine-tune时使用，一般情况请忽略

`Linux_data_path`：数据文件夹的路径，一般为`"/home/caiyi/data/"`

`Linux_output_path`：输出文件夹的路径，一般为`"/home/caiyi/outputs/"`

`pretrained_models_path`：fine-tune时使用，一般可以忽略

#### 训练参数

`auto_end`：是否根据validation loss的收敛情况自动终止训练

`target_epochs`：训练的目标epoch数。当`auto_end`为`true`时，此字段不生效，可填为`null`

`criterion_to_end`：判断是否终止训练的条件，可以是`"r"`，`"rho"`，`"val_loss"`

`epoches_to_end`, `threshold_of_convergence`：当连续`epoches_to_end`个epoch在验证集上的r或rho没有提高超过`threshold_of_convergence`时（或者val_loss没有降低超过`threshold_of_convergence`时），停止训练

`regularization_param`：正则化系数，不使用正则化时，为`0`

#### 其他

`description`：对于此次训练的较详细的描述

