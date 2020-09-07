"""
embedding.py
TODO:
1. get stacked vec for DeepPBS
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import os
from torch.nn import functional as F
import numpy as np
from typing import Dict, Tuple, Union

from process_pdb import process_pdb
import deeppbs
import rep_utils

dataset_name = 'knn_150'

get_rep_p = True
get_vector_p = False
single_chain_pdb = True
mut_p = True
append_knn = False
load_coo = False

# part = 'xah'
# model_path = '/home/caiyi/data/pretrained_models/49_model.pth.tar'
pdb_path = '/home/caiyi/data/rocklin/src/ssm_pdb/'
# pdb_path = '/home/caiyi/data/rocklin/src/pdb/'
# pdb_path = '/home/eric/AttnPBS/attnpbs/dataset/data/nr40/knn/'
# pdb_path = '/home/caiyi/data/nr40/coo_att/'
# pdb_path = '/home/caiyi/workspace/pdb2vec/test/pdb/'
seq_path = '/home/caiyi/data/rocklin/src/mut_seq/'
rep_path = '/home/caiyi/data/rocklin/knn_150/'
# rep_path = '/home/caiyi/data/nr40/knn_180/'
# rep_path = '/home/caiyi/temp/test/test_knn/'# + part
# output_path = '/home/caiyi/temp/test/test_full/'
# output_path = '/home/caiyi/data/rocklin/ssm_knn_full/'# + part
# avg_h_path = '/home/caiyi/temp/test/test_vec/'
# avg_h_path = '/home/caiyi/data/rocklin/ssm_knn_vec/'#+part
# stacked_path = '/home/caiyi/temp/test/test_stacked/'
# stacked_path = '/home/caiyi/data/rocklin/ssm_knn_768/'#+part
# chains_path = '/home/caiyi/data/protherm/src/chain_to_extract.txt'

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
device_name = 'cuda:0'
device = torch.device(device_name)

rep_dict = {'knn_60': 'get_knn', 'knn_75': 'KNNStructRepRelative', 'image_ori': 'ImageStructRep', 'knn_135':
            'get_knn_135', 'knn_150': 'get_knn_150', 'knn_180': 'get_knn_180'}
dataset_dict = {'knn_60': 'DistanceWindow', 'knn_75': 'KNN2tor', 'image_ori': 'Image2tor',
                'knn_75_batch': 'KNN2tor_batch', 'knn_3': 'KNN2TorNoStruct'}
nn_dict = {'knn_75': 'DeepPBS', 'image_ori': 'DeepPBS', 'knn_60': 'LSTM', 'knn_75_batch': 'DeepPBS', 'knn_3': ''}
block_list_dict = {'knn_60': None, 'knn_75': ['MLP', 'LSTM', 'Joint'], 'image_ori': ['ConvOld', 'LSTM', 'Split'],
                   'knn_75_batch': ['MLP', 'BatchLSTM', 'BatchJoint']}
dims_dict = {'knn_60': None, 'knn_75': [75, 1024, 1024, 4], 'image_ori': [5, 512, 512, 4],
             'knn_75_batch': [75, 1024, 1024, 4]}
if get_vector_p:
    exec('network = deeppbs.%s' % nn_dict[dataset_name])
    exec('dataset = deeppbs.%s' % dataset_dict[dataset_name])
    block_list = block_list_dict[dataset_name]
    dims = dims_dict[dataset_name]

chains_to_extract: Dict[str, Tuple[str, str]] = {}  # {pdb_name -> (selected_chain, selected_model)}

if not single_chain_pdb:
    chain_file = open(chains_path, 'r')
    chain_data = chain_file.readlines()
    chain_data = [line.split() for line in chain_data]
    for pdb in range(len(chain_data)):
        chains_to_extract[chain_data[pdb][0]] = (chain_data[pdb][1], chain_data[pdb][2])
    chain_file.close()


def pdb2rep(pdb_path, rep_path, single_chain_pdb, chains_to_extract, mut_p):
    if dataset_name in ['knn_135', 'knn_150']:
        atoms_type = ['N', 'CA', 'C', 'O']
    else:
        atoms_type = ['CA']

    for pdb_filename in os.listdir(pdb_path):
        pdb_name = str(pdb_filename).split('.')[0]
        print(pdb_name)
        if append_knn:
            knn_array = np.load('/home/eric/AttnPBS/attnpbs/dataset/data/nr40/knn/' + pdb_name + '.npy')
            if load_coo:
                coord_array = np.load('/home/eric/AttnPBS/attnpbs/dataset/data/nr40/coo/' + pdb_name + '.npy')
                coord_array = (knn_array, coord_array)
                acid_array = None
            else:
                pdb_profile, atom_lines = process_pdb(os.path.join(pdb_path, pdb_filename), atoms_type=atoms_type)
                if single_chain_pdb:
                    chain, model = 'A', '1' 
                else:
                    chain, model = chains_to_extract[pdb_name]
                atoms_data = atom_lines[chain, model]
                coord_array_ca, acid_array, coord_array = rep_utils.extract_coord(atoms_data, atoms_type=atoms_type)
                coord_array = (knn_array, coord_array)
                acid_array = None
        elif load_coo:
            coord_array = np.load('/home/eric/AttnPBS/attnpbs/dataset/data/nr40/coo/' + pdb_name + '.npy')
            acid_array = rep_utils.seq2array(list(rep_utils.read_fasta(os.path.join('/home/caiyi/data/nr40/seq_fa/',
                                                              pdb_name + '.fasta')).values())[0])
        else:
            pdb_profile, atom_lines = process_pdb(os.path.join(pdb_path, pdb_filename), atoms_type=atoms_type)

            if single_chain_pdb:
                chain, model = 'A', '1'
            else:
                chain, model = chains_to_extract[pdb_name]

            atoms_data = atom_lines[chain, model]
            print('A', len(atoms_data))
            coord_array_ca, acid_array, coord_array = rep_utils.extract_coord(atoms_data, atoms_type=atoms_type)
            print('B', coord_array.shape, acid_array.shape)

            # knn = get_knn(coord_array, acid_array)
            # np.save(os.path.join(knn_path, pdb_name + '.npy'), knn)
            rep_utils.compare_len(coord_array, acid_array, atoms_type)

        rep = get_rep(coord_array, acid_array)
        if dataset_name == 'image':
            print(pdb_filename, coord_array.shape, rep)
        elif append_knn:
            print(pdb_filename, coord_array[0].shape, rep.shape)
        else:
            print(pdb_filename, coord_array.shape, rep.shape)
        np.save(os.path.join(rep_path, pdb_name + '.pdb.npy'), rep)

        if mut_p:
            seq_dict: Dict[str, Union[str, np.ndarray]] = rep_utils.read_fasta(os.path.join(seq_path, pdb_name + '.faa'))
            # Convert str to np.ndarray
            for seq_name in seq_dict:
                # seq = seq_dict[seq_name]
                # print(align(array2seq(aa_names), seq)
                seq_dict[seq_name] = rep_utils.seq2array(seq_dict[seq_name])
            for seq_name in seq_dict:
                rep = get_rep(coord_array, seq_dict[seq_name])
                np.save(os.path.join(rep_path, seq_name + '.npy'), rep)
                print(pdb_filename, coord_array.shape, seq_dict[seq_name].shape)
                # print(pdb_filename, seq_name, 'Succeeded!')


def rep2vec(rep_path, block_list):
    test_dataset = dataset(path={dataset_name: rep_path}, list_path=None)
    data_loader = DataLoader(dataset=test_dataset, shuffle=True, num_workers=64, pin_memory=True)

    with torch.no_grad():
        block_dict = {'local': block_list[0], 'global': block_list[1], 'predict': block_list[2]}
        model = network(blocks=block_dict, dims=dims)
        print(model)
        model.load_model(model_path=model_path)
        model.eval()
        model.is_training = False

        print('Embedding begins:')
        for arrays, output_filename in data_loader:
            arrays = arrays.to(device)
            _, output_vec = model.extract_feature(arrays[0])
            # output_vec, hn_vec, cn_vec = model(arrays[0])
            output = output_vec.data.cpu().numpy()
            # hn = hn_vec.data.cpu().numpy()
            # cn = cn_vec.data.cpu().numpy()
            print(output_filename, output.shape) #, hn.shape, cn.shape)
            avg_h = np.mean(output, axis=0)
            # stacked = np.vstack((avg_h, hn.reshape((1,256)), cn.reshape((1,256))))

            np.save(os.path.join(output_path, output_filename[0]), output)
            np.save(os.path.join(avg_h_path, output_filename[0]), avg_h)
            # np.save(os.path.join(stacked_path, output_filename[0]), stacked)
        print('Embedding completed.')
        print('Dataset size:', len(test_dataset))


if __name__ == '__main__':
    if get_rep_p:
        exec('get_rep = rep_utils.%s' % rep_dict[dataset_name])
        pdb2rep(pdb_path, rep_path, single_chain_pdb, chains_to_extract, mut_p)
    if get_vector_p:
        rep2vec(rep_path=rep_path, block_list=block_list)

