"""
unirep: (3, 1900)
tape: (768,)
tape_full: (1, L, 768), maybe useful for the GFP task.
knn: (L, 15, 4) original KNN, no LSTM. L=223 for GFP.
knn_full: (L, 256) all hidden(output) of LSTM.
knn_vec(knn_avg): (256,) average hidden(output) of LSTM. WITH or WITHOUT bn in the final layer.
knn_768(knn_stacked): (3, 256) average hidden, final hidden and final cell stacked.
"""

from torch.utils.data import Dataset
import os
import numpy as np
import json


class AbstractDataset(Dataset):
    def __init__(self, paths, list_path):
        self.paths = paths  # Dict
        list_file = open(list_path, 'r')
        self.file_list = list_file.read().splitlines()
        list_file.close()

    def __len__(self):
        return len(self.file_list)


class OnehotDataset(AbstractDataset):
    def __getitem__(self, idx):
        filename = self.file_list[idx]
        array = np.load(os.path.join(self.paths['onehot'], filename))
        index = filename.split('.')[0]
        return array, filename


class UnirepDataset(AbstractDataset):
    def __getitem__(self, idx):
        filename = self.file_list[idx]
        array = np.load(os.path.join(self.paths['unirep'], filename))
        index = filename.split('.')[0]
        return array, filename


class Unirep1900Dataset(AbstractDataset):
    def __getitem__(self, idx):
        filename = self.file_list[idx]
        array = np.load(os.path.join(self.paths['unirep_1900'], filename))
        array = array[0].reshape(1, 1900)
        index = filename.split('.')[0]
        return array, filename


class UnirepFullDataset(AbstractDataset):
    def __getitem__(self, idx):
        filename = self.file_list[idx]
        array = np.load(os.path.join(self.paths['unirep_full'], filename))
        index = filename.split('.')[0]
        return array, filename


class UnirepFullKnnSelfSupervised512FullDataset(AbstractDataset):
    def __getitem__(self, idx):
        filename = self.file_list[idx]
        unirep_array = np.load(os.path.join(self.paths['unirep_full'], filename))
        knn_array = np.load(os.path.join(self.paths['knn_self_512_full'], filename))
        zeros = np.zeros((unirep_array.shape[0] - knn_array.shape[0], 512))
        knn_array = knn_array.squeeze(1)
        knn_array = np.vstack((knn_array, zeros))
        array = np.hstack((unirep_array, knn_array))
        return array, filename


class TapeDataset(AbstractDataset):
    def __getitem__(self, idx):
        filename = self.file_list[idx]
        array = np.load(os.path.join(self.paths['tape'], filename))
        array = array.reshape((1, 768))
        index = filename.split('.')[0]
        return array, filename


class TapeFullDataset(AbstractDataset):
    def __getitem__(self, idx):
        filename = self.file_list[idx]
        array = np.load(os.path.join(self.paths['tape_full'], filename))
        # array = array.reshape((228, 768))
        array = array.squeeze(0)
        index = filename.split('.')[0]
        return array, filename


class KnnSelfSupervisedFullDataset(AbstractDataset):
    def __getitem__(self, idx):
        filename = self.file_list[idx]
        array = np.load(os.path.join(self.paths['knn_self_full'], filename))
        array = array.squeeze(1)
        index = filename.split('.')[0]
        return array, filename


class KnnSelfSupervisedVecDataset(AbstractDataset):
    def __getitem__(self, idx):
        filename = self.file_list[idx]
        array = np.load(os.path.join(self.paths['knn_self_vec'], filename))
        index = filename.split('.')[0]
        return array, filename


class KnnSelfSupervised128FullDataset(AbstractDataset):
    def __getitem__(self, idx):
        filename = self.file_list[idx]
        array = np.load(os.path.join(self.paths['knn_self_128_full'], filename))
        array = array.squeeze(1)
        index = filename.split('.')[0]
        return array, filename

class KnnSelfSupervised512FullDataset(AbstractDataset):
    def __getitem__(self, idx):
        filename = self.file_list[idx]
        array = np.load(os.path.join(self.paths['knn_self_512_full'], filename))
#        array = array.squeeze(1)
        index = filename.split('.')[0]
        return array, filename

class KnnDataset(AbstractDataset):
    def __getitem__(self, idx):
        filename = self.file_list[idx]
        knn_array = np.load(os.path.join(self.paths['knn'], filename))
        knn_array = knn_array.reshape(13380)
        zeros = np.zeros(1820)
        knn_array = np.concatenate((knn_array, zeros))
        array = knn_array.reshape((8, 1900))
        index = filename.split('.')[0]
        return array, filename


class Knn135Dataset(AbstractDataset):
    def __getitem__(self, idx):
        filename = self.file_list[idx]
        arrays = np.load(os.path.join(self.paths['knn_135'], filename))
        arrays = arrays.reshape(arrays.shape[0], arrays.shape[1]*arrays.shape[2])
        index = filename.split('.')[0]
        return arrays, filename


class KnnLstmFullDataset(AbstractDataset):
    def __getitem__(self, idx):
        filename = self.file_list[idx]
        knn_array = np.load(os.path.join(self.paths['knn_full'], filename))
        index = filename.split('.')[0]
        return knn_array, filename


class Knn75LstmFullDataset(AbstractDataset):
    def __getitem__(self, idx):
        filename = self.file_list[idx]
        knn_array = np.load(os.path.join(self.paths['knn_75_full'], filename))
        index = filename.split('.')[0]
        return knn_array, filename


class KnnVec768Dataset(AbstractDataset):
    def __getitem__(self, idx):
        filename = self.file_list[idx]
        knn_array = np.load(os.path.join(self.paths['knn_768'], filename))
        knn_array = knn_array.reshape((1, 768))
        index = filename.split('.')[0]
        return knn_array, filename


class KnnVecDataset(AbstractDataset):
    def __getitem__(self, idx):
        filename = self.file_list[idx]
        knn_array = np.load(os.path.join(self.paths['knn_vec'], filename))
        knn_array = knn_array.reshape((1, 256))
        index = filename.split('.')[0]
        return knn_array, filename


class KnnVec512Dataset(AbstractDataset):
    def __getitem__(self, idx):
        filename = self.file_list[idx]
        knn_array = np.load(os.path.join(self.paths['knn_512'], filename))
        knn_array = knn_array.reshape((1, 512))
        index = filename.split('.')[0]
        return knn_array, filename


class KnnVec1024Dataset(AbstractDataset):
    def __getitem__(self, idx):
        filename = self.file_list[idx]
        knn_array = np.load(os.path.join(self.paths['knn_1024'], filename))
        knn_array = knn_array.reshape((1, 1024))
        index = filename.split('.')[0]
        return knn_array, filename


class ImageDataset(AbstractDataset):
    def __getitem__(self, idx):
        filename = self.file_list[idx]
        array = np.load(os.path.join(self.paths['image'], filename))
        index = filename.split('.')[0]
        return array, filename


class ImageVecDataset(AbstractDataset):
    def __getitem__(self, idx):
        filename = self.file_list[idx]
        array = np.load(os.path.join(self.paths['image_vec'], filename))
        array = array.reshape((1, 512))
        index = filename.split('.')[0]
        return array, filename


class UnirepKnnDataset(AbstractDataset):
    def __getitem__(self, idx):
        filename = self.file_list[idx]
        unirep_array = np.load(os.path.join(self.paths['unirep'], filename))
        knn_arrays = np.load(os.path.join(self.paths['knn'], filename))
        knn_arrays = knn_arrays.reshape(13380)
        zeros = np.zeros(1820)
        knn_arrays = np.concatenate((knn_arrays, zeros))
        knn_arrays = knn_arrays.reshape((8, 1900))
        unirep_array = np.vstack((unirep_array, knn_arrays))
        index = filename.split('.')[0]
        unirep_array.dtype = 'float'
        return unirep_array, filename


class TapeKnnDataset(AbstractDataset):
    def __getitem__(self, idx):
        filename = self.file_list[idx]
        tape_array = np.load(os.path.join(self.paths['tape'], filename))
        tape_array = tape_array.reshape((1, 768))
        knn_array = np.load(os.path.join(self.paths['knn'], filename))
        knn_array = knn_array.reshape(13380)
        zeros = np.zeros(444)
        knn_array = np.concatenate((knn_array, zeros))
        knn_array = knn_array.reshape((18, 768))
        array = np.vstack((tape_array, knn_array))
        index = filename.split('.')[0]
        return array, filename


class UnirepKnnVecDataset(AbstractDataset):
    def __getitem__(self, idx):
        filename = self.file_list[idx]
        unirep_array = np.load(os.path.join(self.paths['unirep'], filename))
        knn_array = np.load(os.path.join(self.paths['knn_vec'], filename))
        zeros = np.zeros(1644)
        knn_array = np.hstack((knn_array, zeros))
        unirep_array = np.vstack((unirep_array, knn_array))
        index = filename.split('.')[0]
        unirep_array.dtype = 'float'
        return unirep_array, filename


class TapeFullKnnDataset(AbstractDataset):
    def __getitem__(self, idx):
        filename = self.file_list[idx]
        tape_array = np.load(os.path.join(self.paths['tape_full'], filename))
        tape_array = tape_array.reshape((228, 768))
        knn_array = np.load(os.path.join(self.paths['knn'], filename))
        knn_array = knn_array.reshape(13380)
        zeros = np.zeros(444)
        knn_array = np.concatenate((knn_array, zeros))
        knn_array = knn_array.reshape((18, 768))
        array = np.vstack((tape_array, knn_array))
        index = filename.split('.')[0]
        return array, filename


class TapeFullKnnLstmFullDataset(AbstractDataset):
    def __getitem__(self, idx):
        filename = self.file_list[idx]
        tape_array = np.load(os.path.join(self.paths['tape_full'], filename))
        tape_array = tape_array.reshape((228, 768))
        knn_array = np.load(os.path.join(self.paths['knn_full'], filename))
        zeros = np.zeros((6, 256))
        knn_array = np.vstack((knn_array, zeros))
        array = np.hstack((tape_array, knn_array))
        index = filename.split('.')[0]
        return array, filename


class TapeFullKnn75LstmFullDataset(AbstractDataset):
    def __getitem__(self, idx):
        filename = self.file_list[idx]
        tape_array = np.load(os.path.join(self.paths['tape_full'], filename))
        tape_array = tape_array.reshape((-1, 768))
        knn_array = np.load(os.path.join(self.paths['knn_75_full'], filename))
        # print(filename, tape_array.shape, knn_array.shape)
        zeros = np.zeros((tape_array.shape[0] - knn_array.shape[0], 1024))
        knn_array = np.vstack((knn_array, zeros))
        array = np.hstack((tape_array, knn_array))
        index = filename.split('.')[0]
        return array, filename


class TapeFullKnnSelfSupervisedFullDataset(AbstractDataset):
    def __getitem__(self, idx):
        filename = self.file_list[idx]
        tape_array = np.load(os.path.join(self.paths['tape_full'], filename))
        tape_array = tape_array.reshape((-1, 768))
        knn_array = np.load(os.path.join(self.paths['knn_self_full'], filename))
        # print(filename, tape_array.shape, knn_array.shape)
        zeros = np.zeros((tape_array.shape[0] - knn_array.shape[0], 256))
        # print(zeros.shape, knn_array.shape)
        knn_array = knn_array.squeeze(1)
        knn_array = np.vstack((knn_array, zeros))
        array = np.hstack((tape_array, knn_array))
        return array, filename


class TapeFullKnnSelfSupervised512FullDataset(AbstractDataset):
    def __getitem__(self, idx):
        filename = self.file_list[idx]
        tape_array = np.load(os.path.join(self.paths['tape_full'], filename))
        tape_array = tape_array.reshape((-1, 768))
        knn_array = np.load(os.path.join(self.paths['knn_self_512_full'], filename))
        # print(filename, tape_array.shape, knn_array.shape)
        zeros = np.zeros((tape_array.shape[0] - knn_array.shape[0], 512))
        # print(zeros.shape, knn_array.shape)
        #knn_array = knn_array.squeeze(1)
        knn_array = np.vstack((knn_array, zeros))
        array = np.hstack((tape_array, knn_array))
        return array, filename


class TapeKnnVecDataset(AbstractDataset):
    def __getitem__(self, idx):
        filename = self.file_list[idx]
        tape_array = np.load(os.path.join(self.paths['tape'], filename))
        tape_array = tape_array.reshape(768)
        knn_array = np.load(os.path.join(self.paths['knn_vec'], filename))
        array = np.hstack((tape_array, knn_array))
        array = array.reshape((1, 1024))
        index = filename.split('.')[0]
        return array, filename


class UnirepKnnVec768Dataset(AbstractDataset):
    def __getitem__(self, idx):
        filename = self.file_list[idx]
        unirep_array = np.load(os.path.join(self.paths['unirep'], filename))
        knn_array = np.load(os.path.join(self.paths['knn_768'], filename))
        knn_array = knn_array.reshape(768)
        zeros = np.zeros(1132)
        knn_array = np.hstack((knn_array, zeros))
        unirep_array = np.vstack((unirep_array, knn_array))
        index = filename.split('.')[0]
        unirep_array.dtype = 'float'
        return unirep_array, filename


class TapeKnnVec768Dataset(AbstractDataset):
    def __getitem__(self, idx):
        filename = self.file_list[idx]
        tape_array = np.load(os.path.join(self.paths['tape'], filename))
        tape_array = tape_array.reshape(768)
        knn_array = np.load(os.path.join(self.paths['knn_768'], filename))
        knn_array = knn_array.reshape(768)
        array = np.hstack((tape_array, knn_array))
        array = array.reshape((1, 1536))
        index = filename.split('.')[0]
        return array, filename


class TapeKnnVec1024Dataset(AbstractDataset):
    def __getitem__(self, idx):
        filename = self.file_list[idx]
        tape_array = np.load(os.path.join(self.paths['tape'], filename))
        tape_array = tape_array.reshape(768)
        knn_array = np.load(os.path.join(self.paths['knn_1024'], filename))
        knn_array = knn_array.reshape(1024)
        array = np.hstack((tape_array, knn_array))
        array = array.reshape((1, 1792))
        index = filename.split('.')[0]
        return array, filename


class UnirepTapeConcatDataset(AbstractDataset):
    def __getitem__(self, idx):
        filename = self.file_list[idx]
        tape_array = np.load(os.path.join(self.paths['tape'], filename))
        tape_array = tape_array.reshape((1, 768))
        unirep_array = np.load(os.path.join(self.paths['unirep'], filename))
        unirep_array = unirep_array.reshape((1, 5700))
        array = np.hstack((tape_array, unirep_array)) 
        index = filename.split('.')[0]
        return array, filename


class UnirepTapeVstackDataset(AbstractDataset):
    def __getitem__(self, idx):
        filename = self.file_list[idx]
        tape_arrays = np.load(os.path.join(self.paths['tape'], filename))
        tape_arrays = tape_arrays.reshape((1, 768))
        unirep_arrays = np.load(os.path.join(self.paths['unirep'], filename))
        tape_arrays = np.hstack((tape_arrays, np.zeros((1, 1132))))
        arrays = np.vstack((unirep_arrays, tape_arrays))
        index = filename.split('.')[0]
        arrays.dtype = 'float'
        return arrays, filename


class UnirepTapeKnnDataset(AbstractDataset):
    def __getitem__(self, idx):
        raise Exception('Dataset not implemented!')


class UnirepTapeKnnVecDataset(AbstractDataset):
    def __getitem__(self, idx):
        filename = self.file_list[idx]
        tape_arrays = np.load(os.path.join(self.paths['tape'], filename))
        knn_arrays = np.load(os.path.join(self.paths['knn_vec'], filename))
        unirep_arrays = np.load(os.path.join(self.paths['unirep'], filename))
        zeros = np.zeros(876)
        knn_arrays = np.concatenate((tape_arrays, knn_arrays, zeros))
        knn_arrays = knn_arrays.reshape((1, 1900))
        arrays = np.vstack((unirep_arrays, knn_arrays))
        index = filename.split('.')[0]
        arrays.dtype = 'float'
        return arrays, filename


class UnirepTapeKnnVec768Dataset(AbstractDataset):
    def __getitem__(self, idx):
        filename = self.file_list[idx]
        unirep_array = np.load(os.path.join(self.paths['unirep'], filename))
        knn_array = np.load(os.path.join(self.paths['knn_768'], filename))
        knn_array = knn_array.reshape(768)
        tape_array = np.load(os.path.join(self.paths['tape'], filename))
        tape_array = tape_array.reshape(768)
        zeros = np.zeros(364)
        knn_array = np.hstack((tape_array, knn_array, zeros))
        unirep_array = np.vstack((unirep_array, knn_array))
        index = filename.split('.')[0]
        unirep_array.dtype = 'float'
        return unirep_array, filename


def get_class_name(param_file_path):
    parameter_file = open(param_file_path, 'r')
    p = json.load(parameter_file)
    dataset_names = set(p['dataset_names'])
    combining_method = p['combining_method']
    if 'unirep' in dataset_names:
        dataset_names.remove('unirep')
        if 'tape' in dataset_names:
            dataset_names.remove('tape')
            if 'knn' in dataset_names:
                dataset_names.remove('knn')
                dataset_class = 'UnirepTapeKnn'
                input_shape = (11, 1900)
            elif 'knn_vec' in dataset_names:
                dataset_names.remove('knn_vec')
                dataset_class = 'UnirepTapeKnnVec'
                input_shape = (4, 1900)
            elif 'knn_768' in dataset_names:
                dataset_names.remove('knn_768')
                dataset_class = 'UnirepTapeKnnVec768'
                input_shape = (4, 1900)
            else:
                dataset_class = 'UnirepTape'
                if combining_method == 'vstack':
                    input_shape = (4, 1900)
                elif combining_method == 'concat':
                    input_shape = (1, 6468)
                else:
                    raise Exception('No such combining method!')
        else:
            if 'knn' in dataset_names:
                dataset_names.remove('knn')
                dataset_class = 'UnirepKnn'
                input_shape = (11, 1900)
            elif 'knn_vec' in dataset_names:
                dataset_names.remove('knn_vec')
                dataset_class = 'UnirepKnnVec'
                input_shape = (4, 1900)
            elif 'knn_768' in dataset_names:
                dataset_names.remove('knn_768')
                dataset_class = 'UnirepKnnVec768'
                input_shape = (4, 1900)
            else:
                dataset_class = 'Unirep'
                input_shape = (3, 1900)
    else:
        if 'tape' in dataset_names:
            dataset_names.remove('tape')
            if 'knn' in dataset_names:
                dataset_names.remove('knn')
                dataset_class = 'TapeKnn'
                input_shape = (19, 768)
            elif 'knn_vec' in dataset_names:
                dataset_names.remove('knn_vec')
                dataset_class = 'TapeKnnVec'
                input_shape = (1, 1024)
            elif 'knn_768' in dataset_names:
                dataset_names.remove('knn_768')
                dataset_class = 'TapeKnnVec768'
                input_shape = (1, 1536)
            elif 'knn_1024' in dataset_names:
                dataset_names.remove('knn_1024')
                dataset_class = 'TapeKnnVec1024'
                input_shape = (1, 1792)
            else:
                dataset_class = 'Tape'
                input_shape = (1, 768)
        else:
            if 'tape_full' in dataset_names:
                dataset_names.remove('tape_full')
                if 'knn' in dataset_names:
                    dataset_names.remove('knn')
                    dataset_class = 'TapeFullKnn'
                    input_shape = (246, 768)
                elif 'knn_full' in dataset_names:
                    dataset_names.remove('knn_full')
                    dataset_class = 'TapeFullKnnLstmFull'
                    input_shape = (228, 1024)
                elif 'knn_75_full' in dataset_names:
                    dataset_names.remove('knn_75_full')
                    dataset_class = 'TapeFullKnn75LstmFull'
                    input_shape = (228, 1792)
                elif 'knn_self_full' in dataset_names:
                    dataset_names.remove('knn_self_full')
                    dataset_class = 'TapeFullKnnSelfSupervisedFull'
                    input_shape = (228, 1024)
                elif 'knn_self_512_full' in dataset_names:
                    dataset_names.remove('knn_self_512_full')
                    dataset_class = 'TapeFullKnnSelfSupervised512Full'
                    input_shape = (228, 1280)
                else:
                    dataset_class = 'TapeFull'
                    input_shape = (228, 768)
            else:
                if 'knn' in dataset_names:
                    dataset_names.remove('knn')
                    dataset_class = 'Knn'
                    input_shape = (8, 1900)
                elif 'knn_full' in dataset_names:
                    dataset_names.remove('knn_full')
                    dataset_class = 'KnnLstmFull'
                    input_shape = (222, 256)
                elif 'knn_768' in dataset_names:
                    dataset_names.remove('knn_768')
                    dataset_class = 'KnnVec768'
                    input_shape = (1, 768)
                elif 'knn_vec' in dataset_names:
                    dataset_names.remove('knn_vec')
                    dataset_class = 'KnnVec'
                    input_shape = (1, 256)
    if 'unirep_full' in dataset_names:
        dataset_names.remove('unirep_full')
        if 'knn_self_512_full' in dataset_names:
            dataset_names.remove('knn_self_512_full')
            dataset_class = 'UnirepFullKnnSelfSupervised512Full'
            input_shape = (227, 2412)
        else:
            dataset_class = 'UnirepFull'
            input_shape = (227, 1900)

    if 'image' in dataset_names:
        dataset_names.remove('image')
        dataset_class = 'Image'
        input_shape = (222, 512)
    if 'image_vec' in dataset_names:
        dataset_names.remove('image_vec')
        dataset_class = 'ImageVec'
        input_shape = (1, 512)
    if 'knn_75_full' in dataset_names:
        dataset_names.remove('knn_75_full')
        dataset_class = 'Knn75LstmFull'
        input_shape = (223, 1024)
    if 'knn_512' in dataset_names:
        dataset_names.remove('knn_512')
        dataset_class = 'KnnVec512'
        input_shape = (1, 512)
    if 'knn_1024' in dataset_names:
        dataset_names.remove('knn_1024')
        dataset_class = 'KnnVec1024'
        input_shape = (1, 1024)
    if 'unirep_1900' in dataset_names:
        dataset_names.remove('unirep_1900')
        dataset_class = 'Unirep1900'
        input_shape = (1, 1900)
    if 'knn_self_full' in dataset_names:
        dataset_names.remove('knn_self_full')
        dataset_class = 'KnnSelfSupervisedFull'
        input_shape = (223, 256)
    if 'knn_self_128_full' in dataset_names:
        dataset_names.remove('knn_self_full')
        dataset_class = 'KnnSelfSupervised128Full'
        input_shape = (223, 128)
    if 'knn_self_512_full' in dataset_names:
        dataset_names.remove('knn_self_512_full')
        dataset_class = 'KnnSelfSupervised512Full'
        input_shape = (223, 512)
    if 'knn_self_vec' in dataset_names:
        dataset_names.remove('knn_self_vec')
        dataset_class = 'KnnSelfSupervisedVec'
        input_shape = (1, 256)
    if 'knn_135' in dataset_names:
        dataset_names.remove('knn_135')
        dataset_class = 'Knn135'
        input_shape = (None, 135)
    if 'onehot' in dataset_names:
        dataset_names.remove('onehot')
        dataset_class = 'Onehot'
        input_shape = (226, 20)


    if dataset_names:
        raise Exception('No such dataset!')
    dataset_class += '%sDataset' % p['combining_method'].capitalize()
    return dataset_class, input_shape
