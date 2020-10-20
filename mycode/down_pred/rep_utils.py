# -*- coding: utf-8 -*
import numpy as np
import os
import random
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import matplotlib.colors as mplcolors
import matplotlib.pyplot as plt

from math import ceil
from typing import Dict, Union, Tuple, List


'''
FROM geo.py
'''

def rotation(vec, axis, theta_):
    """[罗德里格旋转公式]

    Arguments:
        vec {[npy]} -- [原始坐标 [[x1,y1,z1],[x2,y2,z2]...]]
        theta {[float]} -- [转角 弧度值]
        axis {[npy]} -- [转轴 [x,y,z]]

    Returns:
        [npy] -- [旋转所得坐标 [[x1,y1,z1],[x2,y2,z2]...]]
    """
    theta = theta_ + np.pi
    cos = np.cos(theta)
    vec_rot = cos*vec + \
        np.sin(theta)*np.cross(axis, vec) + \
        (1 - cos) * batch_dot(vec, axis).reshape(-1, 1) * axis
    if len(vec_rot) == 1:
        vec_rot = vec_rot[0]
    return vec_rot


def batch_dot(vecs1, vecs2):
    if len(vecs2.shape) == 1:
        vecs2 = vecs2.reshape(1, -1)
    return np.diag(np.matmul(vecs1, vecs2.T))


def get_torsion(vec1, vec2, axis):
    """[计算以axis为轴，向量1到向量2的旋转角]

    Arguments:
        vec1 {[npy]} -- [向量1 [x,y,z]]
        vec2 {[npy]} -- [向量2 [x,y,z]]
        axis {[npy]} -- [转轴axis [x,y,z]]

    Returns:
        [float] -- [旋转角]
    """
    n = np.cross(axis, vec2)
    n2 = np.cross(vec1, axis)
    sign = np.sign(batch_cos(vec1, n))
    angle = np.arccos(batch_cos(n2, n))
    torsion = sign*angle
    if len(torsion) == 1:
        return torsion[0]
    else:
        return torsion


def get_len(vec):
    return np.linalg.norm(vec, axis=-1)


def norm(vec):
    return vec / get_len(vec).reshape(-1, 1)


def get_angle(vec1, vec2):
    return np.arccos(np.dot(norm(vec1), norm(vec2)))


def batch_cos(vecs1, vecs2):
    cos = np.diag(np.matmul(norm(vecs1), norm(vecs2).T))
    cos = np.clip(cos, -1, 1)
    return cos

'''
END OF geo.py
'''


def coo2tor(coo):
    ca_c = (coo[2::4] - coo[1::4])[:-1]
    ca_n = (coo[::4] - coo[1::4])[1:]
    ca_ca = (coo[1::4][1:] - coo[1::4][:-1])

    tor_c = get_torsion(ca_ca[1:], ca_c[:-1], ca_ca[:-1])
    tor_n = get_torsion(ca_ca[1:], ca_n[:-1], ca_ca[:-1])

    tor_last_c = get_torsion(-ca_ca[-2], ca_c[-1], ca_ca[-1])
    tor_last_n = get_torsion(-ca_ca[-2], ca_n[-1], ca_ca[-1])

    tor_c = np.concatenate([tor_c, [tor_last_c]])
    tor_n = np.concatenate([tor_n, [tor_last_n]])

    tor = [tor_c, tor_n] 
    return np.array(tor, dtype='float32').T


def tor2sincos(tor):
    sin = np.sin(tor)
    cos = np.cos(tor)
    sincos = np.array((sin[:, 0], cos[:, 0], sin[:, 1], cos[:, 1]))
    return sincos.astype('float32')


def sincos2tor(sincos):
    tor = np.array((np.arctan2(sincos[0], sincos[1]), np.arctan2(
        sincos[2], sincos[3]))).swapaxes(0, 1)
    return tor.astype('float32')


def pept_args(coo):
    ca_ca = (coo[1::4][1:]-coo[1::4][:-1])
    ca_c = (coo[2::4] - coo[1::4])[:-1]
    ca_o = (coo[3::4]-coo[1::4])[:-1]
    ca_n = (coo[::4] - coo[1::4])[1:]

    l_ca_c = get_len(ca_c)
    l_ca_o = get_len(ca_o)
    l_ca_n = get_len(ca_n)

    cos_c = batch_cos(ca_c, ca_ca)
    cos_n = batch_cos(ca_n, ca_ca)
    cos_o = batch_cos(ca_o, ca_ca)

    p_c = l_ca_c * cos_c
    r_c = l_ca_c * (1 - cos_c ** 2) ** 0.5
    p_n = l_ca_n * cos_n
    r_n = l_ca_n * (1 - cos_n ** 2) ** 0.5
    p_o = l_ca_o * cos_o
    r_o = l_ca_o * (1 - cos_o ** 2) ** 0.5
    return np.abs(np.array([p_c, r_c, p_n, r_n, p_o, r_o], dtype='float32'))


def args_sta(args, threshold):
    args_1 = args[args <= threshold]
    args_2 = args[args > threshold]
    args_1 = args_1[int(len(args_1)*0.2):int(len(args_1)*0.8)].mean()
    args_2 = args_2[int(len(args_2)*0.2):int(len(args_2)*0.8)].mean()
    return args_1, args_2


# [cis, trans]
Projection_C = np.array([0.8027563, 1.4310328])
Radius_C = np.array([1.2996129, 0.527541])
Projection_N = np.array([0.8152914, 1.4068587])
Radius_N = np.array([1.2145333, 0.39022177])
Projection_O = np.array([0.22724794, 1.6512747])
Radius_O = np.array([2.3761292, 1.7392269])


def tor2coo(tor, ca):
    ca_ca = (ca[1:]-ca[:-1])
    l_ca_ca = get_len(ca_ca)
    is_tran = (l_ca_ca//3.4).astype(int).reshape(-1, 1)

    ori_ca_ca = norm(ca_ca)
    cos_3ca = batch_cos(ca_ca[:-1], ca_ca[1:])
    projection_ground = (
        l_ca_ca[1:]*cos_3ca).reshape(-1, 1)*norm(ca_ca[:-1])
    last_projection_ground = l_ca_ca[-2]*cos_3ca[-1]*norm(ca_ca[-1])
    ori_ground = np.concatenate(
        (norm(ca_ca[1:]-projection_ground), norm(last_projection_ground-ca_ca[-2])))

    ori_C = rotation(ori_ground, ori_ca_ca, tor[:, 0].reshape(-1, 1))
    ori_N = rotation(ori_ground, ori_ca_ca, tor[:, 1].reshape(-1, 1))

    C = ori_C * Radius_C[is_tran] + ori_ca_ca*Projection_C[is_tran] + ca[:-1]
    O = ori_C * Radius_O[is_tran] + ori_ca_ca*Projection_O[is_tran] + ca[:-1]
    N = ori_N * Radius_N[is_tran] - ori_ca_ca*Projection_N[is_tran] + ca[1:]

    coo = np.concatenate([ca[np.newaxis, :-1], [C, O, N]]
                         ).swapaxes(0, 1).reshape(-1, 3)
    coo = np.concatenate((coo, ca[np.newaxis, -1]))
    return coo.astype('float32')


AA_ALPHABET = {'A': 'ALA', 'F': 'PHE', 'C': 'CYS', 'D': 'ASP', 'N': 'ASN',
               'E': 'GLU', 'Q': 'GLN', 'G': 'GLY', 'H': 'HIS', 'L': 'LEU',
               'I': 'ILE', 'K': 'LYS', 'M': 'MET', 'P': 'PRO', 'R': 'ARG',
               'S': 'SER', 'T': 'THR', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR'}
AA_ALPHABET_REV = {'ALA': 'A', 'PHE': 'F', 'CYS': 'C', 'ASP': 'D', 'ASN': 'N',
              	   'GLU': 'E', 'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'LEU': 'L',
              	   'ILE': 'I', 'LYS': 'K', 'MET': 'M', 'PRO': 'P', 'ARG': 'R',
              	   'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'}
AA_NUM = {'A': 0, 'F': 1, 'C': 2, 'D': 3, 'N': 4,
          'E': 5, 'Q': 6, 'G': 7, 'H': 8, 'L': 9,
          'I': 10, 'K': 11, 'M': 12, 'P': 13, 'R': 14,
          'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19}
AA_HYDROPATHICITY_INDEX = {'R': -4.5, 'K': -3.9, 'N': -3.5, 'D': -3.5, 'Q': -3.5,
                           'E': -3.5, 'H': -3.2, 'P': -1.6, 'Y': -1.3, 'W': -0.9,
                           'S': -0.8, 'T': -0.7, 'G': -0.4, 'A': 1.8, 'M': 1.9,
                           'C': 2.5, 'F': 2.8, 'L': 3.8, 'V': 4.2, 'I': 4.5}
AA_BULKINESS_INDEX = {'R': 14.28, 'K': 15.71, 'N': 12.82, 'D': 11.68, 'Q': 14.45,
                      'E': 13.57, 'H': 13.69,  'P': 17.43, 'Y': 18.03, 'W': 21.67,
                      'S': 9.47, 'T': 15.77, 'G': 3.4, 'A': 11.5, 'M': 16.25,
                      'C': 13.46, 'F': 19.8, 'L': 21.4, 'V': 21.57, 'I': 21.4}
AA_FLEXIBILITY_INDEX = {'R': 2.6, 'K': 1.9, 'N': 14., 'D': 12., 'Q': 4.8,
                        'E': 5.4, 'H': 4., 'P': 0.05, 'Y': 0.05, 'W': 0.05,
                        'S': 19., 'T': 9.3, 'G': 23., 'A': 14., 'M': 0.05,
                        'C': 0.05, 'F': 7.5, 'L': 5.1, 'V': 2.6, 'I': 1.6}
AA_PROPERTY = {}

# CAUTIOUS: MAY HAVE COMPATIBILITY PROBLEMS
for aa in AA_HYDROPATHICITY_INDEX.keys():
    AA_PROPERTY.update({aa: [(5.5 - AA_HYDROPATHICITY_INDEX[aa]) / 10,
                             AA_BULKINESS_INDEX[aa] / 21.67,
                             (25. - AA_FLEXIBILITY_INDEX[aa]) / 25.]})
    aa_long = AA_ALPHABET[aa]
    AA_PROPERTY.update({aa_long: [(5.5 - AA_HYDROPATHICITY_INDEX[aa]) / 10,
                              AA_BULKINESS_INDEX[aa] / 21.67,
                              (25. - AA_FLEXIBILITY_INDEX[aa]) / 25.]})


def MapDis(coo):
    return squareform(pdist(coo, metric='euclidean')).astype('float32')

# CAUTIOUS: Function name changed (spelling corrected)
def KNNStructRep(ca, seq, k=15):
    dismap = MapDis(ca)
    nn_indexs = np.argsort(dismap, axis=1)[:, :k]
    seq_embeded = []
    for aa in seq:
        seq_embeded.append(AA_PROPERTY[aa])
    knn_feature = np.array(seq_embeded)[nn_indexs]
    knn_distance = [dismap[i][nn_indexs[i]] for i in range(len(nn_indexs))]
    knn_distance = np.array(knn_distance).reshape(-1, k, 1)
    knn_rep = np.concatenate((knn_distance, knn_feature), -1)
    return knn_rep.astype('float32')


def KNNStructRepRelative(ca, seq, k=15, index_norm=200):
    dismap = MapDis(ca)
    nn_indexs = np.argsort(dismap, axis=1)[:, :k] 
    relative_indexs = nn_indexs.reshape(-1, k, 1) - \
        nn_indexs[:, 0].reshape(-1, 1, 1).astype('float32')
    relative_indexs /= index_norm
    seq_embeded = []
    for aa in seq:
        seq_embeded.append(AA_PROPERTY[aa])
    knn_feature = np.array(seq_embeded)[nn_indexs]
    knn_distance = [dismap[i][nn_indexs[i]] for i in range(len(nn_indexs))]
    knn_distance = np.array(knn_distance).reshape(-1, k, 1)
    print(knn_distance.shape, relative_indexs.shape)
    knn_rep = np.concatenate((knn_distance, relative_indexs, knn_feature), -1) 
    return knn_rep.astype('float32')


class Atom(object):
    def __init__(self, aminoacid, index, x, y, z):
        self.aa = aminoacid
        self.index = index
        self.x = x
        self.y = y
        self.z = z


## NOT UPDATED, SEE process_relative.py
class Arraylize(object):
    def __init__(self, resolution, size, atoms, indexs, pad=4):
        self.atoms = atoms
        self.ar = size + pad
        self.idx_ary = indexs
        self.scale = size * 2 / resolution
        self.res = resolution + int(2*pad/self.scale)
        self.dim = 5
        self.array = np.zeros(
            [self.res, self.res, self.dim], dtype='float32', order='C')
        self.pad = pad

        self.rec = {}
        self.site = {}
        self.run()

    def pixel_center_dis(self, dot):
        dot.dis_x = dot.x / self.scale % 1 - 0.5
        dot.dis_y = dot.y / self.scale % 1 - 0.5
        dot.dis_sqrt = dot.dis_x ** 2 + dot.dis_y ** 2

    def closer_pixel(self, dot):
        x_sign = int(np.sign(dot.dis_x))
        y_sign = int(np.sign(dot.dis_y))
        if abs(dot.dis_x) < abs(dot.dis_y):
            neighbors = [(0, y_sign), (x_sign, 0), (x_sign, y_sign), (-x_sign, 0),
                         (-x_sign, y_sign), (0, -y_sign), (x_sign, -y_sign), (-x_sign, -y_sign)]
        else:
            neighbors = [(x_sign, 0), (0, y_sign), (x_sign, y_sign), (0, -y_sign),
                         (x_sign, -y_sign), (-x_sign, 0), (-x_sign, y_sign), (-x_sign, -y_sign)]
        for (i, j) in neighbors:
            if -1 < dot.x_ary + i < self.res and -1 < dot.y_ary + j < self.res:
                if self.array[dot.x_ary + i, dot.y_ary + j, -1] == 0:
                    dot.x_ary = dot.x_ary + i
                    dot.y_ary = dot.y_ary + j
                    self.draw_atom(dot)
                    break

    def closer_dot(self, dot1, dot2):  # dot1 is original; dot2 is new
        self.pixel_center_dis(dot1)
        self.pixel_center_dis(dot2)
        if dot1.dis_sqrt > dot2.dis_sqrt:
            self.closer_pixel(dot1)
            self.draw_atom(dot2)
        else:
            self.closer_pixel(dot2)

    def draw_atom(self, dot):
        self.array[dot.x_ary, dot.y_ary] = [
            dot.z, dot.index] + AA_PROPERTY[dot.aa]
        self.rec.update({(dot.x_ary, dot.y_ary): dot})

    def draw_dot(self, x, y, dot, z_add, idx_add):
        if self.rec.get((x, y)) is None:
            if self.array[x, y, 0]:
                if dot.z + z_add > self.array[x, y, 0]:
                    self.array[x, y, :2] = [dot.z + z_add, dot.index + idx_add]
            else:
                self.array[x, y, :2] = [dot.z + z_add, dot.index + idx_add]

    def dots_connection(self, dot1, dot2):
        z_dis = dot2.z - dot1.z
        x_sign = int(np.sign(dot2.x_ary - dot1.x_ary))
        y_sign = int(np.sign(dot2.y_ary - dot1.y_ary))
        x_dis = abs(dot2.x_ary - dot1.x_ary)
        y_dis = abs(dot2.y_ary - dot1.y_ary)
        long_step = max(x_dis, y_dis)
        short_step = min(x_dis, y_dis)

        if short_step == 0:
            if x_dis > y_dis:
                x_step, y_step = 1, 0
            else:
                x_step, y_step = 0, 1
        else:
            slope = long_step / short_step
            if x_dis > y_dis:
                x_step, y_step = 1, 1 / slope
            else:
                x_step, y_step = 1 / slope, 1

        for step in range(1, long_step):
            self.draw_dot(round(dot1.x_ary + step * x_step * x_sign), round(dot1.y_ary + step * y_step * y_sign),
                          dot1, z_dis * step / (long_step + 1), step / (long_step + 1))

    def draw_connection(self):
        for (x, y) in self.rec.keys():
            self.site.update({self.rec[(x, y)]: [x, y]})
        for i in range(len(self.atoms) - 1):
            if self.atoms[i + 1].index - self.atoms[i].index == 1 or self.atoms[i].index == -1:
                self.dots_connection(self.atoms[i], self.atoms[i + 1])

    def crop_image(self):
        padding = int(self.pad / self.scale)
        self.array = self.array[padding:self.res -
                                padding, padding:self.res-padding]

    def height_limit(self):
        self.array[abs(self.array[:, :, 0]) > self.ar - self.pad] = 0

    def height_norm(self):
        self.array[:, :, 0] /= self.ar - self.pad

    def index_norm(self, norm_lenght=200):
        self.array[:, :, 1] /= norm_lenght

    def run(self):
        for atom in self.atoms:
            atom.x_ary = int(atom.x // self.scale + self.res // 2)
            atom.y_ary = int(atom.y // self.scale + self.res // 2)
            if self.rec.get((atom.x_ary, atom.y_ary)):
                self.closer_dot(self.rec[(atom.x_ary, atom.y_ary)], atom)
            else:
                self.draw_atom(atom)

        self.draw_connection()
        self.crop_image()
        self.height_limit()
        self.height_norm()
        self.index_norm()


# NOT UPDATED. SEE process_relative.py
def ImageStructRep(ca, seq, center='peptide_plane', resolution=128, box_size=8, compress=True, pad=4, relative_index=True):
    arrays = []
    tgt_x = np.array([0, 1, 0])
    rot_axis_y = tgt_x
    tgt_y = np.array([1, 0, 0])
    ori_x = norm(ca[1:] - ca[:-1])
    ori_y = np.concatenate((ori_x[1:], -(ori_x[np.newaxis, -2])))
    if center == 'peptide_plane':
        centers = (ca[:-1] + ca[1:]) / 2
    else:
        centers = ca.copy()
        ori_x = np.concatenate((ori_x, ori_x[np.newaxis, -1]))
        ori_y = np.concatenate((ori_y, ori_y[np.newaxis, -1]))
    rot_axis_x = norm(np.cross(ori_x, tgt_x))

    tor_x = get_torsion(ori_x, tgt_x, rot_axis_x)
    ori_y_rot = rotation(ori_y, rot_axis_x, tor_x.reshape(-1, 1))
    ori_y_proj = ori_y_rot.copy()
    ori_y_proj[:, 1] = 0.
    ori_y_proj = norm(ori_y_proj)
    l_ori_y_proj = len(ori_y_proj)
    tor_y = get_torsion(ori_y_proj,
                            np.tile(tgt_y, (l_ori_y_proj, 1)),
                            np.tile(rot_axis_y, (l_ori_y_proj, 1)))

    for i, center in enumerate(centers):
        ca_ = ca - center
        global_indexs = np.where(get_len(
            ca_) < (box_size + pad)*np.sqrt(3))[0]

        if relative_index:
            local_indexs = global_indexs - i
            if center == 'peptide_plane':
                local_indexs[local_indexs <= 0] -= 1
        else:
            local_indexs = global_indexs

        local_atoms = []
        num_local_atoms = len(global_indexs)
        ca_xrot = rotation(ca_[global_indexs],
                               np.tile(rot_axis_x[i], (num_local_atoms, 1)),
                               np.tile(tor_x[i], (num_local_atoms, 1)))
        ca_rot = rotation(ca_xrot,
                              np.tile(rot_axis_y, (num_local_atoms, 1)),
                              np.tile(tor_y[i], (num_local_atoms, 1)))

        count = 0
        for j, idx in enumerate(global_indexs):
            if np.max(np.abs(ca_rot[j])) < box_size + pad:
                count += 1
                local_atoms.append(
                    Atom(seq[idx], local_indexs[j], ca_rot[j][0], ca_rot[j][1], ca_rot[j][2]))

        arrays.append(Arraylize(resolution=resolution,
                                size=box_size,
                                atoms=local_atoms,
                                indexs=local_indexs).array)

    arrays = np.array(arrays, dtype='float32')

    if compress:
        shape = arrays.shape
        keys = arrays.nonzero()
        values = arrays[keys]
        com_ary = [shape, keys, values.astype('float32')]
        return com_ary
    else:
        return arrays


def load_compressed_array(filename):
    shape, keys, values = np.load(filename, allow_pickle=True)
    ary = np.zeros(shape)
    ary[keys] = values
    return ary.astype('float32')


def array_visible(ary):
    signal = ary[:, :, :, :2].nonzero()
    visary = np.zeros_like(ary)
    visary[signal] = (ary[signal]+1)/2
    return visary


def cb_args(coo, seq, cb):
    mask = np.array([aa != 'G' for aa in seq]).nonzero()
    n = coo[::4][mask]
    ca = coo[1::4][mask]
    c = coo[2::4][mask]

    ori_ca_n = norm(n - ca)
    ori_ca_c = norm(c - ca)
    ori_mid = norm(ori_ca_n + ori_ca_c)
    rot_axis_cb = norm(ori_ca_c - ori_ca_n)

    ca_cb = cb - ca
    l_ca_cb = get_len(ca_cb)
    ori_ca_cb = norm(ca_cb)

    tor_cb = []
    for i in range(len(cb)):
        tor_cb.append(get_torsion(
            ori_mid[i], ori_ca_cb[i], rot_axis_cb[i]))
    return np.array([l_ca_cb, tor_cb], dtype='float32')


R_CB = 1.53534496
T_CB = 0.91366992


def coo2cb(coo, seq):
    mask = np.array([aa != 'G' for aa in seq]).nonzero()
    cb = []
    n = coo[::4][mask]
    ca = coo[1::4][mask]
    c = coo[2::4][mask]
    ori_ca_n = norm(n - ca)
    ori_ca_c = norm(c - ca)
    ori_mid = norm(ori_ca_n + ori_ca_c)
    rot_axis_cb = norm(ori_ca_c - ori_ca_n)
    cb = rotation(ori_mid, rot_axis_cb, T_CB) * R_CB + ca
    return np.array(cb, dtype='float32')


def cRMSD(coo1, coo2, norm=100):
    G1 = np.sum(np.square(get_len(coo1)))
    G2 = np.sum(np.square(get_len(coo2)))

    S = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            S[i, j] = np.sum(coo1[:, i]*coo2[:, j])

    K = np.zeros((4, 4))
    K[0, 0] = S[0, 0] + S[1, 1] + S[2, 2]
    K[1, 1] = S[0, 0] - S[1, 1] - S[2, 2]
    K[2, 2] = -S[0, 0] + S[1, 1] - S[2, 2]
    K[3, 3] = -S[0, 0] - S[1, 1] + S[2, 2]
    K[0, 1] = K[1, 0] = S[1, 2] - S[2, 1]
    K[0, 2] = K[2, 0] = S[2, 0] - S[0, 2]
    K[0, 3] = K[3, 0] = S[0, 1] - S[1, 0]
    K[1, 2] = K[2, 1] = S[0, 1] + S[1, 0]
    K[1, 3] = K[3, 1] = S[2, 0] + S[0, 2]
    K[2, 3] = K[3, 2] = S[1, 2] + S[2, 1]

    a, _ = np.linalg.eig(K)
    u = max(a)

    rmsd = np.sqrt(abs(G1 + G2 - 2 * u) / len(coo1))

    if norm:
        norm_rmsd = rmsd / (1 + np.log(np.sqrt(len(coo1) / norm)))
        return norm_rmsd
    else:
        return rmsd


def GDT(coo1, coo2, cutoff):
    distance = get_len(coo1-coo2)
    count = np.sum(distance <= cutoff)
    return count/len(coo1)


def cal_GDT(coo1, coo2, cutoffs):
    gdt = [[GDT(tgt, out, cutoff) for cutoff in cutoffs]
           for tgt, out in zip(coo1, coo2)]
    return np.array(gdt).astype('float32')


def cal_RMSD(coo1, coo2, norm):
    rmsd = [cRMSD(tgt, out, norm) for tgt, out in zip(coo1, coo2)]
    return np.array(rmsd).astype('float32')


def atoms_cluster(coo, cb):
    n = coo[3::4]
    c = coo[1::4]
    o = coo[2::4]
    min_bb = np.concatenate([n, c])
    ext_bb = np.concatenate([n, c, o, cb])
    return n, c, o, cb, min_bb, ext_bb


def strip_tgt(coo, cb, seq):
    if seq[0] != 'G':
        cb = cb[1:]
    if seq[-1] != 'G':
        cb = cb[:-1]
    return coo[1:-2], cb


def tor2GDT(coo, seq, cb, tor, cutoffs=[0.05, 0.1, 0.2]):
    ca = coo[1::4]
    coo_ = tor2coo(tor, ca)
    cb_ = coo2cb(coo_[3:-1], seq[1:-1])

    atoms_ = atoms_cluster(coo_, cb_)
    coo_striped, cb_striped = strip_tgt(coo, cb, seq)
    atoms = atoms_cluster(coo_striped, cb_striped)

    gdt = cal_GDT(atoms, atoms_, cutoffs)
    return gdt


def tor2RMSD(coo, seq, cb, tor, norm=100):
    ca = coo[1::4]
    coo_ = tor2coo(tor, ca)
    cb_ = coo2cb(coo_[3:-1], seq[1:-1])

    atoms_ = atoms_cluster(coo_, cb_)
    coo_striped, cb_striped = strip_tgt(coo, cb, seq)
    atoms = atoms_cluster(coo_striped, cb_striped)

    rmsd = cal_RMSD(atoms, atoms_, norm)
    return rmsd


RAMA_SETTING = {
    "General": {
        "file": os.path.join('./rama/rama_contour', 'pref_general.data'),
        "cmap": mplcolors.ListedColormap(['#FFFFFF', '#B3E8FF', '#7FD9FF']),
        "bounds": [0, 0.0005, 0.02, 1],
    },
    "GLY": {
        "file": os.path.join('./rama/rama_contour', 'pref_glycine.data'),
        "cmap": mplcolors.ListedColormap(['#FFFFFF', '#FFE8C5', '#FFCC7F']),
        "bounds": [0, 0.002, 0.02, 1],
    },
    "PRO": {
        "file": os.path.join('./rama/rama_contour', 'pref_proline.data'),
        "cmap": mplcolors.ListedColormap(['#FFFFFF', '#D0FFC5', '#7FFF8C']),
        "bounds": [0, 0.002, 0.02, 1],
    },
    "PRE-PRO": {
        "file": os.path.join('./rama/rama_contour', 'pref_preproline.data'),
        "cmap": mplcolors.ListedColormap(['#FFFFFF', '#B3E8FF', '#7FD9FF']),
        "bounds": [0, 0.002, 0.02, 1],
    }
}


def load_rama_map(filename):
    rama_map = np.zeros((360, 360), dtype=np.float64)
    with open(filename) as fn:
        for line in fn:
            if line.startswith("#"):
                continue
            else:
                line = line.split()
                x = int(float(line[1]))
                y = int(float(line[0]))
                rama_map[x + 180][y + 180] = \
                    rama_map[x + 179][y + 179] = \
                    rama_map[x + 179][y + 180] = \
                    rama_map[x + 180][y + 179] = float(line[2])
    return rama_map


# for rama_type in RAMA_SETTING.keys():
#     RAMA_SETTING[rama_type]['map'] = load_rama_map(
#         RAMA_SETTING[rama_type]['file'])


def cal_phipsi(coo):
    ca = coo[::4][1:-1]
    n = coo[3::4]
    c = coo[1::4]

    c_n = norm(n-c)
    n_ca = norm(ca-n[:-1])
    ca_c = norm(c[1:] - ca)

    phi = get_torsion(c_n[:-1], ca_c, n_ca) / np.pi * 180 + 180
    psi = get_torsion(n_ca, c_n[1:], ca_c) / np.pi * 180 + 180

    phi[phi >= 360] -= 360
    psi[psi >= 360] -= 360
    return phi, psi


def seq2rama_type(seq):
    rama_types = []
    for aa in seq:
        if aa == 'G':
            rama_types.append('GLY')
        elif aa == 'P':
            rama_types.append('PRO')
            if len(rama_types) != 1:
                if rama_types[-2] != 'PRO':
                    rama_types[-2] = "PRE-PRO"
        else:
            rama_types.append('General')
    return rama_types


def cal_rama(coo, seq, reduce_output=True):
    rama_types = seq2rama_type(seq)[1:-1]
    phis, psis = cal_phipsi(coo)
    core = {}
    allow = {}
    outlier = {}

    for rank in core, allow, outlier:
        for rama_type in RAMA_SETTING.keys():
            rank[rama_type] = {}

    for index, (phi, psi, rama_type) in enumerate(zip(phis, psis, rama_types)):
        if RAMA_SETTING[rama_type]['map'][int(psi)][int(phi)] < RAMA_SETTING[rama_type]["bounds"][1]:
            outlier[rama_type][index] = [psi, phi]
        elif RAMA_SETTING[rama_type]['map'][int(psi)][int(phi)] < RAMA_SETTING[rama_type]["bounds"][2]:
            allow[rama_type][index] = [psi, phi]
        else:
            core[rama_type][index] = [psi, phi]

    core_num = [len(core[rama_type].keys())
                for rama_type in RAMA_SETTING.keys()]
    allow_num = [len(allow[rama_type].keys())
                 for rama_type in RAMA_SETTING.keys()]
    outlier_num = [len(outlier[rama_type].keys())
                   for rama_type in RAMA_SETTING.keys()]

    core_num = np.array(core_num + [sum(core_num)])
    allow_num = np.array(allow_num + [sum(allow_num)])
    outlier_num = np.array(outlier_num + [sum(outlier_num)])
    total_num = core_num + allow_num + outlier_num

    rama_matrix = np.concatenate(
        (core_num, allow_num, outlier_num, total_num)).reshape(4, -1)

    core_rate, allow_rate, outlier_rate = rama_matrix[:3, -1] / \
        rama_matrix[-1, -1]

    if reduce_output:
        return [core_rate, allow_rate, outlier_rate]
    else:
        return rama_matrix, [core, allow, outlier]


def cal_criterias(coo, seq, cb, tor, rmsd_norm=100, gdt_cutoffs=[0.05, 0.1, 0.2], rama_reduce_output=True):
    ca = coo[1::4]
    coo_ = tor2coo(tor, ca)
    cb_ = coo2cb(coo_[3:-1], seq[1:-1])

    atoms_ = atoms_cluster(coo_, cb_)
    coo_striped, cb_striped = strip_tgt(coo, cb, seq)
    atoms = atoms_cluster(coo_striped, cb_striped)

    rmsd = cal_RMSD(atoms, atoms_, rmsd_norm)
    gdt = cal_GDT(atoms, atoms_, gdt_cutoffs)
    rama = cal_rama(coo_, seq, rama_reduce_output)
    return rmsd, gdt, rama


def loss_from_log(train_name):
    with open('./logs/log_%s.txt' % train_name) as f:
        lines = f.readlines()
    val_loss = []
    train_loss = []
    for line in lines:
        if line[:5] == 'epoch':
            line = line.split()
            if line[-1] == 'training':
                train_loss.append([])
            if line[-2] == 'mean_val_loss=':
                val_loss.append(float(line[-1]))
        if line[:5] == 'iters':
            train_loss[-1].append(float(line.split('=')[-1]))
    train_loss = np.array(train_loss)
    val_loss = np.array(val_loss)
    return train_loss, val_loss


def fuse_output(output_paths):
    sincos_fuse = []
    filenames = os.listdir(output_paths[0])
    for filename in filenames:
        filename = filename[:-4]
        sincos_outputs = []
        for output_path in output_paths:
            sincos_output = np.load(os.path.join(
                output_path, '%s.npy' % filename))
            sincos_outputs.append(sincos_output)
        sincos_outputs = np.array(sincos_outputs)
        sincos_fuse.append(np.mean(np.array(sincos_outputs), axis=0))
    return sincos_fuse, filenames


# FOLLOWING IS ADDED FROM process_relative.py
def concate_groups(groups, target_size):
    groups.sort(key=lambda x: sum(x), reverse=True)
    for g in groups:
        g.sort(reverse=True)

    i = 0 
    while True:
        if sum(groups[i])+groups[i+1][-1] <= target_size:
            groups[i].append(groups[i+1].pop(-1))
            if len(groups[i+1]) == 0:
                groups.pop(i+1)
                if i == len(groups)-1:
                    return groups
        else:
            if i+1 < len(groups)-1:
                i += 1
            else:
                return groups


def random_group(lengths, target_size):
    random.shuffle(lengths)
    groups = [[]]
    for l_ in lengths:
        if sum(groups[-1])+l_ <= target_size:
            groups[-1] += [l_]
        else:
            groups.append([l_])

    groups_num = []
    groups_num.append(len(groups))
    while True:
        if len(groups_num) > 5:
            if groups_num[-1] == groups_num[-6]:
                break
        groups = concate_groups(groups, target_size)
        groups_num.append(len(groups))
    return groups


def fill_groups(groups, len_sep_files):
    file_groups = []
    for g in groups:
        g.sort(reverse=True)
        file_groups.append([len_sep_files[l_].pop(0) for l_ in g])
    return file_groups


def group_files(len_sep_files, lengths, target_size):
    len_groups = random_group(lengths, target_size)
    grouped_files = fill_groups(len_groups, len_sep_files)
    return grouped_files


'''
FROM knn_utils.py
'''
def extract_coord(atoms_data, atoms_type):
    coord_array_ca = np.zeros((ceil(len(atoms_data) / len(atoms_type)), 3))  # CA坐标, shape: L * 3
    coord_array_all = np.zeros((len(atoms_data), 3))  # 所有backbone原子(或atom_types中的原子)坐标, shape: 4L * 3
    aa_names = []
    for i in range(len(atoms_data)):
        coord_array_all[i] = [float(atoms_data[i][j]) for j in range(6, 9)] 
        # 写法可能不合适,未考虑氨基酸内部原子顺序不一致的情况
        if i % len(atoms_type) == atoms_type.index('CA'):
            coord_array_ca[i // len(atoms_type)] = [float(atoms_data[i][j]) for j in range(6, 9)] 
            aa_names.append(atoms_data[i][3][-3::])
    aa_names_array = np.array(aa_names)  # shape: L * 1
    return coord_array_ca, aa_names_array, coord_array_all


def read_fasta(file_path):
    seq_dict: Dict[str, str] = {}  # {seq_name -> fasta_seq}
    seq_file = open(file_path, 'r')
    seq_data = seq_file.readlines()
    # 读取fasta文件

    for i in range(len(seq_data)):
        if seq_data[i][0] == '>':
            seq_name = seq_data[i][1:-1]
            seq_dict[seq_name] = ''
            j = 1 
            while True:
                if i + j >= len(seq_data) or seq_data[i + j][0] == '>':
                    break
                else:
                    seq_dict[seq_name] += ''.join(seq_data[i + j].split())
                j += 1
    return seq_dict

def seq2onehot(seq: str) -> np.ndarray:
    seq = list(seq)
    onehot = np.zeros((len(seq), 20))
    for i in range(len(seq)):
        aa_num = AA_NUM[seq[i]]
        onehot[i, aa_num] = 1
    return onehot

def seq2array(aa_seq: str) -> np.ndarray:
    aa_seq = list(aa_seq)
    for i in range(len(aa_seq)):
        aa_seq[i] = AA_ALPHABET[aa_seq[i]]
    aa_seq_array = np.array(aa_seq)
    return aa_seq_array


def array2seq(aa_array: np.ndarray) -> str:
    aa_array = list(aa_array)
    for i in range(len(aa_array)):
        aa_array[i] = list(AA_ALPHABET.keys())[list(AA_ALPHABET.values()).index(aa_array[i])]
    aa_seq = ''.join(aa_array)
    return aa_seq


# Deprecated. 比对两个等长序列，得到突变位点
def align(seq_a: str, seq_b: str) -> Dict[int, Tuple[str, str]]:
    mut_sites = {}
    if len(seq_a) != len(seq_b):
        raise Exception("Lengths of the sequences are not equal!")
    for i in range(len(seq_a)):
        if seq_a[i] != seq_b[i]:
            mut_sites[i + 1] = (seq_a[i], seq_b[i])
    return mut_sites


# 根据序列名比较两个knn的array
def compare_arrays(pdb_name, seq_name1, seq_name2):
    identical_p: List[bool] = []
    seq_name1 = '_' + seq_name1 if seq_name1 else ''
    seq_name2 = '_' + seq_name2 if seq_name2 else ''
    array1 = np.load(os.path.join(knn_path, pdb_name + seq_name1 + '.npy'))
    array2 = np.load(os.path.join(knn_path, pdb_name + seq_name2 + '.npy'))
    if len(array1) != len(array2):
        raise Exception("Lengths of the arrays are not equal!")
    for i in range(len(array1)):
        identical_p.append((array1[i] == array2[i]).all())
        print("compare_arrays:", i + 1, identical_p[-1])
    return identical_p


def compare_len(coord_array, aa_array, atoms_type):
    atoms = len(atoms_type)
    print(coord_array.shape[0] / atoms, aa_array.shape[0])
    if coord_array.shape[0] / atoms > aa_array.shape[0]:
        raise("Seq too short!")
    elif coord_array.shape[0] / atoms < aa_array.shape[0]:
        raise("Seq too long!")

def get_knn(coord_array, aa_array):
    # compare_len(coord_array, aa_array)
    window_size = 15

    dist_ca = pdist(coord_array, metric='euclidean')  # shape: L(L-1)/2 * 1
    dist_ca = squareform(dist_ca).astype('float32')  # shape: L * L
    mark_type = [('distance', float), ('aa', 'S10')]
    dist_windows = []

    # 对于每个aa
    for i in range(len(dist_ca)):
        marked_array = []
        new_array = []
        # 对于每个其他aa到当前aa的距离
        for j in range(len(dist_ca[i])):
            marked_array.append((dist_ca[i, j], aa_array[j]))
        marked_array = np.array(marked_array, dtype=mark_type)  # shape: L * 1, element: (dist, aa_type)
        marked_array = np.sort(marked_array, order='distance')[:window_size]  # shape: 15 * 1
        for j in range(len(marked_array)):
            aa = marked_array[j][1].decode('utf-8')
            new_array.append([marked_array[j][0]] + AA_PROPERTIES[aa])
        dist_windows.append(new_array)
    dist_windows = np.array(dist_windows).astype('float32')
    return dist_windows

'''
END OF knn_utils.py
'''
# 135: 15*9, (distance*1 + seq_relative_pos*1 + aa_property*3) + (coord_diff*3 + tor_angle*1)
def get_knn_135(coo, aa, k=15):
    if len(coo) % 4 != 0:
        raise Exception('Absence of certain atoms!')
    arrays = coo
    tor_arrays = tor2sincos(coo2tor(coo))
    tor_arrays = tor_arrays.transpose()
    print(tor_arrays.shape, coo.shape)
    tor_arrays = np.concatenate((tor_arrays, np.zeros((1, 4))), axis=0)
    ca_coo = []
    for i in range(len(arrays)):
        if i % 4 == 1:
            ca_coo.append(arrays[i]) 
    ca_coo = np.array(ca_coo)
    arrays = np.concatenate((ca_coo, tor_arrays), axis=1)
    structure_feature = []
    tor = []
    for i, array in enumerate(arrays):
        dic = {}
        list_coord = []
        for j, arr in enumerate(arrays):
            x = arrays[j][:3] - arrays[i][:3]
            dis = np.linalg.norm(x,axis=0,keepdims=True)[0]
            a, b, c, d = arrays[j][3:]
            tor_c = np.arctan2(a, b)
            tor_n = np.arctan2(c, d)
            tor_avg = (tor_c + tor_n) / 2
            dic[dis] = list(x) + [tor_avg]
        dic1 = sorted(dic.items(), key = lambda x: x[0], reverse = False)[:20]
        for key in dic1:
            list_coord.append(key[1])
        structure_feature.append(list_coord)

    arrays_orgin = KNNStructRepRelative(ca_coo, aa, k=k)
    print(arrays_orgin.shape, np.array(structure_feature).shape)
    new_feature = np.concatenate((arrays_orgin, np.array(structure_feature)),axis=2)
    return new_feature


# 150: 15*10, (spherical_coord*4 + distance*1 + seq_relative_pos*1 + aa_property*3) + (tor_angle*1)
def get_knn_150_append(knn_coo, aa):
    knn_spher = knn_coo[0]
    #knn_spher = knn_coo
    coo = knn_coo[1]
    print('CC', coo.shape)
    if len(coo) % 4 != 0:
        raise Exception('Absence of certain atoms!')
    arrays = coo
    tor = []
    tor_arrays = tor2sincos(coo2tor(coo))
    tor_arrays = tor_arrays.transpose()
    print(tor_arrays.shape, coo.shape)
    tor_arrays = np.concatenate((tor_arrays, np.zeros((1, 4))), axis=0)
    ca_coo = []
    for i in range(len(arrays)):
        if i % 4 == 1:
            ca_coo.append(arrays[i]) 
    ca_coo = np.array(ca_coo)
    arrays = np.concatenate((ca_coo, tor_arrays), axis=1)
    structure_feature = []
    tor = []
    for i, array in enumerate(arrays):
        dic = {}
        list_tor = []
        for j, arr in enumerate(arrays):
            x = arrays[j][:3] - arrays[i][:3]
            dis = np.linalg.norm(x,axis=0,keepdims=True)[0]
            a, b, c, d = arrays[j][3:]
            tor_c = np.arctan2(a, b)
            tor_n = np.arctan2(c, d)
            tor_avg = (tor_c + tor_n) / 2
            dic[dis] = [tor_avg]
        dic1 = sorted(dic.items(), key = lambda x: x[0], reverse = False)[:15]
        for key in dic1:
            list_tor.append(key[1])
        structure_feature.append(list_tor)
    print('CONCATE', knn_spher.shape, np.array(structure_feature).shape)
    new_feature = np.concatenate((knn_spher, np.array(structure_feature)), axis=2)
    return new_feature


def get_knn_180(coo, aa):
    return get_knn_135(coo, aa, k=20)


def get_knn_150(coo, aa):
    ca_coo = []
    for i in range(len(coo)):
        if i % 4 == 1:
            ca_coo.append(coo[i])
    ca_coo = np.array(ca_coo)
    knn_spher = StrucRep('knn', 'property', 200).knn_struc_rep(ca_coo, aa)
    return get_knn_150_append((knn_spher, coo), aa)


class StrucRep(object):
    def __init__(self, struc_format='knn', aa_format='property', index_norm=200):
        self.aa_encoder = AminoacidEncoder(aa_format)
        self.index_norm = index_norm
        if struc_format == 'knn':
            self.struc_rep = self.knn_struc_rep
        elif struc_format == 'image':
            self.struc_rep = self.image_struc_rep
        elif struc_format == 'conmap':
            self.struc_rep = self.contact_map
        elif struc_format == 'dismap':
            self.struc_rep = self.distance_map

    def knn_struc_rep(self, ca, seq, k=15):
        dismap = MapDis(ca)
        nn_indexs = np.argsort(dismap, axis=1)[:, :k]
        relative_indexs = nn_indexs.reshape(-1, k, 1) - \
            nn_indexs[:, 0].reshape(-1, 1, 1).astype('float32')
        relative_indexs /= self.index_norm
        seq_embeded = self.aa_encoder.encode(seq)
        knn_feature = np.array(seq_embeded)[nn_indexs]
        knn_distance = [dismap[i][nn_indexs[i]] for i in range(len(nn_indexs))]
        knn_distance = np.array(knn_distance).reshape(-1, k, 1)

        tgt_x = np.array([0, 1, 0])
        rot_axis_y = tgt_x
        tgt_y = np.array([1, 0, 0])
        ori_x = norm(ca[1:] - ca[:-1])
        ori_y = np.concatenate((ori_x[1:], -(ori_x[np.newaxis, -2])))
        ori_x = np.concatenate((ori_x, ori_x[np.newaxis, -1]))
        ori_y = np.concatenate((ori_y, ori_y[np.newaxis, -1]))

        rot_axis_x = norm(np.cross(ori_x, tgt_x))
        tor_x = get_torsion(ori_x, tgt_x, rot_axis_x)
        ori_y_rot = rotation(ori_y, rot_axis_x, tor_x.reshape(-1, 1))
        ori_y_proj = ori_y_rot.copy()
        ori_y_proj[:, 1] = 0.
        ori_y_proj = norm(ori_y_proj)
        l_ori_y_proj = len(ori_y_proj)
        tor_y = get_torsion(ori_y_proj,
                                np.tile(tgt_y, (l_ori_y_proj, 1)),
                                np.tile(rot_axis_y, (l_ori_y_proj, 1)))

        knn_sincos = []
        for i, center in enumerate(ca):
            ca_ = ca - center
            global_indexs = nn_indexs[i]
            ca_xrot = rotation(ca_[global_indexs],
                                   np.tile(rot_axis_x[i], (k, 1)),
                                   np.tile(tor_x[i], (k, 1)))
            ca_rot = rotation(ca_xrot,
                                  np.tile(rot_axis_y, (k, 1)),
                                  np.tile(tor_y[i], (k, 1)))

            sin_1 = ca_rot[1:, 0] / \
                np.sqrt(np.square(ca_rot[1:, 0]) + np.square(ca_rot[1:, 1]))
            cos_1 = ca_rot[1:, 1] / \
                np.sqrt(np.square(ca_rot[1:, 0]) + np.square(ca_rot[1:, 1]))

            cos_2 = ca_rot[1:, 2]/knn_distance[i, 1:].reshape(-1)
            sin_2 = np.sqrt(1-np.square(cos_2))

            knn_sincos.append(np.concatenate([np.zeros((1, 4)), np.array(
                [sin_1, cos_1, sin_2, cos_2]).T]))

        knn_sincos = np.array(knn_sincos)
        knn_rep = np.concatenate(
            (knn_sincos, knn_distance, relative_indexs, knn_feature), -1)
        return knn_rep.astype('float32')

    # 3+1+1+3
    def knn_struc_rep_135(self, ca, seq, k=15):
        dismap = MapDis(ca)
        nn_indexs = np.argsort(dismap, axis=1)[:, :k]
        relative_indexs = nn_indexs.reshape(-1, k, 1) - \
            nn_indexs[:, 0].reshape(-1, 1, 1).astype('float32')
        relative_indexs /= self.index_norm
        seq_embeded = self.aa_encoder.encode(seq)
        knn_feature = np.array(seq_embeded)[nn_indexs]
        knn_distance = [dismap[i][nn_indexs[i]] for i in range(len(nn_indexs))]
        knn_distance = np.array(knn_distance).reshape(-1, k, 1)

        knn_orient = []
        for i in range(len(nn_indexs)):
            orient = norm(ca[nn_indexs[i]][1:] - ca[i])
            knn_orient.append(np.concatenate([np.zeros((1, 3)), orient]))
        knn_orient = np.array(knn_orient)

        # print('SPHER', knn_orient.shape, knn_distance.shape, relative_indexs.shape, knn_feature.shape)
        knn_rep = np.concatenate(
            (knn_orient, knn_distance, relative_indexs, knn_feature), -1)
        return knn_rep.astype('float32')

    def contact_map(self, ca, seq='', cutoff=8):
        dismap = MapDis(ca)
        conmap = np.zeros_like(dismap)
        conmap[dismap < cutoff] = 1.
        return conmap.astype('float32')

    def distance_map(self, ca, seq=''):
        return MapDis(ca).astype('float32')

    def image_struc_rep(self, ca, seq, resolution=128, box_size=8, compress=True, pad=4):
        arrays = []
        tgt_x = np.array([0, 1, 0])
        rot_axis_y = tgt_x
        tgt_y = np.array([1, 0, 0])
        ori_x = norm(ca[1:] - ca[:-1])
        ori_y = np.concatenate((ori_x[1:], -(ori_x[np.newaxis, -2])))

        centers = ca.copy()
        ori_x = np.concatenate((ori_x, ori_x[np.newaxis, -1]))
        ori_y = np.concatenate((ori_y, ori_y[np.newaxis, -1]))
        rot_axis_x = norm(np.cross(ori_x, tgt_x))

        tor_x = get_torsion(ori_x, tgt_x, rot_axis_x)
        ori_y_rot = rotation(ori_y, rot_axis_x, tor_x.reshape(-1, 1))
        ori_y_proj = ori_y_rot.copy()
        ori_y_proj[:, 1] = 0.
        ori_y_proj = norm(ori_y_proj)
        l_ori_y_proj = len(ori_y_proj)
        tor_y = get_torsion(ori_y_proj,
                                np.tile(tgt_y, (l_ori_y_proj, 1)),
                                np.tile(rot_axis_y, (l_ori_y_proj, 1)))

        for i, center in enumerate(centers):
            ca_ = ca - center
            global_indexs = np.where(get_len(
                ca_) < (box_size + pad)*np.sqrt(3))[0]
            local_indexs = global_indexs - i

            num_local_atoms = len(global_indexs)
            ca_xrot = rotation(ca_[global_indexs],
                               np.tile(rot_axis_x[i],
                                      (num_local_atoms, 1)),
                               np.tile(tor_x[i], (num_local_atoms, 1)))
            ca_rot = rotation(ca_xrot,
                              np.tile(rot_axis_y, (num_local_atoms, 1)),
                              np.tile(tor_y[i], (num_local_atoms, 1)))

            local_atoms = []
            for j, idx in enumerate(global_indexs):
                if np.max(np.abs(ca_rot[j])) < box_size + pad:
                    local_atoms.append(
                        Atom(seq[idx], local_indexs[j], ca_rot[j][0], ca_rot[j][1], ca_rot[j][2]))

            arrays.append(Arraylize(resolution=resolution,
                                    size=box_size,
                                    atoms=local_atoms,
                                    indexs=local_indexs,
                                    aa_encoder=self.aa_encoder).array)

        arrays = np.array(arrays, dtype='float32')

        if compress:
            shape = arrays.shape
            keys = arrays.nonzero()
            values = arrays[keys]
            com_ary = [shape, keys, values.astype('float32')]
            return com_ary
        else:
            return arrays


class AminoacidEncoder(object):
    def __init__(self, aa_format='property'):
        self.aa_list = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                        'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
        self.index = {}
        for aa in self.aa_list:
            self.index[aa] = self.aa_list.index(aa)

        if aa_format == 'onehot':
            self.encoder = np.eye(20)

        elif aa_format == 'property':
            self.hydropathicity = [1.8, 2.5, -3.5, -3.5, 2.8, -0.4, -3.2, 4.5, -3.9, 3.8,
                                   1.9, -3.5, -1.6, -3.5, -4.5, -0.8, -0.7, 4.2, -0.9, -1.3]
            self.bulkiness = [11.5, 13.46, 11.68, 13.57, 19.8, 3.4, 13.69, 21.4, 15.71, 21.4,
                              16.25, 12.82, 17.43, 14.45, 14.28, 9.47, 15.77, 21.57, 21.67, 18.03]
            self.flexibility = [14.0, 0.05, 12.0, 5.4, 7.5, 23.0, 4.0, 1.6, 1.9, 5.1,
                                0.05, 14.0, 0.05, 4.8, 2.6, 19.0, 9.3, 2.6, 0.05, 0.05]
            self.property_norm()

            self.encoder = np.stack([self.hydropathicity,
                                     self.bulkiness,
                                     self.flexibility]).T.astype('float32')

    def property_norm(self):
        self.hydropathicity = (5.5 - np.array(self.hydropathicity)) / 10
        self.bulkiness = np.array(self.bulkiness) / max(self.bulkiness)
        self.flexibility = (25 - np.array(self.flexibility)) / 25

    def encode(self, seq):
        if len(seq[0]) == 3:
            seq = [AA_ALPHABET_REV[aa] for aa in seq]
        indexs = np.array([self.index[aa] for aa in seq])
        return self.encoder[indexs]
