import os

import numpy as np


import torch
import random
import math

from hgcn.utils.data_utils import sparse_mx_to_torch_sparse_tensor
import torch.nn.functional as F
from tqdm import tqdm
import scipy.io as sio
import networkx as nx
import ts2vg
import scipy.sparse as sp
from data_utils.VGLDataset import VGLDataset

def bonn2graphdataset(data_f, label):
    def tsnp2vg(ts_np):
        vg = ts2vg.NaturalVG()
        vg.build(ts_np)
        graph_edges = vg.edges
        graph = nx.Graph(graph_edges)
        adj = nx.adjacency_matrix(graph)

        node_feature_1 = np.arange(0, len(ts_np))[:, np.newaxis]
        node_feature_2 = ts_np[:, np.newaxis]
        node_features = np.hstack((node_feature_1, node_feature_2))

        feat = torch.Tensor(node_features).numpy()
        adj = sparse_mx_to_torch_sparse_tensor(sp.csr_matrix(adj)).to_dense().numpy()

        return adj, feat

    def divide_train_test(data, ratio=0.8):

        split_index = math.floor(ratio*len(data))

        train_data_list = data[0:split_index]
        test_data_list = data[split_index:]

        return train_data_list, test_data_list

    def data2dataset(data):
        feat_index = 0
        adj_index= 1
        y_index = 2
        data_feats = [row[feat_index] for row in data]
        data_adjs = [row[adj_index] for row in data]
        data_y = [row[y_index] for row in data]

        data_feats = torch.Tensor(np.array(data_feats))
        data_adjs = torch.Tensor(np.array(data_adjs))
        data_y = torch.tensor(data_y, dtype=torch.int64)
        data_y = F.one_hot(data_y, num_classes=2).float()
        dataset = VGLDataset(data_feats, data_adjs, data_y)
        return dataset

    vg_list = []

    for i in tqdm(range(len(data_f))):
        data = data_f[i]
        adj_f, adj_feat = tsnp2vg(data)
        vg_list.append([adj_feat, adj_f, label[i]])

    random.shuffle(vg_list)
    random.shuffle(vg_list)


    train_data, test_data = divide_train_test(vg_list, 0.8)

    train_dataset = data2dataset(train_data)
    test_dataset = data2dataset(test_data)


    return train_dataset, test_dataset

def load_bonn_dataset(args):
    path_data_t = os.path.join(args.data_dir, args.dataset, 'AET.mat')
    path_label = os.path.join(args.data_dir, args.dataset, 'train_label_Bonn.mat')

    data_t = sio.loadmat(path_data_t)["AET"]
    data_label = sio.loadmat(path_label)["train_label_Bonn"]
    data_label = data_label.flatten()

    VGL_train_data, VGL_test_data = bonn2graphdataset(data_t, data_label)
    return VGL_train_data, VGL_test_data