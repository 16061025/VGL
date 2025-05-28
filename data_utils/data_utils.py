import glob

import pickle




import torch



from hgcn.utils.data_utils import sparse_mx_to_torch_sparse_tensor
import torch.nn.functional as F

from data_utils.VGLDataset import VGLDataset

import ts2vg


import numpy as np
import networkx as nx
import scipy.sparse as sp
import math


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
    split_index = math.floor(ratio * len(data))

    train_data_list = data[0:split_index]
    test_data_list = data[split_index:]

    return train_data_list, test_data_list


def data2dataset(data):
    feat_index = 0
    adj_index = 1
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

def split_array(arr, chunk_size=256):
    n_chunks = len(arr) // chunk_size
    return arr[:n_chunks * chunk_size].reshape(-1, chunk_size)
