from multiprocessing import Manager
from multiprocessing import Process

import torch
import os


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



def subprocess(process_args):
    '''
    transfer EEG row data into VG

    :param process_args:
    :return:
    '''

    share_res_list = process_args["share_res_list"]
    share_res_list_Lock = process_args["share_res_list_Lock"]
    data_list = process_args["data_list"]
    Process_id = process_args['Process_id']
    worker = process_args['worker']


    print("start process ", Process_id, "PID", os.getpid())


    vg_list = worker(data_list)


    share_res_list_Lock.acquire()

    for i in range(0, len(vg_list)):
        share_res_list.append(vg_list[i])

    share_res_list_Lock.release()
    print("finish process", process_args['Process_id'])
    return 0


def divide_data_list(data_list, process_cnt):
    '''
    divide N data into M group M=process_cnt
    :param data_list: a list contains N data
    :return: a list contains M sub list of data_list
    '''

    length = len(data_list)
    base_len = length//process_cnt
    remainder = length % process_cnt


    divided_data_list = []
    start = 0
    for i in range(process_cnt):
        end = start + (base_len + 1 if i < remainder else base_len)
        divided_data = data_list[start:end]
        divided_data_list.append(divided_data)
        start = end

    return divided_data_list

def multi_process_data_list(data_list, Worker, process_cnt):
    '''
    convert data into visual graph
    :param data_list: a list contains N data
           worker: function to procee data list
    :return:  a list contains visual graph of data
    '''

    def init_share_res_list(res_list):
        return res_list



    share_res_list = Manager().list()
    share_res_list = init_share_res_list(share_res_list)
    share_res_list_Lock = Manager().Lock()

    divided_data_list = divide_data_list(data_list, process_cnt)

    proecesses = []
    i = 0
    for data_sublist in divided_data_list:
        process_args = ({"data_list": data_sublist,
                         "share_res_list": share_res_list,
                         "share_res_list_Lock": share_res_list_Lock,
                         "Process_id": i,
                         "worker": Worker
                         },)
        p = Process(target=subprocess, args=process_args)
        proecesses.append(p)
        i = i + 1
        p.start()

    for p in proecesses:
        p.join()
    all_graph_list = list(share_res_list)
    return all_graph_list
