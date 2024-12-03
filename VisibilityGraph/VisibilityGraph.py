import os

import ts2vg
from tqdm import tqdm
from multiprocessing import Manager
from multiprocessing import Process
import numpy as np
import networkx as nx
import scipy.sparse as sp
import math


def Worker(process_args):
    '''
    transfer EEG row data into VG

    :param process_args:
    :return:
    '''

    share_res_list = process_args["share_res_list"]
    share_res_list_Lock = process_args["share_res_list_Lock"]
    EEG_data = process_args["EEG_data"]
    Process_id = process_args['Process_id']
    is_devide_ts = False
    sections = 10
    downsample_rate = 0.1

    print("start process ", Process_id, "PID", os.getpid())

    all_graph_list = []
    for patient_EEG_raw in EEG_data:
        N_channels = len(patient_EEG_raw.ch_names)
        for i in range(N_channels):
            channel_ts = patient_EEG_raw[i,:][0]
            channel_ts_np = np.array(channel_ts).flatten()
            if is_devide_ts:
                #devide
                devided_channel_ts_np_list = np.split(channel_ts_np, sections)
                #downsample
                devided_VG_list = []
                for ts in devided_channel_ts_np_list:
                    section_length = math.ceil(1/downsample_rate)
                    pad_length = section_length - (len(ts) % section_length)
                    ts = np.pad(ts, (0, pad_length), mode="constant", constant_values=0)
                    ts = ts.reshape(-1, section_length).mean(axis=1)
                    vg = ts2vg.NaturalVG()
                    vg.build(channel_ts_np)
                    graph_edges = vg.edges
                    graph = nx.Graph(graph_edges)
                    adj = nx.adjacency_matrix(graph)

                    node_feature_1 = np.arange(0, len(channel_ts_np))[:, np.newaxis]
                    node_feature_2 = channel_ts_np[:, np.newaxis]
                    node_features = np.hstack(node_feature_1, node_feature_2)
                    devided_VG_list.append((sp.csr_matrix(adj), node_features))
                all_graph_list.append(devided_VG_list)
            else:
                vg = ts2vg.NaturalVG()
                vg.build(channel_ts_np)
                graph_edges = vg.edges
                graph = nx.Graph(graph_edges)
                adj = nx.adjacency_matrix(graph)

                node_feature_1 = np.arange(0, len(channel_ts_np))[:, np.newaxis]
                node_feature_2 = channel_ts_np[:, np.newaxis]
                node_features = np.hstack(node_feature_1, node_feature_2)

                all_graph_list.append((sp.csr_matrix(adj), node_features))

    share_res_list_Lock.acquire()

    for i in range(0, len(all_graph_list)):
        share_res_list.append(all_graph_list[i])

    share_res_list_Lock.release()
    print("finish process", process_args['Process_id'])
    return 0





def construct_EEG_visibility_grapy(EEG_data):

    def init_share_res_list(res_list):
        return res_list

    share_res_list = Manager().list()
    share_res_list = init_share_res_list(share_res_list)

    share_res_list_Lock = Manager().Lock()

    def divide_EEG_data(EEG_data):
        divided_EEG_data_list = []
        process_cnt = 2
        interval = len(EEG_data)//process_cnt
        start_index = np.arange(0, process_cnt+1) * interval
        start_index[-1] = len(EEG_data)
        for i in range(process_cnt):
            divided_EEG_data = EEG_data[start_index[i]:start_index[i+1]]
            divided_EEG_data_list.append(divided_EEG_data)

        return divided_EEG_data_list

    divided_EEG_data_list = divide_EEG_data(EEG_data)

    proecesses = []
    i = 0
    for divided_EEG_data in divided_EEG_data_list:
        process_args = ({"EEG_data": divided_EEG_data,
                         "share_res_list": share_res_list,
                         "share_res_list_Lock": share_res_list_Lock,
                         "Process_id": i,
                         },)
        p = Process(target=Worker, args=process_args)
        proecesses.append(p)
        i = i + 1
        p.start()

    for p in proecesses:
        p.join()
    all_graph_list = list(share_res_list)
    return all_graph_list