import os

import ts2vg
from tqdm import tqdm
from multiprocessing import Manager
from multiprocessing import Process
import numpy as np
import networkx as nx
import scipy.sparse as sp
import math
import scipy


def Worker(process_args):
    '''
    transfer EEG row data into VG

    :param process_args:
    :return:
    '''

    #paras
    is_devide_ts = False
    sections = 10
    downsample_factor = 10
    resample_num = 30

    share_res_list = process_args["share_res_list"]
    share_res_list_Lock = process_args["share_res_list_Lock"]
    EEG_data = process_args["EEG_data"]
    Process_id = process_args['Process_id']


    print("start process ", Process_id, "PID", os.getpid())

    all_graph_list = []
    for patient_EEG_data in EEG_data:
        patient_EEG_raw = patient_EEG_data['raw']
        N_channels = len(patient_EEG_raw.ch_names)
        patient_VG_list = []
        for i in range(N_channels):
            channel_ts, times = patient_EEG_raw[i, :]
            channel_ts_np = np.array(channel_ts).flatten()

            def tsnp2vg(ts_np):
                vg = ts2vg.NaturalVG()
                vg.build(ts_np)
                graph_edges = vg.edges
                graph = nx.Graph(graph_edges)
                adj = nx.adjacency_matrix(graph)

                node_feature_1 = np.arange(0, len(ts_np))[:, np.newaxis]
                node_feature_2 = ts_np[:, np.newaxis]
                node_features = np.hstack((node_feature_1, node_feature_2))
                return sp.csr_matrix(adj), node_features

            if is_devide_ts:
                #devide chancel ts into N subsections N=secitons
                #make sure it can be divided into N equal arrays
                clip_length = len(channel_ts_np) - (len(channel_ts_np)%sections)
                channel_ts_np = channel_ts_np[0:clip_length]

                devided_channel_ts_np_list = np.split(channel_ts_np, sections)

                #downsample every subsection
                devided_VG_list = []
                for ts in devided_channel_ts_np_list:
                    downsampled_ts = scipy.signal.resample(ts, resample_num)
                    #downsampled_ts = scipy.signal.decimate(ts, downsample_factor)
                    devided_VG_list.append(tsnp2vg(downsampled_ts))
                patient_VG_list.append(devided_VG_list)
            else:
                patient_VG_list.append(tsnp2vg(channel_ts_np))
        patient_EEG_data.pop('raw')
        patient_EEG_data['VG'] = patient_VG_list
        all_graph_list.append(patient_EEG_data)

    share_res_list_Lock.acquire()

    for i in range(0, len(all_graph_list)):
        share_res_list.append(all_graph_list[i])

    share_res_list_Lock.release()
    print("finish process", process_args['Process_id'])
    return 0





def construct_EEG_visibility_graph(EEG_data):
    '''
    convert EEG into visual graph
    :param EEG_data: a list contains EEG_raw data of N patients
    :return:  a list contains visual graph of EEG_raw data of N patients, each patient has N_cha*N_sec graph
    '''

    def init_share_res_list(res_list):
        return res_list

    share_res_list = Manager().list()
    share_res_list = init_share_res_list(share_res_list)

    share_res_list_Lock = Manager().Lock()

    def divide_EEG_data(EEG_data):
        '''
        divide N EEG_raw data into M group M=process_cnt
        :param EEG_data: a list contains EEG_raw data of N patients
        :return: a list contains M list of EEG_raw
        '''
        divided_EEG_data_list = []
        process_cnt = 1
        interval = max(len(EEG_data)//process_cnt, 1)
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

def construct_EEG_visibility_graph_single_process(EEG_data):
    is_devide_ts = True
    downsample_factor = 1000
    sections = 5
    resample_num = 30
    all_graph_list = []
    for patient_EEG_data in EEG_data:
        patient_EEG_raw = patient_EEG_data['raw']
        N_channels = len(patient_EEG_raw.ch_names)
        patient_VG_list = []
        for i in range(N_channels):
            channel_ts, times = patient_EEG_raw[i, :]
            #channel_ts = patient_EEG_raw[i, :]
            channel_ts_np = np.array(channel_ts).flatten()

            def tsnp2vg(ts_np):
                vg = ts2vg.NaturalVG()
                vg.build(ts_np)
                graph_edges = vg.edges
                graph = nx.Graph(graph_edges)
                adj = nx.adjacency_matrix(graph)

                node_feature_1 = np.arange(0, len(ts_np))[:, np.newaxis]
                node_feature_2 = ts_np[:, np.newaxis]
                node_features = np.hstack((node_feature_1, node_feature_2))
                return [sp.csr_matrix(adj), node_features]

            if is_devide_ts:
                # devide chancel ts into N subsections N=secitons
                # make sure it can be divided into N equal arrays
                clip_length = len(channel_ts_np) - (len(channel_ts_np) % sections)
                channel_ts_np = channel_ts_np[0:clip_length]

                devided_channel_ts_np_list = np.split(channel_ts_np, sections)

                # downsample every subsection
                devided_VG_list = []
                for ts in devided_channel_ts_np_list:
                    downsampled_ts = scipy.signal.resample(ts, resample_num)
                    # downsampled_ts = scipy.signal.decimate(ts, downsample_factor)
                    devided_VG_list.append(tsnp2vg(downsampled_ts))
                patient_VG_list.append(devided_VG_list)
            else:
                patient_VG_list.append(tsnp2vg(channel_ts_np))
        patient_EEG_data.pop('raw')
        patient_EEG_data['VG'] = patient_VG_list
        all_graph_list.append(patient_EEG_data)
    return all_graph_list