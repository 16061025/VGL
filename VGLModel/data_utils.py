import os
import mne
import matplotlib.pyplot as plt
import numpy as np
import glob

import torch
import random
import math
from VisibilityGraph.VisibilityGraph import construct_EEG_visibility_graph
from hgcn.utils.data_utils import  sparse_mx_to_torch_sparse_tensor

def load_EEG_data(args):
    '''

    :param args:
    :return: a list contians eeg raw data of N patients
    '''
    # disease to be loaded
    load_disease = ["AD"]

    EEG_DATA_PATH = os.path.join("data/EEG data/")
    set_file_paths = []
    for disease_name in load_disease:
        disease_set_file_paths = glob.glob(EEG_DATA_PATH + '/*'+disease_name+'*/**/*.set', recursive=True)
        set_file_paths += disease_set_file_paths

    all_raw_list = []
    for set_file_path in set_file_paths:
        if os.path.isfile(set_file_path):
            print(set_file_path)
            raw = mne.io.read_raw_eeglab(set_file_path)

            nor_set_file_path = os.path.normpath(set_file_path)
            splited_path = nor_set_file_path.split(os.sep)
            label = splited_path[2][2:]
            patientID = splited_path[4]
            all_raw_list.append({'patientID': patientID,
                                'raw': raw,
                                'label': label})
        if len(all_raw_list) == 2:
            break
    return all_raw_list



def load_MRI_data():

    return




def construct_VGL_dataset(VG_list):
    AD_data = []
    non_AD_data = []
    for patient_data in VG_list:
        x_data = patient_data['VG']
        Xs = [[] for i in range(len(x_data))]
        adjs = [[] for i in range(len(x_data))]
        for i in range(len(x_data)):
            for j in range(len(x_data[i])):
                x = torch.Tensor(x_data[i][j][1])
                adj = sparse_mx_to_torch_sparse_tensor(x_data[i][j][0])
                Xs[i].append(x)
                adjs[i].append(adj)
        if patient_data['label']=="AD":
            y = torch.Tensor(1)
            AD_data.append((Xs,adjs,y))
        else:
            y = torch.Tensor(0)
            non_AD_data.append((Xs,adjs,y))
    ratio = 0.8
    split_index = math.floor(ratio*len(VG_list))
    random.shuffle(AD_data)
    random.shuffle(non_AD_data)
    train_data = AD_data[0:split_index] + non_AD_data[0:split_index]
    test_data = AD_data[split_index:] + non_AD_data[split_index:]
    return train_data, test_data