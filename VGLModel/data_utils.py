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
from torch.utils.data import Dataset
import torch.nn.functional as F

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
        if len(all_raw_list) == 10:
            break
    return all_raw_list



def load_MRI_data():

    return


def VGL_collate_fn(batch):
    return batch

class VGLDataset(Dataset):
    def __init__(self, feats, adjs, labels):
        self.feats = feats
        self.adjs = adjs
        self.labels = labels

    def __len__(self):
        return len(self.feats)

    def __getitem__(self, idx):
        return self.feats[idx], self.adjs[idx], self.labels[idx]

def construct_VGL_dataset(VG_list):
    data_feats = []
    data_adjs = []
    data_y = []
    random.shuffle(VG_list)
    for patient_data in VG_list:
        x_data = patient_data['VG']
        patient_feats = [[] for i in range(len(x_data))]
        patient_adjs = [[] for i in range(len(x_data))]
        for i in range(len(x_data)):
            for j in range(len(x_data[i])):
                feat = torch.Tensor(x_data[i][j][1]).numpy()
                adj = sparse_mx_to_torch_sparse_tensor(x_data[i][j][0]).to_dense().numpy()
                patient_feats[i].append(feat)
                patient_adjs[i].append(adj)
        if patient_data['label']=="AD":
            y = 1
        else:
            y = 0
        data_feats.append(patient_feats)
        data_adjs.append(patient_adjs)
        data_y.append(y)
    data_feats = torch.Tensor(np.array(data_feats))
    data_adjs = torch.Tensor(np.array(data_adjs))
    data_y = torch.tensor(data_y, dtype=torch.int64)
    data_y = F.one_hot(data_y, num_classes=2).float()
    ratio = 0.8
    split_index = math.floor(ratio*len(VG_list))

    train_data_feats = data_feats[0:split_index]
    train_data_adjs = data_adjs[0:split_index]
    train_data_y = data_y[0:split_index]
    train_dataset = VGLDataset(train_data_feats, train_data_adjs, train_data_y)

    test_data_feats = data_feats[0:split_index]
    test_data_adjs = data_adjs[0:split_index]
    test_data_y = data_y[split_index:]
    test_dataset = VGLDataset(test_data_feats, test_data_adjs, test_data_y)

    return train_dataset, test_dataset