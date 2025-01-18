import os
import mne
import matplotlib.pyplot as plt
import numpy as np
import glob

import torch
import random
import math
from VisibilityGraph.VisibilityGraph import construct_EEG_visibility_graph, \
    construct_EEG_visibility_graph_single_process
from hgcn.utils.data_utils import  sparse_mx_to_torch_sparse_tensor
from torch.utils.data import Dataset
import torch.nn.functional as F
import pickle
from tqdm import tqdm





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


def load_EEG_raw_data(args):
    '''

    :param args:
    :return: a list contians eeg raw data of N patients
    '''
    EEG_raw_data_pickle_path = os.path.join(args.data_dir, "all_raw_list.pickle")
    if os.path.exists(EEG_raw_data_pickle_path):
        print("load existing eeg raw data")
        with open(EEG_raw_data_pickle_path, "rb") as f:
            EEG_raw_data = pickle.load(f)
        return EEG_raw_data
    else:
        print("load eeg raw data from .set files")


    DATA_PATH = args.data_dir

    nor_DATA_PATH = os.path.normpath(DATA_PATH)
    splited_DATA_PATH = nor_DATA_PATH.split(os.sep)
    DATA_PATH_len = len(splited_DATA_PATH)
    patientID_in_path_index = DATA_PATH_len + 3
    label_in_path_index = DATA_PATH_len + 1

    # disease to be loaded
    load_disease = ['AD', 'PD', 'MS', "HC", "bvFTD"]

    EEG_DATA_PATH = os.path.join(DATA_PATH, "EEG data")
    set_file_paths = []
    for disease_name in load_disease:
        disease_set_file_paths = glob.glob(EEG_DATA_PATH + '/*' + disease_name + '*/**/*.set', recursive=True)
        set_file_paths += disease_set_file_paths

    all_raw_list = []
    for set_file_path in tqdm(set_file_paths):
        if os.path.isfile(set_file_path):
            print(set_file_path)

            try:
                raw = mne.io.read_raw_eeglab(set_file_path)
            except:
                epochdata = mne.io.read_epochs_eeglab(set_file_path)
                data = epochdata.get_data()
                data_np = np.array(data)
                data_np = np.concatenate(data_np, axis=1)
                data_info = epochdata.info
                raw = mne.io.RawArray(data_np, data_info)
            nor_set_file_path = os.path.normpath(set_file_path)
            splited_path = nor_set_file_path.split(os.sep)
            label = splited_path[label_in_path_index][2:]
            patientID = splited_path[patientID_in_path_index]
            # print(f"new patient:[patientID:{patientID}, label:{label}]")

            all_raw_list.append({'patientID': patientID,
                                 'raw': raw,
                                 'label': label})
        if len(all_raw_list) == 4:
            break

    with open(EEG_raw_data_pickle_path, "wb") as f:
        pickle.dump(all_raw_list, f)
        print(f"eeg raw data has been saved to {EEG_raw_data_pickle_path}")

    return all_raw_list

def construct_VGL_dataset(VG_list):
    data_feats = []
    data_adjs = []
    data_y = []
    random.shuffle(VG_list)
    for patient_data in tqdm(VG_list):
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
        #print("user feat len:", len(patient_feats))
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

    test_data_feats = data_feats[split_index:]
    test_data_adjs = data_adjs[split_index:]
    test_data_y = data_y[split_index:]
    test_dataset = VGLDataset(test_data_feats, test_data_adjs, test_data_y)

    return train_dataset, test_dataset

def load_VG_list_data(args):
    EEG_VG_list_pickle_path = os.path.join(args.data_dir, "all_VG_list.pickle")
    if os.path.exists(EEG_VG_list_pickle_path):
        print("load existing eeg VG data")
        with open(EEG_VG_list_pickle_path, "rb") as f:
            EEG_visibility_graph_list = pickle.load(f)
    else:
        print("construct eeg VG data from eeg raw data")
        EEG_raw_data = load_EEG_raw_data(args)


        EEG_visibility_graph_list = construct_EEG_visibility_graph_single_process(EEG_raw_data)
        with open(EEG_VG_list_pickle_path, "wb") as f:
            pickle.dump(EEG_visibility_graph_list, f)
            print(f"eeg VG data has been saved to {EEG_VG_list_pickle_path}")

    return EEG_visibility_graph_list

def load_VGL_dataset(args):
    VGL_dataset_pickle_path = os.path.join(args.data_dir, "VGL_dataset.pickle")

    if os.path.exists(VGL_dataset_pickle_path):
        print("load existing VGL dataset")
        with open(VGL_dataset_pickle_path, "rb") as f:
            VGL_train_data, VGL_test_data = pickle.load(f)
    else:
        print("construct VGL dataset")
        VG_list = load_VG_list_data(args)
        VGL_train_data, VGL_test_data = construct_VGL_dataset(VG_list)
        with open(VGL_dataset_pickle_path, "wb") as f:
            pickle.dump((VGL_train_data, VGL_test_data), f)
            print(f"VGL dataset has been saved to {VGL_dataset_pickle_path}")


    return VGL_train_data, VGL_test_data