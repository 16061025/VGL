import glob
from data_utils.data_utils import tsnp2vg, divide_train_test, data2dataset, split_array
from data_utils.data_utils import multi_process_data_list
import os
import pickle

import mne
import numpy as np

import random

from tqdm import tqdm

import re

chunk_size = 256



def construct_EEG_visibility_graph(EEG_data):

    vg_list = []

    for patient_EEG_data in tqdm(EEG_data):
        patient_EEG_raw = patient_EEG_data['raw']
        label = patient_EEG_data['label']
        if label == "MDD":
            label = 1
        else:
            label = 0


        N_channels = len(patient_EEG_raw.ch_names)
        N_channels = min(N_channels, 2)

        for i in tqdm(range(N_channels)):
            channel_ts, times = patient_EEG_raw[i, :]
            channel_ts_np = np.array(channel_ts).flatten()
            segments = split_array(channel_ts_np, chunk_size)
            for segment in segments:
                adj, feat = tsnp2vg(segment)

                vg_list.append([feat, adj, label])

    return vg_list


def load_EEG_raw_data(args):
    '''

    :param args:
    :return: a list contians eeg raw data of N patients
    '''

    print("load eeg raw data from .set files")


    edf_path = os.path.join(args.data_dir, args.dataset)
    edf_file_paths = glob.glob(edf_path + "/*.edf")
    # delete first 2 edf that begin with number
    edf_file_paths = edf_file_paths[2:]

    all_raw_list = []
    for edf_file_path in tqdm(edf_file_paths):
        if os.path.isfile(edf_file_path):
            nor_set_file_path = os.path.normpath(edf_file_path)
            splited_path = nor_set_file_path.split(os.sep)
            file_name = splited_path[-1][:-4]
            label, patientID, task = re.split(r"[\s]+", file_name)

            if task != "EC":
                continue

            try:
                raw = mne.io.read_raw_edf(edf_file_path)
            except:
                epochdata = mne.io.read_epochs_eeglab(edf_file_path)
                data = epochdata.get_data()
                data_np = np.array(data)
                data_np = np.concatenate(data_np, axis=1)
                data_info = epochdata.info
                raw = mne.io.RawArray(data_np, data_info)

            # print(f"new patient:[patientID:{patientID}, label:{label}]")

            all_raw_list.append({'patientID': patientID,
                                 'raw': raw,
                                 'label': label})

    return all_raw_list



def load_VG_list_data(args):
    EEG_VG_list_pickle_path = os.path.join(args.data_dir, args.dataset, "all_VG_list.pickle")
    if os.path.exists(EEG_VG_list_pickle_path):
        print("load existing eeg VG data")
        with open(EEG_VG_list_pickle_path, "rb") as f:
            EEG_visibility_graph_list = pickle.load(f)
    else:
        print("construct eeg VG data from eeg raw data")
        EEG_raw_data = load_EEG_raw_data(args)


        #EEG_visibility_graph_list = multi_process_data_list(EEG_raw_data, construct_EEG_visibility_graph, args.n_processor)
        EEG_visibility_graph_list =construct_EEG_visibility_graph(EEG_raw_data)
        with open(EEG_VG_list_pickle_path, "wb") as f:
            pickle.dump(EEG_visibility_graph_list, f)
            print(f"eeg VG data has been saved to {EEG_VG_list_pickle_path}")

    return EEG_visibility_graph_list

def load_MDD_dataset(args):

    vg_list = load_VG_list_data(args)

    random.shuffle(vg_list)
    random.shuffle(vg_list)

    train_data, test_data = divide_train_test(vg_list, 0.8)

    train_dataset = data2dataset(train_data)
    test_dataset = data2dataset(test_data)

    return train_dataset, test_dataset