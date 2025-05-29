import glob
import pickle
import random
import os
import mne
from tqdm import tqdm
import numpy as np


chunk_size = 256
from data_utils.data_utils import tsnp2vg, divide_train_test, data2dataset, split_array

from data_utils.data_utils import multi_process_data_list

def construct_EEG_visibility_graph(EEG_data):

    vg_list = []

    for patient_EEG_data in tqdm(EEG_data):
        patient_EEG_raw = patient_EEG_data['raw']
        N_channels = len(patient_EEG_raw.ch_names)
        label = patient_EEG_data['label']
        if label == "ASD":
            label = 1
        else:
            label = 0

        N_channels = min(2, N_channels)

        for i in range(N_channels):
            channel_ts, times = patient_EEG_raw[i, :]
            #channel_ts = patient_EEG_raw[i, :]
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


    nor_DATA_PATH = os.path.normpath(os.path.join(args.data_dir, args.dataset))
    splited_DATA_PATH = nor_DATA_PATH.split(os.sep)
    DATA_PATH_len = len(splited_DATA_PATH)
    label_in_path_index = DATA_PATH_len +0

    set_path = os.path.join(args.data_dir, args.dataset)
    set_file_paths = glob.glob(set_path + "/*/*.set")


    all_raw_list = []
    for set_file_path in tqdm(set_file_paths):
        if os.path.isfile(set_file_path):
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
            label = splited_path[label_in_path_index][0:3]

            all_raw_list.append({'raw': raw,
                                 'label': label})


    return all_raw_list


def load_VG_list_data(args):
    EEG_VG_list_pickle_path = os.path.join(args.data_dir, args.dataset, "all_VG_list.pickle")
    if os.path.exists(EEG_VG_list_pickle_path):
        print("load existing eeg VG data")
        with open(EEG_VG_list_pickle_path, "rb") as f:
            visibility_graph_list = pickle.load(f)
    else:
        print("construct eeg VG data from eeg raw data")
        EEG_raw_data = load_EEG_raw_data(args)

        visibility_graph_list = multi_process_data_list(EEG_raw_data, construct_EEG_visibility_graph, args.n_processor)
        with open(EEG_VG_list_pickle_path, "wb") as f:
            pickle.dump(visibility_graph_list, f)
            print(f"eeg VG data has been saved to {EEG_VG_list_pickle_path}")

    return visibility_graph_list


def load_autism_dataset(args):
    vg_list = load_VG_list_data(args)

    random.shuffle(vg_list)
    random.shuffle(vg_list)

    train_data, test_data = divide_train_test(vg_list, 0.8)

    train_dataset = data2dataset(train_data)
    test_dataset = data2dataset(test_data)
    return train_dataset, test_dataset



