import os
import mne
import matplotlib.pyplot as plt
import numpy as np
import glob

def load_EEG_data():
    EEG_DATA_PATH = os.path.join("data/EEG data/1_AD/AR")

    set_file_paths = glob.glob(EEG_DATA_PATH + '/**/*.set', recursive=True)
    all_raw_list = []
    for set_file_path in set_file_paths:
        if os.path.isfile(set_file_path):
            print(set_file_path)
            raw = mne.io.read_raw_eeglab(set_file_path)
            all_raw_list.append(raw)
    return all_raw_list

def load_MRI_data():

    return