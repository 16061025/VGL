import glob
import os
import numpy as np
import random
from tqdm import tqdm

from data_utils.data_utils import tsnp2vg, divide_train_test, data2dataset, split_array

chunk_size = 256



def process_ADpatient(data_path):

    vg_list = []

    ADpatient_path = os.path.join(data_path, "AD")
    # only Eye_closed

    patients_dir = os.path.join(ADpatient_path, "Eyes_closed")
    pattern = os.path.join(patients_dir, '*')
    patient_dir_list = [path for path in glob.glob(pattern) if os.path.isdir(path)]

    for patient_dir in tqdm(patient_dir_list):
        channel_files = glob.glob(patient_dir+"/*.txt")

        for channel_file in channel_files:
            channel_ts_np = np.loadtxt(channel_file)
            segments = split_array(channel_ts_np, chunk_size)
            #downsampled_ts_np = signal.resample(channel_ts_np, 256)
            #channel_name = channel_file.split("//")[-1][:-4]
            for segment in segments:
                adj, feat = tsnp2vg(segment)
                vg_list.append([feat, adj, 1])

    return vg_list


def process_Healthy(data_path):
    vg_list = []

    Healthy_path = os.path.join(data_path, "Healthy")
    # only Eye_closed

    patients_dir = os.path.join(Healthy_path, "Eyes_closed")
    pattern = os.path.join(patients_dir, '*')
    patient_dir_list = [path for path in glob.glob(pattern) if os.path.isdir(path)]

    for patient_dir in tqdm(patient_dir_list):
        channel_files = glob.glob(patient_dir + "/*.txt")

        for channel_file in channel_files:
            channel_ts_np = np.loadtxt(channel_file)
            segments = split_array(channel_ts_np, chunk_size)
            # downsampled_ts_np = signal.resample(channel_ts_np, 256)
            # channel_name = channel_file.split("//")[-1][:-4]
            for segment in segments:
                adj, feat = tsnp2vg(segment)
                vg_list.append([feat, adj, 0])

    return vg_list


def load_Alzheimer_dataset(args):
    data_path = os.path.join(args.data_dir, args.dataset)
    AD_vg_list = process_ADpatient(data_path)
    healthy_vg_list = process_Healthy(data_path)
    vg_list = AD_vg_list + healthy_vg_list
    random.shuffle(vg_list)
    random.shuffle(vg_list)

    train_data, test_data = divide_train_test(vg_list, 0.8)

    train_dataset = data2dataset(train_data)
    test_dataset = data2dataset(test_data)
    return train_dataset, test_dataset


