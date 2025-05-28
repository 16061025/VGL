import glob
from scipy.io import loadmat

import os

import numpy as np
from data_utils.data_utils import tsnp2vg, divide_train_test, data2dataset, split_array
import random

from tqdm import tqdm



chunk_size = 256

def construct_Epilepsy_dataset(data_path):

    def process_stage(stage_path, stage_name, stage_label):
        mat_file_paths = glob.glob(stage_path + "/*.mat")
        vg_list = []
        for mat_path in tqdm(mat_file_paths):
            mat_data = loadmat(mat_path)
            channel_ts_np = np.array(mat_data[stage_name])
            segments = split_array(channel_ts_np, chunk_size)

            for segment in segments:
                adj, feat = tsnp2vg(segment)
                vg_list.append([feat, adj, stage_label])


        return vg_list

    vg_list = []
    stage_names = ['ictal', 'interictal', 'preictal']
    stage_labels = [1 ,0, 0]
    for stage_name, stage_label in zip(stage_names, stage_labels):
        vg_list_tmp = process_stage(os.path.join(data_path, stage_name), stage_name, stage_label)
        vg_list = vg_list + vg_list_tmp


    random.shuffle(vg_list)
    random.shuffle(vg_list)


    train_data, test_data = divide_train_test(vg_list, 0.8)

    train_dataset = data2dataset(train_data)
    test_dataset = data2dataset(test_data)


    return train_dataset, test_dataset


def load_Epilepsy_dataset(args):

    data_path = os.path.join(args.data_dir, args.dataset)

    VGL_train_data, VGL_test_data = construct_Epilepsy_dataset(data_path)

    return VGL_train_data, VGL_test_data



