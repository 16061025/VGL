import os

import numpy as np
import random

from tqdm import tqdm
import scipy.io as sio
from data_utils.data_utils import tsnp2vg, divide_train_test, data2dataset, split_array

chunk_size = 256

def DREAMER2graphdataset(DRMEARstruct):



    vg_list = []
    Data = DRMEARstruct['Data'][0][0][0]
    for patient_i in tqdm(range(len(Data))): #23 people
        data = Data[patient_i][0][0]
        ScoreValence = data['ScoreValence']
        EEG = data['EEG'][0][0]
        stimuli = EEG['stimuli']
        for stim_j in range(len(stimuli)): #18 stimuli
            Valence = ScoreValence[stim_j][0]
            if Valence < 2.5:
                Valence = 0
            else:
                Valence = 1
            EEG_data = stimuli[stim_j][0]
            EEG_data_np = np.array(EEG_data).T
            for ch_k in range(len(EEG_data_np)): #14 channel per stimuli
                channel_ts_np = EEG_data_np[ch_k]
                #split 256 per section
                segments = split_array(channel_ts_np, chunk_size)
                for segment in segments:
                    adj, feat = tsnp2vg(segment)
                    vg_list.append([feat, adj, Valence])

    random.shuffle(vg_list)
    random.shuffle(vg_list)

    train_data, test_data = divide_train_test(vg_list, 0.8)

    train_dataset = data2dataset(train_data)
    test_dataset = data2dataset(test_data)


    return train_dataset, test_dataset

def load_DREAMER_dataset(args):
    path_mat_data = os.path.join(args.data_dir, args.dataset, 'DREAMER.mat')

    DREAMERstruct = sio.loadmat(path_mat_data)["DREAMER"]

    VGL_train_data, VGL_test_data = DREAMER2graphdataset(DREAMERstruct)
    return VGL_train_data, VGL_test_data