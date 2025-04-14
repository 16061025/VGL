import datetime
import logging
import math
import os
import pickle
import random
import time

import networkx as nx
import ts2vg
from tqdm import tqdm

from args import parser
from VGLModel.data_utils import load_VGL_dataset, VGLDataset
import scipy.io as sio
import torch
from torch.utils.data import DataLoader
from VGLModel.model import VGLModel, VGLModel_shareHGCN, VGLModel_MLP, VGLModel_MLP_bonn
from VGLModel.model import train_VGLModel, test_VGLModel
import numpy as np
import scipy.sparse as sp
import torch.nn.functional as F
from hgcn.utils.data_utils import  sparse_mx_to_torch_sparse_tensor
bonn_data_path = "./data/bonn/"
def load_data(dataset, train_label):
    path = bonn_data_path

    path_data_t = path + dataset + '.mat'
    path_label = path + train_label + '.mat'

    data_t = sio.loadmat(path_data_t)[dataset]
    data_label = sio.loadmat(path_label)[train_label]
    data_label = data_label.flatten()

    return data_t, data_label


def convert_to_graph(data_f, label):
    def tsnp2vg(ts_np):
        vg = ts2vg.NaturalVG()
        vg.build(ts_np)
        graph_edges = vg.edges
        graph = nx.Graph(graph_edges)
        adj = nx.adjacency_matrix(graph)

        node_feature_1 = np.arange(0, len(ts_np))[:, np.newaxis]
        node_feature_2 = ts_np[:, np.newaxis]
        node_features = np.hstack((node_feature_1, node_feature_2))

        feat = torch.Tensor(node_features).numpy()
        adj = sparse_mx_to_torch_sparse_tensor(sp.csr_matrix(adj)).to_dense().numpy()

        return adj, feat

    vg_list = []

    for i in tqdm(range(len(data_f))):
        data = data_f[i]
        adj_f, adj_feat = tsnp2vg(data)
        vg_list.append([adj_feat, adj_f, label[i]])

    random.shuffle(vg_list)
    random.shuffle(vg_list)

    def divide_train_test(data, ratio=0.8):

        split_index = math.floor(ratio*len(data))

        train_data_list = data[0:split_index]
        test_data_list = data[split_index:]

        return train_data_list, test_data_list

    train_data, test_data = divide_train_test(vg_list, 0.8)

    def data2dataset(data):
        feat_index = 0
        adj_index= 1
        y_index = 2
        data_feats = [row[feat_index] for row in data]
        data_adjs = [row[adj_index] for row in data]
        data_y = [row[y_index] for row in data]

        data_feats = torch.Tensor(np.array(data_feats))
        data_adjs = torch.Tensor(np.array(data_adjs))
        data_y = torch.tensor(data_y, dtype=torch.int64)
        data_y = F.one_hot(data_y, num_classes=2).float()
        dataset = VGLDataset(data_feats, data_adjs, data_y)
        return dataset

    train_dataset = data2dataset(train_data)
    test_dataset = data2dataset(test_data)


    return train_dataset, test_dataset


if __name__ == '__main__':
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    log_dir = args.VGL_save_dir
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(log_dir, 'training.log'),
        level=logging.INFO,
        filemode='w',
        format='%(asctime)s - %(message)s'
    )

    print(args)
    logging.info(
        args
    )

    ##step 1 load dataset
    VGL_dataset_pickle_path = os.path.join(bonn_data_path, "dataset.pickle")
    if os.path.exists(VGL_dataset_pickle_path):
        print("load existing VGL dataset")
        with open(VGL_dataset_pickle_path, "rb") as f:
            VGL_train_data, VGL_test_data = pickle.load(f)
    else:
        dataset='AET'
        train_label = 'train_label_Bonn'
        train_data, train_label = load_data(dataset, train_label)
        # methods = {'WF1': method.overlook_wf1, 'OG': method.overlook, 'WOG': method.overlookg, 'WS': method.overlook_WS, 'V': method.LPvisibility_v, 'LV': method.LPvisibility_lv, 'H': method.LPhorizontal_h, 'LH': method.LPhorizontal_lh}

        # method flop

        VGL_train_data, VGL_test_data= convert_to_graph(train_data, train_label)

        with open(os.path.join(bonn_data_path, "dataset.pickle"), "wb") as f:
            pickle.dump((VGL_train_data, VGL_test_data), f)

    feats, adjs, y = VGL_train_data[0]
    args.n_nodes, args.feat_dim = feats.shape

    train_dataloader = DataLoader(VGL_train_data, batch_size=args.VGL_batch_size)
    test_dataloader = DataLoader(VGL_test_data, batch_size=args.VGL_batch_size)

    args.device = 'cuda:' + str(args.cuda) if int(args.cuda) >= 0 and torch.cuda.is_available() else 'cpu'
    device = args.device

    print(f"device is {device}")

    model = VGLModel_MLP_bonn(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.VGL_lr)

    print(model)
    logging.info(
        model
    )
    print("Trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: shape={param.shape}, dtype={param.dtype}")
    # for name, param in model.named_parameters():
    #     print(f"Name: {name}")
    #     print(f"Shape: {param.shape}")
    #     print(f"Requires Gradient: {param.requires_grad}")
    #     print("-" * 50)


    loss_fn = torch.nn.CrossEntropyLoss()
    _, class_counts = torch.unique(VGL_train_data.labels, dim=0, return_counts=True)
    class_counts = torch.flip(class_counts, dims=[0])
    class_weights = 1.0 / class_counts  # 逆样本频率
    class_weights = class_weights / class_weights.sum()
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    epochs = args.VGL_epochs
    for t in range(epochs):
        print(f"{datetime.datetime.now()} Epoch {t + 1}\n-------------------------------")

        res = train_VGLModel(model, train_dataloader, loss_fn, optimizer, args)
        loss = res["loss"]
        print(f"loss: {loss:>7f}")
        logging.info(
            f"Epoch [{t + 1}/{epochs}], train Loss: {loss:.4f}"
        )

        res = test_VGLModel(model, test_dataloader, loss_fn, args)
        correct = res['correct']
        test_loss = res['Angloss']
        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        logging.info(
            f"Test Error: Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
        )
    print("Done!")
    logging.info(
        f"Done!"
    )



    # ##step 3 hyperbolic Graph convolution
    # ## independet GCN LP task
    # HGCN_data= create_HGCN_data_from_VG(EEG_visibility_graph_list)
    # HGCN_model = train_HGCN(HGCN_data, args)
    #
    # ##get brain region embedding
    # HGCN_model.eval()
    # all_patients_brain_region_embbeding = HGCN_model.encode(HGCN_data['features'], HGCN_data['adj_train_norm'])
    #
    # # ##step 4 Brain Graph
    # # calculate RDM
    # brain_graph_data = {}
    # RDM_model = RDMModel()
    # RDM_model.eval()
    # for brain_region_embbeding in all_patients_brain_region_embbeding:
    #     brain_graph = RDM_model(brain_region_embbeding)
    #     brain_graph_data[patientid] = brain_graph
    #
    # MochaGCN_train_loader,  MochaGCN_test_loader= BGdata2MGCNdataloader(brain_graph_data)
    # # ## step 5 Graph Fusion
    # train_MochaGCN(MochaGCN_train_loader,  MochaGCN_test_loader, args)
