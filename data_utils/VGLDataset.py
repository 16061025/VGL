from torch.utils.data import Dataset
import numpy as np
import torch
from hgcn.utils.data_utils import sparse_mx_to_torch_sparse_tensor
import networkx as nx
import ts2vg
import scipy.sparse as sp

class VGLDataset(Dataset):
    def __init__(self, feats, adjs, labels):
        self.feats = feats
        self.adjs = adjs
        self.labels = labels

    def __len__(self):
        return len(self.feats)

    def __getitem__(self, idx):
        return self.feats[idx], self.adjs[idx], self.labels[idx]
