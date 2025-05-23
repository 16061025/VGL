import argparse
from hgcn.utils.train_utils import add_flags_from_config

parser = argparse.ArgumentParser()

config_args = {
    #HGCN args
    'training_config': {
        'lr': (0.01, 'learning rate'),
        'dropout': (0.0, 'dropout probability'),
        'cuda': (-1, 'which cuda device to use (-1 for cpu training)'),
        'epochs': (5000, 'maximum number of epochs to train for'),
        'weight-decay': (0., 'l2 regularization strength'),
        'optimizer': ('Adam', 'which optimizer to use, can be any of [Adam, RiemannianAdam]'),
        'momentum': (0.999, 'momentum in optimizer'),
        'patience': (100, 'patience for early stopping'),
        'seed': (2025, 'seed for training'),
        'log-freq': (1, 'how often to compute print train/val metrics (in epochs)'),
        'eval-freq': (1, 'how often to compute val metrics (in epochs)'),
        'save': (0, '1 to save model and logs and 0 otherwise'),
        'save-dir': (None, 'path to save training logs and model weights (defaults to logs/task/date/run/)'),
        'sweep-c': (0, ''),
        'lr-reduce-freq': (None, 'reduce lr every lr-reduce-freq or None to keep lr constant'),
        'gamma': (0.5, 'gamma for lr scheduler'),
        'print-epoch': (True, ''),
        'grad-clip': (None, 'max norm for gradient clipping, or None for no gradient clipping'),
        'min-epochs': (100, 'do not early stop before min-epochs'),
        'share_encoder':(True, 'channel section visual graph share encoder')
    },
    'model_config': {
        'task': ('nc', 'which tasks to train on, can be any of [lp, nc]'),
        'model': ('GCN', 'which encoder to use, can be any of [Shallow, MLP, HNN, GCN, GAT, HyperGCN]'),
        'dim': (128, 'embedding dimension'),
        'manifold': ('Euclidean', 'which manifold to use, can be any of [Euclidean, Hyperboloid, PoincareBall]'),
        'c': (1.0, 'hyperbolic radius, set to None for trainable curvature'),
        'r': (2., 'fermi-dirac decoder parameter for lp'),
        't': (1., 'fermi-dirac decoder parameter for lp'),
        'pretrained-embeddings': (None, 'path to pretrained embeddings (.npy file) for Shallow node classification'),
        'pos-weight': (0, 'whether to upweight positive class in node classification tasks'),
        'num-layers': (2, 'number of hidden layers in encoder'),
        'bias': (1, 'whether to use bias (1) or not (0)'),
        'act': ('relu', 'which activation function to use (or None for no activation)'),
        'n-heads': (4, 'number of attention heads for graph attention networks, must be a divisor dim'),
        'alpha': (0.2, 'alpha for leakyrelu in graph attention networks'),
        'double-precision': ('0', 'whether to use double precision'),
        'use-att': (0, 'whether to use hyperbolic attention or not'),
        'local-agg': (0, 'whether to local tangent space aggregation or not')
    },
    'data_config': {
        'dataset': ('cora', 'which dataset to use'),
        'val-prop': (0.05, 'proportion of validation edges for link prediction'),
        'test-prop': (0.1, 'proportion of test edges for link prediction'),
        'use-feats': (1, 'whether to use node features or not'),
        'normalize-feats': (1, 'whether to normalize input node features'),
        'normalize-adj': (1, 'whether to row-normalize the adjacency matrix'),
        'split-seed': (1234, 'seed for data splits (train/test/val)'),
    },
    ##VGL config
    'VGL_dataconfig':{
        'n_channels': (128, 'number of EEG data channels'),
        'n_sections': (5, 'split number of a channel'),
        'n_resample': (30, 'resample number of a section'),
        'n_classes':(2, 'number of label classes'),
        'data_dir':("./data", "data path")
    },
    'VGL_training_config': {
        'VGL_lr': (0.05, 'learning rate'),
        #'VGL_cuda': (-1, 'which cuda device to use (-1 for cpu training)'),
        'VGL_epochs': (100, 'maximum number of epochs to train for'),
        'VGL_seed': (1234, 'seed for training'),
        'VGL_eval-freq': (1, 'how often to compute val metrics (in epochs)'),
        'VGL_save': (0, '1 to save model and logs and 0 otherwise'),
        'VGL_save_dir': ("./logs", 'path to save training logs and model weights (defaults to logs/task/date/run/)'),
        'VGL_epochs': (100, 'maximum number of epochs to train for'),
        'VGL_batch_size': (32, 'batch size'),
    },
    ##mocha config
    'mocha_modelconfig':{
        'device': ('cuda:0', 'training device'),
        #'feat_dim':(128, 'graph node num and feat dim')
        'mocha_feat_dim': (128, 'mocha graph feat dim'),
        'mocha_n_nodes': (128, 'mocha graph n nodes')
    },


}

for _, config_dict in config_args.items():
    parser = add_flags_from_config(parser, config_dict)