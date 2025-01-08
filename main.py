from args import parser
from VisibilityGraph.VisibilityGraph import construct_EEG_visibility_graph
from VisibilityGraph.VisibilityGraph import construct_EEG_visibility_graph_single_process
from hgcn.trainHGCN import train_HGCN
from VGLModel.data_utils import load_EEG_data, construct_VGL_dataset, VGL_collate_fn
from utils.data_utils import process, mask_edges
from RSA import RDMModel

from MochaGCN.trainMochaGCN import train_MochaGCN
from MochaGCN.data import BGdata2MGCNdataloader
import torch
from torch.utils.data import DataLoader
from VGLModel.model import VGLModel
from VGLModel.model import train_VGLModel
def create_HGCN_data_from_VG(EEG_visibility_graph_list):
    '''


    :param EEG_visibility_graph_list:
    :return: data that format satisfy HGCN requirement
    '''
    for EEG_visibility_graph in EEG_visibility_graph_list:
        adj, features, label = EEG_visibility_graph
        data = {'adj_train': adj, 'features': features, 'label': label}
        adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = mask_edges(
            adj, args.val_prop, args.test_prop, args.split_seed
        )
        data['adj_train'] = adj_train
        data['train_edges'], data['train_edges_false'] = train_edges, train_edges_false
        data['val_edges'], data['val_edges_false'] = val_edges, val_edges_false
        data['test_edges'], data['test_edges_false'] = test_edges, test_edges_false
        data['adj_train_norm'], data['features'] = process(
            data['adj_train'], data['features'], args.normalize_adj, args.normalize_feats
        )
    return data



if __name__ == '__main__':
    args = parser.parse_args()
    ##step 1 load eeg data
    ## N matrixs
    EEG_raw_data = load_EEG_data(args)

    ## select one user for demo
    EEG_data = [EEG_raw_data[0], EEG_raw_data[1]]
    EEG_data = EEG_raw_data

    ##step 2 construct visibility graph
    EEG_visibility_graph_list = construct_EEG_visibility_graph_single_process(EEG_data)
    VGL_train_data, VGL_test_data = construct_VGL_dataset(EEG_visibility_graph_list)
    feats, adjs, y = VGL_train_data[0]
    args.n_nodes, args.feat_dim = feats[0][0].shape


    batch_size = 2


    # train_dataloader = DataLoader(VGL_train_data, batch_size=batch_size, collate_fn=VGL_collate_fn)
    # test_dataloader = DataLoader(VGL_test_data, batch_size=batch_size, collate_fn=VGL_collate_fn)

    train_dataloader = DataLoader(VGL_train_data, batch_size=batch_size)
    test_dataloader = DataLoader(VGL_test_data, batch_size=batch_size)


    model = VGLModel(args)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    train_VGLModel(model, train_dataloader, loss_fn, optimizer, args)





    ##step 3 hyperbolic Graph convolution
    ## independet GCN LP task
    HGCN_data= create_HGCN_data_from_VG(EEG_visibility_graph_list)
    HGCN_model = train_HGCN(HGCN_data, args)

    ##get brain region embedding
    HGCN_model.eval()
    all_patients_brain_region_embbeding = HGCN_model.encode(HGCN_data['features'], HGCN_data['adj_train_norm'])

    # ##step 4 Brain Graph
    # calculate RDM
    brain_graph_data = {}
    RDM_model = RDMModel()
    RDM_model.eval()
    for brain_region_embbeding in all_patients_brain_region_embbeding:
        brain_graph = RDM_model(brain_region_embbeding)
        brain_graph_data[patientid] = brain_graph

    MochaGCN_train_loader,  MochaGCN_test_loader= BGdata2MGCNdataloader(brain_graph_data)
    # ## step 5 Graph Fusion
    train_MochaGCN(MochaGCN_train_loader,  MochaGCN_test_loader, args)






    #
    # ##step 1 load data
    # ## N matrixs
    # MRI_data = load_MRI_data()
    #
    # ##step 2 construct visibility graph
    # MRI_visibility_graph = construct_MRI_visibility_grapy(MRI_data)
    #
    # ##step 3 hyperbolic Graph convolution
    # HGCN_model = train_HGCN()
    # MRI_features = Hyperbolic_graph_Convolution(MRI_visibilitY_grapy)
    #
    # ##step 4 Brain Graph
    # MRI_brain_graph =
    # ## step 5 Graph Fusion