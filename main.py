from args import parser
from VisibilityGraph.VisibilityGraph import construct_EEG_visibility_grapy
from hgcn.trainHGCN import train_HGCN
from loaddata import load_EEG_data
from utils.data_utils import process, mask_edges
from RSA import RDM_Module


def create_dateset_from_VG(EEG_visibility_graph_list):
    for EEG_visibility_graph in EEG_visibility_graph_list:
        adj, features = EEG_visibility_graph
        data = {'adj_train': adj, 'features': features}
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
    EEG_raw_data = load_EEG_data()

    ## select one user for demo
    EEG_data = EEG_raw_data[0]

    ##step 2 construct visibility graph
    ## no divided
    EEG_visibility_graph_list = construct_EEG_visibility_grapy(EEG_data)
    pass

    ## divided
    ##step 3 hyperbolic Graph convolution
    ## if independet GCN task
    data = create_dateset_from_VG(EEG_visibility_graph_list)
    HGCN_model = train_HGCN(data, args)
    ## else

    ##get brain region embedding
    HGCN_model.eval()
    brain_region_embbeding = HGCN_model.encode(data['features'], data['adj_train_norm'])

    # ##step 4 Brain Graph
    # calculate RDM
    # input embeddings N brain region * embedding dim
    # output N * N
    RDM_Model = RDM_Module()
    RDM = RDM_Model(brain_region_embbeding)

    # ## step 5 Graph Fusion
    #
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