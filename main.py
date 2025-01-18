import datetime
import logging
import os

from args import parser
from VGLModel.data_utils import load_VGL_dataset

import torch
from torch.utils.data import DataLoader
from VGLModel.model import VGLModel
from VGLModel.model import train_VGLModel, test_VGLModel


if __name__ == '__main__':
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

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
    VGL_train_data, VGL_test_data = load_VGL_dataset(args)
    print(f"test data len is {len(VGL_test_data)}")
    print(f"train data len is {len(VGL_train_data)}")
    feats, adjs, y = VGL_train_data[0]
    args.n_nodes, args.feat_dim = feats[0][0].shape

    train_dataloader = DataLoader(VGL_train_data, batch_size=args.VGL_batch_size)
    test_dataloader = DataLoader(VGL_test_data, batch_size=args.VGL_batch_size)

    args.device = 'cuda:' + str(args.cuda) if int(args.cuda) >= 0 else 'cpu'
    device = args.device

    print(f"device is {device}")
    model = VGLModel(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.VGL_lr)
    loss_fn = torch.nn.CrossEntropyLoss()

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
