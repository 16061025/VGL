import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool

from RSA import RDMModel
from MochaGCN.models.base_models import NCModel


from hgcn.models.base_models import LPModel


class VGLModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_channels = args.n_channels
        self.n_sections = args.n_sections
        self.hgcn_module = [[] for i in range(self.n_channels)]
        for i in range(self.n_channels):
            for j in range(self.n_sections):
                self.hgcn_module[i].append(LPModel(args).to(args.device))
        self.RDM_module = RDMModel().to(args.device)
        args.feat_dim = args.mocha_feat_dim
        args.n_nodes = args.mocha_n_nodes
        self.MochaGCN_module = NCModel(args).to(args.device)
        return

    # def forward(self, feats, adjs):
    #     batch_size = len(feats)
    #     feats_tmp = feats
    #     adjs_tmp = adjs
    #     pred = torch.ones((batch_size, 1))
    #     for i in range(batch_size):
    #         feats = feats_tmp[i]
    #         adjs = adjs_tmp[i]
    #         all_channel_embeddings = [[] for i in range(self.n_channels)]
    #         for i in range(self.n_channels):
    #             for j in range(self.n_sections):
    #                 adj = adjs[i][j].to_sparse()
    #                 all_channel_embeddings[i].append(self.hgcn_module[i][j].encode(feats[i][j], adj))
    #         for i in range(self.n_channels):
    #             channel_embedding = torch.cat(all_channel_embeddings[i], 0)
    #             all_channel_embeddings[i] = channel_embedding
    #         all_channel_embeddings = torch.stack(all_channel_embeddings, dim=0)
    #         brain_graph = self.RDM_module(all_channel_embeddings)
    #
    #         Mocha_encode_embeddings = self.MochaGCN_module.encode(torch.eye(len(brain_graph)) , brain_graph)
    #         pred[i] = self.MochaGCN_module.decode(Mocha_encode_embeddings, brain_graph)
    #     return pred

    def forward(self, feats, adjs):
        device = feats.device
        batch_size = feats.size()[0]
        all_channel_embeddings = [[] for i in range(self.n_channels)]
        for i in range(self.n_channels):
            for j in range(self.n_sections):
                adj_seldimi = adjs.index_select(1, torch.tensor([i]).to(device))
                adj_seldimij = adj_seldimi.index_select(2, torch.tensor([j]).to(device))
                adj = torch.squeeze(adj_seldimij)
                adj = adj.to_sparse()

                feat_seldimi = feats.index_select(1, torch.tensor([i]).to(device))
                feat_seldimij = feat_seldimi.index_select(2, torch.tensor([j]).to(device))
                feat = torch.squeeze(feat_seldimij)

                all_channel_embeddings[i].append(self.hgcn_module[i][j].encode(feat, adj))
        for i in range(self.n_channels):
            channel_embedding = torch.cat(all_channel_embeddings[i], 1)
            all_channel_embeddings[i] = channel_embedding
        all_channel_embeddings = torch.stack(all_channel_embeddings, dim=1)
        brain_graph = self.RDM_module(all_channel_embeddings)

        one_hot_feat = torch.eye(brain_graph.size()[-1]).repeat(batch_size, 1).to(device)
        mocha_collect_brain_graph_list = list(torch.unbind(brain_graph))
        mocha_collect_brain_graph = torch.block_diag(*mocha_collect_brain_graph_list)


        Mocha_encode_embeddings = self.MochaGCN_module.encode(one_hot_feat , mocha_collect_brain_graph)
        decodeoutput = self.MochaGCN_module.decode(Mocha_encode_embeddings, brain_graph)
        batch = torch.arange(0, batch_size, dtype=torch.int64).repeat_interleave(brain_graph.size()[-1]).to(device)

        mean_pool_decodeoutput = global_mean_pool(decodeoutput, batch)
        pred = F.sigmoid(mean_pool_decodeoutput)
        return pred


def train_VGLModel(VGL_model, train_loader, loss_fn, optimizer, args):
    res = {}
    VGL_model.train()
    for batch, data in enumerate(train_loader):
        feats, adjs, y = data
        feats = feats.to(args.device)
        adjs = adjs.to(args.device)
        y = y.to(args.device)
        size = len(train_loader.dataset)
        pre = VGL_model(feats, adjs)

        loss = loss_fn(pre, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


    loss = loss.item()
    res["loss"] = loss
    return res

def test_VGLModel(VGL_model, test_loader, loss_fn, args):
    res = {}
    device = args.device
    size = len(test_loader.dataset)
    num_batches = len(test_loader)
    VGL_model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for batch, data in enumerate(test_loader):
            feats, adjs, y = data
            feats = feats.to(args.device)
            adjs = adjs.to(args.device)
            y = y.to(args.device)
            pred = VGL_model(feats, adjs)
            test_loss += loss_fn(pred, y).item()
            print("prediction res and label")
            print(pred.tolist())
            print(pred.argmax(1).tolist())
            print("true res and label")
            print(y.tolist())
            print(y.argmax(1).tolist())
            correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size

    res['correct'] = correct
    res['Angloss'] = test_loss
    return res

