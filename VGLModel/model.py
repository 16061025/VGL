import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool

from RSA import RDMModel
from MochaGCN.models.base_models import NCModel


from hgcn.models.base_models import LPModel

from MLP.MLP_model import dimreduction_MLP_model, prediction_MLP_model

class VGLModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_channels = args.n_channels
        self.n_sections = args.n_sections
        self.hgcn_module = nn.ModuleList([
            nn.ModuleList([
                LPModel(args).to(args.device) for _ in range(self.n_sections)
            ])
            for _ in range(self.n_channels)
        ])
        self.RDM_module = RDMModel().to(args.device)
        args.feat_dim = args.mocha_feat_dim
        args.n_nodes = args.mocha_n_nodes
        self.MochaGCN_module = NCModel(args).to(args.device)
        return

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

class VGLModel_shareHGCN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_channels = args.n_channels
        self.n_sections = args.n_sections
        self.hgcn_module = LPModel(args).to(args.device)
        self.RDM_module = RDMModel().to(args.device)
        args.feat_dim = args.mocha_feat_dim
        args.n_nodes = args.mocha_n_nodes
        self.MochaGCN_module = NCModel(args).to(args.device)
        return

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

                all_channel_embeddings[i].append(self.hgcn_module.encode(feat, adj))
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

class VGLModel_MLP_bonn(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.hgcn_module = LPModel(args).to(args.device)
        embeddingsdim = args.bonn_N_node * args.dim
        reductiondim = args.dim
        self.dimreduction_MLP = dimreduction_MLP_model(input_dim=embeddingsdim, output_dim=reductiondim).to(args.device)
        self.prediction_MLP = prediction_MLP_model(input_dim=reductiondim, output_dim=2).to(args.device)

        return

    def forward(self, feats, adjs):
        device = feats.device
        batch_size = feats.size()[0]
        embeddings = self.hgcn_module.encode(feats, adjs)

        dimreduction_embeddings = self.dimreduction_MLP(embeddings)
        pred = self.prediction_MLP(dimreduction_embeddings)
        return pred

class VGLModel_MLP(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_channels = args.n_channels
        self.n_sections = args.n_sections
        self.hgcn_module = LPModel(args).to(args.device)
        embeddingsdim = self.n_channels * self.n_sections * args.n_resample * args.dim
        reductiondim = args.dim
        self.dimreduction_MLP = dimreduction_MLP_model(input_dim=embeddingsdim, output_dim=reductiondim)
        self.prediction_MLP = prediction_MLP_model(input_dim=reductiondim, output_dim=2)

        return

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

                all_channel_embeddings[i].append(self.hgcn_module.encode(feat, adj))
        for i in range(self.n_channels):
            channel_embedding = torch.cat(all_channel_embeddings[i], 1)
            all_channel_embeddings[i] = channel_embedding
        all_channel_embeddings = torch.stack(all_channel_embeddings, dim=1)
        dimreduction_embeddings = self.dimreduction_MLP(all_channel_embeddings)
        pred = self.prediction_MLP(dimreduction_embeddings)
        return pred


def train_VGLModel(VGL_model, train_loader, loss_fn, optimizer, args):
    res = {}
    VGL_model.train()
    correct, size = 0, 0
    for batch, data in enumerate(train_loader):
        feats, adjs, y = data
        feats = feats.to(args.device)
        adjs = adjs.to(args.device)
        y = y.to(args.device)

        pred = VGL_model(feats, adjs)

        loss = loss_fn(pred, y)

        loss.backward()
        # loss.backward(retain_graph=True)
        # for name, param in VGL_model.named_parameters():
        #     if param.grad is not None:
        #         print(f"{name} is in the computation graph (grad={param.grad})")
        #     else:
        #         print(f"{name} is NOT in the computation graph")
        optimizer.step()
        optimizer.zero_grad()

        size += len(train_loader.dataset)
        correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()

        if batch==0:
            print("training...")
            print("prediction res and label")
            print(pred.tolist())
            print(pred.argmax(1).tolist())
            print("true res and label")
            print(y.tolist())
            print(y.argmax(1).tolist())

    correct /= size
    loss = loss.item()
    res["loss"] = loss
    res['correct'] = correct
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

