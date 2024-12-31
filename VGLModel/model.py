import torch
import torch.nn as nn

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
                self.hgcn_module[i].append(LPModel(args))
        self.RDM_module = RDMModel()
        self.MochaGCN_module = NCModel(args)
        return

    def forward(self, feats, adjs):
        all_channel_embeddings = [[] for i in range(self.n_channels)]
        for i in range(self.n_channels):
            for j in range(self.n_sections):
                all_channel_embeddings[i][j] = self.hgcn_module[i][j].encode(feats[i][j], adjs[i][j])
        for i in range(self.n_channels):
            channel_embedding = torch.cat(all_channel_embeddings[i], 0)
            all_channel_embeddings[i] = channel_embedding

        brain_graph = self.RDM_module(all_channel_embeddings)

        Mocha_encode_embeddings = self.MochaGCN_module.encode(torch.eye(len(brain_graph)) , brain_graph)
        pred = self.MochaGCN_module.decode(Mocha_encode_embeddings, brain_graph)
        return pred


def train_VGLModel(VGL_model, train_loader, loss_fn, optimizer, args):

    VGL_model.train()
    for batch, (Xs, adjs, y) in enumerate(train_loader):
        size = len(train_loader.dataset)
        pre = VGL_model(Xs, adjs)
        loss = loss_fn(pre, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    return

def test_VGLModel(VGL_model, test_loader, loss_fn, args):
    device = args.device
    size = len(test_loader.dataset)
    num_batches = len(test_loader)
    VGL_model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            pred = VGL_model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

