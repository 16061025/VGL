import torch
import torch.nn as nn

class RDM_Module(nn.Module):
    def __init__(self, distance_matircs="Pearson"):
        super().__init__()
        self.distance_matircs = distance_matircs
        return

    def calculate_similarity(self, embedding1, embedding2):
        return

    def forward(self, embeddings):
        if self.distance_matircs == "Pearson":
            pearson_corr_matrix = torch.corrcoef(embeddings)
            return pearson_corr_matrix
        if self.distance_matircs == "Euclidean":
            RDM_dim = len(embeddings)
            eucl_distance_matrix = torch.Tensor(shape=(RDM_dim, RDM_dim))
            for i in range(0, RDM_dim):
                for j in range(0, RDM_dim):
                    eucl_distance_matrix[i][j] = sum(((embeddings[i] - embeddings[j])**2).flatten())
            return eucl_distance_matrix
        #if self.distance_matircs == ""

    def calculate_RDM(self, embeddings):
        return self.forward(embeddings)
