'''
Representation Similarity Analysis


'''

import torch
import torch.nn as nn

class RDMModel(nn.Module):
    def __init__(self, distance_matircs="Pearson"):
        super().__init__()
        self.distance_matircs = distance_matircs
        self.flatten = nn.Flatten()
        return

    def calculate_similarity(self, embedding1, embedding2):
        return

    def forward(self, embeddings):
        if self.distance_matircs == "Pearson":
            if embeddings.dim() > 2:
                pearson_corr_matrix_batch = []
                batch_size = embeddings.size()[0]
                for i in range(batch_size):
                    var_embedding = embeddings[i]
                    corr_matrix = torch.corrcoef(self.flatten(var_embedding))
                    pearson_corr_matrix_batch.append(corr_matrix)
                pearson_corr_matrix = torch.stack(pearson_corr_matrix_batch)
            else:
                pearson_corr_matrix = torch.corrcoef(self.flatten(embeddings))
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
