'''
Representation Similarity Analysis


'''

import torch
import torch.nn as nn
from scipy.stats import gamma
from DimReduction.DimReductionModel import eighModel


class RDMModel(nn.Module):
    def __init__(self, distance_matircs="Pearson"):
        super().__init__()
        self.distance_matircs = distance_matircs
        self.flatten = nn.Flatten()
        self.dimreduction = eighModel()
        return

    def calculate_similarity(self, embedding1, embedding2):
        return

    def rbf_dot(self, pattern1, pattern2, deg):
        size1 = pattern1.shape
        size2 = pattern2.shape

        G = torch.sum(pattern1 * pattern1, 1).reshape(size1[0], 1)
        H = torch.sum(pattern2 * pattern2, 1).reshape(size2[0], 1)

        Q = torch.tile(G, (1, size2[0]))
        R = torch.tile(H.T, (size1[0], 1))

        H = Q + R - 2 * torch.mm(pattern1, pattern2.T)

        H = torch.exp(-H / 2 / (deg ** 2))

        return H

    def hsic_corr(self, X, alph=0.5):
        hsic_distance_matrix = torch.eye(len(X))
        for i in range(len(X)):
            for j in range(i + 1, len(X)):
                corr_ij = self.hsic_gam_tensor(X[i], X[j], alph)
                hsic_distance_matrix[i][j] = corr_ij
                hsic_distance_matrix[j][i] = corr_ij
        return hsic_distance_matrix

    def hsic_gam_tensor(self, X, Y, alph=0.5):
        """
        X, Y are tensor vectors with row - sample, col - dim
        alph is the significance level
        auto choose median to be the kernel width
        """
        n = X.shape[0]

        # ----- width of X -----
        Xmed = X

        G = torch.sum(Xmed * Xmed, 1).reshape(n, 1)
        Q = torch.tile(G, (1, n))
        R = torch.tile(G.T, (n, 1))

        dists = Q + R - 2 * torch.mm(Xmed, Xmed.T)
        dists = dists - torch.tril(dists)
        dists = dists.reshape(n ** 2, 1)

        width_x = torch.sqrt(0.5 * torch.median(dists[dists > 0]))
        # ----- -----

        # ----- width of X -----
        Ymed = Y

        G = torch.sum(Ymed * Ymed, 1).reshape(n, 1)
        Q = torch.tile(G, (1, n))
        R = torch.tile(G.T, (n, 1))

        dists = Q + R - 2 * torch.mm(Ymed, Ymed.T)
        dists = dists - torch.tril(dists)
        dists = dists.reshape(n ** 2, 1)

        width_y = torch.sqrt(0.5 * torch.median(dists[dists > 0]))
        # ----- -----

        bone = torch.ones((n, 1))
        H = torch.eye(n) - torch.ones((n, n)) / n

        K = self.rbf_dot(X, X, width_x)
        L = self.rbf_dot(Y, Y, width_y)

        Kc = torch.mm(torch.mm(H, K), H)
        Lc = torch.mm(torch.mm(H, L), H)

        testStat = torch.sum(Kc.T * Lc) / n

        varHSIC = (Kc * Lc / 6) ** 2

        varHSIC = (torch.sum(varHSIC) - torch.trace(varHSIC)) / n / (n - 1)

        varHSIC = varHSIC * 72 * (n - 4) * (n - 5) / n / (n - 1) / (n - 2) / (n - 3)

        K = K - torch.diag(torch.diag(K))
        L = L - torch.diag(torch.diag(L))

        muX = torch.mm(torch.mm(bone.T, K), bone) / n / (n - 1)
        muY = torch.mm(torch.mm(bone.T, L), bone) / n / (n - 1)

        mHSIC = (1 + muX * muY - muX - muY) / n

        al = mHSIC ** 2 / varHSIC
        bet = varHSIC * n / mHSIC

        thresh = gamma.ppf(1 - alph, al, scale=bet)[0][0]

        return testStat - thresh

    # def forward(self, embeddings):
    #     if self.distance_matircs == "Pearson":
    #         if embeddings.dim() > 2:
    #             pearson_corr_matrix_batch = []
    #             batch_size = embeddings.size()[0]
    #             for i in range(batch_size):
    #                 var_embedding = embeddings[i]
    #                 corr_matrix = torch.corrcoef(self.flatten(var_embedding))
    #                 pearson_corr_matrix_batch.append(corr_matrix)
    #             pearson_corr_matrix = torch.stack(pearson_corr_matrix_batch)
    #         else:
    #             pearson_corr_matrix = torch.corrcoef(self.flatten(embeddings))
    #         return pearson_corr_matrix
    #     if self.distance_matircs == "Euclidean":
    #         RDM_dim = len(embeddings)
    #         eucl_distance_matrix = torch.Tensor(shape=(RDM_dim, RDM_dim))
    #         for i in range(0, RDM_dim):
    #             for j in range(0, RDM_dim):
    #                 eucl_distance_matrix[i][j] = sum(((embeddings[i] - embeddings[j])**2).flatten())
    #         return eucl_distance_matrix
    #     if self.distance_matircs == "HSIC":
    #         HSIC_distance_matrix_batch = []
    #         batch_size = embeddings.size()[0]
    #         for i in range(batch_size):
    #             var_embedding = embeddings[i]
    #             corr_matrix = self.hsic_corr(var_embedding)
    #
    #             HSIC_distance_matrix_batch.append(corr_matrix)
    #         HSIC_distance_matrix = torch.stack(HSIC_distance_matrix_batch)
    #         return HSIC_distance_matrix

    def forward(self, embeddings):
        '''
        dimensionality reduction
        :param embeddings:
        :return:
        '''
        if self.distance_matircs == "Pearson":
            if embeddings.dim() > 2:
                pearson_corr_matrix_batch = []
                batch_size = embeddings.size()[0]
                for i in range(batch_size):
                    var_embedding = embeddings[i]
                    #var_embedding = self.dimreduction(var_embedding)
                    corr_matrix = torch.corrcoef(self.flatten(var_embedding))
                    pearson_corr_matrix_batch.append(corr_matrix)
                pearson_corr_matrix = torch.stack(pearson_corr_matrix_batch)
            else:
                pearson_corr_matrix = torch.corrcoef(self.flatten(embeddings))
            return pearson_corr_matrix


