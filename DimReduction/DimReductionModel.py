import torch
import torch.nn as nn
import torch.nn.functional as F

class DimReductionModel(nn.Module):
    def __init__(self):
        super(DimReductionModel, self).__init__()
        return

class eighModel(DimReductionModel):
    def __init__(self):
        super(eighModel, self).__init__()
        return

    def forward(self, embeddings):
        embedding_transform = []
        for channel in range(len(embeddings)):
            channel_embedding_matrix = torch.flatten(embeddings[channel], start_dim=1)
            mean = torch.mean(channel_embedding_matrix, dim=0)  # 计算每个特征的均值
            centered_data = channel_embedding_matrix - mean  # 从每个样本减去相应的特征均值

            # Step 3: 计算协方差矩阵
            cov_matrix = torch.mm(centered_data.T, centered_data) / (centered_data.shape[0] - 1)

            # Step 4: 计算特征值和特征向量
            if torch.allclose(cov_matrix, cov_matrix.T):  # Check if the matrix is symmetric
                eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
            else:
                eigenvalues, eigenvectors = torch.linalg.eig(cov_matrix)
            # Step 5: 按照特征值排序，选择主成分
            # 将特征值从大到小排序，并返回排序后的索引
            sorted_indices = torch.argsort(eigenvalues, descending=True)

            # 按照排序后的索引重新排列特征向量和特征值
            sorted_eigenvalues = eigenvalues[sorted_indices]
            sorted_eigenvectors = eigenvectors[:, sorted_indices]

            # Step 6: 选择前 k 个主成分 (假设我们选择前 2 个主成分)
            k = 4
            top_k_eigenvectors = sorted_eigenvectors[:, :k]

            # Step 7: 投影到主成分空间，得到降维后的数据
            projected_data = torch.mm(centered_data, top_k_eigenvectors)
            embedding_transform.append(projected_data)

        embedding_transform = torch.stack(embedding_transform)
        return embedding_transform

class ISOmapModel(DimReductionModel):
    def __init__(self, n_neighbors=5, n_components=2):
        super(ISOmapModel, self).__init__()
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        return

    def construct_knn_graph(self, X):
        """ Construct k-nearest neighbors adjacency graph using sklearn """
        n_samples = X.shape[0]
        nbrs = self.NearestNeighbors(n_neighbors=self.n_neighbors, algorithm='auto').fit(X.cpu().numpy())
        distances, indices = nbrs.kneighbors(X.cpu().numpy())

        # Initialize adjacency matrix with infinity
        adj_matrix = torch.full((n_samples, n_samples), float('inf'))
        for i in range(n_samples):
            adj_matrix[i, indices[i]] = torch.tensor(distances[i])

        return adj_matrix

    def compute_shortest_paths(self, adj_matrix):
        """ Compute all-pairs shortest paths using Floyd-Warshall algorithm """
        n = adj_matrix.size(0)
        dist_matrix = adj_matrix.clone()

        for k in range(n):
            for i in range(n):
                for j in range(n):
                    dist_matrix[i, j] = min(dist_matrix[i, j], dist_matrix[i, k] + dist_matrix[k, j])

        return dist_matrix

    def classical_mds(self, dist_matrix):
        """ Perform Classical MDS using PyTorch SVD """
        # Centering matrix
        n = dist_matrix.shape[0]
        H = torch.eye(n) - torch.ones((n, n)) / n
        B = -0.5 * H @ (dist_matrix ** 2) @ H  # Double centering

        # Compute eigenvalues and eigenvectors using SVD
        U, S, V = torch.svd(B)
        S = torch.sqrt(S[:self.n_components])  # Take sqrt of eigenvalues
        U = U[:, :self.n_components]

        return U * S  # Compute low-dimensional representation

    def forward(self, X):
        """
        Forward pass for Isomap dimensionality reduction.

        X: Input tensor of shape (n_samples, n_features)
        Returns: Transformed tensor of shape (n_samples, n_components)
        """
        # Step 1: Compute distance matrix

        # Step 2: Construct k-NN adjacency graph
        adj_matrix = self.construct_knn_graph(X)

        # Step 3: Compute shortest paths
        shortest_paths = self.compute_shortest_paths(adj_matrix)

        # Step 4: Apply MDS to find low-dimensional embedding
        reduced_X = self.classical_mds(shortest_paths)

        return reduced_X