import os
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch_geometric.data import Data, Batch
import random
from sklearn.preprocessing import OneHotEncoder

class BrainData:
    def __init__(self, Config):
        self.ad_fmri_matrixz_dir = Config.ad_fmri_matrixz_dir
        self.ad_dti_dir = Config.ad_dti_dir
        self.nc_fcn_matrixz_dir = Config.nc_fcn_matrixz_dir
        self.nc_scn_dir = Config.nc_scn_dir
        self.ad_data = {}
        self.nc_data = {}
        self.load_all_data()
        self.get_all_data()

    def load_frmi_matrices(self, directory):
        matrixz_data = {}
        for filename in os.listdir(directory):
            if filename.endswith(".txt"):
                patient_id = os.path.splitext(filename)[0][1:]  # Remove 'z' prefix
                file_path = os.path.join(directory, filename)
                matrixz = np.loadtxt(file_path)
                if np.isnan(matrixz).any():
                    print(f"Warning: NaN values found in fMRI matrix for patient {patient_id} in file {filename}")
                matrixz_data[patient_id] = matrixz
                print(f"Loaded matrixz for patient {patient_id} from {filename}")
        return matrixz_data

    def load_dti_matrices(self, directory, matrix_type="FA"):
        dti_data = {}
        matrix_type_map = {
            "FA": "_Matrix_FA_AAL_Contract_90_2MM_90.txt",
            "FN": "_Matrix_FN_AAL_Contract_90_2MM_90.txt",
            "Length": "_Matrix_Length_AAL_Contract_90_2MM_90.txt"
        }

        if matrix_type not in matrix_type_map:
            raise ValueError(f"Invalid matrix type: {matrix_type}. Choose from 'FA', 'FN', 'Length'.")

        matrix_suffix = matrix_type_map[matrix_type]

        for patient_folder in os.listdir(directory):
            patient_path = os.path.join(directory, patient_folder, "Network", "Deterministic")
            if os.path.isdir(patient_path):
                for filename in os.listdir(patient_path):
                    if filename.endswith(matrix_suffix):
                        patient_id = patient_folder
                        file_path = os.path.join(patient_path, filename)
                        matrix = np.loadtxt(file_path)
                        if np.isnan(matrix).any():
                            print(f"Warning: NaN values found in {matrix_type} matrix for patient {patient_id} in file {filename}")
                        dti_data[patient_id] = matrix
                        print(f"Loaded {matrix_type} matrix for patient {patient_id} from {filename}")
                        break  # Only load one matrix per patient
        return dti_data

    def load_ad_data(self, limit=-1, method='top_k', fusion_mode=True):
        if not self.ad_fmri_matrixz_dir or not self.ad_dti_dir:
            raise ValueError("AD FMRI matrixz, time series, and/or DTI directory is not set.")

        fmri_matrixz_data = self.load_frmi_matrices(self.ad_fmri_matrixz_dir)
        dti_data = self.load_dti_matrices(self.ad_dti_dir)

        ad_data = {}
        count = 0
        for patient_id in fmri_matrixz_data:
            if limit != -1 and count >= limit:
                break
            if patient_id in dti_data:
                dti_adj = dti_data[patient_id]  # Assuming DTI data for adjacency matrix
                n_nodes = dti_adj.shape[0]

                # Generate a diagonal matrix with ones along the diagonal
                modal2_adj = np.eye(n_nodes)

                # Apply sparsification method to the adjacency matrices
                fmri_adj_origin = fmri_matrixz_data[patient_id]
                if method == 'threshold':
                    fmri_adj = threshold_adj(fmri_adj_origin)
                elif method == 'top_k':
                    fmri_adj = top_k_adj(fmri_adj_origin)
                else:
                    fmri_adj = fmri_adj_origin

                dti_adj_origin = dti_data[patient_id]
                if method == 'threshold':
                    dti_adj = threshold_adj(dti_adj_origin)
                elif method == 'top_k':
                    dti_adj = top_k_adj(dti_adj_origin)
                else:
                    dti_adj = dti_adj_origin

                if fusion_mode:
                    # beta1 = np.ones([])
                    # beta2 = np.ones([])
                    # thetas = np.exp(-beta1) + np.exp(-beta2)
                    # theta1 = np.exp(-beta1) / thetas
                    # theta2 = np.exp(-beta2) / thetas
                    # fusion_mode_adj = (
                    #         np.eye(modal2_adj.shape[0])
                    #         + theta1 * fmri_adj_origin
                    #         + theta2 * dti_adj_origin
                    # )
                    fusion_mode_adj = (fmri_adj + dti_adj) / 2
                    if method == 'threshold':
                        fusion_mode_adj = threshold_adj(fusion_mode_adj)
                    elif method == 'top_k':
                        fusion_mode_adj = top_k_adj(fusion_mode_adj)

                    ad_data[patient_id] = {
                        'modal1': {
                            'x': modal2_adj,  # Placeholder for adjacency matrix or one-hot encoding
                            'adj': fmri_adj
                        },
                        'modal2': {
                            'x': modal2_adj,  # Placeholder for adjacency matrix or one-hot encoding
                            'adj': dti_adj  # Use the generated diagonal matrix for modal2
                        },
                        'modal3': {
                            'x': modal2_adj,  # Placeholder for adjacency matrix or one-hot encoding
                            'adj': fusion_mode_adj  # Use the generated diagonal matrix for modal2
                        },
                        'label': 'AD'
                    }
                else:
                    ad_data[patient_id] = {
                        'modal1': {
                            'x': modal2_adj,  # Placeholder for adjacency matrix or one-hot encoding
                            'adj': fmri_adj
                        },
                        'modal2': {
                            'x': modal2_adj,  # Placeholder for adjacency matrix or one-hot encoding
                            'adj': dti_adj  # Use the generated diagonal matrix for modal2
                        },
                        'label': 'AD'
                    }
                count += 1
                print(f"Combined AD data for patient {patient_id}")
            else:
                print(f"Missing data for patient {patient_id}")

        return ad_data

    def load_nc_data(self, limit=-1, method='top_k', fusion_mode=True):
        if not self.nc_fcn_matrixz_dir or not self.nc_scn_dir:
            raise ValueError("NC FCN matrixz, time series, and/or SCN directory is not set.")

        fcn_matrixz_data = self.load_frmi_matrices(self.nc_fcn_matrixz_dir)
        scn_data = self.load_dti_matrices(self.nc_scn_dir)  # Assuming SCN data for modal2

        nc_data = {}
        count = 0
        for patient_id in fcn_matrixz_data:
            if limit != -1 and count >= limit:
                break
            if patient_id in scn_data:
                scn_adj = scn_data[patient_id]  # Assuming SCN data for adjacency matrix
                n_nodes = scn_adj.shape[0]

                # Generate a diagonal matrix with ones along the diagonal
                modal2_adj = np.eye(n_nodes)

                # Apply sparsification method to the adjacency matrices
                fcn_adj_origin = fcn_matrixz_data[patient_id]
                if method == 'threshold':
                    fcn_adj = threshold_adj(fcn_adj_origin)
                elif method == 'top_k':
                    fcn_adj = top_k_adj(fcn_adj_origin)
                else:
                    fcn_adj = fcn_adj_origin

                scn_adj_origin = scn_data[patient_id]
                if method == 'threshold':
                    scn_adj = threshold_adj(scn_adj_origin)
                elif method == 'top_k':
                    scn_adj = top_k_adj(scn_adj_origin)
                else:
                    scn_adj = scn_adj_origin

                if fusion_mode:
                    # beta1 = np.ones([])
                    # beta2 = np.ones([])
                    # thetas = np.exp(-beta1) + np.exp(-beta2)
                    # theta1 = np.exp(-beta1) / thetas
                    # theta2 = np.exp(-beta2) / thetas
                    # fusion_mode_adj = (
                    #         np.eye(modal2_adj.shape[0])
                    #         + theta1 * fcn_adj_origin
                    #         + theta2 * scn_adj_origin
                    # )
                    fusion_mode_adj = (fcn_adj + scn_adj) / 2
                    if method == 'threshold':
                        fusion_mode_adj = threshold_adj(fusion_mode_adj)
                    elif method == 'top_k':
                        fusion_mode_adj = top_k_adj(fusion_mode_adj)
                    nc_data[patient_id] = {
                        'modal1': {
                            'x': modal2_adj,  # Placeholder for adjacency matrix or one-hot encoding
                            'adj': fcn_adj
                        },
                        'modal2': {
                            'x': modal2_adj,  # Placeholder for adjacency matrix or one-hot encoding
                            'adj': scn_adj  # Use the generated diagonal matrix for modal2
                        },
                        'modal3': {
                            'x': modal2_adj,  # Placeholder for adjacency matrix or one-hot encoding
                            'adj': fusion_mode_adj  # Use the generated diagonal matrix for modal2
                        },
                        'label': 'NC'
                    }
                else:
                    nc_data[patient_id] = {
                        'modal1': {
                            'x': modal2_adj,  # Placeholder for adjacency matrix or one-hot encoding
                            'adj': fcn_adj
                        },
                        'modal2': {
                            'x': modal2_adj,  # Placeholder for adjacency matrix or one-hot encoding
                            'adj': scn_adj  # Use the generated diagonal matrix for modal2
                        },
                        'label': 'NC'
                    }
                count += 1
                print(f"Combined NC data for patient {patient_id}")
            else:
                print(f"Missing data for patient {patient_id}")

        return nc_data

    def train_test_group_split(self, n_splits, seed):
        """Split data into train and test sets."""
        # Convert data dictionary to lists
        patient_ids = list(self.data.keys())
        labels = [self.data[pid]['label'] for pid in patient_ids]
        # Create group numbers
        groups = np.arange(len(patient_ids))

        # Split the data into training and testing sets
        train_indices, test_indices = train_test_split(groups, test_size=1/n_splits, random_state=seed)

        train_data = {patient_ids[i]: self.data[patient_ids[i]] for i in train_indices}
        test_data = {patient_ids[i]: self.data[patient_ids[i]] for i in test_indices}

        return train_data, test_data

    def get_ad_data(self):
        return self.ad_data

    def get_nc_data(self):
        return self.nc_data

    def load_all_data(self):
        if not all([self.ad_fmri_matrixz_dir, self.ad_dti_dir,
                    self.nc_fcn_matrixz_dir, self.nc_scn_dir]):
            raise ValueError("One or more directories are not set.")

        self.ad_data = self.load_ad_data()
        self.nc_data = self.load_nc_data()

    def get_all_data(self):
        d = {}
        d.update(self.ad_data)
        d.update(self.nc_data)
        self.data = d


def threshold_adj(adj, threshold=0.5):
        """Retain edges with weights above the given threshold."""
        adj[adj < threshold] = 0
        return adj

def top_k_adj(adj, k=80):
        """Retain only top k edges for each node."""
        for i in range(adj.shape[0]):
            row = adj[i]
            top_k_indices = row.argsort()[-k:]
            new_row = np.zeros_like(row)
            new_row[top_k_indices] = row[top_k_indices]
            adj[i] = new_row
        return adj

def convert_to_pyg_data(data_dict):
    data_modal1 = []
    data_modal2 = []
    data_modal3 = []

    for patient_id, modalities in data_dict.items():
        # Process modal1
        x_modal1 = torch.tensor(modalities['modal1']['x'], dtype=torch.float)
        adj_modal1 = torch.tensor(modalities['modal1']['adj'], dtype=torch.float)
        y = torch.tensor([1 if modalities['label'] == 'AD' else 0], dtype=torch.long)  # Assuming binary classification

        edge_index_modal1 = adj_modal1.nonzero(as_tuple=False).t().contiguous()
        edge_attr_modal1 = adj_modal1[edge_index_modal1[0], edge_index_modal1[1]]

        pyg_data_modal1 = Data(x=x_modal1, edge_index=edge_index_modal1, edge_attr=edge_attr_modal1, dense_adj=adj_modal1, y=y)
        data_modal1.append(pyg_data_modal1)

        # Process modal2
        x_modal2 = torch.tensor(modalities['modal2']['x'], dtype=torch.float)
        adj_modal2 = torch.tensor(modalities['modal2']['adj'], dtype=torch.float)

        edge_index_modal2 = adj_modal2.nonzero(as_tuple=False).t().contiguous()
        edge_attr_modal2 = adj_modal2[edge_index_modal2[0], edge_index_modal2[1]]

        pyg_data_modal2 = Data(x=x_modal2, edge_index=edge_index_modal2, edge_attr=edge_attr_modal2, dense_adj=adj_modal2, y=y)
        data_modal2.append(pyg_data_modal2)

        # Check if modal3 exists and process it
        if 'modal3' in modalities:
            x_modal3 = torch.tensor(modalities['modal3']['x'], dtype=torch.float)
            adj_modal3 = torch.tensor(modalities['modal3']['adj'], dtype=torch.float)

            edge_index_modal3 = adj_modal3.nonzero(as_tuple=False).t().contiguous()
            edge_attr_modal3 = adj_modal3[edge_index_modal3[0], edge_index_modal3[1]]

            pyg_data_modal3 = Data(x=x_modal3, edge_index=edge_index_modal3, edge_attr=edge_attr_modal3,
                                   dense_adj=adj_modal3, y=y)
            data_modal3.append(pyg_data_modal3)


    return data_modal1, data_modal2, data_modal3



def get_positive_negative_samples(batch):
    labels = batch.y.tolist()
    label_to_indices = {label: [] for label in set(labels)}
    for index, label in enumerate(labels):
        label_to_indices[label].append(index)

    positive_samples = []
    negative_samples = []

    for index, label in enumerate(labels):
        positive_index = random.choice(label_to_indices[label])

        # Check if there are any labels different from the current label
        if len(label_to_indices) > 1:
            negative_label = random.choice([l for l in label_to_indices if l != label])
            negative_index = random.choice(label_to_indices[negative_label])
        else:
            # Generate a random feature close to zero for negative sample
            negative_index = None

        positive_samples.append(positive_index)
        negative_samples.append(negative_index)

    return positive_samples, negative_samples


def construct_positive_negative_representations(embeddings, positive_indices, negative_indices):
    positive_embeddings = embeddings[positive_indices]

    negative_embeddings = []
    for i, neg_index in enumerate(negative_indices):
        if neg_index is None:
            # Generate a random vector close to zero
            neg_embedding = torch.zeros_like(embeddings[i]).normal_(mean=0, std=1e-6)
        else:
            neg_embedding = embeddings[neg_index]
        negative_embeddings.append(neg_embedding)

    negative_embeddings = torch.stack(negative_embeddings)

    return positive_embeddings, negative_embeddings


class MultiModalDataLoader:
    def __init__(self, dataset_modal1, dataset_modal2, dataset_modal3, batch_size, shuffle=True, device='cpu'):
        self.dataset_modal1 = dataset_modal1
        self.dataset_modal2 = dataset_modal2
        self.dataset_modal3 = dataset_modal3
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device  # Add device parameter

    def __iter__(self):
        indices = list(range(len(self.dataset_modal1)))
        if self.shuffle:
            np.random.shuffle(indices)

        for start_idx in range(0, len(self.dataset_modal1), self.batch_size):
            end_idx = start_idx + self.batch_size
            batch_indices = indices[start_idx:end_idx]

            batch_modal1 = [self.dataset_modal1[i] for i in batch_indices]
            batch_modal2 = [self.dataset_modal2[i] for i in batch_indices]
            batch_modal3 = [self.dataset_modal3[i] for i in batch_indices]

            batch_modal1 = self._batch_data_list(batch_modal1)
            batch_modal2 = self._batch_data_list(batch_modal2)
            batch_modal3 = self._batch_data_list(batch_modal3)

            # Move batches to the specified device
            batch_modal1 = batch_modal1.to(self.device)
            batch_modal2 = batch_modal2.to(self.device)
            batch_modal3 = batch_modal3.to(self.device)

            yield batch_modal1, batch_modal2, batch_modal3

    def __len__(self):
        return len(self.dataset_modal1) // self.batch_size

    def _batch_data_list(self, data_list):
        batch = Batch.from_data_list(data_list)
        dense_adjs = [data.dense_adj for data in data_list]

        # Initialize an empty list to collect individual dense adjacency matrices
        batch_adj_list = []

        for adj in dense_adjs:
            batch_adj_list.append(adj)

        # Stack the adjacency matrices into a batch, keeping individual sizes
        batch_adj = torch.block_diag(*batch_adj_list)

        batch.dense_adj = batch_adj
        return batch


class SingleModalDataLoader:
    def __init__(self, dataset_modal, batch_size, shuffle=True, device='cpu'):
        self.dataset_modal = dataset_modal
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device  # Add device parameter

    def __iter__(self):
        indices = list(range(len(self.dataset_modal)))
        if self.shuffle:
            np.random.shuffle(indices)

        for start_idx in range(0, len(self.dataset_modal), self.batch_size):
            end_idx = start_idx + self.batch_size
            batch_indices = indices[start_idx:end_idx]

            batch_modal = [self.dataset_modal[i] for i in batch_indices]
            batch_modal = self._batch_data_list(batch_modal)
            # Move batches to the specified device
            batch_modal = batch_modal.to(self.device)


            yield batch_modal

    def __len__(self):
        return len(self.dataset_modal) // self.batch_size

    def _batch_data_list(self, data_list):
        batch = Batch.from_data_list(data_list)
        dense_adjs = [data.dense_adj for data in data_list]

        # Initialize an empty list to collect individual dense adjacency matrices
        batch_adj_list = []

        for adj in dense_adjs:
            batch_adj_list.append(adj)

        # Stack the adjacency matrices into a batch, keeping individual sizes
        batch_adj = torch.block_diag(*batch_adj_list)

        batch.dense_adj = batch_adj
        return batch

def BGdata2MGCNdataloader(brain_graph_data, n_splits, seed, batch_size, shuffle=True):
    '''
    transfer brain graph data into MochaGCN data loader

    :param brain_graph_data:
    :param n_splits:
    :param seed:
    :param batch_size:
    :param shuffle:
    :return: MochaGCN train test data loader
    '''



    """Split data into train and test sets."""
    # Convert data dictionary to lists
    patient_ids = list(brain_graph_data.keys())
    labels = [brain_graph_data[pid]['label'] for pid in patient_ids]
    # Create group numbers
    groups = np.arange(len(patient_ids))

    # Split the data into training and testing sets
    train_indices, test_indices = train_test_split(groups, test_size=1/n_splits, random_state=seed)

    train_data = {patient_ids[i]: brain_graph_data[patient_ids[i]] for i in train_indices}
    test_data = {patient_ids[i]: brain_graph_data[patient_ids[i]] for i in test_indices}

    def convert2pyd_data_single_model(data_dict):

        data_modal = []
        for patient_id, modalities in data_dict.items():
            # Process modal1
            x_modal1 = torch.tensor(modalities['modal1']['x'], dtype=torch.float)
            adj_modal1 = torch.tensor(modalities['modal1']['adj'], dtype=torch.float)
            y = torch.tensor([1 if modalities['label'] == 'AD' else 0],
                             dtype=torch.long)  # Assuming binary classification

            edge_index_modal = adj_modal1.nonzero(as_tuple=False).t().contiguous()
            edge_attr_modal = adj_modal1[edge_index_modal[0], edge_index_modal[1]]

            pyg_data_modal = Data(x=x_modal1, edge_index=edge_index_modal, edge_attr=edge_attr_modal,
                                   dense_adj=adj_modal1, y=y)
            data_modal.append(pyg_data_modal)
        return data_modal

    train_data_modal = convert2pyd_data_single_model(train_data)
    test_data_modal = convert2pyd_data_single_model(test_data)


    train_loader = SingleModalDataLoader(train_data_modal, batch_size=batch_size, shuffle=shuffle)
    test_loader = SingleModalDataLoader(test_data_modal, batch_size=batch_size, shuffle=shuffle)

    return train_loader, test_loader