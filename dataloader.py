import numpy as np
import torch
import pandas as pd
from torch_geometric.data import Data, DataLoader
from scipy.sparse import csr_matrix

class GraphDataset:
    def __init__(self, split='train'):

        if split not in ['train', 'val', 'test']:
            raise ValueError("split must be one of train/val/test")
        
        self.split = split

        # Load graph structure
        self.edges = np.loadtxt("data/public/A.txt", delimiter=",", dtype=int)
        self.node_graph_id = np.load("data/public/node_graph_id.npy")

        # Load graph IDs and labels for this split
        self.graph_ids = np.load(f"data/public/{self.split}_idx.npy")
        if self.split != 'test':
            self.graph_labels = pd.read_csv(f"data/public/{self.split}_labels.csv")
            self.label_map = dict(zip(self.graph_labels["id"], self.graph_labels["y_true"]))

        # Load and concatenate all 3 feature sets
        self.node_features = torch.cat([
            self._load_sparse("data/public/new_spacy_feature.npz"),    # 300
            self._load_sparse("data/public/new_bert_feature.npz"),     # 768
            self._load_sparse("data/public/new_profile_feature.npz"),  # 10
        ], dim=-1)  # → N × 1078

    def _load_sparse(self, path):
        f = np.load(path)
        sparse = csr_matrix(
            (f["data"], f["indices"], f["indptr"]),
            shape=tuple(f["shape"])
        )
        return torch.tensor(sparse.toarray(), dtype=torch.float)

    def build_graph(self, g_id):
        nodes = np.where(self.node_graph_id == g_id)[0]
        mask = np.isin(self.edges[:,0], nodes) & np.isin(self.edges[:,1], nodes)
        edge_index = self.edges[mask]

        node_map = {node: i for i, node in enumerate(nodes)}
        edge_index = np.array([[node_map[u], node_map[v]] for u, v in edge_index]).T
        edge_index = torch.tensor(edge_index, dtype=torch.long)

        x = self.node_features[nodes]
        if self.split != 'test':
            y = torch.tensor([self.label_map[g_id]], dtype=torch.float)
            data = Data(x=x, edge_index=edge_index, y=y)
        else:
            data = Data(x=x, edge_index=edge_index)

        data.graph_id = torch.tensor([g_id], dtype=torch.long)
        return data

    def get_loader(self, batch_size=128, shuffle=True):
        dataset = [self.build_graph(g) for g in self.graph_ids]
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)