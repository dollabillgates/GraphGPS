from typing import List
import os
import os.path as osp
import torch
from torch_geometric.data import Dataset
import re

class Alphafold(Dataset):
    def __init__(self):
        super().__init__()
        self.root = '/content/drive/MyDrive/protein-DATA/sample-final'
        self.split_file = '/content/drive/MyDrive/protein-DATA/dataset-indices.pt'
        self.eigen_path = "/content/drive/MyDrive/protein-DATA/eigens_precomputed"

    @property
    def processed_file_names(self) -> List[str]:
        return sorted(
            [f for f in os.listdir(self.root) if f.endswith('.pt')],
            key=lambda x: int(re.findall(r'\d+', x)[0])
        )

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data_path = osp.join(self.root, self.processed_file_names[idx])
        data = torch.load(data_path)
        data.filename = osp.splitext(self.processed_file_names[idx])[0]

        self.load_eigen_data(data)

        return data

    def load_eigen_data(self, data):
        graph_id = data.filename
        evals_path = osp.join(self.eigen_path, f"{graph_id}_evals.pt")
        evects_path = osp.join(self.eigen_path, f"{graph_id}_evects.pt")

        if osp.exists(evals_path) and osp.exists(evects_path):
            print(f"Loading EigVals and EigVecs for graph {graph_id}")
            data.EigVals = torch.load(evals_path)
            data.EigVecs = torch.load(evects_path)

    def get_idx_split(self):
        split_dict = torch.load(self.split_file)
        return split_dict
