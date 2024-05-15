
from utils.utils import to_smiles
from utils.bridge import bridge
import torch
from torch.utils.data import Dataset
from collections import defaultdict
from torch_geometric.data import Data

# Design a Tree class that store the information of data, like x, edge_index, and SMILES.
class Tree:
    def __init__(self, data, data_name, add_H = False):
        self.smiles = to_smiles(data, True, data_name, add_H=add_H)
        self.y = data.y
    
    def transform(self):
        # Transform the SMILES into a motif tree structure
        # print(f"smiles: {self.smiles}")
        mol, fragments, atom_list, bond_list = bridge(self.smiles)
        self.mol = mol
        self.fragments = fragments
        self.atom_list = atom_list
        self.bond_list = bond_list

class TreeDataset(Dataset):
    def __init__(self, trees):
        self.trees = trees

    def __len__(self):
        return len(self.trees)

    def __getitem__(self, idx):
        tree = self.trees[idx]
        # Process the tree data as needed, e.g., converting to tensor
        idx = torch.tensor([idx], dtype=torch.long)
        x = tree.x
        edge_index = tree.edge_index
        return x, edge_index, idx