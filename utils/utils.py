# Description: This file contains utility functions for data processing and data cleaning.

import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import rdchem
from rdkit.Chem import rdmolops
from rdkit import RDLogger
from typing import Any
from utils.mapping_conf import ATOM, EDGE
from tqdm import tqdm

def clean_dataset(dataset, data_name, add_H = False):
    cleaned_data_0 = []
    cleaned_data_1 = []
    id_0 = []
    id_1 = []
    for i, data in enumerate(tqdm(dataset)):
        smiles = to_smiles(data, True, data_name, add_H=add_H)
        # smiles = sanitize_smiles(smiles)
        if smiles is not None:
            mol = sanitize_mol(get_mol(smiles), addH=add_H)
            new_data = to_tudataset(mol, data_name, data.y.item())
            new_smiles = to_smiles(new_data, True, data_name, add_H=add_H)
            if new_smiles == smiles:
                if data.y.item() == 0:
                    cleaned_data_0.append(new_data)
                    id_0.append(i)
                else:
                    cleaned_data_1.append(new_data)
                    id_1.append(i)
    return cleaned_data_0, cleaned_data_1, (id_0, id_1)

def kekulize_mol(mol):
    Chem.Kekulize(mol)

def get_mol(smiles, addH=False):
    RDLogger.DisableLog('rdApp.*')  
    mol = Chem.MolFromSmiles(smiles)
    if addH == True:
        mol = Chem.AddHs(mol)
    Chem.Kekulize(mol, clearAromaticFlags=True) # Add clearAromaticFlags to avoid error
    return mol

def get_smiles(mol):
    RDLogger.DisableLog('rdApp.*') 
    smiles = Chem.MolToSmiles(mol)
    return smiles

def sanitize_mol(mol, addH=False):
    try:
        mol = get_mol(get_smiles(mol), addH=addH)
    except:
        return None
    return mol
    
def sanitize_smiles(smiles, addH=False):
    try:
        mol = get_mol(smiles, addH=addH)
        smiles = get_smiles(mol)

    except:
        return None
    return smiles

# Function to find potential bonding sites
def find_bonding_sites(mol):
    bonding_sites = []
    for atom in mol.GetAtoms():
        # Check if the atom has free valence
        if atom.GetImplicitValence() > 0:
            bonding_sites.append(atom.GetIdx())
    return bonding_sites

def check_bond_feasibility(mol1, mol2, site1, site2):
    # Temporary combining of fragments for checking
    combined_mol = Chem.CombineMols(mol1, mol2)
    editable_mol = Chem.EditableMol(combined_mol)
    
    # Add a bond between chosen sites
    editable_mol.AddBond(site1, mol1.GetNumAtoms() + site2, order=Chem.rdchem.BondType.SINGLE)
    
    # Create the new molecule
    new_mol = editable_mol.GetMol()
    
    # Try sanitizing the molecule; if it fails, the bond is not feasible
    try:
        sanitize_mol(new_mol)
    except:
        return False
    return True

def can_assemble(mol, fragment):
    # Get bonding sites for each fragment
    sites1 = find_bonding_sites(mol)
    sites2 = find_bonding_sites(fragment)
    site_pair = []
    for site1 in sites1:
        for site2 in sites2:
            if check_bond_feasibility(mol, fragment, site1, site2):
                site_pair.append((site1, site2))
    if len(site_pair) > 0:
        return site_pair
    return None

def to_smiles(data: 'torch_geometric.data.Data',
              kekulize: bool = True, data_name: str = 'MUTAG', add_H = False) -> Any:
    """Converts a :class:`torch_geometric.data.Data` instance to a SMILES
    string.

    Args:
        data (torch_geometric.data.Data): The molecular graph.
        kekulize (bool, optional): If set to :obj:`True`, converts aromatic
            bonds to single/double bonds. (default: :obj:`False`)
        data_name: The name of dataset
    """

    mol = Chem.RWMol()

    for i in range(data.num_nodes):
        # Some dataset does not have 
        if data_name in ["COX2", "BZR", "NCI1"]:
            atom = rdchem.Atom(torch.argmax(data.x[i]).item()+1)
        else:
            atom = rdchem.Atom(ATOM[data_name][torch.argmax(data.x[i]).item()])
        mol.AddAtom(atom)
    edges = [tuple(i) for i in data.edge_index.t().tolist()]
    visited = set()
    deleted = []
    
    for i in range(len(edges)):
        src, dst = edges[i]
        if tuple(sorted(edges[i])) in visited:
            continue
        if "edge_attr" in data.keys():
            bond_type = EDGE[data_name][torch.argmax(data.edge_attr[i]).item()]
            if bond_type == None:
                deleted.append(tuple(edges[i]))
            else:
                mol.AddBond(src, dst, bond_type)
        else:
            mol.AddBond(src, dst)

        visited.add(tuple(sorted(edges[i])))

    mol = mol.GetMol()

    mol = sanitize_mol(mol, add_H)
    if mol is None:
        return None

    # Chem.AssignStereochemistry(mol)

    return sanitize_smiles(get_smiles(mol), add_H)

def to_tudataset(mol, data_name, label=None):
    if mol == None:
        return None
    if mol.GetNumAtoms() == 0 and mol.GetNumBonds() == 0:
        return None
    rdmolops.AssignStereochemistry(mol)
    # if addH == True:
    #     mol = Chem.AddHs(mol)
    # Extract atom-level features
    atom_features = []
    swapped_feature_map = {value: key for key, value in ATOM[data_name].items()}
    for atom in mol.GetAtoms():
        atom_features.append(swapped_feature_map[atom.GetAtomicNum()])
    

    # Extract bond-level features
    bond_features = []
    swapped_edge_feature_map = {value: key for key, value in EDGE[data_name].items()}
    for bond in mol.GetBonds():
        bond_type = bond.GetBondTypeAsDouble()
        if bond_type == 1.0:
            bond_feat = swapped_edge_feature_map[rdchem.BondType.SINGLE]
        elif bond_type == 1.5:
            bond_feat = swapped_edge_feature_map[rdchem.BondType.AROMATIC]
        elif bond_type == 2.0:
            bond_feat = swapped_edge_feature_map[rdchem.BondType.DOUBLE]
        elif bond_type == 3.0:
            bond_feat = swapped_edge_feature_map[rdchem.BondType.TRIPLE]
        else:
            bond_type = swapped_edge_feature_map[None]

        bond_features.append(bond_feat)
        bond_features.append(bond_feat)
    
    atom_features = torch.tensor(atom_features, dtype=torch.long)
    x = F.one_hot(atom_features, num_classes=len(swapped_feature_map)).float()  # Node feature matrix
    edge_index = []
    for bond in mol.GetBonds():
        edge_index.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        edge_index.append([bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # Edge connectivity
    if mol.GetNumBonds() == 0:
        edge_index = torch.tensor([[], []], dtype=torch.long)
    # if mol.GetNumBonds() == 0:
        # edge_index.fill_([])
    bond_features = torch.tensor(bond_features, dtype=torch.long)  # Edge feature matrixd
    edge_attr = F.one_hot(bond_features, num_classes=len(swapped_edge_feature_map))
    if not label == None:
        y = torch.tensor([label])
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    else:
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data