
from rdkit import Chem
from rdkit.Chem import Recap
from tqdm import tqdm
from collections import defaultdict
from utils.utils import sanitize_mol, sanitize_smiles, get_mol, get_smiles

def get_bridge_bonds(mol):
    bridge_bonds = []
    for bond in mol.GetBonds():
        u, v = bond.GetBeginAtom(), bond.GetEndAtom()
        # Check if both u and v have degree >= 2
        if u.GetDegree() >= 2 and v.GetDegree() >= 2:
            # Check if either u or v is part of a ring
            if u.IsInRing() and v.IsInRing():
                continue
            if u.IsInRing() or v.IsInRing():
                bridge_bonds.append(bond.GetIdx())
    return bridge_bonds

def break_bonds(mol, bond_indices):
    emol = Chem.EditableMol(mol)
    breaked_bonds = []
    # Remove bonds based on indices
    for idx in sorted(bond_indices, reverse=True):
        bond = mol.GetBondWithIdx(idx)
        emol.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        breaked_bonds.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
    
    # Get the fragmented molecule
    fragmented_mol = emol.GetMol()
    
    # Generate separate fragments
    atom_indices = Chem.GetMolFrags(fragmented_mol, sanitizeFrags=True)
    fragment_mols = Chem.GetMolFrags(fragmented_mol, asMols=True, sanitizeFrags=True)
    fragments = [get_smiles(sanitize_mol(x, False)) for x in fragment_mols]

    return fragments, atom_indices, breaked_bonds

def bridge(data):
    motif_dict = defaultdict(list)
    motif_id = {}
    mol = get_mol(data, False)
    bridge_bonds = get_bridge_bonds(mol)
    fragments, atom_list, bond_list = break_bonds(mol, bridge_bonds)
    for i, frag in enumerate(fragments):
        motif_dict[frag].append(list(atom_list[i]))
    return mol, fragments, atom_list, bond_list

def bridge_list(data_list):
    tf = []
    df = defaultdict(int)
    motif_list = []

    for i, data in enumerate(tqdm(data_list)):
        motif_dict = defaultdict(list)
        mol = get_mol(data, False)

        bridge_bonds = get_bridge_bonds(mol)
        fragments, atom_list, _ = break_bonds(mol, bridge_bonds)
        for i, frag in enumerate(fragments):
            motif_dict[frag].append(list(atom_list[i]))
        motif_list.append(motif_dict)
        mol_tf = {}
        for fragment in fragments:
            if fragment in mol_tf:
                mol_tf[fragment] += 1
            else:
                mol_tf[fragment] = 1
        tf.append(mol_tf)
        for fragment in mol_tf:
            df[fragment] += 1
    return motif_list, df

def do_bridge(data):
    mol = get_mol(data, False)
    num_atoms = mol.GetNumAtoms()
    bridge_bonds = get_bridge_bonds(mol)
    fragments, atom_list, bond_list = break_bonds(mol, bridge_bonds)
    return num_atoms