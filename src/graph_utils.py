"""
Graph utility functions for molecular and protein data processing.
Contains common functions for one-hot encoding, parsing, and graph creation.
"""

import numpy as np
import io
from rdkit import Chem
from src import const
from src.pdb_utils import Structure


def atom_one_hot(atom):
    """Convert atom type to one-hot encoding."""
    n = const.N_ATOM_TYPES
    one_hot = np.zeros(n)
    if atom not in const.ATOM2IDX:
        atom = 'X'
    one_hot[const.ATOM2IDX[atom]] = 1
    return one_hot


def aa_one_hot(residue):
    """Convert amino acid residue to one-hot encoding."""
    n = const.N_RESIDUE_TYPES
    one_hot = np.zeros(n)
    if residue not in const.RESIDUE2IDX:
        residue = 'UNK'
    one_hot[const.RESIDUE2IDX[residue]] = 1
    return one_hot


def bond_one_hot(bond):
    """Convert bond type to one-hot encoding."""
    one_hot = [0 for i in range(const.N_RDBOND_TYPES)]
    
    bond_type = bond.GetBondType()
    if bond_type not in const.RDBOND2IDX:
        bond_type = Chem.rdchem.BondType.ZERO
    one_hot[const.RDBOND2IDX[bond_type]] = 1
        
    return one_hot


def parse_molecule_from_sdf(sdf_content):
    """Parse molecule from SDF content string"""
    supplier = Chem.SDMolSupplier()
    supplier.SetData(sdf_content)
    
    mol = next(supplier)
    if mol is None:
        return None, None, None
    
    atom_one_hots = []
    non_h_indices = []
    
    # Collect non-hydrogen atoms
    for idx, atom in enumerate(mol.GetAtoms()):
        if atom.GetSymbol() != 'H':
            atom_one_hots.append(atom_one_hot(atom.GetSymbol()))
            non_h_indices.append(idx)

    # Get positions for non-hydrogen atoms
    if mol.GetNumConformers() == 0:
        positions = np.zeros((len(non_h_indices), 3))
    else:
        all_positions = mol.GetConformer().GetPositions()
        positions = all_positions[non_h_indices]

    # Get bonds between non-hydrogen atoms
    bonds = []
    old_idx_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(non_h_indices)}
    
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        if i in old_idx_to_new and j in old_idx_to_new:
            one_hot = bond_one_hot(bond)
            new_i = old_idx_to_new[i]
            new_j = old_idx_to_new[j]
            bonds.append([new_i, new_j] + one_hot)
            bonds.append([new_j, new_i] + one_hot)

    return positions, np.array(atom_one_hots), np.array(bonds)


def parse_pocket_from_pdb(pdb_content):
    """Parse pocket from PDB content string"""
    pdb_io = io.StringIO(pdb_content)
    structure = Structure()
    structure.read(pdb_io)
    
    pocket_coords = []
    pocket_types = []
    residue_info = []  # Store (residue_id, chain_id) for backbone detection

    # Iterate through all models, chains, and residues
    for model in structure.get_models():
        for chain in model.chains:
            for residue in chain.get_residues():
                residue_name = residue.res_name
                residue_id = residue.res_id
                chain_id = residue.chain_id
                
                # Look for CA atom
                ca_atom = residue.get_atom('CA')
                if ca_atom is not None:
                    atom_coord = ca_atom.get_coord()
                    pocket_coords.append(atom_coord.tolist())
                    pocket_types.append(residue_name)
                    residue_info.append((residue_id, chain_id))

    pocket_one_hot = []
    for _type in pocket_types:
        pocket_one_hot.append(aa_one_hot(_type))
    pocket_one_hot = np.array(pocket_one_hot)

    return pocket_coords, pocket_one_hot, residue_info


def create_pocket_edges(pocket_coords, residue_info, distance_cutoff=6.0):
    """Create edges within pocket based on distance and backbone"""
    pocket_size = len(pocket_coords)
    pocket_coords = np.array(pocket_coords)
    
    edge_index = []
    edge_attr = []
    n_bond_feats = const.N_RDBOND_TYPES
    
    # Calculate all pairwise distances
    for i in range(pocket_size):
        for j in range(i + 1, pocket_size):
            coord_i = pocket_coords[i]
            coord_j = pocket_coords[j]
            distance = np.linalg.norm(coord_i - coord_j)
            
            # Check if it's a backbone connection (consecutive residues in same chain)
            res_id_i, chain_i = residue_info[i]
            res_id_j, chain_j = residue_info[j]
            is_backbone = (chain_i == chain_j and abs(res_id_i - res_id_j) == 1)
            
            # Add edge if within distance cutoff or backbone connection
            if distance <= distance_cutoff or is_backbone:
                edge_attr_val = [distance, int(is_backbone), 0, 0] + [0] * n_bond_feats
                
                edge_index.append([i, j])
                edge_attr.append(edge_attr_val)
                edge_index.append([j, i])
                edge_attr.append(edge_attr_val)
    
    return edge_index, edge_attr


def create_pocket_graph(pocket_coords, residue_info, pocket_one_hot):
    """Create pocket graph with internal edges only"""
    # Get pocket internal edges
    pocket_edges_idx, pocket_edges_attr = create_pocket_edges(pocket_coords, residue_info)
    
    return {
        'one_hot': pocket_one_hot,
        'edge_index': np.array(pocket_edges_idx),
        'edge_attr': np.array(pocket_edges_attr),
        'coords': np.array(pocket_coords)
    }


def create_ligand_graph(mol_pos, mol_one_hot, mol_bonds):
    """Create ligand graph with internal bonds only"""
    # Create molecule internal edges
    edge_index = []
    edge_attr = []
    n_bond_feats = const.N_RDBOND_TYPES
    
    for bond in mol_bonds:
        bond_i = bond[0]
        bond_j = bond[1]
        # Edge features: [distance=0, is_backbone=0, is_pocket_mol=0, is_mol_mol=1, bond_features...]
        edge_attr_val = [0.0, 0, 0, 1] + bond[2:].tolist()
        edge_index.append([bond_i, bond_j])
        edge_attr.append(edge_attr_val)
    
    return {
        'one_hot': mol_one_hot,
        'edge_index': np.array(edge_index),
        'edge_attr': np.array(edge_attr),
        'coords': mol_pos
    }
