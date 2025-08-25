#%%

import argparse
import os
import numpy as np
import torch
from rdkit import Chem
import io
from src import const
from src.pocket_ligand_dataset import combine_pocket_ligand_graphs
from src.graph_utils import (
    parse_pocket_from_pdb, parse_molecule_from_sdf, 
    create_pocket_graph, create_ligand_graph
)
from src.lightning import Drift2
from src.utils import set_deterministic
from src.db_utils import db_connection
from src.pdb_utils import Structure
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')



def _prepare_model_input(pdb_content, sdf_content, device):
    """Prepare model input for affinity prediction.
    
    Args:
        pdb_content: string content of PDB file
        sdf_content: string content of SDF file
        device: torch device
    
    Returns:
        dict: Combined graph data ready for model input
    """
    # Parse pocket
    pocket_pos, pocket_one_hot, residue_info = parse_pocket_from_pdb(pdb_content)
    if len(pocket_pos) == 0:
        raise ValueError("No valid residues found in PDB file")
    
    # Parse molecule
    mol_pos, mol_one_hot, mol_bonds = parse_molecule_from_sdf(sdf_content)
    if mol_pos is None:
        raise ValueError("Failed to parse molecule from SDF file")
    
    # Create separate pocket and ligand graphs
    pocket_graph = create_pocket_graph(pocket_pos, residue_info, pocket_one_hot)
    ligand_graph = create_ligand_graph(mol_pos, mol_one_hot, mol_bonds)
    
    # Combine graphs
    combined_graph = combine_pocket_ligand_graphs(pocket_graph, ligand_graph)
    
    # Convert to tensors and add batch dimension
    data = {
        'one_hot': torch.tensor(combined_graph['one_hot'], dtype=const.TORCH_FLOAT, device=device).unsqueeze(0),
        'edge_index': torch.tensor(combined_graph['edge_index'], dtype=torch.long, device=device).unsqueeze(0),
        'edge_attr': torch.tensor(combined_graph['edge_attr'], dtype=const.TORCH_FLOAT, device=device).unsqueeze(0),
        'node_mask': torch.tensor(combined_graph['node_mask'], dtype=const.TORCH_INT, device=device).unsqueeze(0),
        'edge_mask': torch.tensor(combined_graph['edge_mask'], dtype=const.TORCH_INT, device=device).unsqueeze(0),
        'protein_mask': torch.tensor(combined_graph['protein_mask'], dtype=const.TORCH_INT, device=device).unsqueeze(0)
    }
    
    return data

def _predict_affinity(data, model):
    """Predict pocket-ligand affinity using Drift2 model.
    
    Args:
        data: Combined graph data
        model: Drift2 model
    
    Returns:
        float: Affinity score
    """
    # Get model prediction
    with torch.no_grad():
        affinity_score = model.forward(data)  # shape: (batch,) -> (1,)
        # Remove batch dimension
        affinity_score = affinity_score.squeeze().item()  # scalar

    return affinity_score

def predict_affinity(receptor_path, ligand_path, model, device):
    """Predict pocket-ligand affinity using Drift2 model.
    
    Args:
        receptor_path: path to receptor PDB file
        ligand_path: path to ligand SDF file
        model: Drift2 model
        device: torch device
    
    Returns:
        float: Affinity score (higher score indicates better binding)
    """
    # Read file contents as text
    with open(receptor_path, 'r') as f:
        pdb_content = f.read()
    with open(ligand_path, 'r') as f:
        sdf_content = f.read()
    
    # Run prediction
    data = _prepare_model_input(pdb_content, sdf_content, device)
    affinity_score = _predict_affinity(data, model)

    return affinity_score





def main(args):
    # Set random seed if provided
    if args.random_seed is not None:
        set_deterministic(args.random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device)
    drift2_model = Drift2.load_from_checkpoint(args.model, map_location=device).eval().to(device)
    
    affinity_score = predict_affinity(args.receptor, args.ligand, drift2_model, device)
    
    print(f"Predicted affinity score: {affinity_score:.4f}")
    print(f"Higher scores indicate stronger predicted binding affinity")
    
    # Optionally save result to output file
    if hasattr(args, 'output') and args.output:
        with open(args.output, 'w') as f:
            f.write(f"Receptor: {args.receptor}\n")
            f.write(f"Ligand: {args.ligand}\n")
            f.write(f"Predicted affinity score: {affinity_score:.4f}\n")
        print(f"Results saved to: {args.output}")
    
    return affinity_score

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict pocket-ligand binding affinity using Drift2 model')
    parser.add_argument(
        'receptor', action='store', type=str, 
        help='Path to the receptor PDB file'
    )
    parser.add_argument(
        'ligand', action='store', type=str,
        help='Path to the ligand SDF file'
    )
    parser.add_argument(
        '--output', action='store', type=str, required=False,
        help='Path to the output text file (optional)'
    )
    parser.add_argument(
        '--model', action='store', type=str, default='./models/drift2_pdbbind_latest.ckpt',
        help='Path to the Drift2 model checkpoint'
    )
    parser.add_argument(
        '--random_seed', action='store', type=int, required=False, default=None,
        help='Random seed'
    )
    
    args = parser.parse_args()
    main(args)
