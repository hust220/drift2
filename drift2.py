import argparse
import os
import tempfile
import sys
import torch
from torch.utils.data import Dataset, DataLoader
from src import const
from src import utils
from src.datasets import combine_pocket_ligand_graphs, collate_batch_data
from src.graph_utils import (
    parse_pocket_from_pdb, parse_molecules_from_sdf, parse_molecules_from_smi,
    create_pocket_graph, create_ligand_graph
)
from src.lightning import Drift2
from tqdm import tqdm
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

class BatchPredictionDataset(Dataset):
    
    def __init__(self, pdb_content, ligand_path, device=None):
        self.pdb_content = pdb_content
        self.device = device
        self.ligand_path = ligand_path
        self.molecule_count = 0
        
        # Pre-parse pocket data
        pocket_pos, pocket_one_hot, residue_info = parse_pocket_from_pdb(pdb_content)
        if len(pocket_pos) == 0:
            raise ValueError("No valid residues found in PDB file")
        self.pocket_graph = create_pocket_graph(pocket_pos, residue_info, pocket_one_hot)
    
    def __len__(self):
        # Build index and count molecules on first call
        if self.molecule_count == 0:
            if self.ligand_path.lower().endswith('.sdf'):
                self._build_sdf_index()
                self.molecule_count = len(self._sdf_positions)
            elif self.ligand_path.lower().endswith('.smi'):
                self._build_smi_index()
                self.molecule_count = len(self._smi_positions)
            else:
                raise ValueError("Ligand file must be .sdf or .smi format")
        return self.molecule_count
    
    def __getitem__(self, idx):
        # Parse molecule on-demand
        if self.ligand_path.lower().endswith('.sdf'):
            mol_data = self._get_sdf_molecule(idx)
        elif self.ligand_path.lower().endswith('.smi'):
            mol_data = self._get_smi_molecule(idx)
        else:
            raise ValueError("Unsupported file format")
        
        # Create ligand graph
        ligand_graph = create_ligand_graph(mol_data['positions'], mol_data['one_hot'], mol_data['bonds'])
        
        # Combine graphs
        combined_graph = combine_pocket_ligand_graphs(self.pocket_graph, ligand_graph)
        
        # Convert to tensors
        data = {
            'one_hot': torch.tensor(combined_graph['one_hot'], dtype=const.TORCH_FLOAT, device=self.device),
            'edge_index': torch.tensor(combined_graph['edge_index'], dtype=torch.long, device=self.device),
            'edge_attr': torch.tensor(combined_graph['edge_attr'], dtype=const.TORCH_FLOAT, device=self.device),
            'node_mask': torch.tensor(combined_graph['node_mask'], dtype=const.TORCH_INT, device=self.device),
            'edge_mask': torch.tensor(combined_graph['edge_mask'], dtype=const.TORCH_INT, device=self.device),
            'protein_mask': torch.tensor(combined_graph['protein_mask'], dtype=const.TORCH_INT, device=self.device),
            'name': mol_data['name']
        }
        
        return data
    
    def _get_sdf_molecule(self, idx):
        """Get molecule at specific index from SDF file"""
        # For large files, we need to build an index of molecule positions
        if not hasattr(self, '_sdf_positions'):
            self._build_sdf_index()
        
        if idx >= len(self._sdf_positions):
            raise IndexError(f"Index {idx} out of range")
        
        start_pos, end_pos = self._sdf_positions[idx]
        with open(self.ligand_path, 'r') as f:
            f.seek(start_pos)
            mol_content = f.read(end_pos - start_pos)
        
        # Parse the molecule
        mol_data = next(parse_molecules_from_sdf(mol_content, name="DATABASE_ID"))
        return mol_data
    
    def _get_smi_molecule(self, idx):
        """Get molecule at specific index from SMILES file"""
        # For large files, we need to build an index of line positions
        if not hasattr(self, '_smi_positions'):
            self._build_smi_index()
        
        if idx >= len(self._smi_positions):
            raise IndexError(f"Index {idx} out of range")
        
        start_pos, end_pos = self._smi_positions[idx]
        with open(self.ligand_path, 'r') as f:
            f.seek(start_pos)
            line = f.read(end_pos - start_pos).strip()
        
        # Parse the molecule
        mol_data = next(parse_molecules_from_smi(line, generate_coords=False))
        return mol_data
    
    def _build_sdf_index(self):
        """Build index of SDF molecule positions"""
        self._sdf_positions = []
        
        with open(self.ligand_path, 'r') as f:
            pos = 0
            start_pos = 0
            for line in f:
                if line.strip() == '$$$$':
                    end_pos = pos + len(line)
                    self._sdf_positions.append((start_pos, end_pos))
                    start_pos = end_pos
                pos += len(line)
    
    def _build_smi_index(self):
        """Build index of SMILES line positions"""
        self._smi_positions = []
        
        with open(self.ligand_path, 'r') as f:
            pos = 0
            for line in f:
                if line.strip():
                    line_len = len(line)
                    self._smi_positions.append((pos, pos + line_len))
                pos += line_len

def predict_affinities_batch(receptor_path, ligand_path, model, device, output_file, batch=128, include_affinity=False):
    print("Reading receptor file...")
    with open(receptor_path, 'r') as f:
        pdb_content = f.read()
    
    # If ligand_path is not a file, treat it as a SMILES string and write a temporary .smi file
    needs_cleanup = False
    ligand_source = ligand_path
    if not os.path.isfile(ligand_path):
        print("Ligand argument is not a file; writing to a temporary .smi file...")
        tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.smi', delete=False)
        # Write as "SMILES<TAB>NAME" so that downstream parsing has an identifier
        tmp.write(f"{ligand_path}\tQUERY\n")
        tmp.flush()
        tmp.close()
        ligand_source = tmp.name
        needs_cleanup = True

    try:
        print(f"Creating dataset...")
        dataset = BatchPredictionDataset(pdb_content, ligand_source, device)
        dataloader = DataLoader(dataset, batch_size=batch, collate_fn=collate_batch_data, shuffle=False)
        
        print(f"Processing {len(dataset)} molecules in batches of {batch}...")
        
        processed_count = 0
        
        model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Processing batches"):
                affinity_scores = model.forward(batch)
                names = batch['name']
                for name, score in zip(names, affinity_scores):
                    score_val = score.item()
                    if include_affinity:
                        # Convert score to affinity in nM: affinity_nM = (10^(-score)) * 1e9
                        affinity_nm = (10 ** (-score_val)) * 1e9
                        if name:
                            output_file.write(f"{name}\t{score_val:.4f}\t{affinity_nm:.2f}\n")
                        else:
                            output_file.write(f"{score_val:.4f}\t{affinity_nm:.2f}\n")
                    else:
                        if name:
                            output_file.write(f"{name}\t{score_val:.4f}\n")
                        else:
                            output_file.write(f"{score_val:.4f}\n")
                output_file.flush()
                processed_count += len(names)
        print(f"Successfully processed {processed_count} molecules")
        return processed_count
    finally:
        if needs_cleanup:
            try:
                os.remove(ligand_source)
            except Exception:
                pass

def get_device(device_arg):
    """Get the appropriate device based on user input and availability"""
    if device_arg.lower() == 'auto':
        if torch.cuda.is_available():
            return torch.device("cuda"), "CUDA"
        elif torch.backends.mps.is_available():
            return torch.device("mps"), "MPS"
        else:
            return torch.device("cpu"), "CPU"
    elif device_arg.lower() == 'cuda':
        if torch.cuda.is_available():
            return torch.device("cuda"), "CUDA"
        else:
            print("Warning: CUDA requested but not available, falling back to CPU")
            return torch.device("cpu"), "CPU"
    elif device_arg.lower() == 'mps':
        if torch.backends.mps.is_available():
            return torch.device("mps"), "MPS"
        else:
            print("Warning: MPS requested but not available, falling back to CPU")
            return torch.device("cpu"), "CPU"
    elif device_arg.lower() == 'cpu':
        return torch.device("cpu"), "CPU"
    else:
        # Try to use the specified device directly
        try:
            device = torch.device(device_arg)
            return device, device_arg.upper()
        except:
            print(f"Warning: Device '{device_arg}' not recognized, falling back to CPU")
            return torch.device("cpu"), "CPU"

def main(args):
    print("Initializing Drift2 affinity prediction...")
    
    device, device_name = get_device(args.device)
    print(f"Using device: {device_name}")
    
    resolved_ckpt = utils.pick_latest([args.model])
    print(f"Using checkpoint: {resolved_ckpt}")

    print("Loading model...")
    drift2_model = Drift2.load_from_checkpoint(resolved_ckpt, map_location=device).eval().to(device)
    print("Model loaded successfully!")
    
    print(f"Processing {args.receptor} with {args.ligand}")
    if os.path.isfile(args.ligand):
        if not args.output:
            raise SystemExit("Error: output path is required when ligand is a file (.sdf/.smi)")
        print(f"Results will be saved to: {args.output}")
    else:
        if args.output:
            print(f"Results will be saved to: {args.output}")
        else:
            print("No output path provided for SMILES input; writing results to stdout")
    print(f"Batch size: {args.batch}")
    
    if os.path.isfile(args.ligand):
        with open(args.output, 'w') as output_file:
            processed_count = predict_affinities_batch(
                args.receptor, args.ligand, drift2_model, device, output_file, args.batch, args.include_affinity
            )
    else:
        # Write to provided file if given; otherwise stdout
        if args.output:
            with open(args.output, 'w') as output_file:
                processed_count = predict_affinities_batch(
                    args.receptor, args.ligand, drift2_model, device, output_file, args.batch, args.include_affinity
                )
        else:
            processed_count = predict_affinities_batch(
                args.receptor, args.ligand, drift2_model, device, sys.stdout, args.batch, args.include_affinity
            )
    
    print(f"Prediction completed! Processed {processed_count} molecules.")
    print(f"Results saved to: {args.output}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict pocket-ligand binding affinity using Drift2 model')
    parser.add_argument(
        'receptor', action='store', type=str, 
        help='Path to the receptor PDB file'
    )
    parser.add_argument(
        'ligand', action='store', type=str,
        help='Path to the ligand SDF or SMI file'
    )
    parser.add_argument(
        'output', nargs='?', default=None, type=str,
        help='Output file path. Required when ligand is a file; optional for SMILES (stdout if omitted)'
    )
    parser.add_argument(
        '--model', action='store', type=str, default='pdbbind_aff',
        help='Checkpoint path or glob/pattern; latest match will be used'
    )
    parser.add_argument(
        '--batch', action='store', type=int, default=128,
        help='Batch size for prediction (default: 128)'
    )
    parser.add_argument(
        '--device', action='store', type=str, default='auto',
        help='Device to use: auto, cuda, mps, cpu, or specific device (default: auto)'
    )
    parser.add_argument(
        '--include-affinity', action='store_true', default=False,
        help='Include predicted affinity in nM units in output (default: False)'
    )
    
    args = parser.parse_args()
    main(args)
