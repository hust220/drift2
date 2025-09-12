import os
import numpy as np
import pickle
import torch
import time
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import const
from src.db_utils import db_connection

def convert_binding_value_to_pk(binding_value, binding_unit):
    """Convert binding value to pK (-log10(Kd in M))"""
    if binding_value is None or binding_unit is None:
        return None
    
    # Convert to molar (M) first
    unit_lower = binding_unit.lower()
    if 'um' in unit_lower or 'μm' in unit_lower:
        kd_molar = binding_value * 1e-6  # uM to M
    elif 'nm' in unit_lower:
        kd_molar = binding_value * 1e-9  # nM to M
    elif 'mm' in unit_lower:
        kd_molar = binding_value * 1e-3  # mM to M
    elif 'm' in unit_lower and 'u' not in unit_lower and 'n' not in unit_lower:
        kd_molar = binding_value  # M to M
    else:
        return None  # Unknown unit
    
    # Convert to pK
    if kd_molar > 0:
        return -torch.log10(torch.tensor(kd_molar))
    else:
        return None

# Global data attributes configuration
DATA_LIST_ATTRS = ['pdb_id_i', 'pdb_id_j', 'name']
DATA_ATTRS_TO_PAD = ['one_hot', 'edge_index', 'edge_attr', 'node_mask', 'edge_mask', 'protein_mask']
DATA_ATTRS_TO_ADD_LAST_DIM = ['node_mask', 'edge_mask', 'protein_mask']

def combine_pocket_ligand_graphs(pocket_graph, ligand_graph):
    """Combine pocket and ligand graphs into a single graph with cross-edges"""
    pocket_size = len(pocket_graph['coords'])
    ligand_size = len(ligand_graph['coords'])
    
    # Combine node features
    mol_node_features = ligand_graph['one_hot'].shape[1]
    pocket_node_features = pocket_graph['one_hot'].shape[1]
    
    # Pad features to same dimension
    pocket_one_hot_padded = np.concatenate([
        pocket_graph['one_hot'], 
        np.zeros((pocket_size, mol_node_features))
    ], axis=-1)
    
    ligand_one_hot_padded = np.concatenate([
        np.zeros((ligand_size, pocket_node_features)), 
        ligand_graph['one_hot']
    ], axis=-1)
    
    combined_one_hot = np.concatenate([pocket_one_hot_padded, ligand_one_hot_padded], axis=0)
    
    # Combine edges
    combined_edge_index = []
    combined_edge_attr = []
    
    # Add pocket internal edges
    for edge_idx, edge_attr in zip(pocket_graph['edge_index'], pocket_graph['edge_attr']):
        combined_edge_index.append(edge_idx)
        combined_edge_attr.append(edge_attr)
    
    # Add ligand internal edges (shift indices)
    for edge_idx, edge_attr in zip(ligand_graph['edge_index'], ligand_graph['edge_attr']):
        shifted_edge = [edge_idx[0] + pocket_size, edge_idx[1] + pocket_size]
        combined_edge_index.append(shifted_edge)
        combined_edge_attr.append(edge_attr)
    
    # Add pocket-ligand cross edges (fully connected)
    n_bond_feats = const.N_RDBOND_TYPES
    for i in range(pocket_size):
        for j in range(ligand_size):
            ligand_idx = pocket_size + j
            # Edge features: [distance=0, is_backbone=0, is_pocket_mol=1, is_mol_mol=0, bond_features...]
            edge_attr_val = [0.0, 0, 1, 0] + [0] * n_bond_feats
            
            # Bidirectional edges
            combined_edge_index.append([i, ligand_idx])
            combined_edge_attr.append(edge_attr_val)
            combined_edge_index.append([ligand_idx, i])
            combined_edge_attr.append(edge_attr_val)
    
    # Create masks
    total_size = pocket_size + ligand_size
    protein_mask = np.zeros(total_size)
    protein_mask[:pocket_size] = 1
    node_mask = np.ones(total_size)
    edge_mask = np.ones(len(combined_edge_index))
    
    return {
        'one_hot': combined_one_hot,
        'edge_index': np.array(combined_edge_index),
        'edge_attr': np.array(combined_edge_attr),
        'node_mask': node_mask,
        'edge_mask': edge_mask,
        'protein_mask': protein_mask
    }

class PocketLigandDataset(Dataset):
    """Dataset for pocket-ligand interactions with dynamic graph combination"""

    def __init__(self, device=None, split='train', table_name='pdbbind_dataset'):
        self.device = device
        self.split = split
        self.table_name = table_name
        self.cache = {}  # Instance-level cache

        try:
            with db_connection() as conn:
                start_time = time.time()
                print(f'Loading dataset IDs from {table_name} table...')
                cursor = conn.cursor()                    
                cursor.execute(f"""
                    SELECT id, pdb_id
                    FROM {table_name}
                    WHERE split = %s
                    ORDER BY id
                """, (split,))
                results = cursor.fetchall()
                self.ids = [row[0] for row in results]
                self.pdb_ids = [row[1] for row in results]
                cursor.close()
                print(f'Loaded dataset IDs from {table_name}, Time: ', time.time() - start_time, 's')
            
            self.valid_indices = np.arange(len(self.ids))
            print(f"Found {len(self.valid_indices)} valid entries for split '{split}'")
        except Exception as e:
            print(f"Error loading dataset from database: {str(e)}")
            raise e

    def __len__(self):
        return len(self.valid_indices)

    def _create_tensors(self, data):
        """Helper method to create tensors from data dictionary."""
        result = {
            'pdb_id_i': data['pdb_id_i'],
            'pdb_id_j': data['pdb_id_j'],
            'one_hot': torch.tensor(data['one_hot'], dtype=const.TORCH_FLOAT, device=self.device),
            'edge_index': torch.tensor(data['edge_index'], dtype=torch.long, device=self.device),
            'edge_attr': torch.tensor(data['edge_attr'], dtype=const.TORCH_FLOAT, device=self.device),
            'node_mask': torch.tensor(data['node_mask'], dtype=const.TORCH_INT, device=self.device),
            'edge_mask': torch.tensor(data['edge_mask'], dtype=const.TORCH_INT, device=self.device),
            'protein_mask': torch.tensor(data['protein_mask'], dtype=const.TORCH_INT, device=self.device)
        }
        
        # Add affinity if available
        if 'affinity' in data and data['affinity'] is not None:
            if isinstance(data['affinity'], torch.Tensor):
                result['affinity'] = data['affinity'].to(device=self.device)
            else:
                result['affinity'] = torch.tensor(data['affinity'], dtype=const.TORCH_FLOAT, device=self.device)
        
        return result

    def _load_graph_data(self, item_id):
        """Load pocket and ligand graph data from database"""
        if item_id in self.cache:
            return self.cache[item_id]
        
        with db_connection() as conn:
            cursor = conn.cursor()
            # First get the pdb_id from pdbbind_dataset table
            cursor.execute(f"""
                SELECT pdb_id,
                       pocket_one_hot, pocket_edge_index, pocket_edge_attr, pocket_coords,
                       ligand_one_hot, ligand_edge_index, ligand_edge_attr, ligand_coords
                FROM {self.table_name}
                WHERE id = %s
            """, (item_id,))
            sample = cursor.fetchone()
            
            if sample is None:
                cursor.close()
                raise ValueError(f"Sample with id {item_id} not found in database")
            
            pdb_id = sample[0]
            
            # Then get binding affinity data from pdbbind table
            cursor.execute("""
                SELECT binding_value, binding_unit
                FROM pdbbind
                WHERE pdb_id = %s
            """, (pdb_id,))
            affinity_data = cursor.fetchone()
            cursor.close()
            
            # Convert binding affinity to pK directly
            if affinity_data:
                binding_value, binding_unit = affinity_data
                pk_value = convert_binding_value_to_pk(binding_value, binding_unit)
            else:
                binding_value, binding_unit = None, None
                pk_value = None
        
        # Parse pocket graph
        pocket_graph = {
            'one_hot': pickle.loads(sample[1]),
            'edge_index': pickle.loads(sample[2]),
            'edge_attr': pickle.loads(sample[3]),
            'coords': pickle.loads(sample[4])
        }
        
        # Parse ligand graph
        ligand_graph = {
            'one_hot': pickle.loads(sample[5]),
            'edge_index': pickle.loads(sample[6]),
            'edge_attr': pickle.loads(sample[7]),
            'coords': pickle.loads(sample[8])
        }
        
        # Cache the data
        self.cache[item_id] = {
            'pdb_id': pdb_id,
            'pocket_graph': pocket_graph,
            'ligand_graph': ligand_graph,
            'pk_value': pk_value
        }
        
        return self.cache[item_id]

    def __getitem__(self, item):
        # Map the requested index to the actual index in the dataset
        actual_idx = self.valid_indices[item]
        item_id_i = self.ids[actual_idx]
        pdb_id_i = self.pdb_ids[actual_idx]
        
        # Randomly select another ligand (j) from the dataset
        random_idx = np.random.randint(0, len(self.ids))
        while random_idx == actual_idx:  # Ensure different ligand
            random_idx = np.random.randint(0, len(self.ids))
        
        item_id_j = self.ids[random_idx]
        pdb_id_j = self.pdb_ids[random_idx]
        
        # Load graph data for both entries
        data_i = self._load_graph_data(item_id_i)
        data_j = self._load_graph_data(item_id_j)
        
        # Combine pocket_i with ligand_i (positive pair)
        combined_graph_ii = combine_pocket_ligand_graphs(
            data_i['pocket_graph'], 
            data_i['ligand_graph']
        )
        combined_graph_ii.update({
            'pdb_id_i': pdb_id_i,
            'pdb_id_j': pdb_id_i,  # Same ligand
            'affinity': data_i['pk_value']
        })
        
        # Combine pocket_i with ligand_j (negative pair)
        combined_graph_ij = combine_pocket_ligand_graphs(
            data_i['pocket_graph'], 
            data_j['ligand_graph']
        )
        combined_graph_ij.update({
            'pdb_id_i': pdb_id_i,
            'pdb_id_j': pdb_id_j,  # Different ligand
            'affinity': None  # No affinity for negative pairs
        })
        
        return {
            'positive': self._create_tensors(combined_graph_ii),
            'negative': self._create_tensors(combined_graph_ij)
        }

def collate_batch_data(batch_data, data_list_attrs=DATA_LIST_ATTRS, data_attrs_to_pad=DATA_ATTRS_TO_PAD, data_attrs_to_add_last_dim=DATA_ATTRS_TO_ADD_LAST_DIM):
    """Base collate function for batch data"""
    out = {}
    # Collect the list attributes
    for data in batch_data:
        for key, value in data.items():
            if key in data_list_attrs or key in data_attrs_to_pad:
                out.setdefault(key, []).append(value)

    # Pad the tensors
    for key, value in out.items():
        if key in data_attrs_to_pad:
            out[key] = torch.nn.utils.rnn.pad_sequence(value, batch_first=True, padding_value=0)
            continue

    # Add last dimension to the tensor
    for key in data_attrs_to_add_last_dim:
        if key in out.keys():
            out[key] = out[key][:, :, None]
    
    return out

def collate(batch):
    """Collate function for PocketLigandDataset"""
    positive_batch = []
    negative_batch = []
    
    # Separate positive and negative samples
    for sample in batch:
        positive_batch.append(sample['positive'])
        negative_batch.append(sample['negative'])
    
    return {
        'positive': collate_batch_data(positive_batch),
        'negative': collate_batch_data(negative_batch)
    }

def get_pocket_ligand_dataloader(dataset, batch_size, shuffle=False):
    """Get DataLoader for PocketLigandDataset"""
    return DataLoader(dataset, batch_size, collate_fn=collate, shuffle=shuffle)

def test_dataset(table_name='pdbbind_dataset'):
    """Test function to verify dataset functionality"""
    try:
        print(f"Testing PocketLigandDataset with table: {table_name}...")
        
        # Test dataset creation
        dataset = PocketLigandDataset(split='train', table_name=table_name)
        print(f"Dataset size: {len(dataset)}")
        
        if len(dataset) > 0:
            # Test single item
            sample = dataset[0]
            print("Sample keys:", list(sample.keys()))
            print("Sample shapes:")
            for key, value in sample.items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, torch.Tensor):
                            print(f"    {sub_key}: {sub_value.shape}")
                        else:
                            print(f"    {sub_key}: {type(sub_value)} - {sub_value}")
                else:
                    print(f"  {key}: {type(value)} - {value}")
            
            # Test dataloader
            dataloader = get_pocket_ligand_dataloader(dataset, batch_size=2, shuffle=False)
            batch = next(iter(dataloader))
            print("\nBatch keys:", list(batch.keys()))
            print("Batch shapes:")
            for key, value in batch.items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, torch.Tensor):
                            print(f"    {sub_key}: {sub_value.shape}")
                        else:
                            print(f"    {sub_key}: {type(sub_value)} - length {len(sub_value)}")
                else:
                    print(f"  {key}: {type(value)} - length {len(value)}")
        
        print("Dataset test completed successfully!")
        
    except Exception as e:
        print(f"Error in dataset test: {str(e)}")
        raise e

if __name__ == "__main__":
    test_dataset()