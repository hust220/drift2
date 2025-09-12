import os
import sys
import numpy as np
import pickle
import torch
from multiprocessing import Pool
from rdkit import Chem
from tqdm import tqdm
import io

sys.path.append('../..')
from src.db_utils import db_connection
from src import const
from src.pdb_utils import Structure
from src.graph_utils import (
    atom_one_hot, aa_one_hot, bond_one_hot, parse_molecule_from_sdf, 
    parse_pocket_from_pdb, create_pocket_edges, create_pocket_graph, create_ligand_graph
)
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

def process_single_pdbbind_item(row):
    """Process a single PDBbind item"""
    try:
        pdb_id = row[0]
        pocket_pdb = row[1]
        ligand_sdf = row[2]
        
        if not pocket_pdb or not ligand_sdf:
            return None
            
        # Parse molecule
        mol_pos, mol_one_hot, mol_bonds = parse_molecule_from_sdf(ligand_sdf)
        if mol_pos is None:
            return None
            
        # Parse pocket
        pocket_pos, pocket_one_hot, residue_info = parse_pocket_from_pdb(pocket_pdb)
        if len(pocket_pos) == 0:
            return None
            
        pocket_size = len(pocket_pos)
        mol_size = len(mol_pos)
        
        if pocket_size > 1000 or mol_size == 0:
            return None
            
        # Create separate pocket and ligand graphs
        pocket_graph = create_pocket_graph(pocket_pos, residue_info, pocket_one_hot)
        ligand_graph = create_ligand_graph(mol_pos, mol_one_hot, mol_bonds)
        
        return {
            'pdb_id': pdb_id,
            'pocket_graph': pocket_graph,
            'ligand_graph': ligand_graph
        }
        
    except Exception as e:
        print(f"Error processing {row[0] if row else 'unknown'}: {str(e)}")
        return None

def process_chunk(chunk_ids):
    """Process a chunk of PDBbind data IDs and save directly to database"""
    processed_count = 0
    
    try:
        # Fetch data for this chunk of IDs
        with db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT pdb_id, pocket_pdb, ligand_sdf
                FROM pdbbind 
                WHERE id = ANY(%s)
            """, (chunk_ids,))
            chunk_data = cursor.fetchall()
            cursor.close()
        
        # Process each row and collect results
        processed_data = []
        for row in chunk_data:
            result = process_single_pdbbind_item(row)
            if result is not None:
                pocket_graph = result['pocket_graph']
                ligand_graph = result['ligand_graph']
                
                # Convert to bytes for database storage
                processed_data.append((
                    result['pdb_id'],
                    # Pocket graph data
                    pickle.dumps(np.array(pocket_graph['one_hot'], dtype=np.float32)),
                    pickle.dumps(np.array(pocket_graph['edge_index'], dtype=np.int64)),
                    pickle.dumps(np.array(pocket_graph['edge_attr'], dtype=np.float32)),
                    pickle.dumps(np.array(pocket_graph['coords'], dtype=np.float32)),
                    # Ligand graph data
                    pickle.dumps(np.array(ligand_graph['one_hot'], dtype=np.float32)),
                    pickle.dumps(np.array(ligand_graph['edge_index'], dtype=np.int64)),
                    pickle.dumps(np.array(ligand_graph['edge_attr'], dtype=np.float32)),
                    pickle.dumps(np.array(ligand_graph['coords'], dtype=np.float32))
                ))
        
        # Save processed data directly to database
        if processed_data:
            with db_connection() as conn:
                cursor = conn.cursor()
                cursor.executemany("""
                    INSERT INTO pdbbind_dataset (
                        pdb_id, 
                        pocket_one_hot, pocket_edge_index, pocket_edge_attr, pocket_coords,
                        ligand_one_hot, ligand_edge_index, ligand_edge_attr, ligand_coords
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (pdb_id) DO NOTHING
                """, processed_data)
                conn.commit()
                cursor.close()
                processed_count = len(processed_data)
                
    except Exception as e:
        print(f"Error processing chunk: {str(e)}")
    
    return processed_count  # Return count instead of data

def create_pdbbind_dataset_table():
    """Create the pdbbind_dataset table"""
    try:
        with db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS pdbbind_dataset (
                    id SERIAL PRIMARY KEY,
                    pdb_id TEXT UNIQUE,
                    -- Pocket graph data
                    pocket_one_hot BYTEA,
                    pocket_edge_index BYTEA,
                    pocket_edge_attr BYTEA,
                    pocket_coords BYTEA,
                    -- Ligand graph data
                    ligand_one_hot BYTEA,
                    ligand_edge_index BYTEA,
                    ligand_edge_attr BYTEA,
                    ligand_coords BYTEA,
                    split TEXT
                )
            """)
            conn.commit()
            cursor.close()
    except Exception as e:
        print(f"Error creating pdbbind_dataset table: {str(e)}")
        raise e

def convert_binding_value_to_micromolar(binding_value, binding_unit):
    """Convert binding value to micromolar for comparison"""
    if binding_value is None or binding_unit is None:
        return None
    
    # Convert to micromolar (uM)
    unit_lower = binding_unit.lower()
    if 'um' in unit_lower or 'μm' in unit_lower:
        return binding_value
    elif 'nm' in unit_lower:
        return binding_value / 1000.0  # nM to uM
    elif 'mm' in unit_lower:
        return binding_value * 1000.0  # mM to uM
    elif 'm' in unit_lower and 'u' not in unit_lower and 'n' not in unit_lower:
        return binding_value * 1000000.0  # M to uM
    else:
        return None  # Unknown unit

def parallel_process_pdbbind(num_workers=4, batch_size=50, affinity_cutoff_um=10.0):
    """Process PDBbind data in parallel with binding affinity filtering"""
    
    # Create table
    create_pdbbind_dataset_table()
    
    # Get IDs from pdbbind table with binding affinity filter
    try:
        with db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, pdb_id, binding_value, binding_unit
                FROM pdbbind 
                WHERE pocket_pdb IS NOT NULL 
                AND ligand_sdf IS NOT NULL
                AND binding_value IS NOT NULL 
                AND binding_unit IS NOT NULL
                ORDER BY id
            """)
            all_entries = cursor.fetchall()
            cursor.close()
    except Exception as e:
        print(f"Error fetching data from pdbbind table: {str(e)}")
        return
    
    print(f"Found {len(all_entries)} entries with binding data")
    
    # Filter by binding affinity (≤ 10 μM)
    filtered_ids = []
    for entry in all_entries:
        entry_id, pdb_id, binding_value, binding_unit = entry
        affinity_um = convert_binding_value_to_micromolar(binding_value, binding_unit)
        
        if affinity_um is not None and affinity_um <= affinity_cutoff_um:
            filtered_ids.append(entry_id)
    
    print(f"After filtering (≤{affinity_cutoff_um} μM): {len(filtered_ids)} entries")
    
    if len(filtered_ids) == 0:
        print("No entries meet the binding affinity criteria")
        return
    
    # Split IDs into chunks
    chunks = [filtered_ids[i:i + batch_size] for i in range(0, len(filtered_ids), batch_size)]
    
    # Process chunks in parallel
    with Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_chunk, chunks),
            total=len(chunks),
            desc="Processing PDBbind data chunks"
        ))
    
    # Sum up the processed counts from each chunk
    total_processed = sum(results)
    
    print(f"Successfully processed {total_processed} items out of {len(filtered_ids)} filtered items")

def set_pdbbind_split():
    """Set train/val/test splits for PDBbind dataset"""
    try:
        with db_connection() as conn:
            cursor = conn.cursor()
            
            # Get all pdb_ids
            cursor.execute("SELECT pdb_id FROM pdbbind_dataset ORDER BY pdb_id")
            all_ids = [row[0] for row in cursor.fetchall()]
            
            if len(all_ids) == 0:
                print("No data found in pdbbind_dataset table")
                return
            
            # Random split: 80% train, 10% val, 10% test
            np.random.seed(42)
            indices = np.random.permutation(len(all_ids))
            
            train_size = int(0.8 * len(all_ids))
            val_size = int(0.1 * len(all_ids))
            
            train_ids = [all_ids[i] for i in indices[:train_size]]
            val_ids = [all_ids[i] for i in indices[train_size:train_size + val_size]]
            test_ids = [all_ids[i] for i in indices[train_size + val_size:]]
            
            # Update splits
            cursor.executemany("""
                UPDATE pdbbind_dataset SET split = 'train' WHERE pdb_id = %s
            """, [(pdb_id,) for pdb_id in train_ids])
            
            cursor.executemany("""
                UPDATE pdbbind_dataset SET split = 'val' WHERE pdb_id = %s
            """, [(pdb_id,) for pdb_id in val_ids])
            
            cursor.executemany("""
                UPDATE pdbbind_dataset SET split = 'test' WHERE pdb_id = %s
            """, [(pdb_id,) for pdb_id in test_ids])
            
            conn.commit()
            cursor.close()
            
            print(f"Split assignment complete:")
            print(f"Train: {len(train_ids)} samples")
            print(f"Val: {len(val_ids)} samples") 
            print(f"Test: {len(test_ids)} samples")
            
    except Exception as e:
        print(f"Error setting splits: {str(e)}")
        raise e

if __name__ == "__main__":
    # Process only entries with binding affinity ≤ 10 μM
    parallel_process_pdbbind(num_workers=8, batch_size=50, affinity_cutoff_um=10.0)
    set_pdbbind_split()
