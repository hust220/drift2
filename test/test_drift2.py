#!/usr/bin/env python3
"""
Test script for Drift2 affinity prediction using database data.
"""

import os
import torch
from src.db_utils import db_connection
from src.lightning import Drift2
from drift2 import predict_affinity


def test_from_database():
    """Test affinity prediction using data from database."""
    try:
        # Get a random test case from PDBbind dataset
        with db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT pdb_id, pocket_pdb, ligand_sdf
                FROM pdbbind 
                WHERE pocket_pdb IS NOT NULL 
                AND ligand_sdf IS NOT NULL
                ORDER BY RANDOM()
                LIMIT 1
            """)
            result = cursor.fetchone()
            if result is None:
                print("No test data found in database")
                return
                
            pdb_id, pdb_content, sdf_content = result
            print(f"Selected test case: {pdb_id}")
            cursor.close()
        
        # Create output directory
        output_dir = 'test_output'
        os.makedirs(output_dir, exist_ok=True)
        
        # Save files for inspection
        protein_path = os.path.join(output_dir, f"{pdb_id}_protein.pdb")
        ligand_path = os.path.join(output_dir, f"{pdb_id}_ligand.sdf")
        
        # Write files
        with open(protein_path, 'w') as f:
            f.write(pdb_content)
        with open(ligand_path, 'w') as f:
            f.write(sdf_content)
        
        # Set up device and model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = './models/drift2_pdbbind_latest.ckpt'  # Update this path to your model checkpoint
        model = Drift2.load_from_checkpoint(model_path, map_location=device).eval().to(device)
        
        # Run affinity prediction
        affinity_score = predict_affinity(
            protein_path,
            ligand_path,
            model,
            device
        )
        
        print(f"Files saved:")
        print(f"Protein: {protein_path}")
        print(f"Ligand: {ligand_path}")
        print(f"Predicted affinity score: {affinity_score:.4f}")
            
    except Exception as e:
        print(f"Error in test_from_database: {str(e)}")
        raise e


def test_multiple_cases(num_cases=5):
    """Test multiple random cases from database."""
    try:
        # Get multiple random test cases
        with db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT pdb_id, pocket_pdb, ligand_sdf
                FROM pdbbind 
                WHERE pocket_pdb IS NOT NULL 
                AND ligand_sdf IS NOT NULL
                ORDER BY RANDOM()
                LIMIT %s
            """, (num_cases,))
            results = cursor.fetchall()
            if not results:
                print("No test data found in database")
                return
            cursor.close()
        
        # Set up device and model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = './models/drift2_pdbbind_latest.ckpt'
        model = Drift2.load_from_checkpoint(model_path, map_location=device).eval().to(device)
        
        # Create output directory
        output_dir = 'test_output'
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Testing {len(results)} cases:")
        print("-" * 50)
        
        for i, (pdb_id, pdb_content, sdf_content) in enumerate(results, 1):
            print(f"Case {i}: {pdb_id}")
            
            # Save files
            protein_path = os.path.join(output_dir, f"{pdb_id}_protein.pdb")
            ligand_path = os.path.join(output_dir, f"{pdb_id}_ligand.sdf")
            
            with open(protein_path, 'w') as f:
                f.write(pdb_content)
            with open(ligand_path, 'w') as f:
                f.write(sdf_content)
            
            # Run prediction
            try:
                affinity_score = predict_affinity(protein_path, ligand_path, model, device)
                print(f"  Predicted affinity score: {affinity_score:.4f}")
            except Exception as e:
                print(f"  Error: {str(e)}")
            
            print()
            
    except Exception as e:
        print(f"Error in test_multiple_cases: {str(e)}")
        raise e


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Drift2 model with database data')
    parser.add_argument(
        '--multiple', type=int, default=None, metavar='N',
        help='Test N random cases (default: test single case)'
    )
    parser.add_argument(
        '--model', type=str, default='./models/drift2_pdbbind_latest.ckpt',
        help='Path to model checkpoint'
    )
    
    args = parser.parse_args()
    
    if args.multiple:
        test_multiple_cases(args.multiple)
    else:
        test_from_database()
