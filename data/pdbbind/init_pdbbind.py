import os
import sys
sys.path.append('../..')
from src.db_utils import db_connection
from tqdm import tqdm

def create_pdbbind_table(conn):
    cur = conn.cursor()
    
    # Create table if not exists
    cur.execute("""
        CREATE TABLE IF NOT EXISTS pdbbind (
            id SERIAL PRIMARY KEY,
            pdb_id TEXT UNIQUE NOT NULL,
            binding_type TEXT,
            binding_value FLOAT,
            binding_unit TEXT,
            protein_pdb TEXT,
            ligand_sdf TEXT,
            ligand_mol2 TEXT,
            pocket_pdb TEXT,
            year INTEGER,
            resolution TEXT,
            reference TEXT,
            ligand_name TEXT
        )
    """)
    conn.commit()

def parse_binding_data(binding_str):
    """Parse binding data string like 'Kd=49uM' into (type, value, unit)"""
    try:
        binding_type = binding_str[:binding_str.find('=')]
        value_unit = binding_str[binding_str.find('=')+1:]
        
        # Extract numeric value and unit
        for i, c in enumerate(value_unit):
            if not (c.isdigit() or c == '.' or c == '-'):
                value = float(value_unit[:i])
                unit = value_unit[i:]
                return binding_type, value, unit
    except:
        return None, None, None

def get_year_folder(year):
    """Determine which year folder a PDB entry belongs to"""
    if year <= 2000:
        return "1981-2000"
    elif year <= 2010:
        return "2001-2010"
    else:
        return "2011-2019"

def process_pdbbind_data(index_path, pdbbind_dir):
    print("Establishing database connection...")
    
    with db_connection() as conn:
        print("Creating database table...")
        create_pdbbind_table(conn)
        
        print("Reading index file...")
        with open(index_path, 'r') as f:
            index_content = f.read()
        
        # Skip header lines
        data_lines = [line.strip() for line in index_content.split('\n') if line.strip() and not line.startswith('#')]
        total_entries = len(data_lines)
        print(f"Found {total_entries} entries to process")
        
        cur = conn.cursor()
        batch_size = 100
        processed_count = 0
        
        # Process each entry with progress bar
        for line in tqdm(data_lines, desc="Processing PDBbind entries", unit="entry"):
            # Parse line: PDB code, resolution, release year, binding data, reference, ligand name
            parts = line.split()
            pdb_id = parts[0]
            resolution = parts[1]  # Keep as string to handle NMR
            year = int(parts[2])
            binding_str = parts[3]
            
            # Parse reference and ligand name
            reference_parts = ' '.join(parts[4:])
            if '//' in reference_parts:
                reference = reference_parts.split('//')[1].strip()
                # Extract ligand name from parentheses
                if '(' in reference and ')' in reference:
                    ligand_name = reference.split('(')[1].split(')')[0]
                else:
                    ligand_name = None
            else:
                reference = reference_parts
                ligand_name = None
            
            # Parse binding data
            binding_type, binding_value, binding_unit = parse_binding_data(binding_str)
            
            try:
                # Determine year folder
                year_folder = get_year_folder(year)
                pdb_folder = os.path.join(pdbbind_dir, "P-L", year_folder, pdb_id)
                
                # Read all required files
                protein_pdb_path = os.path.join(pdb_folder, f"{pdb_id}_protein.pdb")
                ligand_sdf_path = os.path.join(pdb_folder, f"{pdb_id}_ligand.sdf")
                ligand_mol2_path = os.path.join(pdb_folder, f"{pdb_id}_ligand.mol2")
                pocket_pdb_path = os.path.join(pdb_folder, f"{pdb_id}_pocket.pdb")
                
                # Read file contents as text
                protein_pdb = None
                ligand_sdf = None
                ligand_mol2 = None
                pocket_pdb = None
                
                if os.path.exists(protein_pdb_path):
                    with open(protein_pdb_path, 'r') as f:
                        protein_pdb = f.read()
                
                if os.path.exists(ligand_sdf_path):
                    with open(ligand_sdf_path, 'r') as f:
                        ligand_sdf = f.read()
                
                if os.path.exists(ligand_mol2_path):
                    with open(ligand_mol2_path, 'r') as f:
                        ligand_mol2 = f.read()
                
                if os.path.exists(pocket_pdb_path):
                    with open(pocket_pdb_path, 'r') as f:
                        pocket_pdb = f.read()
                
                # Insert into database with elegant upsert
                update_columns = ['binding_type', 'binding_value', 'binding_unit', 'protein_pdb', 
                                'ligand_sdf', 'ligand_mol2', 'pocket_pdb', 'year', 
                                'resolution', 'reference', 'ligand_name']
                
                update_clause = ', '.join([f"{col} = EXCLUDED.{col}" for col in update_columns])
                
                cur.execute(f"""
                    INSERT INTO pdbbind (
                        pdb_id, binding_type, binding_value, binding_unit,
                        protein_pdb, ligand_sdf, ligand_mol2, pocket_pdb,
                        year, resolution, reference, ligand_name
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (pdb_id) DO UPDATE SET {update_clause}
                """, (
                    pdb_id, binding_type, binding_value, binding_unit,
                    protein_pdb, ligand_sdf, ligand_mol2, pocket_pdb,
                    year, resolution, reference, ligand_name
                ))
                
                processed_count += 1
                # Batch commit for better performance
                if processed_count % batch_size == 0:
                    conn.commit()
                    
            except Exception as e:
                print(f"\nError processing entry {pdb_id}: {str(e)}")
                continue
        
        # Final commit for remaining entries
        conn.commit()
        print(f"Successfully processed {processed_count} entries")

if __name__ == "__main__":
    index_path = os.path.expanduser("~/scratch/datasets/pdbbind/index/INDEX_general_PL.2020R1.lst")
    pdbbind_dir = os.path.expanduser("~/scratch/datasets/pdbbind")
    process_pdbbind_data(index_path, pdbbind_dir)



