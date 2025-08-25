# Drift2 Affinity Prediction Usage Guide

Drift2 is a graph neural network model for predicting pocket-ligand binding affinity. This guide explains how to use the updated `drift2.py` script for affinity prediction.

## Overview

The updated Drift2 model takes a pocket PDB file and a ligand SDF file as input and outputs a scalar affinity score. Higher scores indicate stronger predicted binding affinity.

## Usage

### Basic Usage

```bash
python drift2.py receptor.pdb ligand.sdf
```

### With Optional Output File

```bash
python drift2.py receptor.pdb ligand.sdf --output results.txt
```

### Specify Custom Model

```bash
python drift2.py receptor.pdb ligand.sdf --model ./models/my_drift2_model.ckpt
```

### Run Test with Database Data

```bash
python test_drift2.py
```

### Run Multiple Test Cases

```bash
python test_drift2.py --multiple 5
```

## Command Line Arguments

- `receptor` (required): Path to the receptor PDB file
- `ligand` (required): Path to the ligand SDF file
- `--output`: Path to save results text file (optional)
- `--model`: Path to Drift2 model checkpoint (default: `./models/drift2_pdbbind_latest.ckpt`)
- `--random_seed`: Random seed for reproducibility

## Testing Script Arguments (test_drift2.py)

- `--multiple N`: Test N random cases from database
- `--model`: Path to model checkpoint

## Model Training

To train a new Drift2 model:

### PDBbind Dataset
```bash
python train_drift2.py --config configs/train_pdbbind.yml
```

### MOAD Dataset
```bash
python train_drift2.py --config configs/train_moad.yml
```

### Custom Dataset
```bash
python train_drift2.py --table_name your_dataset_table
```

## Input File Requirements

### PDB File (Receptor)
- Standard PDB format
- Should contain the protein pocket structure
- CA atoms are used for residue representation

### SDF File (Ligand)
- Standard SDF/MOL format
- Should contain the ligand molecule
- Hydrogen atoms are automatically excluded

## Output

The script outputs:
1. **Affinity Score**: A scalar value representing predicted binding affinity
2. **Console Output**: Score and interpretation
3. **Optional File**: Results saved to specified output file

### Interpreting Results

- **Higher scores** indicate stronger predicted binding affinity
- **Lower scores** indicate weaker predicted binding affinity
- The exact score range depends on the training data and model

## Example Output

```
Using device:  cuda
Predicted affinity score: 2.3456
Higher scores indicate stronger predicted binding affinity
Results saved to: results.txt
```

## Model Architecture

The Drift2 model uses:
- **Graph Neural Network**: Processes combined pocket-ligand graphs
- **Node Features**: Amino acid and atom type one-hot encodings
- **Edge Features**: Distance, backbone connections, and bond types
- **Global Pooling**: Aggregates node features to graph-level representation
- **Output Layer**: Produces scalar affinity score

## Training Data Format

The model expects data in the following database table format:
- `pocket_one_hot`: Serialized pocket node features
- `pocket_edge_index`: Pocket edge indices
- `pocket_edge_attr`: Pocket edge attributes
- `pocket_coords`: Pocket coordinates
- `ligand_one_hot`: Serialized ligand node features
- `ligand_edge_index`: Ligand edge indices
- `ligand_edge_attr`: Ligand edge attributes
- `ligand_coords`: Ligand coordinates

## Error Handling

Common issues and solutions:
- **"No valid residues found"**: Check PDB file format and ensure CA atoms are present
- **"Failed to parse molecule"**: Check SDF file format and molecule validity
- **Model loading errors**: Verify model checkpoint path and compatibility
