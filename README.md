# Drift2: Protein-ligand binding affinity prediction with graph neural network

Drift2 is a graph neural network model for predicting pocket-ligand binding affinity. The model takes a pocket PDB file and a ligand SDF file as input and outputs a scalar affinity score. Higher scores indicate stronger predicted binding affinity.

## Environment Setup

Install the necessary packages:

```shell
pip install torch torchvision lightning
pip install rdkit pandas numpy matplotlib scipy scikit-learn tqdm wandb
```

## Model Preparation

The project includes a pre-trained model at `models/last.ckpt`. No additional download is required.

## Usage

### Basic Usage

```bash
python drift2.py receptor.pdb ligand.sdf output.txt
```

### With Custom Model

```bash
python drift2.py receptor.pdb ligand.sdf output.txt --model ./models/my_drift2_model.ckpt
```

### Batch Processing

```bash
python drift2.py receptor.pdb ligand_database.sdf output.txt --batch 128
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
- `ligand` (required): Path to the ligand SDF or SMI file
- `output` (required): Path to save results text file
- `--model`: Path to Drift2 model checkpoint (default: `./models/pdbbind_bs8_date23-08_time09-15-58.399588/last.ckpt`)
- `--batch`: Batch size for prediction (default: 128)

## Training

Train the model using the default PDBbind configuration:

```bash
python train_drift2.py --config configs/train_pdbbind.yml
```

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
3. **Results File**: Results saved to specified output file

### Interpreting Results

- **Higher scores** indicate stronger predicted binding affinity
- **Lower scores** indicate weaker predicted binding affinity
- The exact score range depends on the training data and model

## Example Output

```
Using device: cuda
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

## Project Structure

```
drift2/
├── configs/           # Training configurations
├── data/             # Dataset initialization scripts
├── analysis/         # Analysis and evaluation scripts
├── src/              # Source code
│   ├── lightning.py  # PyTorch Lightning model
│   ├── gnn.py       # Graph neural network implementation
│   ├── datasets.py  # Dataset loading utilities
│   └── utils.py     # Utility functions
├── models/          # Saved model checkpoints
├── drift2.py        # Main prediction script
├── train_drift2.py  # Training script
└── run_screening.sh # Automated screening pipeline
```

## Automated Screening

Use the provided screening script for batch processing:

```bash
./run_screening.sh
```

This script will:
1. Check environment setup
2. Verify project structure
3. Run screening on multiple targets
4. Perform analysis on results

## Contact

If you have any questions, please contact me at jianopt@gmail.com
