
#!/bin/bash

# Drift2 Universal Screening Pipeline
# This script performs virtual screening using Drift2 model
# Usage: ./run_screening.sh [pocket.pdb] [ligand_library.sdf]

set -e  # Exit on any error

echo "=== Drift2 Universal Screening Pipeline ==="
echo "Starting virtual screening workflow..."

# check if environment is set up
echo "Checking environment setup..."

# Check Python
if ! command -v python &> /dev/null; then
    echo "Error: Python is not installed or not in PATH"
    echo ""
    echo "Please install Miniconda (recommended for scientific computing):"
    echo "   - Download from: https://docs.conda.io/en/latest/miniconda.html"
    echo "   - Or use mamba: https://mamba.readthedocs.io/en/latest/installation.html"
    echo ""
    exit 1
fi

# Check GPU availability (CUDA or MPS)
python -c "
import torch
if torch.cuda.is_available():
    print('✓ CUDA available - will use NVIDIA GPU')
elif torch.backends.mps.is_available():
    print('✓ MPS available - will use Apple Silicon GPU')
else:
    print('⚠ No GPU acceleration available, will use CPU')
"

# Check required Python packages
echo "Checking Python dependencies..."
python -c "
import sys
required_packages = ['torch', 'rdkit', 'pandas', 'numpy', 'matplotlib', 'scipy', 'lightning']
missing = []
for pkg in required_packages:
    try:
        __import__(pkg)
    except ImportError:
        missing.append(pkg)

if missing:
    print(f'Missing packages: {missing}')
    print('Please install with: pip install ' + ' '.join(missing))
    sys.exit(1)
else:
    print('✓ All required packages are installed')
"

# check if drift2 folder is set up
echo "Checking drift2 project..."

# Hardcoded repository URL from .git/config
REPO_URL="https://github.com/hust220/drift2.git"

# Check if we're in the drift2 directory
if [ ! -f "drift2.py" ]; then
    echo "Drift2 project not found in current directory"
    echo "Cloning from GitHub: $REPO_URL"
    
    # Go to parent directory and clone
    cd ..
    git clone "$REPO_URL"
    cd drift2
    
    echo "✓ Drift2 project cloned successfully"
else
    echo "✓ Drift2 project found"
fi

# check for model
echo "Checking for model..."

MODEL_PATH="models/last.ckpt"

if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model not found at $MODEL_PATH"
    echo "Please ensure the pre-trained model is available at models/last.ckpt"
    exit 1
else
    echo "✓ Model found at: $MODEL_PATH"
fi

# Get input parameters
POCKET_FILE="$1"
LIGAND_FILE="$2"

# If parameters not provided, ask user
if [ -z "$POCKET_FILE" ]; then
    echo "Please provide the pocket PDB file path:"
    read -r POCKET_FILE
fi

if [ -z "$LIGAND_FILE" ]; then
    echo "Please provide the ligand library SDF file path:"
    read -r LIGAND_FILE
fi

# Validate input files
if [ ! -f "$POCKET_FILE" ]; then
    echo "Error: Pocket file '$POCKET_FILE' not found"
    exit 1
fi

if [ ! -f "$LIGAND_FILE" ]; then
    echo "Error: Ligand file '$LIGAND_FILE' not found"
    exit 1
fi

echo "✓ Input files validated:"
echo "  - Pocket: $POCKET_FILE"
echo "  - Ligand library: $LIGAND_FILE"

# Generate output filename
POCKET_NAME=$(basename "$POCKET_FILE" .pdb)
LIGAND_NAME=$(basename "$LIGAND_FILE" .sdf)
OUTPUT_FILE="${POCKET_NAME}_${LIGAND_NAME}_scores.txt"

echo "Output will be saved to: $OUTPUT_FILE"

# run screening
echo "Starting virtual screening..."

python drift2.py "$POCKET_FILE" "$LIGAND_FILE" "$OUTPUT_FILE" --model "$MODEL_PATH" --batch 128 --device auto

if [ $? -eq 0 ]; then
    echo "✓ Virtual screening completed successfully!"
    echo "Results saved to: $OUTPUT_FILE"
else
    echo "✗ Virtual screening failed"
    exit 1
fi

echo "=== Virtual Screening Pipeline Completed ==="
echo ""
echo "Results saved to: $OUTPUT_FILE"
echo ""
echo "The output file contains compound IDs and their predicted affinity scores."
echo "Higher scores indicate stronger predicted binding affinity."



