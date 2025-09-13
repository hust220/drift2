
#!/bin/bash

# Drift2 Screening Pipeline
# This script automates the complete screening workflow from environment setup to analysis

set -e  # Exit on any error

echo "=== Drift2 Screening Pipeline ==="
echo "Starting screening workflow..."

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

# download yuel2 model
echo "Checking for yuel2 model..."

MODEL_DIR="models"
YUEL2_MODEL="$MODEL_DIR/yuel2_model.ckpt"

if [ ! -f "$YUEL2_MODEL" ]; then
    echo "Yuel2 model not found. Please download it manually:"
    echo "wget https://zenodo.org/records/16921378/files/yuel_pocket.ckpt?download=1 -O $YUEL2_MODEL"
    echo "Or place your trained model checkpoint at: $YUEL2_MODEL"
    echo "Continuing with default drift2 model..."
    YUEL2_MODEL="./models/pdbbind_bs8_date23-08_time09-15-58.399588/last.ckpt"
else
    echo "✓ Yuel2 model found at: $YUEL2_MODEL"
fi

# run screening
echo "Starting screening process..."

# Define screening targets and their parameters
declare -A TARGETS=(
    ["GPR75_1"]="GPR75_pocket1.pdb"
    ["GPR75_2"]="GPR75_pocket2.pdb" 
    ["HCAR1"]="HCAR1_pocket.pdb"
    ["HCAR2"]="HCAR2_pocket.pdb"
)

# Define ligand databases
declare -A LIGAND_DBS=(
    ["GPR75_1"]="hmdb_metabolites.smi"
    ["GPR75_2"]="hmdb_metabolites.smi"
    ["HCAR1"]="hll460k_compounds.smi"
    ["HCAR2"]="hll460k_compounds.smi"
)

# Run screening for each target
for target in "${!TARGETS[@]}"; do
    pocket_file="${TARGETS[$target]}"
    ligand_db="${LIGAND_DBS[$target]}"
    output_file="${target}_${ligand_db%.*}.scores"
    
    echo "Screening $target with $pocket_file against $ligand_db..."
    
    if [ ! -f "$pocket_file" ]; then
        echo "Warning: Pocket file $pocket_file not found, skipping $target"
        continue
    fi
    
    if [ ! -f "$ligand_db" ]; then
        echo "Warning: Ligand database $ligand_db not found, skipping $target"
        continue
    fi
    
    # Run drift2 screening
    python drift2.py "$pocket_file" "$ligand_db" "$output_file" --model "$YUEL2_MODEL" --batch 128
    
    if [ $? -eq 0 ]; then
        echo "✓ Screening completed for $target: $output_file"
    else
        echo "✗ Screening failed for $target"
    fi
done

# run analysis
echo "Starting analysis..."

# Run analysis for GPR75 if scores exist
if [ -f "GPR75_1_hmdb.scores" ] && [ -f "GPR75_2_hmdb.scores" ]; then
    echo "Running GPR75 analysis..."
    cd analysis/GPR75
    python analyze_GPR75.py
    cd ../..
    echo "✓ GPR75 analysis completed"
fi

# Run analysis for HCAR if scores exist  
if [ -f "HCAR1_HLL460K.scores" ] && [ -f "HCAR2_HLL460K.scores" ]; then
    echo "Running HCAR analysis..."
    cd analysis/HCAR
    python analyze_HCAR.py
    cd ../..
    echo "✓ HCAR analysis completed"
fi

# Run PDBbind analysis if available
if [ -f "analysis/pdbbind/analyze_pdbbind.py" ]; then
    echo "Running PDBbind analysis..."
    cd analysis/pdbbind
    python analyze_pdbbind.py
    cd ../..
    echo "✓ PDBbind analysis completed"
fi

echo "=== Screening Pipeline Completed ==="
echo "Check the following for results:"
echo "- Score files: *.scores"
echo "- Analysis results: analysis/*/top100_size_filtered_scores.csv"
echo "- Distribution plots: analysis/*/score_distribution.png"



