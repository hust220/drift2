#!/bin/bash

set -e

# Paths
RECEPTOR="/home/tyq4zn/scratch/codes/drift2/analysis/HCAR/screening/HCAR2_pocket.pdb"
LIGANDS="/home/tyq4zn/scratch/datasets/enamine/Sample95_6M/Sample95_6M.smi"

# Validate inputs
if [ ! -f "$RECEPTOR" ]; then
    echo "Error: receptor file '$RECEPTOR' not found in $(pwd)"
    exit 1
fi

if [ ! -f "$LIGANDS" ]; then
    echo "Error: ligands file '$LIGANDS' not found"
    exit 1
fi


POCKET_NAME=$(basename "$RECEPTOR" .pdb)
OUTPUT_FILE="/home/tyq4zn/scratch/codes/drift2/analysis/HCAR/screening/${POCKET_NAME}_Sample95_6M_scores.txt"

echo "Running Drift2 on $RECEPTOR with $LIGANDS"
python /home/tyq4zn/scratch/codes/drift2/drift2.py "$RECEPTOR" "$LIGANDS" "$OUTPUT_FILE" --model pdbbind_aff --batch 128 --device auto

echo "Done. Results -> $OUTPUT_FILE"


