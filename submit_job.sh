#!/bin/bash

# Generic SLURM job submission script
# Usage: ./submit_job.sh <script_path> [job_name]
# 
# This script can be customized for different projects by modifying:
# - SLURM parameters (partition, resources, time limits)
# - Account name (uncomment and set the account line)
# - Any project-specific module loading or environment setup

if [ $# -eq 0 ]; then
    echo "Error: No script path provided"
    echo "Usage: $0 <script_path> [job_name]"
    echo "Example: $0 ./train.sh train"
    echo "Example: $0 ./run_experiment.sh    (uses default job name based on script name)"
    echo "Note: job_name should be 5 characters or less"
    exit 1
fi

SCRIPT_PATH="$1"
# Extract script name and use first 4 characters as default job name
SCRIPT_NAME=$(basename "$SCRIPT_PATH" .sh)
DEFAULT_JOB_NAME="${SCRIPT_NAME:0:4}"
JOB_NAME="${2:-$DEFAULT_JOB_NAME}"  # Use second argument if provided, otherwise use first 4 chars of script name

# Check if the script file exists
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: Script file '$SCRIPT_PATH' does not exist"
    exit 1
fi

# Validate job name length
if [ ${#JOB_NAME} -gt 5 ]; then
    echo "Warning: Job name '$JOB_NAME' is longer than 5 characters, truncating to '${JOB_NAME:0:5}'"
    JOB_NAME="${JOB_NAME:0:5}"
fi

# Create logs directory if it doesn't exist
LOG_DIR="./slurm_logs"
mkdir -p "$LOG_DIR"

# Generate timestamp for unique log files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Create a temporary SLURM submission script
SLURM_SCRIPT=$(mktemp /tmp/slurm_submit_XXXXXX.sh)

cat > "$SLURM_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --account=dokhlab
#SBATCH --cpus-per-task=2
#SBATCH --mem=64GB
#SBATCH --output=${LOG_DIR}/${JOB_NAME}_${TIMESTAMP}_%j.out
#SBATCH --error=${LOG_DIR}/${JOB_NAME}_${TIMESTAMP}_%j.err

# Print job information
echo "Job started at: \$(date)"
echo "Job ID: \$SLURM_JOB_ID"
echo "Node: \$SLURM_NODELIST"
echo "Working directory: \$(pwd)"
echo "Script being executed: $SCRIPT_PATH"
echo "----------------------------------------"

# Change to the script's directory
cd "\$(dirname "$SCRIPT_PATH")"

# Load any required modules or set up environment
# Example: module load cuda/12.8
# Example: source activate myenv

# Execute the script
bash "$SCRIPT_PATH"

echo "----------------------------------------"
echo "Job finished at: \$(date)"
EOF

# Submit the job
echo "Submitting job: $JOB_NAME"
echo "Script: $SCRIPT_PATH"
echo "SLURM parameters:"
echo "  - Partition: gpu"
echo "  - GPU: 1"
echo "  - CPUs: 2"
echo "  - Memory: 64GB"
echo "  - Time limit: 3-00:00:00 (3 days)"
echo "  - Account: dokhlab"
echo ""

sbatch "$SLURM_SCRIPT"

# Clean up the temporary script
rm "$SLURM_SCRIPT"

echo ""
echo "Job submitted successfully!"
echo "Monitor with: squeue -u \$USER"
echo "Log directory: $LOG_DIR"
echo "Check output files: ${LOG_DIR}/${JOB_NAME}_${TIMESTAMP}_<job_id>.out and ${LOG_DIR}/${JOB_NAME}_${TIMESTAMP}_<job_id>.err"
