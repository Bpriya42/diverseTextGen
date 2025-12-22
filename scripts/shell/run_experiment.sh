#!/bin/bash
#SBATCH --job-name=rag_experiment
#SBATCH --output=/rstor/pi_hzamani_umass_edu/asalemi/priya/diverseTextGen/server_logs/experiment_%j.out
#SBATCH --error=/rstor/pi_hzamani_umass_edu/asalemi/priya/diverseTextGen/server_logs/experiment_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=256GB
#SBATCH --cpus-per-task=16
#SBATCH --partition=cpu

# Run iterative RAG experiment with ICAT tracking
#
# Termination is controlled by:
# 1. Quality metrics (primary - terminates when answer quality is sufficient)
# 2. Memory constraints (terminates when RAM/GPU usage exceeds limits)
#
# Usage:
#   sbatch scripts/run_experiment.sh [N_QUERIES] [DESCRIPTION] [OPTIONS]
#   bash scripts/run_experiment.sh [N_QUERIES] [DESCRIPTION] [OPTIONS]
#
# Options:
#   --max_ram_percent N     Set max RAM usage percentage (default: 90)
#   --max_gpu_percent N     Set max GPU usage percentage (default: 90)
#
# Examples:
#   # Run all queries with memory constraints
#   sbatch scripts/run_experiment.sh all "experiment_v1"
#
#   # Run first 10 queries with custom memory limits
#   sbatch scripts/run_experiment.sh 10 "test_run" --max_ram_percent 85 --max_gpu_percent 85

set -e

# ===========================
# Parse Arguments
# ===========================

N_QUERIES=${1:-all}
DESCRIPTION=${2:-"experiment"}

# Shift past positional arguments
shift 2 2>/dev/null || true

# Default values
MAX_RAM_PERCENT=""
MAX_GPU_PERCENT=""

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --max_ram_percent)
            MAX_RAM_PERCENT="$2"
            shift 2
            ;;
        --max_gpu_percent)
            MAX_GPU_PERCENT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ===========================
# Configuration
# ===========================

# Paths
BASE_DIR="/rstor/pi_hzamani_umass_edu/asalemi/priya"
SCRIPT_DIR="${BASE_DIR}/diverseTextGen"
QUERIES_PATH="${SCRIPT_DIR}/data/antique/train.jsonl"

# ===========================
# Environment Setup
# ===========================

echo "========================================================================"
echo "Setting up environment..."
echo "========================================================================"

# Activate conda environment
source "${BASE_DIR}/.conda/etc/profile.d/conda.sh"
conda activate "${BASE_DIR}/env"

# Set Python path
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

# Change to script directory
cd "$SCRIPT_DIR"

echo "Environment activated: ${BASE_DIR}/env"
echo "Working directory: $(pwd)"

# ===========================
# Run Experiment
# ===========================

echo ""
echo "========================================================================"
echo "Starting Iterative RAG Experiment"
echo "========================================================================"
echo "Queries: $N_QUERIES"
echo "Description: $DESCRIPTION"
echo "Termination: Quality complete OR Memory exceeded"
[ -n "$MAX_RAM_PERCENT" ] && echo "Max RAM: ${MAX_RAM_PERCENT}%"
[ -n "$MAX_GPU_PERCENT" ] && echo "Max GPU: ${MAX_GPU_PERCENT}%"
echo "========================================================================"

# Build command
CMD="python run_full_experiment.py \
    --queries_path $QUERIES_PATH \
    --description \"$DESCRIPTION\""

if [ "$N_QUERIES" != "all" ]; then
    CMD="$CMD -n $N_QUERIES"
fi

if [ -n "$MAX_RAM_PERCENT" ]; then
    CMD="$CMD --max_ram_percent $MAX_RAM_PERCENT"
fi

if [ -n "$MAX_GPU_PERCENT" ]; then
    CMD="$CMD --max_gpu_percent $MAX_GPU_PERCENT"
fi

echo ""
echo "Running: $CMD"
echo ""

# Run experiment
eval $CMD

echo ""
echo "========================================================================"
echo "Experiment Complete!"
echo "========================================================================"
