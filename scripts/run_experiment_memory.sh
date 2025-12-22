#!/bin/bash
#SBATCH --job-name=rag_memory
#SBATCH --output=/rstor/pi_hzamani_umass_edu/asalemi/priya/diverseTextGen/server_logs/experiment_memory_%j.out
#SBATCH --error=/rstor/pi_hzamani_umass_edu/asalemi/priya/diverseTextGen/server_logs/experiment_memory_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=256GB
#SBATCH --cpus-per-task=16
#SBATCH --partition=cpu

# =============================================================================
# Experiment: Quality & Memory-Controlled
# =============================================================================
#
# Runs the iterative RAG system until:
# 1. Quality criteria are met (primary termination condition)
# 2. Memory usage exceeds thresholds (RAM 85%, GPU 85%)
#
# Usage:
#   sbatch scripts/run_experiment_memory.sh
#   sbatch scripts/run_experiment_memory.sh 10  # Run first 10 queries only
#
# =============================================================================

set -e

# ===========================
# Configuration
# ===========================

N_QUERIES=${1:-all}
DESCRIPTION="rag_quality_memory_controlled"

# Memory thresholds (terminate when exceeded)
MAX_RAM_PERCENT=85
MAX_GPU_PERCENT=85

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
echo "EXPERIMENT: Quality & Memory-Controlled"
echo "========================================================================"
echo "Queries: $N_QUERIES"
echo "Description: $DESCRIPTION"
echo "Termination: Quality complete OR Memory exceeded"
echo "Max RAM: ${MAX_RAM_PERCENT}%"
echo "Max GPU: ${MAX_GPU_PERCENT}%"
echo "========================================================================"

# Build command
CMD="python run_full_experiment.py \
    --queries_path $QUERIES_PATH \
    --max_ram_percent $MAX_RAM_PERCENT \
    --max_gpu_percent $MAX_GPU_PERCENT \
    --description \"$DESCRIPTION\""

if [ "$N_QUERIES" != "all" ]; then
    CMD="$CMD -n $N_QUERIES"
fi

echo ""
echo "Running: $CMD"
echo ""

# Run experiment
eval $CMD

echo ""
echo "========================================================================"
echo "Experiment Complete! (Quality & Memory-Controlled)"
echo "========================================================================"
