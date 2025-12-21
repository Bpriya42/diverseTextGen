#!/bin/bash
#SBATCH --job-name=baseline_experiment
#SBATCH --output=/rstor/pi_hzamani_umass_edu/asalemi/priya/diverseTextGen/server_logs/baseline_%j.out
#SBATCH --error=/rstor/pi_hzamani_umass_edu/asalemi/priya/diverseTextGen/server_logs/baseline_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=8
#SBATCH --partition=cpu

# Run baseline experiment with direct LLM responses (no RAG)
#
# Usage:
#   sbatch scripts/run_baseline.sh [N_QUERIES] [DESCRIPTION]
#   bash scripts/run_baseline.sh [N_QUERIES] [DESCRIPTION]
#
# Examples:
#   sbatch scripts/run_baseline.sh 10 "baseline_test"
#   sbatch scripts/run_baseline.sh all "baseline_full"

set -e

# ===========================
# Configuration
# ===========================

# Arguments
N_QUERIES=${1:-all}
DESCRIPTION=${2:-"baseline"}

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
# Run Baseline Experiment
# ===========================

echo ""
echo "========================================================================"
echo "Starting Baseline Experiment (Direct LLM)"
echo "========================================================================"
echo "Queries: $N_QUERIES"
echo "Description: $DESCRIPTION"
echo "Experiment Type: Baseline (no RAG/agent flow)"
echo "========================================================================"

# Build command
CMD="python run_baseline_experiment.py \
    --queries_path $QUERIES_PATH \
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
echo "Baseline Experiment Complete!"
echo "========================================================================"
