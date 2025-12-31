#!/bin/bash

#SBATCH --job-name=rag_full_pipeline
#SBATCH --output=/rstor/pi_hzamani_umass_edu/asalemi/priya/diverseTextGen/server_logs/pipeline_%j.out
#SBATCH --error=/rstor/pi_hzamani_umass_edu/asalemi/priya/diverseTextGen/server_logs/pipeline_%j.err
#SBATCH --partition=gpu,gpu-preempt
#SBATCH --nodes=1
#SBATCH -C a100
#SBATCH -G 1
#SBATCH -c 16
#SBATCH --mem=512G
#SBATCH --time=24:00:00
#SBATCH -q long

# =============================================================================
# Full RAG Pipeline: Start LLM Server + Run Baseline + Run RAG Experiment
# =============================================================================
#
# Usage:
#   sbatch scripts/shell/run_full_pipeline.sh
#
# =============================================================================

set -e

# ===========================
# Configuration
# ===========================

# Number of queries to process (uncomment to limit, default is all)
# N_QUERIES=10
N_QUERIES=${N_QUERIES:-all}

# Experiment description
DESCRIPTION="full_pipeline_$(date +%Y%m%d_%H%M%S)"

# Paths
BASE_DIR="/rstor/pi_hzamani_umass_edu/asalemi/priya"
SCRIPT_DIR="${BASE_DIR}/diverseTextGen"
SERVER_LOGS="${BASE_DIR}/server_logs"
QUERIES_PATH="${SCRIPT_DIR}/data/antique/train.jsonl"

# vLLM settings
DOWNLOAD_DIR="/gypsum/work1/zamani/asalemi/RAG_VS_LoRA_Personalization/cache_vllm"
MODEL="Qwen/Qwen3-4B-Instruct-2507"
PORT=$((5000 + RANDOM % 200))

# ===========================
# Environment Setup
# ===========================

echo "========================================================================"
echo "Setting up environment..."
echo "========================================================================"

# Activate conda environment for experiments
source "${BASE_DIR}/.conda/etc/profile.d/conda.sh"
conda activate "${BASE_DIR}/env"

# Set Python path
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

cd "$SCRIPT_DIR"

echo "Environment activated: ${BASE_DIR}/env"
echo "Working directory: $(pwd)"
echo "Queries: $N_QUERIES"

# ===========================
# Start vLLM Server
# ===========================

echo ""
echo "========================================================================"
echo "Starting vLLM Server..."
echo "========================================================================"
echo "Model: $MODEL"
echo "Port: $PORT"
echo "========================================================================"

# Write server details to log file
echo $(hostname) > "${SERVER_LOGS}/log.txt"
echo "$PORT" >> "${SERVER_LOGS}/log.txt"

# Also write to local server_logs for the refactored code
mkdir -p "${SCRIPT_DIR}/server_logs"
echo $(hostname) > "${SCRIPT_DIR}/server_logs/log.txt"
echo "$PORT" >> "${SCRIPT_DIR}/server_logs/log.txt"

# Start vLLM server in background
# Use the venv that has vllm installed
source /rstor/pi_hzamani_umass_edu/asalemi/pluralistic_thinking/venv/bin/activate

vllm serve "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --download-dir "$DOWNLOAD_DIR" \
    --tensor-parallel-size 1 \
    --max-model-len 32000 &

VLLM_PID=$!
echo "vLLM server started with PID: $VLLM_PID"

# Switch back to conda env for experiments
source "${BASE_DIR}/.conda/etc/profile.d/conda.sh"
conda activate "${BASE_DIR}/env"

# Wait for server to be ready
echo "Waiting for vLLM server to be ready..."
SERVER_URL="http://$(hostname):${PORT}/v1"
MAX_WAIT=300  # 5 minutes
WAITED=0

while [ $WAITED -lt $MAX_WAIT ]; do
    if curl -s "${SERVER_URL}/models" > /dev/null 2>&1; then
        echo "✓ vLLM server is ready!"
        break
    fi
    sleep 5
    WAITED=$((WAITED + 5))
    echo "  Waiting... (${WAITED}s / ${MAX_WAIT}s)"
done

if [ $WAITED -ge $MAX_WAIT ]; then
    echo "ERROR: vLLM server failed to start within ${MAX_WAIT} seconds"
    kill $VLLM_PID 2>/dev/null
    exit 1
fi

# ===========================
# Run Baseline Experiment
# ===========================

echo ""
echo "========================================================================"
echo "Running Baseline Experiment (Direct LLM, no RAG)"
echo "========================================================================"

CMD_BASELINE="python scripts/run_baseline_experiment.py \
    --queries_path $QUERIES_PATH \
    --description \"baseline_${DESCRIPTION}\""

if [ "$N_QUERIES" != "all" ]; then
    CMD_BASELINE="$CMD_BASELINE -n $N_QUERIES"
fi

echo "Command: $CMD_BASELINE"
eval $CMD_BASELINE

echo "✓ Baseline experiment complete!"

# ===========================
# Run RAG Experiment
# ===========================

echo ""
echo "========================================================================"
echo "Running RAG Experiment (Multi-Agent System)"
echo "========================================================================"

CMD_RAG="python scripts/run_full_experiment.py \
    --queries_path $QUERIES_PATH \
    --description \"rag_${DESCRIPTION}\""

if [ "$N_QUERIES" != "all" ]; then
    CMD_RAG="$CMD_RAG -n $N_QUERIES"
fi

echo "Command: $CMD_RAG"
eval $CMD_RAG

echo "✓ RAG experiment complete!"

# ===========================
# Cleanup
# ===========================

echo ""
echo "========================================================================"
echo "Pipeline Complete! Shutting down vLLM server..."
echo "========================================================================"

kill $VLLM_PID 2>/dev/null || true

echo ""
echo "Results saved to: ${SCRIPT_DIR}/output/"
echo "========================================================================"