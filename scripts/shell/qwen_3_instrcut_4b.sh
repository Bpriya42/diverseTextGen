#!/bin/bash

#SBATCH -o /rstor/pi_hzamani_umass_edu/asalemi/priya/server_logs/vllm_server.out
#SBATCH -e /rstor/pi_hzamani_umass_edu/asalemi/priya/server_logs/vllm_server.err
#SBATCH --partition=gpu,gpu-preempt
#SBATCH --nodes=1
#SBATCH -C a100
#SBATCH -G 1
#SBATCH -c 12
#SBATCH --mem=512G
#SBATCH -t 01:30:00
#SBATCH -q long

PORT=$((5000 + RANDOM % 200))

# Define log file path
DETAILS="/rstor/pi_hzamani_umass_edu/asalemi/priya/diverseTextGen/server_logs/log.txt"
DOWNLOAD_DIR="/gypsum/work1/zamani/asalemi/RAG_VS_LoRA_Personalization/cache_vllm"

# Activate virtual environment
source /rstor/pi_hzamani_umass_edu/asalemi/pluralistic_thinking/venv/bin/activate

# Log hostname and port
echo $(hostname) > "$DETAILS"
echo "$PORT" >> "$DETAILS"

# Start vllm server
vllm serve Qwen/Qwen3-4B-Instruct-2507 --host 0.0.0.0 --port "$PORT" --download-dir "$DOWNLOAD_DIR" --tensor-parallel-size 1 --max-model-len 32000