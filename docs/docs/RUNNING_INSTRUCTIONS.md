# Running Instructions and Slurm Guide

This document provides comprehensive instructions for running the Diverse Text Generation system, including both local execution and Slurm job submission.

## Table of Contents

1. [Environment Setup](#environment-setup)
2. [Running the Code](#running-the-code)
3. [Slurm Job Submission](#slurm-job-submission)
4. [Configuration](#configuration)
5. [Troubleshooting](#troubleshooting)

---

## Environment Setup

### 1. Activate Conda Environment

```bash
source /rstor/pi_hzamani_umass_edu/asalemi/priya/.conda/etc/profile.d/conda.sh
conda activate /rstor/pi_hzamani_umass_edu/asalemi/priya/env
```

### 2. Navigate to Project Directory

```bash
cd /rstor/pi_hzamani_umass_edu/asalemi/priya/diverseTextGen
```

### 3. Verify Configuration

Check your current configuration:

```bash
python config.py
```

This will display all paths and settings being used.

---

## Running the Code

### Prerequisites

Before running, ensure:
- ✅ Conda environment is activated
- ✅ vLLM server is running (check `../server_logs/log.txt` for server URL)
- ✅ Corpus file exists at the configured path (default: `./data/antique/corpus_filtered_50.jsonl`)

### Option A: Single Query (Quick Test)

Run a single query for testing:

```bash
python run_langgraph.py \
    --query "What causes headaches and how can they be treated?" \
    --query_id "test_001" \
    --max_iterations 3 \
    --output ./output/test_result.json
```

**Parameters:**
- `--query`: The question to answer
- `--query_id`: Unique identifier for this query
- `--max_iterations`: Maximum number of refinement iterations (default: 3)
- `--output`: Path to save the result JSON file

### Option B: Batch Processing from File

Process multiple queries from a JSONL file:

```bash
# Process first 10 queries
python run_langgraph.py \
    --input_file ./data/antique/test.jsonl \
    -n 10 \
    --max_iterations 3 \
    --output ./output/batch_results

# Process all queries in file
python run_langgraph.py \
    --input_file ./data/antique/test.jsonl \
    --max_iterations 3 \
    --output ./output/batch_results
```

**Parameters:**
- `--input_file`: Path to JSONL file with queries
- `-n`: Number of queries to process (optional, processes all if omitted)
- `--max_iterations`: Maximum iterations per query
- `--output`: Directory to save results (one JSON file per query)

**Input Format:**
The JSONL file should contain one query per line:
```json
{"query_id": "3990512", "query_description": "how can we get concentration on something?"}
{"query_id": "714612", "query_description": "Why doesn't the water fall off earth if it's round?"}
```

### Option C: With Budget Constraints

Add token and time budgets to limit resource usage:

```bash
python run_langgraph.py \
    --query "Explain quantum computing" \
    --query_id "quantum_001" \
    --max_iterations 5 \
    --token_budget 50000 \
    --walltime_budget_s 300 \
    --output ./output/result.json
```

**Additional Parameters:**
- `--token_budget`: Maximum tokens to use across all iterations
- `--walltime_budget_s`: Maximum wall-clock time in seconds

### Output Format

Results are saved as JSON files with the following structure:

```json
{
  "query_id": "test_001",
  "query": "What causes headaches?",
  "final_answer": "...",
  "total_iterations": 3,
  "termination_reason": "max_iterations",
  "iteration_history": [...],
  "total_runtime_seconds": 45.2,
  "timestamps": {...},
  "budget_used": {...}
}
```

### Evaluating Results

After running, evaluate outputs using ICAT:

```bash
python evaluate_icat.py \
    --output_path ./output/batch_results \
    --corpus_path ./data/antique/corpus_filtered_50.jsonl
```

---

## Slurm Job Submission

### CPU Interactive Jobs

Request an interactive CPU session:

```bash
srun --pty --partition=cpu --nodes=1 --mem=32G -c 1 -t 08:00:00 bash
```

**Parameters:**
- `--partition=cpu`: CPU partition
- `--nodes=1`: Single node
- `--mem=32G`: 32GB RAM
- `-c 1`: 1 CPU core
- `-t 08:00:00`: 8 hour time limit

### GPU Interactive Jobs

Request an interactive GPU session:

```bash
srun --pty --partition=gpu --nodes=1 --gres=gpu:1 --mem=64G -c 4 -t 08:00:00 bash
```

**Parameters:**
- `--partition=gpu`: GPU partition (may also be `gpu-preempt`, `gpu-long`, etc.)
- `--nodes=1`: Single node
- `--gres=gpu:1`: Request 1 GPU (can request multiple: `gpu:2`, `gpu:4`)
- `--mem=64G`: Memory allocation (GPU nodes typically have 128GB-512GB available)
- `-c 4`: CPU cores (recommended: 4-8 for GPU jobs)
- `-t 08:00:00`: Time limit

**GPU Memory Allocation:**
- Standard GPU nodes: 128GB - 256GB RAM
- High-memory GPU nodes: 512GB - 1TB RAM
- A100 nodes: Often 512GB+

**Check Available Resources:**
```bash
# Check available GPU resources
sinfo -p gpu -o "%N %c %m %G"

# Check partition details
scontrol show partition gpu

# See specific node specs
scontrol show node <nodename>
```

### Advanced GPU Options

**Request Specific GPU Type:**
```bash
srun --pty --partition=gpu --nodes=1 --gres=gpu:a100:1 --mem=128G -c 8 -t 08:00:00 bash
```

**Request Multiple GPUs:**
```bash
srun --pty --partition=gpu --nodes=1 --gres=gpu:2 --mem=256G -c 16 -t 08:00:00 bash
```

**Use Preemptible/Low-Priority GPU (shorter wait times):**
```bash
srun --pty --partition=gpu-preempt --nodes=1 --gres=gpu:1 --mem=64G -c 4 -t 08:00:00 bash
```

### Starting vLLM Server in GPU Session

Once you have a GPU interactive session, start the vLLM server:

```bash
# Activate environment
source /rstor/pi_hzamani_umass_edu/asalemi/priya/.conda/etc/profile.d/conda.sh
conda activate /rstor/pi_hzamani_umass_edu/asalemi/priya/env

# Start vLLM server
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --host 0.0.0.0 \
    --port 8000

# In another terminal/session, update parent folder's server log
echo "gpu023" > ../server_logs/log.txt
echo "8000" >> ../server_logs/log.txt
```

**Note:** Replace `gpu023` with your actual GPU node hostname. This project uses the parent folder's server_logs directory.

### Batch Job Submission (SBATCH Script)

Create a batch script for non-interactive jobs:

```bash
#!/bin/bash
#SBATCH --job-name=diverse_text_gen
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=output/slurm_%j.out
#SBATCH --error=output/slurm_%j.err

# Activate environment
source /rstor/pi_hzamani_umass_edu/asalemi/priya/.conda/etc/profile.d/conda.sh
conda activate /rstor/pi_hzamani_umass_edu/asalemi/priya/env

# Navigate to project directory
cd /rstor/pi_hzamani_umass_edu/asalemi/priya/diverseTextGen

# Run your code
python run_langgraph.py \
    --input_file ./data/antique/test.jsonl \
    -n 10 \
    --max_iterations 3 \
    --output ./output/batch_results
```

Submit with:
```bash
sbatch your_script.sh
```

---

## Configuration

### Key Configuration Files

**`config.py`** - Main configuration file with:
- Data directories
- Server URLs
- Model names
- Retrieval parameters
- LLM settings

### Environment Variables

You can override config values using environment variables:

```bash
export RAG_DATA_DIR="/path/to/data"
export RAG_CACHE_DIR="/path/to/cache"
export RAG_CORPUS_PATH="/path/to/corpus.jsonl"
export RAG_SERVER_LOG_FILE="../server_logs/log.txt"
export RAG_DEFAULT_MODEL="Qwen/Qwen3-4B-Instruct-2507"
export RAG_DEFAULT_MAX_ITERATIONS=3
export RAG_DEFAULT_TOP_K=5
```

### Server Configuration

The system reads vLLM server information from `../server_logs/log.txt` (parent folder):
- Line 1: Hostname (e.g., `gpu023` or `localhost`)
- Line 2: Port (e.g., `8000`)

To update:
```bash
echo "gpu023" > ../server_logs/log.txt
echo "8000" >> ../server_logs/log.txt
```

---

## Troubleshooting

### Common Issues

**1. Server Connection Error**
```
FileNotFoundError: Server log file not found
```
- **Solution:** Ensure `../server_logs/log.txt` exists with correct hostname and port

**2. Module Not Found**
```
ModuleNotFoundError: No module named 'langgraph'
```
- **Solution:** Activate conda environment and install dependencies:
  ```bash
  conda activate /rstor/pi_hzamani_umass_edu/asalemi/priya/env
  pip install -r requirements.txt
  ```

**3. GPU Not Available**
```
RuntimeError: CUDA out of memory
```
- **Solution:** Request more GPU memory or reduce batch size in config

**4. Corpus File Not Found**
```
FileNotFoundError: corpus_filtered_50.jsonl
```
- **Solution:** Check `CORPUS_PATH` in `config.py` or set `RAG_CORPUS_PATH` environment variable

**5. Slurm Job Pending**
- **Solution:** Check queue status: `squeue -u $USER`
- Try preemptible partition: `--partition=gpu-preempt`
- Reduce requested resources (memory, time)

### Checking Job Status

```bash
# Check your jobs
squeue -u $USER

# Check job details
scontrol show job <job_id>

# Cancel a job
scancel <job_id>
```

### Monitoring Resources

```bash
# Check GPU usage (in GPU session)
nvidia-smi

# Check memory usage
free -h

# Check disk space
df -h
```

---

## Quick Reference

### Most Common Commands

```bash
# Single query
python run_langgraph.py --query "Your question" --query_id "q001"

# Batch processing
python run_langgraph.py --input_file ./data/antique/test.jsonl -n 10

# GPU interactive session
srun --pty --partition=gpu --nodes=1 --gres=gpu:1 --mem=64G -c 4 -t 08:00:00 bash

# CPU interactive session
srun --pty --partition=cpu --nodes=1 --mem=32G -c 1 -t 08:00:00 bash
```

---

## Additional Resources

- **README.md** - Project overview and architecture
- **SOLUTION_DOCUMENT.md** - Complete implementation details
- **ICAT_INTEGRATION_GUIDE.md** - Evaluation setup guide
- **IMPLEMENTATION_SUMMARY.md** - Technical implementation summary

