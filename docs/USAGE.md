# Usage Guide

This guide covers how to run experiments with the DiverseTextGen RAG system.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Running Single Queries](#running-single-queries)
3. [Running Experiments](#running-experiments)
4. [Command Reference](#command-reference)
5. [Monitoring and Output](#monitoring-and-output)

---

## Quick Start

### Prerequisites

1. **Python Environment**: Python 3.10+ with dependencies installed
2. **vLLM Server**: Running LLM server (see setup below)
3. **Data**: Corpus and queries in `data/` directory

### Setup vLLM Server

**Start the server:**

```bash
# Using SLURM (recommended for cluster)
sbatch scripts/shell/start_server.sh

# Or directly
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --host 0.0.0.0 \
    --port 8000
```

**Verify server is running:**

```bash
# Check SLURM job
squeue -u $USER

# Check server log
cat server_logs/log.txt
# Should show:
# hostname
# port_number
```

### First Run (5 minutes)

```bash
cd /rstor/pi_hzamani_umass_edu/asalemi/priya/diverseTextGen

python scripts/run_langgraph.py \
    --query "What causes headaches and how can they be treated?" \
    --query_id "test_001" \
    --output ./output/test_001.json
```

**Expected output:**
```
Building LangGraph workflow...
[Graph] Using in-memory checkpointer

================================================================================
STARTING ITERATIVE RAG SYSTEM
================================================================================
Query ID: test_001
Query: What causes headaches and how can they be treated?
Mode: Quality-controlled (memory-bounded)
Memory Limits: RAM 90%, GPU 90%
================================================================================

--------------------------------------------------------------------------------
ITERATION 1
--------------------------------------------------------------------------------
[Planner] Processing iteration 0
[Planner] Generated initial plan with 5 aspects
...
[Iteration Gate] ✓ Quality complete - no further improvements needed - TERMINATING
```

**Check results:**
```bash
cat output/test_001.json | jq .
```

---

## Running Single Queries

### Basic Usage

```bash
python scripts/run_langgraph.py \
    --query "Your question here" \
    --query_id "unique_id"
```

### With Custom Memory Limits

```bash
python scripts/run_langgraph.py \
    --query "Complex medical question" \
    --query_id "q001" \
    --max_ram_percent 85 \
    --max_gpu_percent 85 \
    --output ./output/q001_result.json
```

### Example Queries

**Medical:**
```bash
python scripts/run_langgraph.py \
    --query "What are the symptoms and treatment options for rheumatoid arthritis?" \
    --query_id "medical_001"
```

**Technical:**
```bash
python scripts/run_langgraph.py \
    --query "How does a neural network learn from backpropagation?" \
    --query_id "tech_001"
```

**General:**
```bash
python scripts/run_langgraph.py \
    --query "What are the main causes of climate change?" \
    --query_id "general_001"
```

---

## Running Experiments

### Batch Processing from JSONL

**Run on first 10 queries:**

```bash
python scripts/run_langgraph.py \
    --input_file ./data/antique/train.jsonl \
    -n 10 \
    --output ./output/batch_10
```

**Run on first 100 queries:**

```bash
python scripts/run_langgraph.py \
    --input_file ./data/antique/train.jsonl \
    -n 100 \
    --output ./output/batch_100
```

**Run on all queries (2,426):**

```bash
python scripts/run_langgraph.py \
    --input_file ./data/antique/train.jsonl \
    --output ./output/full_training
```

### Full Experiment with ICAT Evaluation

**Interactive mode:**

```bash
python scripts/run_full_experiment.py \
    --queries_path data/antique/train.jsonl \
    -n 100 \
    --description "baseline_100"
```

**SLURM batch mode:**

```bash
# Test with 10 queries
sbatch scripts/shell/run_experiment.sh 10 "test_run"

# Run on 100 queries
sbatch scripts/shell/run_experiment.sh 100 "batch_100"

# Run on all queries
sbatch scripts/shell/run_experiment.sh all "full_training"
```

**What happens:**
1. Runs RAG system on each query
2. Evaluates each answer with ICAT (Coverage, Factuality, F1)
3. Tracks all scores automatically
4. Generates visualizations
5. Saves detailed results

### Baseline Experiment (No RAG)

Run queries directly to LLM without retrieval:

```bash
# Interactive
python scripts/run_baseline_experiment.py \
    --queries_path data/antique/train.jsonl \
    -n 100

# SLURM
sbatch scripts/shell/run_baseline.sh 100 "baseline_100"
```

---

## Command Reference

### run_langgraph.py

**Main RAG runner for single queries or batches.**

```bash
python scripts/run_langgraph.py [OPTIONS]
```

**Single Query Mode:**

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `--query` | str | Yes* | Query text |
| `--query_id` | str | Yes* | Unique identifier |
| `--output` | str | No | Output file path (default: `output/result.json`) |
| `--max_ram_percent` | float | No | RAM limit % (default: 90) |
| `--max_gpu_percent` | float | No | GPU memory limit % (default: 90) |

*Required for single query mode

**Batch Mode:**

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `--input_file` | str | Yes* | JSONL file with queries |
| `-n` | int | No | Number of queries to process (default: all) |
| `--output` | str | No | Output directory (default: `output/batch`) |

*Required for batch mode

**Examples:**

```bash
# Single query
python scripts/run_langgraph.py \
    --query "What causes headaches?" \
    --query_id "q001"

# Batch with memory limits
python scripts/run_langgraph.py \
    --input_file data/queries.jsonl \
    -n 50 \
    --max_ram_percent 85 \
    --max_gpu_percent 80 \
    --output output/batch_50
```

### run_full_experiment.py

**Run experiments with automatic ICAT evaluation and tracking.**

```bash
python scripts/run_full_experiment.py [OPTIONS]
```

**Arguments:**

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `--queries_path` | str | Yes | Path to JSONL file with queries |
| `-n` | int | No | Number of queries (default: all) |
| `--description` | str | No | Experiment description |
| `--run_id` | str | No | Resume existing run |
| `--resume_from` | int | No | Resume from query index |

**Examples:**

```bash
# Test run
python scripts/run_full_experiment.py \
    --queries_path data/antique/train.jsonl \
    -n 10 \
    --description "test"

# Resume interrupted experiment
python scripts/run_full_experiment.py \
    --queries_path data/antique/train.jsonl \
    --run_id run_20241231_120000 \
    --resume_from 150
```

### run_baseline_experiment.py

**Run baseline (no RAG) for comparison.**

```bash
python scripts/run_baseline_experiment.py [OPTIONS]
```

**Arguments:**

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `--queries_path` | str | Yes | Path to JSONL queries |
| `-n` | int | No | Number of queries |

**Example:**

```bash
python scripts/run_baseline_experiment.py \
    --queries_path data/antique/train.jsonl \
    -n 100
```

### evaluate_icat.py

**Evaluate existing outputs with ICAT.**

```bash
python scripts/evaluate_icat.py [OPTIONS]
```

**Arguments:**

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `--output_path` | str | Yes | Directory with results |
| `--corpus_path` | str | Yes | Path to corpus JSONL |

**Example:**

```bash
python scripts/evaluate_icat.py \
    --output_path output/batch_100 \
    --corpus_path data/antique/corpus_filtered_50.jsonl
```

---

## Monitoring and Output

### Checking Progress

**Check SLURM job status:**

```bash
# View your jobs
squeue -u $USER

# Watch in real-time
watch -n 10 'squeue -u $USER'

# View job output
tail -f server_logs/experiment_*.out
```

**Check completed queries:**

```bash
# Count completed queries
wc -l output/experiments/run_*/results.jsonl

# View latest result
tail -1 output/experiments/run_*/results.jsonl | jq .
```

### Output Structure

**Single Query Result** (`output/result.json`):

```json
{
  "query_id": "q001",
  "query": "What causes headaches?",
  "final_answer": "...",
  "total_iterations": 3,
  "termination_reason": "quality_complete",
  "iteration_history": [
    {
      "iteration": 0,
      "plan": [...],
      "answer": "...",
      "timestamp": 1704067200.0
    }
  ],
  "total_runtime_seconds": 45.2,
  "timestamps": {
    "planner_iter0": 2.1,
    "retriever_iter0": 1.5,
    ...
  },
  "memory_config": {
    "max_ram_percent": 90.0,
    "max_gpu_percent": 90.0
  }
}
```

**Experiment Output Structure:**

```
output/experiments/
├── run_20241231_120000/          # Run directory
│   ├── config.json               # Experiment configuration
│   ├── results.jsonl             # Per-query results (streaming)
│   ├── summary.json              # Aggregate statistics
│   └── detailed_results/         # Individual query JSONs
│       ├── q001.json
│       ├── q002.json
│       └── ...
├── query_history.json            # Per-query across all runs
├── runs_index.json               # All runs metadata
└── visualizations/               # Generated plots
    ├── aggregate_trends.png
    └── query_tracking.png
```

**Result Fields:**

- `query_id`: Unique query identifier
- `query`: Original query text
- `final_answer`: Generated answer
- `total_iterations`: Number of iterations
- `termination_reason`: Why system stopped
  - `quality_complete`: Quality criteria met
  - `memory_exceeded: ...`: Memory limit reached
- `iteration_history`: Per-iteration snapshots
- `total_runtime_seconds`: Total execution time
- `timestamps`: Per-component timing

### Viewing Results

**View single result:**

```bash
cat output/result.json | jq .
```

**View experiment summary:**

```bash
cat output/experiments/run_*/summary.json | jq .
```

**Extract specific fields:**

```bash
# Get all final answers
cat output/batch_100/*.json | jq -r '.final_answer' > answers.txt

# Get termination reasons
cat output/batch_100/*.json | jq -r '.termination_reason' | sort | uniq -c

# Get average iterations
cat output/batch_100/*.json | jq '.total_iterations' | \
    awk '{sum+=$1} END {print sum/NR}'
```

### ICAT Scores

**View ICAT evaluation:**

```bash
# From experiment
cat output/experiments/run_*/results.jsonl | jq '{query_id, coverage, factuality, f1}'

# View summary statistics
cat output/experiments/run_*/summary.json | jq '.aggregate_stats'
```

**ICAT Metrics:**

- **Coverage**: How well the answer covers the query (0-1)
- **Factuality**: Factual correctness based on corpus (0-1)
- **F1**: Harmonic mean of coverage and factuality

### Visualizations

**Generated automatically in experiments:**

1. `aggregate_trends.png`: Coverage, Factuality, F1 over runs
2. `query_tracking.png`: Per-query F1 scores across runs

**View on local machine:**

```bash
# Copy from cluster
scp user@cluster:path/to/output/experiments/visualizations/*.png .

# Open
open aggregate_trends.png
```

### Logs

**LLM Decision Log** (`artifacts/llm_decisions.jsonl`):

Tracks all LLM decisions (see [OBSERVABILITY.md](OBSERVABILITY.md)).

**Error Logs:**

```bash
# Check for errors
grep -i error server_logs/*.out

# View specific log
cat server_logs/experiment_20241231.out
```

---

## Tips and Best Practices

### 1. Start Small

Always test with `-n 10` before running on full dataset:

```bash
# Test
python scripts/run_full_experiment.py \
    --queries_path data/antique/train.jsonl \
    -n 10 \
    --description "test_run"
```

### 2. Monitor Memory

Watch RAM/GPU usage to tune thresholds:

```bash
# On compute node
watch -n 5 'nvidia-smi; free -h'
```

### 3. Resume Interrupted Runs

If a run is interrupted, resume from last checkpoint:

```bash
# Check progress
wc -l output/experiments/run_*/results.jsonl

# Resume
python scripts/run_full_experiment.py \
    --queries_path data/antique/train.jsonl \
    --run_id run_20241231_120000 \
    --resume_from 150
```

### 4. Compare Runs

Use baseline to measure RAG improvement:

```bash
# Run baseline first
python scripts/run_baseline_experiment.py \
    --queries_path data/antique/train.jsonl \
    -n 100

# Then RAG
python scripts/run_full_experiment.py \
    --queries_path data/antique/train.jsonl \
    -n 100 \
    --description "rag_100"
```

### 5. Batch Processing

Process large datasets in chunks:

```bash
# Queries 0-499
python scripts/run_langgraph.py \
    --input_file data/queries.jsonl \
    -n 500 \
    --output output/batch_0_499

# Queries 500-999
# (manually skip first 500 lines)
tail -n +501 data/queries.jsonl | head -500 > temp_queries.jsonl
python scripts/run_langgraph.py \
    --input_file temp_queries.jsonl \
    --output output/batch_500_999
```

---

## Troubleshooting

### Server not found

```
Error: Server log file not found
```

**Solution:**
```bash
# Ensure server is running
cat server_logs/log.txt

# If not running, start it
sbatch scripts/shell/start_server.sh
```

### Out of memory

```
[Iteration Gate] Memory limit reached: RAM usage at 92%
```

**Solution:**
```bash
# Lower memory thresholds
python scripts/run_langgraph.py \
    --query "..." \
    --query_id "..." \
    --max_ram_percent 80 \
    --max_gpu_percent 80
```

### Slow performance

- **Check server**: Ensure vLLM server is on GPU node
- **Reduce top-k**: Lower `RAG_DEFAULT_TOP_K` in config
- **Limit iterations**: System runs until quality goals met

### Import errors

```
ModuleNotFoundError: No module named 'llm_observability'
```

**Solution:**
```bash
# Ensure running from project root
cd /rstor/pi_hzamani_umass_edu/asalemi/priya/diverseTextGen
python scripts/run_langgraph.py ...
```

---

For architecture details, see [ARCHITECTURE.md](ARCHITECTURE.md).  
For LLM decision tracking, see [OBSERVABILITY.md](OBSERVABILITY.md).

