# Running Experiments Guide

This guide shows you how to run baseline and RAG experiments using SLURM batch jobs.

## Prerequisites

1. **LLM Server Running**: You need a vLLM server running first
2. **Data Available**: Training queries at `data/antique/train.jsonl`
3. **Environment**: Python environment with all dependencies installed

## Step 1: Start the LLM Server

The vLLM server needs to run on a GPU node:

```bash
sbatch scripts/qwen_3_instrcut_4b.sh
```

**Check server status:**
```bash
# Check if job is running
squeue -u $USER

# Once running, check the hostname and port
cat server_logs/log.txt
```

The server will output:
```
hostname
port_number
```

## Step 2: Run Baseline Experiment

The baseline runs queries directly to the LLM without RAG (0 iterations):

```bash
# Run on all 2,426 queries
sbatch scripts/run_baseline.sh all "baseline_full_train"

# Or test with 10 queries first
sbatch scripts/run_baseline.sh 10 "baseline_test"
```

**Resources Allocated:**
- Time: 24 hours
- Memory: 64GB
- CPUs: 8 cores
- Partition: cpu

## Step 3: Run RAG Experiments

### Fixed Iterations

Run experiments with a specific number of iterations:

```bash
# 3 iterations
sbatch scripts/run_experiment.sh all "3iter" 3

# 5 iterations
sbatch scripts/run_experiment.sh all "5iter" 5

# 10 iterations
sbatch scripts/run_experiment.sh all "10iter" 10
```

### Unlimited Iterations (Budget-Controlled)

Run until memory maxes out or budget is exhausted:

```bash
# Run with 2-hour walltime budget per query
sbatch scripts/run_experiment.sh all "unlimited_2h" 999 --no_iteration_limit --walltime_budget 7200

# Run with token budget per query
sbatch scripts/run_experiment.sh all "unlimited_100k_tokens" 999 --no_iteration_limit --token_budget 100000

# Run with both budgets (whichever hits first stops)
sbatch scripts/run_experiment.sh all "unlimited_combined" 999 --no_iteration_limit --walltime_budget 3600 --token_budget 50000
```

**Resources Allocated for RAG Experiments:**
- Time: 24 hours (total job time)
- Memory: 256GB (to allow many iterations)
- CPUs: 16 cores
- Partition: cpu

## Step 4: Monitor Jobs

```bash
# View all your jobs
squeue -u $USER

# Watch jobs in real-time
watch -n 10 'squeue -u $USER'

# Check specific job output
tail -f server_logs/experiment_<JOBID>.out
tail -f server_logs/baseline_<JOBID>.out
```

## Step 5: Compare Results

After experiments complete, compare the runs:

```bash
# List all available runs
python compare_runs.py --list_runs

# Compare specific runs
python compare_runs.py \
    --run_ids run_20251214_123456 run_20251214_234567 run_20251214_345678 \
    --output_dir output/comparisons/baseline_vs_3iter_vs_unlimited \
    --queries_per_page 50
```

This generates:
- Paginated line plots comparing ICAT scores across runs
- Heatmaps showing all queries vs all runs
- Summary statistics (JSON and CSV)

## Understanding the Output

### Directory Structure

```
output/
├── baseline_experiments/
│   ├── run_20251214_123456/
│   │   ├── metadata.json       # Run configuration
│   │   ├── results.jsonl       # Per-query results with ICAT scores
│   │   └── summary.json        # Aggregate statistics
│   └── runs_index.json         # Index of all baseline runs
│
└── experiments/
    ├── run_20251214_234567/
    │   ├── metadata.json
    │   ├── results.jsonl
    │   └── summary.json
    ├── runs_index.json         # Index of all RAG runs
    └── query_history.json      # Per-query tracking across runs
```

### Results Files

Each `results.jsonl` contains per-query results:
```json
{
  "query_id": "123",
  "query": "What is...",
  "answer": "The answer is...",
  "icat_scores": {
    "coverage": 0.85,
    "factuality": 0.92,
    "f1": 0.88
  },
  "num_iterations": 5,
  "timestamp": "2025-12-14T12:34:56"
}
```

## Common Use Cases

### Test Run (Quick Validation)
```bash
# Test with 10 queries, 3 iterations
sbatch scripts/run_experiment.sh 10 "quick_test" 3
```

### Full Comparison Study
```bash
# 1. Baseline
sbatch scripts/run_baseline.sh all "baseline_full"

# 2. Fixed iterations
sbatch scripts/run_experiment.sh all "3iter" 3
sbatch scripts/run_experiment.sh all "5iter" 5
sbatch scripts/run_experiment.sh all "10iter" 10

# 3. Unlimited
sbatch scripts/run_experiment.sh all "unlimited" 999 --no_iteration_limit --walltime_budget 7200
```

### Memory Stress Test
```bash
# Run with minimal budget to see how many iterations fit in memory
sbatch scripts/run_experiment.sh 50 "memory_stress" 999 --no_iteration_limit --walltime_budget 600
```

## Troubleshooting

### Job Failed with "Out of Memory"
- Reduce `--mem` in the script or
- Reduce number of queries with `-n` flag or
- Add walltime budget to prevent too many iterations

### Server Connection Error
```bash
# Check if server is running
squeue -u $USER | grep qwen

# Check server logs
tail -f server_logs/vllm_server.out
tail -f server_logs/vllm_server.err

# Verify hostname and port
cat server_logs/log.txt
```

### Job Stuck in Queue (PD Status)
```bash
# Check why job is pending
squeue -j <JOBID> -o "%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %.20R"

# Common reasons:
# - Resources: Requested resources not available
# - Priority: Other jobs have higher priority
# - QOSMaxCpuPerUserLimit: You've hit user CPU limit
```

### Kill a Job
```bash
scancel <JOBID>
```

## Best Practices

1. **Start Small**: Test with 10 queries before running all 2,426
2. **Monitor Early**: Check the first few minutes of logs to catch errors early
3. **Stagger Submissions**: Don't submit all experiments at once; submit in stages
4. **Use Descriptive Names**: Use meaningful descriptions like "baseline_full_train" not "exp1"
5. **Check Resources**: Use `seff <JOBID>` after job completes to see resource usage
6. **Save Run IDs**: Note the run_ids from logs for later comparison

## Resource Planning

**For 2,426 queries:**

| Experiment Type | Iterations | Est. Time | Memory | Recommendation |
|----------------|-----------|-----------|--------|----------------|
| Baseline | 0 | 8-12 hours | 64GB | Use run_baseline.sh |
| Fixed 3-iter | 3 | 12-16 hours | 128GB | Safe for most clusters |
| Fixed 5-iter | 5 | 16-20 hours | 192GB | Recommended for comparison |
| Fixed 10-iter | 10 | 20-24 hours | 256GB | May hit time limit |
| Unlimited | Variable | 24 hours | 256GB | Use walltime_budget |

**Note**: Actual times depend on server response speed and query complexity.



