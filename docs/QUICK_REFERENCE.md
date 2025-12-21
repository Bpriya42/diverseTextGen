# Quick Reference Card

## Start LLM Server
```bash
sbatch scripts/qwen_3_instrcut_4b.sh
cat server_logs/log.txt  # Check hostname and port
```

## Run Experiments

### Baseline (No RAG)
```bash
sbatch scripts/run_baseline.sh all "baseline_full"
```

### Fixed Iterations
```bash
sbatch scripts/run_experiment.sh all "3iter" 3
sbatch scripts/run_experiment.sh all "5iter" 5
sbatch scripts/run_experiment.sh all "10iter" 10
```

### Unlimited (Until Memory/Budget Exhausted)
```bash
sbatch scripts/run_experiment.sh all "unlimited_2h" 999 --no_iteration_limit --walltime_budget 7200
```

## Monitor Jobs
```bash
squeue -u $USER                          # Check job status
watch -n 10 'squeue -u $USER'           # Real-time monitoring
tail -f server_logs/experiment_*.out     # View live output
scancel <JOBID>                          # Kill a job
```

## Compare Results
```bash
# List all runs
python compare_runs.py --list_runs

# Compare runs
python compare_runs.py \
    --run_ids run_A run_B run_C \
    --output_dir output/comparisons/my_comparison
```

## Output Locations
- Baseline: `output/baseline_experiments/`
- RAG: `output/experiments/`
- Logs: `server_logs/`
- Comparisons: `output/comparisons/`

## Resource Allocations
| Script | Time | Memory | CPUs |
|--------|------|--------|------|
| run_baseline.sh | 24h | 64GB | 8 |
| run_experiment.sh | 24h | 256GB | 16 |

## Common Arguments
- `all` - Run all 2,426 queries
- `-n 10` - Run first 10 queries only
- `--no_iteration_limit` - Run until budget exhausted
- `--walltime_budget 7200` - 2 hours per query
- `--token_budget 100000` - 100k tokens per query



