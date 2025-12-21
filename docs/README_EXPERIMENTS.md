# Quick Start Guide: Experiment Tracking

This is a quick reference for running experiments with ICAT tracking.

## Installation

First, install visualization dependencies:

```bash
pip install matplotlib seaborn pandas
```

## Running Experiments

### 1. Test Run (10 queries, ~5 minutes)

```bash
python run_full_experiment.py \
    --queries_path data/antique/train.jsonl \
    -n 10 \
    --description "test_run"
```

### 2. Small Batch (100 queries, ~30-60 minutes)

```bash
python run_full_experiment.py \
    --queries_path data/antique/train.jsonl \
    -n 100 \
    --description "small_batch"
```

### 3. Full Training Set (2,426 queries, ~12-24 hours)

Using SLURM:

```bash
sbatch scripts/run_experiment.sh all "full_baseline"
```

Or directly:

```bash
python run_full_experiment.py \
    --queries_path data/antique/train.jsonl \
    --description "full_baseline"
```

## Resuming Interrupted Experiments

If an experiment is interrupted:

```bash
# Find how many queries completed
wc -l .diverseTextGen/output/experiments/run_*/results.jsonl

# Resume from that point
python run_full_experiment.py \
    --queries_path data/antique/train.jsonl \
    --run_id run_20231214_143022 \
    --resume_from 150
```

## Generating Visualizations

Visualizations are generated automatically, but you can regenerate them:

```bash
python visualize_only.py --experiments_dir .diverseTextGen/output/experiments
```

## Viewing Results

### Check Experiment Status

```bash
# List all runs
ls -lh .diverseTextGen/output/experiments/

# View run summary
cat .diverseTextGen/output/experiments/run_*/summary.json | jq

# Count queries per run
wc -l .diverseTextGen/output/experiments/run_*/results.jsonl
```

### View Visualizations

```bash
# Open visualizations
open .diverseTextGen/output/experiments/visualizations/aggregate_trends.png
open .diverseTextGen/output/experiments/visualizations/query_tracking.png
```

## Common Use Cases

### Compare Different Configurations

```bash
# Run 1: 2 iterations
python run_full_experiment.py \
    --queries_path data/antique/train.jsonl \
    -n 100 \
    --max_iterations 2 \
    --description "test_2_iter"

# Run 2: 4 iterations
python run_full_experiment.py \
    --queries_path data/antique/train.jsonl \
    -n 100 \
    --max_iterations 4 \
    --description "test_4_iter"

# Visualize comparison
python visualize_only.py
```

### Track Specific Queries

```python
from eval.experiment_tracker import ExperimentTracker

tracker = ExperimentTracker(".diverseTextGen/output/experiments")
history = tracker.get_query_history("3097310")

for run in history["runs"]:
    print(f"{run['run_id']}: F1={run['icat_scores']['f1']:.3f}")
```

## Output Structure

```
.diverseTextGen/output/experiments/
├── runs_index.json              # All runs metadata
├── query_history.json           # Historical scores per query
├── run_20231214_143022/         # Individual run
│   ├── metadata.json
│   ├── results.jsonl
│   ├── summary.json
│   └── rag_outputs/
└── visualizations/
    ├── aggregate_trends.png
    └── query_tracking.png
```

## Troubleshooting

### Missing Dependencies

```bash
pip install matplotlib seaborn pandas
```

### SLURM Job Not Starting

```bash
# Check queue
squeue -u $USER

# Check job details
scontrol show job JOB_ID

# View logs
tail -f server_logs/experiment_*.out
```

### Out of Memory

Process in smaller batches:

```bash
python run_full_experiment.py -n 50 --description "batch_1"
python run_full_experiment.py -n 50 --description "batch_2"
```

## More Information

For detailed documentation, see: [EXPERIMENT_TRACKING.md](EXPERIMENT_TRACKING.md)

