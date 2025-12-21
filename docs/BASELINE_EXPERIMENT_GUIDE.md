# Experiment System Guide

## Overview

This system allows you to run experiments with varying iteration counts and compare their ICAT scores:

- **Baseline** (0 iterations): Direct LLM responses without RAG
- **RAG Experiments**: Multi-agent RAG with configurable iterations (3, 5, 10, unlimited)

## Running Experiments

### 1. Baseline Experiment (0 iterations)

```bash
# Test with 10 queries
python run_baseline_experiment.py --queries_path data/antique/train.jsonl -n 10 --description "baseline"

# Full training set
python run_baseline_experiment.py --queries_path data/antique/train.jsonl --description "baseline"

# Using SLURM
sbatch scripts/run_baseline.sh 10 "baseline_test"
sbatch scripts/run_baseline.sh all "baseline_full"
```

### 2. RAG Experiments (3, 5, 10 iterations)

```bash
# 3 iterations
python run_full_experiment.py --queries_path data/antique/train.jsonl --max_iterations 3 --description "3iter"

# 5 iterations
python run_full_experiment.py --queries_path data/antique/train.jsonl --max_iterations 5 --description "5iter"

# 10 iterations
python run_full_experiment.py --queries_path data/antique/train.jsonl --max_iterations 10 --description "10iter"

# Using SLURM
sbatch scripts/run_experiment.sh 10 "3iter_test" 3
sbatch scripts/run_experiment.sh all "5iter_full" 5
sbatch scripts/run_experiment.sh all "10iter_full" 10
```

### 3. Unlimited Iterations (Budget-Constrained)

```bash
# Use --no_iteration_limit flag with budget constraints
python run_full_experiment.py \
    --queries_path data/antique/train.jsonl \
    --no_iteration_limit \
    --walltime_budget 3600 \
    --description "max_budget"

# Or set max_iterations=999 explicitly
python run_full_experiment.py \
    --queries_path data/antique/train.jsonl \
    --max_iterations 999 \
    --walltime_budget 7200 \
    --description "unlimited_2hr"
```

**Budget Options:**
- `--walltime_budget SECONDS`: Maximum runtime per query in seconds
- `--token_budget TOKENS`: Maximum tokens per query (requires token counting implementation)

## Output Structure

```
output/
├── baseline_experiments/       # Baseline runs (0 iterations)
│   ├── runs_index.json
│   ├── query_history.json
│   └── run_YYYYMMDD_HHMMSS/
│       ├── metadata.json
│       ├── results.jsonl
│       └── summary.json
│
└── experiments/                # RAG runs (3+, 5+, 10+, unlimited iterations)
    ├── runs_index.json
    ├── query_history.json
    └── run_YYYYMMDD_HHMMSS/
        ├── metadata.json
        ├── results.jsonl
        ├── summary.json
        └── rag_outputs/        # Individual query outputs
```

## Comparing Runs

### Using Python

```python
from eval.visualizer import ICATVisualizer

# Initialize visualizer with the experiment directory
viz = ICATVisualizer("output/experiments")

# Compare multiple runs
viz.compare_runs(
    run_ids=[
        "run_20231215_100000",  # baseline
        "run_20231215_110000",  # 3 iterations
        "run_20231215_120000",  # 5 iterations
        "run_20231215_130000",  # 10 iterations
        "run_20231215_140000"   # unlimited
    ],
    output_dir="output/comparisons/baseline_vs_rag/",
    queries_per_page=50
)
```

### Compare Baseline vs RAG

To compare baseline with RAG experiments, you need to use both experiment directories:

```python
from eval.visualizer import ICATVisualizer
from pathlib import Path
import json

# Load run IDs from both directories
baseline_dir = Path("output/baseline_experiments")
rag_dir = Path("output/experiments")

# Get baseline run ID
with open(baseline_dir / "runs_index.json") as f:
    baseline_runs = json.load(f)["runs"]
    baseline_id = baseline_runs[0]["run_id"]  # or select specific one

# Get RAG run IDs
with open(rag_dir / "runs_index.json") as f:
    rag_runs = json.load(f)["runs"]
    rag_3iter_id = [r["run_id"] for r in rag_runs if "3iter" in r.get("description", "")][0]
    rag_5iter_id = [r["run_id"] for r in rag_runs if "5iter" in r.get("description", "")][0]

# Note: You'll need to merge the data or run visualizations separately for each directory
# The current implementation compares runs within the same experiment directory
```

## Generated Visualizations

The `compare_runs()` method generates:

### 1. Paginated Line Plots (`comparison_page_N.png`)
- **X-axis**: Query IDs (paginated, 50 per page by default)
- **Y-axis**: ICAT scores (0-1)
- **Three subplots**: Coverage, Factuality, F1 (stacked vertically)
- **Lines**: One line per run, color-coded
- **Purpose**: See how each query performs across different iteration counts

### 2. Heatmap (`comparison_heatmap.png`)
- **Three heatmaps**: Coverage, Factuality, F1
- **X-axis**: Run IDs
- **Y-axis**: Query IDs (all queries)
- **Color**: Score intensity (green=high, red=low)
- **Purpose**: Quickly identify patterns and outliers across all queries

### 3. Summary Statistics
- **JSON** (`comparison_summary.json`): Detailed statistics
- **CSV** (`comparison_summary.csv`): Table-friendly format
- **Metrics**: Average and standard deviation for Coverage, Factuality, F1

## Example Workflow

### Step 1: Run All Experiments

```bash
# Baseline
sbatch scripts/run_baseline.sh all "baseline"

# 3 iterations
sbatch scripts/run_experiment.sh all "3iter" 3

# 5 iterations
sbatch scripts/run_experiment.sh all "5iter" 5

# 10 iterations
sbatch scripts/run_experiment.sh all "10iter" 10

# Monitor progress
squeue -u $USER
```

### Step 2: List Available Runs

```bash
# Check baseline runs
cat output/baseline_experiments/runs_index.json | jq '.runs[] | {run_id, description, status}'

# Check RAG runs
cat output/experiments/runs_index.json | jq '.runs[] | {run_id, description, status}'
```

### Step 3: Generate Comparisons

Create a Python script (`compare_experiments.py`):

```python
from eval.visualizer import ICATVisualizer

# Get run IDs (replace with your actual run IDs)
run_ids = [
    "run_20231215_100000",  # 3iter
    "run_20231215_110000",  # 5iter
    "run_20231215_120000",  # 10iter
]

viz = ICATVisualizer("output/experiments")
viz.compare_runs(
    run_ids=run_ids,
    output_dir="output/comparisons/iter_comparison/",
    queries_per_page=50
)

print("Comparison complete! Check output/comparisons/iter_comparison/")
```

Run it:

```bash
python compare_experiments.py
```

### Step 4: Analyze Results

```bash
# View summary
cat output/comparisons/iter_comparison/comparison_summary.json | jq '.'

# View images
# - comparison_page_1.png, comparison_page_2.png, ...
# - comparison_heatmap.png
```

## Tips

1. **Use consistent descriptions**: Makes it easier to identify runs later
2. **Test with small N first**: Use `-n 10` to verify setup before full runs
3. **Monitor memory**: Unlimited iterations may consume significant memory
4. **Resume capability**: Both scripts support `--resume_from INDEX` to continue interrupted runs
5. **Same queries**: Only queries that appear in ALL selected runs will be compared

## Performance Expectations

| Iteration Count | Time per Query | Total Time (2,426 queries) |
|----------------|----------------|----------------------------|
| Baseline (0)   | 30-60s         | 20-40 hours               |
| 3 iterations   | 2-3 min        | 80-120 hours              |
| 5 iterations   | 3-4 min        | 120-160 hours             |
| 10 iterations  | 5-7 min        | 200-280 hours             |
| Unlimited      | Varies         | Depends on budget         |

## Troubleshooting

**Issue**: "No common queries found across runs"
- **Solution**: Ensure all runs used the same query file and completed successfully

**Issue**: Visualizations not generated
- **Solution**: Check that all run_ids are valid and exist in the experiments directory

**Issue**: Out of memory with unlimited iterations
- **Solution**: Use `--walltime_budget` to limit runtime per query
