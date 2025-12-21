# Quick Start: Comparing Experiments

## 1. Run Experiments with Different Iterations

```bash
# Baseline (0 iterations - direct LLM)
python run_baseline_experiment.py \
    --queries_path data/antique/train.jsonl \
    -n 100 \
    --description "baseline"

# 3 iterations
python run_full_experiment.py \
    --queries_path data/antique/train.jsonl \
    -n 100 \
    --max_iterations 3 \
    --description "3iter"

# 5 iterations
python run_full_experiment.py \
    --queries_path data/antique/train.jsonl \
    -n 100 \
    --max_iterations 5 \
    --description "5iter"

# 10 iterations
python run_full_experiment.py \
    --queries_path data/antique/train.jsonl \
    -n 100 \
    --max_iterations 10 \
    --description "10iter"

# Unlimited iterations (budget-constrained)
python run_full_experiment.py \
    --queries_path data/antique/train.jsonl \
    -n 100 \
    --no_iteration_limit \
    --walltime_budget 3600 \
    --description "unlimited"
```

## 2. List Available Runs

```bash
# Using the CLI tool
python compare_runs.py --list --experiments_dir output/experiments

# Or using the example script
python example_comparison.py
# Choose option 1
```

## 3. Generate Comparison Visualizations

### Option A: Using the CLI tool

```bash
python compare_runs.py \
    --experiments_dir output/experiments \
    --run_ids run_20231215_100000 run_20231215_110000 run_20231215_120000 \
    --output_dir output/comparisons/my_comparison/
```

### Option B: Using Python directly

```python
from eval.visualizer import ICATVisualizer

viz = ICATVisualizer("output/experiments")
viz.compare_runs(
    run_ids=["run_A", "run_B", "run_C"],
    output_dir="output/comparisons/test/",
    queries_per_page=50
)
```

### Option C: Using the example script

```bash
python example_comparison.py
# Choose option 2 to auto-find runs by description
```

## 4. View Results

The comparison generates:

1. **Paginated line plots**: `comparison_page_1.png`, `comparison_page_2.png`, ...
   - Shows Coverage, Factuality, F1 for each query across runs
   - Default: 50 queries per page

2. **Heatmap**: `comparison_heatmap.png`
   - Shows all queries across all runs in a dense format
   - Color intensity represents score (green=high, red=low)

3. **Summary statistics**:
   - `comparison_summary.json`: Detailed statistics
   - `comparison_summary.csv`: Table format

## Key Features

✓ Compare any number of runs (minimum 2)
✓ Only queries present in ALL runs are compared
✓ Flexible pagination (adjust `queries_per_page`)
✓ Automatic color coding for different runs
✓ Summary statistics with mean and standard deviation

## Tips

- Use consistent `--description` flags when running experiments
- Test with small N first (`-n 10`) before full runs
- Check `runs_index.json` for run IDs and metadata
- Baseline and RAG experiments are in separate directories by default

