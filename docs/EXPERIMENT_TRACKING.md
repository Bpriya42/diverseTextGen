# Experiment Tracking & Visualization System

This document describes the experiment tracking and visualization system for evaluating ICAT scores across multiple experimental runs.

## Overview

The system consists of three main components:

1. **ExperimentTracker** (`eval/experiment_tracker.py`) - Tracks ICAT scores across runs
2. **ICATVisualizer** (`eval/visualizer.py`) - Creates visualizations of tracked data
3. **run_full_experiment.py** - Orchestrates experiments end-to-end

## Features

✅ **Automatic Tracking**: Stores ICAT scores for every query in every run  
✅ **Historical Data**: Maintains complete history of all queries across all runs  
✅ **Resumable**: Can resume interrupted experiments  
✅ **Visualizations**: Automatic generation of trend plots and comparisons  
✅ **Integrated**: Works seamlessly with existing RAG and ICAT systems  

---

## Quick Start

### 1. Run a Test Experiment (10 queries)

```bash
python run_full_experiment.py \
    --queries_path data/antique/train.jsonl \
    -n 10 \
    --description "initial_test" \
    --max_iterations 3
```

### 2. Run Full Training Set

```bash
python run_full_experiment.py \
    --queries_path data/antique/train.jsonl \
    --description "baseline_full_training" \
    --max_iterations 3
```

### 3. Using SLURM

```bash
sbatch scripts/run_experiment.sh 10 "test_run"
sbatch scripts/run_experiment.sh all "full_baseline"
```

---

## Directory Structure

After running experiments, you'll have:

```
.diverseTextGen/output/experiments/
├── runs_index.json              # Index of all runs
├── query_history.json           # Historical scores for each query
├── run_20231214_143022/         # First run
│   ├── metadata.json             # Run configuration
│   ├── results.jsonl             # Per-query results
│   ├── summary.json              # Aggregate statistics
│   └── rag_outputs/              # Individual RAG outputs
│       ├── 3097310.json
│       └── ...
├── run_20231214_150045/         # Second run
│   └── ...
└── visualizations/              # Generated plots
    ├── aggregate_trends.png
    └── query_tracking.png
```

---

## File Descriptions

### Core Files

**`eval/experiment_tracker.py`**
- Manages experiment metadata and results
- Tracks per-query ICAT scores across runs
- Maintains historical data for all queries
- Provides comparison utilities

**`eval/visualizer.py`**
- Creates aggregate trend plots
- Generates per-query tracking visualizations
- Supports custom visualizations

**`run_full_experiment.py`**
- Main experiment orchestration script
- Runs RAG system on queries
- Evaluates with ICAT
- Tracks results automatically
- Generates visualizations

**`scripts/run_experiment.sh`**
- SLURM batch script for running experiments
- Convenient interface for common use cases

---

## Detailed Usage

### Running Experiments

#### Basic Usage

```bash
python run_full_experiment.py \
    --queries_path data/antique/train.jsonl \
    -n 100 \
    --description "baseline_run"
```

#### All Arguments

```bash
python run_full_experiment.py \
    --queries_path PATH           # Path to JSONL with queries
    --run_id RUN_ID              # Unique identifier (auto-generated if omitted)
    --description DESC           # Human-readable description
    -n N                         # Number of queries (default: all)
    --max_iterations N           # Max iterations per query (default: 3)
    --corpus_path PATH           # Corpus file path
    --experiments_dir DIR        # Where to save experiments
    --resume_from INDEX          # Resume from query index (0-based)
```

#### Resume Interrupted Experiment

If an experiment is interrupted, you can resume:

```bash
python run_full_experiment.py \
    --queries_path data/antique/train.jsonl \
    --run_id run_20231214_143022 \
    --resume_from 50
```

---

### Visualization

#### Generate Visualizations

Visualizations are automatically generated after each experiment. To regenerate manually:

```python
from eval.visualizer import ICATVisualizer

viz = ICATVisualizer(".diverseTextGen/output/experiments")
viz.generate_all_visualizations(".diverseTextGen/output/experiments/visualizations")
```

#### Available Plots

1. **Aggregate Trends** (`aggregate_trends.png`)
   - Coverage, Factuality, and F1 scores across all runs
   - Shows progression over time
   - Includes all metrics in one combined plot

2. **Query Tracking** (`query_tracking.png`)
   - Individual F1 scores for 20 sample queries
   - Shows how specific queries perform across runs
   - Useful for identifying consistent patterns

---

### Programmatic Access

#### Using ExperimentTracker

```python
from eval.experiment_tracker import ExperimentTracker

# Initialize tracker
tracker = ExperimentTracker(".diverseTextGen/output/experiments")

# Create a new run
run_id = tracker.create_run(
    run_id="my_experiment",
    config={"max_iterations": 5},
    description="Testing with 5 iterations"
)

# Log query results
tracker.log_query_result(
    run_id=run_id,
    query_id="3097310",
    query="What causes severe swelling and pain in the knees?",
    icat_scores={
        "coverage": 0.85,
        "factuality": 0.90,
        "f1": 0.875
    },
    rag_metrics={
        "total_iterations": 3,
        "runtime_seconds": 12.5
    }
)

# Finalize run
tracker.finalize_run(run_id, status="completed")

# Get query history
history = tracker.get_query_history("3097310")
print(f"Query has {len(history['runs'])} runs")

# Compare runs
comparison = tracker.compare_runs(["run_1", "run_2"])
print(f"Common queries: {len(comparison['common_queries'])}")
```

#### Using ICATVisualizer

```python
from eval.visualizer import ICATVisualizer

viz = ICATVisualizer(".diverseTextGen/output/experiments")

# Plot aggregate trends for specific runs
viz.plot_aggregate_trends(
    "my_trends.png",
    run_ids=["run_1", "run_2", "run_3"]
)

# Track specific queries
viz.plot_query_tracking(
    "specific_queries.png",
    query_ids=["3097310", "3910705", "237390"]
)
```

---

## Data Formats

### runs_index.json

```json
{
  "runs": [
    {
      "run_id": "run_20231214_143022",
      "timestamp": "2023-12-14T14:30:22",
      "description": "baseline experiment",
      "config": {
        "n_queries": 100,
        "max_iterations": 3
      },
      "status": "completed",
      "aggregate_stats": {
        "avg_coverage": 0.65,
        "avg_factuality": 0.72,
        "avg_f1": 0.68
      }
    }
  ]
}
```

### query_history.json

```json
{
  "3097310": {
    "query": "What causes severe swelling and pain in the knees?",
    "runs": [
      {
        "run_id": "run_20231214_143022",
        "timestamp": "2023-12-14T14:30:25",
        "icat_scores": {
          "coverage": 0.85,
          "factuality": 0.90,
          "f1": 0.875
        },
        "rag_metrics": {
          "total_iterations": 3,
          "runtime_seconds": 12.5
        }
      }
    ]
  }
}
```

### results.jsonl (per run)

Each line is a JSON object:

```json
{
  "query_id": "3097310",
  "query": "What causes severe swelling and pain in the knees?",
  "icat_scores": {
    "coverage": 0.85,
    "factuality": 0.90,
    "f1": 0.875
  },
  "rag_metrics": {
    "total_iterations": 3,
    "runtime_seconds": 12.5
  },
  "timestamp": "2023-12-14T14:30:25"
}
```

---

## Examples

### Example 1: Baseline Run

```bash
# Run baseline with 3 iterations on 100 queries
python run_full_experiment.py \
    --queries_path data/antique/train.jsonl \
    -n 100 \
    --max_iterations 3 \
    --description "baseline_3iter"
```

### Example 2: Testing Different Iterations

```bash
# Run 1: 2 iterations
python run_full_experiment.py \
    --queries_path data/antique/train.jsonl \
    -n 50 \
    --max_iterations 2 \
    --description "test_2_iterations"

# Run 2: 5 iterations
python run_full_experiment.py \
    --queries_path data/antique/train.jsonl \
    -n 50 \
    --max_iterations 5 \
    --description "test_5_iterations"

# Compare results in visualizations
```

### Example 3: Full Training Evaluation

```bash
# Submit as SLURM job for all 2,426 queries
sbatch scripts/run_experiment.sh all "full_training_baseline"

# Monitor job
squeue -u $USER
tail -f server_logs/experiment_*.out
```

---

## Best Practices

### 1. Descriptive Run IDs and Descriptions

Use meaningful descriptions to track what you're testing:

```bash
python run_full_experiment.py \
    --queries_path data/antique/train.jsonl \
    --description "baseline_qwen3_4b_3iter" \
    -n 100
```

### 2. Start Small, Scale Up

Test with a small number of queries first:

```bash
# Test with 10 queries
python run_full_experiment.py --queries_path data/antique/train.jsonl -n 10 --description "test"

# If successful, scale to 100
python run_full_experiment.py --queries_path data/antique/train.jsonl -n 100 --description "medium_test"

# Finally, full dataset
python run_full_experiment.py --queries_path data/antique/train.jsonl --description "full_run"
```

### 3. Use Resume for Long Runs

For very long experiments, use checkpointing:

```bash
# If interrupted at query 150
python run_full_experiment.py \
    --queries_path data/antique/train.jsonl \
    --run_id run_20231214_143022 \
    --resume_from 150
```

### 4. Regular Visualization Updates

After adding new runs, regenerate visualizations:

```python
from eval.visualizer import ICATVisualizer
viz = ICATVisualizer(".diverseTextGen/output/experiments")
viz.generate_all_visualizations(".diverseTextGen/output/experiments/visualizations")
```

---

## Troubleshooting

### Issue: Experiment fails partway through

**Solution**: Use `--resume_from` to continue from where it stopped

```bash
# Check the results.jsonl to see how many queries completed
wc -l .diverseTextGen/output/experiments/run_*/results.jsonl

# Resume from that point
python run_full_experiment.py \
    --run_id YOUR_RUN_ID \
    --resume_from LAST_INDEX \
    --queries_path data/antique/train.jsonl
```

### Issue: Out of memory

**Solution**: Reduce batch sizes or process fewer queries at once

```bash
# Process in smaller batches
python run_full_experiment.py -n 50 --description "batch_1"
python run_full_experiment.py -n 50 --description "batch_2"
# etc.
```

### Issue: Visualizations not generating

**Solution**: Ensure matplotlib, seaborn, and pandas are installed

```bash
pip install matplotlib seaborn pandas
```

---

## Advanced Usage

### Custom Visualization

Create your own visualizations using the data:

```python
import json
from pathlib import Path
import matplotlib.pyplot as plt

# Load query history
with open(".diverseTextGen/output/experiments/query_history.json", 'r') as f:
    history = json.load(f)

# Create custom plot
query_id = "3097310"
runs = history[query_id]["runs"]
f1_scores = [r["icat_scores"]["f1"] for r in runs]

plt.figure(figsize=(10, 6))
plt.plot(f1_scores, marker='o')
plt.title(f"F1 Score Progression for Query {query_id}")
plt.xlabel("Run")
plt.ylabel("F1 Score")
plt.grid(True)
plt.savefig("custom_plot.png")
```

### Comparing Specific Runs

```python
from eval.experiment_tracker import ExperimentTracker

tracker = ExperimentTracker(".diverseTextGen/output/experiments")
comparison = tracker.compare_runs(["run_1", "run_2"])

print(f"Common queries: {len(comparison['common_queries'])}")

# Analyze improvements
for query_id in comparison['common_queries'][:10]:
    scores = comparison['per_query_comparison'][query_id]['scores_by_run']
    run1_f1 = scores['run_1']['f1']
    run2_f1 = scores['run_2']['f1']
    improvement = run2_f1 - run1_f1
    print(f"{query_id}: {improvement:+.3f}")
```

---

## Integration with Existing Scripts

The experiment tracking system integrates with your existing infrastructure:

- Uses the same `run_langgraph.py` for RAG execution
- Uses the same `eval/icat.py` for ICAT evaluation
- Respects all configuration in `config.py`
- Saves outputs in the standard `.diverseTextGen/output/` directory

---

## Next Steps

1. **Run a test experiment** with 10 queries to validate setup
2. **Review visualizations** to understand the output format
3. **Run larger experiments** to gather more data
4. **Compare runs** to track improvements over time
5. **Customize visualizations** as needed for your analysis

For questions or issues, refer to the main project documentation or examine the source code in the `eval/` directory.

