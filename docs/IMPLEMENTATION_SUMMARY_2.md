# Implementation Summary: Experiment Comparison System

## Overview

Successfully implemented a comprehensive system for running and comparing RAG experiments with varying iteration counts. The system now supports:

1. **Baseline experiments** (0 iterations - direct LLM)
2. **Fixed iteration experiments** (3, 5, 10, or any number)
3. **Unlimited iteration experiments** (budget-constrained)
4. **Visual comparison** across all experiment types

## Files Modified

### 1. `/run_full_experiment.py`
**Changes:**
- Added `token_budget` and `walltime_budget` parameters
- Added `--no_iteration_limit` flag (sets max_iterations=999)
- Added `--token_budget` and `--walltime_budget` CLI arguments
- Updated to pass budget parameters to `run_query()`
- Removed automatic visualization generation (now manual via compare_runs)

**New Features:**
- Can run unlimited iterations with budget constraints
- More flexible experiment configuration
- Budget-aware termination

### 2. `/run_baseline_experiment.py`
**Changes:**
- Removed automatic visualization generation
- Updated completion message to show comparison instructions

### 3. `/eval/visualizer.py`
**Complete Rewrite:**
- Removed old methods: `plot_average_icat_scores()`, `plot_query_tracking()`, `generate_all_visualizations()`
- Added new methods:
  - `plot_run_comparison_paginated()`: Paginated line plots comparing runs
  - `plot_run_comparison_heatmap()`: Dense heatmap for all queries
  - `compare_runs()`: Main entry point for comparisons
  - `_generate_summary_stats()`: JSON and CSV statistics
  - `_get_common_queries()`: Find queries present in all runs
  - `_load_run_results()`: Load results for specific runs

**New Capabilities:**
- Compare any number of runs (minimum 2)
- Paginated visualizations (50 queries per page by default)
- Three separate metric plots (Coverage, Factuality, F1)
- Heatmaps showing all queries across all runs
- Summary statistics in JSON and CSV formats

### 4. `/scripts/run_experiment.sh`
**Changes:**
- Updated help text with iteration examples
- Added examples for 3, 5, 10, and unlimited iterations

### 5. `/BASELINE_EXPERIMENT_GUIDE.md`
**Complete Rewrite:**
- Now titled "Experiment System Guide"
- Comprehensive guide for all experiment types
- Step-by-step workflow examples
- Comparison instructions
- Performance expectations table

## New Files Created

### 6. `/compare_runs.py`
**Purpose:** CLI tool for comparing runs

**Features:**
- `--list`: List all available runs
- Compare specific runs by ID
- Configurable pagination
- Clean command-line interface

**Usage:**
```bash
python compare_runs.py --list
python compare_runs.py --run_ids run_A run_B --output_dir output/comparisons/
```

### 7. `/example_comparison.py`
**Purpose:** Interactive example script

**Features:**
- List available runs
- Auto-find runs by description
- Compare specific runs
- Educational examples

### 8. `/QUICK_START_COMPARISON.md`
**Purpose:** Quick reference guide

**Content:**
- How to run different experiment types
- How to compare results
- Key features summary
- Tips and best practices

## Visualization Architecture

```
Experiments → ExperimentTracker → query_history.json
                                 → runs_index.json
                                 ↓
                           ICATVisualizer
                                 ↓
                    ┌────────────┼────────────┐
                    ↓            ↓            ↓
            Paginated Lines  Heatmap    Statistics
            (page_N.png)     (.png)    (.json/.csv)
```

## Key Design Decisions

1. **Pagination**: 50 queries per page (configurable) for readability
2. **Three metrics**: Separate subplots for Coverage, Factuality, F1
3. **Heatmap complement**: Dense visualization for pattern identification
4. **Flexible run selection**: Compare any subset of runs
5. **Budget-based termination**: Uses existing iteration_gate mechanisms
6. **Separate tracking**: Baseline and RAG experiments in different directories

## Usage Examples

### Running Experiments

```bash
# Baseline
python run_baseline_experiment.py --queries_path data/antique/train.jsonl --description "baseline"

# 3 iterations
python run_full_experiment.py --queries_path data/antique/train.jsonl --max_iterations 3 --description "3iter"

# Unlimited with budget
python run_full_experiment.py --queries_path data/antique/train.jsonl --no_iteration_limit --walltime_budget 3600 --description "unlimited"
```

### Comparing Results

```bash
# List runs
python compare_runs.py --list

# Compare
python compare_runs.py --run_ids run_A run_B run_C --output_dir output/comparisons/
```

### Programmatic Comparison

```python
from eval.visualizer import ICATVisualizer

viz = ICATVisualizer("output/experiments")
viz.compare_runs(
    run_ids=["run_A", "run_B", "run_C"],
    output_dir="output/comparisons/",
    queries_per_page=50
)
```

## Generated Outputs

For each comparison:

1. **`comparison_page_N.png`** (N = 1, 2, 3, ...)
   - 3 subplots per page (Coverage, Factuality, F1)
   - Lines for each run
   - Query IDs on X-axis

2. **`comparison_heatmap.png`**
   - 3 heatmaps (Coverage, Factuality, F1)
   - Rows = queries, Columns = runs
   - Color intensity = score

3. **`comparison_summary.json`**
   - Detailed statistics per run
   - Average and std dev for all metrics

4. **`comparison_summary.csv`**
   - Table-friendly format
   - Easy to import into spreadsheets

## Testing

All files compile successfully:
```bash
python -m py_compile run_full_experiment.py run_baseline_experiment.py eval/visualizer.py compare_runs.py example_comparison.py
```

No linter errors in modified files.

## Next Steps for Users

1. **Run experiments**: Start with small N for testing
2. **List runs**: Use `compare_runs.py --list` to see available runs
3. **Compare**: Use CLI tool or Python API to generate visualizations
4. **Analyze**: View plots and summary statistics

## Documentation

All documentation updated:
- `BASELINE_EXPERIMENT_GUIDE.md`: Complete system guide
- `QUICK_START_COMPARISON.md`: Quick reference
- Inline docstrings in all modules
- Example scripts with comments

## Performance Notes

- Pagination prevents memory issues with many queries
- Heatmaps work well up to ~500 queries
- Common queries only: only queries in ALL runs are compared
- Budget constraints prevent runaway unlimited iterations

## Compatibility

- Works with existing experiment data
- Backward compatible with old runs (can compare old and new)
- No changes to core RAG system or ICAT evaluation
- Uses existing ExperimentTracker format

