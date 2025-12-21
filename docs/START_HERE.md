# 🚀 Experiment Tracking System - START HERE

## What's New?

A complete **experiment tracking and visualization system** has been added to your diverseTextGen project. This allows you to:

✅ Run experiments on **all 2,426 training queries**  
✅ Track **ICAT scores** (Coverage, Factuality, F1) for every query  
✅ **Resume** interrupted experiments  
✅ **Visualize** trends and improvements over time  
✅ **Compare** different experimental configurations  

---

## 🎯 Quick Start (5 Minutes)

### Step 1: Install Dependencies

```bash
pip install matplotlib seaborn pandas
```

### Step 2: Run Your First Experiment (10 queries)

```bash
cd /rstor/pi_hzamani_umass_edu/asalemi/priya/diverseTextGen

python run_full_experiment.py \
    --queries_path data/antique/train.jsonl \
    -n 10 \
    --description "my_first_test"
```

This will:
1. Run the RAG system on 10 queries
2. Evaluate each with ICAT
3. Track all scores automatically
4. Generate visualizations

### Step 3: View Results

```bash
# View summary
cat .diverseTextGen/output/experiments/run_*/summary.json | jq

# View visualizations (if on local machine with GUI)
open .diverseTextGen/output/experiments/visualizations/aggregate_trends.png
open .diverseTextGen/output/experiments/visualizations/query_tracking.png
```

---

## 📊 What Gets Tracked

For every query in every run:

| Metric | Description |
|--------|-------------|
| **Coverage** | How many topics are covered |
| **Factuality** | How many facts are supported |
| **F1** | Harmonic mean of coverage and factuality |
| **Runtime** | Time taken per query |
| **Iterations** | Number of RAG iterations used |

All data is stored and visualized automatically!

---

## 📁 What Was Created

### Core Scripts
- **`run_full_experiment.py`** - Main experiment runner
- **`visualize_only.py`** - Regenerate visualizations
- **`scripts/run_experiment.sh`** - SLURM batch script

### Tracking System
- **`eval/experiment_tracker.py`** - Data tracking
- **`eval/visualizer.py`** - Visualization generation

### Documentation
- **`README_EXPERIMENTS.md`** - Quick reference
- **`EXPERIMENT_TRACKING.md`** - Complete guide
- **`IMPLEMENTATION_SUMMARY.md`** - What was implemented

---

## 🎮 Common Commands

### Run a Test (10 queries)
```bash
python run_full_experiment.py \
    --queries_path data/antique/train.jsonl \
    -n 10 \
    --description "test"
```

### Run a Batch (100 queries)
```bash
python run_full_experiment.py \
    --queries_path data/antique/train.jsonl \
    -n 100 \
    --description "baseline_100"
```

### Run Full Training Set (2,426 queries)
```bash
# Using SLURM (recommended for full set)
sbatch scripts/run_experiment.sh all "full_baseline"

# Or directly
python run_full_experiment.py \
    --queries_path data/antique/train.jsonl \
    --description "full_training"
```

### Resume Interrupted Experiment
```bash
# If stopped at query 150
python run_full_experiment.py \
    --queries_path data/antique/train.jsonl \
    --run_id run_20231214_143022 \
    --resume_from 150
```

### Regenerate Visualizations
```bash
python visualize_only.py
```

---

## 📈 Example Workflow

### Scenario: Test Different Iteration Counts

```bash
# Run 1: 2 iterations
python run_full_experiment.py \
    --queries_path data/antique/train.jsonl \
    -n 100 \
    --max_iterations 2 \
    --description "test_2_iterations"

# Run 2: 3 iterations (baseline)
python run_full_experiment.py \
    --queries_path data/antique/train.jsonl \
    -n 100 \
    --max_iterations 3 \
    --description "test_3_iterations"

# Run 3: 5 iterations
python run_full_experiment.py \
    --queries_path data/antique/train.jsonl \
    -n 100 \
    --max_iterations 5 \
    --description "test_5_iterations"

# Compare results in visualizations
python visualize_only.py
```

The visualizations will show how F1 scores change across the three runs!

---

## 📂 Output Structure

After running experiments:

```
.diverseTextGen/output/experiments/
├── runs_index.json              # Index of all runs
├── query_history.json           # Per-query scores across runs
├── run_20231214_143022/         # First run
│   ├── metadata.json
│   ├── results.jsonl            # ← Per-query results
│   ├── summary.json             # ← Aggregate statistics
│   └── rag_outputs/
│       ├── 3097310.json
│       └── ...
├── run_20231214_150045/         # Second run
│   └── ...
└── visualizations/
    ├── aggregate_trends.png     # ← Overall trends
    └── query_tracking.png       # ← Per-query tracking
```

---

## 💡 Understanding the Visualizations

### 1. Aggregate Trends (`aggregate_trends.png`)

Shows 4 plots:
- **Top-left**: Average Coverage over runs
- **Top-right**: Average Factuality over runs  
- **Bottom-left**: Average F1 over runs
- **Bottom-right**: All three metrics combined

**Use this to:** See if your system is improving over time

### 2. Query Tracking (`query_tracking.png`)

Shows 20 individual queries and their F1 scores across runs.

**Use this to:** 
- Identify consistently difficult queries
- See which queries improve/worsen
- Understand variance in performance

---

## 🔧 Advanced Usage

### Programmatic Access

```python
from eval.experiment_tracker import ExperimentTracker
from eval.visualizer import ICATVisualizer

# Access tracked data
tracker = ExperimentTracker(".diverseTextGen/output/experiments")

# Get history for a specific query
history = tracker.get_query_history("3097310")
print(f"Query tracked across {len(history['runs'])} runs")

# Compare two runs
comparison = tracker.compare_runs(["run_1", "run_2"])
print(f"Common queries: {len(comparison['common_queries'])}")

# Generate custom visualizations
viz = ICATVisualizer(".diverseTextGen/output/experiments")
viz.plot_aggregate_trends("my_custom_plot.png", run_ids=["run_1", "run_2"])
```

See `example_usage.py` for more examples!

---

## 🐛 Troubleshooting

### "ModuleNotFoundError: No module named 'matplotlib'"

**Solution:**
```bash
pip install matplotlib seaborn pandas
```

### Experiment interrupted midway

**Solution:** Use resume functionality
```bash
# Check how many completed
wc -l .diverseTextGen/output/experiments/run_*/results.jsonl

# Resume from that point
python run_full_experiment.py \
    --run_id YOUR_RUN_ID \
    --resume_from 150 \
    --queries_path data/antique/train.jsonl
```

### SLURM job not starting

**Solution:** Check job status
```bash
squeue -u $USER
scontrol show job JOB_ID
tail -f server_logs/experiment_*.out
```

---

## 📚 Documentation

| Document | Purpose |
|----------|---------|
| **START_HERE.md** (this file) | Quick overview and setup |
| **README_EXPERIMENTS.md** | Quick reference guide |
| **EXPERIMENT_TRACKING.md** | Complete documentation |
| **IMPLEMENTATION_SUMMARY.md** | What was implemented |
| **example_usage.py** | Code examples |

---

## ✨ Key Features

### 1. Automatic Tracking
Every query result is saved automatically. No manual data management!

### 2. Historical Data
See how queries perform across multiple runs.

### 3. Resume Capability
Interrupted experiments can be resumed from where they stopped.

### 4. Visualization
Professional plots generated automatically.

### 5. Comparison
Easy comparison between different experimental configurations.

### 6. Integration
Works seamlessly with existing RAG and ICAT systems.

---

## 🎯 Recommended Workflow

1. **Start Small**
   ```bash
   python run_full_experiment.py -n 10 --description "test"
   ```

2. **Scale to Medium**
   ```bash
   python run_full_experiment.py -n 100 --description "baseline"
   ```

3. **Test Variations**
   ```bash
   python run_full_experiment.py -n 100 --max_iterations 5 --description "more_iterations"
   ```

4. **Compare Results**
   ```bash
   python visualize_only.py
   # View the plots to see differences
   ```

5. **Run Full Set**
   ```bash
   sbatch scripts/run_experiment.sh all "full_baseline"
   ```

---

## 📊 Example Results

After running 3 experiments, your visualizations will show:

```
Run 1 (2 iter): F1 = 0.65
Run 2 (3 iter): F1 = 0.68  ← Improvement!
Run 3 (5 iter): F1 = 0.70  ← Even better!
```

The plots make these trends immediately visible!

---

## 🚦 Next Steps

### Immediate (5 minutes)
1. Install dependencies: `pip install matplotlib seaborn pandas`
2. Run test: `python run_full_experiment.py -n 10 --description "test"`
3. View results: Check `.diverseTextGen/output/experiments/`

### Short-term (1 hour)
1. Run larger batch: `-n 100`
2. Try different configurations: `--max_iterations 5`
3. Compare results in visualizations

### Long-term (Project goal)
1. Run full training set: All 2,426 queries
2. Track improvements over time
3. Optimize based on insights from visualizations

---

## 💬 Questions?

- **Quick questions:** See `README_EXPERIMENTS.md`
- **Detailed info:** See `EXPERIMENT_TRACKING.md`
- **Code examples:** See `example_usage.py`
- **What was built:** See `IMPLEMENTATION_SUMMARY.md`

---

## ✅ Summary

You now have a complete system to:
- ✅ Run experiments on all training queries
- ✅ Track ICAT scores automatically  
- ✅ Visualize trends and improvements
- ✅ Compare different configurations
- ✅ Resume interrupted experiments

**Ready to start?**

```bash
pip install matplotlib seaborn pandas
python run_full_experiment.py --queries_path data/antique/train.jsonl -n 10 --description "first_test"
```

Good luck with your experiments! 🚀

