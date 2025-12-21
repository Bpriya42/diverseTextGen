# Command Cheat Sheet

## 🚀 Quick Commands

### Setup (One-time)
```bash
pip install matplotlib seaborn pandas
```

### Run Experiments

```bash
# Test (10 queries)
python run_full_experiment.py --queries_path data/antique/train.jsonl -n 10 --description "test"

# Small batch (100 queries)
python run_full_experiment.py --queries_path data/antique/train.jsonl -n 100 --description "batch_100"

# Full training set (2,426 queries) - SLURM
sbatch scripts/run_experiment.sh all "full_baseline"

# Full training set (2,426 queries) - Direct
python run_full_experiment.py --queries_path data/antique/train.jsonl --description "full_training"
```

### Resume Interrupted Experiment
```bash
# Check progress first
wc -l .diverseTextGen/output/experiments/run_*/results.jsonl

# Resume from last completed query
python run_full_experiment.py \
    --queries_path data/antique/train.jsonl \
    --run_id run_20231214_143022 \
    --resume_from 150
```

### Generate/Regenerate Visualizations
```bash
python visualize_only.py
```

### View Results
```bash
# List all runs
ls -lh .diverseTextGen/output/experiments/

# View run summary
cat .diverseTextGen/output/experiments/run_*/summary.json | jq

# Count queries per run
wc -l .diverseTextGen/output/experiments/run_*/results.jsonl

# View visualizations
ls -lh .diverseTextGen/output/experiments/visualizations/
```

### SLURM Jobs
```bash
# Submit job
sbatch scripts/run_experiment.sh 100 "my_experiment"

# Check status
squeue -u $USER

# View logs
tail -f server_logs/experiment_*.out

# Cancel job
scancel JOB_ID
```

---

## 🎯 Common Patterns

### Pattern 1: Compare Configurations
```bash
# Run with different iteration counts
python run_full_experiment.py -n 100 --max_iterations 2 --description "test_2iter"
python run_full_experiment.py -n 100 --max_iterations 3 --description "test_3iter"
python run_full_experiment.py -n 100 --max_iterations 5 --description "test_5iter"

# Visualize comparison
python visualize_only.py
```

### Pattern 2: Progressive Testing
```bash
# Start small
python run_full_experiment.py -n 10 --description "tiny_test"

# Scale up if successful
python run_full_experiment.py -n 100 --description "medium_test"

# Go full if all looks good
sbatch scripts/run_experiment.sh all "full_run"
```

### Pattern 3: Resume Long Run
```bash
# Start
sbatch scripts/run_experiment.sh all "full_baseline"

# If interrupted, check progress
wc -l .diverseTextGen/output/experiments/run_*/results.jsonl

# Resume
python run_full_experiment.py \
    --run_id run_20231214_143022 \
    --resume_from 500 \
    --queries_path data/antique/train.jsonl
```

---

## 📊 Data Access

### Python API
```python
from eval.experiment_tracker import ExperimentTracker
from eval.visualizer import ICATVisualizer

# Initialize
tracker = ExperimentTracker(".diverseTextGen/output/experiments")
viz = ICATVisualizer(".diverseTextGen/output/experiments")

# Get all runs
runs = tracker.get_all_runs()

# Get query history
history = tracker.get_query_history("3097310")

# Compare runs
comparison = tracker.compare_runs(["run_1", "run_2"])

# Generate visualizations
viz.generate_all_visualizations("output/viz")
```

### Command Line
```bash
# View summary
cat .diverseTextGen/output/experiments/run_*/summary.json | jq '.aggregate_stats'

# Extract F1 scores
cat .diverseTextGen/output/experiments/run_*/results.jsonl | jq '.icat_scores.f1'

# Find best performing queries
cat .diverseTextGen/output/experiments/run_*/results.jsonl | jq 'select(.icat_scores.f1 > 0.9)'
```

---

## 🔍 Monitoring

### During Experiment
```bash
# Watch progress
watch -n 5 'wc -l .diverseTextGen/output/experiments/run_*/results.jsonl'

# View latest results
tail -f .diverseTextGen/output/experiments/run_*/results.jsonl

# SLURM logs
tail -f server_logs/experiment_*.out
```

### After Completion
```bash
# Summary
cat .diverseTextGen/output/experiments/run_*/summary.json | jq

# Success rate
cat .diverseTextGen/output/experiments/run_*/summary.json | jq '.aggregate_stats | {successful_queries, total_queries}'

# Average scores
cat .diverseTextGen/output/experiments/run_*/summary.json | jq '.aggregate_stats | {avg_coverage, avg_factuality, avg_f1}'
```

---

## 🎓 Tips & Best Practices

### 1. Use Descriptive Names
```bash
# Good
python run_full_experiment.py -n 100 --description "baseline_qwen3_4b_3iter"

# Not as good
python run_full_experiment.py -n 100 --description "test"
```

### 2. Test Before Full Run
```bash
# Always test with -n 10 first
python run_full_experiment.py -n 10 --description "test"

# Then scale up
python run_full_experiment.py -n 100 --description "larger_test"

# Finally go full
sbatch scripts/run_experiment.sh all "full_run"
```

### 3. Save Visualizations
```bash
# Regenerate and save after each new run
python visualize_only.py
cp .diverseTextGen/output/experiments/visualizations/*.png ~/my_results/
```

### 4. Track in Git (Optional)
```bash
# Track experiment summaries in git (not the full data)
git add .diverseTextGen/output/experiments/runs_index.json
git commit -m "Add experiment tracking data"
```

---

## 📞 Help

- **Quick start**: `START_HERE.md`
- **Commands**: This file (`COMMANDS.md`)
- **Full guide**: `EXPERIMENT_TRACKING.md`
- **Examples**: `example_usage.py`

---

## ✅ Ready to Go!

Start your first experiment:

```bash
python run_full_experiment.py \
    --queries_path data/antique/train.jsonl \
    -n 10 \
    --description "my_first_experiment"
```

🎉 Happy experimenting!

