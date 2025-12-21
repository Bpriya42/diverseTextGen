# Experiment Tracking System Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Experiment Tracking System                        │
└─────────────────────────────────────────────────────────────────────┘

┌────────────────┐
│  User Starts   │
│  Experiment    │
└────────┬───────┘
         │
         v
┌────────────────────────────────────────────────┐
│  run_full_experiment.py                        │
│  ┌──────────────────────────────────────────┐ │
│  │ 1. Load queries from train.jsonl         │ │
│  │ 2. Create ExperimentTracker              │ │
│  │ 3. Create run with metadata              │ │
│  │ 4. Initialize ICAT evaluator             │ │
│  └──────────────────────────────────────────┘ │
└────────┬───────────────────────────────────────┘
         │
         v
┌────────────────────────────────────────────────┐
│  For Each Query:                               │
│  ┌──────────────────────────────────────────┐ │
│  │ A. Run RAG System (run_langgraph.py)     │ │
│  │    → Get final_answer                    │ │
│  │                                           │ │
│  │ B. Evaluate with ICAT (eval/icat.py)     │ │
│  │    → Get Coverage, Factuality, F1        │ │
│  │                                           │ │
│  │ C. Track Result (experiment_tracker.py)  │ │
│  │    → Save to results.jsonl               │ │
│  │    → Update query_history.json           │ │
│  └──────────────────────────────────────────┘ │
└────────┬───────────────────────────────────────┘
         │
         v
┌────────────────────────────────────────────────┐
│  Finalize Run                                  │
│  ┌──────────────────────────────────────────┐ │
│  │ 1. Calculate aggregate statistics        │ │
│  │ 2. Save summary.json                     │ │
│  │ 3. Update runs_index.json                │ │
│  └──────────────────────────────────────────┘ │
└────────┬───────────────────────────────────────┘
         │
         v
┌────────────────────────────────────────────────┐
│  Generate Visualizations                       │
│  ┌──────────────────────────────────────────┐ │
│  │ ICATVisualizer (eval/visualizer.py)      │ │
│  │                                           │ │
│  │ → aggregate_trends.png                   │ │
│  │ → query_tracking.png                     │ │
│  └──────────────────────────────────────────┘ │
└────────┬───────────────────────────────────────┘
         │
         v
┌────────────────┐
│ Results Ready! │
└────────────────┘
```

---

## Data Flow

```
Input: train.jsonl (2,426 queries)
  │
  ├─> Query 1 ─────> RAG System ─────> Final Answer ─────> ICAT ─────> Scores ─────┐
  │                                                                                  │
  ├─> Query 2 ─────> RAG System ─────> Final Answer ─────> ICAT ─────> Scores ─────┤
  │                                                                                  │
  ├─> Query 3 ─────> RAG System ─────> Final Answer ─────> ICAT ─────> Scores ─────┤
  │                                                                                  │
  └─> ...                                                                            │
                                                                                     │
                                                                                     v
                                                                    ┌─────────────────────────────┐
                                                                    │  ExperimentTracker          │
                                                                    │  ─────────────────────────  │
                                                                    │  • results.jsonl            │
                                                                    │  • query_history.json       │
                                                                    │  • runs_index.json          │
                                                                    │  • summary.json             │
                                                                    └────────────┬────────────────┘
                                                                                 │
                                                                                 v
                                                                    ┌─────────────────────────────┐
                                                                    │  ICATVisualizer             │
                                                                    │  ─────────────────────────  │
                                                                    │  • aggregate_trends.png     │
                                                                    │  • query_tracking.png       │
                                                                    └─────────────────────────────┘
```

---

## Component Interactions

### ExperimentTracker
- **Reads from**: Nothing initially
- **Writes to**: 
  - `runs_index.json` (all runs metadata)
  - `query_history.json` (per-query scores)
  - `run_*/results.jsonl` (streaming results)
  - `run_*/summary.json` (aggregate stats)
- **Used by**: `run_full_experiment.py`

### ICATVisualizer
- **Reads from**: 
  - `runs_index.json`
  - `query_history.json`
  - `run_*/summary.json`
- **Writes to**: PNG image files
- **Used by**: 
  - `run_full_experiment.py` (automatic)
  - `visualize_only.py` (manual)

### run_full_experiment.py
- **Reads from**: 
  - `data/antique/train.jsonl` (queries)
  - `config.py` (configuration)
- **Uses**: 
  - `run_langgraph.py` (RAG execution)
  - `eval/icat.py` (ICAT evaluation)
  - `ExperimentTracker` (tracking)
  - `ICATVisualizer` (visualization)
- **Writes to**: All experiment data and visualizations

---

## File Relationships

```
run_full_experiment.py
├── imports config.py (paths, settings)
├── imports run_langgraph.py (RAG execution)
├── imports eval/icat.py (ICAT evaluation)
├── imports eval/experiment_tracker.py (tracking)
└── imports eval/visualizer.py (visualization)

eval/experiment_tracker.py
└── standalone (only uses pathlib, json, datetime)

eval/visualizer.py
├── imports experiment_tracker.py data files
└── uses matplotlib, seaborn, pandas

visualize_only.py
└── imports eval/visualizer.py
```

---

## Storage Schema

### runs_index.json
```json
{
  "runs": [
    {
      "run_id": "run_20231214_143022",
      "timestamp": "2023-12-14T14:30:22.123456",
      "description": "Baseline with 3 iterations",
      "config": {
        "n_queries": 100,
        "max_iterations": 3,
        "corpus_path": "..."
      },
      "status": "completed",
      "aggregate_stats": {
        "total_queries": 100,
        "successful_queries": 98,
        "avg_coverage": 0.65,
        "avg_factuality": 0.72,
        "avg_f1": 0.68
      }
    }
  ],
  "metadata": { ... }
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
        "timestamp": "2023-12-14T14:30:25.123456",
        "icat_scores": {
          "coverage": 0.85,
          "factuality": 0.90,
          "f1": 0.875
        },
        "rag_metrics": {
          "total_iterations": 3,
          "runtime_seconds": 12.5
        }
      },
      {
        "run_id": "run_20231214_150045",
        "timestamp": "2023-12-14T15:00:48.789012",
        "icat_scores": {
          "coverage": 0.88,
          "factuality": 0.92,
          "f1": 0.90
        },
        "rag_metrics": {
          "total_iterations": 5,
          "runtime_seconds": 18.3
        }
      }
    ]
  }
}
```

### results.jsonl (per run)
```jsonl
{"query_id": "3097310", "query": "What causes...", "icat_scores": {...}, "rag_metrics": {...}, "timestamp": "..."}
{"query_id": "3910705", "query": "why don't they...", "icat_scores": {...}, "rag_metrics": {...}, "timestamp": "..."}
...
```

---

## Execution Flow

### Sequential Processing (run_full_experiment.py)

```
1. Load all queries from train.jsonl
2. Create ExperimentTracker
3. Create new run with unique ID
4. Initialize ICAT evaluator (once)

FOR EACH query:
  5. Run RAG system → get final_answer
  6. Run ICAT evaluation → get scores
  7. Log to tracker → append to results.jsonl
  8. Print progress

9. Finalize run → calculate aggregates → save summary.json
10. Generate visualizations → create PNG files
11. Done!
```

### Parallel Processing (future enhancement)

The system is designed to be extended with parallel processing:

```python
# Future: Process queries in parallel
from concurrent.futures import ProcessPoolExecutor

with ProcessPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(process_query, q) for q in queries]
    results = [f.result() for f in futures]
```

---

## Error Handling

The system includes robust error handling:

1. **Query-level errors**: Logged and skipped, don't stop experiment
2. **Resume capability**: If entire experiment fails, can resume
3. **Data validation**: Checks for missing/corrupt data
4. **Graceful degradation**: Missing visualizations won't prevent data saving

---

## Performance Characteristics

### Small Batch (10 queries)
- Runtime: ~5-10 minutes
- Memory: ~2-4 GB
- Output: ~10 JSON files + 2 PNG files

### Medium Batch (100 queries)
- Runtime: ~30-60 minutes
- Memory: ~4-8 GB
- Output: ~100 JSON files + 2 PNG files

### Full Set (2,426 queries)
- Runtime: ~12-24 hours (estimated)
- Memory: ~8-16 GB
- Output: ~2,426 JSON files + 2 PNG files
- Recommended: Use SLURM with `sbatch scripts/run_experiment.sh all "baseline"`

---

## Extensibility

The system is designed to be easily extended:

### Add New Metrics
```python
# In experiment_tracker.py
tracker.log_query_result(
    run_id=run_id,
    query_id=query_id,
    query=query,
    icat_scores={...},
    rag_metrics={...},
    custom_metrics={...}  # Add your own metrics!
)
```

### Add New Visualizations
```python
# In visualizer.py
class ICATVisualizer:
    def plot_custom_analysis(self, output_path: str):
        # Your custom visualization
        pass
```

### Add New Analysis
```python
# New file: eval/analyzer.py
class ExperimentAnalyzer:
    def statistical_significance_test(self, run_1, run_2):
        # T-test, ANOVA, etc.
        pass
```

---

## Integration Points

The system integrates at these points:

1. **Input**: `data/antique/train.jsonl` → Training queries
2. **Execution**: `run_langgraph.py` → RAG system
3. **Evaluation**: `eval/icat.py` → ICAT scoring
4. **Tracking**: `experiment_tracker.py` → Data storage
5. **Visualization**: `visualizer.py` → Plot generation
6. **Output**: `.diverseTextGen/output/experiments/` → All results

---

## Backward Compatibility

✅ All existing scripts continue to work  
✅ No breaking changes to existing code  
✅ New functionality is additive  
✅ Can use old and new systems side-by-side  

---

## Testing

The system includes comprehensive tests in `test_experiment_tracking.py`:

- ✅ ExperimentTracker functionality
- ✅ ICATVisualizer generation
- ✅ Integration between components
- ✅ Data persistence
- ✅ Multi-run scenarios

Note: Tests require full environment. For quick validation, run a real experiment with `-n 2`

---

## Future Enhancements (Optional)

Potential additions you could make:

1. **Statistical Testing**: Add significance tests between runs
2. **More Visualizations**: Box plots, heatmaps, correlation matrices
3. **Export to CSV**: For analysis in Excel/R
4. **Web Dashboard**: Real-time monitoring
5. **Parallel Processing**: Speed up large experiments
6. **Auto-tuning**: Optimize based on tracked results

---

## Summary

This architecture provides:
- ✅ Clean separation of concerns
- ✅ Modular design
- ✅ Easy to extend
- ✅ Robust error handling
- ✅ Comprehensive tracking
- ✅ Professional visualizations
- ✅ Integration with existing code

Ready to track and visualize your ICAT experiments! 🎉

