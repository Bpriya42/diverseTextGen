# LLM Observability

This document describes the LLM decision tracking and plateau detection system.

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Using Observability](#using-observability)
4. [Output Examples](#output-examples)
5. [Analyzing Logs](#analyzing-logs)

---

## Overview

The observability system provides **real-time tracking of LLM decisions** throughout the iterative RAG pipeline. It helps understand:

- **What decisions the LLM makes** at each iteration
- **Why the system refines its answers** (factual issues, coverage gaps)
- **When quality improvements plateau** (diminishing returns)
- **How metrics evolve** over iterations

### What It Tracks

**1. LLM Decisions:**
- Planner: Query decomposition and refinement decisions
- Synthesizer: Answer generation metrics

**2. Quality Metrics:**
- Factual accuracy (supported facts / total facts)
- Coverage ratio (well-covered aspects / total aspects)
- Composite quality score (60% accuracy + 40% coverage)

**3. Plateau Detection:**
- Monitors quality improvement over last 4 iterations
- Flags when improvement < 5% threshold
- Informational only - doesn't force termination

### Why It's Useful

✅ **Transparency**: See exactly what the LLM decides at each step  
✅ **Debugging**: Understand why iterations continue or stop  
✅ **Analysis**: Track quality evolution for research  
✅ **Optimization**: Identify when improvements stall  

---

## Features

### 1. LLM Decision Logging

**Automatic tracking of:**

#### Planner Decisions

**Initial Plan (iteration 0):**
```
[PLANNER - Iteration 0] Decision: initial_plan
  num_aspects: 5
  aspects: ['Causes', 'Symptoms', 'Treatment', 'Prevention', 'Risk factors']
```

**Plan Refinement (iteration 1+):**
```
[PLANNER - Iteration 1] Decision: refine_plan
  aspects_before: 5
  aspects_after: 6
  refuted_facts: 2
  unclear_facts: 3
  missing_points: 1
  aspects_added: ['Complications']
  aspects_removed: []
```

**Logged metrics:**
- `aspects_before`/`aspects_after`: Plan size changes
- `refuted_facts`: Number of contradicted facts to address
- `unclear_facts`: Number of facts needing clarification
- `missing_points`: Number of missing salient topics
- `aspects_added`: New aspects (up to 3 shown)
- `aspects_removed`: Removed aspects (up to 3 shown)

#### Synthesizer Decisions

```
[SYNTHESIZER - Iteration 1] Decision: generate_answer
  answer_length: 1245
  answer_word_count: 187
  num_aspects_used: 6
  num_documents_used: 30
```

**Logged metrics:**
- `answer_length`: Character count
- `answer_word_count`: Word count
- `num_aspects_used`: Aspects in the plan
- `num_documents_used`: Total retrieved documents

### 2. Quality Metrics Tracking

**System tracks quality at each iteration:**

```
[SYSTEM - Iteration 1] Decision: iteration_metrics
  factual_accuracy: 0.850
  coverage_ratio: 0.667
  composite_score: 0.777
  refuted_facts: 1
  unclear_facts: 5
  missing_points: 2
```

**Metrics explained:**

- **factual_accuracy**: `supported / total_facts`
  - Shows how many facts are verified by retrieved documents
  
- **coverage_ratio**: `well_covered_aspects / total_aspects`
  - Shows how comprehensively aspects are addressed
  
- **composite_score**: `0.6 × accuracy + 0.4 × coverage`
  - Weighted combination for overall quality
  - Used for plateau detection

### 3. Plateau Detection

**Automatically detects when quality stops improving:**

```
[Iteration Gate] ⚠️  plateau_detected: composite score improvement 0.02 < 0.05 over last 4 iterations
[Iteration Gate] Quality may have stabilized. Continuing to let quality checks decide termination.
```

**Detection parameters:**
- **Window**: 4 iterations (need 4 data points)
- **Threshold**: 0.05 (5% improvement required)
- **Action**: Log warning, don't force termination

**Logged during plateau check:**
```json
{
  "decision_type": "plateau_check",
  "metrics": {
    "plateau_detected": true,
    "avg_improvement": 0.024,
    "threshold": 0.05,
    "window": 4,
    "recent_scores": [0.75, 0.76, 0.77, 0.78]
  }
}
```

**Interpretation:**
- `avg_improvement`: Average change in composite score
- `recent_scores`: Last 4 composite scores
- If improvement is small, quality may have plateaued

---

## Using Observability

### Automatic Logging

**No configuration needed!** The system automatically logs all decisions during execution.

**Run any query:**

```bash
python scripts/run_langgraph.py \
    --query "What causes headaches?" \
    --query_id "test_001"
```

**Logs are saved to:** `artifacts/llm_decisions.jsonl`

### Console Output

**During execution, you'll see:**

```
[PLANNER - Iteration 0] Decision: initial_plan
  num_aspects: 5
  aspects: ['Causes', 'Symptoms', 'Treatment', 'Prevention', 'Risk factors']

[Planner] Processing iteration 0
[Planner] Generated initial plan with 5 aspects

[Retriever] Retrieving documents for 5 aspects
[Retriever] Retrieved 25 documents total

[SYNTHESIZER - Iteration 0] Decision: generate_answer
  answer_length: 1124
  answer_word_count: 168
  num_aspects_used: 5
  num_documents_used: 25

[Synthesizer] Generating answer
[Synthesizer] Generated answer (1124 chars)

[SYSTEM - Iteration 0] Decision: iteration_metrics
  factual_accuracy: 0.800
  coverage_ratio: 0.600
  composite_score: 0.720
  refuted_facts: 2
  unclear_facts: 4
  missing_points: 1

[Iteration Gate] EVALUATION
  Quality Check:
    Facts: 40/50 supported, 2 refuted, 4 unclear (8%)
      → Refuted: ✗ FAIL (must be 0)
      → Unclear: ✓ OK (4 <= 15%)
    Coverage: 1 missing salient points
      → Missing points: ✓ OK (1 <= 1)
    
[Iteration Gate] → Continuing to iteration 2
```

### Log File Location

**Default location:**
```
artifacts/llm_decisions.jsonl
```

**Format:** JSON Lines (one decision per line)

**Create artifacts directory** (if not exists):
```bash
mkdir -p artifacts
```

---

## Output Examples

### Console Output Example

```
================================================================================
STARTING ITERATIVE RAG SYSTEM
================================================================================
Query ID: test_001
Query: What causes headaches and how can they be treated?
Mode: Quality-controlled (memory-bounded)
================================================================================

[PLANNER - Iteration 0] Decision: initial_plan
  num_aspects: 5

[SYNTHESIZER - Iteration 0] Decision: generate_answer
  answer_length: 1124
  answer_word_count: 168

[SYSTEM - Iteration 0] Decision: iteration_metrics
  factual_accuracy: 0.800
  coverage_ratio: 0.600
  composite_score: 0.720

--------------------------------------------------------------------------------
ITERATION 2
--------------------------------------------------------------------------------

[PLANNER - Iteration 1] Decision: refine_plan
  aspects_before: 5
  aspects_after: 6
  refuted_facts: 2
  unclear_facts: 3
  missing_points: 1

[SYNTHESIZER - Iteration 1] Decision: generate_answer
  answer_length: 1456
  answer_word_count: 218

[SYSTEM - Iteration 1] Decision: iteration_metrics
  factual_accuracy: 0.920
  coverage_ratio: 0.833
  composite_score: 0.885

[Iteration Gate] ✓ Quality complete - no further improvements needed - TERMINATING
```

### JSON Log Format

**Planner decision:**
```json
{
  "timestamp": "2025-12-31T10:30:45.123456",
  "iteration": 1,
  "agent": "planner",
  "decision_type": "refine_plan",
  "metrics": {
    "aspects_before": 5,
    "aspects_after": 6,
    "refuted_facts": 2,
    "unclear_facts": 3,
    "missing_points": 1,
    "aspects_added": ["Complications"],
    "aspects_removed": []
  }
}
```

**Synthesizer decision:**
```json
{
  "timestamp": "2025-12-31T10:30:47.456789",
  "iteration": 1,
  "agent": "synthesizer",
  "decision_type": "generate_answer",
  "metrics": {
    "answer_length": 1456,
    "answer_word_count": 218,
    "num_aspects_used": 6,
    "num_documents_used": 30
  }
}
```

**Iteration metrics:**
```json
{
  "timestamp": "2025-12-31T10:30:50.789012",
  "iteration": 1,
  "agent": "system",
  "decision_type": "iteration_metrics",
  "metrics": {
    "factual_accuracy": 0.920,
    "coverage_ratio": 0.833,
    "composite_score": 0.885,
    "refuted_facts": 0,
    "unclear_facts": 2,
    "missing_points": 1
  }
}
```

**Plateau detection:**
```json
{
  "timestamp": "2025-12-31T10:30:51.012345",
  "iteration": 3,
  "agent": "system",
  "decision_type": "plateau_check",
  "metrics": {
    "plateau_detected": true,
    "avg_improvement": 0.024,
    "threshold": 0.05,
    "window": 4,
    "recent_scores": [0.850, 0.865, 0.872, 0.874]
  }
}
```

### Plateau Warning Example

```
[Iteration Gate] EVALUATION
  Current iteration: 4

[SYSTEM - Iteration 4] Decision: plateau_check
  plateau_detected: True
  avg_improvement: 0.024
  threshold: 0.05

[Iteration Gate] ⚠️  plateau_detected: composite score improvement 0.024 < 0.05 over last 4 iterations
[Iteration Gate] Quality may have stabilized. Continuing to let quality checks decide termination.

  Quality Check:
    Facts: 47/50 supported, 0 refuted, 3 unclear (6%)
      → Refuted: ✓ OK (must be 0)
      → Unclear: ✓ OK (3 <= 15% OR 3 <= 3)
    Coverage: 0 missing salient points
      → Missing points: ✓ OK (0 <= 1)
    
[Iteration Gate] ✓ Quality complete - no further improvements needed - TERMINATING
```

---

## Analyzing Logs

### Basic Viewing

**View all decisions:**
```bash
cat artifacts/llm_decisions.jsonl | jq .
```

**View last 5 decisions:**
```bash
tail -5 artifacts/llm_decisions.jsonl | jq .
```

**Pretty print:**
```bash
cat artifacts/llm_decisions.jsonl | jq . | less
```

### Filter by Agent

**Planner decisions only:**
```bash
cat artifacts/llm_decisions.jsonl | jq 'select(.agent=="planner")'
```

**Synthesizer decisions only:**
```bash
cat artifacts/llm_decisions.jsonl | jq 'select(.agent=="synthesizer")'
```

**System metrics only:**
```bash
cat artifacts/llm_decisions.jsonl | jq 'select(.agent=="system")'
```

### Filter by Decision Type

**Initial plans:**
```bash
cat artifacts/llm_decisions.jsonl | jq 'select(.decision_type=="initial_plan")'
```

**Plan refinements:**
```bash
cat artifacts/llm_decisions.jsonl | jq 'select(.decision_type=="refine_plan")'
```

**Iteration metrics:**
```bash
cat artifacts/llm_decisions.jsonl | jq 'select(.decision_type=="iteration_metrics")'
```

**Plateau checks:**
```bash
cat artifacts/llm_decisions.jsonl | jq 'select(.decision_type=="plateau_check")'
```

### Extract Specific Fields

**Quality trajectory:**
```bash
cat artifacts/llm_decisions.jsonl | \
  jq 'select(.decision_type=="iteration_metrics") | 
      {iteration, factual_accuracy, coverage_ratio, composite_score}'
```

**Output:**
```json
{"iteration":0,"factual_accuracy":0.800,"coverage_ratio":0.600,"composite_score":0.720}
{"iteration":1,"factual_accuracy":0.920,"coverage_ratio":0.833,"composite_score":0.885}
{"iteration":2,"factual_accuracy":0.960,"coverage_ratio":0.917,"composite_score":0.943}
```

**Planner changes:**
```bash
cat artifacts/llm_decisions.jsonl | \
  jq 'select(.decision_type=="refine_plan") | 
      {iteration, aspects_before, aspects_after, refuted_facts, unclear_facts}'
```

**Answer growth:**
```bash
cat artifacts/llm_decisions.jsonl | \
  jq 'select(.decision_type=="generate_answer") | 
      {iteration, answer_length, answer_word_count}'
```

### Find Plateau Detections

**All plateaus:**
```bash
cat artifacts/llm_decisions.jsonl | \
  jq 'select(.decision_type=="plateau_check" and .metrics.plateau_detected==true)'
```

**Count plateaus:**
```bash
cat artifacts/llm_decisions.jsonl | \
  jq 'select(.decision_type=="plateau_check" and .metrics.plateau_detected==true)' | \
  wc -l
```

### Statistical Analysis

**Average composite score:**
```bash
cat artifacts/llm_decisions.jsonl | \
  jq 'select(.decision_type=="iteration_metrics") | .metrics.composite_score' | \
  awk '{sum+=$1; count++} END {print sum/count}'
```

**Max/Min accuracy:**
```bash
cat artifacts/llm_decisions.jsonl | \
  jq 'select(.decision_type=="iteration_metrics") | .metrics.factual_accuracy' | \
  sort -n | \
  head -1  # Min
  
cat artifacts/llm_decisions.jsonl | \
  jq 'select(.decision_type=="iteration_metrics") | .metrics.factual_accuracy' | \
  sort -n | \
  tail -1  # Max
```

**Count decisions by agent:**
```bash
cat artifacts/llm_decisions.jsonl | \
  jq -r '.agent' | \
  sort | \
  uniq -c
```

**Output:**
```
  5 planner
  5 synthesizer
  5 system
```

### Visualize Quality Trajectory

**Create CSV for plotting:**
```bash
cat artifacts/llm_decisions.jsonl | \
  jq -r 'select(.decision_type=="iteration_metrics") | 
         [.iteration, .metrics.factual_accuracy, .metrics.coverage_ratio, .metrics.composite_score] | 
         @csv' > quality_trajectory.csv
```

**Result:**
```csv
0,0.800,0.600,0.720
1,0.920,0.833,0.885
2,0.960,0.917,0.943
```

**Plot with Python:**
```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('quality_trajectory.csv', 
                 names=['iteration', 'accuracy', 'coverage', 'composite'])

plt.plot(df['iteration'], df['accuracy'], label='Accuracy', marker='o')
plt.plot(df['iteration'], df['coverage'], label='Coverage', marker='s')
plt.plot(df['iteration'], df['composite'], label='Composite', marker='^')
plt.xlabel('Iteration')
plt.ylabel('Score')
plt.title('Quality Trajectory')
plt.legend()
plt.grid(True)
plt.savefig('quality_trajectory.png')
```

---

## Configuration

### Plateau Detection Parameters

**Adjust in `llm_observability.py`:**

```python
class LLMObservability:
    def __init__(self, ...):
        # Plateau detection parameters
        self.plateau_window = 4       # Number of iterations to check
        self.plateau_threshold = 0.05  # Improvement threshold (5%)
```

**Make stricter (earlier detection):**
```python
self.plateau_window = 3       # Check last 3 iterations
self.plateau_threshold = 0.03  # Require 3% improvement
```

**Make more lenient (later detection):**
```python
self.plateau_window = 5       # Check last 5 iterations
self.plateau_threshold = 0.08  # Require 8% improvement
```

### Composite Score Weights

**Adjust in `llm_observability.py`:**

```python
# Current: 60% accuracy, 40% coverage
composite_score = 0.6 * accuracy + 0.4 * coverage_ratio

# Emphasize accuracy:
composite_score = 0.8 * accuracy + 0.2 * coverage_ratio

# Balance equally:
composite_score = 0.5 * accuracy + 0.5 * coverage_ratio
```

---

## Tips and Best Practices

### 1. Check Logs After Each Run

```bash
# Quick check
tail -20 artifacts/llm_decisions.jsonl | jq .

# Look for plateaus
cat artifacts/llm_decisions.jsonl | \
  jq 'select(.decision_type=="plateau_check")'
```

### 2. Track Planner Decisions

**Understanding refinement:**
```bash
cat artifacts/llm_decisions.jsonl | \
  jq 'select(.decision_type=="refine_plan") | 
      {iteration, refuted: .metrics.refuted_facts, unclear: .metrics.unclear_facts, 
       missing: .metrics.missing_points, added: .metrics.aspects_added}'
```

### 3. Monitor Quality Improvement

**Calculate improvement rate:**
```bash
cat artifacts/llm_decisions.jsonl | \
  jq 'select(.decision_type=="iteration_metrics") | 
      .metrics.composite_score' | \
  awk 'NR==1 {first=$1} END {print "Improvement:", $1-first}'
```

### 4. Identify Problematic Queries

**Queries with many iterations:**
- High refuted/unclear facts
- Many missing points
- Frequent plan changes

**Look for:**
- `refuted_facts > 3` (repeated)
- `unclear_facts > 5` (repeated)
- `missing_points > 2` (repeated)

### 5. Archive Logs

**Per-run archiving:**
```bash
# After each experiment
cp artifacts/llm_decisions.jsonl \
   artifacts/archive/llm_decisions_run_20241231.jsonl

# Clear for next run
rm artifacts/llm_decisions.jsonl
```

---

## Summary

The observability system provides **complete transparency** into LLM decision-making:

✅ **Real-time tracking** of all agent decisions  
✅ **Quality metrics** at each iteration  
✅ **Plateau detection** to identify diminishing returns  
✅ **JSON logs** for easy analysis  
✅ **Zero configuration** - automatic logging  

**Use it to:**
- Understand what LLM decides at each step
- Debug unexpected system behavior
- Optimize iteration strategies
- Analyze quality evolution for research

For system architecture, see [ARCHITECTURE.md](ARCHITECTURE.md).  
For usage instructions, see [USAGE.md](USAGE.md).

