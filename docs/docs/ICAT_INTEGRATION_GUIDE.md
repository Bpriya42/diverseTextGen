# ICAT Integration Guide

## Overview

This guide explains how the ICAT framework from the paper "Beyond Factual Accuracy: Evaluating Coverage of Diverse Factual Information in Long-form Text Generation" (arXiv:2501.03545) is integrated into the LangGraph multi-agent RAG system.

---

## ICAT Framework

### Paper Information

**Title:** Beyond Factual Accuracy: Evaluating Coverage of Diverse Factual Information in Long-form Text Generation

**Authors:** Chris Samarinas, Alexander Krubner, Alireza Salemi, Youngwoo Kim, Hamed Zamani

**Publication:** arXiv:2501.03545 (2025)

**Source Code:** https://github.com/algoprog/ICAT

### Core Concept

ICAT addresses a critical gap in LLM evaluation: existing metrics focus on **either** factual accuracy **or** coverage, but not both. ICAT provides a unified metric that ensures:
- Generated text is factually accurate (grounded in evidence)
- Generated text comprehensively covers diverse aspects of the topic

### Key Innovation

Using **harmonic mean** instead of arithmetic mean to combine factual accuracy and coverage ensures both dimensions must be high - a weakness in either dimension significantly lowers the overall score.

---

## Integration Architecture

### System Mapping to ICAT

```
ICAT Component              →  Our Implementation
─────────────────────────────────────────────────────────
Query                       →  User input
Atomic Claim Extraction     →  Agent 4: Fact Extractor
Claim Verification          →  Agent 5: Verifier
Topic/Aspect Generation     →  Agent 1: Planner
Claim-Topic Alignment       →  Agent 6: Coverage Evaluator
ICAT Score Calculation      →  Iteration Gate Node
```

### Implementation: ICAT-A Variant

We implement **ICAT-A** (fully automated):
- Automatically generates topics/aspects (Agent 1: Planner)
- LLM-based claim-topic alignment (Agent 6: Coverage Evaluator)
- No manual annotations required

**Advantage:** Works for any query without ground-truth topics

**Correlation with human judgments:** Pearson's ρ = 0.256 (BM25) to 0.503 (Dense retrieval)

---

## ICAT Score Calculation

### Location

File: `nodes/iteration_gate.py`

Function: `calculate_icat_score(state: RAGState)`

### Formula

```python
# Component 1: Factual Grounding Score
factual_grounding = (# supported facts) / (# total facts)

# Component 2: Topic Coverage Score  
well_covered = # aspects with status "well-covered"
partially_covered = # aspects with status "partially-covered"
total_aspects = # total planned aspects

topic_coverage = (well_covered + 0.5 × partially_covered) / total_aspects

# Component 3: ICAT Score (harmonic mean)
icat_score = 2 × (factual_grounding × topic_coverage) / 
             (factual_grounding + topic_coverage)
```

### Example Calculation

**Iteration 1:**
- Supported facts: 15/20 → Factual Grounding = 0.750
- Well-covered aspects: 3/5 → Topic Coverage = 0.600
- ICAT Score = 2 × (0.750 × 0.600) / (0.750 + 0.600) = 0.667

**Iteration 2 (after refinement):**
- Supported facts: 22/25 → Factual Grounding = 0.880
- Well-covered aspects: 6/7 → Topic Coverage = 0.857
- ICAT Score = 2 × (0.880 × 0.857) / (0.880 + 0.857) = 0.868

**Improvement:** 0.667 → 0.868 (+0.201, +30%)

---

## Termination Logic

### Decision Tree

```
Iteration N completed
    ↓
Calculate ICAT Score
    ↓
┌─────────────────────────────────────┐
│ Check 1: ICAT >= 0.85?              │
│   YES → Terminate (quality met)     │
│   NO  → Continue checking           │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Check 2: Iteration >= Max?          │
│   YES → Terminate (limit reached)   │
│   NO  → Continue checking           │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Check 3: Budget exceeded?           │
│   YES → Terminate (no resources)    │
│   NO  → Continue checking           │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Check 4: ICAT improvement < 0.02?   │
│   YES → Log warning, continue       │
│   NO  → Continue                    │
└─────────────────────────────────────┘
    ↓
Refine plan based on feedback
    ↓
Loop to Iteration N+1
```

### Termination Reasons

| Reason | Condition | Interpretation |
|--------|-----------|----------------|
| `icat_quality_threshold_met` | ICAT >= 0.85 | Success - High quality achieved |
| `max_iterations` | Iteration limit reached | Best effort - Return best answer |
| `budget_exceeded` | Tokens/time limit hit | Resource constraint - Return current |
| `no_improvement` | Future enhancement | Diminishing returns detected |

---

## Output Structure with ICAT

### JSON Output

```json
{
  "final_scores": {
    "icat_score": 0.889,
    "factual_grounding": 0.880,
    "topic_coverage": 0.898,
    "factuality": 0.88,
    "coverage": 0.90
  },
  "icat_metrics": {
    "icat_score": 0.889,
    "factual_grounding_score": 0.880,
    "topic_coverage_score": 0.898,
    "plan_quality_score": 0.85,
    "plan_completeness_score": 0.90,
    "supported_facts": 22,
    "total_facts": 25,
    "well_covered_aspects": 6,
    "total_aspects": 7
  }
}
```

### Console Output

```
[Iteration Gate] ICAT Evaluation
  Iteration: 2/3
  ICAT Score: 0.889
    Factual Grounding: 0.880 (22/25 supported)
    Topic Coverage: 0.898 (6/7 well-covered)
    Plan Quality: 0.850
    Plan Completeness: 0.900
[Iteration Gate] ICAT quality threshold met (0.889 >= 0.850) - Terminating
```

---

## Comparison: Before vs After ICAT

### Before (Simple Thresholds)

```python
# Check: factuality >= 0.85 AND coverage >= 0.85
if factuality >= 0.85 and coverage >= 0.85:
    terminate()
```

**Problem:** Treats both equally, doesn't capture interaction

**Example Issue:**
- Factuality: 0.95, Coverage: 0.76 → Continue (Coverage low)
- Factuality: 0.76, Coverage: 0.95 → Continue (Factuality low)
- No unified quality metric

### After (ICAT Harmonic Mean)

```python
# ICAT combines with harmonic mean
icat_score = 2 * (factual × coverage) / (factual + coverage)
if icat_score >= 0.85:
    terminate()
```

**Benefit:** Single metric that requires balance

**Example:**
- Factuality: 0.95, Coverage: 0.76 → ICAT: 0.842 (not good enough)
- Factuality: 0.88, Coverage: 0.90 → ICAT: 0.889 (excellent!)
- Clear quality signal for termination

---

## Actionable Feedback

### Priority Guidance

The iteration gate provides actionable feedback based on ICAT components:

```
[Iteration Gate] Continuing to iteration 3
  Target: ICAT >= 0.85 (current: 0.767, gap: 0.083)
  Priority: Improve factual grounding (current: 0.750)
```

**If Factual Grounding < 0.80:**
- Next iteration focuses on better evidence retrieval
- Planner adds aspects to clarify unclear/refuted facts

**If Topic Coverage < 0.80:**
- Next iteration focuses on missing aspects
- Planner adds new aspects for missing salient points

---

## Advantages Over Component-Only Evaluation

### 1. Unified Quality Signal

**Single metric** for decision-making instead of juggling multiple thresholds

### 2. Balanced Optimization

Forces improvement in **both** dimensions (harmonic mean penalty for imbalance)

### 3. Research-Validated

Demonstrated correlation with human judgments in ICAT paper

### 4. Interpretable

Breaks down into components for debugging:
- Which facts are unsupported?
- Which aspects are not covered?
- What's the overall quality?

### 5. Optimizable

Can be used as reward signal for:
- Reinforcement learning
- Prompt optimization
- Model fine-tuning

---

## Configuration

### Adjusting Thresholds

Edit `nodes/iteration_gate.py`:

```python
# Main ICAT threshold
icat_threshold = 0.85  # Higher = stricter quality requirement

# Minimum component thresholds for warnings
min_factual_threshold = 0.80
min_coverage_threshold = 0.80
```

### Recommended Settings

| Use Case | ICAT Threshold | Max Iterations | Rationale |
|----------|----------------|----------------|-----------|
| Research/Medical | 0.90 | 5 | High accuracy critical |
| General Q&A | 0.85 | 3 | Balanced quality/speed |
| Exploratory | 0.75 | 2 | Quick iterations |

---

## Future Enhancements

### 1. Weighted ICAT

Weight factual grounding higher for medical/legal domains:

```python
# Domain-specific weighting
alpha = 0.7  # Weight for factual grounding
beta = 0.3   # Weight for topic coverage

weighted_icat = (alpha * factual_grounding + beta * topic_coverage)
```

### 2. Aspect-Level ICAT

Calculate ICAT per aspect for fine-grained feedback

### 3. ICAT History Tracking

Track ICAT improvement trajectory across iterations for convergence analysis

### 4. NLI-based Verification

As in ICAT paper, use DeBERTa-based NLI model for more efficient fact verification

---

## References

**Primary Paper:**
Samarinas, C., Krubner, A., Salemi, A., Kim, Y., & Zamani, H. (2025). "Beyond Factual Accuracy: Evaluating Coverage of Diverse Factual Information in Long-form Text Generation". arXiv:2501.03545. https://arxiv.org/pdf/2501.03545

**Related Work:**
- FActScore (Min et al., 2023): Factual accuracy only
- VERISCORE (Song et al., 2024): Verification scoring
- ICAT: Combines factuality + coverage with aspect alignment

