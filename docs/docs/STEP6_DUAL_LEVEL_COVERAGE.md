# Step 6: Dual-Level Coverage Feedback - Enhanced Implementation

## Overview
The enhanced Agent 6 performs **dual-level coverage evaluation**, assessing both:
1. **Plan Coverage**: Does the plan identify all salient points needed to answer the query?
2. **Answer Coverage**: Does the answer address all aspects of the plan comprehensively?

This provides actionable feedback for iterative improvement of both query decomposition and answer generation.

---

## Dual-Level Evaluation Framework

### Level 1: Plan Coverage of Query
Evaluates whether the query decomposition (Step 1) identified all important aspects.

**Questions Answered:**
- Are there salient points the query requires that the plan missed?
- Does the plan decompose the query appropriately?
- Are there redundant or overlapping aspects?

**Output:**
- Plan quality score (0.0-1.0)
- Plan completeness score (0.0-1.0)
- List of covered salient points
- **List of missing salient points** → used for plan refinement
- List of redundant aspects → used for plan cleanup

### Level 2: Answer Coverage of Plan
Evaluates whether the generated answer (Step 3) addresses each planned aspect.

**Questions Answered:**
- Does the answer cover all planned aspects?
- Are aspects explained in sufficient depth?
- Does the answer include relevant information beyond the plan?

**Output:**
- Answer quality score (0.0-1.0)
- Breadth, depth, completeness, relevance scores
- Per-aspect coverage status (well-covered | partially-covered | not-covered)
- **List of missing aspects** → used for answer improvement
- Additional coverage (unplanned topics in answer)

---

## Input Format

The system expects `final_answers.json` from Step 3 with this structure:

```json
{
  "query_id": {
    "query": "user question",
    "plan": [
      {
        "aspect": "aspect name",
        "query": "subquery for this aspect",
        "reason": "why this aspect is important"
      },
      ...
    ],
    "answer": "generated answer text"
  }
}
```

---

## Output Structure

### Dual-Level Evaluation Output

```json
{
  "query_id": {
    "query": "...",
    "answer": "...",
    "plan": [...],
    "coverage_evaluation": {
      "plan_coverage": {
        "plan_quality_score": 0.85,
        "plan_completeness": 0.80,
        "covered_salient_points": [
          "Causes and symptoms well-identified",
          "Treatment options included",
          "Anatomical factors covered"
        ],
        "missing_salient_points": [
          "Prevention strategies not identified in plan",
          "Long-term prognosis aspect missing"
        ],
        "redundant_aspects": [
          "Aspects 2 and 3 overlap on inflammatory mechanisms"
        ],
        "plan_feedback": "Plan covers most medical aspects but misses preventive and prognostic dimensions"
      },
      "answer_coverage": {
        "answer_quality_score": 0.90,
        "breadth_score": 0.88,
        "depth_score": 0.92,
        "completeness_score": 0.87,
        "relevance_score": 0.95,
        "aspect_coverage": [
          {
            "aspect": "Anatomical and physiological causes",
            "coverage_status": "well-covered",
            "explanation": "Comprehensive discussion of knee anatomy, ligaments, cartilage"
          },
          {
            "aspect": "Common medical conditions",
            "coverage_status": "well-covered",
            "explanation": "Detailed coverage of arthritis types, gout, injuries"
          },
          {
            "aspect": "Lifestyle factors",
            "coverage_status": "partially-covered",
            "explanation": "Obesity and activity mentioned but lacks specific recommendations"
          }
        ],
        "covered_aspects": [
          "Anatomical causes thoroughly explained",
          "Medical conditions well-detailed",
          "Treatment options comprehensive"
        ],
        "missing_aspects": [
          "Lifestyle factors need more detail",
          "Recovery timelines not provided"
        ],
        "additional_coverage": [
          "Answer includes emergency symptoms not in plan"
        ]
      },
      "overall_coverage_score": 0.87,
      "strengths": [
        "Excellent medical terminology",
        "Well-structured progression",
        "Good depth on pathophysiology"
      ],
      "weaknesses": [
        "Plan missing prevention aspect",
        "Answer lacks specific timelines",
        "Could expand on lifestyle modifications"
      ],
      "improvement_suggestions": [
        {
          "level": "plan",
          "suggestion": "Add aspect for prevention strategies and risk reduction",
          "priority": "high"
        },
        {
          "level": "answer",
          "suggestion": "Expand lifestyle factors section with specific recommendations",
          "priority": "medium"
        },
        {
          "level": "both",
          "suggestion": "Include prognosis and recovery timeline information",
          "priority": "high"
        }
      ],
      "overall_feedback": "Strong coverage of medical causes and treatments, but plan should be expanded to include prevention and prognosis, and answer needs more detail on lifestyle modifications."
    }
  }
}
```

---

## Key Features

### 1. Hierarchical Coverage Analysis

```
Query (What causes knee pain?)
   ↓
Plan Coverage Check
   ├─ ✓ Anatomical causes identified
   ├─ ✓ Medical conditions identified
   ├─ ✓ Inflammatory mechanisms identified
   ├─ ✗ Prevention NOT identified ← Missing
   └─ ✗ Prognosis NOT identified ← Missing
   ↓
Answer Coverage Check (per planned aspect)
   ├─ Anatomical: well-covered
   ├─ Medical conditions: well-covered
   ├─ Inflammatory: partially-covered
   └─ (missing aspects can't be evaluated)
```

### 2. Actionable Improvement Suggestions

Each suggestion specifies:
- **Level**: `plan` | `answer` | `both`
- **Priority**: `high` | `medium` | `low`
- **Actionable text**: Specific guidance for improvement

**Example:**
```json
{
  "level": "plan",
  "suggestion": "Add aspect for prevention strategies including exercise, weight management, and joint protection",
  "priority": "high"
}
```

### 3. Redundancy Detection

Identifies overlapping aspects that can be merged:
```json
{
  "redundant_aspects": [
    "Aspects 2 and 3 both cover immune responses"
  ]
}
```

### 4. Unexpected Coverage

Tracks valuable information in the answer not anticipated by the plan:
```json
{
  "additional_coverage": [
    "Answer includes emergency warning signs",
    "Answer provides cost considerations"
  ]
}
```

---

## Usage in Iterative Improvement Loop

### Iteration Flow

```python
# Iteration N
query → plan → retrieve → answer

# Evaluate coverage
coverage_feedback = step6_coverage_feedback(query, plan, answer)

# Check termination
if coverage_feedback["overall_coverage_score"] >= 0.90:
    return answer  # Good enough
    
# Refine for Iteration N+1
plan_refinements = []
for missing in coverage_feedback["plan_coverage"]["missing_salient_points"]:
    # Add new aspect to plan
    plan_refinements.append({
        "aspect": missing,
        "query": generate_subquery(missing),
        "reason": "Identified as missing salient point"
    })

answer_refinements = []
for missing in coverage_feedback["answer_coverage"]["missing_aspects"]:
    # Retrieve additional docs for this aspect
    # Regenerate answer section
    answer_refinements.append(missing)

# Merge refinements and iterate
```

### Integration with Agent 5 (Verifier)

Combine factuality and coverage feedback:

```python
factual_score = agent5_output["factuality_score"]
coverage_score = agent6_output["overall_coverage_score"]

if factual_score < 0.80:
    # Priority: Fix incorrect facts
    action = "verify_and_correct"
elif coverage_score < 0.80:
    # Priority: Expand coverage
    action = "expand_and_enrich"
else:
    # Both high: done
    action = "finalize"
```

---

## Running Step 6

### Command
```bash
python step6_coverage_feedback.py
```

### Expected Output

```
📂 Loading input data...
🔧 Initializing LLM...

================================================================================
📊 Processing QID: 3097310
Query: What causes severe swelling and pain in the knees?...
Answer length: 1842 characters
📋 Using plan-aware coverage evaluation
🧠 Evaluating coverage...
✓ Overall Coverage Score: 0.87

📋 PLAN COVERAGE:
  - Plan Quality: 0.85
  - Plan Completeness: 0.80
  ⚠️  Missing salient points in plan: 2
      - Prevention strategies
      - Long-term prognosis

📝 ANSWER COVERAGE:
  - Answer Quality: 0.90
  - Breadth: 0.88
  - Depth: 0.92
  - Completeness: 0.87
  - Relevance: 0.95
  ⚠️  Missing from answer: 1
      - Lifestyle modifications lack detail

  🔧 High-priority improvements:
      [plan] Add aspect for prevention strategies
      [both] Include prognosis information

================================================================================
💾 Saving coverage feedback...
✅ Coverage feedback saved to /temp/coverage_feedback.json

================================================================================
📈 SUMMARY STATISTICS
================================================================================
Total Queries Processed: 3

📊 AVERAGE SCORES:
  Overall Coverage:     0.85

  Plan Evaluation:
    - Plan Quality:      0.83
    - Plan Completeness: 0.79

  Answer Evaluation:
    - Answer Quality:    0.88
    - Breadth:           0.87
    - Depth:             0.89

📋 PER-QUERY DETAILS:
  QID 3097310:
    Overall: 0.87
    Plan Quality: 0.85
    ⚠️  Plan missing: Prevention strategies, Long-term prognosis
    ⚠️  Answer missing: Lifestyle details
    🔧 Priority fixes: 2

================================================================================
✨ Coverage evaluation complete!
```

---

## Comparison: Basic vs Dual-Level

| Feature                    | Basic Evaluation | Dual-Level Evaluation |
|----------------------------|------------------|------------------------|
| Plan quality assessed      | ❌               | ✅                     |
| Missing salient points     | ❌               | ✅                     |
| Redundancy detection       | ❌               | ✅                     |
| Per-aspect answer coverage | ✅               | ✅                     |
| Actionable suggestions     | Basic            | Prioritized + targeted |
| Iteration guidance         | Limited          | Comprehensive          |

---

## Metrics Interpretation

### Plan Coverage Scores

| Score Range | Interpretation                                      | Action                        |
|-------------|-----------------------------------------------------|-------------------------------|
| 0.90-1.00   | Excellent - plan captures all salient points        | Proceed to answer evaluation  |
| 0.75-0.89   | Good - minor gaps                                   | Consider adding 1-2 aspects   |
| 0.60-0.74   | Fair - notable missing perspectives                 | Refine plan before answering  |
| < 0.60      | Poor - major salient points missing                 | Redesign query decomposition  |

### Answer Coverage Scores

| Score Range | Interpretation                                      | Action                        |
|-------------|-----------------------------------------------------|-------------------------------|
| 0.90-1.00   | Excellent - comprehensive answer                    | Verify facts (Agent 5)        |
| 0.75-0.89   | Good - most aspects covered                         | Expand missing aspects        |
| 0.60-0.74   | Fair - gaps in planned aspects                      | Retrieve + regenerate         |
| < 0.60      | Poor - many aspects missing or superficial          | Full regeneration needed      |

### Overall Coverage Score

Weighted combination: `0.3 * plan_quality + 0.7 * answer_quality`

Rationale: Answer coverage is more critical, but poor plan limits answer potential.

---

## Benefits for Multi-Agent System

### 1. Early Detection
Identifies plan deficiencies before expensive answer generation

### 2. Targeted Iteration
Pinpoints exactly what to improve (plan vs answer, which aspects)

### 3. Resource Optimization
Avoids wasting retrieval/generation on bad plans

### 4. Quality Assurance
Dual-level validation ensures both decomposition and synthesis quality

### 5. Transparency
Provides detailed explanations for scores and suggestions

---

## Next: Agent 7 (Consolidator)

Agent 7 will use feedback from Agents 5 and 6:

**Inputs:**
- Original answer
- Factual feedback (Agent 5): supported/refuted/unclear facts
- Coverage feedback (Agent 6): missing aspects, suggestions

**Process:**
1. Filter answer to SUPPORTED facts only
2. Identify coverage gaps from Agent 6
3. Retrieve additional evidence for missing aspects
4. Generate content for missing aspects
5. Merge and deduplicate
6. Reorganize for coherence
7. Output improved answer

**Output:**
- Improved answer with verified facts + expanded coverage
- Metadata on changes made
- Revised factuality and coverage scores

---

## Files

- ✅ **Updated**: `step6_coverage_feedback.py` (543 lines)
- ✅ **Documented**: `docs/STEP6_DUAL_LEVEL_COVERAGE.md`
- ✅ **Output**: `/temp/coverage_feedback.json`

---

## Example Use Case

**Query:** "What causes severe swelling and pain in the knees?"

**Step 6 Analysis:**

**Plan Coverage (0.80):**
- ✅ Anatomical causes
- ✅ Medical conditions
- ✅ Inflammatory mechanisms
- ✅ Trauma and injuries
- ✅ Lifestyle factors
- ❌ **Prevention strategies** (missing)
- ❌ **Prognosis and recovery** (missing)

**Answer Coverage (0.90):**
- ✅ Anatomical: well-covered
- ✅ Medical conditions: well-covered
- ✅ Inflammatory: well-covered
- ✅ Trauma: well-covered
- ⚠️ Lifestyle: partially-covered (needs expansion)

**Suggestions:**
1. [HIGH] Add prevention aspect to plan
2. [HIGH] Add prognosis aspect to plan
3. [MEDIUM] Expand lifestyle section in answer with specific recommendations

**Overall:** 0.87 - Good quality but improvable through iteration

