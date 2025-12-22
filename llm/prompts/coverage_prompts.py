"""
Prompts for Agent 6: Coverage Evaluator

Evaluates plan and answer coverage quality.
"""


COVERAGE_PROMPT_TEMPLATE = """You are an expert evaluator assessing the topic coverage quality at two levels:
1. Whether the PLAN adequately covers all salient points of the query
2. Whether the ANSWER adequately covers all aspects of the plan and query

Query:
{query}

Planned Aspects (from query decomposition):
{plan_aspects}

Generated Answer:
{answer}

Your task is to provide comprehensive written feedback at three levels:

**LEVEL 1 - PLAN QUALITY:**
Evaluate whether the plan identifies all important aspects/perspectives that should be addressed to comprehensively answer the query.

**LEVEL 2 - ANSWER QUALITY:**
Evaluate whether the answer addresses each planned aspect adequately and covers the query comprehensively.

**LEVEL 3 - OVERALL COVERAGE QUALITY:**
Provide an integrated assessment of how well the plan and answer work together to address the query.

CRITICAL INSTRUCTION FOR QUALITY-BASED TERMINATION:
- If the answer COMPREHENSIVELY covers the query with NO significant gaps:
  - Set "no_improvements_needed" to true
  - Leave "missing_salient_points" as an empty array []
  - Ensure ALL items in "aspect_coverage_details" have "coverage_status": "well-covered"
  - Set all "points_for_improvement" fields to "No improvements needed"
  - Set all "critical_points_not_covered" fields to "None - coverage is comprehensive"
- If there ARE coverage gaps or missing aspects:
  - Set "no_improvements_needed" to false
  - List specific missing points in "missing_salient_points"
  - Provide actionable improvement suggestions
- Do NOT suggest unnecessary improvements if the coverage is already comprehensive
- Be honest: if the answer truly covers everything important, acknowledge it

Provide your evaluation in the following JSON format:
```json
{{
  "no_improvements_needed": <true if coverage is comprehensive with no gaps, false otherwise>,
  "plan_quality": {{
    "what_is_good": "<what aspects of the plan are well-designed and comprehensive>",
    "critical_points_not_covered": "<important aspects missing from plan, or 'None - plan is comprehensive'>",
    "points_for_improvement": "<suggestions for plan, or 'No improvements needed'>"
  }},
  "answer_quality": {{
    "what_is_good": "<what aspects of the answer are well-covered and demonstrate good depth/breadth>",
    "critical_points_not_covered": "<aspects missing from answer, or 'None - answer is comprehensive'>",
    "points_for_improvement": "<suggestions for answer, or 'No improvements needed'>"
  }},
  "overall_quality": {{
    "what_is_good": "<overall strengths of the plan-answer combination in addressing the query>",
    "critical_points_not_covered": "<key aspects inadequately addressed, or 'None - coverage is comprehensive'>",
    "points_for_improvement": "<improvement recommendations, or 'No improvements needed'>"
  }},
  "covered_salient_points": [
    "<salient point 1 that IS adequately covered>",
    ...
  ],
  "missing_salient_points": [
    "<important aspect/perspective missing or inadequately covered, or empty array [] if none>"
  ],
  "aspect_coverage_details": [
    {{
      "aspect": "<planned aspect name>",
      "coverage_status": "well-covered|partially-covered|not-covered",
      "explanation": "<how this aspect is addressed in the answer>"
    }},
    ...
  ]
}}
```

Output only the JSON block, nothing else."""

