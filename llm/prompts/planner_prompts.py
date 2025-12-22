"""
Prompts for Agent 1: Query Planner

Handles query decomposition and plan refinement.
"""

import json
from typing import List, Dict


PLANNER_REFINEMENT_PROMPT = """You are refining a query decomposition plan based on comprehensive feedback from fact verification and coverage evaluation.

Original Query: {query}

Current Plan:
{current_plan}

FEEDBACK ANALYSIS:

1. MISSING SALIENT POINTS (High Priority - Add New Aspects):
{missing_salient}

2. REFUTED FACTS (Critical - Add Exclusion Instructions):
{refuted_facts}

3. UNCLEAR FACTS (Moderate Priority - Improve Evidence Gathering):
{unclear_facts}

4. PLAN QUALITY IMPROVEMENTS:
{plan_improvements}

REFINEMENT INSTRUCTIONS:

A. ADD aspects for missing salient points (highest priority)
B. MODIFY existing aspects to exclude contradicted information with explicit instructions like:
   - "Exclude information about [specific contradicted topic]"
   - "Focus on [alternative reliable sources/perspectives]" 
C. ENHANCE aspects that led to unclear facts by:
   - Adding more specific retrieval queries
   - Targeting authoritative sources
   - Including evidence requirements
D. REMOVE redundant or ineffective aspects
E. Maximum 7 aspects total

CRITICAL JSON FORMATTING REQUIREMENTS:
- Output ONLY valid JSON array, nothing else
- Enclose your output in ```json and ``` markers
- Use double quotes for all strings (not single quotes)
- Escape special characters in strings (use \\" for quotes, \\n for newlines)
- Ensure all strings are properly closed
- No trailing commas after the last item
- Each object must have exactly these three fields: "aspect", "query", "reason"

Output format:
```json
[
  {{"aspect": "...", "query": "...", "reason": "..."}},
  {{"aspect": "...", "query": "...", "reason": "..."}}
]
```

Example of valid JSON:
```json
[
  {{"aspect": "Medical causes", "query": "What medical conditions cause symptoms?", "reason": "Identifies specific diseases"}},
  {{"aspect": "Treatment options", "query": "How to treat the condition?", "reason": "Provides actionable solutions"}}
]```"""


def format_planner_refinement_prompt(
    query: str,
    current_plan: List[Dict],
    factual_feedback: Dict,
    coverage_feedback: Dict
) -> str:
    """
    Format the planner refinement prompt with actual data.
    
    Args:
        query: Original user query
        current_plan: Current plan to refine
        factual_feedback: Feedback from Agent 5 (verifier)
        coverage_feedback: Feedback from Agent 6 (coverage evaluator)
        
    Returns:
        Formatted prompt string
    """
    # Extract feedback information
    written_feedback = factual_feedback.get("written_feedback", {})
    refuted_facts_details = written_feedback.get("refuted_facts_details", [])
    unclear_facts_details = written_feedback.get("unclear_facts_details", [])
    
    missing_salient = coverage_feedback.get("missing_salient_points", [])
    plan_quality = coverage_feedback.get("plan_quality", {})
    plan_improvements = plan_quality.get("points_for_improvement", "")
    
    # Format missing salient points
    missing_salient_text = "\n".join([f'   - {point}' for point in missing_salient[:5]]) if missing_salient else '   None identified'
    
    # Format refuted facts
    refuted_text = "\n".join([
        f'   - Fact: {detail["fact"][:80]}{"..." if len(detail["fact"]) > 80 else ""}\n'
        f'     Contradiction: {detail["contradiction"]}\n'
        f'     Action: {detail["suggested_correction"]}'
        for detail in refuted_facts_details[:3]
    ]) if refuted_facts_details else '   None identified'
    
    # Format unclear facts
    unclear_text = "\n".join([
        f'   - Fact: {detail["fact"][:80]}{"..." if len(detail["fact"]) > 80 else ""}\n'
        f'     Reason Unclear: {detail["reason"]}\n'
        f'     Evidence Needed: {detail["needed_evidence"]}'
        for detail in unclear_facts_details[:3]
    ]) if unclear_facts_details else '   None identified'
    
    return PLANNER_REFINEMENT_PROMPT.format(
        query=query,
        current_plan=json.dumps(current_plan, indent=2),
        missing_salient=missing_salient_text,
        refuted_facts=refuted_text,
        unclear_facts=unclear_text,
        plan_improvements=plan_improvements if plan_improvements else 'None specified'
    )

