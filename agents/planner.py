"""
Agent 1: Query Planner

Generates and refines query decomposition plans.
"""

import json
import re
from typing import List, Dict

from vllm import SamplingParams

from config.settings import (
    DEFAULT_MODEL, LLM_MAX_RETRIES, LLM_NUM_WORKERS,
    PLANNER_TEMPERATURE, PLANNER_MAX_TOKENS
)
from llm.server_llm import ServerLLM, load_url_from_log_file
from llm.prompts.planner_prompts import format_planner_refinement_prompt
from data.formatters import get_query_planning_formatter


# Module-level LLM instance (singleton pattern)
_llm_instance = None
_sampling_params = None


def get_llm():
    """Get or create shared LLM instance for planning."""
    global _llm_instance, _sampling_params
    
    if _llm_instance is None:
        url = load_url_from_log_file()
        _llm_instance = ServerLLM(
            base_url=url,
            model=DEFAULT_MODEL,
            max_retries=LLM_MAX_RETRIES,
            num_workers=LLM_NUM_WORKERS
        )
        _sampling_params = SamplingParams(
            temperature=PLANNER_TEMPERATURE,
            max_tokens=PLANNER_MAX_TOKENS
        )
    
    return _llm_instance, _sampling_params


def clean_plan_text(text: str) -> List[Dict]:
    """
    Clean and parse plan JSON from LLM output with robust error handling.
    
    Args:
        text: Raw LLM output text
        
    Returns:
        List of aspect dictionaries
    """
    # Step 1: Remove markdown code blocks
    cleaned = re.sub(r"```json|```", "", text, flags=re.IGNORECASE).strip()
    
    # Step 2: Try to extract JSON array from text
    json_match = re.search(r'\[[\s\S]*\]', cleaned)
    if json_match:
        cleaned = json_match.group(0)
    
    # Step 3: Try parsing
    try:
        plan = json.loads(cleaned)
        return plan if isinstance(plan, list) else []
    except json.JSONDecodeError as e:
        # Step 4: Try to fix common JSON issues
        try:
            start_idx = cleaned.find('[')
            if start_idx != -1:
                bracket_count = 0
                end_idx = start_idx
                for i in range(start_idx, len(cleaned)):
                    if cleaned[i] == '[':
                        bracket_count += 1
                    elif cleaned[i] == ']':
                        bracket_count -= 1
                        if bracket_count == 0:
                            end_idx = i + 1
                            break
                
                if end_idx > start_idx:
                    json_str = cleaned[start_idx:end_idx]
                    json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
                    plan = json.loads(json_str)
                    return plan if isinstance(plan, list) else []
        except:
            pass
        
        # Step 5: Try to extract individual JSON objects
        try:
            objects = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', cleaned)
            if objects:
                plan = []
                for obj_str in objects:
                    try:
                        obj = json.loads(obj_str)
                        if isinstance(obj, dict) and 'aspect' in obj:
                            plan.append(obj)
                    except:
                        continue
                if plan:
                    return plan
        except:
            pass
        
        print(f"Warning: Could not parse plan as JSON. Error: {e}")
        return []


def generate_initial_plan(query: str) -> List[Dict]:
    """
    Generate initial query decomposition plan.
    
    Args:
        query: User query string
        
    Returns:
        List of aspect dictionaries with keys: aspect, query, reason
    """
    llm, sampling_params = get_llm()
    formatter = get_query_planning_formatter()
    
    prompts = formatter({"query": [query]})
    responses = llm.generate(
        [[{"role": "user", "content": prompts[0]}]],
        sampling_params
    )
    
    raw_plan = responses[0].outputs[0].text
    plan = clean_plan_text(raw_plan)
    
    return plan


def refine_plan_with_feedback(
    query: str,
    current_plan: List[Dict],
    factual_feedback: Dict,
    coverage_feedback: Dict
) -> List[Dict]:
    """
    Refine plan based on factual and coverage feedback.
    
    Args:
        query: Original user query
        current_plan: Current plan to refine
        factual_feedback: Feedback from Agent 5 (verifier)
        coverage_feedback: Feedback from Agent 6 (coverage evaluator)
        
    Returns:
        Refined plan with added/modified aspects
    """
    llm, sampling_params = get_llm()
    
    # Extract feedback information
    written_feedback = factual_feedback.get("written_feedback", {})
    refuted_facts_details = written_feedback.get("refuted_facts_details", [])
    unclear_facts_details = written_feedback.get("unclear_facts_details", [])
    
    missing_salient = coverage_feedback.get("missing_salient_points", [])
    plan_quality = coverage_feedback.get("plan_quality", {})
    plan_improvements = plan_quality.get("points_for_improvement", "")
    
    # Build refinement prompt
    prompt = f"""You are refining a query decomposition plan based on comprehensive feedback from fact verification and coverage evaluation.

Original Query: {query}

Current Plan:
{json.dumps(current_plan, indent=2)}

FEEDBACK ANALYSIS:

1. MISSING SALIENT POINTS (High Priority - Add New Aspects):
{chr(10).join([f'   - {point}' for point in missing_salient[:5]]) if missing_salient else '   None identified'}

2. REFUTED FACTS (Critical - Add Exclusion Instructions):
{chr(10).join([f'   - Fact: {detail["fact"][:80]}{"..." if len(detail["fact"]) > 80 else ""}' + chr(10) + f'     Contradiction: {detail["contradiction"]}' + chr(10) + f'     Action: {detail["suggested_correction"]}' for detail in refuted_facts_details[:3]]) if refuted_facts_details else '   None identified'}

3. UNCLEAR FACTS (Moderate Priority - Improve Evidence Gathering):
{chr(10).join([f'   - Fact: {detail["fact"][:80]}{"..." if len(detail["fact"]) > 80 else ""}' + chr(10) + f'     Reason Unclear: {detail["reason"]}' + chr(10) + f'     Evidence Needed: {detail["needed_evidence"]}' for detail in unclear_facts_details[:3]]) if unclear_facts_details else '   None identified'}

4. PLAN QUALITY IMPROVEMENTS:
{plan_improvements if plan_improvements else 'None specified'}

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
```"""
    
    response = llm.generate(
        [[{"role": "user", "content": prompt}]],
        sampling_params
    )[0]
    
    refined_plan = clean_plan_text(response.outputs[0].text)
    
    # Enhanced fallback: augment current plan if parsing fails
    if not refined_plan:
        # Extract feedback information for fallback
        written_feedback = factual_feedback.get("written_feedback", {})
        refuted_facts_details = written_feedback.get("refuted_facts_details", [])
        unclear_facts_details = written_feedback.get("unclear_facts_details", [])
        missing_salient = coverage_feedback.get("missing_salient_points", [])
        refined_plan = current_plan.copy()
        
        # Add aspects for missing salient points
        for missing in missing_salient[:2]:
            refined_plan.append({
                "aspect": missing,
                "query": f"Information about {missing} for: {query[:50]}",
                "reason": "Identified as missing salient point in coverage evaluation"
            })
        
        # Modify aspects affected by refuted facts
        for detail in refuted_facts_details[:2]:
            for aspect in refined_plan:
                if any(word in aspect["query"].lower() for word in detail["fact"][:30].lower().split()[:3]):
                    aspect["query"] += f" (Exclude: {detail['contradiction'][:40]})"
                    aspect["reason"] += f"; Modified to exclude contradicted information"
                    break
        
        # Enhance aspects for unclear facts
        for detail in unclear_facts_details[:2]:
            for aspect in refined_plan:
                if any(word in aspect["query"].lower() for word in detail["fact"][:30].lower().split()[:3]):
                    aspect["query"] += f" (Focus on: {detail['needed_evidence'][:40]})"
                    aspect["reason"] += f"; Enhanced for better evidence gathering"
                    break
    
    return refined_plan

