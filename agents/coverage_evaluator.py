"""
Agent 6: Coverage Evaluator

Evaluates plan and answer coverage quality.
"""

import json
import re
from typing import List, Dict

from vllm import SamplingParams

from config.settings import (
    DEFAULT_MODEL, LLM_MAX_RETRIES, LLM_NUM_WORKERS,
    VERIFIER_TEMPERATURE
)
from llm.server_llm import ServerLLM, load_url_from_log_file


# Module-level LLM instance
_llm_instance = None
_sampling_params = None


def get_llm():
    """Get or create shared LLM instance for coverage evaluation."""
    global _llm_instance, _sampling_params
    
    if _llm_instance is None:
        url = load_url_from_log_file()
        _llm_instance = ServerLLM(
            base_url=url,
            model=DEFAULT_MODEL,
            max_retries=LLM_MAX_RETRIES,
            num_workers=LLM_NUM_WORKERS
        )
        _sampling_params = SamplingParams(temperature=VERIFIER_TEMPERATURE, max_tokens=512)
    
    return _llm_instance, _sampling_params


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


def parse_json_response(text):
    """Parse JSON response from LLM."""
    if not text:
        return None
    
    cleaned = re.sub(r"```json|```", "", text, flags=re.IGNORECASE).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        return {
            "plan_quality": {
                "what_is_good": "Unable to evaluate plan quality due to parsing error.",
                "critical_points_not_covered": "Cannot identify missing aspects due to evaluation failure.",
                "points_for_improvement": "Re-run evaluation to get specific improvement suggestions."
            },
            "answer_quality": {
                "what_is_good": "Unable to evaluate answer quality due to parsing error.",
                "critical_points_not_covered": "Cannot identify coverage gaps due to evaluation failure.",
                "points_for_improvement": "Re-run evaluation to get specific coverage improvements."
            },
            "overall_quality": {
                "what_is_good": "Unable to provide overall assessment due to parsing error.",
                "critical_points_not_covered": "Cannot identify overall coverage gaps due to evaluation failure.",
                "points_for_improvement": "Re-run evaluation to get comprehensive improvement recommendations."
            },
            "covered_salient_points": [],
            "missing_salient_points": ["Evaluation failed - cannot determine coverage"],
            "aspect_coverage_details": []
        }


def parse_plan_list(plan):
    """Parse plan from various formats into structured list."""
    if isinstance(plan, list):
        if len(plan) > 0 and isinstance(plan[0], dict):
            return plan
        if len(plan) > 0 and isinstance(plan[0], str):
            try:
                parsed = json.loads(plan[0])
                if isinstance(parsed, list):
                    return parsed
            except:
                pass
    
    if isinstance(plan, str):
        cleaned = re.sub(r"```json|```", "", plan, flags=re.IGNORECASE).strip()
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, list):
                return parsed
        except:
            pass
    
    return []


def format_plan_aspects(plan: List[Dict]) -> str:
    """Format plan aspects into readable text."""
    aspects = parse_plan_list(plan)
    
    if not aspects:
        return "No specific aspects were planned."
    
    formatted_parts = []
    for i, aspect_item in enumerate(aspects, 1):
        if isinstance(aspect_item, dict):
            aspect_name = aspect_item.get("aspect", f"Aspect {i}")
            reason = aspect_item.get("reason", "")
            query = aspect_item.get("query", "")
            
            formatted_parts.append(
                f"{i}. **{aspect_name}**\n"
                f"   - Query: {query}\n"
                f"   - Reason: {reason}"
            )
        else:
            formatted_parts.append(f"{i}. {aspect_item}")
    
    return "\n\n".join(formatted_parts)


def evaluate_coverage(query: str, plan: List[Dict], answer: str) -> Dict:
    """
    Evaluate dual-level coverage (plan + answer) with written feedback.
    
    Args:
        query: Original user query
        plan: Query decomposition plan
        answer: Generated answer
        
    Returns:
        Dictionary with written coverage evaluation results
    """
    llm, sampling_params = get_llm()
    
    # Format plan aspects
    plan_text = format_plan_aspects(plan)
    
    # Build prompt
    prompt = COVERAGE_PROMPT_TEMPLATE.format(
        query=query,
        plan_aspects=plan_text,
        answer=answer
    )
    
    # Generate coverage evaluation
    response = llm.generate(
        [[{"role": "user", "content": prompt}]],
        sampling_params
    )[0]
    
    response_text = response.outputs[0].text.strip()
    
    # Parse response
    coverage_eval = parse_json_response(response_text)
    
    # Enhance feedback if parsing succeeded but fields are missing
    if coverage_eval and "plan_quality" in coverage_eval:
        for level in ["plan_quality", "answer_quality", "overall_quality"]:
            if level not in coverage_eval:
                coverage_eval[level] = {
                    "what_is_good": "No assessment available.",
                    "critical_points_not_covered": "Unable to identify gaps.",
                    "points_for_improvement": "Re-evaluate for suggestions."
                }
            else:
                for field in ["what_is_good", "critical_points_not_covered", "points_for_improvement"]:
                    if field not in coverage_eval[level]:
                        coverage_eval[level][field] = f"No {field.replace('_', ' ')} assessment available."
        
        if "covered_salient_points" not in coverage_eval:
            coverage_eval["covered_salient_points"] = []
        if "missing_salient_points" not in coverage_eval:
            coverage_eval["missing_salient_points"] = []
        if "aspect_coverage_details" not in coverage_eval:
            coverage_eval["aspect_coverage_details"] = []
    
    # Add summary metrics
    missing_count = len(coverage_eval.get("missing_salient_points", []))
    covered_count = len(coverage_eval.get("covered_salient_points", []))
    total_aspects = len(plan) if plan else 1
    
    # Check if all aspects are well-covered
    aspect_details = coverage_eval.get("aspect_coverage_details", [])
    all_aspects_well_covered = all(
        a.get("coverage_status") == "well-covered" 
        for a in aspect_details
    ) if aspect_details else True
    
    # Determine if no improvements are needed based on actual metrics
    # (fallback if LLM didn't set the field correctly)
    llm_no_improve = coverage_eval.get("no_improvements_needed", None)
    computed_no_improve = (missing_count == 0 and all_aspects_well_covered)
    
    # Use LLM's assessment if available, otherwise compute from metrics
    no_improvements_needed = llm_no_improve if llm_no_improve is not None else computed_no_improve
    
    # Ensure the field is set at the top level
    coverage_eval["no_improvements_needed"] = no_improvements_needed
    
    coverage_eval["coverage_summary"] = {
        "total_planned_aspects": total_aspects,
        "covered_salient_points_count": covered_count,
        "missing_salient_points_count": missing_count,
        "all_aspects_well_covered": all_aspects_well_covered,
        "has_critical_gaps": missing_count > 0 or any("critical" in str(v).lower() for v in coverage_eval.get("overall_quality", {}).values())
    }
    
    return coverage_eval

