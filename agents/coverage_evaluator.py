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
from prompts.coverage_prompts import COVERAGE_EVALUATION_PROMPT


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


# Note: COVERAGE_PROMPT_TEMPLATE is now imported from prompts module as COVERAGE_EVALUATION_PROMPT
# Keep legacy variable name for backwards compatibility
COVERAGE_PROMPT_TEMPLATE = COVERAGE_EVALUATION_PROMPT


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

