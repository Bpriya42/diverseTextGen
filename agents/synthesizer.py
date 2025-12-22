"""
Agent 3: Answer Synthesizer

Generates comprehensive answers from plan and retrieved documents.
"""

import json
import re
from typing import List, Dict

from vllm import SamplingParams

from config.settings import (
    DEFAULT_MODEL, LLM_MAX_RETRIES, LLM_NUM_WORKERS,
    SYNTHESIZER_TEMPERATURE, SYNTHESIZER_MAX_TOKENS
)
from llm.server_llm import ServerLLM, load_url_from_log_file
from prompts.synthesizer_prompts import ANSWER_SYNTHESIS_TEMPLATE


# Module-level LLM instance
_llm_instance = None
_sampling_params = None


def get_llm():
    """Get or create shared LLM instance for synthesis."""
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
            temperature=SYNTHESIZER_TEMPERATURE,
            max_tokens=SYNTHESIZER_MAX_TOKENS
        )
    
    return _llm_instance, _sampling_params


def parse_json_block(text):
    """Parse JSON block from LLM output."""
    if not text:
        return []
    if isinstance(text, list) and len(text) == 1 and isinstance(text[0], str):
        text = text[0]
    if isinstance(text, list):
        return text
    
    cleaned = re.sub(r"```json|```", "", text, flags=re.IGNORECASE).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        return []


def build_prompt(query: str, plan: List[Dict], retrieval: List[Dict]) -> str:
    """
    Build synthesis prompt from query, plan, and retrieval.
    
    Args:
        query: Original user query
        plan: List of aspects
        retrieval: Retrieved documents per aspect
        
    Returns:
        Formatted prompt string
    """
    # Parse plan if needed
    if plan and isinstance(plan[0], str):
        plan = parse_json_block(plan[0]) if plan else []
    
    # Build plan summary
    plan_text = "\n".join([
        f"- **{p.get('aspect', 'N/A')}**: {p.get('reason', '')}"
        for p in (plan if isinstance(plan, list) else [])
    ])
    
    # Build retrieval context
    retrieval_texts = []
    for step in retrieval:
        aspect = step.get("aspect", "General")
        subq = step.get("subquery", "")
        docs = step.get("retrieved_docs", [])
        doc_texts = "\n".join([f"- {d['text']}" for d in docs])
        retrieval_texts.append(f"### {aspect}\nSubquery: {subq}\nContext:\n{doc_texts}")
    
    retrieved_text = "\n\n".join(retrieval_texts)
    
    # Build prompt from template
    prompt = ANSWER_SYNTHESIS_TEMPLATE.format(
        query=query,
        plan_text=plan_text,
        retrieved_text=retrieved_text
    )
    
    return prompt.strip()


def generate_answer(query: str, plan: List[Dict], retrieval: List[Dict]) -> str:
    """
    Generate comprehensive answer from query, plan, and retrieved documents.
    
    Args:
        query: Original user query
        plan: Query decomposition plan
        retrieval: Retrieved documents per aspect
        
    Returns:
        Generated answer string
    """
    llm, sampling_params = get_llm()
    
    # Build prompt
    prompt = build_prompt(query, plan, retrieval)
    
    # Generate answer
    response = llm.generate(
        [[{"role": "user", "content": prompt}]],
        sampling_params
    )[0]
    
    answer_text = response.outputs[0].text.strip()
    
    return answer_text

