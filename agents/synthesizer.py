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
from llm.prompts.synthesizer_prompts import format_synthesizer_prompt


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
    
    # Build prompt using centralized template
    prompt = format_synthesizer_prompt(query, plan, retrieval)
    
    # Generate answer
    response = llm.generate(
        [[{"role": "user", "content": prompt}]],
        sampling_params
    )[0]
    
    answer_text = response.outputs[0].text.strip()
    
    return answer_text

