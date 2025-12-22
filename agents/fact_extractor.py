"""
Agent 4: Atomic Fact Extractor

Extracts atomic facts from generated answers.
"""

from typing import List

from vllm import SamplingParams

from config.settings import (
    DEFAULT_MODEL, LLM_MAX_RETRIES, LLM_NUM_WORKERS,
    VERIFIER_TEMPERATURE
)
from llm.server_llm import ServerLLM, load_url_from_log_file
from prompts.fact_extractor_prompts import FACT_EXTRACTION_PROMPT


# Module-level LLM instance
_llm_instance = None
_sampling_params = None


def get_llm():
    """Get or create shared LLM instance for fact extraction."""
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
            temperature=VERIFIER_TEMPERATURE,
            max_tokens=512
        )
    
    return _llm_instance, _sampling_params


def extract_atomic_facts(answer: str) -> List[str]:
    """
    Extract atomic facts from an answer.
    
    Args:
        answer: Generated answer text
        
    Returns:
        List of atomic fact strings
    """
    if not answer.strip():
        return []
    
    llm, sampling_params = get_llm()
    
    # Build prompt
    prompt = FACT_EXTRACTION_PROMPT.format(answer=answer.strip())
    
    # Generate facts
    response = llm.generate(
        [[{"role": "user", "content": prompt}]],
        sampling_params
    )[0]
    
    text_output = response.outputs[0].text.strip()
    
    # Parse output: split by lines and clean
    facts = [
        line.strip("-• ").strip()
        for line in text_output.split("\n")
        if line.strip()
    ]
    
    # Filter out trivial fragments
    facts = [f for f in facts if len(f.split()) > 2]
    
    return facts

