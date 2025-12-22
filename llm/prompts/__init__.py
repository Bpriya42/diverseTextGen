"""
Prompt templates for all agents in the RAG system.

This module centralizes all prompts for easy modification and testing.
"""

from llm.prompts.planner_prompts import (
    PLANNER_REFINEMENT_PROMPT,
    format_planner_refinement_prompt
)
from llm.prompts.synthesizer_prompts import (
    SYNTHESIZER_PROMPT_TEMPLATE,
    format_synthesizer_prompt
)
from llm.prompts.fact_extractor_prompts import (
    FACT_EXTRACTION_PROMPT
)
from llm.prompts.verifier_prompts import (
    VERIFICATION_PROMPT,
    FACTUALITY_SUMMARY_PROMPT
)
from llm.prompts.coverage_prompts import (
    COVERAGE_PROMPT_TEMPLATE
)

__all__ = [
    # Planner
    "PLANNER_REFINEMENT_PROMPT",
    "format_planner_refinement_prompt",
    
    # Synthesizer
    "SYNTHESIZER_PROMPT_TEMPLATE",
    "format_synthesizer_prompt",
    
    # Fact Extractor
    "FACT_EXTRACTION_PROMPT",
    
    # Verifier
    "VERIFICATION_PROMPT",
    "FACTUALITY_SUMMARY_PROMPT",
    
    # Coverage Evaluator
    "COVERAGE_PROMPT_TEMPLATE",
]

