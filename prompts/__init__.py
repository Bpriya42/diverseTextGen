"""
Centralized Prompt Templates for Multi-Agent RAG System

This module contains all prompt templates used by the 6 agents, making them
easy to modify, test, and version control.
"""

from prompts.planner_prompts import (
    INITIAL_PLAN_TEMPLATE,
    REFINEMENT_PLAN_TEMPLATE
)
from prompts.synthesizer_prompts import (
    ANSWER_SYNTHESIS_TEMPLATE
)
from prompts.fact_extractor_prompts import (
    FACT_EXTRACTION_PROMPT
)
from prompts.verifier_prompts import (
    VERIFICATION_PROMPT,
    FACTUALITY_SUMMARY_PROMPT
)
from prompts.coverage_prompts import (
    COVERAGE_EVALUATION_PROMPT
)

__all__ = [
    # Planner
    "INITIAL_PLAN_TEMPLATE",
    "REFINEMENT_PLAN_TEMPLATE",
    # Synthesizer
    "ANSWER_SYNTHESIS_TEMPLATE",
    # Fact Extractor
    "FACT_EXTRACTION_PROMPT",
    # Verifier
    "VERIFICATION_PROMPT",
    "FACTUALITY_SUMMARY_PROMPT",
    # Coverage Evaluator
    "COVERAGE_EVALUATION_PROMPT",
]

