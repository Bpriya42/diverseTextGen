"""
Fact Extractor Node

LangGraph node wrapper for Agent 4 (Fact Extractor).
"""

import time

from state import RAGState
from agents.fact_extractor import extract_atomic_facts


def fact_extractor_node(state: RAGState) -> RAGState:
    """
    Extract atomic facts from the generated answer.
    
    Args:
        state: Current RAGState
        
    Returns:
        Updated RAGState with atomic facts
    """
    answer = state["answer"]
    iteration = state["iteration"]
    
    print(f"[Fact Extractor] Extracting facts")
    start_time = time.time()
    
    atomic_facts = extract_atomic_facts(answer)
    
    elapsed = time.time() - start_time
    print(f"[Fact Extractor] Extracted {len(atomic_facts)} atomic facts")
    
    return {
        "atomic_facts": atomic_facts,
        "timestamps": {
            **state.get("timestamps", {}),
            f"fact_extractor_iter{iteration}": elapsed
        }
    }

