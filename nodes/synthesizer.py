"""
Synthesizer Node

LangGraph node wrapper for Agent 3 (Synthesizer).
"""

import time

from state import RAGState
from agents.synthesizer import generate_answer


def synthesizer_node(state: RAGState) -> RAGState:
    """
    Generate answer from plan and retrieved documents.
    
    Args:
        state: Current RAGState
        
    Returns:
        Updated RAGState with generated answer
    """
    query = state["query"]
    plan = state["plan"]
    retrieval = state["retrieval"]
    iteration = state["iteration"]
    
    print(f"[Synthesizer] Generating answer")
    start_time = time.time()
    
    answer = generate_answer(query, plan, retrieval)
    
    elapsed = time.time() - start_time
    print(f"[Synthesizer] Generated answer ({len(answer)} chars)")
    
    return {
        "answer": answer,
        "timestamps": {
            **state.get("timestamps", {}),
            f"synthesizer_iter{iteration}": elapsed
        }
    }

