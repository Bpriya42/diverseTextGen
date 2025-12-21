"""
Retriever Node

LangGraph node wrapper for Agent 2 (Retriever).
"""

import time

from state import RAGState
from agents.retriever import retrieve_for_plan


def retriever_node(state: RAGState) -> RAGState:
    """
    Retrieve documents for each aspect in the plan.
    
    Args:
        state: Current RAGState
        
    Returns:
        Updated RAGState with retrieval results
    """
    plan = state["plan"]
    iteration = state["iteration"]
    
    print(f"[Retriever] Retrieving documents for {len(plan)} aspects")
    start_time = time.time()
    
    retrieval = retrieve_for_plan(plan, top_k=5)
    
    elapsed = time.time() - start_time
    print(f"[Retriever] Retrieved {sum(len(r.get('retrieved_docs', [])) for r in retrieval)} total documents")
    
    return {
        "retrieval": retrieval,
        "timestamps": {
            **state.get("timestamps", {}),
            f"retriever_iter{iteration}": elapsed
        }
    }

