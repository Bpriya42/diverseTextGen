"""
Planner Node

LangGraph node wrapper for Agent 1 (Planner).
"""

import time

from state import RAGState
from agents.planner import generate_initial_plan, refine_plan_with_feedback


def planner_node(state: RAGState) -> RAGState:
    """
    Generate or refine query decomposition plan.
    
    First iteration: Generate initial plan
    Later iterations: Refine based on feedback
    
    Args:
        state: Current RAGState
        
    Returns:
        Updated RAGState with new plan
    """
    iteration = state["iteration"]
    query = state["query"]
    
    print(f"[Planner] Processing iteration {iteration}")
    start_time = time.time()
    
    if iteration == 0:
        # Initial planning
        plan = generate_initial_plan(query, iteration=iteration)
        print(f"[Planner] Generated initial plan with {len(plan)} aspects")
    else:
        # Refinement based on feedback
        factual_feedback = state["factual_feedback"]
        coverage_feedback = state["coverage_feedback"]
        current_plan = state["plan"]
        
        plan = refine_plan_with_feedback(
            query=query,
            current_plan=current_plan,
            factual_feedback=factual_feedback,
            coverage_feedback=coverage_feedback,
            iteration=iteration
        )
        print(f"[Planner] Refined plan to {len(plan)} aspects")
    
    elapsed = time.time() - start_time
    
    return {
        "plan": plan,
        "timestamps": {
            **state.get("timestamps", {}),
            f"planner_iter{iteration}": elapsed
        }
    }

