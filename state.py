"""
State schema for LangGraph workflow.

Defines the central state object that flows through all nodes.
"""

from typing import TypedDict, List, Dict, Optional, Any
import time


class RAGState(TypedDict):
    """
    Central state object for multi-agent RAG system.
    
    This state flows through all LangGraph nodes, with each node
    reading from and writing to specific fields.
    
    Note: We avoid using Annotated[..., add] reducers as they can cause
    issues with state accumulation across iterations when using checkpointers.
    """
    
    # Query and control
    query_id: str
    query: str
    iteration: int
    max_iterations: Optional[int]  # None means unlimited iterations
    
    # Budget tracking
    budget: Dict[str, Any]
    
    # Agent outputs
    plan: List[Dict[str, str]]
    retrieval: List[Dict[str, Any]]
    answer: str
    atomic_facts: List[str]
    factual_feedback: Dict[str, Any]
    coverage_feedback: Dict[str, Any]
    
    # History and termination
    # Note: history is managed manually in iteration_gate_node to avoid accumulation bugs
    history: List[Dict]
    should_continue: bool
    termination_reason: Optional[str]
    
    # Metadata
    timestamps: Dict[str, float]
    error_log: List[str]


def init_state(
    query: str,
    query_id: str,
    max_iterations: Optional[int] = None,
    token_budget: Optional[int] = None,
    walltime_budget_s: Optional[float] = None,
    max_ram_percent: Optional[float] = None,
    max_gpu_percent: Optional[float] = None
) -> RAGState:
    """
    Initialize state for a new query.
    
    Args:
        query: User query string
        query_id: Unique query identifier
        max_iterations: Maximum number of iterations (None = unlimited)
        token_budget: Optional token budget
        walltime_budget_s: Optional wall-time budget in seconds
        max_ram_percent: Maximum RAM usage percentage before termination (default: 90)
        max_gpu_percent: Maximum GPU memory usage percentage before termination (default: 90)
        
    Returns:
        Initialized RAGState
    """
    return RAGState(
        query_id=query_id,
        query=query,
        iteration=0,
        max_iterations=max_iterations,  # None means unlimited
        budget={
            "token_budget_total": token_budget,
            "tokens_used": 0,
            "walltime_budget_s": walltime_budget_s,
            "walltime_start_ts": time.time(),
            "tokens_per_iteration": [],
            "max_ram_percent": max_ram_percent if max_ram_percent is not None else 90.0,
            "max_gpu_percent": max_gpu_percent if max_gpu_percent is not None else 90.0
        },
        plan=[],
        retrieval=[],
        answer="",
        atomic_facts=[],
        factual_feedback={},
        coverage_feedback={},
        history=[],
        should_continue=True,
        termination_reason=None,
        timestamps={},
        error_log=[]
    )

