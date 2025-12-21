"""
LangGraph Workflow Construction

Builds the multi-agent RAG system workflow with parallel execution.
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.memory import MemorySaver
import sqlite3

from state import RAGState
from nodes.planner import planner_node
from nodes.retriever import retriever_node
from nodes.synthesizer import synthesizer_node
from nodes.fact_extractor import fact_extractor_node
from nodes.parallel_verification import parallel_evaluation_node
from nodes.iteration_gate import iteration_gate_node
from config.settings import CHECKPOINTER_PATH


def should_continue_routing(state: RAGState) -> str:
    """
    Routing function for conditional edge.
    
    Args:
        state: Current RAGState
        
    Returns:
        "continue" to loop back to planner, "end" to terminate
    """
    should_cont = state.get("should_continue", False)
    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", 3)
    
    print(f"[Router] should_continue={should_cont}, iteration={iteration}, max={max_iter}")
    
    return "continue" if should_cont else "end"


def build_graph(checkpointer_path: str = None, use_memory_checkpointer: bool = True):
    """
    Build and compile the LangGraph workflow.
    
    The workflow includes:
    1. Planner - generates/refines query decomposition
    2. Retriever - retrieves documents per aspect
    3. Synthesizer - generates answer
    4. Fact Extractor - extracts atomic facts
    5. Parallel Evaluation - runs Verifier and Coverage Evaluator concurrently
    6. Iteration Gate - decides whether to continue or terminate
    
    Args:
        checkpointer_path: Path to SQLite checkpoint file
        use_memory_checkpointer: If True, use in-memory checkpointer (recommended)
                                 If False, use SQLite checkpointer
        
    Returns:
        Compiled LangGraph application
    """
    # Create state graph
    workflow = StateGraph(RAGState)
    
    # Add nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("synthesizer", synthesizer_node)
    workflow.add_node("fact_extractor", fact_extractor_node)
    workflow.add_node("parallel_evaluation", parallel_evaluation_node)
    workflow.add_node("iteration_gate", iteration_gate_node)
    
    # Set entry point
    workflow.set_entry_point("planner")
    
    # Define linear flow within iteration
    workflow.add_edge("planner", "retriever")
    workflow.add_edge("retriever", "synthesizer")
    workflow.add_edge("synthesizer", "fact_extractor")
    workflow.add_edge("fact_extractor", "parallel_evaluation")
    workflow.add_edge("parallel_evaluation", "iteration_gate")
    
    # Conditional edge for iteration loop
    workflow.add_conditional_edges(
        "iteration_gate",
        should_continue_routing,
        {
            "continue": "planner",  # Loop back
            "end": END              # Terminate
        }
    )
    
    # Compile with checkpointer
    # Use memory checkpointer by default to avoid SQLite serialization issues
    if use_memory_checkpointer:
        print("[Graph] Using in-memory checkpointer")
        checkpointer = MemorySaver()
    else:
        checkpointer_path = checkpointer_path or CHECKPOINTER_PATH
        print(f"[Graph] Using SQLite checkpointer at {checkpointer_path}")
        conn = sqlite3.connect(checkpointer_path, check_same_thread=False)
        checkpointer = SqliteSaver(conn)
    
    app = workflow.compile(checkpointer=checkpointer)
    
    return app


def visualize_graph(app, output_path: str = "./docs/langgraph_visualization.png"):
    """
    Generate and save graph visualization.
    
    Args:
        app: Compiled LangGraph application
        output_path: Where to save the visualization
    """
    try:
        png_bytes = app.get_graph().draw_mermaid_png()
        
        with open(output_path, "wb") as f:
            f.write(png_bytes)
        
        print(f"Graph visualization saved to {output_path}")
        return True
    except Exception as e:
        print(f"Could not generate visualization: {e}")
        return False


if __name__ == "__main__":
    # Test building the graph
    print("Building LangGraph workflow...")
    app = build_graph()
    print("Graph built successfully!")
    
    # Try to visualize
    visualize_graph(app)

