#!/usr/bin/env python3
"""
Main runner for LangGraph multi-agent RAG system.

This script orchestrates the iterative RAG pipeline with parallel execution.
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Add project root to path when running from scripts/
sys.path.insert(0, str(Path(__file__).parent.parent))

from graph import build_graph
from state import init_state
from config.settings import OUTPUT_DIR, DEFAULT_MAX_ITERATIONS, DEFAULT_MAX_RAM_PERCENT, DEFAULT_MAX_GPU_PERCENT


def run_query(
    query: str,
    query_id: str,
    max_iterations: int = None,
    token_budget: int = None,
    walltime_budget_s: float = None,
    max_ram_percent: float = None,
    max_gpu_percent: float = None,
    output_path: str = None
):
    """
    Run the iterative RAG system on a single query.
    
    Args:
        query: User query string
        query_id: Unique query identifier
        max_iterations: Maximum number of iterations (None = unlimited)
        token_budget: Optional token budget
        walltime_budget_s: Optional wall-time budget (seconds)
        max_ram_percent: Maximum RAM usage percentage before termination
        max_gpu_percent: Maximum GPU memory usage percentage before termination
        output_path: Where to save results
        
    Returns:
        Result dictionary with final answer and metrics
    """
    output_path = output_path or f"{OUTPUT_DIR}/result.json"
    
    # Build graph
    print("\nBuilding LangGraph workflow...")
    app = build_graph()
    
    # Initialize state
    initial_state = init_state(
        query=query,
        query_id=query_id,
        max_iterations=max_iterations,
        token_budget=token_budget,
        walltime_budget_s=walltime_budget_s,
        max_ram_percent=max_ram_percent if max_ram_percent is not None else DEFAULT_MAX_RAM_PERCENT,
        max_gpu_percent=max_gpu_percent if max_gpu_percent is not None else DEFAULT_MAX_GPU_PERCENT
    )
    
    # Configure
    # Note: recursion_limit must be high enough to accommodate iterations
    # Each iteration has ~5-6 graph nodes, so limit = max_iterations * 10 (with buffer)
    # Default to 500 to allow ~50 iterations without hitting the limit
    recursion_limit = 500 if max_iterations is None else max(100, max_iterations * 10)
    config = {
        "configurable": {"thread_id": query_id},
        "recursion_limit": recursion_limit
    }
    
    # Print header
    print("\n" + "="*80)
    print("STARTING ITERATIVE RAG SYSTEM")
    print("="*80)
    print(f"Query ID: {query_id}")
    print(f"Query: {query}")
    if max_iterations is not None:
        print(f"Max Iterations: {max_iterations}")
    else:
        print(f"Max Iterations: Unlimited (budget-controlled)")
    if token_budget:
        print(f"Token Budget: {token_budget:,}")
    if walltime_budget_s:
        print(f"Time Budget: {walltime_budget_s}s")
    print(f"Memory Limits: RAM {initial_state['budget']['max_ram_percent']}%, GPU {initial_state['budget']['max_gpu_percent']}%")
    print(f"Graph Recursion Limit: {recursion_limit}")
    print("="*80 + "\n")
    
    # Run graph with streaming
    final_state = None
    start_time = time.time()
    last_iteration = -1
    
    # Setup output path for incremental saves (JSONL - one line per iteration)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    # Use .jsonl extension for iteration log
    iteration_log_file = output_file.parent / f"{output_file.stem}.jsonl"
    
    for event in app.stream(initial_state, config=config, stream_mode="values"):
        # Track progress - only print when iteration changes
        current_iter = event.get("iteration", 0)
        if current_iter != last_iteration:
            print(f"\n{'-'*80}")
            if max_iterations is not None:
                print(f"ITERATION {current_iter + 1} / {max_iterations}")
            else:
                print(f"ITERATION {current_iter + 1} (no limit)")
            print(f"{'-'*80}")
            
            # Append iteration result to JSONL file (one line per iteration)
            if last_iteration >= 0 and final_state is not None:
                elapsed_time = time.time() - start_time
                iteration_result = {
                    "query_id": query_id,
                    "iteration": last_iteration + 1,
                    "answer": final_state.get("answer", ""),
                    "answer_length": len(final_state.get("answer", "")),
                    "status": "in_progress",
                    "elapsed_seconds": round(elapsed_time, 2),
                    "timestamp": time.time()
                }
                # Append mode - each iteration is a new line
                with open(iteration_log_file, "a") as f:
                    f.write(json.dumps(iteration_result) + "\n")
                print(f"[Saved] Iteration {last_iteration + 1} appended to {iteration_log_file.name}")
            
            last_iteration = current_iter
        
        final_state = event
    
    # Calculate total time
    total_time = time.time() - start_time
    
    # Safety check for final_state
    if final_state is None:
        print("ERROR: No final state received from graph!")
        return {"error": "No final state", "query_id": query_id, "query": query}
    
    # Extract results with safe defaults
    result = {
        "query_id": query_id,
        "query": query,
        "final_answer": final_state.get("answer", ""),
        "total_iterations": len(final_state.get("history", [])),
        "termination_reason": final_state.get("termination_reason", "unknown"),
        "iteration_history": final_state.get("history", []),
        "total_runtime_seconds": total_time,
        "timestamps": final_state.get("timestamps", {}),
        "budget_used": final_state.get("budget", {})
    }
    
    # Append final completed entry to iteration log (JSONL)
    final_entry = {
        "query_id": query_id,
        "iteration": "final",
        "answer": final_state.get("answer", ""),
        "answer_length": len(final_state.get("answer", "")),
        "total_iterations": len(final_state.get("history", [])),
        "status": "completed",
        "total_runtime_seconds": round(total_time, 2),
        "termination_reason": final_state.get("termination_reason", "unknown"),
        "timestamp": time.time()
    }
    with open(iteration_log_file, "a") as f:
        f.write(json.dumps(final_entry) + "\n")
    
    # Also save full result as JSON (for detailed inspection)
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)
    
    # Print summary
    print("\n" + "="*80)
    print("COMPLETED")
    print("="*80)
    print(f"Total Iterations: {result['total_iterations']}")
    print(f"Total Runtime: {total_time:.2f}s")
    print(f"Termination Reason: {result['termination_reason']}")
    print(f"\nAnswer Preview:")
    print(f"  {result['final_answer'][:200]}...")
    print(f"\nResults saved to: {output_file}")
    print("="*80 + "\n")
    
    return result


def run_batch_from_jsonl(
    jsonl_path: str,
    n: int = None,
    max_iterations: int = None,
    token_budget: int = None,
    walltime_budget_s: float = None,
    max_ram_percent: float = None,
    max_gpu_percent: float = None,
    output_dir: str = None
):
    """
    Run the system on N queries from a JSONL file.
    
    Args:
        jsonl_path: Path to JSONL file with queries
        n: Number of queries to process (None = all)
        max_iterations: Maximum iterations per query (None = unlimited)
        token_budget: Token budget per query
        walltime_budget_s: Time budget per query
        max_ram_percent: Maximum RAM usage percentage
        max_gpu_percent: Maximum GPU memory usage percentage
        output_dir: Directory to save results
        
    Returns:
        List of results
    """
    output_dir = output_dir or f"{OUTPUT_DIR}/batch"
    
    # Load queries from JSONL
    queries = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line.strip()))
    
    # Limit to first N queries if specified
    if n is not None:
        queries = queries[:n]
    
    print(f"\nLoaded {len(queries)} queries from {jsonl_path}")
    print(f"Processing first {len(queries)} queries\n")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process each query
    results = []
    for idx, query_data in enumerate(queries, 1):
        query_id = query_data.get("query_id", f"query_{idx}")
        query_text = query_data.get("query_description") or query_data.get("query", "")
        
        print(f"\n{'='*80}")
        print(f"Processing Query {idx}/{len(queries)}: {query_id}")
        print(f"{'='*80}")
        
        try:
            result = run_query(
                query=query_text,
                query_id=query_id,
                max_iterations=max_iterations,
                token_budget=token_budget,
                walltime_budget_s=walltime_budget_s,
                max_ram_percent=max_ram_percent,
                max_gpu_percent=max_gpu_percent,
                output_path=str(output_path / f"{query_id}.json")
            )
            results.append(result)
        except Exception as e:
            print(f"\nError processing query {query_id}: {e}")
            results.append({
                "query_id": query_id,
                "query": query_text,
                "error": str(e),
                "status": "failed"
            })
    
    # Save batch summary
    summary_path = output_path / "batch_summary.json"
    summary = {
        "total_queries": len(queries),
        "successful": sum(1 for r in results if "error" not in r),
        "failed": sum(1 for r in results if "error" in r),
        "results": results
    }
    
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"Total Queries: {len(queries)}")
    print(f"Successful: {summary['successful']}")
    print(f"Failed: {summary['failed']}")
    print(f"Summary saved to: {summary_path}")
    print(f"{'='*80}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run iterative multi-agent RAG system with parallel execution"
    )
    
    # Batch mode arguments
    parser.add_argument("--input_file", type=str, help="JSONL file with queries")
    parser.add_argument("-n", type=int, help="Number of queries to process")
    
    # Single query arguments
    parser.add_argument("--query", type=str, help="User query")
    parser.add_argument("--query_id", type=str, help="Query ID")
    
    # Iteration control
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=DEFAULT_MAX_ITERATIONS,
        help="Maximum iterations (default: unlimited if not set)"
    )
    
    # Budget arguments
    parser.add_argument(
        "--token_budget",
        type=int,
        help="Token budget (optional)"
    )
    parser.add_argument(
        "--walltime_budget_s",
        type=float,
        help="Wall-time budget in seconds (optional)"
    )
    
    # Memory limits
    parser.add_argument(
        "--max_ram_percent",
        type=float,
        default=DEFAULT_MAX_RAM_PERCENT,
        help=f"Max RAM usage percentage before termination (default: {DEFAULT_MAX_RAM_PERCENT})"
    )
    parser.add_argument(
        "--max_gpu_percent",
        type=float,
        default=DEFAULT_MAX_GPU_PERCENT,
        help=f"Max GPU memory usage percentage before termination (default: {DEFAULT_MAX_GPU_PERCENT})"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path or directory"
    )
    
    args = parser.parse_args()
    
    # Determine mode
    if args.input_file:
        # Batch mode
        run_batch_from_jsonl(
            jsonl_path=args.input_file,
            n=args.n,
            max_iterations=args.max_iterations,
            token_budget=args.token_budget,
            walltime_budget_s=args.walltime_budget_s,
            max_ram_percent=args.max_ram_percent,
            max_gpu_percent=args.max_gpu_percent,
            output_dir=args.output
        )
    else:
        # Single query mode
        if not args.query or not args.query_id:
            parser.error("--query and --query_id required for single query mode, or use --input_file for batch mode")
        
        run_query(
            query=args.query,
            query_id=args.query_id,
            max_iterations=args.max_iterations,
            token_budget=args.token_budget,
            walltime_budget_s=args.walltime_budget_s,
            max_ram_percent=args.max_ram_percent,
            max_gpu_percent=args.max_gpu_percent,
            output_path=args.output
        )


if __name__ == "__main__":
    main()
