#!/usr/bin/env python3
"""
Run full experiments on training data with ICAT tracking.

This script:
1. Runs LangGraph on all training queries
2. Evaluates with ICAT
3. Tracks scores using ExperimentTracker
4. Generates visualizations

Termination is controlled by quality metrics and memory constraints only.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

# Add project root to path when running from scripts/
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import OUTPUT_DIR, CORPUS_PATH
from scripts.run_langgraph import run_query
from eval.icat import ICAT
from eval.experiment_tracker import ExperimentTracker


def run_full_experiment(
    queries_path: str,
    run_id: Optional[str] = None,
    description: str = "",
    n: Optional[int] = None,
    max_ram_percent: Optional[float] = None,
    max_gpu_percent: Optional[float] = None,
    corpus_path: Optional[str] = None,
    experiments_dir: Optional[str] = None,
    resume_from: Optional[int] = None
):
    """
    Run complete experiment with tracking and visualization.
    
    Args:
        queries_path: Path to JSONL with queries
        run_id: Unique run identifier
        description: Run description
        n: Number of queries (None = all)
        max_ram_percent: Maximum RAM usage percentage before termination
        max_gpu_percent: Maximum GPU memory usage percentage before termination
        corpus_path: Corpus file path
        experiments_dir: Where to save experiments
        resume_from: Resume from query index
    """
    # Setup paths
    corpus_path = corpus_path or CORPUS_PATH
    experiments_dir = experiments_dir or f"{OUTPUT_DIR}/experiments"
    
    # Initialize tracker
    tracker = ExperimentTracker(experiments_dir)
    
    # Load queries
    print(f"\nLoading queries from: {queries_path}")
    queries = []
    with open(queries_path, 'r') as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line.strip()))
    
    if n is not None:
        queries = queries[:n]
    
    total_queries = len(queries)
    print(f"Total queries: {total_queries}\n")
    
    # Create run
    run_config = {
        "queries_path": queries_path,
        "n_queries": total_queries,
        "max_ram_percent": max_ram_percent,
        "max_gpu_percent": max_gpu_percent,
        "corpus_path": corpus_path,
        "termination_mode": "quality_and_memory"
    }
    
    run_id = tracker.create_run(run_id=run_id, config=run_config, description=description)
    print(f"Created run: {run_id}\n")
    
    # Initialize ICAT once
    print("Initializing ICAT evaluator...")
    icat = ICAT(corpus_path=corpus_path, queries_path=None, qrels_path=None, use_web_search=False)
    print("✓ ICAT initialized\n")
    
    print("="*80)
    print("INCREMENTAL SAVING ENABLED")
    print("Results saved after EACH query to: {}/{}/results.jsonl".format(experiments_dir, run_id))
    print("="*80 + "\n")
    
    # Process queries
    start_idx = resume_from or 0
    successful = 0
    failed = 0
    
    for idx in range(start_idx, total_queries):
        query_data = queries[idx]
        query_id = query_data.get("query_id", f"query_{idx}")
        query_text = query_data.get("query_description") or query_data.get("query", "")
        
        print(f"\n{'#'*80}")
        print(f"QUERY {idx + 1}/{total_queries} - Progress: {(idx + 1)/total_queries*100:.1f}%")
        print(f"Query ID: {query_id}")
        print(f"{'#'*80}")
        
        try:
            # Run RAG system
            print(f"\n[1/2] Running RAG system...")
            rag_output_dir = Path(experiments_dir) / run_id / "rag_outputs"
            rag_output_dir.mkdir(parents=True, exist_ok=True)
            
            rag_result = run_query(
                query=query_text,
                query_id=query_id,
                max_ram_percent=max_ram_percent,
                max_gpu_percent=max_gpu_percent,
                output_path=str(rag_output_dir / f"{query_id}.json")
            )
            
            if "error" in rag_result:
                print(f"✗ RAG failed: {rag_result['error']}")
                failed += 1
                continue
            
            # Evaluate with ICAT
            print(f"\n[2/2] Evaluating with ICAT...")
            final_answer = rag_result.get("final_answer", "")
            
            icat_results, _ = icat.icat_score_a(
                model_responses=[final_answer],
                query_ids=[query_id],
                queries=[query_text]
            )
            
            icat_metrics = icat_results[0]["metrics"]
            
            # Track results (saved immediately to disk)
            tracker.log_query_result(
                run_id=run_id,
                query_id=query_id,
                query=query_text,
                icat_scores=icat_metrics,
                rag_metrics={
                    "total_iterations": rag_result.get("total_iterations", 0),
                    "runtime_seconds": rag_result.get("total_runtime_seconds", 0),
                    "termination_reason": rag_result.get("termination_reason", "unknown"),
                    "response_text": final_answer,
                    "response_length": len(final_answer)
                }
            )
            
            print(f"\n✓ Query completed and SAVED to disk")
            print(f"  Coverage: {icat_metrics['coverage']:.3f}, "
                  f"Factuality: {icat_metrics['factuality']:.3f}, "
                  f"F1: {icat_metrics['f1']:.3f}")
            
            successful += 1
            
        except Exception as e:
            print(f"\n✗ Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
        
        print(f"\nProgress: {successful} successful, {failed} failed")
    
    # Finalize run
    print(f"\n{'='*80}")
    print("Finalizing experiment...")
    print(f"{'='*80}")
    
    # Mark as "partial" if some queries failed, "completed" if all succeeded
    final_status = "completed" if failed == 0 else "partial"
    tracker.finalize_run(run_id, status=final_status)
    
    summary = tracker.get_run_summary(run_id)
    stats = summary.get("aggregate_stats", {})
    
    print(f"\nExperiment Complete: {run_id}")
    print(f"  Total Queries: {total_queries}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"\nAggregate ICAT Scores:")
    print(f"  Coverage:   {stats.get('avg_coverage', 0):.4f}")
    print(f"  Factuality: {stats.get('avg_factuality', 0):.4f}")
    print(f"  F1:         {stats.get('avg_f1', 0):.4f}")
    
    print(f"\n{'='*80}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*80}")
    print(f"Results: {experiments_dir}/{run_id}")
    print(f"\nTo generate comparison visualizations, use:")
    print(f"  from eval.visualizer import ICATVisualizer")
    print(f"  viz = ICATVisualizer('{experiments_dir}')")
    print(f"  viz.compare_runs(run_ids=['{run_id}', ...], output_dir='output/comparisons/')")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run full experiment with ICAT tracking"
    )
    
    parser.add_argument(
        "--queries_path",
        type=str,
        required=True,
        help="Path to JSONL with queries"
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Run identifier (auto-generated if not provided)"
    )
    parser.add_argument(
        "--description",
        type=str,
        default="",
        help="Run description"
    )
    parser.add_argument(
        "-n",
        type=int,
        default=None,
        help="Number of queries (default: all)"
    )
    parser.add_argument(
        "--max_ram_percent",
        type=float,
        default=None,
        help="Max RAM usage percentage before termination (default: 90)"
    )
    parser.add_argument(
        "--max_gpu_percent",
        type=float,
        default=None,
        help="Max GPU memory usage percentage before termination (default: 90)"
    )
    parser.add_argument(
        "--corpus_path",
        type=str,
        default=None,
        help=f"Corpus path (default: {CORPUS_PATH})"
    )
    parser.add_argument(
        "--experiments_dir",
        type=str,
        default=None,
        help=f"Experiments directory (default: {OUTPUT_DIR}/experiments)"
    )
    parser.add_argument(
        "--resume_from",
        type=int,
        default=None,
        help="Resume from query index"
    )
    
    args = parser.parse_args()
    
    run_full_experiment(
        queries_path=args.queries_path,
        run_id=args.run_id,
        description=args.description,
        n=args.n,
        max_ram_percent=args.max_ram_percent,
        max_gpu_percent=args.max_gpu_percent,
        corpus_path=args.corpus_path,
        experiments_dir=args.experiments_dir,
        resume_from=args.resume_from
    )


if __name__ == "__main__":
    main()
