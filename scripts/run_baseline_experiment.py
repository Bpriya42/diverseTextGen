#!/usr/bin/env python3
"""
Run baseline experiments with direct LLM responses (no RAG/agent flow).

This script:
1. Loads queries from JSONL
2. Sends queries directly to LLM (no retrieval, no iteration)
3. Evaluates responses with ICAT
4. Tracks scores using ExperimentTracker
5. Generates visualizations

This provides a baseline to compare against the full RAG system.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, List
from vllm import SamplingParams

# Add project root to path when running from scripts/
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import OUTPUT_DIR, CORPUS_PATH
from llm.server_llm import ServerLLM, load_url_from_log_file
from eval.icat import ICAT
from eval.experiment_tracker import ExperimentTracker


def generate_baseline_responses(
    queries: List[str],
    server_log_path: str,
    model: str,
    temperature: float = 0.7,
    max_tokens: int = 512,
    batch_size: int = 1
) -> List[str]:
    """
    Generate responses using direct LLM calls (no RAG).
    
    Args:
        queries: List of query strings
        server_log_path: Path to server log file
        model: Model name
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        batch_size: Number of queries to process at once
        
    Returns:
        List of generated responses
    """
    # Load server URL
    url = load_url_from_log_file(server_log_path)
    
    # Initialize LLM
    llm = ServerLLM(base_url=url, model=model, max_retries=10, num_workers=1)
    
    # Sampling parameters
    sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
    
    # Generate responses in batches
    responses = []
    total_queries = len(queries)
    
    for i in range(0, total_queries, batch_size):
        batch_queries = queries[i:i + batch_size]
        
        # Format as conversations (simple user query)
        batch_conversations = [
            [{"role": "user", "content": query}]
            for query in batch_queries
        ]
        
        print(f"Generating responses for queries {i+1}-{min(i+batch_size, total_queries)} of {total_queries}...")
        
        # Generate
        batch_responses = llm.generate(batch_conversations, sampling_params)
        
        # Extract text
        for response in batch_responses:
            responses.append(response.outputs[0].text)
    
    return responses


def run_baseline_experiment(
    queries_path: str,
    run_id: Optional[str] = None,
    description: str = "",
    n: Optional[int] = None,
    server_log_path: str = None,
    model: str = None,
    temperature: float = 0.7,
    max_tokens: int = 512,
    batch_size: int = 1,
    corpus_path: Optional[str] = None,
    experiments_dir: Optional[str] = None,
    resume_from: Optional[int] = None,
    incremental: bool = True
):
    """
    Run baseline experiment with direct LLM responses.
    
    Args:
        queries_path: Path to JSONL with queries
        run_id: Unique run identifier
        description: Run description
        n: Number of queries (None = all)
        server_log_path: Path to server log file
        model: Model name
        temperature: Sampling temperature
        max_tokens: Max tokens to generate
        batch_size: Batch size for generation
        corpus_path: Corpus file path (for ICAT)
        experiments_dir: Where to save experiments
        resume_from: Resume from query index
        incremental: If True, save results after each query (default: True)
    """
    # Validate queries file exists
    if not Path(queries_path).exists():
        raise FileNotFoundError(f"Queries file not found: {queries_path}")
    
    # Setup paths
    corpus_path = corpus_path or CORPUS_PATH
    experiments_dir = experiments_dir or f"{OUTPUT_DIR}/baseline_experiments"
    server_log_path = server_log_path or "/rstor/pi_hzamani_umass_edu/asalemi/priya/server_logs/log.txt"
    model = model or "Qwen/Qwen3-4B-Instruct-2507"
    
    # Initialize tracker
    tracker = ExperimentTracker(experiments_dir)
    
    # Load queries
    print(f"\nLoading queries from: {queries_path}")
    queries = []
    with open(queries_path, 'r') as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line.strip()))
    
    total_available = len(queries)
    
    # Apply limit if specified
    if n is not None:
        queries = queries[:n]
        print(f"Processing {len(queries)} out of {total_available} available queries")
    else:
        print(f"Processing ALL {len(queries)} queries from file")
    
    total_queries = len(queries)
    print(f"Total queries to process: {total_queries}\n")
    
    # Create run
    run_config = {
        "queries_path": queries_path,
        "total_available_queries": total_available,
        "n_queries_selected": total_queries,
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "batch_size": batch_size,
        "corpus_path": corpus_path,
        "experiment_type": "baseline_direct_llm",
        "incremental_saving": incremental
    }
    
    run_id = tracker.create_run(run_id=run_id, config=run_config, description=description or "baseline")
    print(f"Created baseline run: {run_id}\n")
    
    # Initialize ICAT once
    print("Initializing ICAT evaluator...")
    icat = ICAT(corpus_path=corpus_path, queries_path=None, qrels_path=None, use_web_search=False)
    print("✓ ICAT initialized\n")
    
    # Process queries
    start_idx = resume_from or 0
    successful = 0
    failed = 0
    
    # Handle resume: skip already processed queries
    if start_idx > 0:
        queries = queries[start_idx:]
        print(f"Resuming from query index {start_idx}")
    
    # Extract query texts and IDs
    query_ids = [q.get("query_id", f"query_{start_idx + i}") for i, q in enumerate(queries)]
    query_texts = [q.get("query_description") or q.get("query", "") for q in queries]
    
    print(f"\n{'='*80}")
    print("GENERATING BASELINE RESPONSES (Direct LLM)")
    print(f"{'='*80}\n")
    
    if incremental:
        # INCREMENTAL MODE: Process one query at a time, save after each
        print("Running in INCREMENTAL mode - results saved after each query\n")
        
        # Load server URL and initialize LLM once
        url = load_url_from_log_file(server_log_path)
        llm = ServerLLM(base_url=url, model=model, max_retries=10, num_workers=1)
        sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        
        for idx, (query_data, query_id, query_text) in enumerate(zip(queries, query_ids, query_texts)):
            try:
                print(f"\n--- Query {idx + 1}/{len(queries)}: {query_id} ---")
                print(f"Query: {query_text[:100]}...")
                
                # Generate response for single query
                conversation = [[{"role": "user", "content": query_text}]]
                response_obj = llm.generate(conversation, sampling_params)
                response_text = response_obj[0].outputs[0].text
                
                print(f"✓ Generated response ({len(response_text)} chars)")
                
                # Evaluate with ICAT immediately
                icat_results, _ = icat.icat_score_a(
                    model_responses=[response_text],
                    query_ids=[query_id],
                    queries=[query_text]
                )
                icat_metrics = icat_results[0]["metrics"]
                
                print(f"✓ ICAT: coverage={icat_metrics['coverage']:.3f}, "
                      f"factuality={icat_metrics['factuality']:.3f}, f1={icat_metrics['f1']:.3f}")
                
                # Track result immediately (this saves to disk)
                tracker.log_query_result(
                    run_id=run_id,
                    query_id=query_id,
                    query=query_text,
                    icat_scores=icat_metrics,
                    rag_metrics={
                        "response_length": len(response_text),
                        "response_text": response_text,
                        "experiment_type": "baseline_direct_llm"
                    }
                )
                
                print(f"✓ Saved result for query {idx + 1}/{len(queries)}")
                successful += 1
                
            except Exception as e:
                print(f"\n✗ Error processing query {query_id}: {e}")
                import traceback
                traceback.print_exc()
                failed += 1
                # Continue to next query instead of failing entirely
                continue
    else:
        # BATCH MODE: Original behavior - process all at once
        print("Running in BATCH mode - results saved after all queries complete\n")
        
        try:
            # Generate all responses
            responses = generate_baseline_responses(
                queries=query_texts,
                server_log_path=server_log_path,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                batch_size=batch_size
            )
            
            print(f"\n✓ Generated {len(responses)} responses\n")
            
            # Evaluate with ICAT
            print(f"\n{'='*80}")
            print("EVALUATING WITH ICAT")
            print(f"{'='*80}\n")
            
            icat_results, _ = icat.icat_score_a(
                model_responses=responses,
                query_ids=query_ids,
                queries=query_texts
            )
            
            print(f"\n✓ ICAT evaluation complete\n")
            
            # Track results
            print(f"\n{'='*80}")
            print("TRACKING RESULTS")
            print(f"{'='*80}\n")
            
            for idx, (query_data, icat_result) in enumerate(zip(queries, icat_results)):
                query_id = query_ids[idx]
                query_text = query_texts[idx]
                icat_metrics = icat_result["metrics"]
                
                # Track result
                tracker.log_query_result(
                    run_id=run_id,
                    query_id=query_id,
                    query=query_text,
                    icat_scores=icat_metrics,
                    rag_metrics={
                        "response_length": len(responses[idx]),
                        "response_text": responses[idx],
                        "experiment_type": "baseline_direct_llm"
                    }
                )
                
                print(f"✓ Tracked query {idx + 1}/{len(queries)}: {query_id} "
                      f"(F1: {icat_metrics['f1']:.3f})")
                
                successful += 1
            
            print(f"\n✓ All results tracked\n")
            
        except Exception as e:
            print(f"\n✗ Error: {e}")
            import traceback
            traceback.print_exc()
            failed = len(queries)
    
    # Finalize run
    print(f"\n{'='*80}")
    print("FINALIZING EXPERIMENT")
    print(f"{'='*80}")
    
    tracker.finalize_run(run_id, status="completed" if failed == 0 else "partial")
    
    summary = tracker.get_run_summary(run_id)
    stats = summary.get("aggregate_stats", {})
    
    print(f"\nBaseline Experiment Complete: {run_id}")
    print(f"  Query File: {queries_path}")
    print(f"  Model: {model}")
    print(f"  Total Queries: {total_queries}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"\nAggregate ICAT Scores:")
    print(f"  Coverage:   {stats.get('avg_coverage', 0):.4f}")
    print(f"  Factuality: {stats.get('avg_factuality', 0):.4f}")
    print(f"  F1:         {stats.get('avg_f1', 0):.4f}")
    
    print(f"\n{'='*80}")
    print("BASELINE EXPERIMENT COMPLETE")
    print(f"{'='*80}")
    print(f"Results: {experiments_dir}/{run_id}")
    print(f"\nTo generate comparison visualizations, use:")
    print(f"  from eval.visualizer import ICATVisualizer")
    print(f"  viz = ICATVisualizer('{experiments_dir}')")
    print(f"  viz.compare_runs(run_ids=['{run_id}', ...], output_dir='output/comparisons/')")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run baseline experiment with direct LLM responses (no RAG)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 10 training queries (baseline)
  python run_baseline_experiment.py --queries_path data/antique/train.jsonl -n 10

  # All test queries (baseline)
  python run_baseline_experiment.py --queries_path data/antique/test.jsonl

  # Custom model and parameters
  python run_baseline_experiment.py \\
      --queries_path data/antique/train.jsonl \\
      -n 100 \\
      --model "Qwen/Qwen3-4B-Instruct-2507" \\
      --temperature 0.8 \\
      --max_tokens 1024
        """
    )
    
    parser.add_argument(
        "--queries_path",
        type=str,
        required=True,
        help="Path to JSONL file with queries"
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
        help="Run description (default: 'baseline')"
    )
    parser.add_argument(
        "-n",
        type=int,
        default=None,
        help="Number of queries to process (default: ALL queries in file)"
    )
    parser.add_argument(
        "--server_log_path",
        type=str,
        default=None,
        help="Path to server log file (default: /rstor/pi_hzamani_umass_edu/asalemi/priya/server_logs/log.txt)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (default: Qwen/Qwen3-4B-Instruct-2507)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=512,
        help="Max tokens to generate (default: 512)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for generation (default: 1)"
    )
    parser.add_argument(
        "--corpus_path",
        type=str,
        default=None,
        help=f"Corpus path for ICAT (default: {CORPUS_PATH})"
    )
    parser.add_argument(
        "--experiments_dir",
        type=str,
        default=None,
        help=f"Experiments directory (default: {OUTPUT_DIR}/baseline_experiments)"
    )
    parser.add_argument(
        "--resume_from",
        type=int,
        default=None,
        help="Resume from query index (0-based)"
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        default=True,
        help="Save results after each query (default: True). Use --no-incremental for batch mode."
    )
    parser.add_argument(
        "--no-incremental",
        action="store_true",
        dest="no_incremental",
        help="Disable incremental saving (batch mode - all queries processed before saving)"
    )
    
    args = parser.parse_args()
    
    # Determine incremental mode (default True, disabled by --no-incremental)
    incremental_mode = not args.no_incremental
    
    run_baseline_experiment(
        queries_path=args.queries_path,
        run_id=args.run_id,
        description=args.description,
        n=args.n,
        server_log_path=args.server_log_path,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        corpus_path=args.corpus_path,
        experiments_dir=args.experiments_dir,
        resume_from=args.resume_from,
        incremental=incremental_mode
    )


if __name__ == "__main__":
    main()

