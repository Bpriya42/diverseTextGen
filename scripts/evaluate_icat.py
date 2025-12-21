#!/usr/bin/env python3
"""
Evaluate LangGraph outputs using ICAT-A evaluation.

This script:
1. Loads outputs from LangGraph batch directory or batch_summary.json
2. Uses the eval folder's ICAT implementation with ServerLLM integration
3. Uses the same corpus as the multiagent system
4. Runs ICAT-A evaluation
5. Saves results with comparison to existing scores
"""

import json
import argparse
import sys
from pathlib import Path
from typing import List, Dict

# Add project root to path when running from scripts/
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import CORPUS_PATH, OUTPUT_DIR
from eval.icat import ICAT


def load_langgraph_outputs(output_path: str) -> List[Dict]:
    """
    Load outputs from either a directory or batch_summary.json file.
    
    Args:
        output_path: Path to output directory or batch_summary.json file
        
    Returns:
        List of result dictionaries with query_id, query, and final_answer
    """
    path = Path(output_path)
    
    # If it's a file, assume it's batch_summary.json
    if path.is_file():
        with open(path, 'r') as f:
            summary = json.load(f)
        
        results = []
        for result in summary.get("results", []):
            if "error" not in result and "final_answer" in result:
                results.append({
                    "query_id": result["query_id"],
                    "query": result["query"],
                    "final_answer": result["final_answer"],
                    "existing_scores": result.get("final_scores", {})
                })
        return results
    
    # If it's a directory, try to load batch_summary.json
    batch_summary = path / "batch_summary.json"
    if batch_summary.exists():
        with open(batch_summary, 'r') as f:
            summary = json.load(f)
        
        results = []
        for result in summary.get("results", []):
            if "error" not in result and "final_answer" in result:
                results.append({
                    "query_id": result["query_id"],
                    "query": result["query"],
                    "final_answer": result["final_answer"],
                    "existing_scores": result.get("final_scores", {})
                })
        return results
    
    # Otherwise, load individual JSON files
    results = []
    for json_file in path.glob("*.json"):
        if json_file.name == "batch_summary.json":
            continue
        
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                if "final_answer" in data:
                    results.append({
                        "query_id": data["query_id"],
                        "query": data["query"],
                        "final_answer": data["final_answer"],
                        "existing_scores": data.get("final_scores", {})
                    })
        except Exception as e:
            print(f"Warning: Could not load {json_file}: {e}")
    
    return results


def evaluate_with_icat_a(results: List[Dict], corpus_path: str) -> Dict:
    """
    Evaluate outputs using ICAT-A.
    
    Args:
        results: List of result dictionaries
        corpus_path: Path to corpus file
        
    Returns:
        Dictionary with ICAT results and comparison metrics
    """
    queries = [r["query"] for r in results]
    query_ids = [r["query_id"] for r in results]
    model_responses = [r["final_answer"] for r in results]
    
    print(f"\n{'='*80}")
    print(f"ICAT-A EVALUATION")
    print(f"{'='*80}")
    print(f"Number of queries: {len(queries)}")
    print(f"Corpus path: {corpus_path}")
    print(f"{'='*80}\n")
    
    # Initialize ICAT
    print("Initializing ICAT evaluator...")
    icat = ICAT(
        corpus_path=corpus_path,
        queries_path=None,
        qrels_path=None,
        use_web_search=False
    )
    
    # Run evaluation
    print("\nRunning ICAT-A evaluation...")
    print("This may take a while depending on the number of queries...\n")
    
    icat_results, aggregate_metrics = icat.icat_score_a(
        model_responses=model_responses,
        query_ids=query_ids,
        queries=queries
    )
    
    # Combine with original results
    evaluation_results = []
    for i, (original, icat_result) in enumerate(zip(results, icat_results)):
        evaluation_results.append({
            "query_id": original["query_id"],
            "query": original["query"],
            "icat_metrics": {
                "coverage": icat_result["metrics"]["coverage"],
                "factuality": icat_result["metrics"]["factuality"],
                "f1": icat_result["metrics"]["f1"]
            },
            "existing_metrics": original.get("existing_scores", {}),
            "icat_details": {
                "generated_topics": icat_result.get("generated_topics", []),
                "atomic_facts": icat_result.get("atomic_facts", []),
                "entailed_facts": icat_result.get("entailed_facts", []),
                "covered_topics": icat_result.get("covered_topics", [])
            }
        })
    
    return {
        "aggregate_metrics": aggregate_metrics,
        "per_query_results": evaluation_results,
        "summary": {
            "total_queries": len(results),
            "avg_coverage": aggregate_metrics["avg_coverage"],
            "avg_factuality": aggregate_metrics["avg_factuality"],
            "avg_f1": aggregate_metrics["avg_f1"]
        }
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LangGraph outputs using ICAT-A"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to output directory or batch_summary.json file"
    )
    parser.add_argument(
        "--corpus_path",
        type=str,
        default=None,
        help=f"Path to corpus file (default: {CORPUS_PATH})"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to save evaluation results"
    )
    
    args = parser.parse_args()
    
    corpus_path = args.corpus_path or CORPUS_PATH
    
    # Load outputs
    print(f"Loading outputs from {args.output_path}...")
    results = load_langgraph_outputs(args.output_path)
    
    if not results:
        print("Error: No valid outputs found!")
        return
    
    print(f"Loaded {len(results)} successful outputs\n")
    
    # Validate corpus path
    if not Path(corpus_path).exists():
        print(f"Error: Corpus file not found: {corpus_path}")
        return
    
    # Run evaluation
    evaluation_results = evaluate_with_icat_a(results, corpus_path)
    
    # Determine output file path
    output_path = Path(args.output_path)
    if output_path.is_file():
        output_dir = output_path.parent
    else:
        output_dir = output_path
    
    output_file = args.output_file or output_dir / "icat_evaluation.json"
    output_file_path = Path(output_file)
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save results
    with open(output_file_path, 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    # Print summary
    print(f"\n{'='*80}")
    print("EVALUATION COMPLETE")
    print(f"{'='*80}")
    print(f"\nAggregate Metrics:")
    print(f"  Average Coverage:   {evaluation_results['aggregate_metrics']['avg_coverage']:.2%}")
    print(f"  Average Factuality: {evaluation_results['aggregate_metrics']['avg_factuality']:.2%}")
    print(f"  Average F1:         {evaluation_results['aggregate_metrics']['avg_f1']:.2%}")
    print(f"\nResults saved to: {output_file_path}")
    print(f"{'='*80}\n")
    
    # Print per-query results (first 5)
    print("\nPer-Query Results (first 5):")
    print("-" * 80)
    for result in evaluation_results["per_query_results"][:5]:
        print(f"\nQuery ID: {result['query_id']}")
        print(f"Query: {result['query'][:60]}...")
        print(f"  Coverage:   {result['icat_metrics']['coverage']:.2%}")
        print(f"  Factuality: {result['icat_metrics']['factuality']:.2%}")
        print(f"  F1:         {result['icat_metrics']['f1']:.2%}")
    
    if len(evaluation_results["per_query_results"]) > 5:
        print(f"\n... and {len(evaluation_results['per_query_results']) - 5} more queries")


if __name__ == "__main__":
    main()

