#!/usr/bin/env python3
"""
Example usage of the experiment tracking system.

This demonstrates how to use ExperimentTracker and ICATVisualizer programmatically.
"""

def example_basic_usage():
    """Basic usage example."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Usage")
    print("="*80 + "\n")
    
    from eval.experiment_tracker import ExperimentTracker
    
    # Initialize tracker
    tracker = ExperimentTracker(".diverseTextGen/output/experiments")
    
    # Create a new run
    run_id = tracker.create_run(
        run_id="example_run_1",
        config={"max_iterations": 3, "model": "Qwen3-4B"},
        description="Example experiment for demonstration"
    )
    
    print(f"Created run: {run_id}")
    
    # Log some query results
    for i in range(3):
        tracker.log_query_result(
            run_id=run_id,
            query_id=f"example_query_{i}",
            query=f"Example query {i}?",
            icat_scores={
                "coverage": 0.7 + i * 0.05,
                "factuality": 0.75 + i * 0.05,
                "f1": 0.725 + i * 0.05
            },
            rag_metrics={
                "total_iterations": 3,
                "runtime_seconds": 10.5 + i
            }
        )
        print(f"Logged query {i}")
    
    # Finalize the run
    tracker.finalize_run(run_id, status="completed")
    print(f"\nFinalized run: {run_id}")
    
    # Get summary
    summary = tracker.get_run_summary(run_id)
    print(f"\nAggregate F1: {summary['aggregate_stats']['avg_f1']:.3f}")


def example_query_history():
    """Query history example."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Query History")
    print("="*80 + "\n")
    
    from eval.experiment_tracker import ExperimentTracker
    
    tracker = ExperimentTracker(".diverseTextGen/output/experiments")
    
    # Check if we have any query history
    if not tracker.query_history:
        print("No query history available yet. Run an experiment first!")
        return
    
    # Get a random query
    query_id = list(tracker.query_history.keys())[0]
    history = tracker.get_query_history(query_id)
    
    print(f"Query: {history['query']}")
    print(f"Tracked across {len(history['runs'])} runs\n")
    
    for run in history["runs"]:
        print(f"Run {run['run_id']}:")
        print(f"  F1: {run['icat_scores']['f1']:.3f}")
        print(f"  Coverage: {run['icat_scores']['coverage']:.3f}")
        print(f"  Factuality: {run['icat_scores']['factuality']:.3f}")
        print()


def example_run_comparison():
    """Run comparison example."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Run Comparison")
    print("="*80 + "\n")
    
    from eval.experiment_tracker import ExperimentTracker
    
    tracker = ExperimentTracker(".diverseTextGen/output/experiments")
    
    # Get all runs
    all_runs = tracker.get_all_runs()
    
    if len(all_runs) < 2:
        print("Need at least 2 runs for comparison. Run more experiments!")
        return
    
    # Compare first two runs
    run_ids = [all_runs[0]["run_id"], all_runs[1]["run_id"]]
    comparison = tracker.compare_runs(run_ids)
    
    print(f"Comparing runs: {' vs '.join(run_ids)}")
    print(f"Common queries: {len(comparison['common_queries'])}\n")
    
    # Show first few comparisons
    for query_id in list(comparison['common_queries'])[:3]:
        query_data = comparison['per_query_comparison'][query_id]
        print(f"Query: {query_data['query'][:50]}...")
        
        for rid in run_ids:
            if rid in query_data['scores_by_run']:
                f1 = query_data['scores_by_run'][rid]['f1']
                print(f"  {rid}: F1={f1:.3f}")
        print()


def example_visualization():
    """Visualization example."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Visualization")
    print("="*80 + "\n")
    
    from eval.visualizer import ICATVisualizer
    from pathlib import Path
    
    viz = ICATVisualizer(".diverseTextGen/output/experiments")
    
    # Create output directory
    output_dir = Path(".diverseTextGen/output/experiments/example_visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating visualizations...")
    
    # Generate aggregate trends
    viz.plot_aggregate_trends(str(output_dir / "trends.png"))
    print(f"✓ Created: {output_dir}/trends.png")
    
    # Generate query tracking
    viz.plot_query_tracking(str(output_dir / "tracking.png"), n_queries=10)
    print(f"✓ Created: {output_dir}/tracking.png")
    
    print(f"\nVisualizations saved to: {output_dir}")


def main():
    """Run all examples."""
    print("\n" + "#"*80)
    print("# Experiment Tracking System - Usage Examples")
    print("#"*80)
    
    print("\nThese examples demonstrate how to use the tracking system programmatically.")
    print("For real experiments, use: python run_full_experiment.py")
    
    try:
        example_basic_usage()
        example_query_history()
        example_run_comparison()
        example_visualization()
        
        print("\n" + "#"*80)
        print("# Examples Complete!")
        print("#"*80)
        print("\nTo run a real experiment:")
        print("  python run_full_experiment.py --queries_path data/antique/train.jsonl -n 10")
        print("\n" + "#"*80 + "\n")
        
    except Exception as e:
        print(f"\n⚠ Note: Some examples may not work until you run an actual experiment.")
        print(f"Error: {e}\n")
        print("To get started, run:")
        print("  python run_full_experiment.py --queries_path data/antique/train.jsonl -n 10")


if __name__ == "__main__":
    main()

