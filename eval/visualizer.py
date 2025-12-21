"""
Visualization tools for ICAT experiment tracking.

Creates comprehensive visualizations comparing ICAT scores across runs.
"""

import json
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
from collections import defaultdict

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class ICATVisualizer:
    """Creates visualizations for ICAT experiment tracking and comparison."""
    
    def __init__(self, experiments_dir: str):
        """
        Initialize visualizer.
        
        Args:
            experiments_dir: Directory containing experiment data
        """
        self.experiments_dir = Path(experiments_dir)
        self.runs_index = self._load_runs_index()
        self.query_history = self._load_query_history()
    
    def _load_runs_index(self) -> Dict:
        """Load runs index."""
        index_file = self.experiments_dir / "runs_index.json"
        if index_file.exists():
            with open(index_file, 'r') as f:
                return json.load(f)
        return {"runs": [], "metadata": {}}
    
    def _load_query_history(self) -> Dict:
        """Load query history."""
        history_file = self.experiments_dir / "query_history.json"
        if history_file.exists():
            with open(history_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _load_run_results(self, run_id: str) -> List[Dict]:
        """
        Load all results for a specific run.
        
        Args:
            run_id: Run identifier
            
        Returns:
            List of result dictionaries
        """
        results_file = self.experiments_dir / run_id / "results.jsonl"
        results = []
        if results_file.exists():
            with open(results_file, 'r') as f:
                for line in f:
                    if line.strip():
                        results.append(json.loads(line.strip()))
        return results
    
    def _get_common_queries(self, run_ids: List[str]) -> List[str]:
        """
        Get queries that appear in all specified runs.
        
        Args:
            run_ids: List of run identifiers
            
        Returns:
            List of query IDs common to all runs
        """
        if not run_ids:
            return []
        
        # Get query IDs for each run
        run_queries = []
        for run_id in run_ids:
            results = self._load_run_results(run_id)
            query_ids = {r["query_id"] for r in results}
            run_queries.append(query_ids)
        
        # Find intersection
        common = set.intersection(*run_queries) if run_queries else set()
        return sorted(list(common))
    
    def plot_run_comparison_paginated(
        self,
        run_ids: List[str],
        output_dir: str,
        queries_per_page: int = 50
    ):
        """
        Create paginated line plots comparing ICAT scores across runs.
        
        Args:
            run_ids: List of run identifiers to compare
            output_dir: Directory to save plots
            queries_per_page: Number of queries per page
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get common queries
        common_queries = self._get_common_queries(run_ids)
        
        if not common_queries:
            print("No common queries found across runs")
            return
        
        print(f"\nGenerating paginated comparison plots...")
        print(f"  Common queries: {len(common_queries)}")
        print(f"  Queries per page: {queries_per_page}")
        
        # Load all data for runs
        run_data = {}
        for run_id in run_ids:
            results = self._load_run_results(run_id)
            run_data[run_id] = {r["query_id"]: r for r in results}
        
        # Get run labels
        run_labels = {}
        for run_id in run_ids:
            meta = self.runs_index["metadata"].get(run_id, {})
            desc = meta.get("description", "")
            if desc:
                run_labels[run_id] = desc
            else:
                run_labels[run_id] = run_id[:15]
        
        # Create pages
        num_pages = (len(common_queries) + queries_per_page - 1) // queries_per_page
        
        for page_num in range(num_pages):
            start_idx = page_num * queries_per_page
            end_idx = min(start_idx + queries_per_page, len(common_queries))
            page_queries = common_queries[start_idx:end_idx]
            
            # Create figure with 3 subplots (Coverage, Factuality, F1)
            fig, axes = plt.subplots(3, 1, figsize=(max(16, len(page_queries) * 0.3), 12))
            
            metrics = ['coverage', 'factuality', 'f1']
            titles = ['Coverage', 'Factuality', 'F1 Score']
            colors = plt.cm.tab10(np.linspace(0, 1, len(run_ids)))
            
            for metric_idx, (metric, title) in enumerate(zip(metrics, titles)):
                ax = axes[metric_idx]
                
                # Plot each run
                for run_idx, run_id in enumerate(run_ids):
                    scores = []
                    for query_id in page_queries:
                        result = run_data[run_id].get(query_id, {})
                        icat_scores = result.get("icat_scores", {})
                        scores.append(icat_scores.get(metric, 0))
                    
                    ax.plot(range(len(page_queries)), scores, 
                           marker='o', label=run_labels[run_id],
                           color=colors[run_idx], linewidth=2, markersize=4)
                
                # Customize subplot
                ax.set_ylabel(f'{title} Score', fontsize=11, fontweight='bold')
                ax.set_ylim([0, 1.05])
                ax.grid(True, alpha=0.3)
                ax.legend(loc='upper right', fontsize=9)
                
                if metric_idx == 2:  # Bottom plot
                    ax.set_xlabel('Query ID', fontsize=11, fontweight='bold')
                    ax.set_xticks(range(len(page_queries)))
                    ax.set_xticklabels([q[:15] for q in page_queries], 
                                      rotation=45, ha='right', fontsize=8)
                else:
                    ax.set_xticks([])
            
            plt.suptitle(f'ICAT Score Comparison - Page {page_num + 1}/{num_pages}', 
                        fontsize=14, fontweight='bold', y=0.995)
            plt.tight_layout()
            
            output_file = output_path / f"comparison_page_{page_num + 1}.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  ✓ Saved page {page_num + 1}/{num_pages}: {output_file}")
    
    def plot_run_comparison_heatmap(
        self,
        run_ids: List[str],
        output_path: str
    ):
        """
        Create heatmaps comparing ICAT scores across all runs and queries.
        
        Args:
            run_ids: List of run identifiers to compare
            output_path: Path to save the heatmap
        """
        # Get common queries
        common_queries = self._get_common_queries(run_ids)
        
        if not common_queries:
            print("No common queries found across runs")
            return
        
        print(f"\nGenerating heatmap comparison...")
        print(f"  Queries: {len(common_queries)}")
        print(f"  Runs: {len(run_ids)}")
        
        # Load all data
        run_data = {}
        for run_id in run_ids:
            results = self._load_run_results(run_id)
            run_data[run_id] = {r["query_id"]: r for r in results}
        
        # Get run labels
        run_labels = []
        for run_id in run_ids:
            meta = self.runs_index["metadata"].get(run_id, {})
            desc = meta.get("description", "")
            if desc:
                run_labels.append(desc[:20])
            else:
                run_labels.append(run_id[:15])
        
        # Create data matrices for each metric
        metrics = ['coverage', 'factuality', 'f1']
        titles = ['Coverage', 'Factuality', 'F1 Score']
        
        # Create figure with 3 heatmaps
        fig, axes = plt.subplots(1, 3, figsize=(18, max(10, len(common_queries) * 0.15)))
        
        for metric_idx, (metric, title) in enumerate(zip(metrics, titles)):
            # Build matrix: rows = queries, cols = runs
            matrix = []
            for query_id in common_queries:
                row = []
                for run_id in run_ids:
                    result = run_data[run_id].get(query_id, {})
                    icat_scores = result.get("icat_scores", {})
                    row.append(icat_scores.get(metric, 0))
                matrix.append(row)
            
            matrix = np.array(matrix)
            
            # Create heatmap
            ax = axes[metric_idx]
            im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
            
            # Set ticks and labels
            ax.set_xticks(range(len(run_ids)))
            ax.set_xticklabels(run_labels, rotation=45, ha='right', fontsize=8)
            
            # Show query IDs only if not too many
            if len(common_queries) <= 50:
                ax.set_yticks(range(len(common_queries)))
                ax.set_yticklabels([q[:15] for q in common_queries], fontsize=6)
            else:
                ax.set_yticks([])
                ax.set_ylabel(f'{len(common_queries)} Queries', fontsize=10)
            
            ax.set_title(title, fontsize=12, fontweight='bold')
            
            # Add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        plt.suptitle('ICAT Score Heatmap Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved heatmap: {output_path}")
    
    def _generate_summary_stats(
        self,
        run_ids: List[str],
        output_dir: str
    ):
        """
        Generate summary statistics comparing runs.
        
        Args:
            run_ids: List of run identifiers
            output_dir: Directory to save summary files
        """
        output_path = Path(output_dir)
        
        # Get common queries
        common_queries = self._get_common_queries(run_ids)
        
        # Load data
        run_data = {}
        for run_id in run_ids:
            results = self._load_run_results(run_id)
            run_data[run_id] = {r["query_id"]: r for r in results}
        
        # Calculate statistics
        summary = {
            "num_runs": len(run_ids),
            "num_common_queries": len(common_queries),
            "runs": {}
        }
        
        for run_id in run_ids:
            meta = self.runs_index["metadata"].get(run_id, {})
            
            # Calculate averages across common queries
            coverages = []
            factualities = []
            f1s = []
            
            for query_id in common_queries:
                result = run_data[run_id].get(query_id, {})
                icat_scores = result.get("icat_scores", {})
                coverages.append(icat_scores.get("coverage", 0))
                factualities.append(icat_scores.get("factuality", 0))
                f1s.append(icat_scores.get("f1", 0))
            
            summary["runs"][run_id] = {
                "description": meta.get("description", ""),
                "config": meta.get("config", {}),
                "avg_coverage": np.mean(coverages) if coverages else 0,
                "avg_factuality": np.mean(factualities) if factualities else 0,
                "avg_f1": np.mean(f1s) if f1s else 0,
                "std_coverage": np.std(coverages) if coverages else 0,
                "std_factuality": np.std(factualities) if factualities else 0,
                "std_f1": np.std(f1s) if f1s else 0
            }
        
        # Save as JSON
        json_file = output_path / "comparison_summary.json"
        with open(json_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save as CSV
        csv_data = []
        for run_id, stats in summary["runs"].items():
            csv_data.append({
                "run_id": run_id,
                "description": stats["description"],
                "avg_coverage": f"{stats['avg_coverage']:.4f}",
                "avg_factuality": f"{stats['avg_factuality']:.4f}",
                "avg_f1": f"{stats['avg_f1']:.4f}",
                "std_coverage": f"{stats['std_coverage']:.4f}",
                "std_factuality": f"{stats['std_factuality']:.4f}",
                "std_f1": f"{stats['std_f1']:.4f}"
            })
        
        df = pd.DataFrame(csv_data)
        csv_file = output_path / "comparison_summary.csv"
        df.to_csv(csv_file, index=False)
        
        print(f"\n  ✓ Saved summary statistics:")
        print(f"    - {json_file}")
        print(f"    - {csv_file}")
        
        return summary
    
    def compare_runs(
        self,
        run_ids: List[str],
        output_dir: str,
        queries_per_page: int = 50
    ):
        """
        Main entry point for comparing multiple runs.
        
        Generates:
        1. Paginated line plots
        2. Heatmap visualization
        3. Summary statistics (JSON and CSV)
        
        Args:
            run_ids: List of run identifiers to compare
            output_dir: Directory to save all outputs
            queries_per_page: Number of queries per page in line plots
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*80}")
        print(f"COMPARING RUNS")
        print(f"{'='*80}")
        print(f"Output directory: {output_dir}")
        print(f"Runs to compare: {len(run_ids)}")
        
        # Generate all visualizations
        self.plot_run_comparison_paginated(run_ids, output_dir, queries_per_page)
        
        heatmap_path = output_path / "comparison_heatmap.png"
        self.plot_run_comparison_heatmap(run_ids, str(heatmap_path))
        
        summary = self._generate_summary_stats(run_ids, output_dir)
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"COMPARISON SUMMARY")
        print(f"{'='*80}")
        print(f"Common queries: {summary['num_common_queries']}")
        print(f"\nAverage ICAT Scores:")
        print(f"{'Run':<20} {'Coverage':>10} {'Factuality':>12} {'F1':>10}")
        print(f"{'-'*55}")
        
        for run_id, stats in summary["runs"].items():
            desc = stats["description"][:18] if stats["description"] else run_id[:18]
            print(f"{desc:<20} {stats['avg_coverage']:>10.4f} {stats['avg_factuality']:>12.4f} {stats['avg_f1']:>10.4f}")
        
        print(f"\n{'='*80}")
        print(f"✓ Comparison complete!")
        print(f"{'='*80}\n")
