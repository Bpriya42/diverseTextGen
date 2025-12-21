"""
Experiment Tracker for ICAT Evaluations.

Tracks ICAT scores across multiple experimental runs.
"""

import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from collections import defaultdict


class ExperimentTracker:
    """
    Tracks ICAT scores across experimental runs.
    
    Manages:
    - Run metadata and configuration
    - Per-query ICAT scores over time
    - Aggregate statistics per run
    - Historical data storage
    """
    
    def __init__(self, experiments_dir: str):
        """
        Initialize experiment tracker.
        
        Args:
            experiments_dir: Base directory for all experiments
        """
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        
        self.runs_index_file = self.experiments_dir / "runs_index.json"
        self.query_history_file = self.experiments_dir / "query_history.json"
        
        # Load existing data
        self.runs_index = self._load_runs_index()
        self.query_history = self._load_query_history()
    
    def _load_runs_index(self) -> Dict:
        """Load index of all runs."""
        if self.runs_index_file.exists():
            with open(self.runs_index_file, 'r') as f:
                return json.load(f)
        return {"runs": [], "metadata": {}}
    
    def _load_query_history(self) -> Dict:
        """Load historical query scores."""
        if self.query_history_file.exists():
            with open(self.query_history_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_runs_index(self):
        """Save runs index."""
        with open(self.runs_index_file, 'w') as f:
            json.dump(self.runs_index, f, indent=2)
    
    def _save_query_history(self):
        """Save query history."""
        with open(self.query_history_file, 'w') as f:
            json.dump(self.query_history, f, indent=2)
    
    def create_run(
        self,
        run_id: Optional[str] = None,
        config: Optional[Dict] = None,
        description: str = ""
    ) -> str:
        """
        Create a new experimental run.
        
        Args:
            run_id: Unique identifier (auto-generated if None)
            config: Run configuration
            description: Human-readable description
            
        Returns:
            Run ID
        """
        if run_id is None:
            run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        run_dir = self.experiments_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        
        run_metadata = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "description": description,
            "config": config or {},
            "status": "running",
            "results_file": str(run_dir / "results.jsonl"),
            "summary_file": str(run_dir / "summary.json")
        }
        
        # Add to index
        self.runs_index["runs"].append(run_metadata)
        self.runs_index["metadata"][run_id] = run_metadata
        self._save_runs_index()
        
        # Save run metadata
        with open(run_dir / "metadata.json", 'w') as f:
            json.dump(run_metadata, f, indent=2)
        
        return run_id
    
    def log_query_result(
        self,
        run_id: str,
        query_id: str,
        query: str,
        icat_scores: Dict,
        rag_metrics: Optional[Dict] = None
    ):
        """
        Log ICAT scores for a single query.
        
        Args:
            run_id: Run identifier
            query_id: Query identifier
            query: Query text
            icat_scores: Dict with coverage, factuality, f1
            rag_metrics: Optional RAG system metrics
        """
        # Update query history
        if query_id not in self.query_history:
            self.query_history[query_id] = {
                "query": query,
                "runs": []
            }
        
        self.query_history[query_id]["runs"].append({
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "icat_scores": icat_scores,
            "rag_metrics": rag_metrics or {}
        })
        
        self._save_query_history()
        
        # Append to run's results file
        run_dir = self.experiments_dir / run_id
        results_file = run_dir / "results.jsonl"
        
        result_entry = {
            "query_id": query_id,
            "query": query,
            "icat_scores": icat_scores,
            "rag_metrics": rag_metrics or {},
            "timestamp": datetime.now().isoformat()
        }
        
        with open(results_file, 'a') as f:
            f.write(json.dumps(result_entry) + '\n')
    
    def finalize_run(
        self,
        run_id: str,
        aggregate_stats: Optional[Dict] = None,
        status: str = "completed"
    ):
        """
        Mark a run as complete and save aggregate statistics.
        
        Args:
            run_id: Run identifier
            aggregate_stats: Aggregate statistics for the run
            status: Run status ('completed', 'failed', etc.)
        """
        run_dir = self.experiments_dir / run_id
        
        # Load all results
        results_file = run_dir / "results.jsonl"
        results = []
        if results_file.exists():
            with open(results_file, 'r') as f:
                for line in f:
                    if line.strip():
                        results.append(json.loads(line.strip()))
        
        # Calculate aggregate if not provided
        if aggregate_stats is None and results:
            successful_results = [r for r in results if "icat_scores" in r]
            if successful_results:
                aggregate_stats = {
                    "total_queries": len(results),
                    "successful_queries": len(successful_results),
                    "avg_coverage": sum(r["icat_scores"]["coverage"] for r in successful_results) / len(successful_results),
                    "avg_factuality": sum(r["icat_scores"]["factuality"] for r in successful_results) / len(successful_results),
                    "avg_f1": sum(r["icat_scores"]["f1"] for r in successful_results) / len(successful_results)
                }
        
        # Save summary
        summary = {
            "run_id": run_id,
            "status": status,
            "completed_at": datetime.now().isoformat(),
            "aggregate_stats": aggregate_stats or {},
            "num_results": len(results)
        }
        
        with open(run_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Update runs index
        if run_id in self.runs_index["metadata"]:
            self.runs_index["metadata"][run_id]["status"] = status
            self.runs_index["metadata"][run_id]["aggregate_stats"] = aggregate_stats
            self._save_runs_index()
    
    def get_query_history(self, query_id: str) -> Dict:
        """
        Get all historical data for a specific query.
        
        Args:
            query_id: Query identifier
            
        Returns:
            Dict with query and all run results
        """
        return self.query_history.get(query_id, {"query": "", "runs": []})
    
    def get_all_runs(self) -> List[Dict]:
        """
        Get metadata for all runs.
        
        Returns:
            List of run metadata dicts
        """
        return self.runs_index["runs"]
    
    def get_run_summary(self, run_id: str) -> Dict:
        """
        Get summary for a specific run.
        
        Args:
            run_id: Run identifier
            
        Returns:
            Run summary dict
        """
        summary_file = self.experiments_dir / run_id / "summary.json"
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                return json.load(f)
        return {}
    
    def compare_runs(self, run_ids: List[str]) -> Dict:
        """
        Compare multiple runs.
        
        Args:
            run_ids: List of run identifiers
            
        Returns:
            Comparison statistics
        """
        comparison = {
            "runs": [],
            "common_queries": set(),
            "per_query_comparison": {}
        }
        
        # Load all run data
        for run_id in run_ids:
            run_data = self.get_run_summary(run_id)
            if run_data:
                comparison["runs"].append(run_data)
        
        # Find common queries
        run_queries = []
        for run_id in run_ids:
            results_file = self.experiments_dir / run_id / "results.jsonl"
            if results_file.exists():
                query_ids = set()
                with open(results_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            result = json.loads(line.strip())
                            query_ids.add(result["query_id"])
                run_queries.append(query_ids)
        
        if run_queries:
            comparison["common_queries"] = set.intersection(*run_queries)
            
            # Compare common queries
            for query_id in comparison["common_queries"]:
                history = self.get_query_history(query_id)
                relevant_runs = [r for r in history["runs"] if r["run_id"] in run_ids]
                
                comparison["per_query_comparison"][query_id] = {
                    "query": history["query"],
                    "scores_by_run": {
                        r["run_id"]: r["icat_scores"]
                        for r in relevant_runs
                    }
                }
        
        comparison["common_queries"] = list(comparison["common_queries"])
        return comparison

