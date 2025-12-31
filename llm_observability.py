"""
LLM Observability Module

Minimal tracking system for:
1. LLM decision logging - understand what LLM decides after each iteration
2. Plateau detection - detect when quality improvements stall

Usage:
    from llm_observability import get_observability
    
    obs = get_observability()
    obs.log_decision(iteration, "planner", "refine_plan", metrics_dict)
    obs.track_iteration_metrics(iteration, factual_stats, coverage_stats)
    is_plateau, reason = obs.check_plateau()
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, List


class LLMObservability:
    """
    Lightweight observability for LLM decisions and plateau detection.
    
    Features:
    - Logs LLM decisions to artifacts/llm_decisions.jsonl
    - Tracks iteration quality metrics for plateau detection
    - Prints decisions to console for immediate visibility
    """
    
    def __init__(self, log_file: str = "artifacts/llm_decisions.jsonl"):
        """
        Initialize observability system.
        
        Args:
            log_file: Path to decision log file (JSON Lines format)
        """
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # For plateau detection
        self.iteration_metrics: List[Dict] = []
        
        # Plateau detection parameters (lenient)
        self.plateau_window = 4  # Need 4 iterations to detect
        self.plateau_threshold = 0.05  # 5% improvement required
    
    def log_decision(
        self,
        iteration: int,
        agent: str,
        decision_type: str,
        metrics: Dict[str, Any]
    ):
        """
        Log an LLM decision.
        
        Args:
            iteration: Current iteration number
            agent: Agent name (e.g., "planner", "synthesizer")
            decision_type: Type of decision (e.g., "initial_plan", "refine_plan")
            metrics: Dictionary of metrics to log
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "iteration": iteration,
            "agent": agent,
            "decision_type": decision_type,
            "metrics": metrics
        }
        
        # Print to console for immediate visibility
        print(f"\n[{agent.upper()} - Iteration {iteration}] Decision: {decision_type}")
        for key, value in metrics.items():
            if isinstance(value, (int, float, str, bool)):
                print(f"  {key}: {value}")
            elif isinstance(value, list) and len(value) <= 3:
                print(f"  {key}: {value}")
        
        # Append to log file
        try:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            print(f"Warning: Failed to log decision: {e}")
    
    def track_iteration_metrics(
        self,
        iteration: int,
        factual_stats: Dict,
        coverage_stats: Dict
    ):
        """
        Track iteration metrics for plateau detection.
        
        Args:
            iteration: Current iteration number
            factual_stats: Stats from verifier (total_facts, supported, refuted, unclear)
            coverage_stats: Stats from coverage evaluator (missing_salient_points, etc.)
        """
        # Calculate factual accuracy
        total_facts = factual_stats.get("total_facts", 1)
        supported = factual_stats.get("supported", 0)
        accuracy = supported / max(total_facts, 1)
        
        # Calculate coverage ratio
        missing_points = len(coverage_stats.get("missing_salient_points", []))
        aspect_details = coverage_stats.get("aspect_coverage_details", [])
        if aspect_details:
            well_covered = sum(
                1 for a in aspect_details 
                if a.get("coverage_status") == "well-covered"
            )
            coverage_ratio = well_covered / max(len(aspect_details), 1)
        else:
            coverage_ratio = 0.0
        
        # Calculate composite quality score (60% accuracy, 40% coverage)
        composite_score = 0.6 * accuracy + 0.4 * coverage_ratio
        
        # Store metrics
        metrics = {
            "iteration": iteration,
            "accuracy": accuracy,
            "coverage_ratio": coverage_ratio,
            "composite_score": composite_score,
            "refuted": factual_stats.get("refuted", 0),
            "unclear": factual_stats.get("unclear", 0),
            "missing_points": missing_points
        }
        
        self.iteration_metrics.append(metrics)
        
        # Also log as a decision for tracking
        self.log_decision(
            iteration=iteration,
            agent="system",
            decision_type="iteration_metrics",
            metrics={
                "factual_accuracy": round(accuracy, 3),
                "coverage_ratio": round(coverage_ratio, 3),
                "composite_score": round(composite_score, 3),
                "refuted_facts": factual_stats.get("refuted", 0),
                "unclear_facts": factual_stats.get("unclear", 0),
                "missing_points": missing_points
            }
        )
    
    def check_plateau(self) -> Tuple[bool, str]:
        """
        Check if quality metrics have plateaued.
        
        Plateau is detected when:
        - We have at least 'window' iterations
        - Average improvement over last 'window' iterations < threshold
        
        Returns:
            Tuple of (is_plateaued, reason_string)
        """
        if len(self.iteration_metrics) < self.plateau_window:
            return False, ""
        
        # Get last 'window' iterations
        recent = self.iteration_metrics[-self.plateau_window:]
        composite_scores = [m["composite_score"] for m in recent]
        
        # Calculate changes between consecutive iterations
        changes = [
            composite_scores[i+1] - composite_scores[i]
            for i in range(len(composite_scores) - 1)
        ]
        
        # Average improvement
        avg_improvement = sum(changes) / len(changes)
        
        # Check if below threshold
        if abs(avg_improvement) < self.plateau_threshold:
            reason = (
                f"plateau_detected: composite score improvement "
                f"{avg_improvement:.4f} < {self.plateau_threshold} "
                f"over last {self.plateau_window} iterations"
            )
            
            # Log plateau detection
            self.log_decision(
                iteration=recent[-1]["iteration"],
                agent="system",
                decision_type="plateau_check",
                metrics={
                    "plateau_detected": True,
                    "avg_improvement": round(avg_improvement, 4),
                    "threshold": self.plateau_threshold,
                    "window": self.plateau_window,
                    "recent_scores": [round(s, 3) for s in composite_scores]
                }
            )
            
            return True, reason
        
        return False, ""
    
    def get_metrics_summary(self) -> Dict:
        """Get summary of tracked metrics."""
        if not self.iteration_metrics:
            return {}
        
        return {
            "total_iterations": len(self.iteration_metrics),
            "latest_metrics": self.iteration_metrics[-1],
            "initial_metrics": self.iteration_metrics[0],
            "improvement": {
                "accuracy": self.iteration_metrics[-1]["accuracy"] - self.iteration_metrics[0]["accuracy"],
                "coverage": self.iteration_metrics[-1]["coverage_ratio"] - self.iteration_metrics[0]["coverage_ratio"],
                "composite": self.iteration_metrics[-1]["composite_score"] - self.iteration_metrics[0]["composite_score"]
            }
        }


# Global singleton instance
_observability_instance = None


def get_observability() -> LLMObservability:
    """
    Get or create global observability instance.
    
    Returns:
        Shared LLMObservability instance
    """
    global _observability_instance
    if _observability_instance is None:
        _observability_instance = LLMObservability()
    return _observability_instance


def reset_observability():
    """Reset the global observability instance (useful for testing or new queries)."""
    global _observability_instance
    _observability_instance = None

