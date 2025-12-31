"""
Iteration Gate Node

Decision node that determines whether to continue iterating or terminate.
"""

import time
from typing import Tuple

from state import RAGState
from llm_observability import get_observability

# Try to import psutil for memory monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("[Warning] psutil not installed - memory monitoring disabled")

# Try to import torch for GPU memory monitoring
try:
    import torch
    TORCH_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False


def check_memory_usage(max_ram_percent: float = 90.0, max_gpu_percent: float = 90.0):
    """
    Check if memory usage is within safe limits.
    
    Args:
        max_ram_percent: Maximum allowed RAM usage percentage
        max_gpu_percent: Maximum allowed GPU memory usage percentage
        
    Returns:
        Tuple of (is_safe, reason_string, memory_info_dict)
    """
    memory_info = {}
    
    # Check RAM usage
    if PSUTIL_AVAILABLE:
        ram = psutil.virtual_memory()
        memory_info["ram_percent"] = ram.percent
        memory_info["ram_used_gb"] = ram.used / (1024**3)
        memory_info["ram_total_gb"] = ram.total / (1024**3)
        
        if ram.percent >= max_ram_percent:
            return (
                False, 
                f"RAM usage at {ram.percent:.1f}% (limit: {max_ram_percent}%)",
                memory_info
            )
    
    # Check GPU memory
    if TORCH_AVAILABLE:
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i)
            total = torch.cuda.get_device_properties(i).total_memory
            gpu_percent = (allocated / total) * 100 if total > 0 else 0
            
            memory_info[f"gpu_{i}_percent"] = gpu_percent
            memory_info[f"gpu_{i}_allocated_gb"] = allocated / (1024**3)
            memory_info[f"gpu_{i}_total_gb"] = total / (1024**3)
            
            if gpu_percent >= max_gpu_percent:
                return (
                    False,
                    f"GPU {i} memory at {gpu_percent:.1f}% (limit: {max_gpu_percent}%)",
                    memory_info
                )
    
    return True, "OK", memory_info


def check_quality_termination(
    state: RAGState,
    unclear_facts_threshold_percent: float = 15.0,
    unclear_facts_threshold_absolute: int = 3,
    missing_points_threshold: int = 1
) -> Tuple[bool, str]:
    """
    Check if feedback indicates no further improvements are needed.
    
    Quality-based termination occurs when:
    1. No REFUTED facts
    2. UNCLEAR facts are below threshold (<= threshold_percent OR <= threshold_absolute)
    3. Missing salient points are below threshold (<= missing_points_threshold)
    4. All aspects in aspect_coverage_details are "well-covered"
    
    Args:
        state: Current RAGState with factual_feedback and coverage_feedback
        unclear_facts_threshold_percent: Maximum percentage of unclear facts allowed (default: 15%)
        unclear_facts_threshold_absolute: Maximum absolute number of unclear facts allowed (default: 3)
        missing_points_threshold: Maximum number of missing salient points allowed (default: 1)
        
    Returns:
        Tuple of (should_terminate, reason)
    """
    factual_feedback = state.get("factual_feedback", {})
    coverage_feedback = state.get("coverage_feedback", {})
    
    # Check 1: No REFUTED facts (strict requirement)
    stats = factual_feedback.get("stats", {})
    total_facts = stats.get("total_facts", 0)
    refuted = stats.get("refuted", 0)
    unclear = stats.get("unclear", 0)
    supported = stats.get("supported", 0)
    
    # Need at least some facts to evaluate
    if total_facts == 0:
        return False, ""
    
    # No refuted facts allowed
    if refuted > 0:
        return False, ""
    
    # Check unclear facts against threshold (percentage OR absolute, whichever is more lenient)
    unclear_percent = (unclear / total_facts * 100) if total_facts > 0 else 0
    unclear_below_percent_threshold = unclear_percent <= unclear_facts_threshold_percent
    unclear_below_absolute_threshold = unclear <= unclear_facts_threshold_absolute
    facts_ok = unclear_below_percent_threshold or unclear_below_absolute_threshold
    
    # Check 2: Missing salient points below threshold
    missing_points = coverage_feedback.get("missing_salient_points", [])
    num_missing = len(missing_points)
    coverage_ok = num_missing <= missing_points_threshold
    
    # Check 3: All aspects well-covered
    aspect_details = coverage_feedback.get("aspect_coverage_details", [])
    aspects_ok = all(
        a.get("coverage_status") == "well-covered" 
        for a in aspect_details
    ) if aspect_details else True
    
    # Check 4: Explicit "no improvements needed" flags from agents
    factual_no_improve = factual_feedback.get("written_feedback", {}).get("no_improvements_needed", False)
    coverage_no_improve = coverage_feedback.get("no_improvements_needed", False)
    
    # Log the quality check details
    print(f"  Quality Check:")
    print(f"    Facts: {supported}/{total_facts} supported, {refuted} refuted, {unclear} unclear ({unclear_percent:.1f}%)")
    print(f"      → Refuted: {'✓ OK' if refuted == 0 else '✗ FAIL'} (must be 0)")
    print(f"      → Unclear: {'✓ OK' if facts_ok else '✗ FAIL'} ({unclear} <= {unclear_facts_threshold_percent}% OR {unclear} <= {unclear_facts_threshold_absolute})")
    print(f"    Coverage: {num_missing} missing salient points")
    print(f"      → Missing points: {'✓ OK' if coverage_ok else '✗ FAIL'} ({num_missing} <= {missing_points_threshold})")
    print(f"    Aspects: {'all well-covered' if aspects_ok else 'some gaps'}")
    print(f"    Agent flags: factual={factual_no_improve}, coverage={coverage_no_improve}")
    
    # Terminate if all quality checks pass
    if facts_ok and coverage_ok and aspects_ok:
        return True, "quality_complete"
    
    # Also terminate if both agents explicitly say no improvements needed
    if factual_no_improve and coverage_no_improve:
        return True, "quality_complete_by_agents"
    
    return False, ""


def iteration_gate_node(state: RAGState) -> RAGState:
    """
    Decide whether to continue iterating or terminate.
    
    Termination conditions (priority order):
    1. Quality complete - all facts verified and coverage is comprehensive
    2. Memory limit exceeded (RAM or GPU)
    
    Args:
        state: Current RAGState
        
    Returns:
        Updated RAGState with termination decision
    """
    iteration = state.get("iteration", 0)
    current_history = state.get("history", [])
    budget = state.get("budget", {})
    
    print(f"\n{'='*60}")
    print(f"[Iteration Gate] EVALUATION")
    print(f"{'='*60}")
    print(f"  Current iteration: {iteration}")
    print(f"  Mode: Quality-controlled (memory-bounded)")
    print(f"  Completed iteration: {iteration + 1}")
    
    # Create history entry for current iteration
    history_entry = {
        "iteration": iteration,
        "plan": state.get("plan", []),
        "answer": state.get("answer", ""),
        "timestamp": time.time()
    }
    
    # Append to history (manually managed, not using add reducer)
    new_history = current_history + [history_entry]
    
    next_iteration = iteration + 1
    
    # Track metrics and check for plateau
    obs = get_observability()
    factual_feedback = state.get("factual_feedback", {})
    coverage_feedback = state.get("coverage_feedback", {})
    factual_stats = factual_feedback.get("stats", {})
    
    # Track iteration metrics for plateau detection
    obs.track_iteration_metrics(iteration, factual_stats, coverage_feedback)
    
    # Check for plateau (informational only, doesn't force termination)
    is_plateau, plateau_reason = obs.check_plateau()
    if is_plateau:
        print(f"[Iteration Gate] ⚠️  {plateau_reason}")
        print(f"[Iteration Gate] Quality may have stabilized. Continuing to let quality checks decide termination.")
    
    # Check 1: Quality-based termination (primary termination condition)
    # Terminates when: no refuted facts, unclear facts below threshold, missing points below threshold
    if iteration >= 0:
        quality_done, quality_reason = check_quality_termination(state)
        if quality_done:
            print(f"[Iteration Gate] ✓ Quality complete - no further improvements needed - TERMINATING")
            print(f"  Reason: {quality_reason}")
            return {
                "should_continue": False,
                "termination_reason": quality_reason,
                "iteration": next_iteration,
                "history": new_history
            }
    
    # Check 2: Memory limit exceeded
    max_ram = budget.get("max_ram_percent", 90.0)
    max_gpu = budget.get("max_gpu_percent", 90.0)
    is_safe, memory_reason, memory_info = check_memory_usage(max_ram, max_gpu)
    
    if not is_safe:
        print(f"[Iteration Gate] ✓ Memory limit reached: {memory_reason} - TERMINATING")
        return {
            "should_continue": False,
            "termination_reason": f"memory_exceeded: {memory_reason}",
            "iteration": next_iteration,
            "history": new_history
        }
    
    # Log memory status if available
    if memory_info:
        if "ram_percent" in memory_info:
            print(f"  RAM usage: {memory_info['ram_percent']:.1f}%")
        if "gpu_0_percent" in memory_info:
            print(f"  GPU 0 usage: {memory_info['gpu_0_percent']:.1f}%")
    
    # Continue iterating
    print(f"[Iteration Gate] → Continuing to iteration {next_iteration + 1}")
    print(f"{'='*60}\n")
    
    return {
        "should_continue": True,
        "iteration": next_iteration,
        "history": new_history
    }
