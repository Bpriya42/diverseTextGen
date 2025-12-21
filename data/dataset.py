"""
Dataset Loading Utilities

Functions for loading datasets in various formats.
"""

import json
import datasets


def load_dataset(addr, cache_dir=None):
    """
    Load a dataset (JSON or JSONL) and return a HuggingFace Dataset object.
    
    Supports files with or without 'retrieved_docs' fields.
    
    Args:
        addr: Path to dataset file
        cache_dir: Optional cache directory
        
    Returns:
        HuggingFace Dataset object
    """
    def gen():
        with open(addr, "r", encoding="utf-8") as f:
            try:
                # Try parsing as a full JSON array
                questions = json.load(f)
                for q in questions:
                    yield {
                        "id": q.get("query_id"),
                        "query": q.get("query_description"),
                        **({"context": q["retrieved_docs"]} if "retrieved_docs" in q else {}),
                    }
            except json.JSONDecodeError:
                # Fall back to JSONL format
                f.seek(0)
                for line in f:
                    if not line.strip():
                        continue
                    obj = json.loads(line.strip())
                    yield {
                        "id": obj.get("query_id"),
                        "query": obj.get("query_description"),
                        **({"context": obj["retrieved_docs"]} if "retrieved_docs" in obj else {}),
                    }

    return datasets.Dataset.from_generator(gen, cache_dir=cache_dir)


def load_dataset_with_plan_for_train(addr, cache_dir=None):
    """
    Load a dataset for training with multiple query plans.
    
    Each query may contain multiple 'plans' with parsed sub-queries.
    
    Args:
        addr: Path to dataset file
        cache_dir: Optional cache directory
        
    Returns:
        HuggingFace Dataset object
    """
    def gen():
        with open(addr, 'r', encoding='utf-8') as f:
            questions = json.load(f)
            for q in questions:
                for i, p in enumerate(q.get("plans", [])):
                    plan = []
                    for a in p.get("parsed_queries", []):
                        if not all(k in a for k in ("aspect", "query", "reason")):
                            print(f"⚠️ Skipping malformed sub-query: {a}")
                            continue
                        plan.append({
                            "aspect": a["aspect"],
                            "query": a["query"],
                            "reason": a["reason"],
                            "context": a.get("retrieved_docs", [])
                        })
                    yield {
                        "id": f'{q["query_id"]}@{i}',
                        "query": q.get("query_description"),
                        "plan": plan
                    }

    return datasets.Dataset.from_generator(gen, cache_dir=cache_dir)


def load_responses(addr):
    """
    Load a responses JSON file and flatten it.
    
    Args:
        addr: Path to responses file
        
    Returns:
        Dictionary mapping 'id@index' -> output
    """
    outputs = {}
    with open(addr, 'r', encoding='utf-8') as f:
        temp = json.load(f)
    for k, v in temp.items():
        for i, o in enumerate(v):
            outputs[f'{k}@{i}'] = o
    return outputs


def load_responses_global_local_search(addr, addr_init, max_global=-1, max_local=-1):
    """
    Load responses for global/local search.
    
    Combines multiple search step outputs into grouped and flattened dictionaries.
    
    Args:
        addr: Path to main responses file
        addr_init: Path to initial responses file
        max_global: Maximum global search steps
        max_local: Maximum local search steps
        
    Returns:
        Tuple of (grouped_outputs, individual_responses)
    """
    outputs_grouped, responses_indv = {}, {}

    temp = {}
    if addr:
        with open(addr, 'r', encoding='utf-8') as f:
            temp = json.load(f)

    with open(addr_init, 'r', encoding='utf-8') as f:
        temp_init = json.load(f)
        temp['-1'] = temp_init

    for local_step, local_step_outputs in temp.items():
        if max_local > -1 and int(local_step) >= max_local:
            break

        for k, v in local_step_outputs.items():
            outputs_grouped.setdefault(k, [])
            for i, o in enumerate(v):
                if max_global > -1 and i >= max_global:
                    break
                o['new_id'] = f'{k}@{local_step}@{i}'
                outputs_grouped[k].append(o)
                responses_indv[o['new_id']] = o

    return outputs_grouped, responses_indv

