"""
Agent 2: Document Retriever

Retrieves relevant documents for each aspect in the plan.
"""

import re
import json
from typing import List, Dict

from config.settings import CORPUS_PATH, CACHE_DIR, DEFAULT_TOP_K
from retrieval.retriever import Retriever


# Module-level retriever instance (singleton pattern)
_retriever_instance = None


def get_retriever():
    """Get or create shared retriever instance."""
    global _retriever_instance
    
    if _retriever_instance is None:
        _retriever_instance = Retriever(cache_dir=CACHE_DIR)
        _retriever_instance.process_corpus(CORPUS_PATH)
    
    return _retriever_instance


def extract_subqueries(plan_text: str) -> List[str]:
    """
    Extract subqueries from plan text.
    
    Args:
        plan_text: Plan text with numbered items
        
    Returns:
        List of subquery strings
    """
    steps = re.split(r'\n\d+\.\s*', plan_text)
    steps = [s.strip() for s in steps if s.strip()]
    return steps


def retrieve_for_plan(plan: List[Dict], top_k: int = None) -> List[Dict]:
    """
    Retrieve documents for each aspect in the plan.
    
    Args:
        plan: List of aspect dictionaries with 'query' field
        top_k: Number of documents to retrieve per aspect
        
    Returns:
        List of retrieval results per aspect
    """
    top_k = top_k or DEFAULT_TOP_K
    retriever = get_retriever()
    retrieval_results = []
    
    for aspect_item in plan:
        # Handle both dict and string formats
        if isinstance(aspect_item, dict):
            subquery = aspect_item.get("query", "")
            aspect_name = aspect_item.get("aspect", "")
        else:
            try:
                parsed = json.loads(aspect_item) if isinstance(aspect_item, str) else aspect_item
                subquery = parsed.get("query", "")
                aspect_name = parsed.get("aspect", "")
            except:
                subquery = str(aspect_item)
                aspect_name = ""
        
        if not subquery:
            continue
        
        # Retrieve documents
        docs = retriever.retrieve(subquery, top_k)
        
        retrieval_results.append({
            "aspect": aspect_name,
            "subquery": subquery,
            "retrieved_docs": [
                {
                    "id": doc[0],
                    "text": doc[1],
                    "score": doc[2]
                }
                for doc in docs
            ]
        })
    
    return retrieval_results

