"""
Prompts for Agent 3: Answer Synthesizer

Generates comprehensive answers from plan and retrieved documents.
"""

from typing import List, Dict


SYNTHESIZER_PROMPT_TEMPLATE = """You are a helpful assistant that answers complex user questions using evidence retrieved from multiple aspects of the topic.

Each aspect includes a focused subquery, reasoning, and a set of relevant retrieved documents. 
Read all the information carefully, integrate insights, and write one complete, coherent answer.

---

Main Question:
{query}

Aspects and Reasoning:
{plan_text}

Retrieved Evidence:
{retrieved_text}

---

Write a final, well-organized answer that addresses the user's question.
Synthesize information across all aspects.
Avoid mentioning the retrieval process or document IDs.
Use clear paragraphs and concise factual explanations."""


def format_synthesizer_prompt(query: str, plan: List[Dict], retrieval: List[Dict]) -> str:
    """
    Format the synthesizer prompt with query, plan, and retrieval data.
    
    Args:
        query: Original user query
        plan: List of aspects
        retrieval: Retrieved documents per aspect
        
    Returns:
        Formatted prompt string
    """
    import json
    import re
    
    # Helper to parse JSON blocks
    def parse_json_block(text):
        if not text:
            return []
        if isinstance(text, list) and len(text) == 1 and isinstance(text[0], str):
            text = text[0]
        if isinstance(text, list):
            return text
        
        cleaned = re.sub(r"```json|```", "", text, flags=re.IGNORECASE).strip()
        try:
            return json.loads(cleaned)
        except Exception:
            return []
    
    # Parse plan if needed
    if plan and isinstance(plan[0], str):
        plan = parse_json_block(plan[0]) if plan else []
    
    # Build plan summary
    plan_text = "\n".join([
        f"- **{p.get('aspect', 'N/A')}**: {p.get('reason', '')}"
        for p in (plan if isinstance(plan, list) else [])
    ])
    
    # Build retrieval context
    retrieval_texts = []
    for step in retrieval:
        aspect = step.get("aspect", "General")
        subq = step.get("subquery", "")
        docs = step.get("retrieved_docs", [])
        doc_texts = "\n".join([f"- {d['text']}" for d in docs])
        retrieval_texts.append(f"### {aspect}\nSubquery: {subq}\nContext:\n{doc_texts}")
    
    retrieved_text = "\n\n".join(retrieval_texts)
    
    return SYNTHESIZER_PROMPT_TEMPLATE.format(
        query=query,
        plan_text=plan_text,
        retrieved_text=retrieved_text
    ).strip()

