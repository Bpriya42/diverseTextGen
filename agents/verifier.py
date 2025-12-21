"""
Agent 5: Fact Verifier

Verifies atomic facts against retrieved evidence.
"""

import json
import re
from typing import List, Dict

from vllm import SamplingParams

from config.settings import (
    DEFAULT_MODEL, LLM_MAX_RETRIES, LLM_NUM_WORKERS,
    VERIFIER_TEMPERATURE, VERIFIER_MAX_TOKENS,
    CORPUS_PATH, CACHE_DIR
)
from llm.server_llm import ServerLLM, load_url_from_log_file
from retrieval.retriever import Retriever


# Module-level instances
_llm_instance = None
_sampling_params = None
_retriever_instance = None


def get_llm():
    """Get or create shared LLM instance for verification."""
    global _llm_instance, _sampling_params
    
    if _llm_instance is None:
        url = load_url_from_log_file()
        _llm_instance = ServerLLM(
            base_url=url,
            model=DEFAULT_MODEL,
            max_retries=LLM_MAX_RETRIES,
            num_workers=LLM_NUM_WORKERS
        )
        _sampling_params = SamplingParams(
            temperature=VERIFIER_TEMPERATURE,
            max_tokens=VERIFIER_MAX_TOKENS
        )
    
    return _llm_instance, _sampling_params


def get_retriever():
    """Get or create shared retriever instance."""
    global _retriever_instance
    
    if _retriever_instance is None:
        _retriever_instance = Retriever(cache_dir=CACHE_DIR)
        _retriever_instance.process_corpus(CORPUS_PATH)
    
    return _retriever_instance


VERIFICATION_PROMPT = """You are a precise fact-checking system. Your task is to verify whether a given atomic fact is supported by the provided evidence documents.

Atomic Fact to Verify:
{fact}

Retrieved Evidence Documents:
{evidence}

Instructions:
1. Carefully read the atomic fact and all evidence documents.
2. Determine if the fact is:
   - SUPPORTED: The evidence clearly confirms the fact
   - REFUTED: The evidence contradicts the fact
   - UNCLEAR: The evidence is insufficient, ambiguous, or does not address the fact

3. Provide your verdict in the following JSON format:
```json
{{
  "verdict": "SUPPORTED" or "REFUTED" or "UNCLEAR",
  "confidence": <float between 0.0 and 1.0>,
  "explanation": "<brief explanation of your reasoning>",
  "supporting_doc_ids": ["<doc_id1>", "<doc_id2>", ...]
}}
```

Output only the JSON block, nothing else."""


FACTUALITY_SUMMARY_PROMPT = """You are analyzing the factuality of an answer based on verification results.

Original Query:
{query}

Verification Results:
{verification_summary}

Statistics:
- Total facts verified: {total_facts}
- Supported: {supported_count}
- Refuted: {refuted_count} 
- Unclear: {unclear_count}

IMPORTANT: UNCLEAR facts are more problematic than REFUTED facts because unclear facts indicate poor evidence retrieval or unverifiable claims.

Your task is to provide a comprehensive written assessment:

1. **What is Good**: Highlight the strengths - which facts are well-supported, what aspects of the answer are reliable, what evidence is strong.

2. **Critical Issues**: Identify serious problems that need immediate attention:
   - UNCLEAR FACTS: List specific facts that lack sufficient evidence (these are the most problematic)
   - REFUTED FACTS: List specific facts that are contradicted by evidence
   - Any patterns of unreliability

3. **Overall Assessment**: Provide a brief summary of the answer's factuality and reliability.

4. **How it can be Improved**: Provide specific, actionable suggestions for improving factuality:
   - For UNCLEAR facts: What specific evidence or sources are needed
   - For REFUTED facts: What corrections or removals are needed
   - What retrieval or verification strategies could help

CRITICAL INSTRUCTION FOR QUALITY-BASED TERMINATION:
- If ALL facts are SUPPORTED (no REFUTED, no UNCLEAR), set "no_improvements_needed" to true
- If there are ANY REFUTED or UNCLEAR facts, set "no_improvements_needed" to false
- When "no_improvements_needed" is true:
  - Leave "unclear_facts_details" and "refuted_facts_details" as empty arrays []
  - Set "how_to_improve" to "No improvements needed - all facts are well-supported by evidence"
  - Set "critical_issues" to "No critical issues - all facts are verified"
- Do NOT suggest unnecessary improvements if the answer is already factually sound

Provide your assessment in the following JSON format:
```json
{{
  "no_improvements_needed": <true if ALL facts are SUPPORTED, false otherwise>,
  "what_is_good": "<written assessment of strengths and well-supported aspects>",
  "critical_issues": "<written assessment focusing on UNCLEAR facts first, then REFUTED facts, or 'No critical issues' if all supported>",
  "overall_assessment": "<brief summary of answer factuality and reliability>",
  "how_to_improve": "<specific actionable suggestions, or 'No improvements needed' if all facts supported>",
  "unclear_facts_details": [
    {{
      "fact": "<unclear fact text>",
      "reason": "<why it's unclear>",
      "needed_evidence": "<what evidence would help verify this>"
    }}
  ],
  "refuted_facts_details": [
    {{
      "fact": "<refuted fact text>", 
      "contradiction": "<what evidence contradicts it>",
      "suggested_correction": "<how to fix or replace this fact>"
    }}
  ]
}}
```

Output only the JSON block, nothing else."""


def parse_json_response(text):
    """Parse JSON response from LLM."""
    if not text:
        return None
    
    cleaned = re.sub(r"```json|```", "", text, flags=re.IGNORECASE).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        return {
            "verdict": "UNCLEAR",
            "confidence": 0.0,
            "explanation": "Failed to parse verification response",
            "supporting_doc_ids": []
        }


def format_evidence_text(retrieved_docs: List[Dict]) -> str:
    """Format retrieved documents into evidence text."""
    evidence_parts = []
    for i, doc in enumerate(retrieved_docs, 1):
        doc_id = doc.get("id", f"doc_{i}")
        text = doc.get("text", "")
        score = doc.get("score", 0.0)
        evidence_parts.append(f"[Document {doc_id}, relevance: {score:.3f}]\n{text}")
    
    return "\n\n".join(evidence_parts)


def format_verification_summary(fact_verifications: List[Dict]) -> str:
    """Format verification results into a readable summary for the LLM."""
    summary_parts = []
    for i, v in enumerate(fact_verifications, 1):
        fact = v.get("fact", "")
        verdict = v.get("verdict", "UNCLEAR")
        confidence = v.get("confidence", 0.0)
        explanation = v.get("explanation", "")
        
        summary_parts.append(
            f"Fact {i}: {fact[:100]}{'...' if len(fact) > 100 else ''}\n"
            f"  Verdict: {verdict}\n"
            f"  Confidence: {confidence:.2f}\n"
            f"  Explanation: {explanation}"
        )
    return "\n\n".join(summary_parts)


def get_all_retrieved_docs(retrieval_list: List[Dict]) -> List[Dict]:
    """Extract all retrieved documents from retrieval results."""
    all_docs = []
    seen_ids = set()
    
    for retrieval_step in retrieval_list:
        docs = retrieval_step.get("retrieved_docs", [])
        for doc in docs:
            doc_id = doc.get("id", "")
            if doc_id and doc_id not in seen_ids:
                all_docs.append(doc)
                seen_ids.add(doc_id)
    
    return all_docs


def retrieve_targeted_evidence(fact: str, top_k: int = 3) -> List[Dict]:
    """Retrieve additional evidence specifically for a fact."""
    try:
        retriever = get_retriever()
        results = retriever.retrieve(fact, top_k=top_k)
        return [
            {"id": doc[0], "text": doc[1], "score": doc[2]}
            for doc in results
        ]
    except Exception:
        return []


def verify_facts(atomic_facts: List[str], retrieval: List[Dict], query: str = "") -> Dict:
    """
    Verify atomic facts against retrieved evidence.
    
    Args:
        atomic_facts: List of atomic fact strings
        retrieval: Retrieved documents from Agent 2
        query: Original query
        
    Returns:
        Dictionary with verification results, factuality score, and written feedback
    """
    llm, sampling_params = get_llm()
    
    # Get all retrieved documents
    all_original_docs = get_all_retrieved_docs(retrieval)
    
    fact_verifications = []
    
    for fact in atomic_facts:
        # Start with original retrieved documents
        evidence_docs = all_original_docs.copy()
        
        # If few documents, retrieve targeted evidence
        if len(evidence_docs) < 3:
            targeted_docs = retrieve_targeted_evidence(fact, top_k=5)
            existing_ids = {doc["id"] for doc in evidence_docs}
            for doc in targeted_docs:
                if doc["id"] not in existing_ids:
                    evidence_docs.append(doc)
        
        # Limit to top 10 documents
        evidence_docs = evidence_docs[:10]
        
        # Format evidence
        evidence_text = format_evidence_text(evidence_docs)
        
        # Build verification prompt
        prompt = VERIFICATION_PROMPT.format(fact=fact, evidence=evidence_text)
        
        # Generate verification
        response = llm.generate(
            [[{"role": "user", "content": prompt}]],
            sampling_params
        )[0]
        
        response_text = response.outputs[0].text.strip()
        
        # Parse response
        verification = parse_json_response(response_text)
        verification["fact"] = fact
        verification["evidence_doc_count"] = len(evidence_docs)
        
        fact_verifications.append(verification)
    
    # Calculate aggregate statistics
    supported_count = sum(1 for v in fact_verifications if v.get("verdict") == "SUPPORTED")
    refuted_count = sum(1 for v in fact_verifications if v.get("verdict") == "REFUTED")
    unclear_count = sum(1 for v in fact_verifications if v.get("verdict") == "UNCLEAR")
    total_count = len(fact_verifications)

    # Generate written feedback summary
    verification_summary = format_verification_summary(fact_verifications)
    
    summary_prompt = FACTUALITY_SUMMARY_PROMPT.format(
        query=query,
        verification_summary=verification_summary,
        total_facts=total_count,
        supported_count=supported_count,
        refuted_count=refuted_count,
        unclear_count=unclear_count
    )
    
    # Generate summary feedback
    summary_sampling_params = SamplingParams(temperature=0.3, max_tokens=512)
    summary_response = llm.generate(
        [[{"role": "user", "content": summary_prompt}]],
        summary_sampling_params
    )[0]
    
    summary_text = summary_response.outputs[0].text.strip()
    written_feedback = parse_json_response(summary_text)
    
    # If parsing fails, create detailed feedback structure
    if not written_feedback or "what_is_good" not in written_feedback:
        improvement_suggestions = []
        if unclear_count > 0:
            improvement_suggestions.append(f"Retrieve better evidence for {unclear_count} unclear facts (highest priority)")
        if refuted_count > 0:
            improvement_suggestions.append(f"Remove or correct {refuted_count} contradicted facts")
        if unclear_count > total_count * 0.2 or refuted_count > 0:
            improvement_suggestions.append("Strengthen overall evidence base with more authoritative sources")
        
        how_to_improve = "; ".join(improvement_suggestions) if improvement_suggestions else "Continue maintaining high evidence standards"
        
        unclear_facts_details = []
        refuted_facts_details = []
        
        for v in fact_verifications:
            if v.get("verdict") == "UNCLEAR":
                unclear_facts_details.append({
                    "fact": v.get("fact", "")[:100] + ("..." if len(v.get("fact", "")) > 100 else ""),
                    "reason": v.get("explanation", "Insufficient evidence"),
                    "needed_evidence": "More specific and authoritative sources"
                })
            elif v.get("verdict") == "REFUTED":
                refuted_facts_details.append({
                    "fact": v.get("fact", "")[:100] + ("..." if len(v.get("fact", "")) > 100 else ""),
                    "contradiction": v.get("explanation", "Contradicted by evidence"),
                    "suggested_correction": "Remove or find alternative phrasing"
                })
        
        # Determine if no improvements are needed (all facts supported)
        no_improvements = (unclear_count == 0 and refuted_count == 0)
        
        written_feedback = {
            "no_improvements_needed": no_improvements,
            "what_is_good": f"{supported_count} out of {total_count} facts are well-supported by evidence." + (f" Strong factual foundation in {supported_count} verified claims." if supported_count > 0 else ""),
            "critical_issues": "No critical issues - all facts are verified." if no_improvements else f"UNCLEAR FACTS (most problematic): {unclear_count} facts lack sufficient evidence. REFUTED FACTS: {refuted_count} facts are contradicted.",
            "overall_assessment": f"The answer is {'highly reliable - all facts verified' if no_improvements else 'needs significant improvement due to unclear facts' if unclear_count > total_count * 0.3 else 'mostly reliable but needs refinement'}.",
            "how_to_improve": "No improvements needed - all facts are well-supported by evidence." if no_improvements else how_to_improve,
            "unclear_facts_details": unclear_facts_details,
            "refuted_facts_details": refuted_facts_details
        }
    
    return {
        "verification": fact_verifications,
        "written_feedback": written_feedback,
        "stats": {
            "total_facts": total_count,
            "supported": supported_count,
            "refuted": refuted_count,
            "unclear": unclear_count
        },
        "scoring_weights": {
            "supported": 1.0,
            "refuted": 0.3,
            "unclear": 0.0
        }
    }

