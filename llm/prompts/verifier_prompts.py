"""
Prompts for Agent 5: Fact Verifier

Verifies atomic facts against retrieved evidence.
"""


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

