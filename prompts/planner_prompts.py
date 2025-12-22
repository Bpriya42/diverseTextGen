"""
Agent 1: Query Planner Prompts

Prompts for generating initial query decomposition and refining plans based on feedback.
"""

# Note: Initial plan template is handled by data.formatters.get_query_planning_formatter()
# This is kept here for reference and potential future centralization
INITIAL_PLAN_TEMPLATE = """[Placeholder - Currently handled by formatter in data/formatters.py]

The initial planning prompt is generated dynamically using get_query_planning_formatter()
from data.formatters module. Consider centralizing here in future refactoring."""


REFINEMENT_PLAN_TEMPLATE = """You are refining a query decomposition plan based on comprehensive feedback from fact verification and coverage evaluation.

Original Query: {query}

Current Plan:
{current_plan}

FEEDBACK ANALYSIS:

1. MISSING SALIENT POINTS (High Priority - Add New Aspects):
{missing_salient_points}

2. REFUTED FACTS (Critical - Add Exclusion Instructions):
{refuted_facts_details}

3. UNCLEAR FACTS (Moderate Priority - Improve Evidence Gathering):
{unclear_facts_details}

4. PLAN QUALITY IMPROVEMENTS:
{plan_improvements}

REFINEMENT INSTRUCTIONS:

A. ADD aspects for missing salient points (highest priority)
B. MODIFY existing aspects to exclude contradicted information with explicit instructions like:
   - "Exclude information about [specific contradicted topic]"
   - "Focus on [alternative reliable sources/perspectives]" 
C. ENHANCE aspects that led to unclear facts by:
   - Adding more specific retrieval queries
   - Targeting authoritative sources
   - Including evidence requirements
D. REMOVE redundant or ineffective aspects
E. Maximum 7 aspects total

CRITICAL JSON FORMATTING REQUIREMENTS:
- Output ONLY valid JSON array, nothing else
- Enclose your output in ```json and ``` markers
- Use double quotes for all strings (not single quotes)
- Escape special characters in strings (use \\" for quotes, \\n for newlines)
- Ensure all strings are properly closed
- No trailing commas after the last item
- Each object must have exactly these three fields: "aspect", "query", "reason"

Output format:
```json
[
  {{"aspect": "...", "query": "...", "reason": "..."}},
  {{"aspect": "...", "query": "...", "reason": "..."}}
]
```

Example of valid JSON:
```json
[
  {{"aspect": "Medical causes", "query": "What medical conditions cause symptoms?", "reason": "Identifies specific diseases"}},
  {{"aspect": "Treatment options", "query": "How to treat the condition?", "reason": "Provides actionable solutions"}}
]
```"""

