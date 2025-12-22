"""
Agent 4: Atomic Fact Extractor Prompts

Prompt for extracting atomic facts from generated answers.
"""

FACT_EXTRACTION_PROMPT = """You are a precise information extraction system.

Based on the given text, list all the mentioned atomic fact sentences, one per line. Each sentence should be decontextualized with resolved pronouns (for example, don't use 'this' or 'that', but mention the actual object) and self-explanatory without any additional context.

Text:
{answer}

Output:
- One atomic fact per line.
- No numbering.
- No explanations."""

