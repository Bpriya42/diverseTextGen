"""
Agent 3: Answer Synthesizer Prompts

Prompt for generating comprehensive answers from retrieved evidence.
"""

ANSWER_SYNTHESIS_TEMPLATE = """You are a helpful assistant that answers complex user questions using evidence retrieved from multiple aspects of the topic.

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

