# Prompt Templates

This folder contains all prompt templates used by the 6-agent multi-agent RAG system. Centralizing prompts here makes them easy to modify, test, and version control.

## Structure

```
prompts/
├── __init__.py                    # Exports all prompts
├── planner_prompts.py             # Agent 1: Query planning prompts
├── synthesizer_prompts.py         # Agent 3: Answer synthesis prompt
├── fact_extractor_prompts.py      # Agent 4: Fact extraction prompt
├── verifier_prompts.py            # Agent 5: Verification prompts
├── coverage_prompts.py            # Agent 6: Coverage evaluation prompt
└── README.md                      # This file
```

## Files

### `planner_prompts.py`
- **INITIAL_PLAN_TEMPLATE**: Note - currently handled by `data/formatters.py`
- **REFINEMENT_PLAN_TEMPLATE**: Used to refine plans based on feedback from verifier and coverage evaluator

### `synthesizer_prompts.py`
- **ANSWER_SYNTHESIS_TEMPLATE**: Used to generate comprehensive answers from retrieved evidence

### `fact_extractor_prompts.py`
- **FACT_EXTRACTION_PROMPT**: Used to extract atomic facts from generated answers

### `verifier_prompts.py`
- **VERIFICATION_PROMPT**: Used to verify individual facts against evidence
- **FACTUALITY_SUMMARY_PROMPT**: Used to generate comprehensive factuality assessments

### `coverage_prompts.py`
- **COVERAGE_EVALUATION_PROMPT**: Used to evaluate plan and answer coverage quality

## How to Modify Prompts

1. **Edit the template** in the appropriate file
2. **Maintain placeholder format**: Use `{variable_name}` for variables
3. **Test your changes**: Run a query through the system
4. **Version control**: Commit prompt changes with descriptive messages

## Example: Modifying a Prompt

```python
# prompts/fact_extractor_prompts.py

FACT_EXTRACTION_PROMPT = """You are a precise information extraction system.

Based on the given text, extract all atomic facts.

Text:
{answer}

Output format:
- List facts one per line
- Be specific and concise
- Include only verifiable claims"""
```

## Template Variables

Each prompt uses specific variables that are filled in by the agent:

### Planner (Refinement)
- `query`: Original user query
- `current_plan`: Current plan as JSON
- `missing_salient_points`: Missing coverage points
- `refuted_facts_details`: Contradicted facts
- `unclear_facts_details`: Unclear facts
- `plan_improvements`: Suggested improvements

### Synthesizer
- `query`: User query
- `plan_text`: Formatted plan aspects
- `retrieved_text`: Retrieved documents

### Fact Extractor
- `answer`: Generated answer text

### Verifier
- `fact`: Atomic fact to verify
- `evidence`: Retrieved evidence documents
- `query`: Original query
- `verification_summary`: Summary of verifications
- `total_facts`, `supported_count`, `refuted_count`, `unclear_count`: Statistics

### Coverage Evaluator
- `query`: User query
- `plan_aspects`: Formatted plan
- `answer`: Generated answer

## Best Practices

1. **Be specific**: Clear instructions produce better results
2. **Use examples**: Show the LLM what you want
3. **Specify format**: JSON, list, paragraph, etc.
4. **Add constraints**: Length limits, required fields, etc.
5. **Test iteratively**: Small changes can have big effects

## Testing Prompts

To test a prompt change:

```bash
# Run a single query
python scripts/run_langgraph.py \
    --query "What causes headaches?" \
    --query_id "prompt_test_001"

# Check output quality in output/prompt_test_001.json
```

## Notes

- **Agent 2 (Retriever)**: No prompts - uses embedding-based retrieval
- **Agent 1 Initial Planning**: Currently uses formatter from `data/formatters.py` (may centralize in future)
- **Quality Termination**: Verifier and Coverage prompts include special instructions for quality-based termination

## Contributing

When modifying prompts:
1. Document your changes in commit messages
2. Note any performance impacts (good or bad)
3. Share insights with the team
4. Consider A/B testing major changes

