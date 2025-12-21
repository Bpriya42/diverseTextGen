"""
Prompt Formatters

Templates for formatting prompts for different agents.
"""

import random
import json


def _get_document_text(doc):
    """Format a document for inclusion in a prompt."""
    return f'text: {doc["text"]}'


def get_baseline_no_rag_formatter(train=False):
    """
    Formatter for baseline (no RAG) responses.
    
    Args:
        train: Whether this is for training data
        
    Returns:
        Formatter function
    """
    def formatter(data):
        texts = []
        for i in range(len(data['query'])):
            user_prompt = data['query'][i]
            text = f"""Your task is to generate a comprehensive and factual response to the following query:
            query: {user_prompt}
            response:"""
            if train:
                text = text + f" {data['output'][i]}<eos>"
            texts.append(text)
        return texts
    return formatter


def get_baseline_no_rag_cot_formatter():
    """
    Formatter for baseline (no RAG) with chain-of-thought.
    
    Returns:
        Formatter function
    """
    def formatter(data):
        texts = []
        for i in range(len(data['query'])):
            user_prompt = data['query'][i]
            text = f"""Your task is to generate a comprehensive and factual response to the following query. You should first think step by step about the information that is needed to be present in the answer to the query and then generate a response that is both comprehensive and factually accurate. You should start your thinking by "thought:" and your final response to the query by "response:".
            query: {user_prompt}
            thought:"""
            texts.append(text)
        return texts
    return formatter


def get_baseline_rag_cot_formatter(num_contexts):
    """
    Formatter for RAG with chain-of-thought.
    
    Args:
        num_contexts: Number of context documents to include
        
    Returns:
        Formatter function
    """
    def formatter(data):
        assert "context" in data, 'context should be in the dataset'
        texts = []
        for i in range(len(data['query'])):
            user_prompt = data['query'][i]
            context = data['context'][i][:num_contexts]
            combined_context = "\n\n".join([_get_document_text(doc) for doc in context])
            text = f"""Your task is to generate a comprehensive and factual response to the following query. You should first think step by step about the information that is needed to be present in the answer to the query and then generate a response that is both comprehensive and factually accurate. You should start your thinking by "thought:" and your final response to the query by "response:". You can use the information provided in the context to generate a more comprehensive and factual response.
            query: 
            {user_prompt}
            context: 
            {combined_context}
            thought:"""
            texts.append(text)
        return texts
    return formatter


def get_query_planning_formatter(train=False):
    """
    Formatter for query decomposition planning.
    
    Args:
        train: Whether this is for training data
        
    Returns:
        Formatter function
    """
    def formatter(data):
        texts = []
        for i in range(len(data['query'])):
            user_prompt = data['query'][i]
            text = f"""Your task is to convert the following search query into maximum 5 diverse aspects and perspectives that cover all aspects of the original query. The aspects and perspectives should be non-overlapping and should not be redundant. The aspects and perspectives should cover all aspects that a comprehensive response to the original search query should cover.

CRITICAL JSON FORMATTING REQUIREMENTS:
- Output ONLY valid JSON array, nothing else
- Enclose your output in ```json and ``` markers
- Use double quotes for all strings (not single quotes)
- Escape special characters in strings (use \\" for quotes, \\n for newlines)
- Ensure all strings are properly closed
- No trailing commas after the last item
- Each object must have exactly these three fields: "aspect", "query", "reason"

Your output format:
```json
[
  {{"aspect": "...", "query": "...", "reason": "..."}},
  {{"aspect": "...", "query": "...", "reason": "..."}}
]
```

Example of valid JSON:
```json
[
  {{"aspect": "Medical causes", "query": "What medical conditions cause knee swelling?", "reason": "Identifies specific diseases"}},
  {{"aspect": "Treatment options", "query": "How to treat knee swelling and pain?", "reason": "Provides actionable solutions"}}
]
```

query: {user_prompt}
output: ```json"""
            if train:
                text = text + f" {data['output'][i]}<eos>"
            texts.append(text)
        return texts
    return formatter


def get_query_planning_global_search_formatter(train=False, max_length=4096, tokenizer=None):
    """
    Formatter for global search query planning comparison.
    
    Args:
        train: Whether this is for training data
        max_length: Maximum token length
        tokenizer: Optional tokenizer for length checking
        
    Returns:
        Formatter function
    """
    def formatter(data):
        texts = []
        for i in range(len(data['query'])):
            user_prompt = data['query'][i]
            if train:
                neg_response = data['output_neg'][i]
                pos_response = data['output_pos'][i]
                pos_location = random.choice([0, 1])
                if pos_location == 0:
                    response_1 = pos_response
                    response_2 = neg_response
                    answer = 1
                else:
                    response_1 = neg_response
                    response_2 = pos_response
                    answer = 2
            else:
                response_1 = data['output_1'][i]
                response_2 = data['output_2'][i]
            text = f"""Your task is to choose the response that is more comprehensive and accurate between the two provided responses to the query. 

            query: {user_prompt}

            response 1: 
            {response_1.strip()}
            
            response 2:
            {response_2.strip()}

            selected output: """
            if train:
                text = text + f"{answer}<eos>"
                if tokenizer is not None:
                    tokens = tokenizer(text)['input_ids']
                    if len(tokens) > max_length:
                        print(f"Length of the text is {len(tokens)}")
                        continue
            texts.append(text)
        return texts
    return formatter

