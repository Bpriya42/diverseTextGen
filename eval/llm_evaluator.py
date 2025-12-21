"""
Custom LLMEvaluator for eval folder that uses ServerLLM.

This replaces the original ICAT LLMEvaluator to use our ServerLLM infrastructure.
"""

from typing import List

from vllm import SamplingParams

from config.settings import DEFAULT_MODEL, LLM_MAX_RETRIES, LLM_NUM_WORKERS
from llm.server_llm import ServerLLM, load_url_from_log_file


# ICAT's fact generation prompt
PROMPT_TEMPLATE_FACTS = "Based on the given text, give all the mentioned atomic fact sentences, one per line. Each sentence should be decontextualized with resolved pronouns (eg. don't use 'this' or 'that', mention the actual object) and self-explanatory without any additional context. text: "


class LLMEvaluator:
    """
    Custom LLMEvaluator that uses ServerLLM instead of vLLM or OpenAI API.
    This maintains the same interface as ICAT's LLMEvaluator.
    """
    
    def __init__(
        self,
        server_log_path: str = None,
        base_model: str = None,
        facts_model: str = None,
        max_retries: int = None,
        num_workers: int = None,
        # Keep these for compatibility but ignore them
        api_base_llm: str = None,
        api_facts_llm: str = None,
        hf_token: str = None,
        openai_api_key: str = None,
        openai_base_url: str = None,
        vllm_logging_level: str = None,
        cache_path: str = None,
        **kwargs
    ):
        """
        Initialize with ServerLLM.
        
        Args:
            server_log_path: Path to log file with server URL
            base_model: Model name for general generation
            facts_model: Model name for fact generation (defaults to base_model)
            max_retries: Max retries for API calls
            num_workers: Number of parallel workers
        """
        base_model = base_model or DEFAULT_MODEL
        max_retries = max_retries or LLM_MAX_RETRIES
        num_workers = num_workers or LLM_NUM_WORKERS
        
        url = load_url_from_log_file(server_log_path)
        self.base_llm = ServerLLM(
            base_url=url,
            model=base_model,
            max_retries=max_retries,
            num_workers=num_workers
        )
        
        # Use same model for facts if not specified
        self.facts_model = facts_model or base_model
        if facts_model and facts_model != base_model:
            self.facts_llm = ServerLLM(
                base_url=url,
                model=facts_model,
                max_retries=max_retries,
                num_workers=num_workers
            )
        else:
            self.facts_llm = self.base_llm
    
    def generate(self, texts: List[str]) -> List[str]:
        """
        Generate responses for a list of prompts.
        
        Args:
            texts: List of prompt strings
            
        Returns:
            List of generated text responses
        """
        messages = [
            [{"role": "user", "content": text}]
            for text in texts
        ]
        
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=2048
        )
        
        responses = self.base_llm.generate(messages, sampling_params)
        return [
            r.outputs[0].text if r.outputs[0].text != "cannot generate a response" 
            else ""  # Return empty string for failed generations
            for r in responses
        ]
    
    def generate_facts(self, texts: List[str]) -> List[List[str]]:
        """
        Generate atomic facts from a list of texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of lists of atomic facts (one list per input text)
        """
        prompts = [PROMPT_TEMPLATE_FACTS + text for text in texts]
        
        messages = [
            [{"role": "user", "content": prompt}]
            for prompt in prompts
        ]
        
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=2048
        )
        
        responses = self.facts_llm.generate(messages, sampling_params)
        
        results = []
        for i, response in enumerate(responses):
            generated_text = response.outputs[0].text
            
            # Handle failed generation
            if generated_text == "cannot generate a response":
                print(f"WARNING: Failed to generate facts for response {i+1}/{len(responses)}")
                results.append([])  # Return empty list for failed generations
                continue
            
            facts = generated_text.split('\n')
            facts = [fact.strip() for fact in facts if fact.strip()]
            results.append(facts)
        
        return results


if __name__ == "__main__":
    # Test the evaluator
    llm_evaluator = LLMEvaluator()
    
    print("Testing generate method...")
    responses = llm_evaluator.generate(["What is the capital of France?"])
    print("Response:", responses[0])
    
    print("\nTesting generate_facts method...")
    facts = llm_evaluator.generate_facts(["The quick brown fox jumps over the lazy dog. The dog is a good dog."])
    print("Facts:", facts[0])

