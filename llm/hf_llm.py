"""
Hugging Face LLM Client

Direct inference using Hugging Face transformers (no server required).
Provides same interface as ServerLLM for transparent switching.
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import SamplingParams

from config.settings import DEFAULT_MODEL

# =============================================================================
# HARDCODED HF TOKEN (for testing - do NOT commit to public git!)
# =============================================================================
HF_TOKEN = os.getenv("HF_TOKEN")
# =============================================================================


class HFLLMOutput:
    """Wrapper for LLM output text (matches ServerLLMOutput interface)."""
    
    def __init__(self, text):
        self.text = text

    def __getitem__(self, index):
        return self.text


class HFLLMResponse:
    """Wrapper for LLM response with multiple outputs (matches ServerLLMResponse interface)."""
    
    def __init__(self, texts):
        self.outputs = [HFLLMOutput(text) for text in texts]

    def __getitem__(self, index):
        return self.outputs[index]


class HuggingFaceLLM:
    """
    Direct Hugging Face inference client.
    
    Provides same interface as ServerLLM for transparent delegation.
    Uses singleton pattern to avoid reloading model multiple times.
    
    Args:
        model_name: HuggingFace model name/path (defaults to DEFAULT_MODEL from config)
        device: Device to use ('cuda', 'cpu', or 'auto')
        torch_dtype: Torch dtype for model (defaults to float16 on GPU, float32 on CPU)
    """
    
    _instance = None
    _model = None
    _tokenizer = None
    
    def __new__(cls, model_name: str = None, device: str = "auto", torch_dtype=None):
        # Singleton pattern to avoid loading model multiple times
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(
        self,
        model_name: str = None,
        device: str = "auto",
        torch_dtype=None
    ):
        model_name = model_name or DEFAULT_MODEL
        
        # Only load if not already loaded
        if HuggingFaceLLM._model is None:
            print(f"[HuggingFaceLLM] Loading model: {model_name}")
            
            if torch_dtype is None:
                torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            
            try:
                HuggingFaceLLM._tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    token=HF_TOKEN
                )
                
                HuggingFaceLLM._model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch_dtype,
                    device_map="cpu",
                    trust_remote_code=True,
                    token=HF_TOKEN
                )
                
                if HuggingFaceLLM._tokenizer.pad_token is None:
                    HuggingFaceLLM._tokenizer.pad_token = HuggingFaceLLM._tokenizer.eos_token
                
                print(f"[HuggingFaceLLM] Model loaded successfully on device: {HuggingFaceLLM._model.device}")
            except Exception as e:
                print(f"[HuggingFaceLLM] ERROR loading model: {e}")
                raise
        
        self.model = HuggingFaceLLM._model
        self.tokenizer = HuggingFaceLLM._tokenizer
        self.model_name = model_name

    def generate(
        self,
        messages: list,
        sampling_params: SamplingParams = None
    ):
        """
        Generate responses for multiple message sequences.
        
        Matches ServerLLM.generate() interface exactly.
        
        Args:
            messages: List of message lists (each inner list is a conversation)
            sampling_params: vLLM SamplingParams (converted to HF generate kwargs)
            
        Returns:
            List of HFLLMResponse objects (same interface as ServerLLMResponse)
        """
        if sampling_params is None:
            sampling_params = SamplingParams()
        
        responses = []
        
        for msg_list in messages:
            try:
                # Apply chat template
                text = self.tokenizer.apply_chat_template(
                    msg_list,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
                input_length = inputs.input_ids.shape[1]
                
                # Generate
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=sampling_params.max_tokens or 512,
                        temperature=max(sampling_params.temperature, 0.01),  # Avoid 0
                        top_p=sampling_params.top_p if sampling_params.top_p else 1.0,
                        do_sample=sampling_params.temperature > 0,
                        pad_token_id=self.tokenizer.pad_token_id,
                        num_return_sequences=sampling_params.n or 1
                    )
                
                # Decode only the new tokens
                generated_texts = []
                for output in outputs:
                    new_tokens = output[input_length:]
                    text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                    generated_texts.append(text.strip())
                
                responses.append(HFLLMResponse(generated_texts))
                
            except Exception as e:
                print(f"[HuggingFaceLLM] ERROR generating response: {e}")
                # Return error response matching ServerLLM behavior
                responses.append(HFLLMResponse(["cannot generate a response"]))
        
        return responses

