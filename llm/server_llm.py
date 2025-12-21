"""
Server LLM Client

Wrapper for vLLM server providing OpenAI-compatible API access.
"""

from openai import OpenAI
from vllm import SamplingParams
import time
from concurrent.futures import ThreadPoolExecutor
import tqdm

from config.settings import SERVER_LOG_FILE, USE_HUGGINGFACE_DIRECT


def batchify(items: list, batch_size: int):
    """Split a list into batches."""
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def load_url_from_log_file(log_addr: str = None):
    """
    Load server URL from log file.
    
    Args:
        log_addr: Path to log file. If None, uses config default.
        
    Returns:
        Server URL string
    """
    log_addr = log_addr or SERVER_LOG_FILE
    with open(log_addr, "r") as f:
        lines = f.readlines()
    host = lines[0].strip()
    port = lines[1].strip()
    url = f"http://{host}:{port}/v1"
    return url


class ServerLLMOutput:
    """Wrapper for LLM output text."""
    
    def __init__(self, text):
        self.text = text

    def __getitem__(self, index):
        return self.text


class ServerLLMResponse:
    """Wrapper for LLM response with multiple outputs."""
    
    def __init__(self, texts):
        self.outputs = [ServerLLMOutput(text) for text in texts]

    def __getitem__(self, index):
        return self.outputs[index]


def get_response_from_server(
    client,
    messages,
    model,
    sampling_params,
    max_retries
):
    """
    Get response from vLLM server.
    
    Args:
        client: OpenAI client
        messages: Chat messages
        model: Model name
        sampling_params: vLLM SamplingParams
        max_retries: Maximum retry attempts
        
    Returns:
        Response object or error message
    """
    temperature = sampling_params.temperature
    retries = 0
    base_delay = 1.0
    
    while True:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=sampling_params.max_tokens,
                temperature=temperature,
                top_p=sampling_params.top_p,
                n=sampling_params.n,
            )
            return response
        except Exception as e:
            retries += 1
            if retries >= max_retries:
                print(f"ERROR: Failed after {max_retries} retries. Exception: {e}")
                return "cannot generate a response"
            
            # Exponential backoff: 1s, 2s, 4s, 8s, etc. (max 30s)
            delay = min(base_delay * (2 ** (retries - 1)), 30.0)
            print(f"Retrying... ({retries}/{max_retries}) - waiting {delay:.1f}s. Error: {type(e).__name__}")
            time.sleep(delay)


class ServerLLM:
    """
    Client for vLLM server with parallel request support.
    
    Can transparently delegate to HuggingFaceLLM when USE_HUGGINGFACE_DIRECT is True.
    
    Args:
        base_url: Server URL (e.g., "http://localhost:8000/v1")
        model: Model name
        max_retries: Maximum retries per request
        num_workers: Number of parallel workers
    """
    
    def __init__(
        self,
        base_url: str,
        model: str,
        max_retries: int = 10,
        num_workers: int = 1
    ):
        # If HF direct mode is enabled, use HuggingFaceLLM instead of vLLM server
        if USE_HUGGINGFACE_DIRECT:
            from llm.hf_llm import HuggingFaceLLM
            print("[ServerLLM] Using HuggingFace direct inference instead of vLLM server")
            self._hf_delegate = HuggingFaceLLM(model_name=model)
            # Keep these for interface compatibility, but won't be used
            self.api_key = None
            self.base_url = None
            self.client = None
            self.model = model
            self.max_retries = max_retries
            self.num_workers = 1
            return
        
        # Original vLLM server initialization
        self.api_key = "EMPTY"
        self.base_url = base_url
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.model = model
        self.max_retries = max_retries
        self.num_workers = num_workers
        self._hf_delegate = None

    def generate(
        self,
        messages: list,
        sampling_params: SamplingParams = None
    ):
        """
        Generate responses for multiple message sequences.
        
        Args:
            messages: List of message lists (each inner list is a conversation)
            sampling_params: vLLM SamplingParams
            
        Returns:
            List of ServerLLMResponse objects (or HFLLMResponse when in HF mode)
        """
        # If we are in HF mode, fully delegate to HuggingFaceLLM
        if getattr(self, "_hf_delegate", None) is not None:
            return self._hf_delegate.generate(messages, sampling_params)
        
        # Original vLLM server implementation
        if sampling_params is None:
            sampling_params = SamplingParams()
            
        saved_messages = []
        for msg_batch in tqdm.tqdm(batchify(messages, self.num_workers)):
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = [
                    executor.submit(
                        get_response_from_server,
                        self.client,
                        msg,
                        self.model,
                        sampling_params,
                        self.max_retries
                    )
                    for msg in msg_batch
                ]
                results = [future.result() for future in futures]
                saved_messages.extend(results)
        return [self._post_process(response) for response in saved_messages]

    def _post_process(self, response):
        """Convert raw response to ServerLLMResponse."""
        contents = []
        if response == "cannot generate a response":
            return ServerLLMResponse(["cannot generate a response"])
        for choice in response.choices:
            content = choice.message.content
            contents.append(content)
        return ServerLLMResponse(contents)

