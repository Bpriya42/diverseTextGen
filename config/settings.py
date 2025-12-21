"""
Configuration for Multi-Agent RAG System.

This file centralizes all configurable paths and parameters.
Modify these values for your environment.
"""

import os
from pathlib import Path

# ============================================================================
# BASE PATHS
# ============================================================================

# Project root directory (auto-detected - go up one level from config/)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Data directory - where corpus and queries are stored
DATA_DIR = os.environ.get(
    "RAG_DATA_DIR",
    str(PROJECT_ROOT / "data")
)

# Output directory - where results are saved
OUTPUT_DIR = os.environ.get(
    "RAG_OUTPUT_DIR",
    str(PROJECT_ROOT / "output")
)

# Cache directory - where embeddings and indices are cached
CACHE_DIR = os.environ.get(
    "RAG_CACHE_DIR",
    "/gypsum/work1/zamani/asalemi/RAG_VS_LoRA_Personalization/cache/index_coverage_score_antique"
)

# Server logs directory - where vLLM server logs are stored
# Use parent folder's server_logs instead of diverseTextGen's
SERVER_LOGS_DIR = os.environ.get(
    "RAG_SERVER_LOGS_DIR",
    "/rstor/pi_hzamani_umass_edu/asalemi/priya/server_logs"
)

# ============================================================================
# SERVER CONFIGURATION
# ============================================================================

# Path to log file containing vLLM server URL
SERVER_LOG_FILE = os.environ.get(
    "RAG_SERVER_LOG_FILE",
    str(Path(SERVER_LOGS_DIR) / "log.txt")
)

# Default model name
DEFAULT_MODEL = os.environ.get(
    "RAG_DEFAULT_MODEL",
    "Qwen/Qwen3-4B-Instruct-2507"
)

# Use HuggingFace direct inference instead of vLLM server
# Set to True for local testing without vLLM server
# Can also be set via environment variable: export RAG_USE_HF_DIRECT=true
# USE_HUGGINGFACE_DIRECT = os.environ.get("RAG_USE_HF_DIRECT", "false").lower() == "true"
USE_HUGGINGFACE_DIRECT = False
# ============================================================================
# CORPUS PATHS
# ============================================================================

# Default corpus path
CORPUS_PATH = os.environ.get(
    "RAG_CORPUS_PATH",
    str(Path(DATA_DIR) / "antique" / "corpus_filtered_50.jsonl")
)

# ============================================================================
# RETRIEVAL CONFIGURATION
# ============================================================================

# Sentence transformer model for embeddings
EMBEDDING_MODEL = os.environ.get(
    "RAG_EMBEDDING_MODEL",
    "Snowflake/snowflake-arctic-embed-l"
)

# Default number of documents to retrieve per aspect
DEFAULT_TOP_K = int(os.environ.get("RAG_DEFAULT_TOP_K", "5"))

# Batch size for embedding computation
EMBEDDING_BATCH_SIZE = int(os.environ.get("RAG_EMBEDDING_BATCH_SIZE", "512"))

# ============================================================================
# LLM CONFIGURATION
# ============================================================================

# Maximum retries for LLM API calls
LLM_MAX_RETRIES = int(os.environ.get("RAG_LLM_MAX_RETRIES", "10"))

# Number of parallel workers for LLM
LLM_NUM_WORKERS = int(os.environ.get("RAG_LLM_NUM_WORKERS", "1"))

# Sampling parameters
PLANNER_TEMPERATURE = float(os.environ.get("RAG_PLANNER_TEMPERATURE", "0.7"))
PLANNER_MAX_TOKENS = int(os.environ.get("RAG_PLANNER_MAX_TOKENS", "512"))

SYNTHESIZER_TEMPERATURE = float(os.environ.get("RAG_SYNTHESIZER_TEMPERATURE", "0.7"))
SYNTHESIZER_MAX_TOKENS = int(os.environ.get("RAG_SYNTHESIZER_MAX_TOKENS", "512"))

VERIFIER_TEMPERATURE = float(os.environ.get("RAG_VERIFIER_TEMPERATURE", "0.3"))
VERIFIER_MAX_TOKENS = int(os.environ.get("RAG_VERIFIER_MAX_TOKENS", "256"))

# ============================================================================
# LANGGRAPH CONFIGURATION
# ============================================================================

# Default maximum iterations (None = unlimited, controlled by budget only)
_max_iter_env = os.environ.get("RAG_DEFAULT_MAX_ITERATIONS", "")
DEFAULT_MAX_ITERATIONS = int(_max_iter_env) if _max_iter_env else None

# Default memory thresholds for termination (percentage)
DEFAULT_MAX_RAM_PERCENT = float(os.environ.get("RAG_MAX_RAM_PERCENT", "90"))
DEFAULT_MAX_GPU_PERCENT = float(os.environ.get("RAG_MAX_GPU_PERCENT", "90"))

# Checkpointer path for LangGraph
CHECKPOINTER_PATH = os.environ.get(
    "RAG_CHECKPOINTER_PATH",
    str(PROJECT_ROOT / ".checkpoints.sqlite")
)

# ============================================================================
# NLI MODEL CONFIGURATION (for ICAT evaluation)
# ============================================================================

NLI_MODEL_NAME = os.environ.get(
    "RAG_NLI_MODEL_NAME",
    "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
)

NLI_BATCH_SIZE = int(os.environ.get("RAG_NLI_BATCH_SIZE", "8"))
LLM_BATCH_SIZE = int(os.environ.get("RAG_LLM_BATCH_SIZE", "4"))

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_server_url():
    """Load server URL from log file."""
    log_file = Path(SERVER_LOG_FILE)
    if not log_file.exists():
        raise FileNotFoundError(
            f"Server log file not found: {log_file}\n"
            f"Make sure vLLM server is running and log file is created."
        )
    
    with open(log_file, "r") as f:
        lines = f.readlines()
    
    if len(lines) < 2:
        raise ValueError(
            f"Invalid server log file format: {log_file}\n"
            f"Expected format: host on line 1, port on line 2"
        )
    
    host = lines[0].strip()
    port = lines[1].strip()
    return f"http://{host}:{port}/v1"


def ensure_directories():
    """Create necessary directories if they don't exist."""
    # Only create DATA_DIR and OUTPUT_DIR
    # SERVER_LOGS_DIR is in the parent folder and already exists
    dirs = [DATA_DIR, OUTPUT_DIR]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


# Print configuration on import (useful for debugging)
def print_config():
    """Print current configuration."""
    print("=" * 60)
    print("Multi-Agent RAG System Configuration")
    print("=" * 60)
    print(f"PROJECT_ROOT:      {PROJECT_ROOT}")
    print(f"DATA_DIR:          {DATA_DIR}")
    print(f"OUTPUT_DIR:        {OUTPUT_DIR}")
    print(f"CACHE_DIR:         {CACHE_DIR}")
    print(f"SERVER_LOGS_DIR:   {SERVER_LOGS_DIR}")
    print(f"SERVER_LOG_FILE:   {SERVER_LOG_FILE}")
    print(f"CORPUS_PATH:       {CORPUS_PATH}")
    print(f"DEFAULT_MODEL:     {DEFAULT_MODEL}")
    print("=" * 60)


if __name__ == "__main__":
    print_config()

