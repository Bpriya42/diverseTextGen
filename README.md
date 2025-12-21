# Diverse Text Generation - Multi-Agent RAG System

An iterative multi-agent RAG (Retrieval-Augmented Generation) system built with LangGraph for generating diverse, comprehensive, and factually accurate responses.

## Overview

This system implements a 6-agent pipeline that iteratively refines responses through:

1. **Planner (Agent 1)**: Decomposes queries into aspects/perspectives
2. **Retriever (Agent 2)**: Retrieves relevant documents for each aspect
3. **Synthesizer (Agent 3)**: Generates comprehensive answers
4. **Fact Extractor (Agent 4)**: Extracts atomic facts from answers
5. **Verifier (Agent 5)**: Verifies facts against evidence
6. **Coverage Evaluator (Agent 6)**: Evaluates topic coverage

Agents 5 and 6 run in parallel for improved performance (40-50% faster iterations).

## Project Structure

```
diverseTextGen/
├── state.py               # LangGraph state schema
├── graph.py               # LangGraph workflow construction
├── requirements.txt       # Python dependencies
├── env.example            # Environment variable template
│
├── config/                # Configuration module
│   ├── __init__.py
│   └── settings.py        # All configurable settings
│
├── agents/                # Agent implementations (logic)
│   ├── planner.py
│   ├── retriever.py
│   ├── synthesizer.py
│   ├── fact_extractor.py
│   ├── verifier.py
│   └── coverage_evaluator.py
│
├── nodes/                 # LangGraph node wrappers
│   ├── planner.py
│   ├── retriever.py
│   ├── synthesizer.py
│   ├── fact_extractor.py
│   ├── parallel_verification.py
│   └── iteration_gate.py
│
├── llm/                   # LLM clients and prompts
│   ├── server_llm.py      # vLLM server client
│   ├── hf_llm.py          # HuggingFace direct inference
│   └── prompts/           # Prompt templates (optional)
│
├── retrieval/             # Dense retrieval system
│   └── retriever.py
│
├── data/                  # Data handling
│   ├── formatters.py
│   └── dataset.py
│
├── eval/                  # Evaluation modules
│   ├── icat.py            # ICAT-A evaluation
│   ├── llm_evaluator.py
│   ├── retriever.py
│   ├── experiment_tracker.py
│   └── visualizer.py
│
├── scripts/               # Entry point scripts
│   ├── run_langgraph.py   # Main RAG runner
│   ├── run_baseline_experiment.py
│   ├── run_full_experiment.py
│   ├── evaluate_icat.py
│   ├── compare_runs.py
│   └── *.sh               # SLURM job scripts
│
├── artifacts/             # Generated outputs (gitignored)
│   ├── runs/
│   ├── outputs/
│   └── logs/
│
└── docs/                  # Documentation
```

## Setup

### 1. Create and activate virtual environment

```bash
# Using conda (recommended)
conda create -n rag python=3.10
conda activate rag

# Or use an existing environment
source /path/to/conda/etc/profile.d/conda.sh
conda activate /path/to/env
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

Copy the example environment file and customize:

```bash
cp env.example .env
# Edit .env with your paths
```

Or set environment variables directly:
```bash
export RAG_DATA_DIR="/path/to/data"
export RAG_CACHE_DIR="/path/to/cache"
export RAG_CORPUS_PATH="/path/to/corpus.jsonl"
export RAG_SERVER_LOGS_DIR="/path/to/server_logs"
```

### 4. Start vLLM server

The system requires a vLLM server running:

```bash
# Start vLLM server
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --host 0.0.0.0 \
    --port 8000

# Create server log file with host and port
echo "localhost" > server_logs/log.txt
echo "8000" >> server_logs/log.txt
```

## Usage

All scripts should be run from the project root directory.

### Single Query

```bash
python scripts/run_langgraph.py \
    --query "What causes headaches and how can they be treated?" \
    --query_id "test_001" \
    --max_iterations 3 \
    --output ./output/test_result.json
```

### Batch Processing

```bash
python scripts/run_langgraph.py \
    --input_file ./data/queries.jsonl \
    -n 10 \
    --max_iterations 3 \
    --output ./output/batch_results
```

### With Budget Constraints

```bash
python scripts/run_langgraph.py \
    --query "Explain quantum computing" \
    --query_id "quantum_001" \
    --max_iterations 5 \
    --token_budget 50000 \
    --walltime_budget_s 300 \
    --output ./output/result.json
```

### Baseline Experiment (No RAG)

```bash
python scripts/run_baseline_experiment.py \
    --queries_path data/antique/train.jsonl \
    -n 10
```

### ICAT Evaluation

```bash
python scripts/evaluate_icat.py \
    --output_path ./output/batch_results \
    --corpus_path ./data/antique/corpus_filtered_50.jsonl
```

### Compare Runs

```bash
python scripts/compare_runs.py --list  # List all runs
python scripts/compare_runs.py --runs run1 run2 --output comparisons/
```

## Input Format

Queries should be in JSONL format:

```json
{"query_id": "q001", "query_description": "What causes headaches?"}
{"query_id": "q002", "query_description": "How to learn programming?"}
```

## Output Format

Results are saved as JSON with structure:

```json
{
  "query_id": "q001",
  "query": "What causes headaches?",
  "final_answer": "...",
  "total_iterations": 3,
  "termination_reason": "max_iterations",
  "iteration_history": [...],
  "total_runtime_seconds": 45.2,
  "timestamps": {...},
  "budget_used": {...}
}
```

## Configuration Options

See `env.example` for all available environment variables. Key options:

| Parameter | Environment Variable | Default | Description |
|-----------|---------------------|---------|-------------|
| Data directory | `RAG_DATA_DIR` | `./data` | Data storage path |
| Cache directory | `RAG_CACHE_DIR` | See config | Embedding cache path |
| Server log file | `RAG_SERVER_LOG_FILE` | `./server_logs/log.txt` | vLLM server info |
| Default model | `RAG_DEFAULT_MODEL` | `Qwen/Qwen3-4B-Instruct-2507` | LLM model name |
| Max iterations | `RAG_DEFAULT_MAX_ITERATIONS` | Unlimited | Default iteration limit |
| Top-K retrieval | `RAG_DEFAULT_TOP_K` | `5` | Documents per aspect |

## Architecture

```
Query → Planner → Retriever → Synthesizer → Fact Extractor
                                    ↓
                            ┌───────┴───────┐
                            ↓               ↓
                        Verifier    Coverage Evaluator
                            └───────┬───────┘
                                    ↓
                            Iteration Gate
                                    ↓
                         [Continue or Terminate]
```

## License

MIT License

## Citation

If you use this code, please cite:

```bibtex
@software{diverse_text_gen,
  title={Diverse Text Generation - Multi-Agent RAG System},
  author={...},
  year={2024},
  url={https://github.com/...}
}
```
