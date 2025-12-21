#!/bin/bash
# Setup script for Diverse Text Generation Multi-Agent RAG System

set -e

echo "=========================================="
echo "Setting up Diverse Text Generation System"
echo "=========================================="

# Check if conda is available
if command -v conda &> /dev/null; then
    echo "Using conda environment..."
    
    # Check if environment exists
    if conda env list | grep -q "diverse_rag"; then
        echo "Environment 'diverse_rag' exists. Activating..."
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate diverse_rag
    else
        echo "Creating new conda environment 'diverse_rag'..."
        conda create -n diverse_rag python=3.10 -y
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate diverse_rag
    fi
else
    echo "Conda not found. Using pip with virtual environment..."
    
    if [ ! -d "env" ]; then
        python -m venv env
    fi
    
    source env/bin/activate
fi

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Download NLTK data
echo ""
echo "Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt', quiet=True)"

# Create necessary directories
echo ""
echo "Creating directories..."
mkdir -p output
mkdir -p data

# Print configuration
echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Start your vLLM server"
echo "2. Ensure parent folder's server_logs/log.txt exists with:"
echo "   Line 1: hostname (e.g., localhost)"
echo "   Line 2: port (e.g., 8000)"
echo "   (This project uses ../server_logs/log.txt)"
echo ""
echo "3. Update config.py with your paths:"
echo "   - CORPUS_PATH: Path to your corpus file"
echo "   - CACHE_DIR: Path to embedding cache"
echo ""
echo "4. Run a test query:"
echo "   python run_langgraph.py --query 'Your question' --query_id 'test_001'"
echo ""
echo "For more information, see README.md"

