#!/bin/bash
# SAFIM Composition Environment Build Script

set -e

echo "Building SAFIM environment..."

# Install Python dependencies
echo "Installing Python dependencies..."
python -m pip install \
    Jinja2==3.1.2 \
    openai==0.28.1 \
    tiktoken==0.5.2 \
    transformers==4.36.0 \
    tqdm==4.64.1 \
    tree-sitter==0.20.4 \
    requests==2.28.1 \
    datasets==2.18.0

# Build ExecEval Docker image
echo "Building ExecEval Docker image..."
cd ExecEval
docker build . -t exec-eval:1.0

echo "Build completed successfully!"
echo ""
echo "Next steps:"
echo "  1. Start the ExecEval service: ./start_docker.sh"
echo "  2. Run composition generation: python generate_composition.py <level> <sample_size>"
