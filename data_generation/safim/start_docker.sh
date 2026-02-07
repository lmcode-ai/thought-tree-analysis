#!/bin/bash
# Start ExecEval Docker Service

echo "Starting ExecEval Docker container..."

# Check if Docker image exists
if ! docker image inspect exec-eval:1.0 >/dev/null 2>&1; then
    echo "Error: Docker image 'exec-eval:1.0' not found."
    echo "Please run ./build.sh first to build the required image."
    exit 1
fi

echo "Starting container with ExecEval service on port 5000..."
echo "Container will run with 2 workers"
echo "Press Ctrl+C to stop the container"

docker run -it -p 5000:5000 -e NUM_WORKERS=2 exec-eval:1.0
