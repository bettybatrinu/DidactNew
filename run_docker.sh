#!/usr/bin/env bash
set -euo pipefail

# Build and run the didactai Docker image (Python 3.11 + TensorFlow CPU)
# Usage: ./run_docker.sh

IMAGE=didactai:latest

echo "Building Docker image ${IMAGE}..."
docker build -t "${IMAGE}" .

echo "Running container and mapping port 8501 -> 8501"
docker run --rm -p 8501:8501 -v "$(pwd)":/app -w /app "${IMAGE}"
