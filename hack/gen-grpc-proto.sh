#!/bin/bash

# Generate Python gRPC code from proto files
# Run this script after installing grpcio-tools: pip install grpcio-tools

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# Use .venv Python if available, otherwise fall back to system python/python3
if [ -f ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
elif command -v python3 &> /dev/null; then
    PYTHON="python3"
else
    PYTHON="python"
fi

echo "Generating gRPC code for PyTorch service..."
$PYTHON -m grpc_tools.protoc \
    -I. \
    --python_out=. \
    --pyi_out=. \
    --grpc_python_out=. \
    skytorch/torch/server/service.proto

echo "Generating gRPC code for Health service..."
$PYTHON -m grpc_tools.protoc \
    -I. \
    --python_out=. \
    --pyi_out=. \
    --grpc_python_out=. \
    skytorch/server/health/health.proto

echo "Generating gRPC code for Metrics service..."
$PYTHON -m grpc_tools.protoc \
    -I. \
    --python_out=. \
    --pyi_out=. \
    --grpc_python_out=. \
    skytorch/server/metrics/metrics.proto

echo "Generated gRPC code successfully"
