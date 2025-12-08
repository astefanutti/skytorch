#!/bin/bash

# Generate Python gRPC code from proto file
# Run this script after installing grpcio-tools: pip install grpcio-tools

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

cd "$PROJECT_ROOT"

python -m grpc_tools.protoc \
    -I. \
    --python_out=. \
    --pyi_out=. \
    --grpc_python_out=. \
    kpu/torch/server/service.proto

echo "Generated gRPC code successfully"
