#!/usr/bin/env bash

set -o errexit
set -o nounset

# Source container runtime utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/container-runtime.sh"

# Setup container runtime
setup_container_runtime

API_VERSION="${VERSION}"
API_OUTPUT_PATH="."
PKG_ROOT="${API_OUTPUT_PATH}/kpu/client"

OPENAPI_GENERATOR_VERSION="v7.13.0"
ROOT_DIR="$(pwd)"
SWAGGER_CODEGEN_CONF="hack/swagger/config.json"
SWAGGER_CODEGEN_FILE="hack/swagger/swagger.json"

echo "Generating Python API models using ${CONTAINER_RUNTIME}..."
# We need to add user to allow container override existing files.
${CONTAINER_RUNTIME} run --user "$(id -u)":"$(id -g)" --rm \
  -v "${ROOT_DIR}:/local" docker.io/openapitools/openapi-generator-cli:${OPENAPI_GENERATOR_VERSION} generate \
  -g python \
  -i "local/${SWAGGER_CODEGEN_FILE}" \
  -c "local/${SWAGGER_CODEGEN_CONF}" \
  -o "local/${API_OUTPUT_PATH}" \
  -p=packageVersion="${API_VERSION}" \
  --global-property apis,apiTests=false \
  --global-property models,modelTests=false,modelDocs=false
