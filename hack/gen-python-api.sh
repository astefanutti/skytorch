#!/usr/bin/env bash

set -o errexit
set -o nounset

# Source container runtime utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/container-runtime.sh"

# Setup container runtime
setup_container_runtime

API_VERSION="${VERSION}"
API_OUTPUT_PATH="kpu"
PKG_ROOT="${API_OUTPUT_PATH}/client"

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
  --global-property models,modelTests=false,modelDocs=false,supportingFiles=__init__.py

echo "Removing unused files for the Python API"
git clean -f ${API_OUTPUT_PATH}/.openapi-generator
git clean -f ${API_OUTPUT_PATH}/.github
git clean -f ${API_OUTPUT_PATH}/test
git clean -f ${PKG_ROOT}/api

# Revert manually created files.
#git checkout ${PKG_ROOT}/__init__.py

# Manually modify the SDK version in the __init__.py file.
#if [[ $(uname) == "Darwin" ]]; then
#  sed -i '' -e "s/__version__.*/__version__ = \"${API_VERSION}\"/" ${PKG_ROOT}/__init__.py
#else
#  sed -i -e "s/__version__.*/__version__ = \"${API_VERSION}\"/" ${PKG_ROOT}/__init__.py
#fi
