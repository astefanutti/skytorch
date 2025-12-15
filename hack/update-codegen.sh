#!/usr/bin/env bash

# Copyright (c) 2025 Antonin Stefanutti <antonin.stefanutti@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This shell is used to auto generate some useful tools for k8s, such as clientset, lister, informer.

set -o errexit
set -o nounset
set -o pipefail

CURRENT_DIR=$(dirname "${BASH_SOURCE[0]}")
ROOT_DIR=$(realpath "${CURRENT_DIR}/..")
ROOT_PKG="github.com/astefanutti/kpu"

cd "$CURRENT_DIR/.."

# Get the code-generator binary.
CODEGEN_PKG=$(cd "${TOOLS_DIR}"; "${GO_CMD}" list -m -mod=readonly -f "{{.Dir}}" k8s.io/code-generator)
source "${CODEGEN_PKG}/kube_codegen.sh"
echo ">> Using ${CODEGEN_PKG}"

echo "Generating deepcopy and defaults"

kube::codegen::gen_helpers \
  --boilerplate "${ROOT_DIR}/hack/boilerplate/boilerplate.go.txt" \
  "${ROOT_DIR}/pkg/apis"

echo "Generating clients"

externals=(
  "k8s.io/api/core/v1.EnvVar:k8s.io/client-go/applyconfigurations/core/v1"
  "k8s.io/api/core/v1.EnvFromSource:k8s.io/client-go/applyconfigurations/core/v1"
  "k8s.io/api/core/v1.ResourceRequirements:k8s.io/client-go/applyconfigurations/core/v1"
  "k8s.io/api/core/v1.Toleration:k8s.io/client-go/applyconfigurations/core/v1"
  "k8s.io/api/core/v1.Volume:k8s.io/client-go/applyconfigurations/core/v1"
  "k8s.io/api/core/v1.VolumeMount:k8s.io/client-go/applyconfigurations/core/v1"
)

apply_config_externals="${externals[0]}"
for external in "${externals[@]:1}"; do
  apply_config_externals="${apply_config_externals},${external}"
done

kube::codegen::gen_client \
  --boilerplate "${ROOT_DIR}/hack/boilerplate/boilerplate.go.txt" \
  --output-dir "${ROOT_DIR}/pkg/client" \
  --output-pkg "${ROOT_PKG}/pkg/client" \
  --with-watch \
  --with-applyconfig \
  --applyconfig-externals "${apply_config_externals}" \
  "${ROOT_DIR}/pkg/apis"

echo "Generating OpenAPI specification"

# Get the kube-openapi binary to generate OpenAPI spec.
OPENAPI_PKG=$(go list -m -mod=readonly -f "{{.Dir}}" k8s.io/kube-openapi)
echo ">> Using ${OPENAPI_PKG}"

extra_packages=(
  "k8s.io/apimachinery/pkg/apis/meta/v1"
  "k8s.io/apimachinery/pkg/api/resource"
  "k8s.io/apimachinery/pkg/util/intstr"
  "k8s.io/api/core/v1"
)

gen_openapi_extra_pkgs="--extra-pkgs ${extra_packages[0]}"
for extra_pkg in "${extra_packages[@]:1}"; do
  gen_openapi_extra_pkgs="${gen_openapi_extra_pkgs} --extra-pkgs ${extra_pkg}"
done

echo "gen_openapi_extra" ${gen_openapi_extra_pkgs}

kube::codegen::gen_openapi \
  --boilerplate "${ROOT_DIR}/hack/boilerplate/boilerplate.go.txt" \
  --output-dir "${ROOT_DIR}/pkg/apis/kpu/v1alpha1" \
  --output-pkg "${ROOT_PKG}/pkg/apis/kpu/v1alpha1" \
  ${gen_openapi_extra_pkgs} \
  --update-report \
  "${ROOT_DIR}/pkg/apis/kpu/v1alpha1"

echo "Generate OpenAPI Swagger"
go run hack/swagger/main.go > hack/swagger/swagger.json
