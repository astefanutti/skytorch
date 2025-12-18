# Get the currently used golang install path (in GOPATH/bin, unless GOBIN is set)
ifeq (,$(shell go env GOBIN))
GOBIN=$(shell go env GOPATH)/bin
else
GOBIN=$(shell go env GOBIN)
endif

GO_CMD ?= go

# Setting SHELL to bash allows bash commands to be executed by recipes.
# This is a requirement for 'setup-envtest.sh' in the test target.
# Options are set to exit when a recipe line exits non-zero or a piped command fails.
SHELL = /usr/bin/env bash -o pipefail
.SHELLFLAGS = -ec

PROJECT_DIR := $(shell dirname $(abspath $(lastword $(MAKEFILE_LIST))))
# Location to install tool binaries
BIN_DIR ?= $(PROJECT_DIR)/bin
TOOLS_DIR := $(PROJECT_DIR)/hack

# Tool versions
CONTROLLER_GEN_VERSION ?= $(shell cd $(TOOLS_DIR); $(GO_CMD) list -m -f '{{.Version}}' sigs.k8s.io/controller-tools)
ENVTEST_VERSION ?= release-0.22
GINKGO_VERSION ?= $(shell cd $(TOOLS_DIR); $(GO_CMD) list -m -f '{{.Version}}' github.com/onsi/ginkgo/v2)
GOLANGCI_LINT_VERSION ?= $(shell cd $(TOOLS_DIR); $(GO_CMD) list -m -f '{{.Version}}' github.com/golangci/golangci-lint/v2)
K8S_VERSION ?= 1.34.0
KUSTOMIZE_VERSION ?= $(shell cd $(TOOLS_DIR); $(GO_CMD) list -m -f '{{.Version}}' sigs.k8s.io/kustomize/kustomize/v5)

# Tool binaries
CONTROLLER_GEN ?= $(BIN_DIR)/controller-gen
ENVTEST ?= $(BIN_DIR)/setup-envtest
GINKGO ?= $(BIN_DIR)/ginkgo
GOLANGCI_LINT = $(BIN_DIR)/golangci-lint
GOLANGCI_LINT_KAL ?= $(BIN_DIR)/golangci-lint-kube-api-linter
KUSTOMIZE = $(BIN_DIR)/kustomize

# Container runtime (docker or podman)
CONTAINER_RUNTIME ?= $(shell hack/container-runtime.sh)

ARCH ?= $(shell $(GO_CMD) env GOARCH)
VERSION ?= v0.0.0-dev
IMAGE_BASE ?= ghcr.io/astefanutti
OPERATOR_IMG ?= ${IMAGE_BASE}/kpu-operator:${VERSION}

##@ General

# The help target prints out all targets with their descriptions organized
# beneath their categories. The categories are represented by '##@' and the
# target descriptions by '##'. The awk commands is responsible for reading the
# entire set of makefiles included in this invocation, looking for lines of the
# file as xyz: ## something, and then pretty-format the target and help. Then,
# if there's a line with ##@ something, that gets pretty-printed as a category.
# More info on the usage of ANSI control characters for terminal formatting:
# https://en.wikipedia.org/wiki/ANSI_escape_code#SGR_parameters
# More info on the awk command:
# http://linuxcommand.org/lc3_adv_awk.php

help: ## Display this help.
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Development

# Instructions for code generation.

.PHONY: manifests
manifests: controller-gen ## Generate manifests.
	$(CONTROLLER_GEN) "crd:generateEmbeddedObjectMeta=true" rbac:roleName=kpu-operator webhook \
		paths="./cmd/...;./pkg/apis/kpu/v1alpha1/...;./pkg/controllers/...;./pkg/webhooks/...;./pkg/util/cert/..." \
		output:crd:artifacts:config=config/base/crds \
		output:rbac:artifacts:config=config/base/rbac \
		output:webhook:artifacts:config=config/base/webhook

.PHONY: generate
generate: tools-download manifests ## Generate APIs.
	$(CONTROLLER_GEN) object:headerFile="hack/boilerplate/boilerplate.go.txt" paths="./pkg/apis/..."
	TOOLS_DIR=${TOOLS_DIR} GO_CMD=${GO_CMD} hack/update-codegen.sh
	CONTAINER_RUNTIME=$(CONTAINER_RUNTIME) VERSION=$(VERSION) hack/gen-python-api.sh

.PHONY: tools-download
tools-download: ## Run go mod download for tools.
	cd ${TOOLS_DIR} && $(GO_CMD) mod download

# Instructions for code formatting.

.PHONY: fmt
fmt: ## Run go fmt against the code.
	$(GO_CMD) fmt ./...

.PHONY: vet
vet: ## Run go vet against the code.
	$(GO_CMD) vet ./...

.PHONY: lint
lint: golangci-lint golangci-lint-kal ## Run golangci-lint to verify Go files.
	$(GOLANGCI_LINT_KAL) run -v --timeout 5m ./...

# Instructions to build components.

.PHONY: build-operator-binary
build-operator-binary: fmt vet ## Build operator binary.
	GOOS=linux GOARCH=$(ARCH) $(GO_CMD) build -o cmd/operator/manager-$(ARCH) cmd/operator/main.go

.PHONY: build-operator-image
build-operator-image: build-operator-binary ## Build operator image.
	$(CONTAINER_RUNTIME) build --platform linux/$(ARCH) -f cmd/operator/Containerfile -t ${OPERATOR_IMG} cmd/operator

.PHONY: push-operator-image
push-operator-image: ## Push operator image.
	$(CONTAINER_RUNTIME) push ${OPERATOR_IMG}

# Instructions to run tests.

.PHONY: test
test: ## Run Go unit test.
	$(GO_CMD) test $(shell $(GO_CMD) list ./... | grep -Ev '/(test|cmd|hack|pkg/apis|pkg/client|pkg/util/testing)') -coverprofile cover.out

.PHONY: test-integration
test-integration: ginkgo envtest ## Run Go integration test.
	KUBEBUILDER_ASSETS="$(shell $(ENVTEST) use $(K8S_VERSION) -p path)" $(GINKGO) -v ./test/integration/...

.PHONY: deploy-operator
deploy-operator: kustomize ## Deploy operator.
	cd config/${ENV} && $(KUSTOMIZE) edit set image ghcr.io/astefanutti/kpu-operator=${OPERATOR_IMG}
	$(KUSTOMIZE) build config/${ENV} | kubectl apply --server-side -f -

# Instructions to download tools for development.

.PHONY: controller-gen
controller-gen: ## Download the controller-gen binary if necessary.
	GOBIN=$(BIN_DIR) $(GO_CMD) install sigs.k8s.io/controller-tools/cmd/controller-gen@$(CONTROLLER_GEN_VERSION)

.PHONY: envtest
envtest: ## Download the setup-envtest binary if necessary.
	GOBIN=$(BIN_DIR) $(GO_CMD) install sigs.k8s.io/controller-runtime/tools/setup-envtest@$(ENVTEST_VERSION)

.PHONY: ginkgo
ginkgo: ## Download the ginkgo binary if necessary.
	GOBIN=$(BIN_DIR) $(GO_CMD) install github.com/onsi/ginkgo/v2/ginkgo@$(GINKGO_VERSION)

.PHONY: golangci-lint
golangci-lint: ## Download golangci-lint if necessary.
	@GOBIN=$(BIN_DIR) $(GO_CMD) install github.com/golangci/golangci-lint/v2/cmd/golangci-lint@$(GOLANGCI_LINT_VERSION)

.PHONY: golangci-lint-kal
golangci-lint-kal: ## Build golangci-lint-kal from custom configuration.
	cd $(PROJECT_DIR)/hack/golangci-kal; $(GOLANGCI_LINT) custom -v; mv bin/golangci-lint-kube-api-linter $(BIN_DIR)/

.PHONY: kustomize
kustomize: ## Download kustomize if necessary.
	GOBIN=$(BIN_DIR) $(GO_CMD) install sigs.k8s.io/kustomize/kustomize/v5@$(KUSTOMIZE_VERSION)
