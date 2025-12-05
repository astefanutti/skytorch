# Get the currently used golang install path (in GOPATH/bin, unless GOBIN is set)
ifeq (,$(shell go env GOBIN))
GOBIN=$(shell go env GOPATH)/bin
else
GOBIN=$(shell go env GOBIN)
endif

# Setting SHELL to bash allows bash commands to be executed by recipes.
# This is a requirement for 'setup-envtest.sh' in the test target.
# Options are set to exit when a recipe line exits non-zero or a piped command fails.
SHELL = /usr/bin/env bash -o pipefail
.SHELLFLAGS = -ec

PROJECT_DIR := $(shell dirname $(abspath $(lastword $(MAKEFILE_LIST))))
# Location to install tool binaries
LOCALBIN ?= $(PROJECT_DIR)/bin

# Tool versions
K8S_VERSION ?= 1.34.0
GINKGO_VERSION ?= $(shell go list -m -f '{{.Version}}' github.com/onsi/ginkgo/v2)
ENVTEST_VERSION ?= release-0.22
CONTROLLER_GEN_VERSION ?= v0.18.0
KIND_VERSION ?= $(shell go list -m -f '{{.Version}}' sigs.k8s.io/kind)

# Container runtime (docker or podman)
CONTAINER_RUNTIME ?=

# Tool binaries
GINKGO ?= $(LOCALBIN)/ginkgo
ENVTEST ?= $(LOCALBIN)/setup-envtest
CONTROLLER_GEN ?= $(LOCALBIN)/controller-gen
GOLANGCI_LINT = $(LOCALBIN)/golangci-lint
GOLANGCI_LINT_KAL ?= $(LOCALBIN)/golangci-lint-kube-api-linter

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
generate: go-mod-download manifests ## Generate APIs.
	$(CONTROLLER_GEN) object:headerFile="hack/boilerplate/boilerplate.go.txt" paths="./pkg/apis/..."
	hack/update-codegen.sh

.PHONY: go-mod-download
go-mod-download: ## Run go mod download to download modules.
	go mod download

# Instructions for code formatting.

.PHONY: fmt
fmt: ## Run go fmt against the code.
	go fmt ./...

.PHONY: vet
vet: ## Run go vet against the code.
	go vet ./...

.PHONY: lint
lint: golangci-lint golangci-lint-kal ## Run golangci-lint to verify Go files.
	golangci-lint run --timeout 5m --go 1.24 ./...
	$(GOLANGCI_LINT_KAL) run -v --config $(PROJECT_DIR)/.golangci-kal.yml

# Instructions to build components.

.PHONY: build-operator
build-operator: fmt vet ## Build operator binary.
	go build -o $(LOCALBIN)/operator-manager cmd/operator/main.go

# Instructions to run tests.

.PHONY: test
test: ## Run Go unit test.
	go test $(shell go list ./... | grep -Ev '/(test|cmd|hack|pkg/apis|pkg/client|pkg/util/testing)') -coverprofile cover.out

.PHONY: test-integration
test-integration: ginkgo envtest ## Run Go integration test.
	KUBEBUILDER_ASSETS="$(shell $(ENVTEST) use $(K8S_VERSION) -p path)" $(GINKGO) -v ./test/integration/...

# Instructions to download tools for development.

.PHONY: ginkgo
ginkgo: ## Download the ginkgo binary if required.
	GOBIN=$(LOCALBIN) go install github.com/onsi/ginkgo/v2/ginkgo@$(GINKGO_VERSION)

.PHONY: envtest
envtest: ## Download the setup-envtest binary if required.
	GOBIN=$(LOCALBIN) go install sigs.k8s.io/controller-runtime/tools/setup-envtest@$(ENVTEST_VERSION)

.PHONY: controller-gen
controller-gen: ## Download the controller-gen binary if required.
	GOBIN=$(LOCALBIN) go install sigs.k8s.io/controller-tools/cmd/controller-gen@$(CONTROLLER_GEN_VERSION)

.PHONY: kind
kind: ## Download Kind binary if required.
	GOBIN=$(LOCALBIN) go install sigs.k8s.io/kind@$(KIND_VERSION)

.PHONY: golangci-lint
golangci-lint: ## Download golangci-lint locally if necessary.
	@GOBIN=$(LOCALBIN) go install github.com/golangci/golangci-lint/v2/cmd/golangci-lint@v2.1.5

.PHONY: golangci-lint-kal
golangci-lint-kal: ## Build golangci-lint-kal from custom configuration.
	cd $(PROJECT_DIR)/hack/golangci-kal; $(GOLANGCI_LINT) custom; mv bin/golangci-lint-kube-api-linter $(LOCALBIN)/
