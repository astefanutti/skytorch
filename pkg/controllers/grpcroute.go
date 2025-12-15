/*
Copyright (c) 2025 Antonin Stefanutti <antonin.stefanutti@gmail.com>

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package controllers

import (
	metav1apply "k8s.io/client-go/applyconfigurations/meta/v1"
	gatewayv1 "sigs.k8s.io/gateway-api/apis/v1"
	gatewayv1apply "sigs.k8s.io/gateway-api/applyconfiguration/apis/v1"

	"github.com/astefanutti/kpu/pkg/apis/kpu/v1alpha1"
	"github.com/astefanutti/kpu/pkg/util/constants"
)

func grpcRouteApplyConfiguration(compute *v1alpha1.Compute) *gatewayv1apply.GRPCRouteApplyConfiguration {
	labels := map[string]string{
		"app.kubernetes.io/name":       "kpu-torch-server",
		"app.kubernetes.io/instance":   compute.Name,
		"app.kubernetes.io/component":  "compute",
		"app.kubernetes.io/managed-by": "kpu-operator",
		"app.kubernetes.io/part-of":    "kpu",
	}

	// Merge user-provided labels
	for k, v := range compute.Spec.Labels {
		labels[k] = v
	}

	annotations := make(map[string]string)
	// Merge user-provided annotations
	for k, v := range compute.Spec.Annotations {
		annotations[k] = v
	}

	grpcRoute := gatewayv1apply.GRPCRoute(compute.Name, compute.Namespace).
		WithLabels(labels).
		WithAnnotations(annotations).
		WithOwnerReferences(
			metav1apply.OwnerReference().
				WithAPIVersion(v1alpha1.SchemeGroupVersion.String()).
				WithKind(v1alpha1.ComputeKind).
				WithName(compute.Name).
				WithUID(compute.UID).
				WithController(true).
				WithBlockOwnerDeletion(true),
		).
		WithSpec(
			gatewayv1apply.GRPCRouteSpec().
				WithParentRefs(gatewayv1apply.ParentReference().
					WithKind("Gateway").
					WithNamespace("kpu-system").
					WithName("kpu-gateway"),
				).
				WithRules(
					gatewayv1apply.GRPCRouteRule().
						WithMatches(gatewayv1apply.GRPCRouteMatch().
							WithHeaders(
								gatewayv1apply.GRPCHeaderMatch().
									WithType(gatewayv1.GRPCHeaderMatchExact).
									WithName(constants.ComputeIdHeader).
									WithValue(compute.Namespace + "/" + compute.Name),
							),
						).
						WithBackendRefs(
							gatewayv1apply.GRPCBackendRef().
								WithName(gatewayv1.ObjectName(compute.Name)).
								WithPort(50051),
						),
				),
		)

	return grpcRoute
}
