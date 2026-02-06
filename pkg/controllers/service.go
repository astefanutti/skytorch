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
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	corev1apply "k8s.io/client-go/applyconfigurations/core/v1"
	metav1apply "k8s.io/client-go/applyconfigurations/meta/v1"

	"github.com/astefanutti/skytorch/pkg/apis/compute/v1alpha1"
)

func serviceApplyConfiguration(compute *v1alpha1.Compute) *corev1apply.ServiceApplyConfiguration {
	labels := map[string]string{
		"app.kubernetes.io/name":       "skytorch-server",
		"app.kubernetes.io/instance":   compute.Name,
		"app.kubernetes.io/component":  "compute",
		"app.kubernetes.io/managed-by": "skytorch-operator",
		"app.kubernetes.io/part-of":    "skytorch",
	}

	// Merge user-provided labels
	for k, v := range compute.Spec.Labels {
		labels[k] = v
	}

	annotations := make(map[string]string)
	for k, v := range compute.Spec.Annotations {
		annotations[k] = v
	}

	service := corev1apply.Service(compute.Name, compute.Namespace).
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
			corev1apply.ServiceSpec().
				WithSelector(map[string]string{
					"app.kubernetes.io/name":     "skytorch-server",
					"app.kubernetes.io/instance": compute.Name,
				}).
				WithPorts(
					corev1apply.ServicePort().
						WithName("grpc").
						WithProtocol(corev1.ProtocolTCP).
						WithAppProtocol("kubernetes.io/h2c").
						WithPort(50051).
						WithTargetPort(intstr.FromString("grpc")),
				).
				WithType(corev1.ServiceTypeClusterIP),
		)

	return service
}
