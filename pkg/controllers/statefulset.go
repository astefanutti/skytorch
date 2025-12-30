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
	"k8s.io/apimachinery/pkg/api/resource"
	appsv1apply "k8s.io/client-go/applyconfigurations/apps/v1"
	corev1apply "k8s.io/client-go/applyconfigurations/core/v1"
	metav1apply "k8s.io/client-go/applyconfigurations/meta/v1"
	"k8s.io/utils/ptr"

	"github.com/astefanutti/kpu/pkg/apis/kpu/v1alpha1"
)

func statefulSetApplyConfiguration(compute *v1alpha1.Compute) *appsv1apply.StatefulSetApplyConfiguration {
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
	for k, v := range compute.Spec.Annotations {
		annotations[k] = v
	}

	// Default image if not specified
	image := "kpu-torch-server:latest"
	if compute.Spec.Image != nil && *compute.Spec.Image != "" {
		image = *compute.Spec.Image
	}

	// Build container environment variables
	env := []*corev1apply.EnvVarApplyConfiguration{
		corev1apply.EnvVar().WithName("KPU_PORT").WithValue("50051"),
		corev1apply.EnvVar().WithName("KPU_HOST").WithValue("[::]"),
		corev1apply.EnvVar().WithName("KPU_LOG_LEVEL").WithValue("INFO"),
	}

	// Append user-provided env vars
	for _, e := range compute.Spec.Env {
		envVar := corev1apply.EnvVar().WithName(e.Name)
		if e.Value != "" {
			envVar.WithValue(e.Value)
		}
		if e.ValueFrom != nil {
			envVar.WithValueFrom(corev1apply.EnvVarSource())
			// Handle ValueFrom fields as needed
		}
		env = append(env, envVar)
	}

	// Build resource requirements
	resourceRequirements := corev1apply.ResourceRequirements()
	if len(compute.Spec.Resources) > 0 {
		// Use resources from Compute spec
		resourceRequirements.
			WithRequests(compute.Spec.Resources).
			WithLimits(compute.Spec.Resources)
	} else {
		// Use default resources
		resourceRequirements.
			WithRequests(corev1.ResourceList{
				corev1.ResourceCPU:    resource.MustParse("100m"),
				corev1.ResourceMemory: resource.MustParse("256Mi"),
			}).
			WithLimits(corev1.ResourceList{
				corev1.ResourceCPU:    resource.MustParse("1"),
				corev1.ResourceMemory: resource.MustParse("2Gi"),
			})
	}

	// Build container
	container := corev1apply.Container().
		WithName("server").
		WithImage(image).
		WithPorts(
			corev1apply.ContainerPort().
				WithName("grpc").
				WithContainerPort(50051).
				WithProtocol(corev1.ProtocolTCP),
		).
		WithEnv(env...).
		WithResources(resourceRequirements).
		WithReadinessProbe(
			corev1apply.Probe().
				WithGRPC(
					corev1apply.GRPCAction().
						WithPort(50051).
						// TODO: Make it dependent of the server compute type
						WithService("kpu.torch.Service"),
				).
				WithInitialDelaySeconds(2).
				WithPeriodSeconds(2).
				WithTimeoutSeconds(5).
				WithSuccessThreshold(1).
				WithFailureThreshold(3),
		)

	// Add command if specified
	if len(compute.Spec.Command) > 0 {
		container.WithCommand(compute.Spec.Command...)
	}

	// Add args if specified
	if len(compute.Spec.Args) > 0 {
		container.WithArgs(compute.Spec.Args...)
	}

	replicas := int32(1)
	if ptr.Deref(compute.Spec.Suspend, false) {
		replicas = 0
	}

	// Build StatefulSet apply configuration
	statefulSet := appsv1apply.StatefulSet(compute.Name, compute.Namespace).
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
			appsv1apply.StatefulSetSpec().
				WithReplicas(replicas).
				WithServiceName(compute.Name).
				WithSelector(
					metav1apply.LabelSelector().
						WithMatchLabels(map[string]string{
							"app.kubernetes.io/name":     "kpu-torch-server",
							"app.kubernetes.io/instance": compute.Name,
						}),
				).
				WithTemplate(
					corev1apply.PodTemplateSpec().
						WithLabels(labels).
						WithAnnotations(annotations).
						WithSpec(
							corev1apply.PodSpec().
								WithContainers(container),
						),
				),
		)

	return statefulSet
}
