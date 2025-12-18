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

package v1alpha1

import (
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

const (
	// ComputeKind is the Kind name for the Compute.
	ComputeKind string = "Compute"
)

// +genclient
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:storageversion
// +kubebuilder:printcolumn:name="State",type=string,JSONPath=`.status.conditions[?(@.status=="True")].type`
// +kubebuilder:printcolumn:name="Age",type=date,JSONPath=`.metadata.creationTimestamp`
// +kubebuilder:validation:XValidation:rule="self.metadata.name.matches('^[a-z]([-a-z0-9]*[a-z0-9])?$')", message="metadata.name must match RFC 1035 DNS label format"
// +kubebuilder:validation:XValidation:rule="size(self.metadata.name) <= 63", message="metadata.name must be no more than 63 characters"

// Compute represents the configuration of a compute environment.
type Compute struct {
	metav1.TypeMeta `json:",inline"`

	// metadata of the Compute.
	// +optional
	metav1.ObjectMeta `json:"metadata,omitempty"`

	// spec of the Compute.
	// +optional
	Spec ComputeSpec `json:"spec,omitzero"`

	// status of the Compute.
	// +optional
	Status ComputeStatus `json:"status,omitzero"`
}

const (
	// ComputeSuspended means that Compute is suspended.
	ComputeSuspended string = "Suspended"

	// ComputeReady means that the Compute is ready.
	ComputeReady string = "Ready"
)

const (
	// ComputeSuspendedReason is the "Suspended" condition reason
	// when the Compute is suspended.
	ComputeSuspendedReason string = "Suspended"

	// ComputeResumedReason is the "Suspended" condition reason
	// when the Compute has resume from suspension.
	ComputeResumedReason string = "Resumed"
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +resource:path=computes
// +kubebuilder:object:root=true

// ComputeList is a collection of computes.
type ComputeList struct {
	metav1.TypeMeta `json:",inline"`

	// Standard list metadata.
	metav1.ListMeta `json:"metadata,omitempty"`

	// List of Computes.
	Items []Compute `json:"items"`
}

// ComputeSpec represents the specification of the Compute.
// +kubebuilder:validation:MinProperties=1
type ComputeSpec struct {
	// labels to apply to the Compute resources.
	// +optional
	Labels map[string]string `json:"labels,omitempty"`

	// annotations to apply to the Compute resources.
	// +optional
	Annotations map[string]string `json:"annotations,omitempty"`

	// suspend defines whether to suspend the running Compute.
	// Defaults to false.
	// +kubebuilder:default=false
	// +optional
	Suspend *bool `json:"suspend,omitempty"`

	// image is the container image for the Compute runtime.
	// +optional
	Image *string `json:"image,omitempty"`

	// command for the entrypoint of the Compute runtime.
	// +listType=atomic
	// +optional
	Command []string `json:"command,omitempty"`

	// args for the entrypoint for the Compute runtime.
	// +listType=atomic
	// +optional
	Args []string `json:"args,omitempty"`

	// env is the list of environment variables to set in the Compute runtime.
	// +listType=map
	// +listMapKey=name
	// +optional
	Env []corev1.EnvVar `json:"env,omitempty"`
}

// PodTemplateSpecOverride represents the spec overrides for Pod template.
type PodTemplateSpecOverride struct {
	// serviceAccountName overrides the service account.
	// +optional
	ServiceAccountName *string `json:"serviceAccountName,omitempty"`

	// nodeSelector overrides the node selector.
	// +optional
	NodeSelector map[string]string `json:"nodeSelector,omitempty"`

	// affinity overrides for the affinity.
	// +optional
	Affinity *corev1.Affinity `json:"affinity,omitempty"`

	// tolerations overrides the tolerations.
	// +listType=atomic
	// +optional
	Tolerations []corev1.Toleration `json:"tolerations,omitempty"`

	// volumes overrides the volumes.
	// +listType=map
	// +listMapKey=name
	// +optional
	Volumes []corev1.Volume `json:"volumes,omitempty"`

	// initContainers overrides the init containers.
	// +listType=map
	// +listMapKey=name
	// +optional
	InitContainers []ContainerOverride `json:"initContainers,omitempty"`

	// containers overrides for the containers.
	// +listType=map
	// +listMapKey=name
	// +optional
	Containers []ContainerOverride `json:"containers,omitempty"`

	// schedulingGates overrides the scheduling gates.
	// More info: https://kubernetes.io/docs/concepts/scheduling-eviction/pod-scheduling-readiness/
	// +listType=map
	// +listMapKey=name
	// +optional
	SchedulingGates []corev1.PodSchedulingGate `json:"schedulingGates,omitempty"`

	// imagePullSecrets overrides the image pull secrets.
	// +listType=map
	// +listMapKey=name
	// +optional
	ImagePullSecrets []corev1.LocalObjectReference `json:"imagePullSecrets,omitempty"`
}

// ContainerOverride represents parameters that can be overridden using PodSpecOverrides.
type ContainerOverride struct {
	// name for the container.
	// +kubebuilder:validation:MinLength=1
	// +required
	Name string `json:"name,omitempty"`

	// env is the list of environment variables to set in the container.
	// +listType=map
	// +listMapKey=name
	// +optional
	Env []corev1.EnvVar `json:"env,omitempty"`

	// volumeMounts are the volumes to mount into the container's filesystem.
	// +listType=map
	// +listMapKey=name
	// +optional
	VolumeMounts []corev1.VolumeMount `json:"volumeMounts,omitempty"`
}

// ComputeStatus represents the current status of the Compute.
// +kubebuilder:validation:MinProperties=1
type ComputeStatus struct {
	// conditions of the Compute.
	//
	// +optional
	// +listType=map
	// +listMapKey=type
	Conditions []metav1.Condition `json:"conditions,omitempty"`
}

func init() {
	SchemeBuilder.Register(&Compute{}, &ComputeList{})
}
