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
	corev1apply "k8s.io/client-go/applyconfigurations/core/v1"

	"github.com/astefanutti/skytorch/pkg/apis/compute/v1alpha1"
)

func pvcApplyConfigurations(compute *v1alpha1.Compute) []*corev1apply.PersistentVolumeClaimApplyConfiguration {
	var pvcs []*corev1apply.PersistentVolumeClaimApplyConfiguration

	for _, pvc := range compute.Spec.VolumeClaimTemplates {
		labels := map[string]string{
			"app.kubernetes.io/name":       "skytorch-server",
			"app.kubernetes.io/component":  "compute",
			"app.kubernetes.io/managed-by": "skytorch-operator",
			"app.kubernetes.io/part-of":    "skytorch",
		}

		pvcSpec := corev1apply.PersistentVolumeClaimSpec().
			WithAccessModes(pvc.Spec.AccessModes...).
			WithResources(
				corev1apply.VolumeResourceRequirements().
					WithRequests(pvc.Spec.Resources.Requests),
			)
		if pvc.Spec.StorageClassName != nil {
			pvcSpec.WithStorageClassName(*pvc.Spec.StorageClassName)
		}

		pvcApply := corev1apply.PersistentVolumeClaim(pvc.Name, compute.Namespace).
			WithLabels(labels).
			WithSpec(pvcSpec)

		pvcs = append(pvcs, pvcApply)
	}

	return pvcs
}
