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
	"context"
	"errors"

	"github.com/go-logr/logr"
	"k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/tools/record"
	"k8s.io/klog/v2"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/builder"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller"
	"sigs.k8s.io/controller-runtime/pkg/event"
	"sigs.k8s.io/controller-runtime/pkg/handler"
	"sigs.k8s.io/controller-runtime/pkg/predicate"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"
	"sigs.k8s.io/controller-runtime/pkg/source"

	"github.com/astefanutti/kpu/pkg/apis/kpu/v1alpha1"
	"github.com/astefanutti/kpu/pkg/util/constants"
)

type ComputeReconciler struct {
	log      logr.Logger
	client   client.Client
	recorder record.EventRecorder
}

var _ reconcile.Reconciler = (*ComputeReconciler)(nil)
var _ predicate.TypedPredicate[*v1alpha1.Compute] = (*ComputeReconciler)(nil)

func NewComputeReconciler(client client.Client, recorder record.EventRecorder) *ComputeReconciler {
	return &ComputeReconciler{
		log:      ctrl.Log.WithName("kpu-compute-controller"),
		client:   client,
		recorder: recorder,
	}
}

// +kubebuilder:rbac:groups=compute.kpu.dev,resources=computes,verbs=get;list;watch;update;patch
// +kubebuilder:rbac:groups=compute.kpu.dev,resources=computes/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=compute.kpu.dev,resources=computes/finalizers,verbs=get;update;patch

func (r *ComputeReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	var compute v1alpha1.Compute
	if err := r.client.Get(ctx, req.NamespacedName, &compute); err != nil {
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}
	log := ctrl.LoggerFrom(ctx).WithValues("compute", klog.KObj(&compute))
	ctx = ctrl.LoggerInto(ctx, log)
	log.V(2).Info("Reconciling Compute")

	prevCompute := compute.DeepCopy()
	var err error

	setSuspendedCondition(&compute)

	if statusErr := setComputeStatus(ctx, &compute); statusErr != nil {
		err = errors.Join(err, statusErr)
	}

	if !equality.Semantic.DeepEqual(&compute.Status, prevCompute.Status) {
		// TODO(astefanutti): Consider using SSA once controller-runtime client has SSA support
		// for sub-resources. See: https://github.com/kubernetes-sigs/controller-runtime/issues/3183
		return ctrl.Result{}, errors.Join(err, r.client.Status().Patch(ctx, &compute, client.MergeFrom(prevCompute)))
	}
	return ctrl.Result{}, err
}

func setSuspendedCondition(compute *v1alpha1.Compute) {
	var newCond metav1.Condition
	switch {
	case ptr.Deref(compute.Spec.Suspend, false):
		newCond = metav1.Condition{
			Type:    v1alpha1.ComputeSuspended,
			Status:  metav1.ConditionTrue,
			Message: constants.ComputeSuspendedMessage,
			Reason:  v1alpha1.ComputeSuspendedReason,
		}
	case meta.IsStatusConditionTrue(compute.Status.Conditions, v1alpha1.ComputeSuspended):
		newCond = metav1.Condition{
			Type:    v1alpha1.ComputeSuspended,
			Status:  metav1.ConditionFalse,
			Message: constants.ComputeResumedMessage,
			Reason:  v1alpha1.ComputeResumedReason,
		}
	default:
		return
	}
	meta.SetStatusCondition(&compute.Status.Conditions, newCond)
}

func setComputeStatus(ctx context.Context, compute *v1alpha1.Compute) error {
	return nil
}

func (r *ComputeReconciler) Create(e event.TypedCreateEvent[*v1alpha1.Compute]) bool {
	r.log.WithValues("compute", klog.KObj(e.Object)).Info("Compute create event")
	return true
}

func (r *ComputeReconciler) Delete(e event.TypedDeleteEvent[*v1alpha1.Compute]) bool {
	r.log.WithValues("compute", klog.KObj(e.Object)).Info("Compute delete event")
	return true
}

func (r *ComputeReconciler) Update(e event.TypedUpdateEvent[*v1alpha1.Compute]) bool {
	r.log.WithValues("compute", klog.KObj(e.ObjectNew)).Info("Compute update event")
	return true
}

func (r *ComputeReconciler) Generic(e event.TypedGenericEvent[*v1alpha1.Compute]) bool {
	r.log.WithValues("compute", klog.KObj(e.Object)).Info("Compute generic event")
	return true
}

func (r *ComputeReconciler) SetupWithManager(mgr ctrl.Manager, options controller.Options) error {
	b := builder.TypedControllerManagedBy[reconcile.Request](mgr).
		Named("compute-controller").
		WithOptions(options).
		WatchesRawSource(source.TypedKind(
			mgr.GetCache(),
			&v1alpha1.Compute{},
			&handler.TypedEnqueueRequestForObject[*v1alpha1.Compute]{},
			r,
		))
	return b.Complete(r)
}
