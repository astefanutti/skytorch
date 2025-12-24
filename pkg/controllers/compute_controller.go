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
	"fmt"

	"github.com/go-logr/logr"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
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
	gatewayv1 "sigs.k8s.io/gateway-api/apis/v1"

	"github.com/astefanutti/kpu/pkg/apis/kpu/v1alpha1"
	"github.com/astefanutti/kpu/pkg/util/constants"
)

const (
	fieldManager = "kpu-compute-controller"
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

// +kubebuilder:rbac:groups="",resources=services,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=apps,resources=statefulsets,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=gateway.networking.k8s.io,resources=grpcroutes,verbs=get;list;watch;create;update;patch;delete
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

	if compute.DeletionTimestamp == nil {
		// Reconcile Service
		if serviceErr := r.reconcileService(ctx, &compute); serviceErr != nil {
			log.Error(serviceErr, "Failed to reconcile Service")
			err = errors.Join(err, serviceErr)
		}

		// Reconcile GRPCRoute
		if grpcRouteErr := r.reconcileGRPCRoute(ctx, &compute); grpcRouteErr != nil {
			log.Error(grpcRouteErr, "Failed to reconcile GRPCRoute")
			err = errors.Join(err, grpcRouteErr)
		}

		// Reconcile StatefulSet
		if statefulSetErr := r.reconcileStatefulSet(ctx, &compute); statefulSetErr != nil {
			log.Error(statefulSetErr, "Failed to reconcile StatefulSet")
			err = errors.Join(err, statefulSetErr)
		}
	}

	r.setSuspendedCondition(&compute)

	if readyErr := r.setReadyCondition(ctx, &compute); readyErr != nil {
		log.Error(readyErr, "Failed to set Ready condition")
		err = errors.Join(err, readyErr)
	}

	if !equality.Semantic.DeepEqual(&compute.Status, prevCompute.Status) {
		// TODO(astefanutti): Consider using SSA once controller-runtime client has SSA support
		// for sub-resources. See: https://github.com/kubernetes-sigs/controller-runtime/issues/3183
		return ctrl.Result{}, errors.Join(err, client.IgnoreNotFound(
			r.client.Status().Patch(ctx, &compute, client.MergeFrom(prevCompute))),
		)
	}

	return ctrl.Result{}, err
}

func (r *ComputeReconciler) reconcileStatefulSet(ctx context.Context, compute *v1alpha1.Compute) error {
	log := ctrl.LoggerFrom(ctx)

	statefulSet := statefulSetApplyConfiguration(compute)

	if err := r.client.Apply(ctx, statefulSet, client.FieldOwner(fieldManager), client.ForceOwnership); err != nil {
		return fmt.Errorf("failed to apply StatefulSet: %w", err)
	}

	log.V(2).Info("StatefulSet reconciled", "statefulset", klog.KRef(compute.Namespace, compute.Name))
	return nil
}

func (r *ComputeReconciler) reconcileService(ctx context.Context, compute *v1alpha1.Compute) error {
	log := ctrl.LoggerFrom(ctx)

	serviceApply := serviceApplyConfiguration(compute)

	if err := r.client.Apply(ctx, serviceApply, client.FieldOwner(fieldManager), client.ForceOwnership); err != nil {
		return fmt.Errorf("failed to apply Service: %w", err)
	}

	log.V(2).Info("Service reconciled", "service", klog.KRef(compute.Namespace, compute.Name))
	return nil
}

func (r *ComputeReconciler) reconcileGRPCRoute(ctx context.Context, compute *v1alpha1.Compute) error {
	log := ctrl.LoggerFrom(ctx)

	grpcRoute := grpcRouteApplyConfiguration(compute)

	if err := r.client.Apply(ctx, grpcRoute, client.FieldOwner(fieldManager), client.ForceOwnership); err != nil {
		return fmt.Errorf("failed to apply GRPCRoute: %w", err)
	}

	log.V(2).Info("GRPCRoute reconciled", "grpcroute", klog.KRef(compute.Namespace, compute.Name))
	return nil
}

func (r *ComputeReconciler) setSuspendedCondition(compute *v1alpha1.Compute) {
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

func (r *ComputeReconciler) setReadyCondition(ctx context.Context, compute *v1alpha1.Compute) error {
	// Get the StatefulSet to read its current status
	var statefulSet appsv1.StatefulSet
	if err := r.client.Get(ctx, client.ObjectKeyFromObject(compute), &statefulSet); err != nil {
		if client.IgnoreNotFound(err) == nil {
			// StatefulSet not found, set Ready to False
			meta.SetStatusCondition(&compute.Status.Conditions, metav1.Condition{
				Type:    v1alpha1.ComputeReady,
				Status:  metav1.ConditionFalse,
				Reason:  "StatefulSetNotAvailable",
				Message: "StatefulSet is not available",
			})
			return nil
		}
		return fmt.Errorf("failed to get StatefulSet: %w", err)
	}

	// Check if StatefulSet is ready
	replicas := ptr.Deref(statefulSet.Spec.Replicas, 0)
	readyReplicas := statefulSet.Status.ReadyReplicas

	if readyReplicas == replicas && replicas > 0 {
		// StatefulSet is ready
		meta.SetStatusCondition(&compute.Status.Conditions, metav1.Condition{
			Type:    v1alpha1.ComputeReady,
			Status:  metav1.ConditionTrue,
			Reason:  "StatefulSetReady",
			Message: fmt.Sprintf("All %d replicas are ready", replicas),
		})
	} else {
		// StatefulSet is not ready
		meta.SetStatusCondition(&compute.Status.Conditions, metav1.Condition{
			Type:    v1alpha1.ComputeReady,
			Status:  metav1.ConditionFalse,
			Reason:  "StatefulSetNotReady",
			Message: fmt.Sprintf("StatefulSet has %d/%d ready replicas", readyReplicas, replicas),
		})
	}
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
		)).
		WatchesRawSource(source.TypedKind(
			mgr.GetCache(),
			&appsv1.StatefulSet{},
			handler.TypedEnqueueRequestForOwner[*appsv1.StatefulSet](
				mgr.GetScheme(),
				mgr.GetRESTMapper(),
				&v1alpha1.Compute{},
				handler.OnlyControllerOwner(),
			),
		)).
		WatchesRawSource(source.TypedKind(
			mgr.GetCache(),
			&corev1.Service{},
			handler.TypedEnqueueRequestForOwner[*corev1.Service](
				mgr.GetScheme(),
				mgr.GetRESTMapper(),
				&v1alpha1.Compute{},
				handler.OnlyControllerOwner(),
			),
		)).
		WatchesRawSource(source.TypedKind(
			mgr.GetCache(),
			&gatewayv1.GRPCRoute{},
			handler.TypedEnqueueRequestForOwner[*gatewayv1.GRPCRoute](
				mgr.GetScheme(),
				mgr.GetRESTMapper(),
				&v1alpha1.Compute{},
				handler.OnlyControllerOwner(),
			),
		))
	return b.Complete(r)
}
