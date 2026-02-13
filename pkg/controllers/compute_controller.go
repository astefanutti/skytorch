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
	apierrors "k8s.io/apimachinery/pkg/api/errors"
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

	"github.com/astefanutti/skytorch/pkg/apis/compute/v1alpha1"
	"github.com/astefanutti/skytorch/pkg/util/constants"
)

const (
	fieldManager = "skytorch-compute-controller"
)

type ComputeReconciler struct {
	log               logr.Logger
	client            client.Client
	recorder          record.EventRecorder
	operatorNamespace string
}

var _ reconcile.Reconciler = (*ComputeReconciler)(nil)
var _ predicate.TypedPredicate[*v1alpha1.Compute] = (*ComputeReconciler)(nil)

func NewComputeReconciler(client client.Client, recorder record.EventRecorder, operatorNamespace string) *ComputeReconciler {
	return &ComputeReconciler{
		log:               ctrl.Log.WithName("skytorch-compute-controller"),
		client:            client,
		recorder:          recorder,
		operatorNamespace: operatorNamespace,
	}
}

// +kubebuilder:rbac:groups="",resources=persistentvolumeclaims,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups="",resources=services,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=apps,resources=statefulsets,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=gateway.networking.k8s.io,resources=grpcroutes,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=gateway.networking.k8s.io,resources=gateways,verbs=get;list;watch
// +kubebuilder:rbac:groups=gateway.networking.k8s.io,resources=gateways/status,verbs=get
// +kubebuilder:rbac:groups=compute.skytorch.dev,resources=computes,verbs=get;list;watch;update;patch
// +kubebuilder:rbac:groups=compute.skytorch.dev,resources=computes/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=compute.skytorch.dev,resources=computes/finalizers,verbs=get;update;patch

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
		if grpcRouteErr := r.reconcileGRPCRoute(ctx, &compute, r.operatorNamespace); grpcRouteErr != nil {
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
		if statusErr := r.client.Status().Patch(ctx, &compute, client.MergeFrom(prevCompute)); apierrors.IsNotFound(statusErr) {
			// Requeue to make sure it goes over a full reconcile cycle
			return ctrl.Result{RequeueAfter: 0}, err
		} else if statusErr != nil {
			log.Error(statusErr, "Failed to patch status")
			err = errors.Join(err, statusErr)
		}
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

func (r *ComputeReconciler) reconcileGRPCRoute(ctx context.Context, compute *v1alpha1.Compute, operatorNamespace string) error {
	log := ctrl.LoggerFrom(ctx)

	grpcRoute := grpcRouteApplyConfiguration(compute, operatorNamespace)

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
	// Check StatefulSet status
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

	replicas := ptr.Deref(statefulSet.Spec.Replicas, 0)
	readyReplicas := statefulSet.Status.ReadyReplicas
	statefulSetReady := readyReplicas == replicas && replicas > 0

	if !statefulSetReady {
		meta.SetStatusCondition(&compute.Status.Conditions, metav1.Condition{
			Type:    v1alpha1.ComputeReady,
			Status:  metav1.ConditionFalse,
			Reason:  "StatefulSetNotReady",
			Message: fmt.Sprintf("StatefulSet has %d/%d ready replicas", readyReplicas, replicas),
		})
		return nil
	}

	// Check GRPCRoute status
	var grpcRoute gatewayv1.GRPCRoute
	if err := r.client.Get(ctx, client.ObjectKeyFromObject(compute), &grpcRoute); err != nil {
		if client.IgnoreNotFound(err) == nil {
			meta.SetStatusCondition(&compute.Status.Conditions, metav1.Condition{
				Type:    v1alpha1.ComputeReady,
				Status:  metav1.ConditionFalse,
				Reason:  "GRPCRouteNotAvailable",
				Message: "GRPCRoute is not available",
			})
			return nil
		}
		return fmt.Errorf("failed to get GRPCRoute: %w", err)
	}

	// Find the first Gateway parent ref that's in the operator namespace
	var gatewayParentRef *gatewayv1.ParentReference
	for i := range grpcRoute.Spec.ParentRefs {
		ref := &grpcRoute.Spec.ParentRefs[i]
		if ref.Kind != nil && *ref.Kind == "Gateway" {
			// Check if the Gateway is in the operator namespace
			refNamespace := r.operatorNamespace
			if ref.Namespace != nil {
				refNamespace = string(*ref.Namespace)
			}
			if refNamespace == r.operatorNamespace {
				gatewayParentRef = ref
				break
			}
		}
	}

	if gatewayParentRef == nil {
		meta.SetStatusCondition(&compute.Status.Conditions, metav1.Condition{
			Type:    v1alpha1.ComputeReady,
			Status:  metav1.ConditionFalse,
			Reason:  "GatewayNotReferenced",
			Message: fmt.Sprintf("GRPCRoute does not reference any Gateway"),
		})
		return nil
	}

	// Get the Gateway
	gatewayKey := client.ObjectKey{
		Namespace: r.operatorNamespace,
		Name:      string(gatewayParentRef.Name),
	}
	var gateway gatewayv1.Gateway
	if err := r.client.Get(ctx, gatewayKey, &gateway); err != nil {
		if client.IgnoreNotFound(err) == nil {
			meta.SetStatusCondition(&compute.Status.Conditions, metav1.Condition{
				Type:    v1alpha1.ComputeReady,
				Status:  metav1.ConditionFalse,
				Reason:  "GatewayNotAvailable",
				Message: fmt.Sprintf("Gateway %s is not available", gatewayKey.Name),
			})
			return nil
		}
		return fmt.Errorf("failed to get Gateway: %w", err)
	}

	log := ctrl.LoggerFrom(ctx)
	log.V(4).Info("Found Gateway from GRPCRoute parent ref", "gateway", gatewayKey)

	// Check if Gateway is Programmed
	gatewayProgrammed := false
	for _, cond := range gateway.Status.Conditions {
		if cond.Type == string(gatewayv1.GatewayConditionProgrammed) && cond.Status == metav1.ConditionTrue {
			gatewayProgrammed = true
			break
		}
	}

	if !gatewayProgrammed {
		meta.SetStatusCondition(&compute.Status.Conditions, metav1.Condition{
			Type:    v1alpha1.ComputeReady,
			Status:  metav1.ConditionFalse,
			Reason:  "GatewayNotProgrammed",
			Message: fmt.Sprintf("Gateway %s is not programmed", gateway.Name),
		})
		return nil
	}

	// Check if GRPCRoute is accepted by the Gateway
	grpcRouteAccepted := false
	for _, parent := range grpcRoute.Status.Parents {
		if parent.ParentRef.Kind != nil && *parent.ParentRef.Kind == "Gateway" &&
			parent.ParentRef.Name == gatewayParentRef.Name {
			// Check for Accepted condition
			for _, cond := range parent.Conditions {
				if cond.Type == string(gatewayv1.RouteConditionAccepted) && cond.Status == metav1.ConditionTrue {
					grpcRouteAccepted = true
					break
				}
			}
			break
		}
	}

	if !grpcRouteAccepted {
		meta.SetStatusCondition(&compute.Status.Conditions, metav1.Condition{
			Type:    v1alpha1.ComputeReady,
			Status:  metav1.ConditionFalse,
			Reason:  "GRPCRouteNotAccepted",
			Message: fmt.Sprintf("GRPCRoute is not accepted by Gateway %s", gateway.Name),
		})
		return nil
	}

	// Extract addresses from Gateway status
	addresses := make([]v1alpha1.ComputeAddress, 0, len(gateway.Status.Addresses))
	for _, addr := range gateway.Status.Addresses {
		addrType := "Hostname"
		if addr.Type != nil {
			addrType = string(*addr.Type)
		}
		addresses = append(addresses, v1alpha1.ComputeAddress{
			Type:  addrType,
			Value: addr.Value,
		})
	}
	compute.Status.Addresses = addresses

	// All checks passed - set Ready to True
	meta.SetStatusCondition(&compute.Status.Conditions, metav1.Condition{
		Type:    v1alpha1.ComputeReady,
		Status:  metav1.ConditionTrue,
		Reason:  "AllComponentsReady",
		Message: fmt.Sprintf("Pods (%d/%d replicas), GRPCRoute, and Gateway are ready", readyReplicas, replicas),
	})
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
		)).
		WatchesRawSource(source.TypedKind(
			mgr.GetCache(),
			&gatewayv1.Gateway{},
			handler.TypedEnqueueRequestsFromMapFunc(func(ctx context.Context, gateway *gatewayv1.Gateway) []reconcile.Request {
				// Reconcile all Computes when the Gateway changes
				var computes v1alpha1.ComputeList
				if err := mgr.GetClient().List(ctx, &computes); err != nil {
					ctrl.LoggerFrom(ctx).Error(err, "Failed to list Computes for Gateway watch")
					return nil
				}

				requests := make([]reconcile.Request, 0, len(computes.Items))
				for _, compute := range computes.Items {
					requests = append(requests, reconcile.Request{
						NamespacedName: client.ObjectKeyFromObject(&compute),
					})
				}
				return requests
			}),
		))
	return b.Complete(r)
}
