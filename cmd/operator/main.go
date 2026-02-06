package main

import (
	"errors"
	"flag"
	"net/http"
	"os"
	"strings"

	zaplog "go.uber.org/zap"
	"go.uber.org/zap/zapcore"
	apiruntime "k8s.io/apimachinery/pkg/runtime"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/cache"
	"sigs.k8s.io/controller-runtime/pkg/client"
	ctrlpkg "sigs.k8s.io/controller-runtime/pkg/controller"
	"sigs.k8s.io/controller-runtime/pkg/healthz"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"
	gatewayv1 "sigs.k8s.io/gateway-api/apis/v1"

	"github.com/astefanutti/skytorch/pkg/apis/compute/v1alpha1"
	configapi "github.com/astefanutti/skytorch/pkg/apis/config/v1alpha1"
	"github.com/astefanutti/skytorch/pkg/config"
	"github.com/astefanutti/skytorch/pkg/controllers"
	"github.com/astefanutti/skytorch/pkg/util/cert"
	"github.com/astefanutti/skytorch/pkg/webhooks"
)

const (
	webhookConfigurationName = "validator.skytorch.dev"
)

var (
	scheme   = apiruntime.NewScheme()
	setupLog = ctrl.Log.WithName("setup")
)

func init() {
	utilruntime.Must(clientgoscheme.AddToScheme(scheme))
	utilruntime.Must(v1alpha1.AddToScheme(scheme))
	utilruntime.Must(gatewayv1.AddToScheme(scheme))
}

// +kubebuilder:rbac:groups="",resources=events,verbs=create;watch;update;patch
// +kubebuilder:rbac:groups=coordination.k8s.io,resources=leases,verbs=create;get;list;update

func main() {
	var configFile string
	var enableHTTP2 bool

	flag.StringVar(&configFile, "config", "",
		"The controller will load its initial configuration from this file. "+
			"Omit this flag to use the default configuration values. "+
			"Command-line flags override configuration from this file.")
	// if the enable-http2 flag is false (the default), http/2 should be disabled
	// due to its vulnerabilities. More specifically, disabling http/2 will
	// prevent from being vulnerable to the HTTP/2 Stream Cancellation and
	// Rapid Reset CVEs. For more information see:
	// - https://github.com/advisories/GHSA-qppj-fm5r-hxr3
	// - https://github.com/advisories/GHSA-4374-p667-p6c8
	flag.BoolVar(&enableHTTP2, "enable-http2", false,
		"If set, HTTP/2 will be enabled for the metrics and webhook servers")

	zapOpts := zap.Options{
		TimeEncoder: zapcore.RFC3339NanoTimeEncoder,
		ZapOpts:     []zaplog.Option{zaplog.AddCaller()},
	}
	zapOpts.BindFlags(flag.CommandLine)
	flag.Parse()

	ctrl.SetLogger(zap.New(zap.UseFlagOptions(&zapOpts)))

	setupLog.Info("Loading configuration", "configFile", configFile)
	options, cfg, err := config.Load(scheme, configFile, enableHTTP2)
	if err != nil {
		setupLog.Error(err, "Unable to load configuration")
		os.Exit(1)
	}

	// Get the operator namespace to scope the Gateway watch
	operatorNamespace, err := getOperatorNamespace()
	if err != nil {
		setupLog.Error(err, "Unable to determine operator namespace")
		os.Exit(1)
	}
	setupLog.Info("Detected operator namespace", "namespace", operatorNamespace)

	// Configure cache to only watch Gateway resources in the operator namespace
	if options.Cache.ByObject == nil {
		options.Cache.ByObject = make(map[client.Object]cache.ByObject)
	}
	options.Cache.ByObject[&gatewayv1.Gateway{}] = cache.ByObject{
		Namespaces: map[string]cache.Config{
			operatorNamespace: {},
		},
	}

	setupLog.Info("Creating manager")
	mgr, err := ctrl.NewManager(ctrl.GetConfigOrDie(), options)
	if err != nil {
		setupLog.Error(err, "unable to start manager")
		os.Exit(1)
	}

	certsReady := make(chan struct{})
	if config.IsCertManagementEnabled(&cfg) {
		setupLog.Info("Setting up certificate management")
		if err = cert.ManageCerts(mgr, cert.Config{
			WebhookSecretName:        cfg.CertManagement.WebhookSecretName,
			WebhookServiceName:       cfg.CertManagement.WebhookServiceName,
			WebhookConfigurationName: webhookConfigurationName,
		}, certsReady); err != nil {
			setupLog.Error(err, "unable to set up cert rotation")
			os.Exit(1)
		}
	} else {
		setupLog.Info("Certificate management is disabled, certificates must be provided externally")
		close(certsReady)
	}

	ctx := ctrl.SetupSignalHandler()

	setupProbeEndpoints(mgr, cfg, certsReady)

	// Set up controllers using goroutines to start the manager quickly.
	go setupControllers(mgr, cfg, certsReady, operatorNamespace)

	setupLog.Info("Starting manager")
	if err = mgr.Start(ctx); err != nil {
		setupLog.Error(err, "Could not run manager")
		os.Exit(1)
	}
}

func setupControllers(mgr ctrl.Manager, cfg configapi.Configuration, certsReady <-chan struct{}, operatorNamespace string) {
	setupLog.Info("Waiting for certificate generation to complete")
	<-certsReady
	setupLog.Info("Certs ready")

	if failedCtrlName, err := controllers.Setup(mgr, ctrlpkg.Options{}, operatorNamespace); err != nil {
		setupLog.Error(err, "Could not create controller", "controller", failedCtrlName)
		os.Exit(1)
	}
	if config.IsCertManagementEnabled(&cfg) {
		if failedWebhook, err := webhooks.Setup(mgr); err != nil {
			setupLog.Error(err, "Could not create webhook", "webhook", failedWebhook)
			os.Exit(1)
		}
	}
}

func setupProbeEndpoints(mgr ctrl.Manager, cfg configapi.Configuration, certsReady <-chan struct{}) {
	defer setupLog.Info("Probe endpoints are configured on healthz and readyz")

	if err := mgr.AddHealthzCheck("healthz", healthz.Ping); err != nil {
		setupLog.Error(err, "unable to set up health check")
		os.Exit(1)
	}

	if config.IsCertManagementEnabled(&cfg) {
		// Wait for the webhook server to be listening before advertising the
		// replica as ready. This allows users to wait with sending the first
		// requests, requiring webhooks, until the deployment is available, so
		// that the early requests are not rejected during startup.
		// We wrap the call to GetWebhookServer in a closure to delay calling
		// the function, otherwise a not fully-initialized webhook server (without
		// ready certs) fails the start of the manager.
		if err := mgr.AddReadyzCheck("readyz", func(req *http.Request) error {
			select {
			case <-certsReady:
				return mgr.GetWebhookServer().StartedChecker()(req)
			default:
				return errors.New("certificates are not ready")
			}
		}); err != nil {
			setupLog.Error(err, "unable to set up ready check")
			os.Exit(1)
		}
	} else {
		if err := mgr.AddReadyzCheck("readyz", healthz.Ping); err != nil {
			setupLog.Error(err, "unable to set up ready check")
			os.Exit(1)
		}
	}
}

// getOperatorNamespace returns the namespace where the operator is running.
// This way assumes you've set the NAMESPACE environment variable either manually, when running
// the operator standalone, or using the downward API, when running the operator in-cluster.
func getOperatorNamespace() (string, error) {
	if ns := os.Getenv("NAMESPACE"); ns != "" {
		return ns, nil
	}

	// Fall back to the namespace associated with the service account token, if available
	if data, err := os.ReadFile("/var/run/secrets/kubernetes.io/serviceaccount/namespace"); err == nil {
		if ns := strings.TrimSpace(string(data)); len(ns) > 0 {
			return ns, nil
		}
	}

	return "", errors.New("unable to determine operator namespace")
}
