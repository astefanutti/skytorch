"""
Base metrics source interface for modular metrics collection.
"""
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Type


# Static registry mapping source names to (module_path, class_name) tuples
# This allows dynamic importing of metrics sources based on CLI configuration
_METRICS_SOURCES_REGISTRY: Dict[str, tuple[str, str]] = {
    "nvidia-gpu": ("kpu.server.nvidia.metrics", "NvidiaGPUMetricsSource"),
}


class MetricData:
    """
    Represents a single metric data point.
    """
    def __init__(
        self,
        name: str,
        value: float,
        *,
        metric_type: str = "GAUGE",
        unit: str = "",
        labels: Optional[Dict[str, str]] = None,
        timestamp: Optional[int] = None,
        help_text: str = ""
    ):
        """
        Initialize a metric data point.

        Args:
            name: Metric name (e.g., "gpu.temperature")
            value: Numeric value
            metric_type: Type of metric ("GAUGE", "COUNTER", "HISTOGRAM")
            unit: Unit of measurement (e.g., "celsius", "percent")
            labels: Dictionary of labels/tags
            timestamp: Unix timestamp in seconds (default: current time)
            help_text: Description of the metric
        """
        self.name = name
        self.value = value
        self.metric_type = metric_type
        self.unit = unit
        self.labels = labels or {}
        self.timestamp = timestamp or int(time.time())
        self.help_text = help_text


class MetricsSource(ABC):
    """
    Abstract base class for metrics sources.

    Metrics sources provide metrics from different backends (NVIDIA GPU, AMD GPU, CPU, etc.)
    """

    @abstractmethod
    def get_source_name(self) -> str:
        """
        Get the name/identifier of this metrics source.

        Returns:
            Source name (e.g., "nvidia-gpu", "amd-gpu")
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if this metrics source is available.

        Returns:
            True if the source can provide metrics, False otherwise
        """
        pass

    @abstractmethod
    def collect(
        self,
        *,
        metric_names: Optional[List[str]] = None,
        label_filters: Optional[Dict[str, str]] = None
    ) -> List[MetricData]:
        """
        Collect current metrics from this source.

        Args:
            metric_names: Optional list of specific metric names to collect
            label_filters: Optional label filters to apply

        Returns:
            List of metric data points
        """
        pass

    def get_metadata(self) -> Dict[str, str]:
        """
        Get metadata about this metrics source.

        Returns:
            Dictionary of metadata (e.g., version info, capabilities)
        """
        return {}

    def cleanup(self) -> None:
        """
        Cleanup resources used by this metrics source.

        Called when the metrics service is shutting down.
        """
        pass


def load_metrics_source(name: str) -> Optional[MetricsSource]:
    """
    Dynamically load and instantiate a metrics source by name.

    Args:
        name: Name of the metrics source

    Returns:
        The instantiated metrics source, or None if not found or failed to load
    """
    import importlib
    import logging

    logger = logging.getLogger(__name__)

    if name not in _METRICS_SOURCES_REGISTRY:
        logger.warning(f"Metrics source '{name}' not found in registry")
        return None

    module_path, class_name = _METRICS_SOURCES_REGISTRY[name]

    try:
        # Dynamically import the module
        module = importlib.import_module(module_path)

        # Get the class from the module
        source_class = getattr(module, class_name)

        # Instantiate the metrics source
        source = source_class()

        logger.debug(f"Loaded metrics source '{name}' from {module_path}.{class_name}")
        return source

    except ImportError as e:
        logger.warning(f"Failed to import metrics source '{name}': {e}")
        return None
    except AttributeError as e:
        logger.error(f"Class '{class_name}' not found in module '{module_path}': {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to instantiate metrics source '{name}': {e}")
        return None


def list_metrics_source_names() -> List[str]:
    """
    Get list of all registered metrics source names.

    Returns:
        List of source names
    """
    return list(_METRICS_SOURCES_REGISTRY.keys())
