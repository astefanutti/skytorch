"""
NVIDIA GPU metrics source using NVIDIA Management Library (NVML).
"""
import logging
from typing import Dict, List, Optional

from kpu.server.metrics.source import MetricsSource, MetricData

logger = logging.getLogger(__name__)


class NvidiaGPUMetricsSource(MetricsSource):
    """
    Metrics source for NVIDIA GPUs using pynvml (Python bindings to NVML).

    Collects GPU metrics such as:
    - GPU utilization (compute and memory)
    - Temperature
    - Power usage
    - Memory usage
    - Clock speeds
    - Fan speed
    - Performance state
    """

    def __init__(self):
        """Initialize the NVIDIA GPU metrics source."""
        self._initialized = False
        self._device_count = 0
        self._nvml = None

        try:
            import pynvml
            self._nvml = pynvml
            self._nvml.nvmlInit()
            self._device_count = self._nvml.nvmlDeviceGetCount()
            self._initialized = True
            logger.info(f"NVIDIA GPU metrics source initialized with {self._device_count} device(s)")
        except ImportError:
            logger.warning(
                "pynvml not available. Install with: pip install nvidia-ml-py"
            )
        except Exception as e:
            logger.warning(f"Failed to initialize NVML: {e}")

    def get_source_name(self) -> str:
        """Get the source name."""
        return "nvidia-gpu"

    def is_available(self) -> bool:
        """Check if NVIDIA GPUs are available."""
        return self._initialized and self._device_count > 0

    def collect(
        self,
        *,
        metric_names: Optional[List[str]] = None,
        label_filters: Optional[Dict[str, str]] = None
    ) -> List[MetricData]:
        """
        Collect NVIDIA GPU metrics.

        Args:
            metric_names: Optional list of specific metrics to collect
            label_filters: Optional label filters

        Returns:
            List of metric data points
        """
        if not self.is_available():
            return []

        metrics = []

        # Collect metrics for each GPU
        for i in range(self._device_count):
            try:
                handle = self._nvml.nvmlDeviceGetHandleByIndex(i)
                device_name = self._nvml.nvmlDeviceGetName(handle)

                # Base labels for all metrics from this GPU
                base_labels = {
                    "device": str(i),
                    "gpu_name": device_name,
                }

                # Apply label filters if provided
                if label_filters:
                    if not all(base_labels.get(k) == v for k, v in label_filters.items()):
                        continue

                # Collect GPU utilization
                if self._should_collect("gpu.utilization.compute", metric_names):
                    try:
                        utilization = self._nvml.nvmlDeviceGetUtilizationRates(handle)
                        metrics.append(MetricData(
                            name="gpu.utilization.compute",
                            value=float(utilization.gpu),
                            metric_type="GAUGE",
                            unit="percent",
                            labels=base_labels,
                            help_text="GPU compute utilization"
                        ))
                    except Exception as e:
                        logger.debug(f"Failed to get GPU utilization for device {i}: {e}")

                # Collect memory utilization
                if self._should_collect("gpu.utilization.memory", metric_names):
                    try:
                        utilization = self._nvml.nvmlDeviceGetUtilizationRates(handle)
                        metrics.append(MetricData(
                            name="gpu.utilization.memory",
                            value=float(utilization.memory),
                            metric_type="GAUGE",
                            unit="percent",
                            labels=base_labels,
                            help_text="GPU memory bandwidth utilization"
                        ))
                    except Exception as e:
                        logger.debug(f"Failed to get memory utilization for device {i}: {e}")

                # Collect temperature
                if self._should_collect("gpu.temperature", metric_names):
                    try:
                        temp = self._nvml.nvmlDeviceGetTemperature(
                            handle,
                            self._nvml.NVML_TEMPERATURE_GPU
                        )
                        metrics.append(MetricData(
                            name="gpu.temperature",
                            value=float(temp),
                            metric_type="GAUGE",
                            unit="celsius",
                            labels=base_labels,
                            help_text="GPU temperature"
                        ))
                    except Exception as e:
                        logger.debug(f"Failed to get temperature for device {i}: {e}")

                # Collect power usage
                if self._should_collect("gpu.power.usage", metric_names):
                    try:
                        power = self._nvml.nvmlDeviceGetPowerUsage(handle)
                        # Convert milliwatts to watts
                        metrics.append(MetricData(
                            name="gpu.power.usage",
                            value=float(power) / 1000.0,
                            metric_type="GAUGE",
                            unit="watts",
                            labels=base_labels,
                            help_text="GPU power usage"
                        ))
                    except Exception as e:
                        logger.debug(f"Failed to get power usage for device {i}: {e}")

                # Collect power limit
                if self._should_collect("gpu.power.limit", metric_names):
                    try:
                        power_limit = self._nvml.nvmlDeviceGetPowerManagementLimit(handle)
                        # Convert milliwatts to watts
                        metrics.append(MetricData(
                            name="gpu.power.limit",
                            value=float(power_limit) / 1000.0,
                            metric_type="GAUGE",
                            unit="watts",
                            labels=base_labels,
                            help_text="GPU power limit"
                        ))
                    except Exception as e:
                        logger.debug(f"Failed to get power limit for device {i}: {e}")

                # Collect memory info
                if self._should_collect("gpu.memory.used", metric_names) or \
                   self._should_collect("gpu.memory.free", metric_names) or \
                   self._should_collect("gpu.memory.total", metric_names):
                    try:
                        mem_info = self._nvml.nvmlDeviceGetMemoryInfo(handle)

                        if self._should_collect("gpu.memory.used", metric_names):
                            metrics.append(MetricData(
                                name="gpu.memory.used",
                                value=float(mem_info.used),
                                metric_type="GAUGE",
                                unit="bytes",
                                labels=base_labels,
                                help_text="GPU memory used"
                            ))

                        if self._should_collect("gpu.memory.free", metric_names):
                            metrics.append(MetricData(
                                name="gpu.memory.free",
                                value=float(mem_info.free),
                                metric_type="GAUGE",
                                unit="bytes",
                                labels=base_labels,
                                help_text="GPU memory free"
                            ))

                        if self._should_collect("gpu.memory.total", metric_names):
                            metrics.append(MetricData(
                                name="gpu.memory.total",
                                value=float(mem_info.total),
                                metric_type="GAUGE",
                                unit="bytes",
                                labels=base_labels,
                                help_text="GPU total memory"
                            ))
                    except Exception as e:
                        logger.debug(f"Failed to get memory info for device {i}: {e}")

                # Collect clock speeds
                if self._should_collect("gpu.clock.graphics", metric_names):
                    try:
                        clock = self._nvml.nvmlDeviceGetClockInfo(
                            handle,
                            self._nvml.NVML_CLOCK_GRAPHICS
                        )
                        metrics.append(MetricData(
                            name="gpu.clock.graphics",
                            value=float(clock),
                            metric_type="GAUGE",
                            unit="mhz",
                            labels=base_labels,
                            help_text="GPU graphics clock speed"
                        ))
                    except Exception as e:
                        logger.debug(f"Failed to get graphics clock for device {i}: {e}")

                if self._should_collect("gpu.clock.sm", metric_names):
                    try:
                        clock = self._nvml.nvmlDeviceGetClockInfo(
                            handle,
                            self._nvml.NVML_CLOCK_SM
                        )
                        metrics.append(MetricData(
                            name="gpu.clock.sm",
                            value=float(clock),
                            metric_type="GAUGE",
                            unit="mhz",
                            labels=base_labels,
                            help_text="GPU SM (streaming multiprocessor) clock speed"
                        ))
                    except Exception as e:
                        logger.debug(f"Failed to get SM clock for device {i}: {e}")

                if self._should_collect("gpu.clock.memory", metric_names):
                    try:
                        clock = self._nvml.nvmlDeviceGetClockInfo(
                            handle,
                            self._nvml.NVML_CLOCK_MEM
                        )
                        metrics.append(MetricData(
                            name="gpu.clock.memory",
                            value=float(clock),
                            metric_type="GAUGE",
                            unit="mhz",
                            labels=base_labels,
                            help_text="GPU memory clock speed"
                        ))
                    except Exception as e:
                        logger.debug(f"Failed to get memory clock for device {i}: {e}")

                # Collect fan speed
                if self._should_collect("gpu.fan.speed", metric_names):
                    try:
                        fan_speed = self._nvml.nvmlDeviceGetFanSpeed(handle)
                        metrics.append(MetricData(
                            name="gpu.fan.speed",
                            value=float(fan_speed),
                            metric_type="GAUGE",
                            unit="percent",
                            labels=base_labels,
                            help_text="GPU fan speed"
                        ))
                    except Exception as e:
                        logger.debug(f"Failed to get fan speed for device {i}: {e}")

                # Collect performance state
                if self._should_collect("gpu.performance.state", metric_names):
                    try:
                        pstate = self._nvml.nvmlDeviceGetPerformanceState(handle)
                        metrics.append(MetricData(
                            name="gpu.performance.state",
                            value=float(pstate),
                            metric_type="GAUGE",
                            unit="",
                            labels=base_labels,
                            help_text="GPU performance state (0=max, 15=min)"
                        ))
                    except Exception as e:
                        logger.debug(f"Failed to get performance state for device {i}: {e}")

            except Exception as e:
                logger.error(f"Error collecting metrics for GPU {i}: {e}")

        return metrics

    def _should_collect(self, metric_name: str, filter_names: Optional[List[str]]) -> bool:
        """
        Check if a metric should be collected based on filter.

        Args:
            metric_name: Name of the metric
            filter_names: Optional list of metric names to collect

        Returns:
            True if the metric should be collected
        """
        if filter_names is None:
            return True
        return metric_name in filter_names

    def get_metadata(self) -> Dict[str, str]:
        """Get metadata about NVIDIA GPUs."""
        if not self.is_available():
            return {}

        metadata = {
            "device_count": str(self._device_count),
        }

        try:
            metadata["driver_version"] = self._nvml.nvmlSystemGetDriverVersion()
            metadata["nvml_version"] = self._nvml.nvmlSystemGetNVMLVersion()
        except Exception as e:
            logger.debug(f"Failed to get NVML metadata: {e}")

        return metadata

    def cleanup(self) -> None:
        """Cleanup NVML resources."""
        if self._initialized:
            try:
                self._nvml.nvmlShutdown()
                logger.info("NVIDIA GPU metrics source cleaned up")
            except Exception as e:
                logger.error(f"Error during NVML cleanup: {e}")
