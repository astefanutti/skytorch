from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class MetricType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    UNKNOWN: _ClassVar[MetricType]
    GAUGE: _ClassVar[MetricType]
    COUNTER: _ClassVar[MetricType]
    HISTOGRAM: _ClassVar[MetricType]
UNKNOWN: MetricType
GAUGE: MetricType
COUNTER: MetricType
HISTOGRAM: MetricType

class Metric(_message.Message):
    __slots__ = ("name", "type", "value", "unit", "labels", "timestamp", "help")
    class LabelsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    NAME_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    UNIT_FIELD_NUMBER: _ClassVar[int]
    LABELS_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    HELP_FIELD_NUMBER: _ClassVar[int]
    name: str
    type: MetricType
    value: float
    unit: str
    labels: _containers.ScalarMap[str, str]
    timestamp: int
    help: str
    def __init__(self, name: _Optional[str] = ..., type: _Optional[_Union[MetricType, str]] = ..., value: _Optional[float] = ..., unit: _Optional[str] = ..., labels: _Optional[_Mapping[str, str]] = ..., timestamp: _Optional[int] = ..., help: _Optional[str] = ...) -> None: ...

class MetricsSnapshot(_message.Message):
    __slots__ = ("metrics", "source", "timestamp")
    METRICS_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    metrics: _containers.RepeatedCompositeFieldContainer[Metric]
    source: str
    timestamp: int
    def __init__(self, metrics: _Optional[_Iterable[_Union[Metric, _Mapping]]] = ..., source: _Optional[str] = ..., timestamp: _Optional[int] = ...) -> None: ...

class GetMetricsRequest(_message.Message):
    __slots__ = ("metric_names", "sources", "label_filters")
    class LabelFiltersEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    METRIC_NAMES_FIELD_NUMBER: _ClassVar[int]
    SOURCES_FIELD_NUMBER: _ClassVar[int]
    LABEL_FILTERS_FIELD_NUMBER: _ClassVar[int]
    metric_names: _containers.RepeatedScalarFieldContainer[str]
    sources: _containers.RepeatedScalarFieldContainer[str]
    label_filters: _containers.ScalarMap[str, str]
    def __init__(self, metric_names: _Optional[_Iterable[str]] = ..., sources: _Optional[_Iterable[str]] = ..., label_filters: _Optional[_Mapping[str, str]] = ...) -> None: ...

class GetMetricsResponse(_message.Message):
    __slots__ = ("snapshots",)
    SNAPSHOTS_FIELD_NUMBER: _ClassVar[int]
    snapshots: _containers.RepeatedCompositeFieldContainer[MetricsSnapshot]
    def __init__(self, snapshots: _Optional[_Iterable[_Union[MetricsSnapshot, _Mapping]]] = ...) -> None: ...

class StreamMetricsRequest(_message.Message):
    __slots__ = ("interval_seconds", "metric_names", "sources", "label_filters")
    class LabelFiltersEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    INTERVAL_SECONDS_FIELD_NUMBER: _ClassVar[int]
    METRIC_NAMES_FIELD_NUMBER: _ClassVar[int]
    SOURCES_FIELD_NUMBER: _ClassVar[int]
    LABEL_FILTERS_FIELD_NUMBER: _ClassVar[int]
    interval_seconds: float
    metric_names: _containers.RepeatedScalarFieldContainer[str]
    sources: _containers.RepeatedScalarFieldContainer[str]
    label_filters: _containers.ScalarMap[str, str]
    def __init__(self, interval_seconds: _Optional[float] = ..., metric_names: _Optional[_Iterable[str]] = ..., sources: _Optional[_Iterable[str]] = ..., label_filters: _Optional[_Mapping[str, str]] = ...) -> None: ...
