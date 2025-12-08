from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class TensorChunk(_message.Message):
    __slots__ = ("tensor_id", "chunk_number", "data", "total_chunks", "is_last", "metadata")
    class MetadataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    TENSOR_ID_FIELD_NUMBER: _ClassVar[int]
    CHUNK_NUMBER_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    TOTAL_CHUNKS_FIELD_NUMBER: _ClassVar[int]
    IS_LAST_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    tensor_id: str
    chunk_number: int
    data: bytes
    total_chunks: int
    is_last: bool
    metadata: _containers.ScalarMap[str, str]
    def __init__(self, tensor_id: _Optional[str] = ..., chunk_number: _Optional[int] = ..., data: _Optional[bytes] = ..., total_chunks: _Optional[int] = ..., is_last: bool = ..., metadata: _Optional[_Mapping[str, str]] = ...) -> None: ...

class TensorResponse(_message.Message):
    __slots__ = ("success", "message", "received_tensor_ids")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    RECEIVED_TENSOR_IDS_FIELD_NUMBER: _ClassVar[int]
    success: bool
    message: str
    received_tensor_ids: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, success: bool = ..., message: _Optional[str] = ..., received_tensor_ids: _Optional[_Iterable[str]] = ...) -> None: ...

class TensorRequest(_message.Message):
    __slots__ = ("count", "parameters")
    class ParametersEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    COUNT_FIELD_NUMBER: _ClassVar[int]
    PARAMETERS_FIELD_NUMBER: _ClassVar[int]
    count: int
    parameters: _containers.ScalarMap[str, str]
    def __init__(self, count: _Optional[int] = ..., parameters: _Optional[_Mapping[str, str]] = ...) -> None: ...
