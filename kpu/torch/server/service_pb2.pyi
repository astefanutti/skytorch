from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

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
    tensor_id: int
    chunk_number: int
    data: bytes
    total_chunks: int
    is_last: bool
    metadata: _containers.ScalarMap[str, str]
    def __init__(self, tensor_id: _Optional[int] = ..., chunk_number: _Optional[int] = ..., data: _Optional[bytes] = ..., total_chunks: _Optional[int] = ..., is_last: bool = ..., metadata: _Optional[_Mapping[str, str]] = ...) -> None: ...

class TensorResponse(_message.Message):
    __slots__ = ("success", "message", "received_tensor_ids")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    RECEIVED_TENSOR_IDS_FIELD_NUMBER: _ClassVar[int]
    success: bool
    message: str
    received_tensor_ids: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, success: bool = ..., message: _Optional[str] = ..., received_tensor_ids: _Optional[_Iterable[str]] = ...) -> None: ...

class CreateTensorRequest(_message.Message):
    __slots__ = ("tensor_id", "shape", "dtype", "nbytes", "device_type", "stride", "storage_offset", "device_index")
    TENSOR_ID_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    DTYPE_FIELD_NUMBER: _ClassVar[int]
    NBYTES_FIELD_NUMBER: _ClassVar[int]
    DEVICE_TYPE_FIELD_NUMBER: _ClassVar[int]
    STRIDE_FIELD_NUMBER: _ClassVar[int]
    STORAGE_OFFSET_FIELD_NUMBER: _ClassVar[int]
    DEVICE_INDEX_FIELD_NUMBER: _ClassVar[int]
    tensor_id: int
    shape: _containers.RepeatedScalarFieldContainer[int]
    dtype: str
    nbytes: int
    device_type: str
    stride: _containers.RepeatedScalarFieldContainer[int]
    storage_offset: int
    device_index: int
    def __init__(self, tensor_id: _Optional[int] = ..., shape: _Optional[_Iterable[int]] = ..., dtype: _Optional[str] = ..., nbytes: _Optional[int] = ..., device_type: _Optional[str] = ..., stride: _Optional[_Iterable[int]] = ..., storage_offset: _Optional[int] = ..., device_index: _Optional[int] = ...) -> None: ...

class CreateTensorResponse(_message.Message):
    __slots__ = ("success", "message", "tensor_id")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    TENSOR_ID_FIELD_NUMBER: _ClassVar[int]
    success: bool
    message: str
    tensor_id: int
    def __init__(self, success: bool = ..., message: _Optional[str] = ..., tensor_id: _Optional[int] = ...) -> None: ...

class GetStorageDataRequest(_message.Message):
    __slots__ = ("tensor_id", "shape", "dtype", "stride", "storage_offset")
    TENSOR_ID_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    DTYPE_FIELD_NUMBER: _ClassVar[int]
    STRIDE_FIELD_NUMBER: _ClassVar[int]
    STORAGE_OFFSET_FIELD_NUMBER: _ClassVar[int]
    tensor_id: int
    shape: _containers.RepeatedScalarFieldContainer[int]
    dtype: str
    stride: _containers.RepeatedScalarFieldContainer[int]
    storage_offset: int
    def __init__(self, tensor_id: _Optional[int] = ..., shape: _Optional[_Iterable[int]] = ..., dtype: _Optional[str] = ..., stride: _Optional[_Iterable[int]] = ..., storage_offset: _Optional[int] = ...) -> None: ...

class CopyTensorRequest(_message.Message):
    __slots__ = ("src_tensor_id", "dst_tensor_id", "src_offset", "dst_offset", "num_bytes")
    SRC_TENSOR_ID_FIELD_NUMBER: _ClassVar[int]
    DST_TENSOR_ID_FIELD_NUMBER: _ClassVar[int]
    SRC_OFFSET_FIELD_NUMBER: _ClassVar[int]
    DST_OFFSET_FIELD_NUMBER: _ClassVar[int]
    NUM_BYTES_FIELD_NUMBER: _ClassVar[int]
    src_tensor_id: int
    dst_tensor_id: int
    src_offset: int
    dst_offset: int
    num_bytes: int
    def __init__(self, src_tensor_id: _Optional[int] = ..., dst_tensor_id: _Optional[int] = ..., src_offset: _Optional[int] = ..., dst_offset: _Optional[int] = ..., num_bytes: _Optional[int] = ...) -> None: ...

class TensorReference(_message.Message):
    __slots__ = ("tensor_id", "shape", "dtype", "nbytes", "device_type", "stride", "storage_offset", "device_index")
    TENSOR_ID_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    DTYPE_FIELD_NUMBER: _ClassVar[int]
    NBYTES_FIELD_NUMBER: _ClassVar[int]
    DEVICE_TYPE_FIELD_NUMBER: _ClassVar[int]
    STRIDE_FIELD_NUMBER: _ClassVar[int]
    STORAGE_OFFSET_FIELD_NUMBER: _ClassVar[int]
    DEVICE_INDEX_FIELD_NUMBER: _ClassVar[int]
    tensor_id: int
    shape: _containers.RepeatedScalarFieldContainer[int]
    dtype: str
    nbytes: int
    device_type: str
    stride: _containers.RepeatedScalarFieldContainer[int]
    storage_offset: int
    device_index: int
    def __init__(self, tensor_id: _Optional[int] = ..., shape: _Optional[_Iterable[int]] = ..., dtype: _Optional[str] = ..., nbytes: _Optional[int] = ..., device_type: _Optional[str] = ..., stride: _Optional[_Iterable[int]] = ..., storage_offset: _Optional[int] = ..., device_index: _Optional[int] = ...) -> None: ...

class AtenArgument(_message.Message):
    __slots__ = ("tensor", "scalar_float", "scalar_int", "scalar_bool", "scalar_string", "list_value", "none_value")
    TENSOR_FIELD_NUMBER: _ClassVar[int]
    SCALAR_FLOAT_FIELD_NUMBER: _ClassVar[int]
    SCALAR_INT_FIELD_NUMBER: _ClassVar[int]
    SCALAR_BOOL_FIELD_NUMBER: _ClassVar[int]
    SCALAR_STRING_FIELD_NUMBER: _ClassVar[int]
    LIST_VALUE_FIELD_NUMBER: _ClassVar[int]
    NONE_VALUE_FIELD_NUMBER: _ClassVar[int]
    tensor: TensorReference
    scalar_float: float
    scalar_int: int
    scalar_bool: bool
    scalar_string: str
    list_value: AtenArgumentList
    none_value: bool
    def __init__(self, tensor: _Optional[_Union[TensorReference, _Mapping]] = ..., scalar_float: _Optional[float] = ..., scalar_int: _Optional[int] = ..., scalar_bool: bool = ..., scalar_string: _Optional[str] = ..., list_value: _Optional[_Union[AtenArgumentList, _Mapping]] = ..., none_value: bool = ...) -> None: ...

class AtenArgumentList(_message.Message):
    __slots__ = ("values", "is_tuple")
    VALUES_FIELD_NUMBER: _ClassVar[int]
    IS_TUPLE_FIELD_NUMBER: _ClassVar[int]
    values: _containers.RepeatedCompositeFieldContainer[AtenArgument]
    is_tuple: bool
    def __init__(self, values: _Optional[_Iterable[_Union[AtenArgument, _Mapping]]] = ..., is_tuple: bool = ...) -> None: ...

class ExecuteAtenRequest(_message.Message):
    __slots__ = ("op_name", "args", "outputs", "kwargs")
    class KwargsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: AtenArgument
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[AtenArgument, _Mapping]] = ...) -> None: ...
    OP_NAME_FIELD_NUMBER: _ClassVar[int]
    ARGS_FIELD_NUMBER: _ClassVar[int]
    OUTPUTS_FIELD_NUMBER: _ClassVar[int]
    KWARGS_FIELD_NUMBER: _ClassVar[int]
    op_name: str
    args: _containers.RepeatedCompositeFieldContainer[AtenArgument]
    outputs: _containers.RepeatedCompositeFieldContainer[TensorReference]
    kwargs: _containers.MessageMap[str, AtenArgument]
    def __init__(self, op_name: _Optional[str] = ..., args: _Optional[_Iterable[_Union[AtenArgument, _Mapping]]] = ..., outputs: _Optional[_Iterable[_Union[TensorReference, _Mapping]]] = ..., kwargs: _Optional[_Mapping[str, AtenArgument]] = ...) -> None: ...

class ExecuteAtenResponse(_message.Message):
    __slots__ = ("success", "message", "output_tensors")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_TENSORS_FIELD_NUMBER: _ClassVar[int]
    success: bool
    message: str
    output_tensors: _containers.RepeatedCompositeFieldContainer[TensorReference]
    def __init__(self, success: bool = ..., message: _Optional[str] = ..., output_tensors: _Optional[_Iterable[_Union[TensorReference, _Mapping]]] = ...) -> None: ...
