from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class DeleteTensorsRequest(_message.Message):
    __slots__ = ("tensor_ids",)
    TENSOR_IDS_FIELD_NUMBER: _ClassVar[int]
    tensor_ids: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, tensor_ids: _Optional[_Iterable[int]] = ...) -> None: ...

class TensorMetadata(_message.Message):
    __slots__ = ("tensor_id", "shape", "dtype", "nbytes", "device_type", "stride", "storage_offset", "device_index", "tensor_ref")
    TENSOR_ID_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    DTYPE_FIELD_NUMBER: _ClassVar[int]
    NBYTES_FIELD_NUMBER: _ClassVar[int]
    DEVICE_TYPE_FIELD_NUMBER: _ClassVar[int]
    STRIDE_FIELD_NUMBER: _ClassVar[int]
    STORAGE_OFFSET_FIELD_NUMBER: _ClassVar[int]
    DEVICE_INDEX_FIELD_NUMBER: _ClassVar[int]
    TENSOR_REF_FIELD_NUMBER: _ClassVar[int]
    tensor_id: int
    shape: _containers.RepeatedScalarFieldContainer[int]
    dtype: str
    nbytes: int
    device_type: str
    stride: _containers.RepeatedScalarFieldContainer[int]
    storage_offset: int
    device_index: int
    tensor_ref: int
    def __init__(self, tensor_id: _Optional[int] = ..., shape: _Optional[_Iterable[int]] = ..., dtype: _Optional[str] = ..., nbytes: _Optional[int] = ..., device_type: _Optional[str] = ..., stride: _Optional[_Iterable[int]] = ..., storage_offset: _Optional[int] = ..., device_index: _Optional[int] = ..., tensor_ref: _Optional[int] = ...) -> None: ...

class TensorChunk(_message.Message):
    __slots__ = ("tensor_id", "chunk_number", "data", "total_chunks", "shape", "stride", "storage_offset", "dtype", "total_bytes", "metadata")
    TENSOR_ID_FIELD_NUMBER: _ClassVar[int]
    CHUNK_NUMBER_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    TOTAL_CHUNKS_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    STRIDE_FIELD_NUMBER: _ClassVar[int]
    STORAGE_OFFSET_FIELD_NUMBER: _ClassVar[int]
    DTYPE_FIELD_NUMBER: _ClassVar[int]
    TOTAL_BYTES_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    tensor_id: int
    chunk_number: int
    data: bytes
    total_chunks: int
    shape: _containers.RepeatedScalarFieldContainer[int]
    stride: _containers.RepeatedScalarFieldContainer[int]
    storage_offset: int
    dtype: str
    total_bytes: int
    metadata: TensorMetadata
    def __init__(self, tensor_id: _Optional[int] = ..., chunk_number: _Optional[int] = ..., data: _Optional[bytes] = ..., total_chunks: _Optional[int] = ..., shape: _Optional[_Iterable[int]] = ..., stride: _Optional[_Iterable[int]] = ..., storage_offset: _Optional[int] = ..., dtype: _Optional[str] = ..., total_bytes: _Optional[int] = ..., metadata: _Optional[_Union[TensorMetadata, _Mapping]] = ...) -> None: ...

class TensorResponse(_message.Message):
    __slots__ = ("success", "message")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    success: bool
    message: str
    def __init__(self, success: bool = ..., message: _Optional[str] = ...) -> None: ...

class GetScalarRequest(_message.Message):
    __slots__ = ("tensor_id", "tensor_metadata")
    TENSOR_ID_FIELD_NUMBER: _ClassVar[int]
    TENSOR_METADATA_FIELD_NUMBER: _ClassVar[int]
    tensor_id: int
    tensor_metadata: _containers.RepeatedCompositeFieldContainer[TensorMetadata]
    def __init__(self, tensor_id: _Optional[int] = ..., tensor_metadata: _Optional[_Iterable[_Union[TensorMetadata, _Mapping]]] = ...) -> None: ...

class GetScalarResponse(_message.Message):
    __slots__ = ("float_value", "int_value", "bool_value")
    FLOAT_VALUE_FIELD_NUMBER: _ClassVar[int]
    INT_VALUE_FIELD_NUMBER: _ClassVar[int]
    BOOL_VALUE_FIELD_NUMBER: _ClassVar[int]
    float_value: float
    int_value: int
    bool_value: bool
    def __init__(self, float_value: _Optional[float] = ..., int_value: _Optional[int] = ..., bool_value: bool = ...) -> None: ...

class CreateTensorRequest(_message.Message):
    __slots__ = ("tensor_id", "shape", "dtype", "nbytes", "device_type", "stride", "storage_offset", "device_index", "tensor_ref")
    TENSOR_ID_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    DTYPE_FIELD_NUMBER: _ClassVar[int]
    NBYTES_FIELD_NUMBER: _ClassVar[int]
    DEVICE_TYPE_FIELD_NUMBER: _ClassVar[int]
    STRIDE_FIELD_NUMBER: _ClassVar[int]
    STORAGE_OFFSET_FIELD_NUMBER: _ClassVar[int]
    DEVICE_INDEX_FIELD_NUMBER: _ClassVar[int]
    TENSOR_REF_FIELD_NUMBER: _ClassVar[int]
    tensor_id: int
    shape: _containers.RepeatedScalarFieldContainer[int]
    dtype: str
    nbytes: int
    device_type: str
    stride: _containers.RepeatedScalarFieldContainer[int]
    storage_offset: int
    device_index: int
    tensor_ref: int
    def __init__(self, tensor_id: _Optional[int] = ..., shape: _Optional[_Iterable[int]] = ..., dtype: _Optional[str] = ..., nbytes: _Optional[int] = ..., device_type: _Optional[str] = ..., stride: _Optional[_Iterable[int]] = ..., storage_offset: _Optional[int] = ..., device_index: _Optional[int] = ..., tensor_ref: _Optional[int] = ...) -> None: ...

class GetTensorRequest(_message.Message):
    __slots__ = ("tensor_id", "shape", "dtype", "stride", "storage_offset", "metadata")
    TENSOR_ID_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    DTYPE_FIELD_NUMBER: _ClassVar[int]
    STRIDE_FIELD_NUMBER: _ClassVar[int]
    STORAGE_OFFSET_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    tensor_id: int
    shape: _containers.RepeatedScalarFieldContainer[int]
    dtype: str
    stride: _containers.RepeatedScalarFieldContainer[int]
    storage_offset: int
    metadata: TensorMetadata
    def __init__(self, tensor_id: _Optional[int] = ..., shape: _Optional[_Iterable[int]] = ..., dtype: _Optional[str] = ..., stride: _Optional[_Iterable[int]] = ..., storage_offset: _Optional[int] = ..., metadata: _Optional[_Union[TensorMetadata, _Mapping]] = ...) -> None: ...

class CopyTensorRequest(_message.Message):
    __slots__ = ("src_tensor_id", "dst_tensor_id", "src_offset", "dst_offset", "num_bytes", "src_metadata", "dst_metadata")
    SRC_TENSOR_ID_FIELD_NUMBER: _ClassVar[int]
    DST_TENSOR_ID_FIELD_NUMBER: _ClassVar[int]
    SRC_OFFSET_FIELD_NUMBER: _ClassVar[int]
    DST_OFFSET_FIELD_NUMBER: _ClassVar[int]
    NUM_BYTES_FIELD_NUMBER: _ClassVar[int]
    SRC_METADATA_FIELD_NUMBER: _ClassVar[int]
    DST_METADATA_FIELD_NUMBER: _ClassVar[int]
    src_tensor_id: int
    dst_tensor_id: int
    src_offset: int
    dst_offset: int
    num_bytes: int
    src_metadata: TensorMetadata
    dst_metadata: TensorMetadata
    def __init__(self, src_tensor_id: _Optional[int] = ..., dst_tensor_id: _Optional[int] = ..., src_offset: _Optional[int] = ..., dst_offset: _Optional[int] = ..., num_bytes: _Optional[int] = ..., src_metadata: _Optional[_Union[TensorMetadata, _Mapping]] = ..., dst_metadata: _Optional[_Union[TensorMetadata, _Mapping]] = ...) -> None: ...

class TensorReference(_message.Message):
    __slots__ = ("tensor_id",)
    TENSOR_ID_FIELD_NUMBER: _ClassVar[int]
    tensor_id: int
    def __init__(self, tensor_id: _Optional[int] = ...) -> None: ...

class AtenArgument(_message.Message):
    __slots__ = ("tensor", "scalar_float", "scalar_int", "scalar_bool", "scalar_string", "list_value", "none_value", "scalar_dtype", "scalar_memory_format", "scalar_layout")
    TENSOR_FIELD_NUMBER: _ClassVar[int]
    SCALAR_FLOAT_FIELD_NUMBER: _ClassVar[int]
    SCALAR_INT_FIELD_NUMBER: _ClassVar[int]
    SCALAR_BOOL_FIELD_NUMBER: _ClassVar[int]
    SCALAR_STRING_FIELD_NUMBER: _ClassVar[int]
    LIST_VALUE_FIELD_NUMBER: _ClassVar[int]
    NONE_VALUE_FIELD_NUMBER: _ClassVar[int]
    SCALAR_DTYPE_FIELD_NUMBER: _ClassVar[int]
    SCALAR_MEMORY_FORMAT_FIELD_NUMBER: _ClassVar[int]
    SCALAR_LAYOUT_FIELD_NUMBER: _ClassVar[int]
    tensor: TensorReference
    scalar_float: float
    scalar_int: int
    scalar_bool: bool
    scalar_string: str
    list_value: AtenArgumentList
    none_value: bool
    scalar_dtype: str
    scalar_memory_format: str
    scalar_layout: str
    def __init__(self, tensor: _Optional[_Union[TensorReference, _Mapping]] = ..., scalar_float: _Optional[float] = ..., scalar_int: _Optional[int] = ..., scalar_bool: bool = ..., scalar_string: _Optional[str] = ..., list_value: _Optional[_Union[AtenArgumentList, _Mapping]] = ..., none_value: bool = ..., scalar_dtype: _Optional[str] = ..., scalar_memory_format: _Optional[str] = ..., scalar_layout: _Optional[str] = ...) -> None: ...

class AtenArgumentList(_message.Message):
    __slots__ = ("values", "is_tuple")
    VALUES_FIELD_NUMBER: _ClassVar[int]
    IS_TUPLE_FIELD_NUMBER: _ClassVar[int]
    values: _containers.RepeatedCompositeFieldContainer[AtenArgument]
    is_tuple: bool
    def __init__(self, values: _Optional[_Iterable[_Union[AtenArgument, _Mapping]]] = ..., is_tuple: bool = ...) -> None: ...

class ExecuteAtenRequest(_message.Message):
    __slots__ = ("op_name", "args", "outputs", "kwargs", "tensor_metadata", "output_metadata")
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
    TENSOR_METADATA_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_METADATA_FIELD_NUMBER: _ClassVar[int]
    op_name: str
    args: _containers.RepeatedCompositeFieldContainer[AtenArgument]
    outputs: _containers.RepeatedCompositeFieldContainer[TensorReference]
    kwargs: _containers.MessageMap[str, AtenArgument]
    tensor_metadata: _containers.RepeatedCompositeFieldContainer[TensorMetadata]
    output_metadata: _containers.RepeatedCompositeFieldContainer[TensorMetadata]
    def __init__(self, op_name: _Optional[str] = ..., args: _Optional[_Iterable[_Union[AtenArgument, _Mapping]]] = ..., outputs: _Optional[_Iterable[_Union[TensorReference, _Mapping]]] = ..., kwargs: _Optional[_Mapping[str, AtenArgument]] = ..., tensor_metadata: _Optional[_Iterable[_Union[TensorMetadata, _Mapping]]] = ..., output_metadata: _Optional[_Iterable[_Union[TensorMetadata, _Mapping]]] = ...) -> None: ...

class ExecuteAtenResponse(_message.Message):
    __slots__ = ("success", "message", "output_tensors")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_TENSORS_FIELD_NUMBER: _ClassVar[int]
    success: bool
    message: str
    output_tensors: _containers.RepeatedCompositeFieldContainer[TensorReference]
    def __init__(self, success: bool = ..., message: _Optional[str] = ..., output_tensors: _Optional[_Iterable[_Union[TensorReference, _Mapping]]] = ...) -> None: ...

class UpdateTensorRequest(_message.Message):
    __slots__ = ("tensor_id", "data", "shape", "dtype", "stride", "storage_offset", "metadata")
    TENSOR_ID_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    DTYPE_FIELD_NUMBER: _ClassVar[int]
    STRIDE_FIELD_NUMBER: _ClassVar[int]
    STORAGE_OFFSET_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    tensor_id: int
    data: bytes
    shape: _containers.RepeatedScalarFieldContainer[int]
    dtype: str
    stride: _containers.RepeatedScalarFieldContainer[int]
    storage_offset: int
    metadata: TensorMetadata
    def __init__(self, tensor_id: _Optional[int] = ..., data: _Optional[bytes] = ..., shape: _Optional[_Iterable[int]] = ..., dtype: _Optional[str] = ..., stride: _Optional[_Iterable[int]] = ..., storage_offset: _Optional[int] = ..., metadata: _Optional[_Union[TensorMetadata, _Mapping]] = ...) -> None: ...

class GetTensorResponse(_message.Message):
    __slots__ = ("success", "message", "data", "shape", "dtype", "stride", "storage_offset")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    DTYPE_FIELD_NUMBER: _ClassVar[int]
    STRIDE_FIELD_NUMBER: _ClassVar[int]
    STORAGE_OFFSET_FIELD_NUMBER: _ClassVar[int]
    success: bool
    message: str
    data: bytes
    shape: _containers.RepeatedScalarFieldContainer[int]
    dtype: str
    stride: _containers.RepeatedScalarFieldContainer[int]
    storage_offset: int
    def __init__(self, success: bool = ..., message: _Optional[str] = ..., data: _Optional[bytes] = ..., shape: _Optional[_Iterable[int]] = ..., dtype: _Optional[str] = ..., stride: _Optional[_Iterable[int]] = ..., storage_offset: _Optional[int] = ...) -> None: ...

class ExecuteFunctionRequest(_message.Message):
    __slots__ = ("callable", "args", "kwargs", "callable_source", "callable_name")
    CALLABLE_FIELD_NUMBER: _ClassVar[int]
    ARGS_FIELD_NUMBER: _ClassVar[int]
    KWARGS_FIELD_NUMBER: _ClassVar[int]
    CALLABLE_SOURCE_FIELD_NUMBER: _ClassVar[int]
    CALLABLE_NAME_FIELD_NUMBER: _ClassVar[int]
    callable: bytes
    args: bytes
    kwargs: bytes
    callable_source: str
    callable_name: str
    def __init__(self, callable: _Optional[bytes] = ..., args: _Optional[bytes] = ..., kwargs: _Optional[bytes] = ..., callable_source: _Optional[str] = ..., callable_name: _Optional[str] = ...) -> None: ...

class RemoteTensorInfo(_message.Message):
    __slots__ = ("name", "storage_id", "shape", "dtype", "stride", "storage_offset", "storage_nbytes", "device_type", "device_index")
    NAME_FIELD_NUMBER: _ClassVar[int]
    STORAGE_ID_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    DTYPE_FIELD_NUMBER: _ClassVar[int]
    STRIDE_FIELD_NUMBER: _ClassVar[int]
    STORAGE_OFFSET_FIELD_NUMBER: _ClassVar[int]
    STORAGE_NBYTES_FIELD_NUMBER: _ClassVar[int]
    DEVICE_TYPE_FIELD_NUMBER: _ClassVar[int]
    DEVICE_INDEX_FIELD_NUMBER: _ClassVar[int]
    name: str
    storage_id: int
    shape: _containers.RepeatedScalarFieldContainer[int]
    dtype: str
    stride: _containers.RepeatedScalarFieldContainer[int]
    storage_offset: int
    storage_nbytes: int
    device_type: str
    device_index: int
    def __init__(self, name: _Optional[str] = ..., storage_id: _Optional[int] = ..., shape: _Optional[_Iterable[int]] = ..., dtype: _Optional[str] = ..., stride: _Optional[_Iterable[int]] = ..., storage_offset: _Optional[int] = ..., storage_nbytes: _Optional[int] = ..., device_type: _Optional[str] = ..., device_index: _Optional[int] = ...) -> None: ...

class ExecuteFunctionResponse(_message.Message):
    __slots__ = ("success", "error_message", "tensors")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    TENSORS_FIELD_NUMBER: _ClassVar[int]
    success: bool
    error_message: str
    tensors: _containers.RepeatedCompositeFieldContainer[RemoteTensorInfo]
    def __init__(self, success: bool = ..., error_message: _Optional[str] = ..., tensors: _Optional[_Iterable[_Union[RemoteTensorInfo, _Mapping]]] = ...) -> None: ...

class RegisterTensorsRequest(_message.Message):
    __slots__ = ("registrations",)
    REGISTRATIONS_FIELD_NUMBER: _ClassVar[int]
    registrations: _containers.RepeatedCompositeFieldContainer[TensorRegistration]
    def __init__(self, registrations: _Optional[_Iterable[_Union[TensorRegistration, _Mapping]]] = ...) -> None: ...

class TensorRegistration(_message.Message):
    __slots__ = ("storage_id", "tensor_id")
    STORAGE_ID_FIELD_NUMBER: _ClassVar[int]
    TENSOR_ID_FIELD_NUMBER: _ClassVar[int]
    storage_id: int
    tensor_id: int
    def __init__(self, storage_id: _Optional[int] = ..., tensor_id: _Optional[int] = ...) -> None: ...

class BatchedExecuteAtenRequest(_message.Message):
    __slots__ = ("operations",)
    OPERATIONS_FIELD_NUMBER: _ClassVar[int]
    operations: _containers.RepeatedCompositeFieldContainer[ExecuteAtenRequest]
    def __init__(self, operations: _Optional[_Iterable[_Union[ExecuteAtenRequest, _Mapping]]] = ...) -> None: ...

class StreamRequest(_message.Message):
    __slots__ = ("execute_aten", "delete_tensors", "copy_tensor", "update_tensor", "get_tensor", "register_tensors", "batched_execute_aten", "raw_execute_aten", "raw_batched_execute_aten", "get_scalar", "chunk_number", "total_chunks", "total_bytes")
    EXECUTE_ATEN_FIELD_NUMBER: _ClassVar[int]
    DELETE_TENSORS_FIELD_NUMBER: _ClassVar[int]
    COPY_TENSOR_FIELD_NUMBER: _ClassVar[int]
    UPDATE_TENSOR_FIELD_NUMBER: _ClassVar[int]
    GET_TENSOR_FIELD_NUMBER: _ClassVar[int]
    REGISTER_TENSORS_FIELD_NUMBER: _ClassVar[int]
    BATCHED_EXECUTE_ATEN_FIELD_NUMBER: _ClassVar[int]
    RAW_EXECUTE_ATEN_FIELD_NUMBER: _ClassVar[int]
    RAW_BATCHED_EXECUTE_ATEN_FIELD_NUMBER: _ClassVar[int]
    GET_SCALAR_FIELD_NUMBER: _ClassVar[int]
    CHUNK_NUMBER_FIELD_NUMBER: _ClassVar[int]
    TOTAL_CHUNKS_FIELD_NUMBER: _ClassVar[int]
    TOTAL_BYTES_FIELD_NUMBER: _ClassVar[int]
    execute_aten: ExecuteAtenRequest
    delete_tensors: DeleteTensorsRequest
    copy_tensor: CopyTensorRequest
    update_tensor: UpdateTensorRequest
    get_tensor: GetTensorRequest
    register_tensors: RegisterTensorsRequest
    batched_execute_aten: BatchedExecuteAtenRequest
    raw_execute_aten: bytes
    raw_batched_execute_aten: bytes
    get_scalar: GetScalarRequest
    chunk_number: int
    total_chunks: int
    total_bytes: int
    def __init__(self, execute_aten: _Optional[_Union[ExecuteAtenRequest, _Mapping]] = ..., delete_tensors: _Optional[_Union[DeleteTensorsRequest, _Mapping]] = ..., copy_tensor: _Optional[_Union[CopyTensorRequest, _Mapping]] = ..., update_tensor: _Optional[_Union[UpdateTensorRequest, _Mapping]] = ..., get_tensor: _Optional[_Union[GetTensorRequest, _Mapping]] = ..., register_tensors: _Optional[_Union[RegisterTensorsRequest, _Mapping]] = ..., batched_execute_aten: _Optional[_Union[BatchedExecuteAtenRequest, _Mapping]] = ..., raw_execute_aten: _Optional[bytes] = ..., raw_batched_execute_aten: _Optional[bytes] = ..., get_scalar: _Optional[_Union[GetScalarRequest, _Mapping]] = ..., chunk_number: _Optional[int] = ..., total_chunks: _Optional[int] = ..., total_bytes: _Optional[int] = ...) -> None: ...

class StreamResponse(_message.Message):
    __slots__ = ("success", "error_message", "execute_aten", "delete_tensors", "copy_tensor", "update_tensor", "get_tensor", "register_tensors", "get_scalar", "chunk_number", "total_chunks")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    EXECUTE_ATEN_FIELD_NUMBER: _ClassVar[int]
    DELETE_TENSORS_FIELD_NUMBER: _ClassVar[int]
    COPY_TENSOR_FIELD_NUMBER: _ClassVar[int]
    UPDATE_TENSOR_FIELD_NUMBER: _ClassVar[int]
    GET_TENSOR_FIELD_NUMBER: _ClassVar[int]
    REGISTER_TENSORS_FIELD_NUMBER: _ClassVar[int]
    GET_SCALAR_FIELD_NUMBER: _ClassVar[int]
    CHUNK_NUMBER_FIELD_NUMBER: _ClassVar[int]
    TOTAL_CHUNKS_FIELD_NUMBER: _ClassVar[int]
    success: bool
    error_message: str
    execute_aten: ExecuteAtenResponse
    delete_tensors: TensorResponse
    copy_tensor: TensorResponse
    update_tensor: TensorResponse
    get_tensor: GetTensorResponse
    register_tensors: TensorResponse
    get_scalar: GetScalarResponse
    chunk_number: int
    total_chunks: int
    def __init__(self, success: bool = ..., error_message: _Optional[str] = ..., execute_aten: _Optional[_Union[ExecuteAtenResponse, _Mapping]] = ..., delete_tensors: _Optional[_Union[TensorResponse, _Mapping]] = ..., copy_tensor: _Optional[_Union[TensorResponse, _Mapping]] = ..., update_tensor: _Optional[_Union[TensorResponse, _Mapping]] = ..., get_tensor: _Optional[_Union[GetTensorResponse, _Mapping]] = ..., register_tensors: _Optional[_Union[TensorResponse, _Mapping]] = ..., get_scalar: _Optional[_Union[GetScalarResponse, _Mapping]] = ..., chunk_number: _Optional[int] = ..., total_chunks: _Optional[int] = ...) -> None: ...
