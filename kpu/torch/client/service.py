"""
Client for the PyTorch tensor streaming gRPC service.
"""

import logging

from typing import Optional

try:
    import grpc
    import torch
except ImportError as e:
    raise ImportError(f"Required dependency not found: {e}")

try:
    from kpu.torch.server import service_pb2
    from kpu.torch.server import service_pb2_grpc
except ImportError:
    raise ImportError(
        "Generated gRPC code not found. Run hack/gen-grpc-proto.sh first."
    )

from kpu.torch.server.serialization import (
    serialize_tensor_to_chunks,
    TensorAssembler
)

from grpc.aio._typing import MetadataType

logger = logging.getLogger(__name__)


class TensorClient:
    """
    Async gRPC client for streaming PyTorch tensors.

    Uses a shared gRPC channel provided by the caller (typically GRPCClient).
    """

    def __init__(self, channel: grpc.aio.Channel, metadata: Optional[MetadataType] = None):
        """
        Initialize the client.

        Args:
            channel: gRPC channel to use for communication
            metadata: Optional metadata to include in requests
        """
        self.channel = channel
        self.metadata = metadata
        self.stub = service_pb2_grpc.ServiceStub(self.channel)

    async def send_tensors(self, *tensors: torch.Tensor) -> service_pb2.TensorResponse:
        """
        Send tensors to the server.

        Args:
            *tensors: Tensors to send

        Returns:
            Response from server
        """
        async def tensor_generator():
            for tensor in tensors:
                logger.info(f"Sending tensor with shape {tensor.shape}")
                for chunk_data in serialize_tensor_to_chunks(tensor):
                    t_id, c_num, data, total, is_last, metadata = chunk_data

                    chunk = service_pb2.TensorChunk(
                        tensor_id=t_id,
                        chunk_number=c_num,
                        data=data,
                        total_chunks=total,
                        is_last=is_last
                    )

                    if metadata:
                        chunk.metadata.update(metadata)

                    yield chunk

        response = await self.stub.ReceiveTensors(tensor_generator(), metadata=self.metadata)
        return response

    async def receive_tensors(
        self,
        count: int = 1,
        parameters: dict = None
    ) -> list[torch.Tensor]:
        """
        Receive tensors from the server.

        Args:
            count: Number of tensors to request
            parameters: Optional parameters for the request

        Returns:
            List of received tensors
        """
        request = service_pb2.TensorRequest(count=count)
        if parameters:
            request.parameters.update(parameters)

        assembler = TensorAssembler()
        tensors = []

        async for chunk in self.stub.SendTensors(request, metadata=self.metadata):
            logger.debug(
                f"Received chunk {chunk.chunk_number}/{chunk.total_chunks} "
                f"for tensor {chunk.tensor_id}"
            )

            metadata = dict(chunk.metadata) if chunk.metadata else None

            tensor = assembler.add_chunk(
                tensor_id=chunk.tensor_id,
                chunk_number=chunk.chunk_number,
                data=chunk.data,
                total_chunks=chunk.total_chunks,
                is_last=chunk.is_last,
                metadata=metadata
            )

            if tensor is not None:
                tensors.append(tensor)
                logger.info(f"Received tensor with shape {tensor.shape}")

        return tensors

    async def stream_tensors(
        self,
        *tensors: torch.Tensor
    ) -> list[torch.Tensor]:
        """
        Bidirectional streaming: send tensors and receive processed tensors.

        Args:
            *tensors: Tensors to send

        Returns:
            List of processed tensors received from server
        """
        async def tensor_generator():
            for tensor in tensors:
                logger.info(f"Streaming tensor with shape {tensor.shape}")
                for chunk_data in serialize_tensor_to_chunks(tensor):
                    t_id, c_num, data, total, is_last, metadata = chunk_data

                    chunk = service_pb2.TensorChunk(
                        tensor_id=t_id,
                        chunk_number=c_num,
                        data=data,
                        total_chunks=total,
                        is_last=is_last
                    )

                    if metadata:
                        chunk.metadata.update(metadata)

                    yield chunk

        assembler = TensorAssembler()
        processed_tensors = []

        async for chunk in self.stub.StreamTensors(tensor_generator(), metadata=self.metadata):
            logger.debug(
                f"Received processed chunk {chunk.chunk_number}/{chunk.total_chunks} "
                f"for tensor {chunk.tensor_id}"
            )

            metadata = dict(chunk.metadata) if chunk.metadata else None

            tensor = assembler.add_chunk(
                tensor_id=chunk.tensor_id,
                chunk_number=chunk.chunk_number,
                data=chunk.data,
                total_chunks=chunk.total_chunks,
                is_last=chunk.is_last,
                metadata=metadata
            )

            if tensor is not None:
                processed_tensors.append(tensor)
                logger.info(f"Received processed tensor with shape {tensor.shape}")

        return processed_tensors
