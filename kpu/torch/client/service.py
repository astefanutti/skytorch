"""
Example client for the PyTorch tensor streaming gRPC service.

This demonstrates how to use the async streaming methods to send and receive tensors.
"""

import asyncio
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
    """

    def __init__(self, host: str = 'localhost', port: int = 50051, metadata: Optional[MetadataType] = None):
        """
        Initialize the client.

        Args:
            host: Server host
            port: Server port
        """
        self.address = f'{host}:{port}'
        self.metadata = metadata
        self.channel = None
        self.stub = None

    async def __aenter__(self):
        """Async context manager entry."""
        self.channel = grpc.aio.insecure_channel(self.address)
        self.stub = service_pb2_grpc.ServiceStub(self.channel)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.channel:
            await self.channel.close()

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


async def main():
    """Example usage of the tensor client."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create example tensors
    tensor1 = torch.randn(100, 100)
    tensor2 = torch.randn(50, 50)

    async with TensorClient() as client:
        # Example 1: Send tensors to server
        logger.info("Example 1: Sending tensors to server")
        response = await client.send_tensors(tensor1, tensor2)
        logger.info(f"Server response: {response.message}")
        logger.info(f"Received tensor IDs: {response.received_tensor_ids}")

        # Example 2: Receive tensors from server
        logger.info("Example 2: Receiving tensors from server")
        received = await client.receive_tensors(
            count=2,
            parameters={'shape': '20,20'}
        )
        logger.info(f"Received {len(received)} tensors")
        for i, tensor in enumerate(received):
            logger.info(f"  Tensor {i}: shape={tensor.shape}, dtype={tensor.dtype}")

        # Example 3: Bidirectional streaming
        logger.info("Example 3: Bidirectional streaming")
        processed = await client.stream_tensors(tensor1, tensor2)
        logger.info(f"Received {len(processed)} processed tensors")
        for i, tensor in enumerate(processed):
            logger.info(f"  Processed tensor {i}: shape={tensor.shape}, dtype={tensor.dtype}")


if __name__ == '__main__':
    asyncio.run(main())
