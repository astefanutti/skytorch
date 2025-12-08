"""
Async gRPC server implementation for streaming PyTorch tensors.
"""

import logging
from typing import AsyncIterator

try:
    import grpc
    import torch
except ImportError as e:
    raise ImportError(f"Required dependency not found: {e}. Install with: pip install grpcio torch")

# These imports will work after running generate_proto.sh
try:
    from kpu.torch.server import service_pb2
    from kpu.torch.server import service_pb2_grpc
except ImportError:
    raise ImportError(
        "Generated gRPC code not found. Run ./generate_proto.sh first.\n"
        "Make sure to install grpcio-tools: pip install grpcio-tools"
    )

from kpu.torch.server.serialization import (
    serialize_tensor_to_chunks,
    TensorAssembler,
    DEFAULT_CHUNK_SIZE
)


logger = logging.getLogger(__name__)


class TensorServicer(service_pb2_grpc.ServiceServicer):
    """
    Async gRPC servicer for streaming PyTorch tensors.
    """

    def __init__(self, chunk_size: int = DEFAULT_CHUNK_SIZE):
        """
        Initialize the tensor servicer.

        Args:
            chunk_size: Size of chunks for streaming tensors
        """
        self.chunk_size = chunk_size
        self.received_tensors = {}  # Store received tensors (in production, use proper storage)

    async def ReceiveTensors(
        self,
        request_iterator: AsyncIterator[service_pb2.TensorChunk],
        context: grpc.aio.ServicerContext
    ) -> service_pb2.TensorResponse:
        """
        Receive tensors from client via streaming.

        Args:
            request_iterator: Async iterator of tensor chunks from client
            context: gRPC context

        Returns:
            Response indicating success and received tensor IDs
        """
        assembler = TensorAssembler()
        received_ids = []

        try:
            async for chunk in request_iterator:
                logger.debug(
                    f"Received chunk {chunk.chunk_number}/{chunk.total_chunks} "
                    f"for tensor {chunk.tensor_id}"
                )

                # Convert metadata from proto map to dict
                metadata = dict(chunk.metadata) if chunk.metadata else None

                # Add chunk to assembler
                tensor = assembler.add_chunk(
                    tensor_id=chunk.tensor_id,
                    chunk_number=chunk.chunk_number,
                    data=chunk.data,
                    total_chunks=chunk.total_chunks,
                    is_last=chunk.is_last,
                    metadata=metadata
                )

                # If tensor is complete, store it
                if tensor is not None:
                    self.received_tensors[chunk.tensor_id] = tensor
                    received_ids.append(chunk.tensor_id)
                    logger.info(
                        f"Successfully received tensor {chunk.tensor_id} "
                        f"with shape {tensor.shape}"
                    )

            return service_pb2.TensorResponse(
                success=True,
                message=f"Received {len(received_ids)} tensor(s)",
                received_tensor_ids=received_ids
            )

        except Exception as e:
            logger.error(f"Error receiving tensors: {e}")
            return service_pb2.TensorResponse(
                success=False,
                message=f"Error: {str(e)}",
                received_tensor_ids=received_ids
            )

    async def SendTensors(
        self,
        request: service_pb2.TensorRequest,
        context: grpc.aio.ServicerContext
    ) -> AsyncIterator[service_pb2.TensorChunk]:
        """
        Send tensors to client via streaming.

        Args:
            request: Request parameters (count, filters, etc.)
            context: gRPC context

        Yields:
            Tensor chunks
        """
        try:
            # Get tensors to send (in this example, we'll send random tensors)
            # In production, implement your own logic to select/generate tensors
            count = request.count if request.count > 0 else 1
            parameters = dict(request.parameters) if request.parameters else {}

            logger.info(f"Sending {count} tensor(s) with parameters: {parameters}")

            for i in range(count):
                # Example: create a random tensor
                # In production, replace this with your tensor source
                tensor = self._get_tensor_to_send(i, parameters)

                # Serialize and stream tensor
                for chunk_data in serialize_tensor_to_chunks(
                    tensor,
                    chunk_size=self.chunk_size
                ):
                    tensor_id, chunk_num, data, total, is_last, metadata = chunk_data

                    chunk = service_pb2.TensorChunk(
                        tensor_id=tensor_id,
                        chunk_number=chunk_num,
                        data=data,
                        total_chunks=total,
                        is_last=is_last
                    )

                    # Add metadata to first chunk
                    if metadata:
                        chunk.metadata.update(metadata)

                    logger.debug(
                        f"Sending chunk {chunk_num}/{total} for tensor {tensor_id}"
                    )

                    yield chunk

                logger.info(f"Sent tensor {i+1}/{count} with shape {tensor.shape}")

        except Exception as e:
            logger.error(f"Error sending tensors: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, f"Error sending tensors: {e}")

    async def StreamTensors(
        self,
        request_iterator: AsyncIterator[service_pb2.TensorChunk],
        context: grpc.aio.ServicerContext
    ) -> AsyncIterator[service_pb2.TensorChunk]:
        """
        Bidirectional streaming: receive and send tensors simultaneously.

        This method receives tensors from the client and sends back processed tensors.
        In this example, we echo back the received tensors, but in production
        you would implement your own processing logic.

        Args:
            request_iterator: Async iterator of tensor chunks from client
            context: gRPC context

        Yields:
            Processed tensor chunks
        """
        assembler = TensorAssembler()

        try:
            async for chunk in request_iterator:
                logger.debug(
                    f"Received chunk {chunk.chunk_number}/{chunk.total_chunks} "
                    f"for tensor {chunk.tensor_id}"
                )

                # Convert metadata from proto map to dict
                metadata = dict(chunk.metadata) if chunk.metadata else None

                # Add chunk to assembler
                tensor = assembler.add_chunk(
                    tensor_id=chunk.tensor_id,
                    chunk_number=chunk.chunk_number,
                    data=chunk.data,
                    total_chunks=chunk.total_chunks,
                    is_last=chunk.is_last,
                    metadata=metadata
                )

                # If tensor is complete, process and send back
                if tensor is not None:
                    logger.info(
                        f"Received complete tensor {chunk.tensor_id} "
                        f"with shape {tensor.shape}"
                    )

                    # Process tensor (example: just pass through, add your logic here)
                    processed_tensor = self._process_tensor(tensor)

                    # Send back processed tensor
                    for chunk_data in serialize_tensor_to_chunks(
                        processed_tensor,
                        chunk_size=self.chunk_size,
                        tensor_id=f"processed_{chunk.tensor_id}"
                    ):
                        t_id, c_num, data, total, is_last, meta = chunk_data

                        response_chunk = service_pb2.TensorChunk(
                            tensor_id=t_id,
                            chunk_number=c_num,
                            data=data,
                            total_chunks=total,
                            is_last=is_last
                        )

                        if meta:
                            response_chunk.metadata.update(meta)

                        logger.debug(
                            f"Sending chunk {c_num}/{total} for tensor {t_id}"
                        )

                        yield response_chunk

                    logger.info(
                        f"Sent processed tensor for {chunk.tensor_id} "
                        f"with shape {processed_tensor.shape}"
                    )

        except Exception as e:
            logger.error(f"Error in bidirectional streaming: {e}")
            await context.abort(
                grpc.StatusCode.INTERNAL,
                f"Error in bidirectional streaming: {e}"
            )

    def _get_tensor_to_send(
        self,
        index: int,
        parameters: dict
    ) -> torch.Tensor:
        """
        Get a tensor to send to the client.

        This is an example implementation that creates random tensors.
        Override this method or implement your own logic to provide tensors.

        Args:
            index: Tensor index
            parameters: Request parameters

        Returns:
            Tensor to send
        """
        # Example: create a random tensor
        # Parse shape from parameters if provided
        shape_str = parameters.get('shape', '10,10')
        try:
            shape = tuple(map(int, shape_str.split(',')))
        except ValueError:
            shape = (10, 10)

        return torch.randn(*shape)

    def _process_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Process a received tensor.

        This is an example implementation that just returns the tensor as-is.
        Override this method to implement your own tensor processing logic.

        Args:
            tensor: Input tensor

        Returns:
            Processed tensor
        """
        # Example: just return the tensor (echo)
        # Implement your own processing logic here
        return tensor


# Coroutines to be invoked when the event loop is shutting down.
_cleanup_coroutines = []


async def serve(
    host: str = "[::]",
    port: int = 50051,
    chunk_size: int = DEFAULT_CHUNK_SIZE
) -> None:
    """
    Start the async gRPC server.

    Args:
        host: Host to listen on
        port: Port to listen on
        chunk_size: Size of chunks for streaming tensors
    """
    server = grpc.aio.server()

    servicer = TensorServicer(chunk_size=chunk_size)
    service_pb2_grpc.add_ServiceServicer_to_server(servicer, server)

    listen_addr = f'{host}:{port}'
    server.add_insecure_port(listen_addr)

    logger.info(f"Starting server on {listen_addr}")
    await server.start()

    logger.info(f"Server listening on port {port}")
    logger.info(f"Chunk size: {chunk_size} bytes")

    async def server_graceful_shutdown():
        logging.info("Graceful shutdown...")
        # Shuts down the server with 0 seconds of grace period. During the
        # grace period, the server won't accept new connections and allow
        # existing RPCs to continue within the grace period.
        await server.stop(5.0)
        logging.info("Graceful shutdown complete")

    _cleanup_coroutines.append(server_graceful_shutdown())
    await server.wait_for_termination()
