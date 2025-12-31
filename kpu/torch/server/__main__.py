"""
Main entry point for the PyTorch tensor streaming gRPC server.
"""

import argparse
import asyncio
import logging
import os
import signal

import grpc

from kpu.torch.server.server import serve, graceful_shutdown
from kpu.torch.server.serialization import DEFAULT_CHUNK_SIZE
from kpu.server.metrics.source import load_metrics_source, list_metrics_source_names


parser = argparse.ArgumentParser(
    description='KPU PyTorch Server',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument(
    '--port',
    type=int,
    default=int(os.environ.get('KPU_PORT', 50051)),
    help='Port to listen on'
)

parser.add_argument(
    '--host',
    type=str,
    default=os.environ.get('KPU_HOST', '[::]'),
    help='Host address to bind to (use [::] for all interfaces)'
)

parser.add_argument(
    '--chunk-size',
    type=int,
    default=int(os.environ.get('KPU_CHUNK_SIZE', DEFAULT_CHUNK_SIZE)),
    help='Size of chunks for streaming tensors (in bytes)'
)

parser.add_argument(
    '--log-level',
    type=str,
    default=os.environ.get('KPU_LOG_LEVEL', 'INFO'),
    choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
    help='Logging level'
)

parser.add_argument(
    '--metrics-sources',
    type=str,
    nargs='*',
    default=os.environ.get('KPU_METRICS_SOURCES', '').split(',') if os.environ.get('KPU_METRICS_SOURCES', '') else [],
    help=f'Metrics sources to enable (available: {", ".join(list_metrics_source_names())})'
)

args = parser.parse_args()

logging.basicConfig(
    level=getattr(logging, args.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

logger = logging.getLogger(__name__)
logger.info(f"Starting KPU PyTorch Server")
logger.info(f"  Port: {args.port}")
logger.info(f"  Host: {args.host}")
logger.info(f"  Chunk Size: {args.chunk_size} bytes ({args.chunk_size / 1024 / 1024:.2f} MB)")
logger.info(f"  Log Level: {args.log_level}")
logger.info(f"  Metrics Sources: {', '.join(args.metrics_sources) if args.metrics_sources else 'none'}")

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
server = grpc.aio.server()

# Initialize metrics sources based on CLI configuration
metrics_sources = []
for source_name in args.metrics_sources:
    source = load_metrics_source(source_name)
    if source is not None:
        if source.is_available():
            metrics_sources.append(source)
            logger.info(f"Enabled metrics source: {source_name}")
        else:
            logger.warning(f"Metrics source '{source_name}' is not available")

for sig in (signal.SIGINT, signal.SIGTERM):
    loop.add_signal_handler(sig, lambda: asyncio.ensure_future(graceful_shutdown(server, metrics_sources, grace=5.0)))

try:
    loop.run_until_complete(
        serve(
            server,
            host=args.host,
            port=args.port,
            chunk_size=args.chunk_size,
            metrics_sources=metrics_sources,
        )
    )
finally:
    loop.run_until_complete(server.wait_for_termination())
    loop.close()
