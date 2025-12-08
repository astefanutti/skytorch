"""
Main entry point for the PyTorch tensor streaming gRPC server.
"""

import argparse
import asyncio
import logging
import os

from kpu.torch.server.server import serve, _cleanup_coroutines
from kpu.torch.server.serialization import DEFAULT_CHUNK_SIZE


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

loop = asyncio.get_event_loop()
try:
    loop.run_until_complete(
        serve(
            host=args.host,
            port=args.port,
            chunk_size=args.chunk_size
        )
    )
except KeyboardInterrupt:
    pass
finally:
    loop.run_until_complete(*_cleanup_coroutines)
    loop.close()
