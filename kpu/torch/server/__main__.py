import asyncio
import logging
from kpu.torch.server.server import serve, _cleanup_coroutines

"""
Main entry point for the server.
"""

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

loop = asyncio.get_event_loop()
try:
    loop.run_until_complete(serve())
except KeyboardInterrupt:
    pass
finally:
    loop.run_until_complete(*_cleanup_coroutines)
    loop.close()
