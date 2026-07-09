"""Standalone platform job worker for production (Railway, Fly, etc.).

Dispatches segment/transcribe jobs to the inference API. Disable the in-process
worker on serverless API hosts with JOB_WORKER_ENABLED=false.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import signal

from backend.jobs.infrastructure.worker import worker_loop

logger = logging.getLogger(__name__)


async def _run() -> None:
    stop_event = asyncio.Event()

    def _request_stop(*_args: object) -> None:
        logger.info("platform job worker shutdown requested")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, _request_stop)

    logger.info("platform job worker started")
    await worker_loop(stop_event)
    logger.info("platform job worker stopped")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(_run())
