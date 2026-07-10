"""ASGI admission middleware shared by the inference API and local helper."""

from __future__ import annotations

import asyncio
import secrets
import time
from collections import deque
from collections.abc import Awaitable, Callable

from inference.admission import REQUEST_LIMIT_ERROR
from inference.contracts.webhooks import INFERENCE_SERVICE_SECRET_HEADER

ASGIApp = Callable[
    [dict, Callable[[], Awaitable[dict]], Callable[[dict], Awaitable[None]]],
    Awaitable[None],
]


async def _send_error(send: Callable[[dict], Awaitable[None]], status: int, detail: str) -> None:
    body = ('{"detail":"' + detail + '"}').encode()
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


class RequestBodyLimitMiddleware:
    """Bound request bytes before FastAPI parses JSON or base64 content."""

    def __init__(self, app: ASGIApp, *, max_body_bytes: int) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(
        self,
        scope: dict,
        receive: Callable[[], Awaitable[dict]],
        send: Callable[[dict], Awaitable[None]],
    ) -> None:
        if scope["type"] != "http" or scope["method"] not in {"POST", "PUT", "PATCH"}:
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        content_length = headers.get(b"content-length")
        if content_length is not None:
            try:
                if int(content_length) > self.max_body_bytes:
                    await _send_error(send, 413, REQUEST_LIMIT_ERROR)
                    return
            except ValueError:
                await _send_error(send, 400, REQUEST_LIMIT_ERROR)
                return

        body = bytearray()
        while True:
            message = await receive()
            if message["type"] != "http.request":
                await self.app(scope, receive, send)
                return
            body.extend(message.get("body", b""))
            if len(body) > self.max_body_bytes:
                await _send_error(send, 413, REQUEST_LIMIT_ERROR)
                return
            if not message.get("more_body", False):
                break

        delivered = False

        async def replay_body() -> dict:
            nonlocal delivered
            if delivered:
                return {"type": "http.disconnect"}
            delivered = True
            return {"type": "http.request", "body": bytes(body), "more_body": False}

        await self.app(scope, replay_body, send)


class ServiceRateLimitMiddleware:
    """Simple per-process sliding-window limiter for expensive inference routes."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        requests_per_minute: int,
        service_secret: str | None = None,
        limit_only_authenticated_service_requests: bool = False,
    ) -> None:
        self.app = app
        self.requests_per_minute = requests_per_minute
        self.service_secret = service_secret
        self.limit_only_authenticated_service_requests = limit_only_authenticated_service_requests
        self._requests: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def __call__(
        self,
        scope: dict,
        receive: Callable[[], Awaitable[dict]],
        send: Callable[[dict], Awaitable[None]],
    ) -> None:
        if scope["type"] != "http" or scope["method"] != "POST":
            await self.app(scope, receive, send)
            return

        if self.limit_only_authenticated_service_requests:
            supplied = dict(scope.get("headers", [])).get(
                INFERENCE_SERVICE_SECRET_HEADER.encode().lower()
            )
            if (
                self.service_secret is None
                or supplied is None
                or not secrets.compare_digest(
                    supplied.decode("latin-1"),
                    self.service_secret,
                )
            ):
                await self.app(scope, receive, send)
                return

        now = time.monotonic()
        async with self._lock:
            cutoff = now - 60.0
            while self._requests and self._requests[0] <= cutoff:
                self._requests.popleft()
            if len(self._requests) >= self.requests_per_minute:
                await _send_error(send, 429, "Inference service is busy; retry later")
                return
            self._requests.append(now)
        await self.app(scope, receive, send)
