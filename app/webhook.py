"""Dispatcher webhook asynchronous ke backend eksternal (mis. Go backend)."""

from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from typing import Any

import httpx

from app import config
from app.schemas import WebhookPayload

logger = logging.getLogger(__name__)


class WebhookDispatcher:
    """Queue + worker agar pengiriman webhook tidak memblokir request API."""

    def __init__(
        self,
        url: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
        retries: int | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.url = (url if url is not None else config.WEBHOOK_URL).strip()
        self.api_key = (api_key if api_key is not None else config.WEBHOOK_API_KEY).strip()
        self.timeout = float(timeout if timeout is not None else config.WEBHOOK_TIMEOUT)
        self.retries = max(0, int(retries if retries is not None else config.WEBHOOK_RETRIES))
        self._client = client
        self._owns_client = client is None
        self._queue: asyncio.Queue[WebhookPayload] | None = None
        self._worker: asyncio.Task[None] | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.url)

    async def start(self) -> None:
        if not self.enabled:
            logger.info("WEBHOOK_URL kosong; webhook dinonaktifkan.")
            return
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                follow_redirects=True,
            )
        self._queue = asyncio.Queue()
        self._worker = asyncio.create_task(self._run(), name="parking-webhook-worker")
        logger.info("Webhook dispatcher aktif -> %s", self.url)

    def enqueue(self, payload: WebhookPayload) -> bool:
        """Masukkan payload ke queue. Return False jika webhook nonaktif."""
        if self._queue is None:
            return False
        self._queue.put_nowait(payload)
        return True

    async def _run(self) -> None:
        assert self._queue is not None
        while True:
            payload = await self._queue.get()
            try:
                await self._deliver(payload)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pragma: no cover - jaring pengaman worker
                logger.error("Webhook worker error: %s", exc)
            finally:
                self._queue.task_done()

    async def _deliver(self, payload: WebhookPayload) -> None:
        if self._client is None:
            return

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key

        body = payload.model_dump(mode="json")
        last_error: str | None = None

        for attempt in range(self.retries + 1):
            try:
                response = await self._client.post(self.url, json=body, headers=headers)
                if 200 <= response.status_code < 300:
                    logger.info(
                        "Webhook terkirim: %d slot berubah (HTTP %s)",
                        len(payload.changed_slots),
                        response.status_code,
                    )
                    return
                last_error = f"HTTP {response.status_code}"
            except httpx.HTTPError as exc:
                last_error = str(exc)

            if attempt < self.retries:
                delay = min(0.5 * (2 ** attempt), 10.0)
                logger.warning(
                    "Webhook gagal (%s), retry %d/%d dalam %.1fs",
                    last_error,
                    attempt + 1,
                    self.retries,
                    delay,
                )
                await asyncio.sleep(delay)

        logger.error("Webhook gagal permanen setelah %d percobaan: %s", self.retries + 1, last_error)

    async def close(self) -> None:
        if self._queue is not None:
            try:
                await asyncio.wait_for(
                    self._queue.join(),
                    timeout=self.timeout * (self.retries + 1) + 5,
                )
            except asyncio.TimeoutError:
                logger.warning("Timeout menunggu queue webhook selesai.")

        if self._worker is not None:
            self._worker.cancel()
            with suppress(asyncio.CancelledError):
                await self._worker
            self._worker = None

        if self._client is not None and self._owns_client:
            await self._client.aclose()
        self._client = None
        self._queue = None

    def stats(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "url_configured": bool(self.url),
            "queued": self._queue.qsize() if self._queue is not None else 0,
            "retries": self.retries,
            "timeout": self.timeout,
        }
