from __future__ import annotations

import httpx
import pytest

from app.schemas import ChangedSlot, SlotStatus, Summary, WebhookPayload
from app.webhook import WebhookDispatcher


def _payload() -> WebhookPayload:
    return WebhookPayload(
        timestamp="2026-10-26T17:27:00Z",
        summary=Summary(total_slots=1, occupied_slots=1, empty_slots=0),
        slots=[SlotStatus(slot_id="Slot_1", is_occupied=True, confidence=0.92)],
        changed_slots=[
            ChangedSlot(
                slot_id="Slot_1",
                previous_is_occupied=False,
                is_occupied=True,
                confidence=0.92,
            )
        ],
    )


def test_disabled_webhook_does_not_enqueue() -> None:
    dispatcher = WebhookDispatcher(url="")
    assert dispatcher.enabled is False
    assert dispatcher.enqueue(_payload()) is False


@pytest.mark.asyncio
async def test_webhook_delivers_payload_with_api_key_header() -> None:
    received: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        received.append(
            {
                "headers": dict(request.headers),
                "body": request.content,
            }
        )
        return httpx.Response(200, json={"ok": True})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        dispatcher = WebhookDispatcher(
            url="http://backend.local/webhook",
            api_key="secret-key",
            client=client,
        )
        await dispatcher.start()
        assert dispatcher.enqueue(_payload()) is True
        assert dispatcher._queue is not None
        await dispatcher._queue.join()
        await dispatcher.close()

    assert len(received) == 1
    assert received[0]["headers"]["x-api-key"] == "secret-key"
    assert b"Slot_1" in received[0]["body"]


@pytest.mark.asyncio
async def test_webhook_retries_then_succeeds() -> None:
    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        if attempts["count"] == 1:
            return httpx.Response(500)
        return httpx.Response(204)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        dispatcher = WebhookDispatcher(
            url="http://backend.local/webhook",
            api_key="",
            retries=2,
            client=client,
        )
        await dispatcher.start()
        dispatcher.enqueue(_payload())
        assert dispatcher._queue is not None
        await dispatcher._queue.join()
        await dispatcher.close()

    assert attempts["count"] == 2
