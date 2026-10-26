"""Cache status slot terakhir yang dibagikan ke endpoint API dan webhook."""

from __future__ import annotations

import threading
from collections.abc import Iterable, Sequence
from typing import Any

from app import config
from app.schemas import ChangedSlot, SlotStatus, StatusPayload, Summary


class SlotStatusStore:
    """Thread-safe snapshot status slot terakhir."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._slots: dict[str, SlotStatus] = {}
        self._timestamp: str = config.utc_now_iso()
        self._last_inference: str | None = None

    def initialize(self, slots: Iterable[dict[str, Any]]) -> None:
        """Inisialisasi semua slot sebagai EMPTY (dipakai saat startup/reset)."""
        with self._lock:
            self._slots = {
                str(slot["slot_id"]): SlotStatus(
                    slot_id=str(slot["slot_id"]),
                    is_occupied=False,
                    confidence=0.0,
                )
                for slot in slots
            }
            self._timestamp = config.utc_now_iso()
            self._last_inference = None

    def reset(self, slots: Iterable[dict[str, Any]]) -> None:
        self.initialize(slots)

    def update(
        self,
        slot_results: Sequence[dict[str, Any]],
        timestamp: str | None = None,
    ) -> list[ChangedSlot]:
        """Simpan snapshot baru dan kembalikan slot yang berubah statusnya."""
        with self._lock:
            stamped_at = timestamp or config.utc_now_iso()
            previous = self._slots
            next_slots: dict[str, SlotStatus] = {}
            changes: list[ChangedSlot] = []

            for item in slot_results:
                slot_id = str(item["slot_id"])
                status = SlotStatus(
                    slot_id=slot_id,
                    is_occupied=bool(item["is_occupied"]),
                    confidence=float(item.get("confidence", 0.0)),
                )
                next_slots[slot_id] = status

                previous_status = previous.get(slot_id)
                previous_occupied = (
                    previous_status.is_occupied if previous_status is not None else False
                )
                if previous_occupied != status.is_occupied:
                    changes.append(
                        ChangedSlot(
                            slot_id=slot_id,
                            previous_is_occupied=previous_occupied,
                            is_occupied=status.is_occupied,
                            confidence=status.confidence,
                        )
                    )

            self._slots = next_slots
            self._timestamp = stamped_at
            self._last_inference = stamped_at
            return changes

    def snapshot(self) -> StatusPayload:
        """Ambil salinan snapshot publik."""
        with self._lock:
            slots = [
                SlotStatus(
                    slot_id=status.slot_id,
                    is_occupied=status.is_occupied,
                    confidence=status.confidence,
                )
                for status in self._slots.values()
            ]
            occupied = sum(1 for slot in slots if slot.is_occupied)
            total = len(slots)
            return StatusPayload(
                timestamp=self._timestamp,
                summary=Summary(
                    total_slots=total,
                    occupied_slots=occupied,
                    empty_slots=total - occupied,
                ),
                slots=slots,
                last_inference=self._last_inference,
            )


    @property
    def last_inference(self) -> str | None:
        with self._lock:
            return self._last_inference

    @property
    def total_slots(self) -> int:
        with self._lock:
            return len(self._slots)
