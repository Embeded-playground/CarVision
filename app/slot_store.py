"""Penyimpanan koordinat slot parkir dalam format JSON (parking_slots.json)."""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import time

from app import geometry

logger = logging.getLogger(__name__)

SLOTS_PATH: str = os.getenv("SLOTS_PATH", "parking_slots.json")

_SLOT_ID_RE = re.compile(r"^Slot_(\d+)$")


def _normalize_slot(slot: dict) -> dict | None:
    """Validasi & normalisasi satu entri slot. Return None jika tidak valid.

    Mendukung polygon 4 titik umum (perspective-aware), bukan hanya persegi:
    memeriksa convexitas, self-intersection, dan batas luas, serta
    menormalkan urutan vertex menjadi counter-clockwise (CCW).
    """
    slot_id = slot.get("slot_id")
    polygon = slot.get("polygon")

    if not isinstance(slot_id, str) or not slot_id.strip():
        logger.warning("Slot dilewati: slot_id kosong/invalid (%r)", slot_id)
        return None

    if not isinstance(polygon, (list, tuple)) or len(polygon) != 4:
        logger.warning("Slot %s dilewati: polygon harus terdiri dari 4 titik", slot_id)
        return None

    points: list[list[float]] = []
    for point in polygon:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            logger.warning("Slot %s dilewati: setiap titik harus [x, y]", slot_id)
            return None
        try:
            points.append([float(point[0]), float(point[1])])
        except (TypeError, ValueError):
            logger.warning("Slot %s dilewati: koordinat bukan angka", slot_id)
            return None

    ok, reason, normalized = geometry.validate_polygon(points)
    if not ok or normalized is None:
        logger.warning("Slot %s dilewati: %s", slot_id, reason)
        return None

    return {"slot_id": slot_id.strip(), "polygon": normalized}


def load_slots(path: str = SLOTS_PATH) -> list[dict]:
    """Baca slot dari JSON. Return [] jika file tidak ada/korup (tidak crash)."""
    if not os.path.isfile(path):
        logger.info("File %s belum ada, menganggap 0 slot.", path)
        return []

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        logger.error("File %s korup (%s). Menganggap 0 slot.", path, exc)
        return []

    raw_slots = data.get("slots") if isinstance(data, dict) else data
    if not isinstance(raw_slots, list):
        logger.error('Struktur %s tidak valid (harus {"slots": [...]}).', path)
        return []

    slots: list[dict] = []
    seen_ids: set[str] = set()
    for raw in raw_slots:
        if not isinstance(raw, dict):
            continue
        slot = _normalize_slot(raw)
        if slot is None:
            continue
        if slot["slot_id"] in seen_ids:
            logger.warning("ID ganda %s dilewati.", slot["slot_id"])
            continue
        seen_ids.add(slot["slot_id"])
        slots.append(slot)
    return slots


def save_slots(slots: list[dict], path: str = SLOTS_PATH) -> list[dict]:
    """Validasi dan tulis ulang seluruh slot ke JSON (atomik + retry khusus Windows)."""
    normalized = [s for s in (_normalize_slot(raw) for raw in slots) if s is not None]
    text = json.dumps({"slots": normalized}, indent=2)

    # Lewati penulisan bila isi file sudah identik (mengurangi lock churn di Windows)
    try:
        with open(path, "r", encoding="utf-8") as f:
            if f.read() == text:
                return normalized
    except OSError:
        pass

    directory = os.path.dirname(os.path.abspath(path))
    fd, tmp_path = tempfile.mkstemp(suffix=".json", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    last_exc: OSError | None = None
    for attempt in range(5):
        try:
            os.replace(tmp_path, path)
            last_exc = None
            break
        except OSError as exc:  # PermissionError (WinError 5) termasuk di sini
            last_exc = exc
            time.sleep(0.05 * (attempt + 1))

    if last_exc is not None:
        logger.warning("os.replace gagal (%s). Fallback tulis langsung ke %s.", last_exc, path)
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
        except OSError as exc:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise RuntimeError(f"Gagal menyimpan {path}: {exc}") from exc

    if os.path.exists(tmp_path):
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    logger.info("%d slot disimpan ke %s.", len(normalized), path)
    return normalized


def next_slot_id(slots: list[dict]) -> str:
    """Generate ID berikutnya (Slot_N) berdasarkan angka tertinggi yang ada."""
    max_n = 0
    for slot in slots:
        match = _SLOT_ID_RE.match(str(slot.get("slot_id", "")))
        if match:
            max_n = max(max_n, int(match.group(1)))
    return f"Slot_{max_n + 1}"
