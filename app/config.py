"""Konfigurasi runtime dari environment / file .env.

Modul ini memuat `.env` sekali saat diimpor, lalu menyediakan helper bertipe
untuk membaca nilai environment yang dipakai service.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import TypeVar

try:  # python-dotenv opsional saat menjalankan unit test tanpa instalasi penuh
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - fallback kecil jika dependency belum ada
    def load_dotenv(*_args: object, **_kwargs: object) -> bool:
        return False


load_dotenv()

T = TypeVar("T")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


MODEL_PATH: str = os.getenv("MODEL_PATH", "models/yolo26m.pt").strip()
SLOTS_PATH: str = os.getenv("SLOTS_PATH", "parking_slots.json").strip()
DEVICE: str = os.getenv("DEVICE", "cuda").strip().lower() or "cuda"
ALLOW_CPU_FALLBACK: bool = _env_bool("ALLOW_CPU_FALLBACK", True)
YOLO_IMGSZ: int = _env_int("YOLO_IMGSZ", 1280)
OCCUPANCY_IOA_THRESHOLD: float = _env_float("OCCUPANCY_IOA_THRESHOLD", 0.30)

SLOT_STATE_FRAMES: int = _env_int("SLOT_STATE_FRAMES", 5)
SLOT_OCCUPIED_FRAMES: int = _env_int("SLOT_OCCUPIED_FRAMES", SLOT_STATE_FRAMES)
SLOT_EMPTY_FRAMES: int = _env_int("SLOT_EMPTY_FRAMES", SLOT_STATE_FRAMES)

API_KEY: str = os.getenv("API_KEY", "").strip()
WEBHOOK_URL: str = os.getenv("WEBHOOK_URL", "").strip()
WEBHOOK_API_KEY: str = os.getenv("WEBHOOK_API_KEY", "").strip() or API_KEY
WEBHOOK_TIMEOUT: float = _env_float("WEBHOOK_TIMEOUT", 5.0)
WEBHOOK_RETRIES: int = max(0, _env_int("WEBHOOK_RETRIES", 3))


def _env_list(name: str, default: str = "*") -> list[str]:
    raw = os.getenv(name, default).strip()
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


CORS_ORIGINS: list[str] = _env_list("CORS_ORIGINS", "*")



def utc_now_iso() -> str:
    """Timestamp UTC ISO-8601 tanpa mikrodetik, selalu berakhiran ``Z``."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
