"""Detektor parkir: YOLO + pencocokan geometrik slot + temporal smoothing.

Peningkatan akurasi:
- Titik kontak tanah ``(x_center, y_max)`` sebagai tes utama occupancy.
- Fallback IoA vectorized antara bbox kendaraan dan polygon slot.
- Temporal buffer K-frame agar status slot tidak flicker.
- Confidence per slot dari deteksi kendaraan yang paling kuat.
- Inferensi di ``torch.no_grad()`` dengan CUDA jika tersedia.
"""

from __future__ import annotations

import logging
import os
import threading
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Sequence

import cv2
import numpy as np
import supervision as sv
import torch
from ultralytics import YOLO

from app import config, geometry
from app.slot_store import load_slots

logger = logging.getLogger(__name__)

_MODEL_CANDIDATES = [
    "models/yolo26m.pt",
    "models/yolo26n.pt",
    "models/yolo26s.pt",
    "models/yolo26x.pt",
    "models/yolov8m.pt",
    "models/yolov8n.pt",
]

# Kelas kendaraan pada dataset COCO: car(2), motorcycle(3), bus(5), truck(7)
TARGET_CLASSES: list[int] = [2, 3, 5, 7]

# Ambang confidence per-kelas: motor lebih sensitif, bus/truk lebih ketat.
CLASS_CONF: dict[int, float] = {
    2: 0.30,  # car
    3: 0.25,  # motorcycle
    5: 0.45,  # bus
    7: 0.45,  # truck
}
DEFAULT_CONF = 0.40
MIN_INFER_CONF = min(CLASS_CONF.values())

IMGSZ = config.YOLO_IMGSZ
IOA_THRESHOLD = config.OCCUPANCY_IOA_THRESHOLD

OCCUPIED_FRAMES = config.SLOT_OCCUPIED_FRAMES
EMPTY_FRAMES = config.SLOT_EMPTY_FRAMES
STATE_WINDOW = config.SLOT_STATE_FRAMES


def _find_available_model(preferred: str | None = None) -> str:
    """Pilih model: argumen -> env MODEL_PATH -> scan models/*.pt -> default."""
    candidates = [preferred, config.MODEL_PATH]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    for candidate in _MODEL_CANDIDATES:
        if os.path.exists(candidate):
            return candidate
    try:
        for path in sorted(Path("models").glob("*.pt")):
            return str(path)
    except OSError:
        pass
    return preferred or config.MODEL_PATH or "models/yolo26m.pt"


def filter_vehicle_detections(detections: sv.Detections, raw: bool = False) -> sv.Detections:
    """Filter deteksi ke kelas kendaraan dengan ambang confidence per-kelas."""
    if not len(detections):
        return detections
    class_ids = np.asarray(detections.class_id).astype(int)
    in_target = np.isin(class_ids, TARGET_CLASSES)
    if raw:
        return detections[in_target]
    thresholds = np.array(
        [CLASS_CONF.get(int(c), DEFAULT_CONF) for c in class_ids], dtype=np.float64
    )
    mask = in_target & (detections.confidence >= thresholds)
    return detections[mask]


class TemporalSmoother:
    """Temporal buffer per-slot dengan konfirmasi N frame berturut-turut."""

    def __init__(
        self,
        occupied_frames: int = OCCUPIED_FRAMES,
        empty_frames: int = EMPTY_FRAMES,
        window_size: int | None = None,
    ) -> None:
        self.occupied_frames = max(1, int(occupied_frames))
        self.empty_frames = max(1, int(empty_frames))
        self.window_size = max(
            1,
            int(window_size) if window_size is not None else max(self.occupied_frames, self.empty_frames),
        )
        self._state: dict[str, bool] = {}
        self._windows: dict[str, Deque[bool]] = {}
        self._confidence: dict[str, float] = {}

    @staticmethod
    def _confirmed(window: Deque[bool], target: bool, required: int) -> bool:
        if len(window) < required:
            return False
        return all(value is target for value in list(window)[-required:])

    def update(self, slot_id: str, observed: bool, confidence: float = 0.0) -> tuple[bool, float]:
        """Update satu slot dan kembalikan ``(state, confidence)`` yang stabil."""
        window = self._windows.get(slot_id)
        if window is None:
            window = deque(maxlen=self.window_size)
            self._windows[slot_id] = window
        window.append(observed)

        state = self._state.get(slot_id, False)
        if observed and confidence > 0:
            self._confidence[slot_id] = float(confidence)

        if not state and self._confirmed(window, True, self.occupied_frames):
            state = True
        elif state and self._confirmed(window, False, self.empty_frames):
            state = False
            self._confidence[slot_id] = 0.0

        self._state[slot_id] = state
        final_confidence = self._confidence.get(slot_id, float(confidence)) if state else 0.0
        return state, final_confidence

    def prune(self, valid_ids: set[str]) -> None:
        for key in list(self._state):
            if key not in valid_ids:
                self._state.pop(key, None)
                self._windows.pop(key, None)
                self._confidence.pop(key, None)

    def reset(self) -> None:
        self._state.clear()
        self._windows.clear()
        self._confidence.clear()


class ParkingDetector:
    """Wrapper YOLO + evaluasi occupancy slot yang thread-safe."""

    def __init__(self, model_path: str | None = None) -> None:
        self._lock = threading.RLock()
        self.device = self._resolve_device()
        self.cuda_available = bool(torch.cuda.is_available())

        chosen = _find_available_model(model_path)
        try:
            self.model = YOLO(chosen)
            self._move_model_to_device()
        except Exception as exc:
            fallback = "models/yolo26n.pt"
            logger.error("Gagal memuat model %s (%s). Coba fallback %s", chosen, exc, fallback)
            self.model = YOLO(fallback)
            self._move_model_to_device()
            chosen = fallback

        self.model_path_loaded = chosen
        self.target_classes = TARGET_CLASSES
        logger.info(
            "Detector siap: model=%s device=%s (cuda_available=%s)",
            chosen,
            self.device,
            self.cuda_available,
        )

        self._slots_cache: List[Dict[str, Any]] = []
        self._slots_mtime: float | None = None
        self._slots_loaded = False
        self._zone_cache: dict[str, sv.PolygonZone] = {}
        self._normalized_cache: np.ndarray | None = None
        self._normalized_cache_key: tuple[float | None, bytes] | None = None

        self.smoother = TemporalSmoother(
            occupied_frames=OCCUPIED_FRAMES,
            empty_frames=EMPTY_FRAMES,
            window_size=STATE_WINDOW,
        )

        self._homography: np.ndarray | None = None
        self._perspective_size: tuple[int, int] | None = None
        self._init_perspective()

        self.slots_path = Path(config.SLOTS_PATH)

    # ------------------------------------------------------------------ device
    def _resolve_device(self) -> str:
        requested = config.DEVICE or "cuda"
        cuda_available = bool(torch.cuda.is_available())

        if requested.startswith("cuda"):
            if cuda_available:
                return requested
            if not config.ALLOW_CPU_FALLBACK:
                raise RuntimeError(
                    "DEVICE meminta CUDA tetapi CUDA tidak tersedia. "
                    "Set ALLOW_CPU_FALLBACK=true untuk fallback ke CPU."
                )
            logger.warning("CUDA tidak tersedia; fallback ke CPU.")
            return "cpu"

        return requested

    def _move_model_to_device(self) -> None:
        try:
            self.model.to(self.device)
        except Exception:
            if self.device.startswith("cuda") and config.ALLOW_CPU_FALLBACK:
                logger.warning(
                    "Tidak bisa memindahkan model ke %s; fallback ke CPU.",
                    self.device,
                )
                self.device = "cpu"
                self.model.to(self.device)
            else:
                raise

    def cuda_status(self) -> dict[str, Any]:
        """Metadata CUDA untuk endpoint health."""
        info: dict[str, Any] = {
            "available": bool(torch.cuda.is_available()),
            "device_count": 0,
            "device_name": None,
            "memory_allocated_mb": None,
            "memory_reserved_mb": None,
        }
        if not info["available"]:
            return info
        try:
            info["device_count"] = int(torch.cuda.device_count())
            if info["device_count"] > 0:
                info["device_name"] = torch.cuda.get_device_name(0)
                info["memory_allocated_mb"] = round(
                    torch.cuda.memory_allocated(0) / (1024 ** 2), 2
                )
                info["memory_reserved_mb"] = round(
                    torch.cuda.memory_reserved(0) / (1024 ** 2), 2
                )
        except Exception as exc:  # pragma: no cover - bergantung driver CUDA
            logger.warning("Gagal membaca status CUDA: %s", exc)
        return info

    # ------------------------------------------------------------------ perspektif
    def _init_perspective(self) -> None:
        raw = os.getenv("PARKING_PERSPECTIVE")
        if not raw:
            return
        try:
            values = [float(v) for v in raw.replace(";", ",").split(",") if v.strip()]
            if len(values) != 8:
                raise ValueError("butuh 8 angka")
            src = np.array(values, dtype=np.float32).reshape(4, 2)
            dst, w, h = geometry.quad_to_rect_dst(src)
            self._homography = geometry.perspective_transform(src, dst)
            self._perspective_size = (w, h)
            logger.info("Normalisasi perspektif aktif: %dx%d", w, h)
        except Exception as exc:
            logger.warning("PARKING_PERSPECTIVE tidak valid (%s), perspektif diabaikan.", exc)

    def configure_perspective(
        self, src_quad: Sequence[Sequence[float]] | None, dst_size: tuple[int, int] | None = None
    ) -> None:
        """Aktifkan normalisasi top-down dari 4 titik sumber. None untuk mematikan."""
        with self._lock:
            if src_quad is None:
                self._homography = None
                self._perspective_size = None
            else:
                dst, w, h = geometry.quad_to_rect_dst(src_quad)
                if dst_size:
                    w, h = int(dst_size[0]), int(dst_size[1])
                    dst = [[0, 0], [w, 0], [w, h], [0, h]]
                self._homography = geometry.perspective_transform(src_quad, dst)
                self._perspective_size = (w, h)
            self._invalidate_normalized_cache()

    def _normalize_polygons(self, polygons: list[np.ndarray]) -> list[np.ndarray]:
        if self._homography is None:
            return polygons
        return [
            np.array(geometry.warp_polygon(p, self._homography), dtype=np.float64)
            for p in polygons
        ]

    def _normalize_points(self, points: np.ndarray) -> np.ndarray:
        if self._homography is None or points.size == 0:
            return points
        pts = points.astype(np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self._homography).reshape(-1, 2).astype(np.float64)

    def _normalize_boxes(self, xyxy: np.ndarray) -> np.ndarray:
        """Transformasi tiap bbox ke ruang perspektif, lalu ambil bbox pembatasnya."""
        if self._homography is None or xyxy.size == 0:
            return xyxy
        boxes = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
        corners = np.stack(
            [
                np.stack([boxes[:, 0], boxes[:, 1]], axis=1),
                np.stack([boxes[:, 2], boxes[:, 1]], axis=1),
                np.stack([boxes[:, 2], boxes[:, 3]], axis=1),
                np.stack([boxes[:, 0], boxes[:, 3]], axis=1),
            ],
            axis=1,
        )
        flat = corners.reshape(-1, 1, 2)
        warped = cv2.perspectiveTransform(flat, self._homography).reshape(-1, 4, 2)
        return np.stack(
            [
                warped[..., 0].min(axis=1),
                warped[..., 1].min(axis=1),
                warped[..., 0].max(axis=1),
                warped[..., 1].max(axis=1),
            ],
            axis=1,
        ).astype(np.float64)

    def _invalidate_normalized_cache(self) -> None:
        self._normalized_cache = None
        self._normalized_cache_key = None

    # ------------------------------------------------------------------ slot cache
    def _load_slots_if_changed(self) -> None:
        path = self.slots_path
        try:
            mtime = path.stat().st_mtime
        except FileNotFoundError:
            mtime = -1.0

        if self._slots_loaded and mtime == self._slots_mtime:
            return

        raw = load_slots(str(path))
        self._slots_cache = raw
        self._slots_mtime = mtime
        self._slots_loaded = True
        self._zone_cache.clear()
        self._invalidate_normalized_cache()
        self.smoother.prune({s["slot_id"] for s in raw})

    @property
    def parking_slots(self) -> dict[str, np.ndarray]:
        with self._lock:
            self._load_slots_if_changed()
            return {
                s["slot_id"]: np.array(s["polygon"], dtype=np.int32) for s in self._slots_cache
            }

    def get_slots_list(self) -> list[dict]:
        with self._lock:
            self._load_slots_if_changed()
            return [dict(s) for s in self._slots_cache]

    def _get_zone(self, polygon: np.ndarray, slot_id: str) -> sv.PolygonZone:
        key = f"{slot_id}:{polygon.tobytes()}"
        if key not in self._zone_cache:
            self._zone_cache[key] = sv.PolygonZone(polygon=polygon)
        return self._zone_cache[key]

    def reset_temporal(self) -> None:
        with self._lock:
            self.smoother.reset()

    def reload_slots(self, slots: list[dict] | None = None) -> None:
        """Paksa reload konfigurasi slot ke memori dan reset cache zona/normalisasi."""
        with self._lock:
            if slots is not None:
                self._slots_cache = [dict(s) for s in slots]
            else:
                self._slots_cache = load_slots(str(self.slots_path))
            try:
                self._slots_mtime = self.slots_path.stat().st_mtime
            except FileNotFoundError:
                self._slots_mtime = -1.0
            self._slots_loaded = True
            self._zone_cache.clear()
            self._invalidate_normalized_cache()
            self.smoother.reset()
            self.smoother.prune({s["slot_id"] for s in self._slots_cache})


    # ------------------------------------------------------------------ occupancy
    def _normalized_polygon_stack(self, polygons: list[np.ndarray]) -> np.ndarray:
        homography_key = self._homography.tobytes() if self._homography is not None else b""
        cache_key = (self._slots_mtime, homography_key)
        if self._normalized_cache is None or self._normalized_cache_key != cache_key:
            self._normalized_cache = np.asarray(
                self._normalize_polygons(polygons), dtype=np.float64
            )
            self._normalized_cache_key = cache_key
        return self._normalized_cache

    def _evaluate_occupancy(
        self, polygons: list[np.ndarray], detections: sv.Detections
    ) -> tuple[list[bool], list[float]]:
        """Hitung status + confidence semua slot secara vectorized."""
        n_slots = len(polygons)
        if n_slots == 0:
            return [], []

        if len(detections):
            grounds = geometry.ground_points(detections.xyxy)
            boxes = np.asarray(detections.xyxy, dtype=np.float64)
            confidences = np.asarray(detections.confidence, dtype=np.float64)
        else:
            grounds = np.zeros((0, 2), dtype=np.float64)
            boxes = np.zeros((0, 4), dtype=np.float64)
            confidences = np.zeros((0,), dtype=np.float64)

        norm_polys = self._normalized_polygon_stack(polygons)
        if len(grounds) == 0:
            return [False] * n_slots, [0.0] * n_slots

        norm_grounds = self._normalize_points(grounds)
        norm_boxes = self._normalize_boxes(boxes)

        ground_in = geometry.points_in_polygons(norm_grounds, norm_polys)  # (D, S)
        ioa = geometry.ioa_bbox_vs_polygons(norm_boxes, norm_polys)  # (D, S)
        matched = ground_in | (ioa >= IOA_THRESHOLD)

        occupied = matched.any(axis=0)
        weighted = np.where(matched, confidences[:, None], 0.0)
        per_slot_confidence = weighted.max(axis=0)
        return occupied.tolist(), per_slot_confidence.tolist()

    def process_frame(self, image: np.ndarray, *, use_temporal: bool = True) -> Dict[str, Any]:
        """Jalankan inferensi dan kembalikan status seluruh slot."""
        with self._lock:
            self._load_slots_if_changed()

            try:
                with torch.no_grad():
                    results = self.model(
                        image,
                        verbose=False,
                        device=self.device,
                        conf=MIN_INFER_CONF,
                        imgsz=IMGSZ,
                    )[0]
            except Exception as exc:
                logger.error("Inference gagal (%s)", exc)
                raise

            detections = sv.Detections.from_ultralytics(results)
            detections = filter_vehicle_detections(detections)

            polygons = [
                np.array(s["polygon"], dtype=np.float64) for s in self._slots_cache
            ]
            raw_status, raw_confidence = self._evaluate_occupancy(polygons, detections)

            slots_status: List[Dict[str, Any]] = []
            occupied_count = 0
            for slot, observed, observed_confidence in zip(
                self._slots_cache, raw_status, raw_confidence
            ):
                slot_id = slot["slot_id"]
                if use_temporal:
                    final, confidence = self.smoother.update(
                        slot_id, observed, observed_confidence
                    )
                else:
                    final, confidence = observed, observed_confidence

                if final:
                    occupied_count += 1
                slots_status.append(
                    {
                        "slot_id": slot_id,
                        "is_occupied": bool(final),
                        "confidence": round(float(confidence), 4),
                        "is_occupied_raw": bool(observed),
                        "polygon": slot["polygon"],
                    }
                )

            total = len(self._slots_cache)
            return {
                "timestamp": config.utc_now_iso(),
                "summary": {
                    "total_slots": total,
                    "occupied_slots": occupied_count,
                    "empty_slots": total - occupied_count,
                },
                "slots": [
                    {
                        "slot_id": s["slot_id"],
                        "is_occupied": s["is_occupied"],
                        "confidence": s["confidence"],
                    }
                    for s in slots_status
                ],
                "slots_detail": slots_status,
                "detections": detections,
            }

    def annotate(self, image: np.ndarray, result: Dict[str, Any] | None = None) -> np.ndarray:
        """Gambar polygon slot (merah=terisi, hijau=kosong) + bbox + titik tanah."""
        scene = image.copy()
        if result is None:
            result = self.process_frame(scene)

        status_by_slot = {
            slot["slot_id"]: slot["is_occupied"] for slot in result["slots_detail"]
        }

        with self._lock:
            self._load_slots_if_changed()
            slots = [dict(s) for s in self._slots_cache]

        for slot in slots:
            slot_id = slot["slot_id"]
            polygon = np.array(slot["polygon"], dtype=np.int32)
            zone = self._get_zone(polygon, slot_id)
            is_occupied = bool(status_by_slot.get(slot_id))
            color = sv.Color.RED if is_occupied else sv.Color.GREEN
            annotator = sv.PolygonZoneAnnotator(
                zone=zone, color=color, thickness=2, display_in_zone_count=False
            )
            scene = annotator.annotate(scene=scene)

        detections = result.get("detections")
        if detections is not None and len(detections):
            scene = sv.BoxAnnotator(thickness=2).annotate(scene=scene, detections=detections)
            grounds = geometry.ground_points(detections.xyxy)
            for gx, gy in grounds:
                cv2.circle(scene, (int(gx), int(gy)), 4, (0, 255, 255), -1)

        return scene
