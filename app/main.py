"""FastAPI service untuk deteksi slot parkir dan integrasi webhook."""

from __future__ import annotations

import base64
import logging
import secrets
from contextlib import asynccontextmanager
from typing import AsyncIterator

import cv2
import numpy as np
from fastapi import APIRouter, Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from starlette.concurrency import run_in_threadpool

from app import config, geometry, slot_store
from app.detector import ParkingDetector
from app.schemas import (
    CudaStatus,
    DetectResponse,
    HealthPayload,
    SlotConfig,
    SlotMutationResponse,
    SlotStatus,
    SlotsListResponse,
    SlotsPayload,
    StatusPayload,
    Summary,
    WebhookPayload,
)
from app.state import SlotStatusStore
from app.webhook import WebhookDispatcher

logger = logging.getLogger(__name__)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _get_detector() -> ParkingDetector:
    try:
        return ParkingDetector(model_path=config.MODEL_PATH)
    except Exception as exc:
        logger.error("Gagal inisialisasi detector (%s); fallback ke yolo26n.", exc)
        return ParkingDetector(model_path="models/yolo26n.pt")


async def require_api_key(x_api_key: str | None = Depends(api_key_header)) -> None:
    """Validasi header X-API-Key jika API_KEY dikonfigurasi."""
    expected = config.API_KEY
    if not expected:
        return
    if not x_api_key or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "X-API-Key"},
        )


def get_detector(request: Request) -> ParkingDetector:
    return request.app.state.detector


def get_store(request: Request) -> SlotStatusStore:
    return request.app.state.status_store


def get_webhook(request: Request) -> WebhookDispatcher:
    return request.app.state.webhook


def validate_slot_definition(slot_id: str, polygon: list[list[int]]) -> list[list[int]]:
    """Validasi ketat polygon 4-titik (non-convex, self-intersect, koordinat invalid)."""
    if not isinstance(polygon, (list, tuple)) or len(polygon) != 4:
        raise HTTPException(
            status_code=422,
            detail=f"Slot '{slot_id}' tidak valid: polygon harus memiliki 4 titik.",
        )
    for point in polygon:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise HTTPException(
                status_code=422,
                detail=f"Slot '{slot_id}' tidak valid: setiap titik harus berupa koordinat [x, y].",
            )
        try:
            float(point[0])
            float(point[1])
        except (TypeError, ValueError):
            raise HTTPException(
                status_code=422,
                detail=f"Slot '{slot_id}' tidak valid: koordinat harus berupa angka.",
            )
    ok, reason, normalized = geometry.validate_polygon(polygon)
    if not ok or normalized is None:
        raise HTTPException(
            status_code=422,
            detail=f"Slot '{slot_id}' tidak valid: {reason}.",
        )
    return normalized


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    detector = _get_detector()
    store = SlotStatusStore()
    store.initialize(detector.get_slots_list())
    webhook = WebhookDispatcher()
    await webhook.start()

    app.state.detector = detector
    app.state.status_store = store
    app.state.webhook = webhook

    logger.info(
        "Service siap: model=%s device=%s auth=%s webhook=%s",
        detector.model_path_loaded,
        detector.device,
        bool(config.API_KEY),
        webhook.enabled,
    )
    if not config.API_KEY:
        logger.warning("API_KEY kosong; endpoint /api/v1/* terbuka tanpa autentikasi.")

    try:
        yield
    finally:
        await webhook.close()


app = FastAPI(
    title="Parking Vision Service",
    version="2.0.0",
    description=(
        "Deteksi slot parkir berbasis YOLO dengan REST API, autentikasi API key, "
        "CORS middleware, dan outbound webhook untuk integrasi backend/Web UI."
    ),
    lifespan=lifespan,
)

# Konfigurasi CORS Middleware untuk integrasi Web UI
cors_origins = config.CORS_ORIGINS
allow_all_origins = cors_origins == ["*"] or "*" in cors_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if allow_all_origins else cors_origins,
    allow_credentials=not allow_all_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

router = APIRouter(prefix="/api/v1", dependencies=[Depends(require_api_key)])


@app.get("/", tags=["system"])
def liveness() -> dict[str, str]:
    """Liveness sederhana tanpa autentikasi (untuk load balancer)."""
    return {"status": "online", "service": "Parking Vision Service"}


@router.get("/health", response_model=HealthPayload, tags=["system"])
def health_check(
    detector: ParkingDetector = Depends(get_detector),
    store: SlotStatusStore = Depends(get_store),
    webhook: WebhookDispatcher = Depends(get_webhook),
) -> HealthPayload:
    """Health check service, model, GPU/CUDA, auth, dan webhook."""
    cuda = detector.cuda_status()
    return HealthPayload(
        status="ok" if detector.device.startswith("cuda") else "degraded",
        timestamp=config.utc_now_iso(),
        service="parking-vision-service",
        model_loaded=detector.model_path_loaded,
        device=detector.device,
        cuda=CudaStatus(**cuda),
        auth_enabled=bool(config.API_KEY),
        webhook_configured=webhook.enabled,
        last_inference=store.last_inference,
    )


@router.get("/slots/status", response_model=StatusPayload, tags=["slots"])
def get_slot_status(store: SlotStatusStore = Depends(get_store)) -> StatusPayload:
    """Snapshot status slot terakhir (diperbarui oleh POST /detect)."""
    return store.snapshot()


@router.get(
    "/slots",
    response_model=SlotsListResponse,
    tags=["slots"],
    summary="Ambil seluruh konfigurasi polygon slot",
)
def get_slots(detector: ParkingDetector = Depends(get_detector)) -> SlotsListResponse:
    """Mengembalikan daftar semua slot parkir beserta koordinat polygon 4-titik."""
    raw = detector.get_slots_list()
    slots = [SlotConfig(slot_id=s["slot_id"], polygon=s["polygon"]) for s in raw]
    return SlotsListResponse(success=True, total=len(slots), slots=slots)


@router.get(
    "/slots/{slot_id}",
    response_model=SlotConfig,
    tags=["slots"],
    summary="Ambil konfigurasi satu slot",
)
def get_slot(slot_id: str, detector: ParkingDetector = Depends(get_detector)) -> SlotConfig:
    """Mengambil konfigurasi polygon satu slot berdasarkan slot_id."""
    raw = detector.get_slots_list()
    found = next((s for s in raw if s["slot_id"] == slot_id), None)
    if not found:
        raise HTTPException(status_code=404, detail=f"Slot '{slot_id}' tidak ditemukan.")
    return SlotConfig(slot_id=found["slot_id"], polygon=found["polygon"])


@router.post(
    "/slots",
    response_model=SlotMutationResponse,
    status_code=201,
    tags=["slots"],
    summary="Tambah satu slot baru",
)
def create_slot(
    slot: SlotConfig,
    detector: ParkingDetector = Depends(get_detector),
    store: SlotStatusStore = Depends(get_store),
) -> SlotMutationResponse:
    """Menambahkan satu slot baru ke konfigurasi dengan validasi geometri."""
    current = detector.get_slots_list()
    if any(s["slot_id"] == slot.slot_id for s in current):
        raise HTTPException(status_code=409, detail=f"Slot '{slot.slot_id}' sudah ada.")
    normalized = validate_slot_definition(slot.slot_id, slot.polygon)
    new_entry = {"slot_id": slot.slot_id, "polygon": normalized}
    updated = current + [new_entry]
    saved = slot_store.save_slots(updated)
    detector.reload_slots(saved)
    store.reset(saved)
    return SlotMutationResponse(
        success=True,
        message=f"Slot '{slot.slot_id}' berhasil ditambahkan.",
        slot=SlotConfig(slot_id=slot.slot_id, polygon=normalized),
        slots=[SlotConfig(slot_id=s["slot_id"], polygon=s["polygon"]) for s in saved],
    )


@router.put(
    "/slots/{slot_id}",
    response_model=SlotMutationResponse,
    tags=["slots"],
    summary="Perbarui satu slot yang ada",
)
def update_slot(
    slot_id: str,
    slot_in: SlotConfig,
    detector: ParkingDetector = Depends(get_detector),
    store: SlotStatusStore = Depends(get_store),
) -> SlotMutationResponse:
    """Memperbarui polygon satu slot yang sudah ada."""
    current = detector.get_slots_list()
    index = next((i for i, s in enumerate(current) if s["slot_id"] == slot_id), None)
    if index is None:
        raise HTTPException(status_code=404, detail=f"Slot '{slot_id}' tidak ditemukan.")
    if slot_in.slot_id != slot_id and any(s["slot_id"] == slot_in.slot_id for s in current):
        raise HTTPException(status_code=409, detail=f"Slot ID '{slot_in.slot_id}' sudah digunakan.")
    normalized = validate_slot_definition(slot_in.slot_id, slot_in.polygon)
    updated = list(current)
    updated[index] = {"slot_id": slot_in.slot_id, "polygon": normalized}
    saved = slot_store.save_slots(updated)
    detector.reload_slots(saved)
    store.reset(saved)
    return SlotMutationResponse(
        success=True,
        message=f"Slot '{slot_id}' berhasil diperbarui.",
        slot=SlotConfig(slot_id=slot_in.slot_id, polygon=normalized),
        slots=[SlotConfig(slot_id=s["slot_id"], polygon=s["polygon"]) for s in saved],
    )


@router.delete(
    "/slots/{slot_id}",
    response_model=SlotMutationResponse,
    tags=["slots"],
    summary="Hapus satu slot",
)
def delete_single_slot(
    slot_id: str,
    detector: ParkingDetector = Depends(get_detector),
    store: SlotStatusStore = Depends(get_store),
) -> SlotMutationResponse:
    """Menghapus satu slot berdasarkan slot_id."""
    current = detector.get_slots_list()
    if not any(s["slot_id"] == slot_id for s in current):
        raise HTTPException(status_code=404, detail=f"Slot '{slot_id}' tidak ditemukan.")
    updated = [s for s in current if s["slot_id"] != slot_id]
    saved = slot_store.save_slots(updated)
    detector.reload_slots(saved)
    store.reset(saved)
    return SlotMutationResponse(
        success=True,
        message=f"Slot '{slot_id}' berhasil dihapus.",
        slots=[SlotConfig(slot_id=s["slot_id"], polygon=s["polygon"]) for s in saved],
    )


@router.put(
    "/slots",
    response_model=SlotMutationResponse,
    tags=["slots"],
    summary="Overwrite seluruh konfigurasi slot",
)
def put_slots(
    payload: SlotsPayload,
    detector: ParkingDetector = Depends(get_detector),
    store: SlotStatusStore = Depends(get_store),
) -> SlotMutationResponse:
    """Mengganti seluruh konfigurasi slot parkir secara atomik dengan validasi geometri."""
    seen_ids: set[str] = set()
    validated_slots: list[dict] = []
    for slot in payload.slots:
        if slot.slot_id in seen_ids:
            raise HTTPException(
                status_code=422,
                detail=f"Duplikat slot_id terdeteksi: '{slot.slot_id}'.",
            )
        seen_ids.add(slot.slot_id)
        normalized = validate_slot_definition(slot.slot_id, slot.polygon)
        validated_slots.append({"slot_id": slot.slot_id, "polygon": normalized})

    saved = slot_store.save_slots(validated_slots)
    detector.reload_slots(saved)
    store.reset(saved)
    return SlotMutationResponse(
        success=True,
        message=f"{len(saved)} slot berhasil disimpan.",
        slots=[SlotConfig(slot_id=s["slot_id"], polygon=s["polygon"]) for s in saved],
    )


@router.delete(
    "/slots",
    response_model=SlotMutationResponse,
    tags=["slots"],
    summary="Hapus semua slot",
)
def delete_slots(
    detector: ParkingDetector = Depends(get_detector),
    store: SlotStatusStore = Depends(get_store),
) -> SlotMutationResponse:
    """Menghapus semua konfigurasi slot parkir."""
    saved = slot_store.save_slots([])
    detector.reload_slots(saved)
    store.reset(saved)
    return SlotMutationResponse(
        success=True,
        message="Semua slot berhasil dihapus.",
        slots=[],
    )


@router.post(
    "/detect",
    response_model=DetectResponse,
    response_model_exclude_none=True,
    tags=["detection"],
)
async def detect_parking(
    file: UploadFile = File(...),
    annotated: bool = Query(False, description="Sertakan gambar anotasi base64"),
    use_temporal: bool = Query(
        True,
        description="Gunakan temporal smoothing (default true untuk aliran frame)",
    ),
    detector: ParkingDetector = Depends(get_detector),
    store: SlotStatusStore = Depends(get_store),
    webhook: WebhookDispatcher = Depends(get_webhook),
) -> DetectResponse:
    """Terima frame gambar, jalankan inference, update cache, dan trigger webhook."""
    content_type = file.content_type or ""
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File harus berupa gambar")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="File gambar kosong")

    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Gambar corrupt atau tidak valid")

    try:
        result = await run_in_threadpool(
            detector.process_frame, image, use_temporal=use_temporal
        )
    except Exception as exc:
        logger.exception("Deteksi gagal")
        raise HTTPException(status_code=500, detail="Deteksi gagal") from exc

    changes = store.update(result["slots"], timestamp=result["timestamp"])
    if changes:
        snapshot = store.snapshot()
        webhook.enqueue(
            WebhookPayload(
                timestamp=snapshot.timestamp,
                summary=snapshot.summary,
                slots=snapshot.slots,
                changed_slots=changes,
            )
        )

    response = DetectResponse(
        timestamp=result["timestamp"],
        summary=Summary(**result["summary"]),
        slots=[SlotStatus(**slot) for slot in result["slots"]],
    )

    if annotated:
        annotated_image = await run_in_threadpool(detector.annotate, image, result)
        success, buffer = await run_in_threadpool(cv2.imencode, ".jpg", annotated_image)
        if success:
            encoded = base64.b64encode(buffer).decode("utf-8")
            response.image_annotated = f"data:image/jpeg;base64,{encoded}"

    return response


app.include_router(router)

