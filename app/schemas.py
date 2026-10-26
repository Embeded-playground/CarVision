"""Model Pydantic untuk kontrak API dan webhook."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SlotStatus(BaseModel):
    slot_id: str
    is_occupied: bool
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class Summary(BaseModel):
    total_slots: int = Field(ge=0)
    occupied_slots: int = Field(ge=0)
    empty_slots: int = Field(ge=0)


class StatusPayload(BaseModel):
    timestamp: str
    summary: Summary
    slots: list[SlotStatus]
    last_inference: str | None = None


class SlotConfig(BaseModel):
    slot_id: str = Field(..., description="ID unik slot parkir, mis. Slot_1")
    polygon: list[list[int]] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="4 koordinat vertex polygon [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]",
    )


class SlotsPayload(BaseModel):
    slots: list[SlotConfig] = Field(..., description="Daftar konfigurasi slot")


class SlotsListResponse(BaseModel):
    success: bool = True
    total: int = Field(ge=0, description="Total slot yang terdaftar")
    slots: list[SlotConfig]


class SlotMutationResponse(BaseModel):
    success: bool = True
    message: str
    slot: SlotConfig | None = None
    slots: list[SlotConfig] | None = None



class ChangedSlot(BaseModel):
    slot_id: str
    previous_is_occupied: bool
    is_occupied: bool
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class WebhookPayload(StatusPayload):
    event: str = "slot_state_changed"
    changed_slots: list[ChangedSlot]


class DetectResponse(StatusPayload):
    image_annotated: str | None = None


class CudaStatus(BaseModel):
    available: bool
    device_count: int = 0
    device_name: str | None = None
    memory_allocated_mb: float | None = None
    memory_reserved_mb: float | None = None


class HealthPayload(BaseModel):
    status: str
    timestamp: str
    service: str
    model_loaded: str
    device: str
    cuda: CudaStatus
    auth_enabled: bool
    webhook_configured: bool
    last_inference: str | None = None
