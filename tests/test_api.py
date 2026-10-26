from __future__ import annotations

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.main as main


class FakeDetector:
    model_path_loaded = "fake-model.pt"
    device = "cuda"

    def __init__(self) -> None:
        self.slots = [
            {"slot_id": "Slot_1", "polygon": [[0, 0], [10, 0], [10, 10], [0, 10]]},
            {"slot_id": "Slot_2", "polygon": [[20, 0], [30, 0], [30, 10], [20, 10]]},
        ]
        self.occupied = False
        self.use_temporal_seen: bool | None = None

    def get_slots_list(self) -> list[dict]:
        return [dict(slot) for slot in self.slots]

    def cuda_status(self) -> dict:
        return {
            "available": True,
            "device_count": 1,
            "device_name": "Fake GPU",
            "memory_allocated_mb": 12.5,
            "memory_reserved_mb": 20.0,
        }

    def reset_temporal(self) -> None:
        pass

    def reload_slots(self, slots: list[dict] | None = None) -> None:
        if slots is not None:
            self.slots = [dict(s) for s in slots]


    def process_frame(self, image: np.ndarray, *, use_temporal: bool = True) -> dict:
        self.use_temporal_seen = use_temporal
        occupied = 1 if self.occupied else 0
        slots = []
        for index, slot in enumerate(self.slots):
            is_occupied = index == 0 and self.occupied
            slots.append(
                {
                    "slot_id": slot["slot_id"],
                    "is_occupied": is_occupied,
                    "confidence": 0.92 if is_occupied else 0.0,
                }
            )
        detail = [
            {
                **slot,
                "is_occupied_raw": slot["is_occupied"],
                "polygon": next(
                    s["polygon"] for s in self.slots if s["slot_id"] == slot["slot_id"]
                ),
            }
            for slot in slots
        ]
        return {
            "timestamp": "2026-10-26T17:27:00Z",
            "summary": {
                "total_slots": len(slots),
                "occupied_slots": occupied,
                "empty_slots": len(slots) - occupied,
            },
            "slots": slots,
            "slots_detail": detail,
            "detections": None,
        }

    def annotate(self, image: np.ndarray, result: dict | None = None) -> np.ndarray:
        return image


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch):
    detector = FakeDetector()
    monkeypatch.setattr(main, "_get_detector", lambda: detector)
    monkeypatch.setattr(main.config, "API_KEY", "test-key")
    monkeypatch.setattr(main.config, "WEBHOOK_URL", "")
    with TestClient(main.app) as test_client:
        yield test_client, detector


def _auth() -> dict[str, str]:
    return {"X-API-Key": "test-key"}


def _image_bytes() -> bytes:
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    success, buffer = cv2.imencode(".jpg", image)
    assert success
    return buffer.tobytes()


def test_health_requires_api_key(client) -> None:
    test_client, _ = client
    assert test_client.get("/api/v1/health").status_code == 401

    response = test_client.get("/api/v1/health", headers=_auth())
    assert response.status_code == 200
    payload = response.json()
    assert payload["device"] == "cuda"
    assert payload["cuda"]["available"] is True
    assert payload["auth_enabled"] is True


def test_status_starts_empty(client) -> None:
    test_client, _ = client
    response = test_client.get("/api/v1/slots/status", headers=_auth())
    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"] == {
        "total_slots": 2,
        "occupied_slots": 0,
        "empty_slots": 2,
    }
    assert all(slot["confidence"] == 0.0 for slot in payload["slots"])


def test_detect_updates_status_and_returns_schema(client) -> None:
    test_client, detector = client
    detector.occupied = True

    response = test_client.post(
        "/api/v1/detect",
        files={"file": ("frame.jpg", _image_bytes(), "image/jpeg")},
        headers=_auth(),
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["timestamp"] == "2026-10-26T17:27:00Z"
    assert payload["summary"] == {
        "total_slots": 2,
        "occupied_slots": 1,
        "empty_slots": 1,
    }
    assert payload["slots"][0] == {
        "slot_id": "Slot_1",
        "is_occupied": True,
        "confidence": 0.92,
    }
    assert detector.use_temporal_seen is True

    status = test_client.get("/api/v1/slots/status", headers=_auth()).json()
    assert status["summary"]["occupied_slots"] == 1


def test_detect_with_annotation(client) -> None:
    test_client, _ = client
    response = test_client.post(
        "/api/v1/detect?annotated=true&use_temporal=false",
        files={"file": ("frame.jpg", _image_bytes(), "image/jpeg")},
        headers=_auth(),
    )
    assert response.status_code == 200
    assert response.json()["image_annotated"].startswith("data:image/jpeg;base64,")


def test_detect_rejects_non_image(client) -> None:
    test_client, _ = client
    response = test_client.post(
        "/api/v1/detect",
        files={"file": ("frame.txt", b"not an image", "text/plain")},
        headers=_auth(),
    )
    assert response.status_code == 400


def test_cors_headers_present(client) -> None:
    test_client, _ = client
    response = test_client.options(
        "/api/v1/slots/status",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "*"


def test_slots_crud_lifecycle(client, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    test_client, detector = client
    slots_file = str(tmp_path / "test_slots.json")
    monkeypatch.setattr(main.slot_store, "SLOTS_PATH", slots_file)
    detector.slots_path = tmp_path / "test_slots.json"

    # GET all slots
    res = test_client.get("/api/v1/slots", headers=_auth())
    assert res.status_code == 200
    assert res.json()["total"] == 2

    # GET single slot
    res = test_client.get("/api/v1/slots/Slot_1", headers=_auth())
    assert res.status_code == 200
    assert res.json()["slot_id"] == "Slot_1"

    # GET non-existent slot
    res = test_client.get("/api/v1/slots/NonExistent", headers=_auth())
    assert res.status_code == 404

    # POST new slot
    new_slot = {
        "slot_id": "Slot_3",
        "polygon": [[50, 0], [60, 0], [60, 10], [50, 10]],
    }
    res = test_client.post("/api/v1/slots", json=new_slot, headers=_auth())
    assert res.status_code == 201
    assert res.json()["success"] is True
    assert res.json()["slot"]["slot_id"] == "Slot_3"

    # POST duplicate slot returns 409
    res = test_client.post("/api/v1/slots", json=new_slot, headers=_auth())
    assert res.status_code == 409

    # PUT update slot
    updated_slot = {
        "slot_id": "Slot_3",
        "polygon": [[55, 0], [65, 0], [65, 15], [55, 15]],
    }
    res = test_client.put("/api/v1/slots/Slot_3", json=updated_slot, headers=_auth())
    assert res.status_code == 200
    assert res.json()["success"] is True

    # DELETE single slot
    res = test_client.delete("/api/v1/slots/Slot_3", headers=_auth())
    assert res.status_code == 200
    assert res.json()["success"] is True

    # Reject invalid polygon (non-convex or self-intersecting)
    invalid_slot = {
        "slot_id": "Slot_Bad",
        "polygon": [[0, 0], [10, 10], [10, 0], [0, 10]],  # self-intersecting bowtie
    }
    res = test_client.post("/api/v1/slots", json=invalid_slot, headers=_auth())
    assert res.status_code == 422
    assert "self-intersecting" in res.json()["detail"]

