from __future__ import annotations

from app.detector import TemporalSmoother


def test_occupied_requires_consecutive_frames() -> None:
    smoother = TemporalSmoother(occupied_frames=3, empty_frames=3, window_size=5)

    assert smoother.update("Slot_1", True, 0.9) == (False, 0.0)
    assert smoother.update("Slot_1", True, 0.9) == (False, 0.0)
    state, confidence = smoother.update("Slot_1", True, 0.9)
    assert state is True
    assert confidence == 0.9


def test_single_frame_dropout_does_not_flicker() -> None:
    smoother = TemporalSmoother(occupied_frames=2, empty_frames=3, window_size=5)

    smoother.update("Slot_1", True, 0.8)
    state, _ = smoother.update("Slot_1", True, 0.8)
    assert state is True

    state, confidence = smoother.update("Slot_1", False, 0.0)
    assert state is True
    assert confidence == 0.8

    state, _ = smoother.update("Slot_1", False, 0.0)
    assert state is True

    state, confidence = smoother.update("Slot_1", False, 0.0)
    assert state is False
    assert confidence == 0.0


def test_prune_and_reset_clear_state() -> None:
    smoother = TemporalSmoother(occupied_frames=1, empty_frames=1)
    smoother.update("Slot_1", True, 0.7)
    smoother.update("Slot_2", False, 0.0)

    smoother.prune({"Slot_1"})
    assert smoother.update("Slot_2", True, 0.6) == (True, 0.6)

    smoother.reset()
    assert smoother.update("Slot_1", True, 0.5) == (True, 0.5)
