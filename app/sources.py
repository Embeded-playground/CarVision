"""Penemuan sumber video test + prompt pemilihan sumber.

Semua video test disimpan di folder `test_videos/` (dapat diubah via env
`TEST_VIDEO_DIR`). Fungsi di sini dipakai bersama oleh `slot_editor.py` dan
`test_video.py` agar daftar/prompt selalu sinkron dan video baru otomatis
muncul tanpa mengubah kode.
"""

from __future__ import annotations

import os
from pathlib import Path

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v")


def video_dir() -> Path:
    return Path(os.getenv("TEST_VIDEO_DIR", "test_videos"))


def _scan(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return [
        p
        for p in sorted(directory.iterdir())
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    ]


def list_videos() -> list[Path]:
    """Daftar video: folder test_videos dulu, lalu direktori kerja (root)."""
    found: list[Path] = []
    seen: set[Path] = set()
    for directory in (video_dir(), Path(".")):
        for path in _scan(directory):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            found.append(path)
    return found


def default_video() -> str | None:
    """Video default: env DEFAULT_VIDEO (nama/path), else video pertama ditemukan."""
    env_default = os.getenv("DEFAULT_VIDEO")
    if env_default:
        return resolve_video(env_default)
    videos = list_videos()
    return str(videos[0]) if videos else None


def resolve_video(value: str) -> str:
    """Resolusi path video: exact path -> isi folder test_videos -> apa adanya."""
    path = Path(value)
    if path.is_file():
        return str(path)
    candidate = video_dir() / value
    if candidate.is_file():
        return str(candidate)
    return str(path)


def resolve_source(value: str) -> str | int:
    """Normalisasi argumen sumber: digit -> index webcam, selain itu -> path video."""
    stripped = value.strip()
    if stripped.lstrip("-").isdigit():
        return int(stripped)
    return resolve_video(stripped)


def choose_source() -> str | int:
    """Prompt interaktif: daftar semua video test + opsi webcam."""
    videos = list_videos()
    default = default_video()

    print("=" * 40)
    print("Pilih sumber video")
    print("=" * 40)
    if videos:
        for i, path in enumerate(videos, start=1):
            mark = " (default)" if default and str(path) == default else ""
            print(f"[{i}] {path}{mark}")
    else:
        print(f"(tidak ada video di '{video_dir()}/')")
    webcam_index = len(videos) + 1
    print(f"[{webcam_index}] Webcam (device index)")

    try:
        choice = input(f"Masukkan pilihan (default: {default or 'webcam'}): ").strip()
    except EOFError:
        choice = ""

    if choice == "":
        return default if default else 0

    if choice.isdigit():
        n = int(choice)
        if 1 <= n <= len(videos):
            return str(videos[n - 1])
        if n == webcam_index:
            return _ask_webcam()

    # fallback: anggap nama file
    return resolve_video(choice)


def _ask_webcam() -> int:
    try:
        idx = input("Device index webcam (default: 0): ").strip()
    except EOFError:
        idx = "0"
    try:
        return int(idx) if idx else 0
    except ValueError:
        return 0
