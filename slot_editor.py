import argparse
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
import torch
from ultralytics import YOLO

from app import geometry, slot_store
from app.detector import MIN_INFER_CONF, filter_vehicle_detections
from app.sources import choose_source, resolve_source

WINDOW_NAME = "Parking Slot Editor"
TOPVIEW_NAME = "Top-Down View"
IOU_DEDUPE = 0.5
MIN_SLOT_AREA = 50.0
ROW_GAP_FACTOR = 0.6
ANGLE_THRESHOLD = 0.12  # rad (~7 deg) -> snap ke axis-aligned
MAX_SLOTS_PER_ROW = 100

DEFAULT_PAD = 1.15
DEFAULT_EXTEND = 1
DEFAULT_ROI_ROWS = 1
DEFAULT_ROI_COLS = 4

# --- State global ---
points_temp: list[tuple[int, int]] = []
current_slots: list[dict] = []
mouse_pos: tuple[int, int] | None = None
last_status = "Ready"

roi_active = False
roi_ready = False
roi_need_params = False
roi_points: list[tuple[int, int]] = []
last_roi_quad: list[tuple[int, int]] = []
roi_rows = DEFAULT_ROI_ROWS
roi_cols = DEFAULT_ROI_COLS

last_pad = DEFAULT_PAD
last_extend = DEFAULT_EXTEND
grid_params_prompted = False
perspective_view = False


# ============================ Utilitas model/sumber ============================
def list_models() -> list[str]:
    models_dir = Path("models")
    if models_dir.is_dir():
        return [str(p) for p in sorted(models_dir.glob("*.pt"))]
    return []


def choose_model() -> str:
    env_model = os.getenv("MODEL_PATH")
    if env_model and os.path.exists(env_model):
        return env_model

    available = list_models()
    default = "models/yolo26m.pt" if os.path.exists("models/yolo26m.pt") else "models/yolo26n.pt"

    print("=" * 40)
    print("Pilih model")
    print("=" * 40)
    for i, model in enumerate(available, start=1):
        mark = " (default)" if model == default else ""
        print(f"[{i}] {model}{mark}")
    try:
        choice = input(f"Masukkan pilihan (1-{len(available)}) [Enter = default]: ").strip()
    except EOFError:
        choice = ""
    if choice.isdigit() and 1 <= int(choice) <= len(available):
        return available[int(choice) - 1]
    return default


# ============================ Geometri polygon ============================
def _poly_area(poly: list) -> float:
    return geometry.polygon_area(poly)


def polygon_iou(a: list, b: list) -> float:
    return geometry.polygon_iou(a, b)


def _validate_click_polygon(points: list[tuple[int, int]]) -> tuple[bool, str, list | None]:
    """Validasi 4 titik hasil klik manual sebelum dijadikan slot."""
    ok, reason, normalized = geometry.validate_polygon(points, min_area=MIN_SLOT_AREA)
    if not ok or normalized is None:
        return False, reason, None
    return True, "ok", normalized


def dedupe_slots(slots: list[dict], iou_thresh: float = IOU_DEDUPE) -> list[dict]:
    kept: list[dict] = []
    for slot in slots:
        if any(polygon_iou(slot["polygon"], k["polygon"]) > iou_thresh for k in kept):
            continue
        kept.append(slot)
    return kept


def renumber(slots: list[dict]) -> list[dict]:
    for i, slot in enumerate(slots, start=1):
        slot["slot_id"] = f"Slot_{i}"
    return slots


def _principal_angle(points: np.ndarray) -> float:
    """Sudut (rad) sumbu utama baris kendaraan via PCA. Dinormalkan ke [-pi/2, pi/2]."""
    if len(points) < 2:
        return 0.0
    pts = np.asarray(points, dtype=np.float64)
    centered = pts - pts.mean(axis=0)
    try:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return 0.0
    d = vt[0]
    angle = math.atan2(float(d[1]), float(d[0]))
    if angle > math.pi / 2:
        angle -= math.pi
    elif angle < -math.pi / 2:
        angle += math.pi
    return angle


def _slot_quad(center: tuple[float, float], w: float, h: float, angle: float) -> list[list[int]]:
    """Bangun quad 4 titik (oriented) berpusat di center dengan lebar w, tinggi h, rotasi angle."""
    cx, cy = center
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    ux, uy = cos_a * (w / 2.0), sin_a * (w / 2.0)      # arah baris
    vx, vy = -sin_a * (h / 2.0), cos_a * (h / 2.0)     # tegak lurus baris
    quad = [
        [cx - ux - vx, cy - uy - vy],
        [cx + ux - vx, cy + uy - vy],
        [cx + ux + vx, cy + uy + vy],
        [cx - ux + vx, cy - uy + vy],
    ]
    return [[int(round(p[0])), int(round(p[1]))] for p in quad]


# ============================ Deteksi & simpan ============================
def _detect_vehicles(frame: np.ndarray, model: YOLO, device: str):
    global last_status
    try:
        with torch.no_grad():
            results = model(frame, verbose=False, device=device, conf=MIN_INFER_CONF)[0]
    except Exception as exc:
        last_status = f"ERROR infer: {exc}"
        print(f"[DETECT] gagal: {exc}")
        return None

    detections = sv.Detections.from_ultralytics(results)
    detections = filter_vehicle_detections(detections)
    return detections


def _merge_and_save(polys: list[list], tag: str = "Auto") -> None:
    global last_status
    added = 0
    skipped = 0
    for poly in polys:
        if any(polygon_iou(poly, s["polygon"]) > IOU_DEDUPE for s in current_slots):
            skipped += 1
            continue
        current_slots.append({"slot_id": slot_store.next_slot_id(current_slots), "polygon": poly})
        added += 1

    if added:
        try:
            slot_store.save_slots(current_slots)
            last_status = f"{tag} +{added} (skip {skipped})"
            print(f"[{tag}] +{added} slot (skip {skipped} duplikat)")
        except Exception as exc:
            last_status = f"ERROR save: {exc}"
            print(f"[{tag}] gagal simpan: {exc}")
    else:
        last_status = f"{tag}: 0 baru (skip {skipped})"
        print(f"[{tag}] Tidak ada slot baru (skip {skipped} duplikat)")


def generate_grid_slots(frame: np.ndarray, model: YOLO, device: str, pad_factor: float, extend: int, oriented: bool) -> None:
    """Mode A: proyeksi grid baris/kolom dari anchor kendaraan (slot kosong ikut dibuat)."""
    global last_status
    detections = _detect_vehicles(frame, model, device)
    if detections is None:
        return
    if len(detections) == 0:
        last_status = "Grid: tidak ada kendaraan"
        print("[GRID] Tidak ada kendaraan terdeteksi")
        return

    boxes = detections.xyxy.astype(np.float64)
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    valid = (widths > 5) & (heights > 5)
    if not np.any(valid):
        last_status = "Grid: bbox tidak valid"
        return
    boxes, widths, heights = boxes[valid], widths[valid], heights[valid]

    slot_w = max(10.0, float(np.median(widths)) * pad_factor)
    slot_h = max(10.0, float(np.median(heights)) * pad_factor)

    centers = np.stack(
        [(boxes[:, 0] + boxes[:, 2]) / 2.0, (boxes[:, 1] + boxes[:, 3]) / 2.0], axis=1
    )
    row_gap = max(10.0, float(np.median(heights)) * ROW_GAP_FACTOR)

    # Cluster baris berdasarkan kedekatan cy
    order = np.argsort(centers[:, 1])
    rows: list[list[int]] = []
    current = [int(order[0])]
    for idx in order[1:]:
        if abs(centers[idx, 1] - centers[current[-1], 1]) <= row_gap:
            current.append(int(idx))
        else:
            rows.append(current)
            current = [int(idx)]
    rows.append(current)

    polys: list[list] = []
    for row in rows:
        pts = centers[row]
        pts = pts[np.argsort(pts[:, 0])]
        row_y = float(np.median(pts[:, 1]))

        if len(pts) >= 2:
            diffs = np.diff(pts[:, 0])
            diffs = diffs[diffs > 1]
            pitch = float(np.median(diffs)) if len(diffs) else slot_w
        else:
            pitch = slot_w
        pitch = max(pitch, slot_w * 0.5)

        angle = _principal_angle(pts) if oriented else 0.0
        if abs(angle) < ANGLE_THRESHOLD:
            angle = 0.0

        x_min = float(pts[0, 0]) - extend * pitch
        x_max = float(pts[-1, 0]) + extend * pitch
        n = int(round((x_max - x_min) / pitch)) + 1
        n = max(1, min(n, MAX_SLOTS_PER_ROW))

        for k in range(n):
            cx = x_min + k * pitch
            polys.append(_slot_quad((cx, row_y), slot_w, slot_h, angle))

    _merge_and_save(polys, "Grid")


# ============================ Mode B: ROI perspective ============================
def _warp_roi_cells(roi: list[tuple[int, int]], rows: int, cols: int) -> list[list]:
    if len(roi) != 4 or rows < 1 or cols < 1:
        return []
    src = np.array(roi, dtype=np.float32)

    top_w = float(np.linalg.norm(src[1] - src[0]))
    bottom_w = float(np.linalg.norm(src[2] - src[3]))
    left_h = float(np.linalg.norm(src[3] - src[0]))
    right_h = float(np.linalg.norm(src[2] - src[1]))
    w = max(top_w, bottom_w, 1.0)
    h = max(left_h, right_h, 1.0)

    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    try:
        m = cv2.getPerspectiveTransform(src, dst)
        m_inv = np.linalg.inv(m)
    except (cv2.error, np.linalg.LinAlgError):
        return []

    cell_w = w / cols
    cell_h = h / rows
    polys: list[list] = []
    for r in range(rows):
        for c in range(cols):
            x0, y0 = c * cell_w, r * cell_h
            x1, y1 = x0 + cell_w, y0 + cell_h
            cell = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32).reshape(-1, 1, 2)
            back = cv2.perspectiveTransform(cell, m_inv).reshape(-1, 2)
            polys.append([[int(round(p[0])), int(round(p[1]))] for p in back])
    return polys


def _start_roi() -> None:
    global roi_active, roi_ready, roi_need_params, roi_points, last_roi_quad, last_status
    roi_active = True
    roi_ready = False
    roi_need_params = False
    roi_points = []
    last_roi_quad = []
    last_status = "ROI: klik 4 sudut aisle (dekat-kiri, dekat-kanan, jauh-kanan, jauh-kiri)"
    print("[ROI] Klik 4 sudut aisle searah jarum jam, mulai dari sisi terdekat.")


def _cancel_roi() -> None:
    global roi_active, roi_ready, roi_need_params, roi_points, last_status
    roi_points = []
    roi_ready = False
    roi_need_params = False
    roi_active = False
    last_status = "ROI dibatalkan"


def _prompt_roi_params() -> None:
    global roi_rows, roi_cols, last_status
    try:
        r = input(f"Jumlah baris (rows) [{roi_rows}]: ").strip()
        if r.isdigit():
            roi_rows = max(1, int(r))
        c = input(f"Jumlah kolom (cols) [{roi_cols}]: ").strip()
        if c.isdigit():
            roi_cols = max(1, int(c))
    except EOFError:
        pass
    last_status = f"ROI {roi_rows}x{roi_cols} | +/- kolom, . / , baris, Enter simpan, Esc batal"


def commit_roi_slots() -> None:
    global roi_active, roi_ready, roi_need_params, roi_points, last_roi_quad, last_status
    if len(roi_points) != 4:
        last_status = "ROI tidak lengkap"
        return
    valid, reason, _ = geometry.validate_polygon(roi_points, min_area=100.0)
    if not valid:
        last_status = f"ROI ditolak: {reason}"
        print(f"[ROI] polygon ditolak: {reason}")
        return
    polys = _warp_roi_cells(roi_points, roi_rows, roi_cols)
    if not polys:
        last_status = "ROI gagal (perspective transform)"
        return
    polys = [p for p in polys if geometry.polygon_area(p) >= MIN_SLOT_AREA]
    if not polys:
        last_status = "ROI: semua sel terlalu kecil"
        return
    _merge_and_save(polys, "ROI")
    last_roi_quad = list(roi_points)
    roi_points = []
    roi_ready = False
    roi_need_params = False
    roi_active = False


def _prompt_grid_params() -> None:
    global last_pad, last_extend
    try:
        p = input(f"Padding factor [{last_pad}]: ").strip()
        if p:
            last_pad = max(0.5, float(p))
        e = input(f"Extra slot per ujung [{last_extend}]: ").strip()
        if e:
            last_extend = max(0, int(e))
    except (EOFError, ValueError):
        pass


# ============================ Mouse ============================
def on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
    global mouse_pos, last_status, roi_points, roi_need_params
    try:
        if event == cv2.EVENT_MOUSEMOVE:
            mouse_pos = (x, y)
            return

        if roi_active:
            if event == cv2.EVENT_LBUTTONDOWN and len(roi_points) < 4:
                roi_points.append((x, y))
                if len(roi_points) == 4:
                    roi_need_params = True
                else:
                    last_status = f"ROI titik {len(roi_points)}/4"
            elif event == cv2.EVENT_RBUTTONDOWN and roi_points:
                roi_points.pop()
                roi_need_params = False
                last_status = f"ROI titik {len(roi_points)}/4"
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points_temp) < 4:
                points_temp.append((x, y))
            if len(points_temp) == 4:
                valid, reason, normalized = _validate_click_polygon(points_temp)
                points_temp.clear()
                if not valid or normalized is None:
                    last_status = f"Slot ditolak: {reason}"
                    print(f"[MANUAL] polygon ditolak: {reason}")
                    return
                slot = {"slot_id": slot_store.next_slot_id(current_slots), "polygon": normalized}
                current_slots.append(slot)
                try:
                    slot_store.save_slots(current_slots)
                    last_status = f"Saved {slot['slot_id']}"
                except Exception as exc:
                    last_status = f"ERROR save: {exc}"
                    print(f"[ERROR] {exc}")

        if event == cv2.EVENT_RBUTTONDOWN:
            if points_temp:
                points_temp.pop()
                last_status = "Hapus titik terakhir"
            elif current_slots:
                removed = current_slots.pop()
                try:
                    slot_store.save_slots(current_slots)
                    last_status = f"Hapus {removed['slot_id']}"
                except Exception as exc:
                    last_status = f"ERROR save: {exc}"
                    print(f"[ERROR] {exc}")
    except Exception as exc:
        last_status = f"ERROR mouse: {exc}"
        print(f"[ERROR mouse] {exc}")


# ============================ Render ============================
def draw(frame: np.ndarray) -> None:
    for slot in current_slots:
        poly = np.array(slot["polygon"], dtype=np.int32)
        cv2.polylines(frame, [poly], isClosed=True, color=(0, 255, 0), thickness=2)
        pt = poly[0]
        cv2.putText(frame, slot["slot_id"], (pt[0] + 5, pt[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if roi_active and roi_points:
        for i in range(len(roi_points) - 1):
            cv2.line(frame, roi_points[i], roi_points[i + 1], (0, 165, 255), 2)
        if len(roi_points) == 4:
            cv2.line(frame, roi_points[3], roi_points[0], (0, 165, 255), 2)
            if roi_ready:
                for poly in _warp_roi_cells(roi_points, roi_rows, roi_cols):
                    cv2.polylines(frame, [np.array(poly, dtype=np.int32)], True, (255, 200, 0), 1)
        for pt in roi_points:
            cv2.circle(frame, pt, 4, (0, 165, 255), -1)

    elif points_temp:
        for pt in points_temp:
            cv2.circle(frame, pt, 3, (0, 255, 255), -1)
        for i in range(len(points_temp) - 1):
            cv2.line(frame, points_temp[i], points_temp[i + 1], (0, 255, 255), 2)
        if mouse_pos is not None:
            cv2.line(frame, points_temp[-1], mouse_pos, (0, 255, 255), 1)

    mode = "ROI" if roi_active else "MANUAL"
    extra = f" {roi_rows}x{roi_cols}" if roi_active else ""
    hud = f"[{mode}{extra}] Slots: {len(current_slots)} | Pts: {len(points_temp)} | {last_status}"
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(frame, hud, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


def render_topdown(frame: np.ndarray) -> None:
    """Tampilkan normalisasi perspektif (top-down) memakai 4 titik ROI + grid slot."""
    global last_status
    quad = roi_points if len(roi_points) == 4 else last_roi_quad
    if len(quad) != 4:
        last_status = "Top-down: butuh 4 titik ROI"
        return
    try:
        dst, w, h = geometry.quad_to_rect_dst(quad)
        m = geometry.perspective_transform(quad, dst)
        warped = cv2.warpPerspective(frame, m, (w, h))
    except Exception as exc:  # pragma: no cover - jaga GUI tidak crash
        last_status = f"Top-down gagal: {exc}"
        return
    for cell in _warp_roi_cells(quad, roi_rows, roi_cols):
        warped_cell = geometry.warp_polygon(cell, m)
        cv2.polylines(warped, [np.array(warped_cell, dtype=np.int32)], True, (255, 200, 0), 1)
    cv2.imshow(TOPVIEW_NAME, warped)


def clear_all() -> None:
    global last_status
    current_slots.clear()
    points_temp.clear()
    try:
        slot_store.save_slots(current_slots)
        last_status = "Semua slot dihapus"
    except Exception as exc:
        last_status = f"ERROR save: {exc}"
        print(f"[ERROR] {exc}")
    print("[INFO] Semua slot dihapus")


# ============================ Main ============================
def main() -> None:
    global current_slots, last_status
    global roi_active, roi_ready, roi_need_params, roi_rows, roi_cols
    global last_pad, last_extend, grid_params_prompted, perspective_view

    parser = argparse.ArgumentParser(description="Slot Editor (Parkir) - grid/ROI generation")
    parser.add_argument("--source", help="Path video / 0 untuk webcam")
    parser.add_argument("--model", help="Path model .pt (default: models/yolo26m.pt)")
    parser.add_argument("--display-scale", type=float, default=1.0, help="Skala resize tampilan (mis. 0.75)")
    parser.add_argument("--no-loop", action="store_true", help="Jangan mengulang video saat habis")
    parser.add_argument("--dedupe", action="store_true", help="Rapikan slot (buang tumpang-tindih) lalu keluar")
    parser.add_argument("--pad-factor", type=float, default=DEFAULT_PAD, help="Faktor ukuran slot dari kendaraan")
    parser.add_argument("--extend", type=int, default=DEFAULT_EXTEND, help="Jumlah slot tambahan per ujung baris")
    parser.add_argument("--roi-rows", type=int, default=DEFAULT_ROI_ROWS, help="Default baris ROI")
    parser.add_argument("--roi-cols", type=int, default=DEFAULT_ROI_COLS, help="Default kolom ROI")
    parser.add_argument("--axis-aligned", dest="oriented", action="store_false", help="Paksa slot axis-aligned")
    parser.set_defaults(oriented=True)
    args = parser.parse_args()

    last_pad = max(0.5, args.pad_factor)
    last_extend = max(0, args.extend)
    roi_rows = max(1, args.roi_rows)
    roi_cols = max(1, args.roi_cols)

    current_slots = slot_store.load_slots()

    if args.dedupe:
        before = len(current_slots)
        current_slots = renumber(dedupe_slots(current_slots))
        slot_store.save_slots(current_slots)
        print(f"[DEDUPE] {before} -> {len(current_slots)} slot")
        return

    if args.source is None:
        source: str | int = choose_source()
    else:
        source = resolve_source(args.source)

    model_path = args.model or choose_model()
    print(f"[INFO] {len(current_slots)} slot dimuat dari {slot_store.SLOTS_PATH}")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Gagal membuka sumber: {source}")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(model_path)
    try:
        model.to(device)
    except Exception:
        pass
    print(f"[INFO] model={model_path} device={device}")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 720)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)
    try:
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)
    except Exception:
        pass

    print("=" * 52)
    print("KONTROL (pastikan jendela video dalam fokus)")
    print("=" * 52)
    print("Klik kiri 4x -> Slot manual (polygon)")
    print("Klik kanan  -> Hapus titik/slot terakhir")
    print("A -> Auto GRID dari anchor kendaraan (slot kosong ikut dibuat)")
    print("R -> Mode ROI 4-titik (perspective), lalu atur baris/kolom")
    print("     ROI: +/- kolom | . / , baris | Enter simpan | Esc batal")
    print("T -> Toggle preview top-down (normalisasi perspektif)")
    print("C -> Clear semua slot")
    print("Q -> Simpan & Keluar")
    print("=" * 52)

    is_file = isinstance(source, str) and Path(source).is_file()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            if is_file and not args.no_loop:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            print("[INFO] Video berakhir")
            break

        if roi_need_params:
            _prompt_roi_params()
            roi_need_params = False
            roi_ready = True

        draw(frame)

        if perspective_view:
            render_topdown(frame.copy())

        display = frame
        if args.display_scale != 1.0:
            display = cv2.resize(frame, None, fx=args.display_scale, fy=args.display_scale)

        cv2.imshow(WINDOW_NAME, display)
        key = cv2.waitKey(1) & 0xFF

        try:
            if key == ord("q"):
                break
            elif key == ord("c"):
                clear_all()
            elif key == ord("r"):
                _start_roi()
            elif key == ord("a") and not roi_active:
                if not grid_params_prompted:
                    _prompt_grid_params()
                    grid_params_prompted = True
                generate_grid_slots(frame, model, device, last_pad, last_extend, args.oriented)
            elif key == ord("t"):
                perspective_view = not perspective_view
                if not perspective_view:
                    try:
                        cv2.destroyWindow(TOPVIEW_NAME)
                    except Exception:
                        pass
                last_status = f"Top-down: {'ON' if perspective_view else 'OFF'}"
            elif roi_active and roi_ready:
                if key in (13, 10):
                    commit_roi_slots()
                elif key == 27:
                    _cancel_roi()
                elif key in (ord("+"), ord("=")):
                    roi_cols = min(50, roi_cols + 1)
                    last_status = f"ROI {roi_rows}x{roi_cols}"
                elif key in (ord("-"), ord("_")):
                    roi_cols = max(1, roi_cols - 1)
                    last_status = f"ROI {roi_rows}x{roi_cols}"
                elif key in (ord("."), ord(">")):
                    roi_rows = min(50, roi_rows + 1)
                    last_status = f"ROI {roi_rows}x{roi_cols}"
                elif key in (ord(","), ord("<")):
                    roi_rows = max(1, roi_rows - 1)
                    last_status = f"ROI {roi_rows}x{roi_cols}"
        except Exception as exc:
            last_status = f"ERROR: {exc}"
            print(f"[ERROR] {exc}")

    try:
        slot_store.save_slots(current_slots)
    except Exception as exc:
        print(f"[ERROR] gagal simpan akhir: {exc}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] {len(current_slots)} slot tersimpan ke {slot_store.SLOTS_PATH}")


if __name__ == "__main__":
    main()
