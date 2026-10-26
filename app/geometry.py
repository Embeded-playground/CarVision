"""Utilitas geometri polygon: validasi, winding, IoU/IoA, point-in-polygon vektor, perspektif.

Semua fungsi menerima polygon sebagai list 4 titik [[x, y], ...] (umum/konveks)
dan tidak mengubah input.
"""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

Polygon = Sequence[Sequence[float]]

DEFAULT_MIN_AREA = 50.0
DEFAULT_MAX_AREA = 1.0e7


def to_array(poly: Polygon) -> np.ndarray:
    """Konversi polygon ke ndarray (N, 2) float64."""
    return np.asarray(poly, dtype=np.float64).reshape(-1, 2)


def to_int_list(poly: Polygon) -> list[list[int]]:
    """Konversi polygon ke list [[int, int], ...] (siap JSON)."""
    arr = to_array(poly)
    return [[int(round(float(p[0]))), int(round(float(p[1])))] for p in arr]


def polygon_area(poly: Polygon) -> float:
    """Luas absolut polygon (piksel^2)."""
    arr = to_array(poly).astype(np.float32)
    return float(abs(cv2.contourArea(arr)))


def signed_area(poly: Polygon) -> float:
    """Luas bertanda (shoelace). Positif = counter-clockwise (CCW)."""
    arr = to_array(poly)
    x = arr[:, 0]
    y = arr[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def is_clockwise(poly: Polygon) -> bool:
    return signed_area(poly) < 0


def ensure_ccw(poly: Polygon) -> list[list[int]]:
    """Kembalikan polygon dengan urutan vertex counter-clockwise."""
    arr = to_array(poly)
    if signed_area(arr) < 0:
        arr = arr[::-1]
    return to_int_list(arr)


def is_convex(poly: Polygon, eps: float = 1e-6) -> bool:
    """Cek konveksitas via konsistensi tanda cross-product antar edge."""
    arr = to_array(poly)
    n = len(arr)
    if n < 3:
        return False
    sign = 0
    for i in range(n):
        a, b, c = arr[i], arr[(i + 1) % n], arr[(i + 2) % n]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        if abs(cross) < eps:
            continue
        cur = 1 if cross > 0 else -1
        if sign == 0:
            sign = cur
        elif cur != sign:
            return False
    return True


def _ccw(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float((c[1] - a[1]) * (b[0] - a[0]) - (b[1] - a[1]) * (c[0] - a[0]))


def _segments_intersect(p1, p2, p3, p4) -> bool:
    d1 = _ccw(p3, p4, p1)
    d2 = _ccw(p3, p4, p2)
    d3 = _ccw(p1, p2, p3)
    d4 = _ccw(p1, p2, p4)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def has_self_intersection(poly: Polygon) -> bool:
    """Deteksi sisi yang saling berpotongan (bow-tie)."""
    arr = to_array(poly)
    n = len(arr)
    if n < 4:
        return False
    segs = [(arr[i], arr[(i + 1) % n]) for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if abs(i - j) <= 1 or (i == 0 and j == n - 1):
                continue
            if _segments_intersect(segs[i][0], segs[i][1], segs[j][0], segs[j][1]):
                return True
    return False


def validate_polygon(
    poly: Polygon,
    min_area: float = DEFAULT_MIN_AREA,
    max_area: float = DEFAULT_MAX_AREA,
) -> tuple[bool, str, list[list[int]] | None]:
    """Validasi & normalisasi polygon 4 titik.

    Return (valid, reason, normalized_ccw_polygon).
    """
    arr = to_array(poly)
    if arr.shape != (4, 2):
        return False, "polygon harus 4 titik [x, y]", None
    if not np.all(np.isfinite(arr)):
        return False, "koordinat tidak valid (NaN/Inf)", None
    if has_self_intersection(arr):
        return False, "polygon self-intersecting", None
    if not is_convex(arr):
        return False, "polygon tidak convex", None
    area = polygon_area(arr)
    if area < min_area:
        return False, f"area terlalu kecil ({area:.0f} < {min_area:.0f})", None
    if area > max_area:
        return False, f"area terlalu besar ({area:.0f} > {max_area:.0f})", None
    return True, "ok", ensure_ccw(arr)


def _polygon_stack(polygons: np.ndarray | Sequence[Polygon]) -> np.ndarray:
    """Normalisasi input polygon menjadi ndarray (N, V, 2) float64."""
    arr = np.asarray(polygons, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim != 3 or arr.shape[-1] != 2:
        raise ValueError("polygons harus memiliki bentuk (N, V, 2)")
    return arr


def _ensure_ccw_stack(polygons: np.ndarray) -> np.ndarray:
    """Pastikan semua polygon pada stack berurutan counter-clockwise."""
    x = polygons[..., 0]
    y = polygons[..., 1]
    area = np.sum(x * np.roll(y, -1, axis=-1) - np.roll(x, -1, axis=-1) * y, axis=-1)
    return np.where((area < 0)[..., None, None], polygons[..., ::-1, :], polygons)


def _cross_2d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def points_in_polygons(points: np.ndarray, polygons: np.ndarray | Sequence[Polygon]) -> np.ndarray:
    """Uji titik terhadap banyak polygon sekaligus.

    Args:
        points: ndarray (P, 2).
        polygons: ndarray (S, V, 2) atau sequence polygon.

    Returns:
        ndarray bool (P, S).
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    polys = _polygon_stack(polygons)
    n_points = pts.shape[0]
    n_polys = polys.shape[0]
    if n_points == 0 or n_polys == 0:
        return np.zeros((n_points, n_polys), dtype=bool)

    x = pts[:, 0][:, None, None]
    y = pts[:, 1][:, None, None]
    x1 = polys[:, :, 0][None, :, :]
    y1 = polys[:, :, 1][None, :, :]
    x2 = np.roll(polys[:, :, 0], -1, axis=1)[None, :, :]
    y2 = np.roll(polys[:, :, 1], -1, axis=1)[None, :, :]

    cond = (y1 > y) != (y2 > y)
    denom = np.where((y2 - y1) == 0, 1e-12, (y2 - y1))
    x_inters = (x2 - x1) * (y - y1) / denom + x1
    crossings = np.sum(cond & (x < x_inters), axis=2)
    return (crossings % 2) == 1


def points_in_polygon(points: np.ndarray, polygon: Polygon) -> np.ndarray:
    """Vectorized ray-casting untuk satu polygon. points (N, 2) -> bool mask (N,)."""
    return points_in_polygons(points, np.asarray(polygon, dtype=np.float64)[None, ...])[:, 0]


def polygon_iou(a: Polygon, b: Polygon) -> float:
    """Intersection over Union antara dua polygon konveks."""
    arr_a = to_array(a).astype(np.float32)
    arr_b = to_array(b).astype(np.float32)
    try:
        inter, _ = cv2.intersectConvexConvex(arr_a, arr_b)
    except cv2.error:
        return 0.0
    if inter <= 0:
        return 0.0
    union = polygon_area(a) + polygon_area(b) - float(inter)
    return float(inter) / union if union > 0 else 0.0


def _clip_convex_polygons(subject: np.ndarray, clip: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sutherland-Hodgman vectorized untuk pasangan polygon konveks.

    Args:
        subject: (B, Va, 2) polygon subjek.
        clip: (S, Vc, 2) polygon pemotong.

    Returns:
        Tuple ``(vertices, counts)`` dengan vertices (B, S, MAXV, 2).
    """
    b_count, subject_edges = subject.shape[0], subject.shape[1]
    c_count, clip_edges = clip.shape[0], clip.shape[1]
    max_vertices = max(8, subject_edges + clip_edges)

    base = np.broadcast_to(subject[:, None, :, :], (b_count, c_count, subject_edges, 2))
    clip_b = np.broadcast_to(clip[None, :, :, :], (b_count, c_count, clip_edges, 2))

    current = np.zeros((b_count, c_count, max_vertices, 2), dtype=np.float64)
    current[:, :, :subject_edges, :] = base
    count = np.full((b_count, c_count), subject_edges, dtype=np.int32)

    vertex_index = np.arange(max_vertices, dtype=np.int32)
    eps = 1e-9

    for edge_index in range(clip_edges):
        c1 = clip_b[:, :, edge_index, :]
        c2 = clip_b[:, :, (edge_index + 1) % clip_edges, :]
        edge = c2 - c1

        valid = vertex_index[None, None, :] < count[:, :, None]
        safe_count = np.maximum(count[:, :, None], 1)
        prev_index = (vertex_index[None, None, :] - 1) % safe_count
        prev_index_full = np.broadcast_to(prev_index[..., None], current.shape)
        previous = np.take_along_axis(current, prev_index_full, axis=2)

        inside_current = _cross_2d(edge[:, :, None, :], current - c1[:, :, None, :]) >= -eps
        inside_previous = _cross_2d(edge[:, :, None, :], previous - c1[:, :, None, :]) >= -eps
        inside_current &= valid
        inside_previous &= valid
        crossing = inside_current != inside_previous

        direction = current - previous
        denominator = _cross_2d(direction, edge[:, :, None, :])
        numerator = _cross_2d(c1[:, :, None, :] - previous, edge[:, :, None, :])
        t = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=np.abs(denominator) > 1e-12,
        )
        intersection = previous + t[..., None] * direction

        # Untuk tiap vertex: intersection (jika melintasi edge) lalu vertex itu
        # sendiri (jika berada di dalam). Urutan ini menjaga winding polygon.
        candidates = np.zeros((b_count, c_count, max_vertices * 2, 2), dtype=np.float64)
        candidate_valid = np.zeros((b_count, c_count, max_vertices * 2), dtype=bool)
        candidates[:, :, 0::2, :] = intersection
        candidate_valid[:, :, 0::2] = crossing
        candidates[:, :, 1::2, :] = current
        candidate_valid[:, :, 1::2] = inside_current

        flat_valid = candidate_valid.reshape(b_count, c_count, -1)
        positions = np.cumsum(flat_valid, axis=2) - 1
        new_count = flat_valid.sum(axis=2)
        output = np.zeros((b_count, c_count, max_vertices, 2), dtype=np.float64)

        b_idx, c_idx, candidate_idx = np.nonzero(flat_valid & (positions < max_vertices))
        if b_idx.size:
            output[b_idx, c_idx, positions[b_idx, c_idx, candidate_idx], :] = candidates[
                b_idx, c_idx, candidate_idx, :
            ]

        current = output
        count = np.clip(new_count, 0, max_vertices).astype(np.int32)

    return current, count


def convex_intersection_areas(
    polys_a: np.ndarray | Sequence[Polygon],
    polys_b: np.ndarray | Sequence[Polygon],
) -> np.ndarray:
    """Luas irisan semua pasangan polygon konveks.

    Returns:
        ndarray float64 (N_a, N_b).
    """
    a = _ensure_ccw_stack(_polygon_stack(polys_a))
    b = _ensure_ccw_stack(_polygon_stack(polys_b))
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float64)

    vertices, counts = _clip_convex_polygons(a, b)
    max_vertices = vertices.shape[2]
    index = np.arange(max_vertices, dtype=np.int32)
    valid = index[None, None, :] < counts[:, :, None]
    next_index = (index[None, None, :] + 1) % np.maximum(counts[:, :, None], 1)
    next_index_full = np.broadcast_to(next_index[..., None], vertices.shape)
    next_vertices = np.take_along_axis(vertices, next_index_full, axis=2)

    terms = (
        vertices[..., 0] * next_vertices[..., 1]
        - next_vertices[..., 0] * vertices[..., 1]
    )
    terms = np.where(valid, terms, 0.0)
    return 0.5 * np.abs(np.sum(terms, axis=2))


def bbox_to_polygon(bbox: Sequence[float]) -> list[list[float]]:
    x1, y1, x2, y2 = (float(v) for v in bbox[:4])
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def ioa_bbox_vs_polygons(
    boxes: np.ndarray,
    polygons: np.ndarray | Sequence[Polygon],
) -> np.ndarray:
    """IoA semua pasangan bbox kendaraan vs polygon slot.

    Args:
        boxes: ndarray (N, 4) dalam format xyxy.
        polygons: ndarray (S, V, 2) atau sequence polygon.

    Returns:
        ndarray float64 (N, S), nilai 0..1.
    """
    boxes_arr = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    polys = _polygon_stack(polygons)
    n_boxes = boxes_arr.shape[0]
    n_polys = polys.shape[0]
    if n_boxes == 0 or n_polys == 0:
        return np.zeros((n_boxes, n_polys), dtype=np.float64)

    x1, y1, x2, y2 = (boxes_arr[:, i] for i in range(4))
    rects = np.stack(
        [
            np.stack([x1, y1], axis=1),
            np.stack([x2, y1], axis=1),
            np.stack([x2, y2], axis=1),
            np.stack([x1, y2], axis=1),
        ],
        axis=1,
    )
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    intersections = convex_intersection_areas(rects, polys)
    ratio = np.divide(
        intersections,
        areas[:, None],
        out=np.zeros_like(intersections),
        where=areas[:, None] > 0,
    )
    return np.clip(ratio, 0.0, 1.0)


def ioa_bbox_vs_polygon(bbox: Sequence[float], polygon: Polygon) -> float:
    """Intersection over (vehicle bbox) Area. bbox = (x1, y1, x2, y2)."""
    boxes = np.asarray(bbox, dtype=np.float64).reshape(1, 4)
    polys = np.asarray(polygon, dtype=np.float64)[None, ...]
    return float(ioa_bbox_vs_polygons(boxes, polys)[0, 0])


def ground_points(xyxy: np.ndarray) -> np.ndarray:
    """Titik kontak tanah kendaraan = (center_x, y_max) untuk tiap bbox (N, 4)."""
    boxes = np.asarray(xyxy, dtype=np.float64).reshape(-1, 4)
    if boxes.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
    return np.stack([cx, boxes[:, 3]], axis=1)


def perspective_transform(src_quad: Polygon, dst_quad: Polygon) -> np.ndarray:
    src = to_array(src_quad).astype(np.float32)
    dst = to_array(dst_quad).astype(np.float32)
    return cv2.getPerspectiveTransform(src, dst)


def quad_to_rect_dst(quad: Polygon) -> tuple[list[list[float]], int, int]:
    """Bangun dst rectangle top-down (w, h) dari rata-rata panjang sisi lawan."""
    q = to_array(quad)
    top_w = float(np.linalg.norm(q[1] - q[0]))
    bottom_w = float(np.linalg.norm(q[2] - q[3]))
    left_h = float(np.linalg.norm(q[3] - q[0]))
    right_h = float(np.linalg.norm(q[2] - q[1]))
    w = int(max(top_w, bottom_w, 1.0))
    h = int(max(left_h, right_h, 1.0))
    dst = [[0, 0], [w, 0], [w, h], [0, h]]
    return dst, w, h


def warp_polygon(polygon: Polygon, matrix: np.ndarray) -> list[list[int]]:
    pts = to_array(polygon).astype(np.float32).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts, matrix).reshape(-1, 2)
    return to_int_list(out)
