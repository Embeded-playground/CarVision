from __future__ import annotations

import cv2
import numpy as np

from app import geometry


def _random_boxes(rng: np.random.Generator, count: int) -> np.ndarray:
    boxes = []
    for _ in range(count):
        x1, y1 = rng.uniform(0, 150, 2)
        width, height = rng.uniform(10, 80, 2)
        boxes.append([x1, y1, x1 + width, y1 + height])
    return np.asarray(boxes, dtype=np.float64)


def _random_quads(rng: np.random.Generator, count: int) -> np.ndarray:
    quads = []
    for _ in range(count):
        center = rng.uniform(20, 180, 2)
        size = rng.uniform(15, 100, 2)
        angle = rng.uniform(0, 360)
        quads.append(
            cv2.boxPoints(
                (
                    (float(center[0]), float(center[1])),
                    (float(size[0]), float(size[1])),
                    float(angle),
                )
            )
        )
    return np.asarray(quads, dtype=np.float64)


def test_points_in_polygons_matches_single_polygon() -> None:
    rng = np.random.default_rng(7)
    points = rng.uniform(0, 200, (100, 2))
    polygons = _random_quads(rng, 12)

    matrix = geometry.points_in_polygons(points, polygons)
    assert matrix.shape == (len(points), len(polygons))

    for index, polygon in enumerate(polygons):
        expected = geometry.points_in_polygon(points, polygon)
        np.testing.assert_array_equal(matrix[:, index], expected)


def test_ioa_matrix_matches_cv2_reference() -> None:
    rng = np.random.default_rng(11)
    boxes = _random_boxes(rng, 15)
    polygons = _random_quads(rng, 20)

    matrix = geometry.ioa_bbox_vs_polygons(boxes, polygons)
    assert matrix.shape == (len(boxes), len(polygons))
    assert np.all(matrix >= 0.0)
    assert np.all(matrix <= 1.0)

    for box_index, box in enumerate(boxes):
        for poly_index, polygon in enumerate(polygons):
            expected = geometry.ioa_bbox_vs_polygon(box, polygon)
            np.testing.assert_allclose(matrix[box_index, poly_index], expected, atol=1e-6)


def test_convex_intersection_area_of_identical_polygons() -> None:
    polygon = np.array([[0, 0], [10, 0], [10, 5], [0, 5]], dtype=np.float64)
    areas = geometry.convex_intersection_areas(polygon[None, ...], polygon[None, ...])
    np.testing.assert_allclose(areas[0, 0], geometry.polygon_area(polygon), atol=1e-9)


def test_empty_inputs_return_empty_matrices() -> None:
    polygons = np.array([[[0, 0], [10, 0], [10, 10], [0, 10]]], dtype=np.float64)
    assert geometry.points_in_polygons(np.zeros((0, 2)), polygons).shape == (0, 1)
    assert geometry.ioa_bbox_vs_polygons(np.zeros((0, 4)), polygons).shape == (0, 1)
