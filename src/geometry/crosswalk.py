from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


class CrosswalkZone:
    def __init__(self, polygon):
        self.polygon = np.array(polygon[:4], dtype=np.int32)
        self._mask_canvas: Optional[np.ndarray] = None
        self._canvas_shape: Optional[tuple] = None
    def draw(self, frame):
        cv2.polylines(frame, [self.polygon], True, (255, 255, 0), 2)
    def intersects_box(self, box, min_ratio=0.02):
        x1, y1, x2, y2 = map(int, box)
        box_poly = np.array([
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2]
        ], dtype=np.int32)
        inter_area, _ = cv2.intersectConvexConvex(
            self.polygon.astype(np.float32),
            box_poly.astype(np.float32)
        )
        box_area = (x2 - x1) * (y2 - y1)
        if box_area <= 0:
            return False
        overlap_ratio = inter_area / box_area
        return overlap_ratio >= min_ratio

    def intersects_mask(
        self,
        mask: Optional[np.ndarray],
        frame_shape: tuple,
        min_ratio: float = 0.02,
        box_fallback=None,
    ) -> bool:
        if mask is None:
            if box_fallback is not None:
                return self.intersects_box(box_fallback, min_ratio=min_ratio)
            return False

        fh, fw = frame_shape[:2]

        if self._canvas_shape != (fh, fw):
            self._mask_canvas = np.zeros((fh, fw), dtype=np.uint8)
            pts = self.polygon.reshape((-1, 1, 2))
            cv2.fillPoly(self._mask_canvas, [pts], 1)
            self._canvas_shape = (fh, fw)

        mh, mw = mask.shape[:2]
        m = mask
        if (mh, mw) != (fh, fw):
            m = cv2.resize(mask, (fw, fh), interpolation=cv2.INTER_NEAREST)

        inter = int(np.count_nonzero(m & self._mask_canvas))
        mask_area = int(np.count_nonzero(m))
        if mask_area == 0:
            return False
        return (inter / mask_area) >= min_ratio
    def compute_split_boundary(self, ratio=0.5):
        p0, p1, p2, p3 = self.polygon
        left_point = (
            int(p0[0] + ratio * (p3[0] - p0[0])),
            int(p0[1] + ratio * (p3[1] - p0[1]))
        )
        right_point = (
            int(p1[0] + ratio * (p2[0] - p1[0])),
            int(p1[1] + ratio * (p2[1] - p1[1]))
        )
        return left_point, right_point
    def draw_half_split(self, frame, ratio=0.5):
        left_point, right_point = self.compute_split_boundary(ratio)
        cv2.line(
            frame,
            left_point,
            right_point,
            (0, 255, 255),
            3
        )
    def get_split_polygons(self, ratio=0.5):
        p0, p1, p2, p3 = self.polygon
        left_split, right_split = self.compute_split_boundary(ratio)
        upper = np.array([p0, p1, right_split, left_split], dtype=np.int32)
        lower = np.array([left_split, right_split, p2, p3], dtype=np.int32)
        return upper, lower
    def intersects_mask_foot(
        self,
        mask: Optional[np.ndarray],
        frame_shape: tuple,
        bottom_ratio: float = 0.20,
        box_fallback=None,
    ) -> bool:
        if mask is None:
            if box_fallback is not None:
                return self.intersects_box(box_fallback, min_ratio=0.01)
            return False

        fh, fw = frame_shape[:2]
        if self._canvas_shape != (fh, fw):
            self._mask_canvas = np.zeros((fh, fw), dtype=np.uint8)
            pts = self.polygon.reshape((-1, 1, 2))
            cv2.fillPoly(self._mask_canvas, [pts], 1)
            self._canvas_shape = (fh, fw)

        mh, mw = mask.shape[:2]
        m = mask
        if (mh, mw) != (fh, fw):
            m = cv2.resize(mask, (fw, fh), interpolation=cv2.INTER_NEAREST)

        cut = max(0, int(m.shape[0] * (1.0 - bottom_ratio)))
        foot_mask = np.zeros_like(m)
        foot_mask[cut:, :] = m[cut:, :]
        return bool(np.any(foot_mask & self._mask_canvas))

    def intersects_polygon(self, box, polygon, min_ratio=0.02):
        x1, y1, x2, y2 = map(int, box)
        box_poly = np.array([
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2]
        ], dtype=np.int32)
        inter_area, _ = cv2.intersectConvexConvex(
            polygon.astype(np.float32),
            box_poly.astype(np.float32)
        )
        box_area = (x2 - x1) * (y2 - y1)
        if box_area <= 0:
            return False
        overlap_ratio = inter_area / box_area
        return overlap_ratio >= min_ratio
    def is_in_upper_half(self, point, ratio=0.5):
        left_point, right_point = self.compute_split_boundary(ratio)
        x1, y1 = left_point
        x2, y2 = right_point
        x, y = point
        A = y2 - y1
        B = x1 - x2
        C = x2 * y1 - x1 * y2
        value = A * x + B * y + C
        p0 = self.polygon[0]
        ref = A * p0[0] + B * p0[1] + C
        return value * ref > 0
