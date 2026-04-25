import cv2
import numpy as np


class CrosswalkZone:
    def __init__(self, polygon):
        """
        polygon: list of at least 4 ordered points
        Expected order:
            p0 = top-left
            p1 = top-right
            p2 = bottom-right
            p3 = bottom-left
        """
        self.polygon = np.array(polygon[:4], dtype=np.int32)

    # ----------------------------------------------------
    # Draw full crosswalk polygon
    # ----------------------------------------------------
    def draw(self, frame):
        cv2.polylines(frame, [self.polygon], True, (255, 255, 0), 2)

    # ----------------------------------------------------
    # Bounding box intersection with full crosswalk
    # ----------------------------------------------------
    def intersects_box(self, box, min_ratio=0.02):
        """
        box: [x1, y1, x2, y2]
        min_ratio: minimum overlap ratio to count as inside
        """
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

    # ----------------------------------------------------
    # Compute horizontal split boundary inside polygon
    # ----------------------------------------------------
    def compute_split_boundary(self, ratio=0.5):
        """
        ratio = 0.5  → exact middle
        ratio < 0.5  → move line upward
        ratio > 0.5  → move line downward
        """

        p0, p1, p2, p3 = self.polygon

        # Interpolate along left edge (p0 -> p3)
        left_point = (
            int(p0[0] + ratio * (p3[0] - p0[0])),
            int(p0[1] + ratio * (p3[1] - p0[1]))
        )

        # Interpolate along right edge (p1 -> p2)
        right_point = (
            int(p1[0] + ratio * (p2[0] - p1[0])),
            int(p1[1] + ratio * (p2[1] - p1[1]))
        )

        return left_point, right_point

    # ----------------------------------------------------
    # Draw split line only
    # ----------------------------------------------------
    def draw_half_split(self, frame, ratio=0.5):
        left_point, right_point = self.compute_split_boundary(ratio)

        cv2.line(
            frame,
            left_point,
            right_point,
            (0, 255, 255),
            3
        )

    # ----------------------------------------------------
    # Return two sub-polygons (upper and lower halves)
    # ----------------------------------------------------
    def get_split_polygons(self, ratio=0.5):
        p0, p1, p2, p3 = self.polygon
        left_split, right_split = self.compute_split_boundary(ratio)

        upper = np.array([p0, p1, right_split, left_split], dtype=np.int32)
        lower = np.array([left_split, right_split, p2, p3], dtype=np.int32)

        return upper, lower

    # ----------------------------------------------------
    # Check if bounding box intersects given sub-polygon
    # ----------------------------------------------------
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

    # ----------------------------------------------------
    # Determine if a point lies in upper half
    # ----------------------------------------------------
    def is_in_upper_half(self, point, ratio=0.5):
        """
        Returns True if point lies above split line.
        """
        left_point, right_point = self.compute_split_boundary(ratio)

        x1, y1 = left_point
        x2, y2 = right_point
        x, y = point

        # Line equation Ax + By + C = 0
        A = y2 - y1
        B = x1 - x2
        C = x2 * y1 - x1 * y2

        value = A * x + B * y + C

        # Determine which side is "upper" by checking p0
        p0 = self.polygon[0]
        ref = A * p0[0] + B * p0[1] + C

        return value * ref > 0
