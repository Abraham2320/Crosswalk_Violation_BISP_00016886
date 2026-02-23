import numpy as np
import cv2


class ViolationDetector:
    """
    Violation rule (bidirectional, half-based):

    START:
      - vehicle state == "enter"
      - at least one pedestrian present
      - vehicle and pedestrian are in the same half of the crosswalk

    KEEP:
      - keep the violation while vehicle state == "inside"

    CLEAR:
      - remove violation when vehicle state == "outside"
    """

    def __init__(self, polygon, margin=0):
        """
        polygon: list of points (at least 4) describing the crosswalk polygon (screen coords)
        margin: not used now but kept for possible spatial tolerance later
        """
        self.active_violations = set()
        self.margin = margin
        self.polygon = polygon
        self._compute_axis()

    # --------------------------
    # Axis helpers (for debugging / optional future use)
    # --------------------------
    def _compute_axis(self):
        pts = np.array(self.polygon[:4], dtype=float)
        # entry = midpoint between p0 and p1; exit = midpoint between p2 and p3
        entry_mid = (pts[0] + pts[1]) / 2.0
        exit_mid = (pts[2] + pts[3]) / 2.0
        axis = exit_mid - entry_mid
        norm = np.linalg.norm(axis)
        if norm == 0:
            # fallback: horizontal axis to avoid crash
            self.axis_origin = entry_mid
            self.axis_vector = np.array([1.0, 0.0])
            return
        self.axis_origin = entry_mid
        self.axis_vector = axis / norm

    def project_progress(self, point):
        """Project a screen point onto the computed axis and return scalar progress."""
        vec = np.array(point, dtype=float) - self.axis_origin
        return float(np.dot(vec, self.axis_vector))

    # Violation logic
    def detect_violation(
    self,
    obj_id,
    obj_class,
    obj_state,
    vehicle_zone,
    pedestrians_data
    ):
        """
        START:
            - vehicle ENTERS
            - at least one pedestrian inside
            - EITHER:
                (a) same zone
                (b) pedestrian moving toward vehicle zone

        KEEP:
            - keep while vehicle INSIDE

        CLEAR:
            - remove when vehicle OUTSIDE
        """

        if obj_class != "vehicle":
            return False

        # ---------------- START ----------------
        if obj_state == "enter" and pedestrians_data:

            for ped in pedestrians_data:

                ped_zone = ped.get("zone")
                ped_direction = ped.get("direction")

                # Case 1: same zone
                if ped_zone == vehicle_zone:
                    self.active_violations.add(obj_id)
                    return True

                # Case 2: pedestrian moving toward vehicle zone
                if ped_direction == "DOWN" and vehicle_zone == "lower":
                    self.active_violations.add(obj_id)
                    return True

                if ped_direction == "UP" and vehicle_zone == "upper":
                    self.active_violations.add(obj_id)
                    return True

        # ---------------- KEEP ----------------
        if obj_id in self.active_violations and obj_state == "inside":
            return True

        # ---------------- CLEAR ----------------
        if obj_id in self.active_violations and obj_state == "outside":
            self.active_violations.discard(obj_id)

        return False    # --------------------------
    # Debug drawing
    # --------------------------
    def draw_axis(self, frame, length=400, color=(255, 0, 0)):
        """
        Draw the principal axis (arrow) for debugging.
        """
        origin = tuple(self.axis_origin.astype(int))
        end = tuple((self.axis_origin + self.axis_vector * length).astype(int))
        cv2.arrowedLine(frame, origin, end, color, 2, tipLength=0.08)