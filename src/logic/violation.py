from __future__ import annotations

from typing import Iterable, Optional

import cv2
import numpy as np

from schemas import ViolationTrigger


class ViolationDetector:
    """
    Bidirectional half-zone violation detector with transition-aware triggers.

    `detect_violation` preserves the legacy boolean interface.
    `evaluate_vehicle` adds an event trigger so downstream services can capture
    evidence only once when a vehicle transitions from not violating to violating.
    """

    def __init__(self, polygon, margin: int = 0):
        self.active_violations: set[int] = set()
        self.margin = margin
        self.polygon = polygon
        self._compute_axis()

    def _compute_axis(self) -> None:
        pts = np.array(self.polygon[:4], dtype=float)
        entry_mid = (pts[0] + pts[1]) / 2.0
        exit_mid = (pts[2] + pts[3]) / 2.0
        axis = exit_mid - entry_mid
        norm = np.linalg.norm(axis)
        if norm == 0:
            self.axis_origin = entry_mid
            self.axis_vector = np.array([1.0, 0.0])
            return
        self.axis_origin = entry_mid
        self.axis_vector = axis / norm

    def project_progress(self, point):
        vec = np.array(point, dtype=float) - self.axis_origin
        return float(np.dot(vec, self.axis_vector))

    def _find_trigger(
        self,
        vehicle_zone: Optional[str],
        pedestrians_data: Iterable[dict],
    ) -> Optional[ViolationTrigger]:
        for ped in pedestrians_data:
            ped_zone = ped.get("zone")
            ped_direction = ped.get("direction", "STATIC")
            ped_id = int(ped.get("id", -1))

            if ped_zone == vehicle_zone and ped_zone is not None:
                return ViolationTrigger(
                    vehicle_id=ped_id,
                    vehicle_zone=vehicle_zone,
                    pedestrian_direction=ped_direction,
                    pedestrian_zone=ped_zone,
                    reason="same_zone",
                )

            if ped_direction == "DOWN" and vehicle_zone == "lower":
                return ViolationTrigger(
                    vehicle_id=ped_id,
                    vehicle_zone=vehicle_zone,
                    pedestrian_direction=ped_direction,
                    pedestrian_zone=ped_zone,
                    reason="pedestrian_approaching_vehicle_zone",
                )

            if ped_direction == "UP" and vehicle_zone == "upper":
                return ViolationTrigger(
                    vehicle_id=ped_id,
                    vehicle_zone=vehicle_zone,
                    pedestrian_direction=ped_direction,
                    pedestrian_zone=ped_zone,
                    reason="pedestrian_approaching_vehicle_zone",
                )

        return None

    def evaluate_vehicle(
        self,
        obj_id: int,
        obj_class: str,
        obj_state: str,
        vehicle_zone: Optional[str],
        pedestrians_data: Iterable[dict],
    ) -> tuple[bool, Optional[ViolationTrigger]]:
        if obj_class != "vehicle":
            return False, None

        trigger = None
        if obj_state == "enter" and pedestrians_data:
            trigger = self._find_trigger(vehicle_zone, pedestrians_data)
            if trigger is not None:
                self.active_violations.add(obj_id)
                return True, trigger

        if obj_id in self.active_violations and obj_state in ("inside", "exit"):
            return True, None

        if obj_id in self.active_violations and obj_state == "outside":
            self.active_violations.discard(obj_id)

        return False, None

    def detect_violation(
        self,
        obj_id,
        obj_class,
        obj_state,
        vehicle_zone,
        pedestrians_data,
    ):
        is_violating, _ = self.evaluate_vehicle(
            obj_id=obj_id,
            obj_class=obj_class,
            obj_state=obj_state,
            vehicle_zone=vehicle_zone,
            pedestrians_data=pedestrians_data,
        )
        return is_violating

    def draw_axis(self, frame, length=400, color=(255, 0, 0)):
        origin = tuple(self.axis_origin.astype(int))
        end = tuple((self.axis_origin + self.axis_vector * length).astype(int))
        cv2.arrowedLine(frame, origin, end, color, 2, tipLength=0.08)
