from __future__ import annotations

from collections import deque

import cv2

from config import (
    CONF_THRESHOLD,
    DETECTION_CLASSES,
    IMG_SIZE,
    MODEL_PATH,
    VIDEO_PATH,
    settings,
)
from detector.tracker import ObjectFSM
from detector.yolo_detector import YOLODetector
from geometry.crosswalk import CrosswalkZone
from geometry.polygon_editor import PolygonEditor
from logic.violation import ViolationDetector
from schemas import ViolationEvent
from services.pipeline import EnforcementPipeline
from vision.draw import draw_box


WINDOW_NAME = "Crosswalk Violation System"


def draw_zone_overlay(frame, polygon, color, alpha=0.2):
    overlay = frame.copy()
    cv2.fillPoly(overlay, [polygon], color)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.polylines(frame, [polygon], True, color, 2)


def build_event(frame_index, box, obj_id, vehicle_zone, trigger, polygon):
    x1, y1, x2, y2 = [int(value) for value in box]
    return ViolationEvent.create(
        vehicle_id=obj_id,
        frame_index=frame_index,
        vehicle_bbox=(x1, y1, x2, y2),
        vehicle_zone=vehicle_zone,
        polygon=[tuple(map(int, point)) for point in polygon],
        pedestrian_direction=trigger.pedestrian_direction,
        pedestrian_zone=trigger.pedestrian_zone,
        confidence=1.0,
        location=settings.runtime.location_name,
    )


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    editor = PolygonEditor(WINDOW_NAME)
    polygon_loaded = editor.load()
    cv2.setMouseCallback(WINDOW_NAME, editor.mouse_callback)

    if not polygon_loaded:
        print("Calibration mode:")
        print("LEFT click -> add polygon point")
        print("RIGHT click -> finish polygon")
        while True:
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError("Cannot read frame during calibration")

            editor.draw(frame)
            cv2.imshow(WINDOW_NAME, frame)
            cv2.waitKey(1)

            if editor.done:
                editor.save()
                break

    polygon = editor.get_polygon()
    if polygon is None:
        raise RuntimeError("Polygon missing or invalid")

    crosswalk = CrosswalkZone(polygon)
    upper_poly, lower_poly = crosswalk.get_split_polygons(ratio=settings.runtime.split_ratio)

    detector = YOLODetector(MODEL_PATH, DETECTION_CLASSES, CONF_THRESHOLD, IMG_SIZE)
    fsm = ObjectFSM()
    violation_detector = ViolationDetector(polygon)
    enforcement_pipeline = EnforcementPipeline(settings)

    pedestrians_progress = {}
    vehicles_inside = set()
    frame_index = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_index += 1
            results = detector.detect(frame)

            crosswalk.draw(frame)
            crosswalk.draw_half_split(frame, ratio=settings.runtime.split_ratio)
            draw_zone_overlay(frame, upper_poly, (255, 0, 0), alpha=0.15)
            draw_zone_overlay(frame, lower_poly, (0, 255, 0), alpha=0.15)

            cv2.putText(
                frame,
                f"P:{len(pedestrians_progress)} V:{len(vehicles_inside)}",
                (16, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )

            if results and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy().astype(int)
                ids = results[0].boxes.id.cpu().numpy().astype(int)

                for box, cls, obj_id in zip(boxes, classes, ids):
                    obj_class = "person" if cls == 0 else "vehicle"
                    x1, y1, x2, y2 = box
                    cx = int((x1 + x2) / 2)
                    cy = int(y2)

                    inside = (
                        crosswalk.intersects_box(box, min_ratio=0.005)
                        if obj_class == "person"
                        else crosswalk.intersects_box(box, min_ratio=0.02)
                    )
                    state = fsm.update(obj_id, inside)
                    violation_active = False

                    if obj_class == "person":
                        if inside:
                            ped_data = pedestrians_progress.get(obj_id)
                            if ped_data is None:
                                ped_data = {
                                    "id": obj_id,
                                    "y_history": deque(maxlen=settings.runtime.history_length),
                                    "zone": None,
                                    "direction": "STATIC",
                                }
                                pedestrians_progress[obj_id] = ped_data

                            ped_data["y_history"].append(cy)

                            zone = None
                            if crosswalk.intersects_polygon(box, upper_poly, 0.005):
                                zone = "upper"
                            elif crosswalk.intersects_polygon(box, lower_poly, 0.005):
                                zone = "lower"

                            ped_data["zone"] = zone
                            if len(ped_data["y_history"]) >= settings.runtime.history_length:
                                dy = ped_data["y_history"][-1] - ped_data["y_history"][0]
                                if abs(dy) > settings.runtime.pedestrian_direction_threshold:
                                    ped_data["direction"] = "DOWN" if dy > 0 else "UP"
                                else:
                                    ped_data["direction"] = "STATIC"
                        else:
                            pedestrians_progress.pop(obj_id, None)

                    else:
                        if state in ("enter", "inside"):
                            vehicles_inside.add(obj_id)
                        else:
                            vehicles_inside.discard(obj_id)

                        vehicle_zone = None
                        if crosswalk.intersects_polygon(box, upper_poly):
                            vehicle_zone = "upper"
                        elif crosswalk.intersects_polygon(box, lower_poly):
                            vehicle_zone = "lower"

                        violation_active, trigger = violation_detector.evaluate_vehicle(
                            obj_id=obj_id,
                            obj_class=obj_class,
                            obj_state=state,
                            vehicle_zone=vehicle_zone,
                            pedestrians_data=list(pedestrians_progress.values()),
                        )

                        if trigger is not None:
                            event = build_event(
                                frame_index=frame_index,
                                box=box,
                                obj_id=obj_id,
                                vehicle_zone=vehicle_zone,
                                trigger=trigger,
                                polygon=polygon,
                            )
                            enforcement_pipeline.submit_violation(frame.copy(), event)

                    box_color = (
                        (0, 0, 255)
                        if violation_active
                        else (0, 255, 255) if state in ("enter", "inside") else (0, 255, 0)
                    )
                    draw_box(frame, box, obj_class, box_color)

                    if obj_class == "person":
                        ped_data = pedestrians_progress.get(obj_id)
                        if ped_data:
                            cv2.putText(
                                frame,
                                ped_data["direction"],
                                (cx, cy - 40),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (255, 255, 0),
                                2,
                            )

                    cv2.putText(
                        frame,
                        state,
                        (cx, cy + 18),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                    )

                    if violation_active:
                        cv2.putText(
                            frame,
                            "VIOLATION",
                            (cx, cy - 25),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (0, 0, 255),
                            2,
                        )

            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(1) == 27:
                break
    finally:
        enforcement_pipeline.shutdown()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
