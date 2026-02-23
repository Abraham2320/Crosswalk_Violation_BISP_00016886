import cv2
from config import *

from detector.yolo_detector import YOLODetector
from detector.tracker import ObjectFSM

from geometry.crosswalk import CrosswalkZone
from geometry.polygon_editor import PolygonEditor
from logic.violation import ViolationDetector
from vision.draw import draw_box


WINDOW_NAME = "Crosswalk Violation System"


def draw_zone_overlay(frame, polygon, color, alpha=0.2):
    overlay = frame.copy()
    cv2.fillPoly(overlay, [polygon], color)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.polylines(frame, [polygon], True, color, 2)


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    # -------------------------
    # Polygon Calibration
    # -------------------------
    editor = PolygonEditor(WINDOW_NAME)
    polygon_loaded = editor.load()
    cv2.setMouseCallback(WINDOW_NAME, editor.mouse_callback)

    if not polygon_loaded:
        print("Calibration mode:")
        print("LEFT click  → add polygon point")
        print("RIGHT click → finish polygon")
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

    # Compute split zones ONCE
    upper_poly, lower_poly = crosswalk.get_split_polygons(ratio=0.32)

    detector = YOLODetector(
        MODEL_PATH,
        DETECTION_CLASSES,
        CONF_THRESHOLD,
        IMG_SIZE
    )

    fsm = ObjectFSM()
    violation_detector = ViolationDetector(polygon)

    pedestrians_zones = {}   # {ped_id: "upper"/"lower"}
    vehicles_inside = set()

    # -------------------------
    # Main Loop
    # -------------------------
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = detector.detect(frame)

        # Draw original crosswalk
        crosswalk.draw(frame)

        # Draw split line
        crosswalk.draw_half_split(frame, ratio=0.32)

        # Draw sub-zones
        draw_zone_overlay(frame, upper_poly, (255, 0, 0), alpha=0.15)
        draw_zone_overlay(frame, lower_poly, (0, 255, 0), alpha=0.15)

        cv2.putText(frame, f"P:{len(pedestrians_zones)} V:{len(vehicles_inside)}",
                    (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            ids = results[0].boxes.id.cpu().numpy().astype(int)

            for box, cls, obj_id in zip(boxes, classes, ids):

                obj_class = "person" if cls == 0 else "vehicle"

                x1, y1, x2, y2 = box
                cx = int((x1 + x2) / 2)
                cy = int(y2)

                # Geometry inside main crosswalk
                if obj_class == "person":
                    inside = crosswalk.intersects_box(box, min_ratio=0.005)
                else:
                    inside = crosswalk.intersects_box(box, min_ratio=0.02)

                state = fsm.update(obj_id, inside)

                violation_active = False

                # -------------------------
                # PEDESTRIAN LOGIC
                # -------------------------
                if obj_class == "person":

                    if inside:
                        zone = None
                        if crosswalk.intersects_polygon(box, upper_poly, 0.005):
                            zone = "upper"
                        elif crosswalk.intersects_polygon(box, lower_poly, 0.005):
                            zone = "lower"

                        if zone:
                            pedestrians_zones[obj_id] = zone
                        else:
                            pedestrians_zones.pop(obj_id, None)
                    else:
                        pedestrians_zones.pop(obj_id, None)

                # -------------------------
                # VEHICLE LOGIC
                # -------------------------
                else:

                    if state == "inside":
                        vehicles_inside.add(obj_id)
                    else:
                        vehicles_inside.discard(obj_id)

                    vehicle_zone = None
                    if crosswalk.intersects_polygon(box, upper_poly):
                        vehicle_zone = "upper"
                    elif crosswalk.intersects_polygon(box, lower_poly):
                        vehicle_zone = "lower"

                    violation_active = violation_detector.detect_violation(
                        obj_id=obj_id,
                        obj_class=obj_class,
                        obj_state=state,
                        vehicle_zone=vehicle_zone,
                        pedestrians_zones=list(pedestrians_zones.values())
                    )

                # -------------------------
                # DRAWING
                # -------------------------
                if violation_active:
                    box_color = (0, 0, 255)
                elif state in ("enter", "inside"):
                    box_color = (0, 255, 255)
                else:
                    box_color = (0, 255, 0)

                draw_box(frame, box, obj_class, box_color)

                cv2.putText(
                    frame,
                    state,
                    (cx, cy + 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )

                if violation_active:
                    cv2.putText(
                        frame,
                        "VIOLATION",
                        (cx, cy - 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 0, 255),
                        2
                    )

        cv2.imshow(WINDOW_NAME, frame)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()