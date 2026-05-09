import cv2
import json
import os
import numpy as np
class PolygonEditor:
    def __init__(self, window_name, save_path="crosswalk_polygon.json"):
        self.window_name = window_name
        self.save_path = save_path
        self.points = []
        self.done = False
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.done = True
    def draw(self, frame):
        for p in self.points:
            cv2.circle(frame, p, 5, (0, 0, 255), -1)
        if len(self.points) > 1:
            cv2.polylines(
                frame,
                [np.array(self.points, dtype=np.int32)],
                False,
                (255, 255, 0),
                2
            )
    def save(self):
        with open(self.save_path, "w") as f:
            json.dump(self.points, f)
    def load(self):
        if os.path.exists(self.save_path):
            with open(self.save_path, "r") as f:
                self.points = [tuple(p) for p in json.load(f)]
            return True
        return False
    def get_polygon(self):
        if len(self.points) >= 3:
            return np.array(self.points, dtype=np.int32)
        return None
