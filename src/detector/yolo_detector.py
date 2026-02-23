from ultralytics import YOLO

class YOLODetector:
    def __init__(self, model_path, classes, conf, imgsz):
        self.model = YOLO(model_path)
        self.classes = classes
        self.conf = conf
        self.imgsz = imgsz

    def detect(self, frame):
        return self.model.track(
            frame,
            persist=True,
            conf=self.conf,
            imgsz=self.imgsz,
            classes=self.classes
        )