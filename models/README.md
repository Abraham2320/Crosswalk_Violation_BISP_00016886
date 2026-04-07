# Custom Model Directory

Place your trained YOLO model files here.

## License Plate Detection Model

Train a YOLOv8 model on license plate images (class 0 = plate bounding box).

After training, copy the `.pt` file here:

```
models/plate_detector.pt
```

Then set `PLATE_MODEL_PATH=models/plate_detector.pt` in your `.env` file.

The system will automatically use your custom model for plate detection. If the
file is not found, it falls back to the built-in OpenCV Haar cascade.

## Main Vehicle/Person Detection

The main detection model (`yolov8n.pt` or `yolov8x.pt`) lives in the project root.
Set `DETECTION_MODEL_PATH=yolov8x.pt` in `.env` to use the larger, more accurate model.
