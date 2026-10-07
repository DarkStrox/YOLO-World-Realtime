# YOLO-World Real-Time Detection

Real-time zero-shot open-vocabulary object detection pipeline powered by Ultralytics YOLO-World.

## Features
- **Zero-Shot Object Detection**: Detect novel classes dynamically by changing natural language prompts without retraining.
- **Livestream & Webcam Support**: Low-latency video streaming inference (`yolo_world_livestream.py`).
- **Custom Filtering**: Dynamic confidence and class filters (`custom_filter.py`) for specific domain tracking.

## Usage
```bash
pip install ultralytics opencv-python torch
python yolo_world_detector.py
```
