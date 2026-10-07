import cv2
import numpy as np
from ultralytics import YOLOWorld

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[0])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[0])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0

def filter_detections(boxes, scores, class_ids, names):
    """
    Class Priority Filter:
    1. If Spider-Man is detected, suppress traffic light, fire, barrier, helmet, scooter on his body.
    2. Keep valid guns, cars, taxis, vans, people.
    """
    filtered = []
    spider_boxes = []

    # First pass: collect spider-man boxes
    for b, s, c in zip(boxes, scores, class_ids):
        lbl = names.get(int(c), "")
        if lbl == "spider-man" and s >= 0.22:
            spider_boxes.append((b, s, c))
            filtered.append((b, s, c))

    # Second pass: check other detections
    for b, s, c in zip(boxes, scores, class_ids):
        lbl = names.get(int(c), "")
        if lbl == "spider-man":
            continue

        # Suppress false positives on Spider-Man's body
        if lbl in ["traffic light", "fire", "construction barrier", "helmet", "scooter"]:
            if any(compute_iou(b, sb[0]) > 0.35 for sb in spider_boxes):
                continue

        # Suppress weak gun noise on Spider-Man's body unless confidence >= 0.40
        if lbl == "gun" and s < 0.35:
            if any(compute_iou(b, sb[0]) > 0.45 for sb in spider_boxes):
                continue

        filtered.append((b, s, c))

    return filtered

print("Custom pipeline helper ready.")
