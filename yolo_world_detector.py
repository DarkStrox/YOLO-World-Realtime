import os
import sys
import time
import subprocess
import cv2
import torch
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

    # Collect spider-man boxes
    for b, s, c in zip(boxes, scores, class_ids):
        lbl = names.get(int(c), "")
        if lbl == "spider-man" and s >= 0.20:
            spider_boxes.append((b, s, c))
            filtered.append((b, s, c))

    # Process other detections
    for b, s, c in zip(boxes, scores, class_ids):
        lbl = names.get(int(c), "")
        if lbl == "spider-man":
            continue

        # Suppress false positive street labels on Spider-Man's body
        if lbl in ["traffic light", "fire", "construction barrier", "helmet", "scooter"]:
            if any(compute_iou(b, sb[0]) > 0.30 for sb in spider_boxes):
                continue

        # Suppress weak gun noise on Spider-Man's body unless score >= 0.35
        if lbl == "gun" and s < 0.35:
            if any(compute_iou(b, sb[0]) > 0.40 for sb in spider_boxes):
                continue

        filtered.append((b, s, c))

    return filtered

def main():
    prompt_classes = [
        # Spider-Man prompts
        "spider-man superhero in red and blue suit", 
        "man in spider-man suit", 
        "spider-man character",
        "head of spider-man mask",
        "spider-man suit leg or boot in motion",
        "spider-man in dark alley",

        # Gun / Weapon prompts
        "grenade launcher weapon",
        "gun barrel facing forward",
        "handheld gun or rifle or firearm",
        "metallic firearm weapon held by driver or person",

        # Yellow Taxi
        "bright yellow NYC taxi cab with roof sign",
        "yellow taxi cab",

        # Van
        "van or white cargo delivery van seen from top down view",
        "white van",

        # Cars
        "distant car on street",
        "silver or grey sedan car",
        "car or automobile",
        "black sedan car",

        # Other categories with disambiguated prompts
        "man in tactical vest or coat", "man", "child", "pedestrian walking on street", "helmet on head", 
        "military SWAT armored tank vehicle on road", "police car", "bus", "riding motor scooter",
        "skyscraper", "city traffic light signal mounted on post", "orange or yellow street construction barrier fence", 
        "spider web", "flames of fire burning", "tools"
    ]

    display_names = {
        0: "spider-man", 1: "spider-man", 2: "spider-man", 3: "spider-man", 4: "spider-man", 5: "spider-man",
        6: "gun", 7: "gun", 8: "gun", 9: "gun",
        10: "yellow taxi", 11: "yellow taxi",
        12: "van", 13: "van",
        14: "car", 15: "car", 16: "car",
        17: "black car",
        18: "man", 19: "man", 20: "child", 21: "pedestrian", 22: "helmet",
        23: "armored vehicle", 24: "police car", 25: "bus", 26: "scooter",
        27: "skyscraper", 28: "traffic light", 29: "construction barrier",
        30: "spider web", 31: "fire", 32: "tools"
    }

    colors = {
        "spider-man": (255, 0, 0),      # Blue
        "gun": (0, 0, 255),             # Red
        "yellow taxi": (0, 255, 255),   # Yellow
        "van": (255, 0, 255),          # Magenta
        "car": (0, 255, 0),             # Green
        "black car": (0, 200, 100),     # Dark Green
        "man": (255, 255, 0),           # Cyan
        "pedestrian": (255, 128, 0),    # Orange
        "armored vehicle": (128, 0, 255),
        "skyscraper": (200, 200, 200)
    }

    video_filename = "SPIDER-MAN_ BRAND NEW DAY  _ INTRO  (2026) Movie CLIP 4K.mp4"
    video_path = os.path.join(os.path.dirname(__file__), video_filename)

    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        sys.exit(1)

    print("Initializing YOLO-World model with Master Priority Pipeline...")
    model = YOLOWorld("yolov8s-worldv2.pt")
    model.set_classes(prompt_classes)
    model.model.names = display_names

    output_dir = os.path.join(os.path.dirname(__file__), "runs", "detect")
    os.makedirs(output_dir, exist_ok=True)
    temp_avi = os.path.join(output_dir, "temp_clean_render.avi")
    final_output_mp4 = os.path.join(output_dir, "SPIDER-MAN_DETECTION_WITH_SOUND.mp4")

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(temp_avi, fourcc, fps, (width, height))

    print(f"Starting GPU frame-by-frame clean detection loop ({total_frames} frames)...")
    start_time = time.time()
    frame_count = 0

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        res = model.predict(frame, conf=0.15, verbose=False, device=device)[0]

        if len(res.boxes) > 0:
            boxes = res.boxes.xyxy.cpu().numpy()
            scores = res.boxes.conf.cpu().numpy()
            class_ids = res.boxes.cls.cpu().numpy()

            filtered = filter_detections(boxes, scores, class_ids, display_names)

            for b, s, c in filtered:
                x1, y1, x2, y2 = map(int, b)
                label = display_names.get(int(c), f"class_{int(c)}")
                color = colors.get(label, (0, 255, 0))

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                text = f"{label} {s:.2f}"

                (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - 22), (x1 + w + 6, y1), color, -1)
                cv2.putText(frame, text, (x1 + 3, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        out.write(frame)

        if frame_count % 500 == 0:
            print(f"Processed {frame_count}/{total_frames} frames...")

    cap.release()
    out.release()

    elapsed_time = time.time() - start_time
    print(f"\nCustom Clean Render completed in {elapsed_time:.2f} seconds.")

    print("Remuxing 100% working stereo audio track at 12 Mbps Crystal Clear HD quality...")
    cmd = [
        "ffmpeg", "-y",
        "-i", temp_avi,
        "-i", video_path,
        "-c:v", "h264_mf",
        "-b:v", "12M",
        "-c:a", "aac",
        "-b:a", "192k",
        "-ar", "44100",
        "-ac", "2",
        "-map", "0:v:0",
        "-map", "1:a:0?",
        final_output_mp4
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"\n🎉 PERFECT CRYSTAL-CLEAR HD VIDEO WITH SOUND SAVED AT:\n{final_output_mp4}")
    except Exception as e:
        print(f"FFmpeg error: {e}")

    if os.path.exists(temp_avi):
        try:
            os.remove(temp_avi)
        except:
            pass

if __name__ == "__main__":
    main()
