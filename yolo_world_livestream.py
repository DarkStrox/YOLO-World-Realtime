import cv2
import torch
from ultralytics import YOLOWorld

def run_livestream(source=0):
    """
    Real-time live streaming object detection with YOLO-World.
    source: 0 for default webcam, or path/URL to RTSP/HTTP video stream.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device.upper()}")

    prompt_classes = [
        # Spider-Man prompts
        "spider-man superhero in red and blue suit", 
        "man in spider-man suit", 
        "spider-man character",
        
        # Gun / Weapon prompts
        "gun barrel facing forward",
        "firearm held by driver or person",
        "handheld gun",
        "heavy weapon or rifle or shotgun",
        "gun",

        # Other requested classes
        "man", "child", "pedestrian", "helmet", 
        "armored vehicle", "police car", "yellow taxi", "black car", "van", "bus", "scooter",
        "skyscraper", "traffic light", "traffic cone", "construction barrier", 
        "spider web", "smoke", "fire", "tools"
    ]

    display_names = {
        0: "spider-man", 1: "spider-man", 2: "spider-man",
        3: "gun", 4: "gun", 5: "gun", 6: "gun", 7: "gun",
        8: "man", 9: "child", 10: "pedestrian", 11: "helmet",
        12: "armored vehicle", 13: "police car", 14: "yellow taxi", 15: "black car",
        16: "van", 17: "bus", 18: "scooter", 19: "skyscraper", 20: "traffic light",
        21: "traffic cone", 22: "construction barrier", 23: "spider web",
        24: "smoke", 25: "fire", 26: "tools"
    }

    print("Loading YOLO-World model with complete multi-prompt vocabulary...")
    model = YOLOWorld("yolov8s-worldv2.pt")
    model.set_classes(prompt_classes)
    model.model.names = display_names

    print(f"Starting real-time live stream from source: {source}")
    print("Press 'q' in the display window to exit.")

    results = model.predict(
        source=source,
        stream=True,
        show=True,
        conf=0.18,
        device=device
    )

    for r in results:
        pass

if __name__ == "__main__":
    video_file = "SPIDER-MAN_ BRAND NEW DAY  _ INTRO  (2026) Movie CLIP 4K.mp4"
    run_livestream(source=video_file)
