"""Bake a custom vocabulary into an open-vocabulary detector (YOLO-World).

YOLO11n only knows 80 fixed COCO classes. YOLO-World detects whatever words you
give here - including things COCO lacks (keys, remote control, cardboard box...).
The words are turned into CLIP text embeddings and frozen into the ONNX at export,
so the live app (detect.py) runs it on plain onnxruntime-directml with no CLIP or torch.

Run this once (needs CLIP + torch, CPU is fine), and again whenever you want to
change what it looks for:

    pip install ftfy regex
    pip install git+https://github.com/openai/CLIP.git --no-deps
    python export_world.py

Then set USE_WORLD = True in detect.py (or it auto-detects yolo_world.onnx).
"""
import json
import os

# Edit this list to change what the camera looks for. Keep it short-ish:
# every word is a class the detector scores on each frame.
VOCAB = [
    "person", "bottle", "cup", "laptop", "cell phone", "keyboard", "mouse",
    "book", "backpack", "chair", "keys", "remote control", "scissors",
    "knife", "wallet", "glasses", "cardboard box", "water bottle", "headphones",
]

MODEL = "yolov8s-worldv2.pt"        # v2: redesigned head, exports to ONNX where v1 fails
OUT = os.path.join(os.path.dirname(__file__), "yolo_world.onnx")
NAMES_OUT = os.path.join(os.path.dirname(__file__), "yolo_world.names.json")


def main():
    from ultralytics import YOLOWorld
    model = YOLOWorld(MODEL)
    model.set_classes(VOCAB)          # bakes CLIP embeddings of these words into the model
    # opset 12: the low-opset path Ultralytics 8.4.189+ fixed for YOLO-World/YOLOE export.
    path = model.export(format="onnx", opset=12, imgsz=640)
    # export() writes next to the .pt; move it to our fixed name so detect.py finds it.
    if os.path.abspath(path) != os.path.abspath(OUT):
        os.replace(path, OUT)
    with open(NAMES_OUT, "w", encoding="utf-8") as f:
        json.dump(VOCAB, f)
    print(f"Exported {OUT} with {len(VOCAB)} classes:")
    print("  " + ", ".join(VOCAB))
    print("Now set USE_WORLD = True in detect.py (or it auto-detects this file).")


if __name__ == "__main__":
    main()
