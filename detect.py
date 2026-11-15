"""Fast object detection on the GPU, fixed 80 classes or an open vocabulary.

By default: YOLO11n, 80 everyday COCO classes, ~4 ms/frame on the RTX 5050 via
DirectML (Windows), ~33 ms on CPU. On a Raspberry Pi, `pip install onnxruntime`
gives the CPU path with the same code.

Open vocabulary: run `python export_world.py` to bake a custom word list into
yolo_world.onnx (YOLO-World; detects things COCO lacks, e.g. keys, wallet, box).
If that file exists it is used automatically, ~12 ms/frame; delete it to fall
back to YOLO11n. Both share the same output head, so the parsing below is identical.

OpenCV does the image preparation, duplicate-box removal (NMS) and drawing.
(OpenCV's own DNN module was tried first: the pip build has no CUDA, and its
OpenCL path was slower than CPU on this machine.)

How the raw output is read:
  the model returns 8400 candidate boxes; each is [cx, cy, w, h, <N> class scores]
  (N = 80 for YOLO11n, or the custom vocab size). We keep candidates whose best
  score > CONF, then NMS removes duplicate boxes that cover the same object.
"""
import os

import cv2
import numpy as np
import onnxruntime as ort

SIZE = 640   # this ONNX export only accepts 640x640 input
NMS = 0.45   # overlap above which two boxes count as the same object
# CONF is set below per model: YOLO-World scores lower than COCO-trained YOLO11n
# (it matches text embeddings), so it needs a lower bar to catch the same objects.

COCO = (
    "person,bicycle,car,motorcycle,airplane,bus,train,truck,boat,traffic light,"
    "fire hydrant,stop sign,parking meter,bench,bird,cat,dog,horse,sheep,cow,"
    "elephant,bear,zebra,giraffe,backpack,umbrella,handbag,tie,suitcase,frisbee,"
    "skis,snowboard,sports ball,kite,baseball bat,baseball glove,skateboard,"
    "surfboard,tennis racket,bottle,wine glass,cup,fork,knife,spoon,bowl,banana,"
    "apple,sandwich,orange,broccoli,carrot,hot dog,pizza,donut,cake,chair,couch,"
    "potted plant,bed,dining table,toilet,tv,laptop,mouse,remote,keyboard,"
    "cell phone,microwave,oven,toaster,sink,refrigerator,book,clock,vase,"
    "scissors,teddy bear,hair drier,toothbrush"
).split(",")

# Open-vocabulary: if export_world.py has produced yolo_world.onnx, use it and its
# custom word list; otherwise fall back to the fixed-80-class YOLO11n. Same output
# layout (YOLOv8/world head), so detect() below is unchanged either way.
WORLD = os.path.join(os.path.dirname(__file__), "yolo_world.onnx")
if os.path.exists(WORLD):
    MODEL = WORLD
    CONF = 0.25   # YOLO-World scores lower; 0.25 catches objects 0.4 would drop (measured)
    import json
    with open(os.path.join(os.path.dirname(__file__), "yolo_world.names.json"), encoding="utf-8") as _f:
        NAMES = json.load(_f)
else:
    MODEL = os.path.join(os.path.dirname(__file__), "yolo11n.onnx")
    CONF = 0.4
    NAMES = COCO

if not os.path.exists(MODEL):
    raise SystemExit(f"Missing {MODEL}. Make it with: yolo export model=yolo11n.pt format=onnx")

providers = ["CPUExecutionProvider"]
if "DmlExecutionProvider" in ort.get_available_providers():
    # high_performance = the NVIDIA GPU. Without it DirectML picks device 0,
    # which on this laptop is the slow Intel integrated GPU (26 ms vs 4 ms).
    providers.insert(0, ("DmlExecutionProvider", {"performance_preference": "high_performance"}))
session = ort.InferenceSession(MODEL, providers=providers)
INPUT = session.get_inputs()[0].name
DEVICE = "GPU" if session.get_providers()[0] == "DmlExecutionProvider" else "CPU"


def detect(frame):
    """Return [(name, confidence, (x, y, w, h)), ...] in frame pixel coordinates."""
    h, w = frame.shape[:2]
    # Pad to a square (not stretch) so objects keep their shape, as YOLO was trained.
    side = max(h, w)
    square = np.zeros((side, side, 3), np.uint8)
    square[:h, :w] = frame
    blob = cv2.dnn.blobFromImage(square, 1 / 255, (SIZE, SIZE), swapRB=True)
    out = session.run(None, {INPUT: blob})[0][0].T  # (8400, 84)
    scores = out[:, 4:]
    cls, conf = scores.argmax(1), scores.max(1)
    keep = conf > CONF
    out, cls, conf = out[keep], cls[keep], conf[keep]
    scale = side / SIZE
    boxes = [[int((cx - bw / 2) * scale), int((cy - bh / 2) * scale), int(bw * scale), int(bh * scale)]
             for cx, cy, bw, bh in out[:, :4]]
    idx = cv2.dnn.NMSBoxes(boxes, conf.tolist(), CONF, NMS)
    return [(NAMES[cls[i]], float(conf[i]), boxes[i]) for i in np.array(idx).flatten()]


def draw(img, detections):
    """Draw a green box and 'name 87%' label for each detection."""
    for name, conf, (x, y, bw, bh) in detections:
        cv2.rectangle(img, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        label = f"{name} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        x, y = max(x, 0), max(y, th + 6)  # keep the label on screen when the box touches an edge
        cv2.rectangle(img, (x, y - th - 6), (x + tw + 4, y), (0, 255, 0), -1)
        cv2.putText(img, label, (x + 2, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


if __name__ == "__main__":
    # python detect.py photo.jpg  -> boxes on a file; no argument -> one webcam frame.
    import sys

    if len(sys.argv) > 1:
        frame = cv2.imread(sys.argv[1])
    else:
        cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        for _ in range(10):  # let auto-exposure settle
            ok, frame = cam.read()
        cam.release()
    assert detect(np.zeros_like(frame)) == [], "blank image should have no detections"
    import time
    t = time.time()
    for _ in range(50):
        dets = detect(frame)
    print(f"YOLO on {DEVICE}: {(time.time() - t) / 50 * 1000:.1f} ms per frame (incl. pre/post-processing)")
    for name, conf, box in dets:
        print(f"{name:15} {conf:.0%}  {box}")
    draw(frame, dets)
    cv2.imwrite("detect_test.jpg", frame)
    print("Saved detect_test.jpg")
