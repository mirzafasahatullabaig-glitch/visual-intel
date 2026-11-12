"""Fast object detection: OpenCV DNN + YOLO11n (80 everyday COCO classes).

~34 ms per frame on CPU, so it can box objects on every frame while the
slower VLM describes the scene. Pure OpenCV (no PyTorch), so it also runs
on a Raspberry Pi.

How YOLO's raw output is read:
  the model returns 8400 candidate boxes; each is [cx, cy, w, h, 80 class scores].
  We keep candidates whose best score > CONF, then NMS removes duplicate boxes
  that cover the same object.
"""
import os

import cv2
import numpy as np

MODEL = os.path.join(os.path.dirname(__file__), "yolo11n.onnx")
SIZE = 640   # this ONNX export only accepts 640x640 input
CONF = 0.4   # minimum confidence to show a box
NMS = 0.45   # overlap above which two boxes count as the same object

NAMES = (
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

if not os.path.exists(MODEL):
    raise SystemExit(f"Missing {MODEL}. Make it with: yolo export model=yolo11n.pt format=onnx")
net = cv2.dnn.readNetFromONNX(MODEL)


def detect(frame):
    """Return [(name, confidence, (x, y, w, h)), ...] in frame pixel coordinates."""
    h, w = frame.shape[:2]
    # Pad to a square (not stretch) so objects keep their shape, as YOLO was trained.
    side = max(h, w)
    square = np.zeros((side, side, 3), np.uint8)
    square[:h, :w] = frame
    net.setInput(cv2.dnn.blobFromImage(square, 1 / 255, (SIZE, SIZE), swapRB=True))
    out = net.forward()[0].T                      # (8400, 84)
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
    dets = detect(frame)
    for name, conf, box in dets:
        print(f"{name:15} {conf:.0%}  {box}")
    draw(frame, dets)
    cv2.imwrite("detect_test.jpg", frame)
    print("Saved detect_test.jpg")
