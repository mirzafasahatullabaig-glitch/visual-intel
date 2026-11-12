"""Object tracking: follow each object from frame to frame so it keeps an ID.

YOLO alone only says "there is a person in THIS frame". The tracker links
detections across frames: a new box that overlaps a box from the previous
frame (same class) is the same object. That gives us:
  - a stable ID per object ("person #1")
  - 'entered' and 'left' events
  - how many of each object are visible, and where each one is
"""
import time

CONFIRM = 15    # frames (~0.5 s) an object must be seen before it counts (ignores flickers)
FORGET = 3.0    # seconds unseen before an object is declared gone (bridges brief misses)
MATCH_IOU = 0.2  # minimum overlap to call two boxes the same object
NEAR = 0.4      # two objects are 'near' if their centres are within this fraction of the frame width


def iou(a, b):
    """Overlap of two (x, y, w, h) boxes: 0 = apart, 1 = identical."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union else 0


def centre(box):
    return box[0] + box[2] / 2, box[1] + box[3] / 2


def where(box, frame_w, frame_h):
    """Position of a box in the picture, e.g. 'top-left', 'right', 'centre'.

    The camera's view: 'left' means the left side of the image.
    """
    cx, cy = centre(box)
    col = "left" if cx < frame_w / 3 else "right" if cx > frame_w * 2 / 3 else ""
    row = "top" if cy < frame_h / 3 else "bottom" if cy > frame_h * 2 / 3 else ""
    return "-".join(p for p in (row, col) if p) or "centre"


class Tracker:
    def __init__(self):
        self.tracks = []   # each track is a dict, see update()
        self.next_id = 1

    def update(self, detections, frame_w, frame_h, now=None):
        """Feed one frame's detections. Returns events: [('entered' | 'left', track), ...]."""
        now = time.time() if now is None else now
        events = []
        unmatched = list(self.tracks)
        for name, conf, box in sorted(detections, key=lambda d: -d[1]):
            # Best-overlapping existing track of the same class, if any.
            best = max((t for t in unmatched if t["name"] == name),
                       key=lambda t: iou(t["box"], box), default=None)
            if best and iou(best["box"], box) >= MATCH_IOU:
                unmatched.remove(best)
                best["hits"] += 1
            else:
                best = {"id": 0, "name": name, "hits": 1, "first": now, "confirmed": False, "near": ""}
                self.tracks.append(best)
            best.update(box=box, conf=conf, last=now, where=where(box, frame_w, frame_h))
            if not best["confirmed"] and best["hits"] >= CONFIRM:
                best["confirmed"] = True
                best["id"], self.next_id = self.next_id, self.next_id + 1  # ID only once it is real
                events.append(("entered", best))

        for t in [t for t in self.tracks if now - t["last"] > FORGET]:
            self.tracks.remove(t)
            if t["confirmed"]:
                events.append(("left", t))

        # Remember each object's nearest neighbour ("bottle near laptop") for memory and questions.
        seen = self.visible(now)
        for t in seen:
            tx, ty = centre(t["box"])
            def dist(o):
                ox, oy = centre(o["box"])
                return ((ox - tx) ** 2 + (oy - ty) ** 2) ** 0.5
            nearest = min((o for o in seen if o is not t), key=dist, default=None)
            t["near"] = nearest["name"] if nearest and dist(nearest) < frame_w * NEAR else ""
        return events

    def visible(self, now=None):
        """Confirmed tracks seen in the last 0.3 s (what is in view right now)."""
        now = time.time() if now is None else now
        return [t for t in self.tracks if t["confirmed"] and now - t["last"] < 0.3]


def label(t):
    return f"{t['name']} #{t['id']}"


def counts(tracks):
    """'2 person, 1 cup' for a list of tracks."""
    names = [t["name"] for t in tracks]
    return ", ".join(f"{names.count(n)} {n}" for n in dict.fromkeys(names))


def summary(tracks):
    """One line for the VLM prompt and memory: 'person #1 at centre; cup #2 at right (near person)'."""
    return "; ".join(f"{label(t)} at {t['where']}" + (f" (near {t['near']})" if t["near"] else "")
                     for t in tracks) or "nothing detected"


if __name__ == "__main__":
    # Self-check with made-up boxes and a fake clock (no camera needed).
    tr, t0 = Tracker(), 1000.0
    events = []
    for i in range(20):  # a cup slides right, 20 frames; a person stands still
        dets = [("cup", 0.9, [100 + 10 * i, 300, 60, 60]), ("person", 0.8, [300, 50, 200, 400])]
        events += tr.update(dets, 640, 480, now=t0 + i / 30)
    assert [(k, label(t)) for k, t in events] == [("entered", "cup #1"), ("entered", "person #2")], events
    assert counts(tr.visible(t0 + 19 / 30)) == "1 cup, 1 person"
    cup = tr.tracks[0]
    assert cup["where"] == "bottom" and cup["near"] == "person", cup  # slid from left into the middle third
    # The cup disappears; the person stays. After FORGET seconds the cup has 'left'.
    events = []
    for i in range(20, 150):
        events += tr.update([("person", 0.8, [300, 50, 200, 400])], 640, 480, now=t0 + i / 30)
    assert [(k, label(t)) for k, t in events] == [("left", "cup #1")], events
    assert summary(tr.visible(t0 + 149 / 30)) == "person #2 at centre", "stale 'near cup' must be cleared"
    # A 2-frame flicker never gets confirmed, so it makes no events.
    tr2 = Tracker()
    ev = tr2.update([("dog", 0.5, [0, 0, 50, 50])], 640, 480, now=t0)
    ev += tr2.update([], 640, 480, now=t0 + 5)
    assert ev == [] and where([0, 0, 50, 50], 640, 480) == "top-left"
    print("summary:", summary(tr.visible(t0 + 149 / 30)))
    print("OK: tracker works.")
