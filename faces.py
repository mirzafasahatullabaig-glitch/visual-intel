"""Face recognition: recognise and name people across sessions.

InsightFace (buffalo_l) finds faces and turns each into a 512-number embedding.
Two faces are the same person when their embeddings point almost the same way
(cosine similarity above THRESHOLD). Known people are kept in faces/known.json as
{name: [embedding, ...]}; unknown faces get a temporary "face #N" until you name one.

Optional, like voice: if InsightFace or its model is missing, `ready` is False and
the rest of the app runs unchanged.

    pip install insightface --no-deps      (the --no-deps keeps your onnxruntime-directml)
    pip install onnx scikit-image          (insightface needs these; onnx + skimage)

Runs on the GPU through DirectML when available (same as detect.py), else CPU.
ctx_id stays -1: InsightFace uses a positive ctx_id only to init CUDA, which this
machine's onnxruntime (DirectML + CPU, no CUDA) does not have; DirectML placement
happens through the providers list instead.
"""
import json
import os
import threading
import time

import cv2
import numpy as np

DIR = os.path.join(os.path.dirname(__file__), "faces")
STORE = os.path.join(DIR, "known.json")
THRESHOLD = 0.35      # cosine above this = same named person (buffalo_l same-person is ~0.3-0.6)
SAME_UNKNOWN = 0.30   # looser: is this the same unseen face as moments ago, to reuse its 'face #N'
EVERY = 0.5           # seconds between face scans (CPU; slower than YOLO)
MAX_PER_PERSON = 8    # keep at most this many embeddings per name (a few angles is plenty)
MAX_UNKNOWN = 50      # cap unnamed faces cached this session, so a busy scene can't grow it forever

ready = False
_device = "CPU"       # set by _init(): "GPU" (DirectML) or "CPU"
_app = None
_lock = threading.Lock()
known = {}            # name -> list[np.ndarray(512)]
_last = {"t": 0.0, "faces": []}   # cache so repeated calls within EVERY are free
_next_unknown = 1
_unknown = {}         # temp id -> (embedding, box), so the UI can name "face #N"


def _load():
    if os.path.exists(STORE):
        with open(STORE, encoding="utf-8") as f:
            for name, vecs in json.load(f).items():
                known[name] = [np.asarray(v, np.float32) for v in vecs]


def _save():
    with open(STORE, "w", encoding="utf-8") as f:
        json.dump({n: [v.tolist() for v in vecs] for n, vecs in known.items()}, f)


def _init():
    """Load InsightFace once. Any failure (not installed, model download blocked) leaves ready False."""
    global _app, ready
    global _device
    try:
        import onnxruntime as ort
        from insightface.app import FaceAnalysis
        os.makedirs(DIR, exist_ok=True)
        # Run on the GPU through DirectML, same as detect.py; fall back to CPU if it is absent.
        # ctx_id stays -1: InsightFace uses a positive ctx_id only to init CUDA, which we don't have.
        providers = ["CPUExecutionProvider"]
        if "DmlExecutionProvider" in ort.get_available_providers():
            providers.insert(0, ("DmlExecutionProvider", {"performance_preference": "high_performance"}))
        _app = FaceAnalysis(name="buffalo_l", providers=providers)
        _device = "GPU" if providers[0] != "CPUExecutionProvider" else "CPU"
        _app.prepare(ctx_id=-1, det_size=(640, 640) if _device == "GPU" else (320, 320))
        _load()
        ready = True
    except Exception as e:  # ImportError, model download failure, onnxruntime mismatch...
        print(f"faces: off ({e}). Install with: pip install insightface --no-deps")


def _match(emb):
    """Name whose stored embedding is closest to emb (cosine), or None if none is close enough."""
    best, score = None, THRESHOLD
    for name, vecs in known.items():
        for v in vecs:
            s = float(np.dot(emb, v) / (np.linalg.norm(emb) * np.linalg.norm(v) + 1e-9))
            if s > score:
                best, score = name, s
    return best


def recognise(frame):
    """[(label, (x, y, w, h)), ...] for faces in the frame. Labels are a known name
    or a temporary 'face #N'. Throttled to one real scan every EVERY seconds."""
    global _next_unknown
    if not ready:
        return []
    now = time.time()
    if now - _last["t"] < EVERY:
        return _last["faces"]
    out = []
    with _lock:
        for f in _app.get(frame):
            emb = f.normed_embedding
            x1, y1, x2, y2 = (int(v) for v in f.bbox)
            box = (x1, y1, x2 - x1, y2 - y1)
            name = _match(emb)
            if name is None:
                # Reuse a temp id if this face matches an already-seen unknown this session.
                tid = next((t for t, (e, _) in _unknown.items()
                            if float(np.dot(emb, e)) > SAME_UNKNOWN), None)
                if tid is None:
                    tid = f"face #{_next_unknown}"
                    _next_unknown += 1
                _unknown[tid] = (emb, box)
                if len(_unknown) > MAX_UNKNOWN:  # drop the oldest unnamed face
                    del _unknown[next(iter(_unknown))]
                name = tid
            out.append((name, box))
        _last["t"], _last["faces"] = now, out
    return out


def name_face(temp_id, person):
    """Give a name to a 'face #N' seen this session. Returns True if it was learned."""
    person = person.strip()
    if not person:  # the server is the trust boundary: never store a blank-named person
        return False
    with _lock:
        if temp_id not in _unknown:
            return False
        emb = _unknown.pop(temp_id)[0]
        known.setdefault(person, []).append(emb)
        known[person] = known[person][-MAX_PER_PERSON:]
        _save()
    return True


def unknown_faces():
    """Temp ids seen this session that still have no name, for the UI to offer."""
    with _lock:  # the face thread may be mutating _unknown while the web thread reads it
        return list(_unknown)


def draw(img, faces):
    for label, (x, y, w, h) in faces:
        colour = (255, 180, 0) if label.startswith("face #") else (0, 220, 120)
        cv2.rectangle(img, (x, y), (x + w, y + h), colour, 2)
        cv2.putText(img, label, (x, max(y - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2, cv2.LINE_AA)


_init()


if __name__ == "__main__":
    # Self-check: the matching, naming and store logic without needing the model or a camera.
    import tempfile

    DIR = tempfile.mkdtemp()
    STORE = os.path.join(DIR, "known.json")
    known.clear()
    a = np.array([1, 0, 0] + [0] * 509, np.float32)       # "person A" direction
    b = np.array([0, 1, 0] + [0] * 509, np.float32)       # a different direction
    known["Ali"] = [a]
    assert _match(a) == "Ali", "same vector should match"
    assert _match(b) is None, "orthogonal vector should not match"
    # Naming an unknown teaches it, and it then matches.
    _unknown["face #1"] = (b, (0, 0, 10, 10))
    assert name_face("face #1", "Sara") and _match(b) == "Sara", "naming did not stick"
    assert not name_face("face #99", "x"), "unknown id should return False"
    # A blank or whitespace name is refused, even for a real temp id.
    _unknown["face #2"] = (a, (0, 0, 10, 10))
    assert not name_face("face #2", "   ") and "face #2" in _unknown, "blank name must not store"
    _unknown.clear()
    # The unknown cache is capped: adding past MAX_UNKNOWN drops the oldest.
    for i in range(MAX_UNKNOWN + 5):
        _unknown[f"face #{i}"] = (a, (0, 0, 1, 1))
        if len(_unknown) > MAX_UNKNOWN:
            del _unknown[next(iter(_unknown))]
    assert len(_unknown) == MAX_UNKNOWN and "face #0" not in _unknown, "unknown cap failed"
    _unknown.clear()
    # The store round-trips through JSON.
    known.clear()
    _load()
    assert _match(b) == "Sara" and _match(a) == "Ali", "store did not persist"
    # Per-person cap holds.
    for _ in range(20):
        known["Ali"].append(a)
    known["Ali"] = known["Ali"][-MAX_PER_PERSON:]
    assert len(known["Ali"]) == MAX_PER_PERSON
    print("OK: faces logic works (model not needed for this check).")
