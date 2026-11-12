"""Visual memory: a timestamped diary of what the camera saw.

Every observation is one line in memory/log.jsonl, for example
  {"time": "2026-11-13 14:05:33", "kind": "alert",
   "text": "bottle #4 disappeared; last seen at right (near laptop)", "image": "20261113-140533-alert.jpg"}
Snapshots are saved next to the log. The diary survives restarts, so the
system can answer "where did I leave my bottle?" later.
"""
import json
import os
import threading
import time

import cv2

DIR = os.path.join(os.path.dirname(__file__), "memory")
LOG = os.path.join(DIR, "log.jsonl")
STOPWORDS = {"where", "what", "when", "which", "there", "that", "this", "with", "have", "does", "were", "been"}

os.makedirs(DIR, exist_ok=True)
# shortcut: the whole diary is kept in RAM; fine to ~100k lines, switch to SQLite beyond that.
entries = []
if os.path.exists(LOG):
    with open(LOG, encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]
_lock = threading.Lock()  # two threads write here (tracker events, VLM descriptions)


def remember(kind, text, frame=None):
    """Add one observation; pass frame to keep a snapshot with it."""
    entry = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "kind": kind, "text": text}
    with _lock:
        if frame is not None:
            entry["image"] = time.strftime("%Y%m%d-%H%M%S") + f"-{kind}.jpg"
            cv2.imwrite(os.path.join(DIR, entry["image"]), frame)
        entries.append(entry)
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    return entry


def recall(question, recent=8, matching=8):
    """Diary lines to show the model for this question: the latest few, plus
    older lines that mention a word from the question (e.g. 'bottle')."""
    words = {w.strip("?.,!'\"").lower() for w in question.split()}
    words = {w for w in words if len(w) > 3 and w not in STOPWORDS}
    older = [e for e in entries[:-recent] if any(w in e["text"].lower() for w in words)][-matching:]
    return "\n".join(f"{e['time']} {e['text']}" for e in older + entries[-recent:]) or "(empty)"


if __name__ == "__main__":
    # Self-check in a throwaway folder so the real diary is untouched.
    import tempfile

    import numpy as np

    DIR = tempfile.mkdtemp()
    LOG = os.path.join(DIR, "log.jsonl")
    entries.clear()
    e = remember("alert", "bottle #4 disappeared; last seen at right (near laptop)", np.zeros((48, 64, 3), np.uint8))
    for i in range(20):
        remember("scene", f"A man sits at a desk, observation {i}.")
    assert os.path.exists(os.path.join(DIR, e["image"])), "snapshot not saved"
    with open(LOG, encoding="utf-8") as f:
        assert len(f.readlines()) == 21
    got = recall("Where did I leave my water bottle?")
    assert "bottle #4 disappeared" in got and "observation 19" in got and "observation 3." not in got, got
    print(got.splitlines()[0])
    print("OK: memory works.")
