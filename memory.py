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
STOPWORDS = {"where", "what", "when", "which", "there", "that", "this", "with", "have", "does", "were", "been",
             "the", "was", "did", "you", "are", "any", "has", "for", "and", "not", "who", "how", "why", "can",
             "see", "put", "get", "its", "his", "her", "our", "out", "now"}
# Words the camera never uses (YOLO's labels) matched to how a person asks for them.
# Each line is one group: asking for any word in it searches the diary for all of them.
SYNONYMS = [{"bottle", "drink", "water", "flask"}, {"laptop", "computer", "pc", "macbook"},
            {"cell", "phone", "mobile", "smartphone"}, {"person", "people", "someone", "somebody", "man", "woman"},
            {"cup", "mug", "coffee", "tea"}, {"remote", "controller"}, {"tv", "television", "monitor", "screen"},
            {"book", "notebook"}, {"backpack", "bag"}, {"keyboard", "keys"}]
SNAPSHOT_DAYS = 14   # delete saved snapshots older than this on startup
MAX_ENTRIES = 50000  # cap the diary: oldest lines drop out of RAM and the log is rewritten


def load(path):
    """Read the diary file. A line cut short by a crash or power loss is skipped, not fatal."""
    found = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    found.append(json.loads(line))
                except ValueError:
                    pass
    return found


def prune_snapshots(days=SNAPSHOT_DAYS):
    """Delete .jpg snapshots older than `days`. The log keeps the text line either way;
    only the image goes, so old entries stay searchable, they just lose their picture."""
    cutoff = time.time() - days * 86400
    removed = 0
    for name in os.listdir(DIR) if os.path.isdir(DIR) else []:
        path = os.path.join(DIR, name)
        if name.endswith(".jpg") and os.path.getmtime(path) < cutoff:
            os.remove(path)
            removed += 1
    return removed


os.makedirs(DIR, exist_ok=True)
# shortcut: the whole diary is kept in RAM; fine to ~100k lines, switch to SQLite beyond that.
entries = load(LOG)
prune_snapshots()
_lock = threading.Lock()  # two threads write here (tracker events, VLM descriptions)


def remember(kind, text, frame=None):
    """Add one observation; pass frame to keep a snapshot with it."""
    entry = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "kind": kind, "text": text}
    with _lock:
        if frame is not None:
            entry["image"] = time.strftime("%Y%m%d-%H%M%S") + f"-{kind}.jpg"
            cv2.imwrite(os.path.join(DIR, entry["image"]), frame)
        entries.append(entry)
        if len(entries) > MAX_ENTRIES:  # drop the oldest and rewrite the log to match
            del entries[:-MAX_ENTRIES]
            with open(LOG, "w", encoding="utf-8") as f:
                f.writelines(json.dumps(e) + "\n" for e in entries)
        else:
            with open(LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
    return entry


def recall(question, recent=8, matching=8):
    """Diary lines to show the model for this question: the latest few, plus
    older lines that mention a word from the question (e.g. 'bottle')."""
    words = {w.strip("?.,!'\"").lower() for w in question.split()}
    words = {w for w in words if len(w) > 2 and w not in STOPWORDS}
    # Pull in synonyms so "drink" also finds diary lines about "bottle".
    for group in SYNONYMS:
        if words & group:
            words |= group
    # Only what the camera observed counts as memory. Earlier answers stay in the
    # file, but are not fed back: a wrong answer would be repeated as if it were a fact.
    facts = [e for e in entries if e["kind"] != "question"]
    older = [e for e in facts[:-recent] if any(w in e["text"].lower() for w in words)][-matching:]
    return "\n".join(f"{e['time']} {e['text']}" for e in older + facts[-recent:]) or "(empty)"


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
    # Short object names must be found too, and old answers must not come back as facts.
    entries.insert(0, {"time": "2026-01-01 09:00:00", "kind": "event", "text": "cup #2 appeared at left"})
    remember("question", "user asked: where is the cup | answer: The cup is on the moon.")
    got = recall("Where is my cup?")
    assert "cup #2 appeared" in got and "moon" not in got, got
    # A half-written last line (power cut) must not stop the diary from loading.
    with open(LOG, "a", encoding="utf-8") as f:
        f.write('{"time": "2026-11-13 10:00:00", "kind": "ev')
    assert len(load(LOG)) == 22, len(load(LOG))
    # Synonyms: a word the camera never uses must still find the camera's word.
    got = recall("where is my drink?")
    assert "bottle #4 disappeared" in got, got
    # Prune: an old snapshot goes, a fresh one stays, and the log line is untouched.
    old = os.path.join(DIR, "20200101-000000-alert.jpg")
    cv2.imwrite(old, np.zeros((4, 4, 3), np.uint8))
    os.utime(old, (0, 0))  # mtime = 1970
    assert prune_snapshots() == 1 and not os.path.exists(old)
    assert os.path.exists(os.path.join(DIR, e["image"])), "pruned a fresh snapshot"
    assert len(load(LOG)) == 22, "prune touched the log"
    # The cap: once over MAX_ENTRIES, RAM and the log both hold exactly the newest MAX_ENTRIES.
    MAX_ENTRIES = 5
    entries.clear()
    for i in range(8):
        remember("scene", f"line {i}")
    assert len(entries) == 5 and entries[0]["text"] == "line 3", entries[0]
    assert len(load(LOG)) == 5, "log not trimmed to the cap"
    print("OK: memory works.")
