"""Real-time webcam visual intelligence.

The video stays live at full speed. Background threads work on the newest frame:
  - YOLO + tracker (detect.py, track.py): boxes with IDs, counts, positions, ~30x a second
  - watch.py: events ('person #1 appeared') and alerts (object disappeared,
    restricted zone entered, object left unattended)
  - VLM (vlm.py): describes the scene every few seconds and answers your questions,
    using the image + the detector's object list + the memory diary (memory.py)

Video window (click it first):
  SPACE       pause / resume scene descriptions
  m           switch model (Gemma <-> Qwen)
  v           voice question: press, speak, press again; the answer is spoken
  drag mouse  draw the restricted zone       z  clear it
  ESC         quit
Terminal: type a question + Enter, e.g.  Is my phone on the desk?
          Where did I leave my water bottle?     (empty Enter clears the answer)

Run:  python main.py            (uses vlm.DEFAULT_MODEL)
      python main.py qwen       (start with Qwen3-VL 2B)
"""
import sys
import textwrap
import threading
import time

import cv2

import detect
import memory
import track
import vlm
import watch

try:
    import voice
except ImportError:  # faster-whisper / sounddevice not installed: everything else still works
    voice = None

DESCRIBE = (
    "Describe what you see in one sentence of at most 30 words: any person, "
    "the main objects and where they are. No preamble."
)
QUESTION = (
    "You are a visual assistant looking through a camera. Answer the question in one or two "
    "short sentences, no preamble. Use the image first; use the notes below for exact positions "
    "and for things that happened earlier. If you cannot tell, say so.\n"
    "Object detector, in view right now: {objects}\n"
    "Memory diary of earlier observations:\n{memory}\n"
    "Time now: {now}\n"
    "Question: {question}"
)
MAX_TOKENS = 60     # shorter answers = faster updates
MEMORY_EVERY = 30   # seconds between scene descriptions saved to the diary
EVENT_SECONDS = 6   # how long an event stays on screen
WIN = "Visual Intelligence"
FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE, YELLOW, RED, CYAN = (255, 255, 255), (0, 255, 255), (60, 60, 255), (255, 255, 0)

# Shared between the camera loop and the background threads.
# Each value is replaced whole (never edited in place), which is safe across threads in Python.
S = {
    "frame": None,
    "model": vlm.QWEN if "qwen" in sys.argv[1:] else vlm.DEFAULT_MODEL,
    "header": "Analysing...",   # latest scene description
    "text": "",
    "question": None,           # (text, speak_answer) waiting for the VLM
    "qa": None,                 # (question, answer) pinned on screen
    "tracks": [],               # objects in view right now
    "events": [],               # recent (time, text, is_alert)
    "zone": None,               # restricted area (x1, y1, x2, y2)
    "drag": None,
    "recording": False,
    "paused": False,
    "running": True,
}


def to_jpeg(frame):
    h, w = frame.shape[:2]
    if w > 640:
        frame = cv2.resize(frame, (640, int(h * 640 / w)))
    return cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()


def note(text, alert=False, frame=None):
    """Show an event on screen, print it, and save it (with a snapshot) to the diary."""
    S["events"] = S["events"][-3:] + [(time.time(), text, alert)]
    print(("ALERT " if alert else "      ") + time.strftime("%H:%M:%S ") + text, flush=True)
    memory.remember("alert" if alert else "event", text, frame)


def detect_loop():
    """Background thread: boxes, tracking, events and alerts for each new frame."""
    tracker, watcher = track.Tracker(), watch.Watcher()
    last = None
    while S["running"]:
        frame = S["frame"]
        if frame is None or frame is last:  # don't re-detect the same frame
            time.sleep(0.005)
            continue
        last = frame
        h, w = frame.shape[:2]
        events = tracker.update(detect.detect(frame), w, h)
        S["tracks"] = tracker.visible()
        for text, alert in watcher.check(S["tracks"], events, S["zone"]):
            note(text, alert, frame)


def analyse_loop():
    """Background thread: answer a waiting question, otherwise describe the newest frame."""
    last_saved = 0
    while S["running"]:
        frame, asked = S["frame"], S["question"]
        if frame is None or (S["paused"] and not asked):
            time.sleep(0.1)
            continue
        model = S["model"]
        if asked:
            S["question"] = None
            question, speak = asked
            prompt = QUESTION.format(objects=track.summary(S["tracks"]), memory=memory.recall(question),
                                     now=time.strftime("%Y-%m-%d %H:%M:%S"), question=question)
        else:
            prompt = DESCRIBE
        t = time.time()
        try:
            answer = vlm.ask(to_jpeg(frame), prompt, model=model, max_tokens=MAX_TOKENS + 30 * bool(asked))
        except Exception as e:  # Ollama down, model missing, timeout...
            S["header"], S["text"] = "ERROR (is Ollama running?)", str(e)
            time.sleep(2)  # don't hammer a dead server
            continue
        secs = time.time() - t
        if answer.rfind(".") > 0:  # the token cap can cut mid-sentence; keep only full sentences
            answer = answer[: answer.rfind(".") + 1]
        if asked:
            S["qa"] = (question, answer)
            print(f"\nQ: {question}\nA: {answer}   [{model} {secs:.1f}s]\n", flush=True)
            memory.remember("question", f"user asked: {question} | answer: {answer}")
            if speak:
                voice.say(answer)
        else:
            S["header"], S["text"] = f"{model} | {secs:.1f}s", answer
            if time.time() - last_saved > MEMORY_EVERY:
                last_saved = time.time()
                memory.remember("scene", answer, frame)


def read_questions():
    """Background thread: a typed line becomes a question for the VLM."""
    while S["running"]:
        try:
            question = input().strip()
        except EOFError:  # no terminal attached
            return
        S["qa"] = (question, "...") if question else None
        if question:
            S["question"] = (question, False)


def ask_by_voice():
    """After the second 'v': turn the recording into a question (about 1 s, so not in the camera loop)."""
    question = voice.stop()
    print(f"Heard: {question!r}", flush=True)
    if question:
        S["qa"] = (question, "...")
        S["question"] = (question, True)


def on_mouse(event, x, y, flags, _):
    """Drag with the left button to draw the restricted zone."""
    if event == cv2.EVENT_LBUTTONDOWN:
        S["drag"] = (x, y)
    elif S["drag"] and (event == cv2.EVENT_LBUTTONUP
                        or (event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON)):
        x0, y0 = S["drag"]
        S["zone"] = (min(x0, x), min(y0, y), max(x0, x), max(y0, y))
        if event == cv2.EVENT_LBUTTONUP:
            S["drag"] = None


def lines(text, colour):
    # putText can't draw unicode or markdown; strip them, then wrap to the frame width.
    clean = text.replace("*", "").encode("ascii", "ignore").decode()
    return [(line, colour) for line in textwrap.wrap(clean, 64)]


def text_block(img, rows, top):
    """Draw (text, colour) rows on a dark see-through strip starting at y=top."""
    box = img.copy()
    cv2.rectangle(box, (0, top), (img.shape[1], top + 22 * len(rows) + 10), (0, 0, 0), -1)
    cv2.addWeighted(box, 0.6, img, 0.4, 0, img)
    for i, (text, colour) in enumerate(rows):
        cv2.putText(img, text, (8, top + 20 + 22 * i), FONT, 0.5, colour, 1, cv2.LINE_AA)


def draw_overlay(img):
    tracks = S["tracks"]
    detect.draw(img, [(track.label(t), t["conf"], t["box"]) for t in tracks])
    if S["zone"]:
        x1, y1, x2, y2 = S["zone"]
        cv2.rectangle(img, (x1, y1), (x2, y2), RED, 2)
        cv2.putText(img, "RESTRICTED", (x1 + 4, y1 + 16), FONT, 0.5, RED, 1, cv2.LINE_AA)

    # Top: what is in view, then recent events (alerts in red).
    top = lines("In view: " + (track.counts(tracks) or "nothing"), WHITE)
    for when, text, alert in S["events"]:
        if time.time() - when < EVENT_SECONDS:
            top += lines(("ALERT: " if alert else "") + text, RED if alert else CYAN)
    if S["recording"]:
        top += lines("REC - speak, then press v again", RED)
    text_block(img, top, 0)

    # Bottom: live scene description, then the pinned question and answer.
    bottom = lines(S["header"] + ("  [PAUSED]" if S["paused"] else ""), YELLOW) + lines(S["text"], WHITE)
    if S["qa"]:
        bottom += lines("Q: " + S["qa"][0], YELLOW) + lines("A: " + S["qa"][1], WHITE)
    bottom = bottom[-9:]
    text_block(img, bottom, img.shape[0] - 22 * len(bottom) - 10)


def main():
    # CAP_DSHOW opens the Windows webcam much faster than the default backend.
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cam.isOpened():
        sys.exit("Could not open webcam 0. Is another app (Teams, Camera) using it?")

    print(__doc__)
    print(f"Model: {S['model']} | YOLO on {detect.DEVICE} | voice: {'on' if voice else 'off (see voice.py)'} "
          f"| diary: {len(memory.entries)} entries")
    cv2.namedWindow(WIN)
    cv2.setMouseCallback(WIN, on_mouse)
    for loop in (analyse_loop, detect_loop, read_questions):
        threading.Thread(target=loop, daemon=True).start()

    while S["running"]:
        ok, frame = cam.read()
        if not ok:
            print("Lost camera frame; stopping.")
            break
        S["frame"] = frame
        shown = frame.copy()  # draw on a copy so neither model sees the overlay
        draw_overlay(shown)
        cv2.imshow(WIN, shown)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord(" "):
            S["paused"] = not S["paused"]
        elif key == ord("m"):
            S["model"] = vlm.QWEN if S["model"] == vlm.GEMMA else vlm.GEMMA
            print(f"Switched to {S['model']}", flush=True)
        elif key == ord("z"):
            S["zone"] = None
        elif key == ord("v") and voice:
            S["recording"] = not S["recording"]
            if S["recording"]:
                voice.start()
            else:
                threading.Thread(target=ask_by_voice, daemon=True).start()

    S["running"] = False
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
