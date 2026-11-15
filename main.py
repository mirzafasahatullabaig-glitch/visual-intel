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
  v           voice question: press once and speak; it stops when you stop
              talking, and the answer is spoken
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
import faces
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
    "and for things that happened earlier. Left and right mean the left and right of the image. "
    "If you cannot tell, say so.\n"
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
    "history": [],              # the last 20 (question, answer) pairs, for the web page
    "tracks": [],               # objects in view right now
    "faces": [],                # (label, box) for faces in view: a name or 'face #N'
    "events": [],               # the last 50 (time, text, is_alert)
    "zone": None,               # restricted area (x1, y1, x2, y2)
    "drag": None,
    "listening": False,         # microphone is recording a voice question
    "paused": False,
    "running": True,
}


def to_jpeg(frame):
    h, w = frame.shape[:2]
    if w > 640:
        frame = cv2.resize(frame, (640, int(h * 640 / w)))
    return cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()


def show(text, alert=False):
    """Show a message on screen for a few seconds and print it."""
    S["events"] = S["events"][-49:] + [(time.time(), text, alert)]
    print(("ALERT " if alert else "      ") + time.strftime("%H:%M:%S ") + text, flush=True)


def note(text, alert=False, frame=None):
    """An observation about the scene: show it and save it (with a snapshot) to the diary."""
    show(text, alert)
    memory.remember("alert" if alert else "event", text, frame)


def guarded(loop):
    """Wrap a background loop so a crash is reported and the loop restarts.
    (A thread that dies silently would freeze its feature for the rest of the session.)"""
    def run():
        while S["running"]:
            try:
                loop()
            except Exception as e:
                show(f"{loop.__name__} error, restarting: {e}", alert=True)
                time.sleep(1)
    return run


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


def faces_loop():
    """Background thread: face recognition. Its own thread because it runs on the CPU
    and is far slower than YOLO; sharing detect_loop would stall tracking and alerts."""
    if not faces.ready:
        return
    last, seen = None, set()
    while S["running"]:
        frame = S["frame"]
        if frame is None or frame is last:
            time.sleep(0.05)
            continue
        last = frame
        S["faces"] = faces.recognise(frame)
        for label, _ in S["faces"]:  # a known person appearing is a diary event
            if not label.startswith("face #") and label not in seen:
                seen.add(label)
                note(f"{label} appeared", frame=frame)


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
            if asked:  # don't leave the question hanging on '...'
                S["qa"] = (question, "No answer: the model is not reachable. Is Ollama running?")
                S["history"] = S["history"][-19:] + [S["qa"]]
            time.sleep(2)  # don't hammer a dead server
            continue
        secs = time.time() - t
        # The token cap can cut mid-sentence: drop an unfinished last sentence.
        # ('. ' with a space, so a decimal such as 0.5 is never mistaken for a sentence end.)
        if not answer.endswith((".", "!", "?")) and ". " in answer:
            answer = answer[: answer.rfind(". ") + 1]
        if asked:
            S["qa"] = (question, answer)
            S["history"] = S["history"][-19:] + [S["qa"]]
            print(f"\nQ: {question}\nA: {answer}   [{model} {secs:.1f}s]\n", flush=True)
            memory.remember("question", f"user asked: {question} | answer: {answer}")
            if speak:
                voice.say(answer)
        else:
            S["header"], S["text"] = f"{model} | {secs:.1f}s", answer
            if time.time() - last_saved > MEMORY_EVERY:
                last_saved = time.time()
                memory.remember("scene", answer)  # text only: a snapshot every 30 s would be ~100 MB a day


def ask(question, speak=False):
    """Hand a question to the VLM thread; it shows as '...' until the answer arrives."""
    S["qa"] = (question, "...")
    S["question"] = (question, speak)


def switch_model():
    S["model"] = vlm.QWEN if S["model"] == vlm.GEMMA else vlm.GEMMA
    show(f"Switched to {S['model']} (first answer takes ~10 s)")


def read_questions():
    """Background thread: a typed line becomes a question for the VLM."""
    while S["running"]:
        try:
            question = input().strip()
        except EOFError:  # no terminal attached
            return
        if question:
            ask(question)
        else:
            S["qa"] = None


def ask_by_voice():
    """Own thread: record one spoken question and hand it to the VLM."""
    try:
        question = voice.listen()
    except Exception as e:  # microphone busy or unplugged: say so, never crash the app
        show(f"Microphone error: {e}", alert=True)
        return
    finally:
        S["listening"] = False
    if question:
        show(f"Heard: {question}")
        ask(question, speak=True)
    else:
        show("Heard nothing. Is the microphone muted?", alert=True)


_voice_lock = threading.Lock()


def start_voice():
    """Start recording a voice question. Does nothing while one is already being recorded,
    so a held key or a double click cannot open the microphone twice."""
    if not voice:
        show("Voice is not installed; see the note at the top of voice.py", alert=True)
        return
    with _voice_lock:
        if S["listening"]:
            return
        S["listening"] = True
    threading.Thread(target=ask_by_voice, daemon=True).start()


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
            if abs(x - x0) < 10 or abs(y - y0) < 10:  # a plain click, not a drag: clear the zone
                S["zone"] = None


def window_closed():
    """True once the user has closed the video window with the X button."""
    try:
        return cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:  # some OpenCV builds raise once the window is gone
        return True


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


def draw_boxes(img):
    """Object boxes with IDs, and the restricted zone."""
    detect.draw(img, [(track.label(t), t["conf"], t["box"]) for t in S["tracks"]])
    faces.draw(img, S["faces"])
    if S["zone"]:
        x1, y1, x2, y2 = S["zone"]
        cv2.rectangle(img, (x1, y1), (x2, y2), RED, 2)
        cv2.putText(img, "RESTRICTED", (x1 + 4, y1 + 16), FONT, 0.5, RED, 1, cv2.LINE_AA)


def draw_overlay(img):
    """Everything the OpenCV window shows: boxes plus the text strips."""
    tracks = S["tracks"]
    draw_boxes(img)

    # Top: what is in view, then recent events (alerts in red).
    top = lines("In view: " + (track.counts(tracks) or "nothing"), WHITE)
    for when, text, alert in S["events"][-4:]:
        if time.time() - when < EVENT_SECONDS:
            top += lines(("ALERT: " if alert else "") + text, RED if alert else CYAN)
    if S["listening"]:  # the bar moves when the microphone hears you
        bar = "|" * min(30, int(voice.level * 300))
        top += lines("LISTENING - speak now  " + bar if voice.stage == "listening" else "Working out what you said...", RED)
    text_block(img, top, 0)

    # Bottom: live scene description, then the pinned question and answer.
    bottom = lines(S["header"] + ("  [PAUSED]" if S["paused"] else ""), YELLOW) + lines(S["text"], WHITE)
    if S["qa"]:
        bottom += lines("Q: " + S["qa"][0], YELLOW) + lines("A: " + S["qa"][1], WHITE)
    bottom = bottom[-9:]
    text_block(img, bottom, img.shape[0] - 22 * len(bottom) - 10)


def camera_frames():
    """Yield webcam frames (also stored in S['frame']) until S['running'] goes False
    or the camera stops. Both front ends, the OpenCV window and web.py, run on this."""
    # CAP_DSHOW opens the Windows webcam much faster than the default backend.
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cam.isOpened():
        sys.exit("Could not open webcam 0. Is another app (Teams, Camera) using it?")
    failed = 0
    try:
        while S["running"]:
            ok, frame = cam.read()
            if not ok:
                failed += 1
                if failed > 30:  # one dropped frame is normal; a full second of them is not
                    print("Camera stopped giving frames; stopping.")
                    return
                continue
            failed = 0
            S["frame"] = frame
            yield frame
    finally:
        cam.release()


def start_workers():
    """Start the background threads and print the status line."""
    sys.stdout.reconfigure(errors="replace")  # an emoji in a model answer must not crash a print
    print(f"Model: {S['model']} | YOLO on {detect.DEVICE} | faces: {faces._device if faces.ready else 'off'} "
          f"| voice: {'on' if voice else 'off (see voice.py)'} | diary: {len(memory.entries)} entries")
    for loop in (guarded(analyse_loop), guarded(detect_loop), guarded(faces_loop), read_questions):
        threading.Thread(target=loop, daemon=True).start()


def main():
    print(__doc__)
    cv2.namedWindow(WIN)
    cv2.setMouseCallback(WIN, on_mouse)
    start_workers()
    try:
        for frame in camera_frames():
            shown = frame.copy()  # draw on a copy so neither model sees the overlay
            draw_overlay(shown)
            cv2.imshow(WIN, shown)
            key = chr(cv2.waitKey(1) & 0xFF).lower()  # lower(): keys still work with Caps Lock on
            if key == "\x1b" or window_closed():  # ESC, or the X button
                break
            elif key == " ":
                S["paused"] = not S["paused"]
            elif key == "m":
                switch_model()
            elif key == "z":
                S["zone"] = None
            elif key == "v":
                start_voice()
    except KeyboardInterrupt:  # Ctrl+C in the terminal is a normal way to quit
        pass
    finally:
        S["running"] = False
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
