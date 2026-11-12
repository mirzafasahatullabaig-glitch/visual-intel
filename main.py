"""Real-time webcam visual intelligence.

The video stays live at full speed. Two background threads work on the
newest frame:
  - YOLO (detect.py) draws green boxes around objects, ~30 times a second.
  - The VLM describes the scene; it needs ~3 s per answer on an 8 GB GPU,
    so the text updates every few seconds while video and boxes stay live.

Video window keys (click the window first):
  SPACE  pause / resume analysis
  m      switch model (Gemma <-> Qwen)
  ESC    quit
Terminal: type a question + Enter and it is asked about every new frame
          (e.g. "Is there a person?"). Empty Enter = back to describing.

Run:  python main.py            (uses vlm.DEFAULT_MODEL)
      python main.py qwen       (start with Qwen3-VL 2B)
"""
import sys
import textwrap
import threading
import time

import cv2

import detect
import vlm

DESCRIBE = (
    "Describe what you see in one sentence of at most 30 words: any person, "
    "the main objects and where they are. No preamble."
)
MAX_TOKENS = 60  # shorter answers = faster updates

# Shared between the camera loop, the model thread and the question thread.
# Each value is replaced whole (never edited in place), which is safe across threads in Python.
S = {
    "frame": None,
    "prompt": DESCRIBE,
    "model": vlm.QWEN if "qwen" in sys.argv[1:] else vlm.DEFAULT_MODEL,
    "header": "Analysing...",
    "text": "",
    "detections": [],
    "paused": False,
    "running": True,
}


def to_jpeg(frame):
    h, w = frame.shape[:2]
    if w > 640:
        frame = cv2.resize(frame, (640, int(h * 640 / w)))
    return cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()


def analyse_loop():
    """Background thread: as soon as one answer is done, ask about the newest frame."""
    while S["running"]:
        frame = S["frame"]
        if S["paused"] or frame is None:
            time.sleep(0.1)
            continue
        prompt, model = S["prompt"], S["model"]
        t = time.time()
        try:
            answer = vlm.ask(to_jpeg(frame), prompt, model=model, max_tokens=MAX_TOKENS)
        except Exception as e:  # Ollama down, model missing, timeout...
            S["header"], S["text"] = "ERROR (is Ollama running?)", str(e)
            time.sleep(2)  # don't hammer a dead server
            continue
        secs = time.time() - t
        if answer.rfind(".") > 0:  # MAX_TOKENS can cut mid-sentence; keep only full sentences
            answer = answer[: answer.rfind(".") + 1]
        S["header"] = f"{model} | {secs:.1f}s" + ("" if prompt == DESCRIBE else f" | Q: {prompt}")
        S["text"] = answer
        print(f"[{model} {secs:.1f}s] {answer}", flush=True)


def detect_loop():
    """Background thread: YOLO boxes for each new frame (~34 ms each on CPU)."""
    last = None
    while S["running"]:
        frame = S["frame"]
        if frame is None or frame is last:  # don't re-detect the same frame
            time.sleep(0.005)
            continue
        last = frame
        S["detections"] = detect.detect(frame)


def read_questions():
    """Background thread: terminal input changes what the model is asked."""
    while S["running"]:
        try:
            question = input().strip()
        except EOFError:  # no terminal attached
            return
        S["prompt"] = question or DESCRIBE
        print(f"Now asking: {S['prompt']}", flush=True)


def draw_answer(img):
    """Dark box at the bottom of the video with the latest answer."""
    # putText can't draw unicode or markdown; strip them.
    clean = S["text"].replace("*", "").encode("ascii", "ignore").decode()
    status = "  [PAUSED]" if S["paused"] else ""
    lines = (textwrap.wrap(S["header"] + status, 64) + textwrap.wrap(clean, 64))[:8]
    h, w = img.shape[:2]
    top = h - 22 * len(lines) - 10
    box = img.copy()
    cv2.rectangle(box, (0, top), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(box, 0.6, img, 0.4, 0, img)
    for i, line in enumerate(lines):
        color = (0, 255, 255) if i == 0 else (255, 255, 255)  # yellow header
        cv2.putText(img, line, (8, top + 20 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def main():
    # CAP_DSHOW opens the Windows webcam much faster than the default backend.
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cam.isOpened():
        sys.exit("Could not open webcam 0. Is another app (Teams, Camera) using it?")

    print(__doc__)
    print(f"Model: {S['model']}")
    threading.Thread(target=analyse_loop, daemon=True).start()
    threading.Thread(target=detect_loop, daemon=True).start()
    threading.Thread(target=read_questions, daemon=True).start()

    while S["running"]:
        ok, frame = cam.read()
        if not ok:
            print("Lost camera frame; stopping.")
            break
        S["frame"] = frame
        shown = frame.copy()  # draw on a copy so neither model sees the overlay
        detect.draw(shown, S["detections"])
        draw_answer(shown)
        cv2.imshow("Visual Intelligence (SPACE pause, m model, ESC quit)", shown)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord(" "):
            S["paused"] = not S["paused"]
        elif key == ord("m"):
            S["model"] = vlm.QWEN if S["model"] == vlm.GEMMA else vlm.GEMMA
            print(f"Switched to {S['model']}", flush=True)

    S["running"] = False
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
