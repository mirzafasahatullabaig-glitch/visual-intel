"""Webcam visual intelligence, terminal version.

Live camera window. Keys (click the video window first so it gets the keys):
  SPACE  capture a frame and describe it
  q      capture a frame and type your own question in the terminal
  m      switch model (Gemma <-> Qwen)
  ESC    quit

Run:  python main.py            (uses vlm.DEFAULT_MODEL)
      python main.py qwen       (start with Qwen3-VL 2B)
"""
import sys
import time

import cv2

import vlm

DESCRIBE = (
    "Describe this scene. List the objects you can see, where they are "
    "(left/right/centre, front/back), and any notable colours or details. "
    "Say whether a person is present. Be concise."
)


def capture_jpeg(frame):
    # Smaller image = fewer tokens = faster answer; 640px wide is enough for a desk scene.
    h, w = frame.shape[:2]
    if w > 640:
        frame = cv2.resize(frame, (640, int(h * 640 / w)))
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes()


def run_query(frame, prompt, model):
    print(f"\n[{model}] {prompt}")
    t = time.time()
    try:
        answer = vlm.ask(capture_jpeg(frame), prompt, model=model)
    except Exception as e:  # Ollama down, model missing, timeout...
        print(f"  ERROR: {e}  (is Ollama running? try: ollama ps)")
        return
    print(f"  ({time.time() - t:.1f}s)\n{answer}\n")


def main():
    model = vlm.QWEN if "qwen" in sys.argv[1:] else vlm.DEFAULT_MODEL
    # CAP_DSHOW opens the Windows webcam much faster than the default backend.
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cam.isOpened():
        sys.exit("Could not open webcam 0. Is another app (Teams, Camera) using it?")

    print(__doc__)
    print(f"Model: {model}")
    while True:
        ok, frame = cam.read()
        if not ok:
            print("Lost camera frame; stopping.")
            break
        cv2.imshow("Visual Intelligence (SPACE describe, q ask, m model, ESC quit)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord(" "):
            run_query(frame, DESCRIBE, model)
        elif key == ord("q"):
            # The video window freezes while you type; that's expected for a terminal prototype.
            question = input("Your question: ").strip()
            if question:
                run_query(frame, question, model)
        elif key == ord("m"):
            model = vlm.QWEN if model == vlm.GEMMA else vlm.GEMMA
            print(f"Switched to {model}")

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
