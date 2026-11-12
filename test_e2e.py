"""End-to-end test: runs the real app (~90 s) and behaves like a careless human.

Uses the real webcam, microphone, YOLO, Ollama and speakers. Only the hands
are simulated: scripted key presses, typed lines and mouse drags go through
the same handlers a person would trigger. The diary is redirected to a
throwaway folder. Sit in front of the camera (the zone alert needs a person)
and stay quiet around 14-20 s (the microphone is really recording then).

    python test_e2e.py

What the 'human' does:
   4 s  drags a restricted zone, then clears it with Z while Caps Lock is on
   6 s  drags the zone again (a person in view must raise an alert)
   8 s  types a question
  14 s  HOLDS the v key for 2 s (60 repeated key events; this used to crash)
  31 s  asks a question by voice (the microphone is faked here, the rest is real)
  41 s  Ollama 'goes down' for 8 s while a question is asked
  51 s  the detector thread crashes once; a message with an emoji is printed
  53 s  clicks the video without dragging (must clear the zone)
  55 s  asks a memory question
  68 s  closes the window with the X button
Then a second short run is ended with Ctrl+C.
"""
import _thread
import builtins
import os
import sys
import tempfile
import threading
import time

import cv2

import main
from main import S, detect, memory, vlm, voice

memory.DIR = tempfile.mkdtemp()
memory.LOG = os.path.join(memory.DIR, "log.jsonl")
memory.entries.clear()

t0 = time.time()
now = lambda: time.time() - t0
seen = {"frames": 0, "answers": [], "zone_after_Z": "?", "zone_after_click": "?", "unicode_ok": False,
        "listen_calls": 0, "listen_overlap": 0, "detect_ok_after_crash": 0, "last_description": 0, "errors_shown": []}
keys, typed, actions = [], [], []


def press(at, chars, gap=0.033):
    keys.extend((at + i * gap, c) for i, c in enumerate(chars))


def drag(x0, y0, x1, y1):
    main.on_mouse(cv2.EVENT_LBUTTONDOWN, x0, y0, 0, None)
    main.on_mouse(cv2.EVENT_MOUSEMOVE, x1, y1, cv2.EVENT_FLAG_LBUTTON, None)
    main.on_mouse(cv2.EVENT_LBUTTONUP, x1, y1, 0, None)


# ---- the scripted human -------------------------------------------------
actions += [(4.0, lambda: drag(120, 60, 520, 440))]
press(5.0, "Z")                                            # Caps Lock on
actions += [(5.6, lambda: seen.update(zone_after_Z=S["zone"]))]
actions += [(6.0, lambda: drag(120, 60, 520, 440))]
typed += [(8.0, "Is there a person in the image, and where?")]
press(14.0, "v" * 60)                                      # key held down for 2 s
real_listen = voice.listen
busy = threading.Lock()


def counted_listen():
    seen["listen_calls"] += 1
    if not busy.acquire(blocking=False):
        seen["listen_overlap"] += 1
        return ""
    try:
        return real_listen()
    finally:
        busy.release()


voice.listen = counted_listen
actions += [(30.5, lambda: setattr(voice, "listen", lambda: "What objects can you see?"))]
press(31.0, "V")
real_url = vlm.OLLAMA_URL
actions += [(41.0, lambda: setattr(vlm, "OLLAMA_URL", "http://localhost:9/api/generate"))]
typed += [(42.0, "What colour is my shirt?")]
actions += [(49.0, lambda: setattr(vlm, "OLLAMA_URL", real_url))]
real_detect = detect.detect
crash = {"armed": False}


def flaky_detect(frame):
    if crash["armed"]:
        crash["armed"] = False
        raise RuntimeError("simulated detector crash")
    out = real_detect(frame)
    if now() > 52:
        seen["detect_ok_after_crash"] += 1
    return out


detect.detect = flaky_detect
actions += [(51.0, lambda: crash.update(armed=True))]
actions += [(51.5, lambda: (main.show("unicode check ✓ \U0001F600"), seen.update(unicode_ok=True)))]
actions += [(53.0, lambda: drag(300, 200, 300, 200))]     # click, no drag
actions += [(53.5, lambda: seen.update(zone_after_click=S["zone"]))]
typed += [(55.0, "Did anyone enter the restricted zone? When?")]
CLOSE_AT, WATCHDOG = 68.0, 100.0

# ---- hooks into the real app --------------------------------------------
real_imshow, real_waitkey = cv2.imshow, cv2.waitKey


def imshow(name, img):
    seen["frames"] += 1
    real_imshow(name, img)


def waitkey(ms):
    real_waitkey(ms)
    while actions and actions[0][0] <= now():
        actions.pop(0)[1]()
    if phase == 1 and now() > CLOSE_AT:
        cv2.destroyWindow(main.WIN)                       # the X button
    if keys and keys[0][0] <= now():
        return ord(keys.pop(0)[1])
    return -1


def fake_input():
    if not typed:
        time.sleep(3600)
    at, text = typed.pop(0)
    while now() < at:
        time.sleep(0.05)
    return text


def observer():
    last_text = None
    while True:
        qa = S["qa"]
        if qa and qa[1] != "..." and qa not in seen["answers"]:
            seen["answers"].append(qa)
        if S["text"] != last_text and not S["header"].startswith("ERROR"):
            last_text, seen["last_description"] = S["text"], now()
        for _, text, alert in S["events"]:
            if alert and text not in seen["errors_shown"]:
                seen["errors_shown"].append(text)
        if now() > WATCHDOG:
            S["running"] = False
        time.sleep(0.1)


main.cv2.imshow, main.cv2.waitKey, builtins.input = imshow, waitkey, fake_input
actions.sort(key=lambda a: a[0])
keys.sort(key=lambda k: k[0])
threading.Thread(target=observer, daemon=True).start()

phase = 1
main.main()
run1 = now()
fps = seen["frames"] / run1

# ---- second run: quit with Ctrl+C ---------------------------------------
phase = 2
time.sleep(6)  # let the first run's background threads finish their last model call
S["running"] = True
threading.Timer(6, _thread.interrupt_main).start()
t_ctrlc = time.time()
main.main()                                                # must return, not raise KeyboardInterrupt
ctrlc_secs = time.time() - t_ctrlc

# ---- verdict ------------------------------------------------------------
diary = memory.entries
answers = dict(seen["answers"])
checks = {
    "video stayed above 20 fps": fps > 20,
    "window X button quits the app": CLOSE_AT - 1 < run1 < CLOSE_AT + 8,
    "Ctrl+C quits cleanly": 4 < ctrlc_secs < 20,
    "Caps Lock: Z cleared the zone": seen["zone_after_Z"] is None,
    "click without drag cleared the zone": seen["zone_after_click"] is None,
    "restricted-zone alert was raised": any(e["kind"] == "alert" and "restricted zone" in e["text"] for e in diary),
    "typed question answered": "Is there a person in the image, and where?" in answers,
    "holding v: one recording, no overlap": seen["listen_calls"] == 1 and seen["listen_overlap"] == 0,
    "voice question answered": "What objects can you see?" in answers,
    "Ollama down: question got a clear reply": answers.get("What colour is my shirt?", "").startswith("No answer"),
    "Ollama back: descriptions resumed": seen["last_description"] > 51,
    "detector crash was reported": any("detect_loop error" in t for t in seen["errors_shown"]),
    "detector kept working after its crash": seen["detect_ok_after_crash"] > 100,
    "emoji in a message did not crash": seen["unicode_ok"],
    "memory question answered": "Did anyone enter the restricted zone? When?" in answers,
    "scene notes are text-only, events have snapshots":
        all("image" not in e for e in diary if e["kind"] == "scene")
        and all("image" in e for e in diary if e["kind"] in ("event", "alert")),
}
print(f"\n===== END-TO-END RESULT  ({run1:.0f} s run, {fps:.1f} fps) =====")
for question, answer in seen["answers"]:
    print(f"  Q: {question}\n  A: {answer}")
print()
for name, ok in checks.items():
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
print(f"\n{sum(checks.values())}/{len(checks)} checks passed")
sys.exit(0 if all(checks.values()) else 1)
