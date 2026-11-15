"""Browser front end: the same system as main.py, shown as a clean web page.

    python web.py              the page opens at http://localhost:8765
    python web.py qwen         start with Qwen3-VL 2B
    python web.py --no-browser do not open the browser automatically

The page (ui.html) shows the live video with boxes, what is in view, the scene
description, a chat with the assistant (typed or by microphone) and the
activity feed with alerts. Drag on the video to mark the restricted zone.

Only this computer can open the page: the server listens on 127.0.0.1 and
refuses requests that come from other websites. It is Python's built-in
http.server, so there is nothing extra to install.

How the page and Python talk:
  GET  /        the page
  GET  /video   the live video (a stream of JPEG pictures, 'MJPEG')
  GET  /state   everything else as JSON; the page asks for it a few times a second
  POST /ask     {"question": "..."}
  POST /do      {"action": "pause" | "model" | "voice" | "clear_zone" | "stop"}
  POST /zone    {"x1": .., "y1": .., "x2": .., "y2": ..}  in camera pixels
"""
import json
import os
import sys
import threading
import time
import webbrowser
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2

import main as app
from main import S, detect, faces, memory, track, voice

HOST, PORT = "127.0.0.1", 8765
HOSTS = {f"localhost:{PORT}", f"127.0.0.1:{PORT}"}
ORIGINS = {"http://" + h for h in HOSTS}
PAGE = os.path.join(os.path.dirname(__file__), "ui.html")
MAX_QUESTION = 300   # characters
fps = 0.0


def state():
    """Everything the page shows, as one JSON-ready dictionary."""
    frame, tracks, qa = S["frame"], S["tracks"], S["qa"]
    return {
        "live": frame is not None,
        "fps": round(fps),
        "size": [frame.shape[1], frame.shape[0]] if frame is not None else [640, 480],
        "model": S["model"],
        "yolo": detect.DEVICE,
        "voice": bool(voice),
        "diary": len(memory.entries),
        "paused": S["paused"],
        "listening": S["listening"],
        "stage": voice.stage if voice else "",
        "level": voice.level if voice else 0,
        "counts": sorted(Counter(t["name"] for t in tracks).items()),
        "tracks": [{"label": track.label(t), "where": t["where"]} for t in tracks],
        "faces_on": faces.ready,
        "unknown": faces.unknown_faces(),   # temp ids the user can name
        "faces": [label for label, _ in S["faces"]],
        "scene": S["text"],
        "scene_info": S["header"],
        "history": S["history"],
        "pending": qa[0] if qa and qa[1] == "..." else None,
        "events": [[time.strftime("%H:%M:%S", time.localtime(when)), text, alert]
                   for when, text, alert in S["events"]],
        "zone": S["zone"],
    }


def diary(path):
    """Newest 100 diary entries, optionally filtered by ?q=word. For the Memory panel."""
    q = parse_qs(urlparse(path).query).get("q", [""])[0].lower().strip()
    rows = [e for e in memory.entries if not q or q in e["text"].lower()]
    return [{"time": e["time"], "kind": e["kind"], "text": e["text"], "image": e.get("image")}
            for e in rows[-100:][::-1]]


def set_zone(data):
    """Zone from the page, in camera pixels. Too small to be a drag = clear it."""
    h, w = S["frame"].shape[:2]
    x1, x2 = sorted(max(0, min(w, int(data[k]))) for k in ("x1", "x2"))
    y1, y2 = sorted(max(0, min(h, int(data[k]))) for k in ("y1", "y2"))
    S["zone"] = (x1, y1, x2, y2) if x2 - x1 >= 10 and y2 - y1 >= 10 else None


def do(action):
    if action == "pause":
        S["paused"] = not S["paused"]
    elif action == "model":
        app.switch_model()
    elif action == "voice":
        app.start_voice()
    elif action == "clear_zone":
        S["zone"] = None
    elif action == "stop":
        S["running"] = False
    else:
        raise ValueError(action)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # keep the terminal for events and answers
        pass

    def allowed(self):
        """Only this computer's own page may use the server. The Host check stops a
        website from reaching it through a fake DNS name; the Origin check stops a
        website from sending commands from the user's browser."""
        origin = self.headers.get("Origin")
        return self.headers.get("Host") in HOSTS and (origin is None or origin in ORIGINS)

    def reply(self, code, body=b"", kind="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if not self.allowed():
            self.reply(403)
        elif self.path == "/":
            with open(PAGE, "rb") as f:  # read each time, so edits to ui.html show on refresh
                self.reply(200, f.read(), "text/html; charset=utf-8")
        elif self.path == "/state":
            self.reply(200, json.dumps(state()).encode())
        elif self.path.startswith("/memory"):
            self.reply(200, json.dumps(diary(self.path)).encode())
        elif self.path.startswith("/snap/"):
            self.snapshot(self.path[len("/snap/"):])
        elif self.path.startswith("/video"):
            self.video()
        else:
            self.reply(404)

    def do_POST(self):
        if not self.allowed() or self.headers.get("Content-Type") != "application/json":
            return self.reply(403)
        try:
            length = int(self.headers.get("Content-Length", 0))
            if length > 2000:
                raise ValueError("too long")
            data = json.loads(self.rfile.read(length))
            if self.path == "/ask":
                question = str(data["question"]).strip()[:MAX_QUESTION]
                if question:
                    app.ask(question)
            elif self.path == "/do":
                do(data["action"])
            elif self.path == "/zone":
                set_zone(data)
            elif self.path == "/name":
                faces.name_face(str(data["id"]), str(data["name"]).strip()[:40])
            else:
                return self.reply(404)
        except (ValueError, KeyError, TypeError, AttributeError):  # bad or incomplete request
            return self.reply(400)
        self.reply(200, b"{}")

    def snapshot(self, name):
        """Serve a diary snapshot by filename. Only plain .jpg names from the memory
        folder: no path separators, so the browser cannot reach other files."""
        if "/" in name or "\\" in name or not name.endswith(".jpg"):
            return self.reply(403)
        path = os.path.join(memory.DIR, name)
        if not os.path.isfile(path):
            return self.reply(404)
        with open(path, "rb") as f:
            self.reply(200, f.read(), "image/jpeg")

    def video(self):
        """Live video with boxes: one JPEG after another on the same connection."""
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        last = None
        try:
            while S["running"]:
                frame = S["frame"]
                if frame is None or frame is last:
                    time.sleep(0.01)
                    continue
                last = frame
                img = frame.copy()  # draw on a copy so neither model sees the boxes
                app.draw_boxes(img)
                jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpg)
                                 + jpg + b"\r\n")
        except OSError:  # the browser tab was closed
            pass


def main():
    global fps
    try:
        server = ThreadingHTTPServer((HOST, PORT), Handler)
    except OSError:
        sys.exit(f"Port {PORT} is already in use. Is web.py already running? Close it, or change PORT in web.py.")
    server.daemon_threads = True  # an open video stream must not keep the program alive
    threading.Thread(target=server.serve_forever, daemon=True).start()
    app.start_workers()
    url = f"http://localhost:{PORT}"
    print(f"Page: {url}   (Ctrl+C here, or Stop on the page, to quit)")
    if "--no-browser" not in sys.argv:
        webbrowser.open(url)

    count, since = 0, time.time()
    try:
        for _ in app.camera_frames():
            count += 1
            if time.time() - since >= 1:
                fps, count, since = count / (time.time() - since), 0, time.time()
    except KeyboardInterrupt:  # Ctrl+C in the terminal is a normal way to quit
        pass
    finally:
        S["running"] = False
        server.shutdown()


if __name__ == "__main__":
    main()
