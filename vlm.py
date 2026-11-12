"""Talk to a local vision-language model through Ollama.

This is the ONLY file that knows about Ollama. To switch models, change
DEFAULT_MODEL below (or pass model=... to ask). Everything else stays the same.

Ollama's /api/generate accepts images as base64 strings, so we only need the
standard library here -- no extra 'ollama' or 'requests' package to install.
"""
import base64
import json
import urllib.request

OLLAMA_URL = "http://localhost:11434/api/generate"

# The two vision models you have installed. Pick one as default; switch freely.
GEMMA = "gemma3:4b"
QWEN = "qwen3-vl:2b-instruct"
DEFAULT_MODEL = GEMMA


def ask(image_bytes, prompt, model=DEFAULT_MODEL, timeout=120, max_tokens=None):
    """Send JPEG/PNG bytes + a question to the model, return its text answer.

    image_bytes: raw encoded image (e.g. from cv2.imencode or a file read)
    prompt:      what to ask about the image
    max_tokens:  cap the answer length; fewer tokens = faster reply (real-time mode)
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [base64.b64encode(image_bytes).decode()],
        "stream": False,  # one JSON object back, not a token stream
        "keep_alive": "30m",  # keep the model in VRAM so there is no reload pause
    }
    if max_tokens:
        payload["options"] = {"num_predict": max_tokens}
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())["response"].strip()


if __name__ == "__main__":
    import sys

    # Stage 1: python vlm.py photo.jpg ["question"]  -> ask about an existing image file
    if len(sys.argv) > 1:
        with open(sys.argv[1], "rb") as f:
            question = sys.argv[2] if len(sys.argv) > 2 else "Describe this image in detail."
            print(ask(f.read(), question))
        sys.exit()

    # Self-check: answer a question about a solid-red test image.
    # Red should be obvious to any working VLM; proves Ollama + model + GPU path.
    import cv2
    import numpy as np

    red = np.zeros((64, 64, 3), dtype=np.uint8)
    red[:, :] = (0, 0, 255)  # OpenCV is BGR
    ok, buf = cv2.imencode(".jpg", red)
    answer = ask(buf.tobytes(), "What is the single dominant color in this image? One word.")
    print("Model replied:", answer)
    assert "red" in answer.lower(), f"expected 'red' in answer, got: {answer!r}"
    print("OK: VLM pipeline works.")
