# visual-intel

A webcam + local AI system that sees, tracks, remembers and answers questions about its surroundings. Everything runs on the laptop; nothing goes to the cloud.

```
Webcam -> OpenCV (main.py)
            |-> YOLO11n on GPU (detect.py) -> tracker (track.py) -> events + alerts (watch.py)
            |-> Ollama VLM (vlm.py): scene descriptions, question answering
            |-> memory diary (memory.py): timestamped observations + snapshots
            `-> voice (voice.py): speech-to-text questions, spoken answers
```

## What it does
| Feature | How |
|---|---|
| Object detection and localisation | Boxes with names and IDs, position of each object (`top-left`, `centre`...), live count (`In view: 1 person, 2 cup`) |
| Real-time tracking | Each object keeps its ID across frames; `appeared` / `left` events |
| Visual reasoning | Ask `Is my phone on the desk?`; the VLM answers from the current image plus the detector's object list |
| Visual memory | Events, alerts, a scene description every 30 s and all Q&A go to `memory/log.jsonl` with snapshots; `Where did I leave my bottle?` is answered from it. Survives restarts |
| Anomaly alerts | Object disappeared (after at least 10 s in view); person entered the restricted zone; object left unattended (20 s with nobody in view) |
| Voice | Press `v` once and speak; recording stops when you stop talking; the answer is spoken in a soft female voice |

Known limits:
- Detecting that an object fell is not covered (needs a motion/pose model).
- "Left" and "right" mean the left and right of the camera image, which is the opposite of your own left and right when you face the camera.
- YOLO11n knows 80 object types and sometimes mislabels things; brief detections are ignored for alerts for that reason.
- The 4B model can misread times in the diary; treat exact times in memory answers as approximate.
- The on-screen text is ASCII only (OpenCV cannot draw other scripts); the terminal shows the full text.
- Voice input is English (`WHISPER = "base.en"` in `voice.py`).

## Requirements
- Ollama running, with `gemma3:4b` and/or `qwen3-vl:2b-instruct` pulled
- Python packages:
  ```
  pip install opencv-python onnxruntime-directml sounddevice
  pip install --no-deps faster-whisper
  pip install ctranslate2 tokenizers av
  ```
  Do not run a plain `pip install faster-whisper`: it installs CPU `onnxruntime`, which replaces `onnxruntime-directml` and puts YOLO back on the CPU. On a Raspberry Pi / Linux use `onnxruntime` (CPU) and `espeak` for speech.
- `yolo11n.onnx` in this folder (not in git, 10 MB). Make it with `pip install ultralytics` then `yolo export model=yolo11n.pt format=onnx`

## Use
```
python main.py          # or: python main.py qwen
```
| Where | Action |
|---|---|
| Video window | `SPACE` pause/resume descriptions, `m` switch model, `v` voice question, `z` clear zone, `ESC` or the X button to quit |
| Video window | Drag the mouse to draw the restricted zone (red box); a plain click clears it. A person whose centre enters it raises an alert |
| Terminal | Type a question + Enter. Empty Enter clears the pinned answer |

Each module also runs alone as a self-check:

| Command | Checks |
|---|---|
| `python vlm.py` | The model calls a red test image "red". `python vlm.py photo.jpg "question"` asks about a file |
| `python detect.py [photo.jpg]` | YOLO on one webcam frame (or a file); prints speed, saves `detect_test.jpg` |
| `python track.py` / `watch.py` / `memory.py` | Tracker, alert rules and diary, with made-up data (no camera) |
| `python voice.py` | The computer speaks a sentence into a file and Whisper must hear it back; the stop-when-silent rule; the microphone opens and closes 5 times |
| `python test_e2e.py` | The whole app for ~90 s with a simulated careless user: held keys, Caps Lock, Ollama going down, a crashing thread, closing the window, Ctrl+C. Sit in front of the camera while it runs |

## Measured on an RTX 5050 8 GB laptop
- Video 29 fps with everything running
- YOLO on every frame: 4 ms model time on GPU (DirectML), ~16 ms with pre/post-processing; CPU was ~35 ms
- VLM: ~3.5 s per description or answer (Gemma, 60-token cap). Ollama keeps only one of the two models in VRAM, so the first answer after `m` takes ~10 s

## Files and settings
- `vlm.py`: the only Ollama-specific code. `DEFAULT_MODEL` picks the model.
- `detect.py`: YOLO11n via ONNX Runtime (GPU on Windows, CPU elsewhere). `CONF`, `NMS`.
- `track.py`: IDs, positions, counts. `CONFIRM` (frames before an object counts), `FORGET` (seconds before it is gone).
- `watch.py`: events and alerts. `UNATTENDED` and `SETTLED` seconds.
- `memory.py`: the diary in `memory/` (not in git: it contains camera snapshots of your room and face; delete the folder to forget everything).
- `voice.py`: `WHISPER` model, `SILENCE` threshold (raise it in a noisy room), `VOICE_RATE`, `VOICE_VOLUME`.
- `main.py`: camera loop, threads, overlay, keys. `MEMORY_EVERY`, `MAX_TOKENS`, prompts.
