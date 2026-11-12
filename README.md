# visual-intel

Webcam + local vision-language model (Ollama) that describes the scene and answers questions about it. Nothing goes to the cloud.

```
Webcam -> OpenCV (main.py) -+-> YOLO11n via ONNX Runtime GPU (detect.py) -> green boxes, every frame
                            +-> Ollama VLM (vlm.py) gemma3 / qwen3-vl -> scene description, every ~3 s
```

## Requirements
- Ollama running, with `gemma3:4b` and/or `qwen3-vl:2b-instruct` pulled
- Python with `opencv-python` and ONNX Runtime: `pip install opencv-python onnxruntime-directml` (Windows, GPU through DirectX 12). On a Raspberry Pi / Linux use `pip install onnxruntime` instead (CPU)
- `yolo11n.onnx` in this folder (not in git, 10 MB). Make it with `pip install ultralytics` then `yolo export model=yolo11n.pt format=onnx`

## Use
| Command | What it does |
|---|---|
| `python vlm.py` | Self-check: model must call a red test image "red" |
| `python vlm.py photo.jpg "What is on the table?"` | Ask about an existing image file |
| `python detect.py [photo.jpg]` | YOLO boxes on one webcam frame (or a file), saved to `detect_test.jpg` |
| `python main.py` | Real-time: live video with YOLO boxes and the VLM's latest answer drawn on it |
| `python main.py qwen` | Same, starting with Qwen3-VL 2B |

While `main.py` runs:
- In the video window: SPACE pauses/resumes analysis, `m` switches model, ESC quits.
- In the terminal: type a question and press Enter (e.g. `Is there a person?`). It is asked about every new frame. An empty Enter goes back to describing the scene.

How "real time" works: the camera loop only shows frames, so video stays at full speed. One background thread runs YOLO on each new frame; another sends the newest frame to the VLM as soon as the previous answer is done. Measured on an RTX 5050 8 GB: video 29 fps, YOLO on every frame (GPU via DirectML: 4 ms model time, ~16 ms with pre/post-processing; CPU was ~35 ms); Gemma ~3.3 s per answer (60-token cap, `MAX_TOKENS` in `main.py`). Ollama keeps only one of the two models in VRAM, so the first answer after pressing `m` takes ~10 s while the other model loads.

To change the default model, edit `DEFAULT_MODEL` in `vlm.py`.

## Files
- `vlm.py`: the only Ollama-specific code. Swap models or backends here.
- `detect.py`: YOLO11n object boxes. ONNX Runtime runs the model (GPU on Windows, CPU elsewhere, no PyTorch); OpenCV prepares the image, removes duplicate boxes and draws. Tune `CONF` / `NMS` there.
- `main.py`: camera loop, background threads, overlay and key handling.
