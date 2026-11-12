# visual-intel

Webcam + local vision-language model (Ollama) that describes the scene and answers questions about it. Nothing goes to the cloud.

```
Webcam -> OpenCV (main.py) -> JPEG -> Ollama HTTP API (vlm.py) -> gemma3:4b / qwen3-vl:2b-instruct -> text
```

## Requirements
- Ollama running, with `gemma3:4b` and/or `qwen3-vl:2b-instruct` pulled
- Python with `opencv-python` (`pip install opencv-python`)

## Use
| Command | What it does |
|---|---|
| `python vlm.py` | Self-check: model must call a red test image "red" |
| `python vlm.py photo.jpg "What is on the table?"` | Ask about an existing image file |
| `python main.py` | Real-time: live video with the model's latest answer drawn on it, refreshed continuously |
| `python main.py qwen` | Same, starting with Qwen3-VL 2B |

While `main.py` runs:
- In the video window: SPACE pauses/resumes analysis, `m` switches model, ESC quits.
- In the terminal: type a question and press Enter (e.g. `Is there a person?`). It is asked about every new frame. An empty Enter goes back to describing the scene.

How "real time" works: the camera loop only shows frames, so video stays at full speed. A background thread sends the newest frame to the model as soon as the previous answer is done. Measured on an RTX 5050 8 GB: video 29 fps; Gemma ~3.3 s per answer (60-token cap, `MAX_TOKENS` in `main.py`). Ollama keeps only one of the two models in VRAM, so the first answer after pressing `m` takes ~10 s while the other model loads.

To change the default model, edit `DEFAULT_MODEL` in `vlm.py`.

## Files
- `vlm.py`: the only Ollama-specific code. Swap models or backends here.
- `main.py`: camera loop and key handling.
