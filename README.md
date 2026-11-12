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
| `python main.py` | Live webcam. SPACE describes, `q` asks your own question, `m` switches model, ESC quits |
| `python main.py qwen` | Same, starting with Qwen3-VL 2B |

To change the default model, edit `DEFAULT_MODEL` in `vlm.py`.

## Files
- `vlm.py`: the only Ollama-specific code. Swap models or backends here.
- `main.py`: camera loop and key handling.
