"""Voice: ask questions by speaking, hear the answers.

Speech-to-text: faster-whisper ('base.en', runs on CPU in about a second).
Text-to-speech: the voice built into the OS, run as a separate process so the
video never waits for it (Windows: System.Speech; Raspberry Pi: espeak).

Install note: use  pip install --no-deps faster-whisper ; pip install ctranslate2 tokenizers av
A plain 'pip install faster-whisper' pulls in CPU onnxruntime, which replaces
onnxruntime-directml and puts YOLO back on the CPU.
"""
import os
import subprocess

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

RATE = 16000              # Whisper wants 16 kHz mono
WHISPER = "base.en"       # English only; use "base" or "small" for other languages
MAX_SECONDS = 30          # stop collecting audio after this, in case 'v' is never pressed again

_model = None
_chunks = []
_stream = None
_speaking = None

VOICE_RATE = -1           # -10 (slow) .. 10 (fast); a little slow sounds softer
VOICE_VOLUME = 80         # 0 .. 100

# A soft female voice: Windows picks its installed female voice (Microsoft Zira).
# The text arrives on stdin, never inside the command line, so nothing the model says can run as code.
_SPEAK_WINDOWS = ["powershell", "-NoProfile", "-Command",
                  "Add-Type -AssemblyName System.Speech; "
                  "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                  f"$s.SelectVoiceByHints('Female'); $s.Rate = {VOICE_RATE}; $s.Volume = {VOICE_VOLUME}; "
                  "$s.Speak([Console]::In.ReadToEnd())"]
_SPEAK_LINUX = ["espeak", "-v", "en+f3", "-s", "150", "--stdin"]  # f3 = a female espeak voice


def transcribe(audio):
    """float32 mono 16 kHz samples -> text."""
    global _model
    if _model is None:  # first use downloads the model (~75 MB) and loads it
        _model = WhisperModel(WHISPER, device="cpu", compute_type="int8")
    segments, _ = _model.transcribe(audio, language="en" if WHISPER.endswith(".en") else None)
    return " ".join(s.text.strip() for s in segments).strip()


def start():
    """Start recording from the default microphone."""
    global _stream
    _chunks.clear()

    def collect(data, frames, when, status):
        if len(_chunks) * frames < MAX_SECONDS * RATE:
            _chunks.append(data.copy())

    _stream = sd.InputStream(samplerate=RATE, channels=1, dtype="float32", callback=collect)
    _stream.start()


def stop():
    """Stop recording and return what was said as text ('' if nothing)."""
    _stream.stop()
    _stream.close()
    if not _chunks:
        return ""
    return transcribe(np.concatenate(_chunks)[:, 0])


def say(text):
    """Speak text aloud; returns immediately. New speech cuts off old speech."""
    global _speaking
    if _speaking and _speaking.poll() is None:
        _speaking.kill()
    _speaking = subprocess.Popen(_SPEAK_WINDOWS if os.name == "nt" else _SPEAK_LINUX,
                                 stdin=subprocess.PIPE, text=True)
    _speaking.stdin.write(text)
    _speaking.stdin.close()


if __name__ == "__main__":
    # Self-check without a human: the computer's own voice writes a WAV file,
    # Whisper must hear the same words back. (Windows only; uses the built-in voice.)
    import tempfile
    import wave

    wav = os.path.join(tempfile.mkdtemp(), "check.wav")
    subprocess.run(["powershell", "-NoProfile", "-Command",
                    "Add-Type -AssemblyName System.Speech; "
                    "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                    "$f = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, 'Sixteen', 'Mono'); "
                    f"$s.SetOutputToWaveFile('{wav}', $f); $s.Speak('Is my phone on the desk?'); $s.Dispose()"],
                   check=True)
    with wave.open(wav) as f:
        assert f.getframerate() == RATE and f.getnchannels() == 1
        audio = np.frombuffer(f.readframes(f.getnframes()), np.int16).astype(np.float32) / 32768
    heard = transcribe(audio)
    print("Whisper heard:", heard)
    assert "phone" in heard.lower() and "desk" in heard.lower(), heard
    say("Hello. Voice check passed.")
    _speaking.wait(timeout=30)
    assert _speaking.returncode == 0, "text-to-speech failed"
    print("OK: voice works.")
