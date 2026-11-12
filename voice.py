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
BLOCK = 0.1               # seconds of audio per loudness measurement
SILENCE = 0.004           # loudness (RMS) below this is silence; raise it in a noisy room
MIN_SPEECH = 0.3          # seconds of loud audio needed to count as speech (a key click is shorter)
END_AFTER = 1.0           # seconds of silence after speech that end the recording
WAIT_FOR_SPEECH = 6       # give up if nobody speaks for this long
MAX_SECONDS = 15          # longest question

level = 0.0               # loudness of the newest block, for the on-screen meter
stage = ""                # "listening", "transcribing" or "" (idle), for the on-screen status
_model = None
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
    # Whisper invents text for noise; it also reports how likely a piece is 'not speech'. Drop those.
    return " ".join(s.text.strip() for s in segments if s.no_speech_prob < 0.6).strip()


def heard_enough(levels):
    """Look at the loudness history and decide whether to keep recording.

    Returns None = keep going, True = someone spoke and has now stopped (or time
    is up), False = nobody spoke at all.
    """
    threshold = max(SILENCE, 3 * min(levels))  # the quietest block is the room noise; speech is well above it
    loud = [x > threshold for x in levels]
    seconds = len(levels) * BLOCK
    if sum(loud) * BLOCK < MIN_SPEECH:  # nothing yet, or only a click (the 'v' key is right next to the mic)
        return False if seconds >= WAIT_FOR_SPEECH else None
    quiet_tail = loud[::-1].index(True) * BLOCK  # seconds since the last loud block
    return True if quiet_tail >= END_AFTER or seconds >= MAX_SECONDS else None


def listen():
    """Record one spoken question and return it as text ('' if nobody spoke).

    Recording stops by itself when the speaker stops, so there is no stop key.
    Only audio that contained speech is given to Whisper, because Whisper
    invents sentences when it is fed faint noise.
    """
    global level, stage
    if _speaking and _speaking.poll() is None:
        _speaking.kill()  # stop talking when the user wants to talk
    size = int(RATE * BLOCK)
    blocks, levels, done = [], [], None
    stage = "listening"
    try:
        with sd.InputStream(samplerate=RATE, channels=1, dtype="float32", blocksize=size) as mic:
            while done is None:
                data = mic.read(size)[0][:, 0]
                blocks.append(data.copy())
                level = float(np.sqrt((data ** 2).mean()))
                levels.append(level)
                done = heard_enough(levels)
        level, stage = 0.0, "transcribing"
        return transcribe(np.concatenate(blocks)) if done else ""
    finally:
        level, stage = 0.0, ""


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

    # The stop-by-itself rule, with made-up loudness values (one per 0.1 s).
    talk, hush = [0.05] * 10, [0.0] * 10
    assert heard_enough(talk) is None                      # still talking
    assert heard_enough(talk + hush[:5]) is None           # short pause between words
    assert heard_enough(talk + hush) is True               # spoke, then 1 s of silence
    assert heard_enough(hush * 3) is None and heard_enough(hush * 6) is False  # nobody spoke
    assert heard_enough([0.01] * 20 + [0.1] * 5 + [0.01] * 10) is True  # works over steady room noise
    assert heard_enough(([0.05] * 9 + [0.0]) * 15) is True  # 15 s of talking with short gaps: time is up
    click = [0.0] * 5 + [0.2] + [0.0] * 20
    assert heard_enough(click) is None and heard_enough(click + hush * 4) is False  # a key click is not speech

    # The microphone must open and close cleanly many times in a row (this crashed before).
    WAIT_FOR_SPEECH = 0.3
    for _ in range(5):
        listen()
    print("Microphone opened and closed 5 times without error.")

    say("Hello. Voice check passed.")
    _speaking.wait(timeout=30)
    assert _speaking.returncode == 0, "text-to-speech failed"
    print("OK: voice works.")
