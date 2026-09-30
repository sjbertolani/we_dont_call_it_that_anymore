from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf


def record_take(
    output_path: Path,
    *,
    phrase: str,
    duration: float,
    sample_rate: int,
    channels: int,
    input_device: int | str | None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nRecording: {phrase!r}")
    for remaining in range(3, 0, -1):
        print(f"{remaining}...", flush=True)
        time.sleep(1)
    print("speak now", flush=True)
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=channels,
        dtype="float32",
        device=input_device,
    )
    sd.wait()
    trimmed = trim_silence(audio, sample_rate)
    sf.write(output_path, trimmed, sample_rate)
    print(f"saved {output_path}")


def play_file(
    path: Path,
    *,
    output_device: int | str | None = None,
    gain: float = 1.0,
    repeat: int = 1,
) -> None:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    audio = np.clip(audio * gain, -1.0, 1.0)
    if repeat > 1:
        gap = np.zeros((int(sample_rate * 0.35), audio.shape[1]), dtype=np.float32)
        audio = np.concatenate([np.concatenate([audio, gap]) for _ in range(repeat)])
    print(f"playing {path} ({len(audio) / sample_rate:.3f}s, gain={gain:g}, repeat={repeat})")
    sd.play(audio, samplerate=sample_rate, device=output_device)
    sd.wait()


def trim_silence(audio: np.ndarray, sample_rate: int, threshold: float = 0.012) -> np.ndarray:
    mono = np.mean(np.abs(audio), axis=1)
    active = np.flatnonzero(mono > threshold)
    if active.size == 0:
        return audio

    pad = int(sample_rate * 0.08)
    start = max(0, int(active[0]) - pad)
    end = min(len(audio), int(active[-1]) + pad)
    return audio[start:end]
