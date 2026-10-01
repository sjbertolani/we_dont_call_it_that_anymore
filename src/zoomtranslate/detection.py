from __future__ import annotations

import queue
import time
from dataclasses import dataclass

import numpy as np
import sounddevice as sd


@dataclass(frozen=True)
class DetectionConfig:
    phrase: str
    input_device: int | str | None = None
    sample_rate: int = 16_000
    block_ms: int = 100
    chunk_seconds: float = 2.0
    model_size: str = "tiny.en"
    compute_type: str = "int8"
    language: str = "en"

    @property
    def block_size(self) -> int:
        return int(self.sample_rate * self.block_ms / 1000)

    @property
    def chunk_frames(self) -> int:
        return int(self.sample_rate * self.chunk_seconds)


def normalize_text(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else " " for ch in value)


def stream_detect(config: DetectionConfig) -> None:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise SystemExit(
            "Missing faster-whisper. Install ASR deps with: "
            ".venv/bin/python -m pip install -e '.[asr]'"
        ) from exc

    model = WhisperModel(
        config.model_size,
        device="cpu",
        compute_type=config.compute_type,
    )
    phrase = normalize_text(config.phrase).strip()
    audio_blocks: queue.Queue[np.ndarray] = queue.Queue()

    def callback(indata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            print(f"input status: {status}", flush=True)
        audio_blocks.put(indata[:, 0].copy())

    print(
        f"listening for {config.phrase!r} "
        f"({config.model_size}, {config.chunk_seconds:g}s chunks)",
        flush=True,
    )
    print("press Ctrl-C to stop", flush=True)

    buffer = np.empty(0, dtype=np.float32)
    last_detection_at = 0.0
    with sd.InputStream(
        samplerate=config.sample_rate,
        blocksize=config.block_size,
        channels=1,
        device=config.input_device,
        dtype="float32",
        callback=callback,
    ):
        while True:
            block = audio_blocks.get()
            buffer = np.concatenate([buffer, block])
            if len(buffer) < config.chunk_frames:
                continue

            chunk = buffer[: config.chunk_frames]
            buffer = buffer[config.block_size :]
            started_at = time.time()
            segments, _info = model.transcribe(
                chunk,
                language=config.language,
                beam_size=1,
                vad_filter=True,
                condition_on_previous_text=False,
            )
            transcript = " ".join(segment.text.strip() for segment in segments).strip()
            if not transcript:
                continue

            elapsed_ms = int((time.time() - started_at) * 1000)
            now = time.strftime("%H:%M:%S")
            print(f"[{now}] {transcript} ({elapsed_ms} ms)", flush=True)

            normalized = normalize_text(transcript)
            if phrase in normalized and time.time() - last_detection_at > 1.5:
                last_detection_at = time.time()
                print(f">>> DETECTED {config.phrase!r} at {now}", flush=True)
