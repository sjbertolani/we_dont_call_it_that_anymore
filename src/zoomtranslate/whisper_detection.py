from __future__ import annotations

import queue
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import sounddevice as sd
import soundfile as sf


@dataclass(frozen=True)
class WhisperDetectionConfig:
    phrase: str
    aliases: tuple[str, ...] = ()
    input_device: int | str | None = None
    model: str = "base.en"
    sample_rate: int = 16_000
    block_ms: int = 100
    chunk_seconds: float = 2.4
    cooldown_seconds: float = 1.5
    min_rms: float = 0.006

    @property
    def block_size(self) -> int:
        return int(self.sample_rate * self.block_ms / 1000)

    @property
    def chunk_frames(self) -> int:
        return int(self.sample_rate * self.chunk_seconds)


def normalize_text(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else " " for ch in value)


def load_whisper_model(model_name: str):
    try:
        import whisper
    except ImportError as exc:
        raise SystemExit(
            "Missing openai-whisper. Install it with the pinned dependency flow in README."
        ) from exc
    return whisper.load_model(model_name)


def transcribe_chunk(model, chunk: np.ndarray) -> str:
    result = model.transcribe(
        chunk,
        language="en",
        fp16=False,
        condition_on_previous_text=False,
        verbose=False,
        without_timestamps=True,
    )
    return str(result.get("text", "")).strip()


def stream_whisper_detect(
    config: WhisperDetectionConfig,
    *,
    on_detection: Callable[[], None] | None = None,
) -> None:
    model = load_whisper_model(config.model)
    targets = tuple(
        value
        for value in [normalize_text(config.phrase).strip()]
        + [normalize_text(alias).strip() for alias in config.aliases]
        if value
    )
    audio_blocks: queue.Queue[np.ndarray] = queue.Queue()

    def callback(indata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            print(f"input status: {status}", flush=True)
        audio_blocks.put(indata[:, 0].copy())

    print(
        f"listening for {config.phrase!r} with openai-whisper {config.model} "
        f"({config.chunk_seconds:g}s chunks)",
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

            chunk = buffer[-config.chunk_frames :]
            buffer = buffer[-int(config.sample_rate * 0.6) :]
            rms = float(np.sqrt(np.mean(chunk**2)))
            if rms < config.min_rms:
                continue

            started_at = time.time()
            transcript = transcribe_chunk(model, chunk)
            elapsed_ms = int((time.time() - started_at) * 1000)
            if not transcript:
                continue

            stamp = time.strftime("%H:%M:%S")
            print(f"[{stamp}] {transcript} ({elapsed_ms} ms, rms={rms:.4f})", flush=True)
            normalized = normalize_text(transcript)
            now = time.time()
            if any(target in normalized for target in targets) and now - last_detection_at >= config.cooldown_seconds:
                last_detection_at = now
                print(f">>> DETECTED {config.phrase!r} at {stamp}", flush=True)
                if on_detection is not None:
                    on_detection()


def consume_whisper_audio(
    config: WhisperDetectionConfig,
    *,
    audio_blocks: queue.Queue[np.ndarray],
    input_sample_rate: int,
    on_detection: Callable[[], None] | None = None,
) -> None:
    model = load_whisper_model(config.model)
    targets = tuple(
        value
        for value in [normalize_text(config.phrase).strip()]
        + [normalize_text(alias).strip() for alias in config.aliases]
        if value
    )

    print(
        f"listening for {config.phrase!r} with openai-whisper {config.model} "
        f"from shared mic stream ({config.chunk_seconds:g}s chunks)",
        flush=True,
    )

    buffer = np.empty(0, dtype=np.float32)
    last_detection_at = 0.0
    while True:
        block = audio_blocks.get()
        block = resample_mono(block, input_sample_rate, config.sample_rate)
        buffer = np.concatenate([buffer, block])
        if len(buffer) < config.chunk_frames:
            continue

        chunk = buffer[-config.chunk_frames :]
        buffer = buffer[-int(config.sample_rate * 0.6) :]
        rms = float(np.sqrt(np.mean(chunk**2)))
        if rms < config.min_rms:
            continue

        started_at = time.time()
        transcript = transcribe_chunk(model, chunk)
        elapsed_ms = int((time.time() - started_at) * 1000)
        if not transcript:
            continue

        stamp = time.strftime("%H:%M:%S")
        print(f"[{stamp}] {transcript} ({elapsed_ms} ms, rms={rms:.4f})", flush=True)
        normalized = normalize_text(transcript)
        now = time.time()
        if any(target in normalized for target in targets) and now - last_detection_at >= config.cooldown_seconds:
            last_detection_at = now
            print(f">>> DETECTED {config.phrase!r} at {stamp}", flush=True)
            if on_detection is not None:
                on_detection()


def resample_mono(block: np.ndarray, input_sample_rate: int, output_sample_rate: int) -> np.ndarray:
    if block.ndim > 1:
        block = block[:, 0]
    block = block.astype(np.float32)
    if input_sample_rate == output_sample_rate:
        return block

    ratio = input_sample_rate / output_sample_rate
    if ratio.is_integer():
        return block[:: int(ratio)]

    input_times = np.arange(len(block), dtype=np.float32) / input_sample_rate
    output_len = int(len(block) * output_sample_rate / input_sample_rate)
    output_times = np.arange(output_len, dtype=np.float32) / output_sample_rate
    return np.interp(output_times, input_times, block).astype(np.float32)


def test_whisper_samples(
    *,
    phrase: str,
    aliases: tuple[str, ...] = (),
    paths: list[Path],
    model_name: str,
    min_rms: float,
) -> None:
    model = load_whisper_model(model_name)
    targets = tuple(
        value
        for value in [normalize_text(phrase).strip()]
        + [normalize_text(alias).strip() for alias in aliases]
        if value
    )
    for path in paths:
        audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
        if sample_rate != 16_000:
            raise SystemExit(f"{path} is {sample_rate} Hz; expected 16000 Hz")
        if audio.ndim > 1:
            audio = audio[:, 0]
        rms = float(np.sqrt(np.mean(audio**2))) if len(audio) else 0.0
        transcript = "" if rms < min_rms else transcribe_chunk(model, audio.astype(np.float32))
        detected = any(target in normalize_text(transcript) for target in targets)
        marker = "DETECTED" if detected else "miss"
        print(f"{marker}: {path.name}: {transcript!r} rms={rms:.4f}", flush=True)
