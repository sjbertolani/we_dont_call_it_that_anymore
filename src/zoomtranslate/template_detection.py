from __future__ import annotations

import queue
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf


@dataclass(frozen=True)
class TemplateDetectionConfig:
    template_dir: Path
    template_pattern: str = "*.wav"
    input_device: int | str | None = None
    sample_rate: int = 16_000
    block_ms: int = 50
    window_seconds: float = 1.4
    threshold: float = 0.82
    min_rms: float = 0.006
    cooldown_seconds: float = 1.5
    debug_scores: bool = False

    @property
    def block_size(self) -> int:
        return int(self.sample_rate * self.block_ms / 1000)

    @property
    def window_frames(self) -> int:
        return int(self.sample_rate * self.window_seconds)


def load_templates(template_dir: Path, sample_rate: int, pattern: str) -> list[np.ndarray]:
    paths = sorted(template_dir.glob(pattern))
    if not paths:
        raise SystemExit(f"no template WAV files matching {pattern!r} found in {template_dir}")

    templates: list[np.ndarray] = []
    for path in paths:
        audio, file_rate = sf.read(path, dtype="float32", always_2d=False)
        if file_rate != sample_rate:
            raise SystemExit(
                f"{path} is {file_rate} Hz, expected {sample_rate} Hz. "
                "Record templates with the same sample rate as detection."
            )
        active = active_speech_core(audio, sample_rate)
        templates.append(fingerprint(normalize_audio(active), sample_rate))
    return templates


def load_named_templates(template_dir: Path, sample_rate: int, pattern: str) -> list[tuple[Path, np.ndarray]]:
    paths = sorted(template_dir.glob(pattern))
    if not paths:
        raise SystemExit(f"no template WAV files matching {pattern!r} found in {template_dir}")

    templates: list[tuple[Path, np.ndarray]] = []
    for path in paths:
        audio, file_rate = sf.read(path, dtype="float32", always_2d=False)
        if file_rate != sample_rate:
            raise SystemExit(f"{path} is {file_rate} Hz, expected {sample_rate} Hz")
        active = active_speech_core(audio, sample_rate)
        templates.append((path, fingerprint(normalize_audio(active), sample_rate)))
    return templates


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    if audio.ndim > 1:
        audio = audio[:, 0]
    audio = audio.astype(np.float32)
    audio = audio - float(np.mean(audio))
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak > 0:
        audio = audio / peak
    return audio


def active_speech_core(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    if audio.ndim > 1:
        audio = audio[:, 0]
    audio = audio.astype(np.float32)
    frame = max(1, int(sample_rate * 0.02))
    energies = []
    for start in range(0, len(audio), frame):
        chunk = audio[start : start + frame]
        energies.append(float(np.sqrt(np.mean(chunk**2))) if len(chunk) else 0.0)
    if not energies:
        return audio

    values = np.array(energies)
    threshold = max(float(np.percentile(values, 65)) * 0.55, 0.004)
    active = np.flatnonzero(values >= threshold)
    if active.size == 0:
        return audio

    pad = int(0.08 * sample_rate / frame)
    start_frame = max(0, int(active[0]) - pad)
    end_frame = min(len(values), int(active[-1]) + pad + 1)
    return audio[start_frame * frame : min(len(audio), end_frame * frame)]


def best_template_score(window: np.ndarray, templates: list[np.ndarray], sample_rate: int) -> float:
    window_features = fingerprint(normalize_audio(window), sample_rate)
    best = 0.0
    for template in templates:
        score = max_feature_similarity(window_features, template)
        best = max(best, score)
    return best


def fingerprint(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    frame_size = int(sample_rate * 0.04)
    hop = int(sample_rate * 0.02)
    if len(audio) < frame_size:
        audio = np.pad(audio, (0, frame_size - len(audio)))

    bands = np.array([80, 160, 260, 400, 600, 850, 1200, 1700, 2400, 3400, 4800, 6800])
    freqs = np.fft.rfftfreq(frame_size, 1 / sample_rate)
    band_masks = []
    for low, high in zip(bands[:-1], bands[1:]):
        band_masks.append((freqs >= low) & (freqs < high))

    rows: list[np.ndarray] = []
    window = np.hanning(frame_size).astype(np.float32)
    for start in range(0, len(audio) - frame_size + 1, hop):
        frame = audio[start : start + frame_size] * window
        spectrum = np.abs(np.fft.rfft(frame))
        values = []
        for mask in band_masks:
            values.append(float(np.mean(spectrum[mask])) if np.any(mask) else 0.0)
        rows.append(np.log1p(values))

    features = np.array(rows, dtype=np.float32)
    if features.size == 0:
        return np.zeros((1, len(band_masks)), dtype=np.float32)
    features -= np.mean(features, axis=0, keepdims=True)
    std = np.std(features, axis=0, keepdims=True)
    features /= np.where(std > 1e-6, std, 1.0)
    return features


def max_feature_similarity(window: np.ndarray, template: np.ndarray) -> float:
    if len(window) < len(template):
        return 0.0
    template_flat = template.reshape(-1)
    template_norm = float(np.linalg.norm(template_flat))
    if template_norm == 0:
        return 0.0

    best = 0.0
    step = max(1, len(template) // 12)
    for start in range(0, len(window) - len(template) + 1, step):
        segment = window[start : start + len(template)]
        segment_flat = segment.reshape(-1)
        segment_norm = float(np.linalg.norm(segment_flat))
        if segment_norm == 0:
            continue
        score = max(0.0, float(np.dot(segment_flat, template_flat)) / (segment_norm * template_norm))
        best = max(best, score)
    return best


def best_feature_alignment(window: np.ndarray, template: np.ndarray) -> tuple[float, int]:
    if len(window) < len(template):
        return 0.0, 0
    template_flat = template.reshape(-1)
    template_norm = float(np.linalg.norm(template_flat))
    if template_norm == 0:
        return 0.0, 0

    best = 0.0
    best_start = 0
    step = max(1, len(template) // 16)
    for start in range(0, len(window) - len(template) + 1, step):
        segment = window[start : start + len(template)]
        segment_flat = segment.reshape(-1)
        segment_norm = float(np.linalg.norm(segment_flat))
        if segment_norm == 0:
            continue
        score = max(0.0, float(np.dot(segment_flat, template_flat)) / (segment_norm * template_norm))
        if score > best:
            best = score
            best_start = start
    return best, best_start


def align_templates(
    *,
    template_dir: Path,
    template_pattern: str,
    target_dir: Path,
    target_pattern: str,
    sample_rate: int,
) -> None:
    templates = load_named_templates(template_dir, sample_rate, template_pattern)
    targets = sorted(target_dir.glob(target_pattern))
    if not targets:
        raise SystemExit(f"no target WAV files matching {target_pattern!r} found in {target_dir}")

    feature_hop_seconds = 0.02
    for target in targets:
        audio, file_rate = sf.read(target, dtype="float32", always_2d=False)
        if file_rate != sample_rate:
            raise SystemExit(f"{target} is {file_rate} Hz, expected {sample_rate} Hz")
        target_features = fingerprint(normalize_audio(audio), sample_rate)

        best_score = 0.0
        best_start = 0
        best_template = templates[0][0]
        best_length = 0
        for template_path, template_features in templates:
            score, start = best_feature_alignment(target_features, template_features)
            if score > best_score:
                best_score = score
                best_start = start
                best_template = template_path
                best_length = len(template_features)

        start_seconds = best_start * feature_hop_seconds
        end_seconds = (best_start + best_length) * feature_hop_seconds
        print(
            f"{target.name}: modeler≈{start_seconds:.2f}-{end_seconds:.2f}s "
            f"score={best_score:.3f} template={best_template.name}",
            flush=True,
        )


def crop_aligned_templates(
    *,
    template_dir: Path,
    template_pattern: str,
    target_dir: Path,
    target_pattern: str,
    output_dir: Path,
    sample_rate: int,
    pad_seconds: float = 0.08,
    min_score: float = 0.45,
) -> None:
    templates = load_named_templates(template_dir, sample_rate, template_pattern)
    targets = sorted(target_dir.glob(target_pattern))
    if not targets:
        raise SystemExit(f"no target WAV files matching {target_pattern!r} found in {target_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_hop_seconds = 0.02
    written = 0
    for target in targets:
        audio, file_rate = sf.read(target, dtype="float32", always_2d=False)
        if file_rate != sample_rate:
            raise SystemExit(f"{target} is {file_rate} Hz, expected {sample_rate} Hz")
        target_features = fingerprint(normalize_audio(audio), sample_rate)

        best_score = 0.0
        best_start = 0
        best_length = 0
        for _template_path, template_features in templates:
            score, start = best_feature_alignment(target_features, template_features)
            if score > best_score:
                best_score = score
                best_start = start
                best_length = len(template_features)

        if best_score < min_score:
            print(f"skip {target.name}: score={best_score:.3f} < {min_score:.3f}", flush=True)
            continue

        start_seconds = max(0.0, best_start * feature_hop_seconds - pad_seconds)
        end_seconds = min(
            len(audio) / sample_rate,
            (best_start + best_length) * feature_hop_seconds + pad_seconds,
        )
        start_frame = int(start_seconds * sample_rate)
        end_frame = int(end_seconds * sample_rate)
        if end_frame <= start_frame:
            print(f"skip {target.name}: empty crop", flush=True)
            continue

        written += 1
        output_path = output_dir / f"modeler-word-{written:02d}.wav"
        sf.write(output_path, audio[start_frame:end_frame], sample_rate)
        print(
            f"wrote {output_path}: {start_seconds:.2f}-{end_seconds:.2f}s "
            f"score={best_score:.3f}",
            flush=True,
        )


def stream_template_detect(config: TemplateDetectionConfig) -> None:
    templates = load_templates(config.template_dir, config.sample_rate, config.template_pattern)
    audio_blocks: queue.Queue[np.ndarray] = queue.Queue()

    def callback(indata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            print(f"input status: {status}", flush=True)
        audio_blocks.put(indata[:, 0].copy())

    print(
        f"listening with {len(templates)} templates from {config.template_dir} "
        f"matching {config.template_pattern!r} "
        f"(threshold={config.threshold:g})",
        flush=True,
    )
    print("press Ctrl-C to stop", flush=True)

    buffer = np.empty(0, dtype=np.float32)
    last_detection_at = 0.0
    last_score_print_at = 0.0
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
            if len(buffer) > config.window_frames:
                buffer = buffer[-config.window_frames :]
            if len(buffer) < config.window_frames:
                continue

            rms = float(np.sqrt(np.mean(buffer**2)))
            score = best_template_score(buffer, templates, config.sample_rate)
            now = time.time()
            if config.debug_scores and now - last_score_print_at >= 0.25:
                last_score_print_at = now
                print(f"score={score:.3f} rms={rms:.4f}", flush=True)
            if (
                rms >= config.min_rms
                and score >= config.threshold
                and now - last_detection_at >= config.cooldown_seconds
            ):
                last_detection_at = now
                stamp = time.strftime("%H:%M:%S")
                print(f">>> DETECTED template at {stamp} score={score:.3f}", flush=True)
