from __future__ import annotations

import queue
import signal
import threading
import time as time_module
from dataclasses import dataclass

import numpy as np
import sounddevice as sd


@dataclass(frozen=True)
class AudioLoopConfig:
    sample_rate: int = 48_000
    channels: int = 1
    block_ms: int = 20
    delay_ms: int = 600
    input_device: int | str | None = None
    output_device: int | str | None = None
    monitor_device: int | str | None = None

    @property
    def block_size(self) -> int:
        return int(self.sample_rate * self.block_ms / 1000)

    @property
    def delay_blocks(self) -> int:
        return max(1, int(self.delay_ms / self.block_ms))


class DelayedPassThrough:
    """A fixed-delay mic-to-output loop.

    This is milestone 1: prove routing before adding detection or rewrite logic.
    """

    def __init__(self, config: AudioLoopConfig) -> None:
        self.config = config
        self._queues: list[queue.Queue[np.ndarray]] = []
        self._stop = threading.Event()
        self._new_queue()

    def _new_queue(self) -> queue.Queue[np.ndarray]:
        blocks: queue.Queue[np.ndarray] = queue.Queue()
        silence = np.zeros((self.config.block_size, self.config.channels), dtype=np.float32)
        for _ in range(self.config.delay_blocks):
            blocks.put(silence.copy())
        self._queues.append(blocks)
        return blocks

    def input_callback(self, indata: np.ndarray, frames: int, time, status) -> None:
        if status:
            print(f"input status: {status}", flush=True)
        for blocks in self._queues:
            blocks.put(indata.copy())

    def output_callback_for(self, blocks: queue.Queue[np.ndarray]):
        def callback(outdata: np.ndarray, frames: int, time, status) -> None:
            self.output_callback(outdata, frames, time, status, blocks=blocks)

        return callback

    def output_callback(
        self,
        outdata: np.ndarray,
        frames: int,
        time,
        status,
        *,
        blocks: queue.Queue[np.ndarray] | None = None,
    ) -> None:
        if status:
            print(f"output status: {status}", flush=True)
        blocks = blocks or self._queues[0]
        try:
            block = blocks.get_nowait()
        except queue.Empty:
            block = np.zeros((frames, self.config.channels), dtype=np.float32)
        outdata[:] = block[:frames]

    def run(self, *, duration: float | None = None) -> None:
        signal.signal(signal.SIGINT, lambda *_: self._stop.set())
        signal.signal(signal.SIGTERM, lambda *_: self._stop.set())

        main_queue = self._queues[0]
        streams = [
            sd.InputStream(
                samplerate=self.config.sample_rate,
                blocksize=self.config.block_size,
                channels=self.config.channels,
                device=self.config.input_device,
                dtype="float32",
                callback=self.input_callback,
            ),
            sd.OutputStream(
                samplerate=self.config.sample_rate,
                blocksize=self.config.block_size,
                channels=self.config.channels,
                device=self.config.output_device,
                dtype="float32",
                callback=self.output_callback_for(main_queue),
            ),
        ]
        if self.config.monitor_device is not None:
            monitor_queue = self._new_queue()
            streams.append(
                sd.OutputStream(
                    samplerate=self.config.sample_rate,
                    blocksize=self.config.block_size,
                    channels=self.config.channels,
                    device=self.config.monitor_device,
                    dtype="float32",
                    callback=self.output_callback_for(monitor_queue),
                )
            )

        with streams[0], streams[1]:
            if len(streams) > 2:
                streams[2].start()
            try:
                self._wait(duration=duration)
            finally:
                if len(streams) > 2:
                    streams[2].stop()

    def _wait(self, *, duration: float | None = None) -> None:
        print(
            "audio loop running "
            f"({self.config.sample_rate} Hz, {self.config.delay_ms} ms delay)",
            flush=True,
        )
        stop_at = time_module.monotonic() + duration if duration is not None else None
        while not self._stop.wait(0.2):
            if stop_at is not None and time_module.monotonic() >= stop_at:
                break
            pass

    def _old_run(self) -> None:
        with sd.InputStream(
            samplerate=self.config.sample_rate,
            blocksize=self.config.block_size,
            channels=self.config.channels,
            device=self.config.input_device,
            dtype="float32",
            callback=self.input_callback,
        ), sd.OutputStream(
            samplerate=self.config.sample_rate,
            blocksize=self.config.block_size,
            channels=self.config.channels,
            device=self.config.output_device,
            dtype="float32",
            callback=self.output_callback,
        ):
            self._wait()


class TimedSampleInjectingLoop(DelayedPassThrough):
    """Mic pass-through that replaces output with a sample after a fixed delay."""

    def __init__(
        self,
        config: AudioLoopConfig,
        *,
        sample: np.ndarray,
        inject_after: float,
        sample_gain: float = 1.0,
    ) -> None:
        super().__init__(config)
        self.sample = self._match_channels(sample.astype(np.float32), config.channels) * sample_gain
        self.sample = np.clip(self.sample, -1.0, 1.0)
        self.inject_after = inject_after
        self._started_at: float | None = None
        self._sample_offsets: dict[int, int | None] = {}

    def _match_channels(self, audio: np.ndarray, channels: int) -> np.ndarray:
        if audio.ndim == 1:
            audio = audio[:, None]
        if audio.shape[1] == channels:
            return audio
        if audio.shape[1] == 1 and channels > 1:
            return np.repeat(audio, channels, axis=1)
        return audio[:, :channels]

    def output_callback(
        self,
        outdata: np.ndarray,
        frames: int,
        time,
        status,
        *,
        blocks: queue.Queue[np.ndarray] | None = None,
    ) -> None:
        offset_key = id(blocks) if blocks is not None else id(self._queues[0])
        if offset_key not in self._sample_offsets:
            self._sample_offsets[offset_key] = None

        if self._started_at is None:
            self._started_at = time_module.monotonic()

        elapsed = time_module.monotonic() - self._started_at
        sample_offset = self._sample_offsets[offset_key]
        if sample_offset is None and elapsed >= self.inject_after:
            sample_offset = 0

        if sample_offset is not None and sample_offset < len(self.sample):
            start = sample_offset
            end = min(start + frames, len(self.sample))
            chunk = self.sample[start:end]
            outdata.fill(0)
            outdata[: len(chunk)] = chunk
            self._sample_offsets[offset_key] = end
            return

        super().output_callback(outdata, frames, time, status, blocks=blocks)


class ManualSampleInjectingLoop(DelayedPassThrough):
    """Mic pass-through that injects a sample when `trigger()` is called."""

    def __init__(
        self,
        config: AudioLoopConfig,
        *,
        sample: np.ndarray,
        sample_gain: float = 1.0,
    ) -> None:
        super().__init__(config)
        self.sample = self._match_channels(sample.astype(np.float32), config.channels) * sample_gain
        self.sample = np.clip(self.sample, -1.0, 1.0)
        self._lock = threading.Lock()
        self._trigger_generation = 0
        self._sample_offsets: dict[int, tuple[int, int | None]] = {}

    def _match_channels(self, audio: np.ndarray, channels: int) -> np.ndarray:
        if audio.ndim == 1:
            audio = audio[:, None]
        if audio.shape[1] == channels:
            return audio
        if audio.shape[1] == 1 and channels > 1:
            return np.repeat(audio, channels, axis=1)
        return audio[:, :channels]

    def trigger(self) -> None:
        with self._lock:
            self._trigger_generation += 1
        print("injected sample", flush=True)

    def output_callback(
        self,
        outdata: np.ndarray,
        frames: int,
        time,
        status,
        *,
        blocks: queue.Queue[np.ndarray] | None = None,
    ) -> None:
        offset_key = id(blocks) if blocks is not None else id(self._queues[0])
        with self._lock:
            generation = self._trigger_generation

        seen_generation, sample_offset = self._sample_offsets.get(offset_key, (0, None))
        if generation > seen_generation:
            sample_offset = 0
            seen_generation = generation

        if sample_offset is not None and sample_offset < len(self.sample):
            start = sample_offset
            end = min(start + frames, len(self.sample))
            chunk = self.sample[start:end]
            outdata.fill(0)
            outdata[: len(chunk)] = chunk
            self._sample_offsets[offset_key] = (seen_generation, end)
            return

        self._sample_offsets[offset_key] = (seen_generation, sample_offset)
        super().output_callback(outdata, frames, time, status, blocks=blocks)
