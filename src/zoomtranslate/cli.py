from __future__ import annotations

import argparse
import queue
import select
import sys
import termios
import threading
import tty
from pathlib import Path

import sounddevice as sd
import soundfile as sf

from zoomtranslate.audio_loop import (
    AudioLoopConfig,
    DelayedPassThrough,
    ManualSampleInjectingLoop,
    TimedSampleInjectingLoop,
)
from zoomtranslate.detection import DetectionConfig, stream_detect
from zoomtranslate.recording import play_file, record_take
from zoomtranslate.template_detection import (
    TemplateDetectionConfig,
    align_templates,
    crop_aligned_templates,
    stream_template_detect,
)
from zoomtranslate.whisper_detection import (
    WhisperDetectionConfig,
    consume_whisper_audio,
    stream_whisper_detect,
)
from zoomtranslate.whisper_detection import test_whisper_samples


def parse_device(value: str | None) -> int | str | None:
    if value is None:
        return None
    return int(value) if value.isdigit() else value


def resolve_device(value: str | None, kind: str) -> int | str | None:
    parsed = parse_device(value)
    if not isinstance(parsed, str):
        return parsed

    devices = sd.query_devices()
    matches: list[int] = []
    for index, device in enumerate(devices):
        channel_key = f"max_{kind}_channels"
        if parsed.lower() in device["name"].lower() and device[channel_key] > 0:
            matches.append(index)

    if len(matches) == 1:
        return matches[0]
    if not matches:
        return parsed
    names = ", ".join(f"{index}: {devices[index]['name']}" for index in matches)
    raise SystemExit(f"device name {value!r} is ambiguous for {kind}: {names}")


def list_devices() -> int:
    print(sd.query_devices())
    return 0


def run_loop(args: argparse.Namespace) -> int:
    config = AudioLoopConfig(
        sample_rate=args.sample_rate,
        block_ms=args.block_ms,
        delay_ms=args.delay_ms,
        input_device=resolve_device(args.input_device, "input"),
        output_device=resolve_device(args.output_device, "output"),
        monitor_device=resolve_device(args.monitor_device, "output"),
    )
    DelayedPassThrough(config).run(duration=args.duration)
    return 0


def record_samples(args: argparse.Namespace) -> int:
    slug = args.slug or args.phrase.lower().replace(" ", "-")
    for index in range(1, args.count + 1):
        output_path = args.output_dir / f"{slug}-{index:02d}.wav"
        record_take(
            output_path,
            phrase=args.phrase,
            duration=args.duration,
            sample_rate=args.sample_rate,
            channels=1,
            input_device=resolve_device(args.input_device, "input"),
        )
    return 0


def record_prompt_samples(args: argparse.Namespace) -> int:
    prompts = [line.strip() for line in args.prompts.read_text().splitlines() if line.strip()]
    if not prompts:
        raise SystemExit(f"no prompts found in {args.prompts}")

    for index, phrase in enumerate(prompts, start=1):
        output_path = args.output_dir / f"{args.slug}-{index:02d}.wav"
        print(f"\nPrompt {index}/{len(prompts)}: {phrase}")
        record_take(
            output_path,
            phrase=phrase,
            duration=args.duration,
            sample_rate=args.sample_rate,
            channels=1,
            input_device=resolve_device(args.input_device, "input"),
        )
    return 0


def audition_samples(args: argparse.Namespace) -> int:
    paths: list[Path] = []
    for value in args.paths:
        path = Path(value)
        if path.is_dir():
            paths.extend(sorted(path.glob("*.wav")))
        else:
            paths.append(path)

    if not paths:
        raise SystemExit("no WAV files found")

    for path in paths:
        play_file(
            path,
            output_device=resolve_device(args.output_device, "output"),
            gain=args.gain,
            repeat=args.repeat,
        )
    return 0


def inject_sample(args: argparse.Namespace) -> int:
    sample, sample_rate = sf.read(args.sample, dtype="float32", always_2d=True)
    config = AudioLoopConfig(
        sample_rate=sample_rate,
        block_ms=args.block_ms,
        delay_ms=args.delay_ms,
        input_device=resolve_device(args.input_device, "input"),
        output_device=resolve_device(args.output_device, "output"),
        monitor_device=resolve_device(args.monitor_device, "output"),
    )
    TimedSampleInjectingLoop(
        config,
        sample=sample,
        inject_after=args.inject_after,
        sample_gain=args.sample_gain,
    ).run(duration=args.duration)
    return 0


def manual_inject(args: argparse.Namespace) -> int:
    sample, sample_rate = sf.read(args.sample, dtype="float32", always_2d=True)
    config = AudioLoopConfig(
        sample_rate=sample_rate,
        block_ms=args.block_ms,
        delay_ms=args.delay_ms,
        input_device=resolve_device(args.input_device, "input"),
        output_device=resolve_device(args.output_device, "output"),
        monitor_device=resolve_device(args.monitor_device, "output"),
    )
    loop = ManualSampleInjectingLoop(
        config,
        sample=sample,
        sample_gain=args.sample_gain,
        duck_gain=args.duck_gain,
        mute_after_ms=args.mute_after_ms,
    )

    def read_triggers() -> None:
        print(f"press {args.trigger_key!r} to inject; press 'q' to quit", flush=True)
        if not sys.stdin.isatty():
            for line in sys.stdin:
                value = line.strip().lower()
                if value == "q":
                    loop._stop.set()
                    return
                if value in ("", args.trigger_key):
                    loop.trigger()
            return

        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())
            while not loop._stop.is_set():
                readable, _, _ = select.select([sys.stdin], [], [], 0.1)
                if not readable:
                    continue
                value = sys.stdin.read(1).lower()
                if value == "q":
                    loop._stop.set()
                    return
                if value == args.trigger_key:
                    loop.trigger()
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    threading.Thread(target=read_triggers, daemon=True).start()
    loop.run(duration=args.duration)
    return 0


def auto_whisper_replace(args: argparse.Namespace) -> int:
    sample, sample_rate = sf.read(args.sample, dtype="float32", always_2d=True)
    config = AudioLoopConfig(
        sample_rate=sample_rate,
        block_ms=args.block_ms,
        delay_ms=args.delay_ms,
        input_device=resolve_device(args.input_device, "input"),
        output_device=resolve_device(args.output_device, "output"),
        monitor_device=resolve_device(args.monitor_device, "output"),
    )
    loop = ManualSampleInjectingLoop(
        config,
        sample=sample,
        sample_gain=args.sample_gain,
        duck_gain=args.duck_gain,
        mute_after_ms=args.mute_after_ms,
    )
    detect_blocks: queue.Queue = queue.Queue()
    loop.add_input_tap(detect_blocks.put)
    detect_config = WhisperDetectionConfig(
        phrase=args.phrase,
        aliases=tuple(args.alias),
        input_device=resolve_device(args.input_device, "input"),
        model=args.model,
        sample_rate=args.detect_sample_rate,
        block_ms=args.detect_block_ms,
        chunk_seconds=args.chunk_seconds,
        min_rms=args.min_rms,
    )

    def schedule_trigger() -> None:
        if args.inject_delay_ms > 0:
            threading.Timer(args.inject_delay_ms / 1000, loop.trigger).start()
        else:
            loop.trigger()

    def run_detector() -> None:
        consume_whisper_audio(
            detect_config,
            audio_blocks=detect_blocks,
            input_sample_rate=config.sample_rate,
            on_detection=schedule_trigger,
        )

    threading.Thread(target=run_detector, daemon=True).start()
    loop.run(duration=args.duration)
    return 0


def detect_phrase(args: argparse.Namespace) -> int:
    config = DetectionConfig(
        phrase=args.phrase,
        aliases=tuple(args.alias),
        input_device=resolve_device(args.input_device, "input"),
        sample_rate=args.sample_rate,
        block_ms=args.block_ms,
        chunk_seconds=args.chunk_seconds,
        min_rms=args.min_rms,
        model_size=args.model,
        compute_type=args.compute_type,
    )
    stream_detect(config)
    return 0


def detect_template(args: argparse.Namespace) -> int:
    config = TemplateDetectionConfig(
        template_dir=args.template_dir,
        template_pattern=args.template_pattern,
        input_device=resolve_device(args.input_device, "input"),
        sample_rate=args.sample_rate,
        block_ms=args.block_ms,
        window_seconds=args.window_seconds,
        threshold=args.threshold,
        min_rms=args.min_rms,
        debug_scores=args.debug_scores,
    )
    stream_template_detect(config)
    return 0


def detect_whisper(args: argparse.Namespace) -> int:
    config = WhisperDetectionConfig(
        phrase=args.phrase,
        input_device=resolve_device(args.input_device, "input"),
        model=args.model,
        sample_rate=args.sample_rate,
        block_ms=args.block_ms,
        chunk_seconds=args.chunk_seconds,
    )
    stream_whisper_detect(config)
    return 0


def test_whisper(args: argparse.Namespace) -> int:
    paths: list[Path] = []
    for value in args.paths:
        path = Path(value)
        if path.is_dir():
            paths.extend(sorted(path.glob(args.pattern)))
        else:
            paths.append(path)
    if not paths:
        raise SystemExit("no sample files found")
    test_whisper_samples(
        phrase=args.phrase,
        paths=paths,
        model_name=args.model,
        min_rms=args.min_rms,
        aliases=tuple(args.alias),
    )
    return 0


def align_template(args: argparse.Namespace) -> int:
    align_templates(
        template_dir=args.template_dir,
        template_pattern=args.template_pattern,
        target_dir=args.target_dir,
        target_pattern=args.target_pattern,
        sample_rate=args.sample_rate,
    )
    return 0


def crop_template(args: argparse.Namespace) -> int:
    crop_aligned_templates(
        template_dir=args.template_dir,
        template_pattern=args.template_pattern,
        target_dir=args.target_dir,
        target_pattern=args.target_pattern,
        output_dir=args.output_dir,
        sample_rate=args.sample_rate,
        pad_seconds=args.pad_seconds,
        min_score=args.min_score,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="zoomtranslate")
    subparsers = parser.add_subparsers(dest="command", required=True)

    devices = subparsers.add_parser("devices", help="list audio devices")
    devices.set_defaults(func=lambda args: list_devices())

    loop = subparsers.add_parser("loop", help="run delayed pass-through audio")
    loop.add_argument("--input-device", default=None)
    loop.add_argument("--output-device", default=None)
    loop.add_argument("--monitor-device", default=None)
    loop.add_argument("--sample-rate", type=int, default=48_000)
    loop.add_argument("--block-ms", type=int, default=20)
    loop.add_argument("--delay-ms", type=int, default=600)
    loop.add_argument("--duration", type=float, default=None)
    loop.set_defaults(func=run_loop)

    record = subparsers.add_parser("record", help="record replacement phrase samples")
    record.add_argument("phrase", help="phrase to speak for every take")
    record.add_argument("--slug", default=None, help="filename prefix, defaults from phrase")
    record.add_argument("--count", type=int, default=10)
    record.add_argument("--duration", type=float, default=1.8)
    record.add_argument("--output-dir", type=Path, default=Path("samples"))
    record.add_argument("--input-device", default=None)
    record.add_argument("--sample-rate", type=int, default=48_000)
    record.set_defaults(func=record_samples)

    prompts = subparsers.add_parser("record-prompts", help="record one take per prompt line")
    prompts.add_argument("prompts", type=Path, help="text file with one prompt per line")
    prompts.add_argument("--slug", default="prompt")
    prompts.add_argument("--duration", type=float, default=2.4)
    prompts.add_argument("--output-dir", type=Path, default=Path("samples/prompts"))
    prompts.add_argument("--input-device", default=None)
    prompts.add_argument("--sample-rate", type=int, default=16_000)
    prompts.set_defaults(func=record_prompt_samples)

    audition = subparsers.add_parser("audition", help="play sample WAV files")
    audition.add_argument("paths", nargs="+", help="WAV file(s) or directories")
    audition.add_argument("--output-device", default=None)
    audition.add_argument("--gain", type=float, default=1.0)
    audition.add_argument("--repeat", type=int, default=1)
    audition.set_defaults(func=audition_samples)

    inject = subparsers.add_parser("inject", help="run mic pass-through and inject a sample")
    inject.add_argument("sample", type=Path)
    inject.add_argument("--input-device", default=None)
    inject.add_argument("--output-device", default=None)
    inject.add_argument("--monitor-device", default=None)
    inject.add_argument("--delay-ms", type=int, default=600)
    inject.add_argument("--block-ms", type=int, default=20)
    inject.add_argument("--inject-after", type=float, default=3.0)
    inject.add_argument("--sample-gain", type=float, default=1.0)
    inject.add_argument("--duration", type=float, default=8.0)
    inject.set_defaults(func=inject_sample)

    manual = subparsers.add_parser("manual", help="run mic pass-through and inject on Enter")
    manual.add_argument("sample", type=Path)
    manual.add_argument("--input-device", default=None)
    manual.add_argument("--output-device", default=None)
    manual.add_argument("--monitor-device", default=None)
    manual.add_argument("--delay-ms", type=int, default=600)
    manual.add_argument("--block-ms", type=int, default=20)
    manual.add_argument("--sample-gain", type=float, default=1.0)
    manual.add_argument("--duck-gain", type=float, default=0.35)
    manual.add_argument("--mute-after-ms", type=int, default=0)
    manual.add_argument("--trigger-key", default="t")
    manual.add_argument("--duration", type=float, default=None)
    manual.set_defaults(func=manual_inject)

    auto = subparsers.add_parser(
        "auto-whisper",
        help="run mic pass-through and inject a sample when Whisper detects a phrase",
    )
    auto.add_argument("sample", type=Path)
    auto.add_argument("--phrase", default="modeler")
    auto.add_argument("--alias", action="append", default=[])
    auto.add_argument("--input-device", default=None)
    auto.add_argument("--output-device", default=None)
    auto.add_argument("--monitor-device", default=None)
    auto.add_argument("--delay-ms", type=int, default=1200)
    auto.add_argument("--inject-delay-ms", type=int, default=0)
    auto.add_argument("--block-ms", type=int, default=20)
    auto.add_argument("--sample-gain", type=float, default=8.0)
    auto.add_argument("--duck-gain", type=float, default=0.35)
    auto.add_argument("--mute-after-ms", type=int, default=0)
    auto.add_argument("--model", default="base.en")
    auto.add_argument("--detect-sample-rate", type=int, default=16_000)
    auto.add_argument("--detect-block-ms", type=int, default=100)
    auto.add_argument("--chunk-seconds", type=float, default=2.4)
    auto.add_argument("--min-rms", type=float, default=0.006)
    auto.add_argument("--duration", type=float, default=None)
    auto.set_defaults(func=auto_whisper_replace)

    detect = subparsers.add_parser("detect", help="listen for a phrase and print detections")
    detect.add_argument("phrase", nargs="?", default="modeler")
    detect.add_argument("--input-device", default=None)
    detect.add_argument("--sample-rate", type=int, default=16_000)
    detect.add_argument("--block-ms", type=int, default=100)
    detect.add_argument("--chunk-seconds", type=float, default=2.0)
    detect.add_argument("--model", default="tiny.en")
    detect.add_argument("--compute-type", default="int8")
    detect.set_defaults(func=detect_phrase)

    whisper_detect = subparsers.add_parser(
        "detect-whisper",
        help="listen for a phrase with openai-whisper",
    )
    whisper_detect.add_argument("phrase", nargs="?", default="modeler")
    whisper_detect.add_argument("--alias", action="append", default=[])
    whisper_detect.add_argument("--input-device", default=None)
    whisper_detect.add_argument("--model", default="base.en")
    whisper_detect.add_argument("--sample-rate", type=int, default=16_000)
    whisper_detect.add_argument("--block-ms", type=int, default=100)
    whisper_detect.add_argument("--chunk-seconds", type=float, default=2.4)
    whisper_detect.add_argument("--min-rms", type=float, default=0.006)
    whisper_detect.set_defaults(func=detect_whisper)

    whisper_test = subparsers.add_parser(
        "test-whisper",
        help="transcribe sample files and report phrase detections",
    )
    whisper_test.add_argument("paths", nargs="+")
    whisper_test.add_argument("--pattern", default="*.wav")
    whisper_test.add_argument("--phrase", default="modeler")
    whisper_test.add_argument("--alias", action="append", default=[])
    whisper_test.add_argument("--model", default="base.en")
    whisper_test.add_argument("--min-rms", type=float, default=0.006)
    whisper_test.set_defaults(func=test_whisper)

    template = subparsers.add_parser(
        "detect-template",
        help="listen for a word using local recorded templates",
    )
    template.add_argument("template_dir", type=Path)
    template.add_argument("--template-pattern", default="*.wav")
    template.add_argument("--input-device", default=None)
    template.add_argument("--sample-rate", type=int, default=16_000)
    template.add_argument("--block-ms", type=int, default=50)
    template.add_argument("--window-seconds", type=float, default=1.2)
    template.add_argument("--threshold", type=float, default=0.82)
    template.add_argument("--min-rms", type=float, default=0.006)
    template.add_argument("--debug-scores", action="store_true")
    template.set_defaults(func=detect_template)

    align = subparsers.add_parser(
        "align-template",
        help="find likely template timestamps inside recorded WAV files",
    )
    align.add_argument("template_dir", type=Path)
    align.add_argument("target_dir", type=Path)
    align.add_argument("--template-pattern", default="modeler-0*.wav")
    align.add_argument("--target-pattern", default="modeler-context-*.wav")
    align.add_argument("--sample-rate", type=int, default=16_000)
    align.set_defaults(func=align_template)

    crop = subparsers.add_parser(
        "crop-template",
        help="crop likely word snippets from recorded WAV files",
    )
    crop.add_argument("template_dir", type=Path)
    crop.add_argument("target_dir", type=Path)
    crop.add_argument("output_dir", type=Path)
    crop.add_argument("--template-pattern", default="modeler-0*.wav")
    crop.add_argument("--target-pattern", default="modeler-context-*.wav")
    crop.add_argument("--sample-rate", type=int, default=16_000)
    crop.add_argument("--pad-seconds", type=float, default=0.08)
    crop.add_argument("--min-score", type=float, default=0.45)
    crop.set_defaults(func=crop_template)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
