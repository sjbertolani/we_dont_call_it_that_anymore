from __future__ import annotations

import argparse
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
from zoomtranslate.recording import play_file, record_take


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
    manual.add_argument("--trigger-key", default="t")
    manual.add_argument("--duration", type=float, default=None)
    manual.set_defaults(func=manual_inject)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
