#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

.venv/bin/zoomtranslate auto-whisper samples/rel-ui-07.wav \
  --phrase modeler \
  --alias model \
  --alias "model review" \
  --input-device "MacBook Pro Microphone" \
  --output-device BlackHole \
  --monitor-device "MacBook Pro Speakers" \
  --delay-ms 1200 \
  --sample-gain 8 \
  --duck-gain 0.35 \
  --mute-after-ms 0 \
  --model base.en \
  --chunk-seconds 2.4 \
  --min-rms 0.006
