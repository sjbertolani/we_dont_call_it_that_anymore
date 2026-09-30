#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

.venv/bin/zoomtranslate manual samples/rel-ui-07.wav \
  --input-device "MacBook Pro Microphone" \
  --output-device BlackHole \
  --monitor-device "MacBook Pro Speakers" \
  --delay-ms 600 \
  --sample-gain 8 \
  --trigger-key t
