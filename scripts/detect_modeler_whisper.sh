#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

.venv/bin/zoomtranslate detect-whisper modeler \
  --alias model \
  --alias "model review" \
  --input-device "MacBook Pro Microphone" \
  --model base.en \
  --chunk-seconds 2.4 \
  --min-rms 0.006
