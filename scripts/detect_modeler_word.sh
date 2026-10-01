#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

.venv/bin/zoomtranslate detect-template samples/modeler-word \
  --template-pattern "*.wav" \
  --input-device "MacBook Pro Microphone" \
  --sample-rate 16000 \
  --window-seconds 1.3 \
  --threshold 0.50 \
  --min-rms 0.006 \
  --debug-scores
