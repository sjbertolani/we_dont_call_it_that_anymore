#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

.venv/bin/zoomtranslate detect-template samples/modeler \
  --template-pattern "modeler-0*.wav" \
  --input-device "MacBook Pro Microphone" \
  --sample-rate 16000 \
  --window-seconds 1.4 \
  --threshold 0.72 \
  --min-rms 0.006
