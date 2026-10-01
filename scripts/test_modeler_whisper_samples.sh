#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

.venv/bin/zoomtranslate test-whisper samples/modeler \
  --pattern "modeler-context-*.wav" \
  --phrase modeler \
  --alias model \
  --alias "model review" \
  --model base.en
