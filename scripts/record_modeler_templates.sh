#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

.venv/bin/zoomtranslate record modeler \
  --slug modeler \
  --count 8 \
  --duration 1.4 \
  --output-dir samples/modeler \
  --input-device "MacBook Pro Microphone" \
  --sample-rate 16000
