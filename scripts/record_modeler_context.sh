#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

.venv/bin/zoomtranslate record-prompts prompts/modeler_context.txt \
  --slug modeler-context \
  --duration 2.4 \
  --output-dir samples/modeler \
  --input-device "MacBook Pro Microphone" \
  --sample-rate 16000
