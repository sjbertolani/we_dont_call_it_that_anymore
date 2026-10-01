#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

.venv/bin/zoomtranslate detect modeler \
  --input-device "MacBook Pro Microphone" \
  --model tiny.en \
  --chunk-seconds 2
