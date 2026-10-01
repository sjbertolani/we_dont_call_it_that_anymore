#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

.venv/bin/zoomtranslate crop-template samples/modeler samples/modeler samples/modeler-word \
  --template-pattern "modeler-0*.wav" \
  --target-pattern "modeler-context-*.wav" \
  --sample-rate 16000 \
  --pad-seconds 0.08 \
  --min-score 0.45
