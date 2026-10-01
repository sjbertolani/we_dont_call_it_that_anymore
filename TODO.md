# ZoomTranslate TODO

## Current Status

The prototype can route live microphone audio through a local Python process into
Zoom via BlackHole.

Working pieces:

- MacBook microphone capture with `sounddevice`.
- Delayed pass-through to `BlackHole 2ch`.
- Local monitor output to speakers/headphones.
- Manual `t` key trigger for injecting a prerecorded `rel UI` clip.
- OpenAI Whisper `base.en` detection for `modeler`.
- Alias matching for common Whisper mishearings:
  - `model`
  - `model review`
- Automatic trigger from Whisper detection.
- Shared mic stream for pass-through and Whisper, so Whisper no longer opens a
  second Core Audio input stream.
- Replacement-window controls:
  - `--delay-ms`
  - `--inject-delay-ms`
  - `--duck-gain`
  - `--mute-after-ms`

Known working test command:

```bash
.venv/bin/zoomtranslate auto-whisper samples/rel-ui-07.wav \
  --phrase modeler \
  --alias model \
  --alias "model review" \
  --input-device "MacBook Pro Microphone" \
  --output-device BlackHole \
  --monitor-device "MacBook Pro Speakers" \
  --delay-ms 3000 \
  --inject-delay-ms 800 \
  --sample-gain 8 \
  --duck-gain 0 \
  --mute-after-ms 400 \
  --model base.en \
  --chunk-seconds 2.4 \
  --min-rms 0.006
```

Zoom setup:

- Microphone: `BlackHole 2ch`
- Original Sound / musician mode: on
- Background noise suppression: low/off
- Automatic microphone volume: off if available

## Installation Notes

Installed native pieces:

- Homebrew `portaudio`
- BlackHole 2ch
- FFmpeg already present

Python runtime:

- Python 3.11 virtualenv in `.venv`
- `sounddevice`
- `soundfile`
- `numpy`
- `pyyaml`
- `openai-whisper`
- `torch==2.2.2`
- `numba==0.59.1`
- `llvmlite==0.42.0`
- `numpy==1.26.4`

Important dependency note:

- `faster-whisper` failed because PyAV would not build cleanly.
- Homebrew `whisper.cpp` failed because the current formula requires newer Xcode
  than this Intel Mac has.
- OpenAI Whisper works with pinned `numpy`/`numba`/`llvmlite`.

## Open Problems

### Timing Alignment

Detection works, but replacement timing is still rough. Whisper detects after
the word has been spoken, so the outgoing audio needs enough delay to allow
replacement before Zoom hears the original word.

Current test values:

```text
delay-ms:        3000
inject-delay-ms: 800
duck-gain:       0
mute-after-ms:   400
```

Need to tune:

- Whether `inject-delay-ms` should be smaller or larger.
- Whether `delay-ms` can be reduced below 3000ms.
- Whether the mute window should begin earlier.
- Whether the mute window is too aggressive and cuts neighboring words.

### Word Removal

The system can inject `rel UI`, but fully hiding the original `modeler` depends
on timing. Current approach mutes the mic under the replacement window, but the
window is not yet aligned to the exact original word location.

Future approach:

- Track the position of the detected word inside the Whisper chunk.
- Map that timestamp into the delayed audio queue.
- Apply mute/crossfade exactly over that region.

### Audio Quality

Current replacement uses one recorded sample:

```text
samples/rel-ui-07.wav
```

Need to:

- Rotate between multiple good `rel UI` samples.
- Normalize sample loudness.
- Add short crossfades at mute/replacement boundaries.
- Avoid clipping when sample gain is high.

### Logging Noise

OpenAI Whisper prints progress bars during live detection. This is harmless but
messy. Suppress or redirect progress output.

### Core Audio Warning

Sometimes this appears:

```text
PaMacCore (AUHAL) err='-50'
```

The stream usually continues. Investigate if it correlates with monitor output,
BlackHole, sample rate mismatch, or multiple output streams.

## Next Tasks

1. Tune replacement timing by ear with:
   - `--delay-ms`
   - `--inject-delay-ms`
   - `--mute-after-ms`
2. Add replacement clip rotation across good `rel UI` samples.
3. Add crossfade around mute/replacement window.
4. Suppress Whisper progress bars.
5. Add a single script for the current recommended auto test settings.
6. Consider timestamp-aware Whisper decoding so replacement can be scheduled
   against word timing rather than rough post-detection delay.
7. Commit future sample metadata, but keep raw voice samples ignored.

