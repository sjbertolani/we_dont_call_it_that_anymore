# ZoomTranslate

Real-time same-language speech rewriting for meetings.

Goal: speak naturally into the microphone, rewrite selected words or phrases
before they reach Zoom/Google Meet, and keep the output fast enough that it
feels like a normal live call.

Example:

```text
spoken: "the modeler should update this"
heard:  "the rel UI should update this"
```

## Core Decision

The live meeting path should run locally.

Runpod is useful for offline experiments such as training/evaluating speech
models, generating replacement phrase samples, or batch-testing accuracy. It is
not a good fit for the real-time audio path because round-trip network latency,
cold starts, jitter, and failure modes are too risky for a live call.

## Recommended V1

Start with deterministic local phrase replacement:

1. Capture microphone audio locally.
2. Keep a short delay buffer, probably 300-900 ms.
3. Detect configured words/phrases in the stream.
4. Replace only the matched audio span with a prerecorded replacement phrase in
   the speaker's own voice.
5. Output the rewritten stream to a virtual microphone selected by Zoom/Meet.

This avoids solving full real-time voice cloning on day one. For a small set of
replacement terms, prerecorded clips will sound more natural, run faster, and be
much easier to debug.

## Audio Routing

On macOS, the expected routing is:

```text
Physical mic -> ZoomTranslate app -> virtual audio device -> Zoom/Meet mic input
                                      |
                                      +-> local monitor/headphones, optional
```

Practical virtual device options:

- BlackHole: free/open-source virtual audio driver.
- Loopback: paid, much easier routing and monitoring UI.

For the first prototype, BlackHole is enough if we only need one processed
virtual mic. Loopback may be worth it once routing gets more complex.

## Detection Strategy

Whisper-style transcription is useful, but it may not be the fastest first
choice for a tiny phrase list.

Use a staged detector:

1. Voice activity detection to split speech from silence.
2. Lightweight keyword/phrase spotting for configured phrases.
3. Whisper/faster-whisper only as a fallback or validation path.

For "modeler" -> "rel UI", the system does not need a complete transcript. It
needs to know when that exact word was spoken and where the word starts/ends in
the audio buffer.

## Replacement Strategy

Best V1 replacement:

- Record 10-30 samples of each target replacement phrase in the user's voice.
- Normalize loudness.
- Trim leading/trailing silence.
- Keep multiple variants.
- Select a variant based on surrounding energy/speaking rate.
- Crossfade into and out of the original audio.

Potential V2 replacement:

- Use local voice conversion or TTS to generate arbitrary replacement phrases.
- Train/adapt voice samples offline, possibly on Runpod.
- Cache generated phrases locally before meetings.

Avoid generating fresh voice-cloned audio during a meeting unless latency tests
prove it is stable.

## Latency Budget

Target end-to-end delay:

```text
excellent: < 300 ms
usable:    300-900 ms
risky:     > 1000 ms
```

Some delay is unavoidable because the app must hear enough of the word before it
can safely rewrite it. The prototype should expose this delay as a setting.

## Proposed Local Stack

- Language: Python first, Swift/Rust later if needed.
- Audio I/O: `sounddevice` or `pyaudio` for prototype.
- VAD: Silero VAD or WebRTC VAD.
- ASR/spotting: `faster-whisper` for validation; dedicated keyword spotting if
  Whisper is too slow.
- DSP: NumPy/SciPy for trimming, crossfade, loudness, and buffering.
- Virtual device: BlackHole or Loopback.

If Python cannot meet the latency target, move the audio engine to Swift/Rust and
keep Python only for model experiments.

## Prototype Milestones

### Milestone 1: Audio Loop

- Capture physical microphone.
- Add fixed delay buffer.
- Output unchanged audio to virtual mic.
- Confirm Zoom/Meet can use the virtual mic.

### Milestone 2: Manual Rewrite

- Press a hotkey to replace the next buffered word-sized region with a recorded
  `rel UI` clip.
- Tune crossfade and gain matching.

### Milestone 3: Automatic Keyword Rewrite

- Detect `modeler`.
- Replace it with a selected `rel UI` clip.
- Log timings: detection latency, replacement latency, output delay.

### Milestone 4: Phrase Config

- Add a config file:

```yaml
phrases:
  - from: modeler
    to: rel UI
    samples:
      - samples/rel-ui-01.wav
      - samples/rel-ui-02.wav
```

### Milestone 5: Voice/Model Experiments

- Record a small voice dataset.
- Test local TTS/voice conversion for cached replacement phrases.
- Use Runpod only if local training/inference is too slow.

## Open Questions

- macOS only, or do we need Windows support later?
- Outgoing mic only, or should incoming meeting audio be processed too?
- How many phrases need rewriting initially?
- Is a small delay acceptable if it makes replacement much cleaner?
- Should the app fail open, meaning original mic audio passes through if the
  model crashes?
