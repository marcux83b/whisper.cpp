# whisper.cpp/examples/stream

This is a naive example of performing real-time inference on audio from your microphone
or from raw PCM provided on stdin. The `whisper-stream` tool samples the audio every
half a second (or uses end‑of‑speech when reading stdin) and runs the transcription continuously.
More info is available in [issue #10](https://github.com/ggerganov/whisper.cpp/issues/10).

```bash
./build/bin/whisper-stream -m ./models/ggml-base.en.bin -t 8 --step 500 --length 5000
```

https://user-images.githubusercontent.com/1991296/194935793-76afede7-cfa8-48d8-a80f-28ba83be7d09.mp4

## Sliding window mode with VAD

Setting the `--step` argument to `0` enables the sliding window mode:

```bash
 ./build/bin/whisper-stream -m ./models/ggml-base.en.bin -t 6 --step 0 --length 30000 -vth 0.6
```

In this mode, the tool will transcribe only after some speech activity is detected. A very
basic VAD detector is used, but in theory a more sophisticated approach can be added. The
`-vth` argument determines the VAD threshold - higher values will make it detect silence more often.
It's best to tune it to the specific use case, but a value around `0.6` should be OK in general.
When silence is detected, it will transcribe the last `--length` milliseconds of audio and output
a transcription block that is suitable for parsing.

## Stdin streaming with EOS (end‑of‑speech)

`whisper-stream` can read PCM from stdin via `--stdin`, using an internal end‑of‑speech detector
that frames audio in 20 ms chunks and flushes utterances as soon as speech ends.

- Input format: `--stdin-format f32le` (default) or `s16le`, mono, 16 kHz
- Behavior: adds a small pre‑roll to avoid clipping starts, ignores tiny blips, and
  flushes immediately on silence or long zero runs. Long, continuous speech is still sliced by
  `--length`/`--keep` just like microphone mode.

Examples

```bash
# Raw f32le @ 16 kHz
cat audio.raw | ./build/bin/whisper-stream -m ./models/ggml-base.en.bin \
  --stdin --stdin-format f32le --debug-eos

# Raw s16le @ 16 kHz
cat audio_s16le.raw | ./build/bin/whisper-stream -m ./models/ggml-base.en.bin \
  --stdin --stdin-format s16le

# From a WAV/MP3 using ffmpeg to pipe f32le @ 16 kHz mono
ffmpeg -nostdin -i input.wav -f f32le -ac 1 -ar 16000 - 2>/dev/null \
  | ./build/bin/whisper-stream -m ./models/ggml-base.en.bin --stdin --stdin-format f32le
```

Tunable EOS flags (defaults in brackets):

- `--eos-on <float>`: RMS to enter speech [0.0085]
- `--eos-off <float>`: RMS to leave speech [0.0045]
- `--eos-hang-ms <int>`: keep speech after last loud frame [200]
- `--eos-preroll-ms <int>`: audio to prepend at speech start [200]
- `--eos-flush-zero-ms <int>`: flush if this many ms of zeros [1000]
- `--eos-min-chunk-ms <int>`: ignore blips shorter than this [600]
- `--debug-eos`: print EOS state transitions to stderr

### Loading EOS parameters from a config file

You can load tuned EOS parameters from a simple key=value config file using `--eos-config`.
This allows you to keep clean commands and still apply your tuned values. CLI flags always
override values loaded from the config file.

Example config (saved here as `eos_defaults.conf`, located in the repository root):

```
eos-on=0.0105
eos-off=0.0055
eos-preroll-ms=350
eos-hang-ms=350
eos-min-chunk-ms=800
```

Usage:

```bash
cat mic_test.raw | ./build/bin/whisper-stream \
  -m ./models/ggml-large-v3.bin \
  --stdin --stdin-format f32le \
  --eos-config ./eos_defaults.conf \
  --debug-eos
```

Note:
- Config is applied at parse time; any EOS flags specified after `--eos-config` will override.
- `eos_defaults.conf` in this repo is provided as a tuned example for a deterministic input.

## Piped Output (non‑TTY)

When stdout is not a TTY (for example, when piping output to another command or file),
`whisper-stream` automatically replaces in‑place progress updates (carriage return `\r`)
with newline `\n`. This preserves earlier output in logs and avoids lines being overwritten.
This change only affects console printing; internal buffers and debug logs are unchanged.

Example:

```bash
cat mic_test.raw | ./build/bin/whisper-stream \
  -m ./models/ggml-large-v3.bin \
  --stdin --stdin-format f32le \
  --eos-config ./eos_defaults.conf \
  | tee -a whisper_output.txt
```

## Streaming Buffering (mbuffer + sox)

For robust real‑time use, insert buffering between `ffmpeg` and `whisper-stream` to smooth bursts
and prevent underruns. The helper script `whisper.sh` demonstrates a practical setup using `sox`
and `mbuffer`.

- Requirements: `ffmpeg`, `sox`, and `mbuffer` must be installed.
  - Debian/Ubuntu: `sudo apt-get install mbuffer`
  - Fedora: `sudo dnf install mbuffer`
  - macOS (Homebrew): `brew install mbuffer`

- Environment knobs (override as needed when running `whisper.sh`):
  - `BUFFER_SEC` (default `1.0`): seconds of end padding via sox.
  - `MBUFFER_MEM` (default `2G`): target memory for mbuffer (e.g., `5G`).
  - `MBUFFER_BLOCK` (default `64k`): mbuffer block size.
  - `MONITOR=1`: insert `pv -rab` to monitor throughput (debug only).
  - `FFMPEG_DEBUG=1`: enable ffmpeg verbose logs and timestamp debug.

Examples

```bash
# Run with defaults (output required)
./whisper.sh -o whisper_output.txt

# Larger buffer and monitor throughput
MBUFFER_MEM=5G BUFFER_SEC=2.0 MONITOR=1 ./whisper.sh -o whisper_output.txt
```

## Building

The `whisper-stream` tool depends on SDL2 library to capture audio from the microphone. You can build it like this:

```bash
# Install SDL2
# On Debian based linux distributions:
sudo apt-get install libsdl2-dev

# On Fedora Linux:
sudo dnf install SDL2 SDL2-devel

# Install SDL2 on Mac OS
brew install sdl2

cmake -B build -DWHISPER_SDL2=ON
cmake --build build --config Release

./build/bin/whisper-stream
```

Note: SDL2 is only required for microphone capture. Stdin mode (`--stdin`) does not use SDL.

## Web version

This tool can also run in the browser: [examples/stream.wasm](/examples/stream.wasm)
