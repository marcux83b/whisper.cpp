# 🎙️ Whisper-Stream Live Transcription Project — Primer & Technical Summary

### 🧩 Core Goal

Create a **deterministic, reproducible live transcription pipeline** around `whisper.cpp`, capable of robust voice detection (EOS/VAD) and adaptive noise handling — suitable for continuous dictation, meetings, or conversation monitoring in real time.

---

## 1️⃣ System Architecture Overview

### 🧠 Components

* **Audio Ingestion:** `ffmpeg` captures two PulseAudio sources:

    * `MIC_SRC` — the live microphone input
    * `SPEAKER_SRC` — the system monitor stream (optional mix)
* **Noise Control:**

    * RNNoise model (`std.rnnn`) for neural noise suppression
    * `highpass`, `lowpass`, `compand` filters for analog shaping
* **Signal Mix:** `amix` merges mic + monitor (or mic-only variant)
* **Whisper Streamer:**

    * `./build/bin/whisper-stream` reads f32le PCM from stdin
    * Uses `--eos-config` for finely tuned end-of-speech detection
    * Optional `--debug-eos` emits detailed state transitions
* **Output Control:** `tee` appends transcribed text to log files
* **Mux Logger:** Bash wrapper merges stdout (`[OUT]`) + stderr (`[ERR]`) into a single chronological trace file for debugging and EOS state analysis.

---

## 2️⃣ Deterministic Test Path

To make tuning reproducible and quantifiable:

### 🧪 Process

1. **Record controlled input**

    * `mic_test_long_gain_68.raw` — a single-speaker narration with calibrated gain.
2. **Play back deterministically**

    * Feed into pipeline via `ffmpeg -f f32le -ar 16000 -ac 1 -i mic_test_long_gain_68.raw`.
3. **Log full EOS state machine**

    * `stdout` = recognized text
    * `stderr` = `[ERR] EOS: ...` transitions
    * Both merged into `.mux` files for analysis.
4. **Iterate configuration files**

    * Successive versions: `v68b`, `v68c`, `v68c_rnnoise`, `v68d_rnnoise`, `v68h`
    * Each change isolated, compared by metrics (flush count, span length, false triggers).

### 📊 Outcome Metrics

| Metric                   | Ideal Trend                      |
| ------------------------ | -------------------------------- |
| Total transitions        | ↓ fewer flips between HANG/VOICE |
| Flush events             | ↓ stable chunk boundaries        |
| DROP tiny                | ✅ eliminated                     |
| Avg VOICE span           | ↑ longer continuous speech       |
| Early FLUSH              | ✅ merged                         |
| End capture completeness | ✅ full                           |

---

## 3️⃣ Live Test Path

After deterministic stability was proven, live tests were run with the same EOS config and RNNoise active:

### 🎧 Configuration (final form)

```bash
ffmpeg -hide_banner -nostats -loglevel warning \
  -f pulse -thread_queue_size 1024 -i "$SPEAKER_SRC" \
  -f pulse -thread_queue_size 1024 -i "$MIC_SRC" \
  -filter_complex "[1:a]aformat=sample_fmts=flt:channel_layouts=mono,aresample=16000, \
                      arnndn=m=$RNN:mix=0.90,highpass=f=80,lowpass=f=3400[mic]; \
                   [0:a]aformat=sample_fmts=flt:channel_layouts=mono,aresample=16000[speaker]; \
                   [mic][speaker]amix=inputs=2:duration=shortest:dropout_transition=2, \
                      highpass=f=120,lowpass=f=3500,compand=attacks=0.2:decays=0.4:points=-90/-900|-70/-70|-40/-20|0/0, \
                      volume=1.9,aresample=resampler=soxr:async=1:first_pts=0,asetpts=N/SR/TB" \
  -ac 1 -ar 16000 -f f32le - \
| ./build/bin/whisper-stream "${WHISPER_FLAGS[@]}" \
| tee -a whisper_output.txt
```

### ⚙️ Final Parameters (`v68h.conf`)

```ini
eos-on=0.0170
eos-off=0.0023
eos-preroll-ms=350
eos-hang-ms=800
eos-min-chunk-ms=950
eos-flush-zero-ms=1000
```

### 💬 Live Results

* No false “thank you” or phantom activations.
* Stable EOS hysteresis: single clean VOICE/HANG/FLUSH per phrase.
* Pauses & resumes handled naturally.
* End-of-stream tail merged correctly.

---

## 4️⃣ Tuning Philosophy

### 🧠 Key Concepts

* **Determinism first:** all tuning against a known input before live chaos.
* **Isolate variables:** only change one dimension (EOS, RNNoise mix, filters) per test.
* **State-aware logs:** `[ERR] EOS:` traces are gold — reveal every decision.
* **Metric tracking:** count transitions, flushes, and VOICE spans to quantify progress.
* **Reproducible labeling:** commit each EOS config with version tags in git for auditability.

---

## 5️⃣ Project State & Next Steps

✅ **Baseline achieved:**
`v68h` — stable, low-false-positive, live-ready EOS config.

🚀 **Next phases:**

* Field-test in meetings and Discord (evaluate latency & multi-speaker behavior)
* Optionally benchmark smaller Whisper models (medium/small) for speed vs accuracy
* Add optional per-chunk timestamping for real-time UI or post-processing alignment
* Eventually wrap in a thin Python or Rust layer for streaming to an app interface

---

### 🧭 Essence

This project transformed `whisper.cpp` from a passive model into a **real-time, speech-aware transcription instrument** —
data-driven, empirically tuned, and reproducible down to each dB threshold.

---
