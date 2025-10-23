perfect — here’s your **checkpoint merge**: it preserves all the earlier structure and insights from `v69b_mbuffer-sync`, while fully integrating the latest multilingual + low-confidence work. only the **“Project State & Next Steps”** section was replaced and expanded to capture everything achieved and still pending as of `v71.2_auto-lang-flush`.

---

# 🎙️ Whisper-Stream Live Transcription Project — Updated Technical Checkpoint (`v71.2_auto-lang-flush`)

### 🧩 Core Goal

Create a **deterministic, reproducible live transcription pipeline** around `whisper.cpp`, capable of robust voice detection (EOS/VAD), adaptive noise handling, **dynamic multilingual awareness**, and **decoupled streaming throughput** — suitable for continuous dictation, meetings, or multi-hour live audio with no dropouts.

---

## 1️⃣ System Architecture Overview

### 🧠 Components

* **Audio Ingestion:** `ffmpeg` captures two PulseAudio sources:

    * `MIC_SRC` — live microphone input
    * `SPEAKER_SRC` — system monitor stream (optional mix)
* **Noise Control:**

    * RNNoise model (`std.rnnn`) for neural noise suppression
    * `highpass`, `lowpass`, and `compand` filters for analog shaping
* **Signal Mix:** `amix` merges mic + monitor (or mic-only variant)
* **SoX Stage (pad):**

    * Inserts small silence pad (`BUFFER_SEC`, default 1.0s) to smooth edges and absorb micro-jitter.
    * Also useful for future language-switch reinitialization.
* **Asynchronous Buffer (`mbuffer`):**

    * Provides a large in-RAM (or disk-backed) FIFO queue between producer (`ffmpeg`) and consumer (`whisper-stream`).
    * Decouples real-time audio from model throughput — **no underruns, no dropouts.**
* **Whisper Streamer:**

    * `./build/bin/whisper-stream` reads f32le PCM from stdin.
    * Uses `--eos-config` for finely tuned end-of-speech detection.
    * Now supports **low-confidence suppression** and **auto-language re-evaluation**.
* **Output Control:**

    * `tail -f output.txt` for live monitoring.
    * `mbuffer` stderr shows in/out rates and queue fullness for performance insight.

---

## 2️⃣ Deterministic Test Path

### 🧪 Process

1. **Record controlled input**
   `mic_test_long_gain_68.raw` — single-speaker narration with calibrated gain.
2. **Playback deterministically**
   Feed into pipeline via `ffmpeg -f f32le -ar 16000 -ac 1 -i mic_test_long_gain_68.raw`.
3. **Log full EOS state machine**
   `stderr` captures `[ERR] EOS:` transitions for quantitative comparison.
4. **Iterate configuration files**
   `v68b` → `v68h` (final EOS stable).
5. **Quantify performance** by metrics (transition count, flush timing, VOICE span).

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

## 3️⃣ Live Test Path — Continuous Stream Mode

After achieving EOS stability, the live pipeline was upgraded to **fully decoupled mode**:

### 🎧 Configuration (`v69b_mbuffer-sync`)

```bash
ffmpeg -hide_banner -nostats -loglevel warning \
  -f pulse -thread_queue_size 1024 -i "$SPEAKER_SRC" \
  -f pulse -thread_queue_size 1024 -i "$MIC_SRC" \
  -filter_complex "[1:a]aformat=sample_fmts=flt:channel_layouts=mono,aresample=16000, \
                      arnndn=m=$RNN:mix=0.90,highpass=f=80,lowpass=f=3400[mic]; \
                   [0:a]aformat=sample_fmts=flt:channel_layouts=mono,aresample=16000[speaker]; \
                   [mic][speaker]amix=inputs=2:duration=shortest:dropout_transition=2, \
                      highpass=f=120,lowpass=f=3500,compand=attacks=0.2:decays=0.4:points=-90/-900|-70/-70|-40/-20|0/0, \
                      volume=1.9,aresample=resampler=soxr:async=0:first_pts=0,asetpts=N/SR/TB" \
  -ac 1 -ar 16000 -f f32le - \
| sox -t f32 - -t f32 - -- pad 0 "${BUFFER_SEC:-1.0}" \
| mbuffer -m "${MBUFFER_MEM:-2G}" -s "${MBUFFER_BLOCK:-64k}" -o - \
| ./build/bin/whisper-stream "${WHISPER_FLAGS[@]}" -o output.txt
```

### ⚙️ EOS Parameters (`v68h.conf`)

```ini
eos-on=0.0170
eos-off=0.0023
eos-preroll-ms=350
eos-hang-ms=800
eos-min-chunk-ms=950
eos-flush-zero-ms=1000
```

### 💬 Live Results

* No more “you/thank you” repetition loops.
* No underruns, no dropouts — steady processing regardless of speech duration.
* EOS remains stable with clear VOICE/HANG/FLUSH boundaries.
* GPU stays active consistently; throughput stable around 64 kB/s.
* Tested against continuous YouTube speech streams (multi-minute monologues).

---

## 4️⃣ Tuning Philosophy

* **Determinism first:** all thresholds tuned offline against deterministic input.
* **Isolation:** change one variable per iteration (EOS, filters, buffer, etc.).
* **Reproducibility:** every config version committed with tag (`v68b`→`v69b`).
* **Separation of concerns:** ffmpeg = capture, sox = signal conditioning, mbuffer = queue, whisper = inference.
* **Resilience:** model can lag safely; never loses data.
* **Adaptivity (new):** the system now learns language context dynamically and filters low-confidence noise without retriggering EOS.

---

## 5️⃣ Project State & Next Steps

✅ **Stable baseline:**
`v71.2_auto-lang-flush` — continuous multilingual transcription with EOS-stable core and asynchronous buffering.

### ⚙️ Core Enhancements Achieved

* **Low-Confidence Filtering:** suppresses false tokens and ambient “thank-you” ghosts.
* **Auto-Language Re-Evaluation:** detects spoken language on-the-fly via short rolling windows.
* **Clean Language Handover:** clears prompt tokens and avoids cross-language interference.
* **Immediate Flush on Switch (stdin path):** prevents dead air after language change.
* **CPU efficiency:** idle `ffmpeg` load reduced dramatically.

---

### 🚧 Active Work (`v71.3` in progress)

* **Decode Language Binding:** ensure every `whisper_full()` uses current `wparams.language`.
* **Force-Flush Verification:** confirm immediate decode after switch produces text.
* **Diagnostic Expansion:** add logs for

    * `[decode] using lang=...`
    * `[lowconf] pavg=... text='...'`
    * Top-3 detected language probabilities.
* **Window Validation:** verify tail-window slicing (`--auto-lang-window-sec`) uses the latest PCM tail.

---

### 🌍 Phase 2 — Fast Dynamic Language Adaptation

Next-generation bilingual handling:

* **Adaptive Hysteresis:** allow rapid but stable switching (<1 s protection from flip-flop).
* **Confidence-Weighted Replay:** re-decode 1–2 s of buffered PCM after switch to recover missed words.
* **Fallback Intelligence:** `--auto-lang-fallback` used only before first detection, not between switches.
* **Real-time Transparency:** live display of top-probability languages with per-chunk updates.

---

### 🧭 Essence

The pipeline has evolved from a timing-sensitive prototype into a **context-adaptive transcription system**.
`mbuffer` ensures real-time stability, SoX and RNNoise maintain audio clarity, and Whisper now begins to *listen with a thousand ears* — adapting to language context and confidence while preserving determinism and resilience.

---

✅ **Current Baseline:** `v71.2_auto-lang-flush`
🧠 **Core Principles:** determinism · decoupling · adaptivity · resilience · clarity

---
