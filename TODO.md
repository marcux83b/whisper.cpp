perfect — let’s lock this in as an **internal change to `whisper-stream`** so Codecs can implement it without needing ffmpeg in the sandbox. Below is a tight, self-contained task list + drop-in C++ snippets. ffmpeg is mentioned only as *context* for how we’ll feed audio in production.

---

# Goal

Make `whisper-stream` finalize utterances **on end-of-speech**, not only on the fixed `--step` timer, and avoid clipping starts/ends. After this change, you do **not** need any Python helper or exotic ffmpeg gating; the tool will:

* detect start/end of speech from incoming **f32le mono 16 kHz** frames,
* **prepend a small pre-roll** at speech start (no “Mar**k**o”),
* **flush immediately** on confirmed silence/zeros (no “wait for the next sentence”),
* ignore tiny blips.

---

# High-level tasks for Codecs

1. **Add EOS config & flags** (with sane defaults)
2. **Frame the stdin stream at 20 ms (320 samples)** and compute per-frame RMS + zero-run
3. **State machine**: `{IDLE, VOICE, HANG, ARMED}` with hysteresis & hang
4. **Pre-roll ring buffer**, append on VOICE entry
5. **Flush** the chunk on EOS (zeros or RMS-silence), respecting a **min duration**
6. Keep existing long-utterance slicing (`--length/--keep`) as is
7. **Debug logging** behind a flag

---

## 1) Flags & config (defaults “just work”)

Add to your CLI parser (names prefixed `--eos-*`):

```txt
--eos-on <float>            (default 0.0085)   // RMS to enter speech  (~ -44 dBFS)
--eos-off <float>           (default 0.0045)   // RMS to leave speech  (~ -49 dBFS)
--eos-hang-ms <int>         (default 200)      // keep 'speech' this long after last loud frame
--eos-preroll-ms <int>      (default 200)      // prepend this much audio at speech start
--eos-flush-zero-ms <int>   (default 1000)     // flush if this many ms of zeros seen
--eos-min-chunk-ms <int>    (default 600)      // ignore blips shorter than this
--debug-eos                 (optional)         // log EOS decisions
```

C++ config:

```cpp
struct eos_params {
    float on  = 0.0085f;
    float off = 0.0045f;
    int   hang_ms = 200;
    int   preroll_ms = 200;
    int   flush_zero_ms = 1000;
    int   min_chunk_ms  = 600;
    bool  debug = false;
};
```

> Keep existing `--step/--length/--keep`. With EOS, short lines won’t wait for `--step`, but long speech still gets mid-decodes by `--length`.

---

## 2) Framing stdin + utilities

Assume stdin is **f32le mono @ 16 kHz** (you already enforce this with `--stdin-format f32le`). We’ll process **20 ms frames**.

```cpp
constexpr int SR = 16000;
constexpr int FRAME_SAMPLES = SR * 20 / 1000; // 320

inline float frame_rms(const float* p, int n) {
    double acc = 0.0;
    for (int i = 0; i < n; ++i) { double x = p[i]; acc += x*x; }
    return (float) std::sqrt(acc / n);
}

inline bool is_zero_frame(const float* p, int n) {
    for (int i = 0; i < n; ++i) if (std::fabs(p[i]) > 1e-7f) return false;
    return true;
}

inline int ms_to_frames(int ms) { return std::max(1, ms / 20); }
```

---

## 3) Pre-roll ring buffer (last ~200 ms)

Use any simple ring buffer; here’s a minimal one:

```cpp
struct float_ring {
    std::vector<float> buf; size_t w = 0; bool full = false;
    explicit float_ring(size_t cap) : buf(cap) {}
    void push(const float* p, size_t n) {
        for (size_t i=0;i<n;++i) { buf[w] = p[i]; w=(w+1)%buf.size(); if (w==0) full=true; }
    }
    void dump_to(std::vector<float>& out) const {
        if (!full && w==0) return;
        if (!full) { out.insert(out.end(), buf.begin(), buf.begin()+w); }
        else {
            out.insert(out.end(), buf.begin()+w, buf.end());
            out.insert(out.end(), buf.begin(), buf.begin()+w);
        }
    }
    void clear() { w=0; full=false; }
};
```

---

## 4) EOS state machine

```cpp
enum class EOSState { IDLE, VOICE, HANG, ARMED };

struct eos_state {
    EOSState st = EOSState::IDLE;
    int hang = 0;
    int zero_run = 0;
    int frames_in_chunk = 0;
};
```

Main loop (inside your stdin reader):

```cpp
void process_stream_with_eos(const eos_params& P,
                             std::function<bool(float*,int)> read_frame, // returns false on EOF
                             std::function<void(const std::vector<float>&)> decode_chunk,
                             std::function<void(const char*)> dbg) {

    float_ring preroll(ms_to_frames(P.preroll_ms) * FRAME_SAMPLES);
    std::vector<float> chunk; chunk.reserve(8*FRAME_SAMPLES);

    eos_state S;

    std::array<float, FRAME_SAMPLES> fr;

    const int HANG_FR   = ms_to_frames(P.hang_ms);
    const int FLUSHZ_FR = ms_to_frames(P.flush_zero_ms);
    const int MIN_FR    = ms_to_frames(P.min_chunk_ms);

    auto log = [&](const char* m){ if (P.debug) dbg(m); };

    while (read_frame(fr.data(), FRAME_SAMPLES)) {
        // metrics
        const bool zero = is_zero_frame(fr.data(), FRAME_SAMPLES);
        S.zero_run = zero ? (S.zero_run + 1) : 0;
        const float rms = frame_rms(fr.data(), FRAME_SAMPLES);
        const bool on  = (rms >= P.on);
        const bool off = (rms <  P.off);

        // preroll always records
        preroll.push(fr.data(), FRAME_SAMPLES);

        switch (S.st) {
            case EOSState::IDLE:
                if (on && !zero) {
                    // start: prepend preroll so starts aren't clipped
                    preroll.dump_to(chunk);
                    chunk.insert(chunk.end(), fr.begin(), fr.end());
                    S.frames_in_chunk = (int)(chunk.size()/FRAME_SAMPLES);
                    S.hang = HANG_FR;
                    S.st = EOSState::VOICE;
                    log("EOS: IDLE->VOICE");
                }
                break;

            case EOSState::VOICE:
                chunk.insert(chunk.end(), fr.begin(), fr.end());
                ++S.frames_in_chunk;
                if (on && !zero) {
                    S.hang = HANG_FR; // keep extending
                } else {
                    S.st = EOSState::HANG; log("EOS: VOICE->HANG");
                }
                break;

            case EOSState::HANG:
                chunk.insert(chunk.end(), fr.begin(), fr.end());
                ++S.frames_in_chunk;
                if (on && !zero) {
                    S.st = EOSState::VOICE; S.hang = HANG_FR; log("EOS: HANG->VOICE");
                } else {
                    if (--S.hang <= 0) { S.st = EOSState::ARMED; log("EOS: HANG->ARMED"); }
                }
                break;

            case EOSState::ARMED:
                // Wait for either a run of zeros OR clear silence (off)
                if (S.zero_run >= FLUSHZ_FR || off) {
                    if (S.frames_in_chunk >= MIN_FR) {
                        log("EOS: FLUSH");
                        decode_chunk(chunk); // <-- call your existing decode path
                    } else {
                        log("EOS: DROP tiny");
                    }
                    chunk.clear(); S.frames_in_chunk = 0;
                    preroll.clear();
                    S.st = EOSState::IDLE;
                    // remain idle until next on
                } else if (on && !zero) {
                    S.st = EOSState::VOICE; S.hang = HANG_FR;
                    chunk.insert(chunk.end(), fr.begin(), fr.end());
                    ++S.frames_in_chunk;
                    log("EOS: ARMED->VOICE");
                }
                break;
        }

        // Optional: mid-slice long utterances still allowed by your existing logic:
        // if (S.st != IDLE && S.frames_in_chunk * 20 >= user_length_ms) { ... mid-decode, keep last --keep ms ... }
    }
}
```

**Hook points you already have:**

* `read_frame` → your stdin reader (`fread` on `stdin`) that fills `fr`.
* `decode_chunk` → wraps your current “run Whisper on a PCM buffer” path (the same path used when `--step` fires; now you call it on EOS too).
* `dbg` → `fprintf(stderr, ...)` guarded by `--debug-eos`.

---

## 5) Where to integrate

In your current `main`:

* After model init, set up `eos_params` from CLI.
* Replace the old “fixed timer decode” loop with `process_stream_with_eos(...)`.
* Keep your **existing** mid-chunk slicing: if the speaker never pauses (podcast, etc.), still do periodic decodes based on `--length/--keep`. That’s orthogonal to EOS and continues to work.

---

## 6) Defaults & behavior

With defaults:

* Any short sentence that ends in silence/zeros **flushes immediately**.
* Starts aren’t clipped (pre-roll).
* Tiny chirps are ignored (min-chunk, hysteresis).
* Long speech still gets time-sliced.

---

## 7) Debugging / validation (no ffmpeg required in sandbox)

Add `--debug-eos` and run unit-ish tests by feeding recorded f32le files from stdin:

```
cat sample_f32le_16k_mono.raw | ./whisper-stream ... --stdin --stdin-format f32le --debug-eos
```

(You can create `sample_f32le_16k_mono.raw` offline from a WAV in dev env; Codecs won’t need ffmpeg in sandbox to run the logic.)

---

## 8) (Context only) Production feed line

When you test on your machine, you’ll feed the mic via ffmpeg with RNNoise; **no helper scripts needed**:

```bash
ffmpeg -hide_banner -nostats -loglevel warning \
  -f pulse -fflags nobuffer -thread_queue_size 64 \
  -i alsa_input.pci-...Mic1__source \
  -filter_complex "aformat=sample_fmts=flt:channel_layouts=mono,aresample=16000,arnndn=m=$HOME/rnnoise-models/cb.rnnn:mix=0.95,highpass=f=80,lowpass=f=3600" \
  -ac 1 -ar 16000 -f f32le - \
| ./build/bin/whisper-stream \
     -m models/ggml-large-v3.bin \
     --stdin --stdin-format f32le \
     --step 800 --length 3200 --keep 0 \
     --eos-on 0.0085 --eos-off 0.0045 --eos-hang-ms 200 \
     --eos-preroll-ms 200 --eos-flush-zero-ms 1000 --eos-min-chunk-ms 600
```

(Again: Codecs doesn’t run this; it’s just to show how we’ll use the new EOS behavior.)

---

## 9) README delta

* Add a section “End-of-Speech (EOS) decoding” explaining why short utterances no longer wait for `--step`.
* Document the `--eos-*` flags and defaults.
* Note that external VAD/pad helpers are no longer required.

---

## 10) Acceptance check (your field test)

Use your two-line probe:

> “Hello, my name is Marco.” *(pause ~1.2 s)*
> “This is a test.” *(pause)*

**Expected now:**

* “Marco” appears **once** (no “Mark/come”), then GPU drops quickly.
* “This is a test.” appears **without** needing a third utterance.
* Birds don’t re-arm it unless genuinely loud/continuous.

---

