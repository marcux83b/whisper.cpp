
## ✅ TASKLIST: Low-Confidence Suppression for `whisper-stream`

### 🎯 Goal

Add a configurable **low-confidence filter** to `examples/stream/stream.cpp`.
If the average probability (`res.p_prob`) of a recognized segment is below the threshold, skip emitting text (or mark as `[GARBLED]` in debug).

---

### 📂 Files

* `examples/stream/stream.cpp`
* `examples/stream/whisper_stream_params.h` (or equivalent header where CLI params are stored)

---

### 🧩 Tasks

#### 1. Add new CLI parameter

```cpp
// in params struct
float low_conf_threshold = 0.0f;  // default off (no filtering)

// in CLI parser
else if (arg == "--lowconf-threshold" && i + 1 < argc) {
    params.low_conf_threshold = std::stof(argv[++i]);
}
```

#### 2. Apply filter before printing output

Locate the block that prints recognized text, e.g.:

```cpp
printf("%s", res.text.c_str());
```

Insert above it:

```cpp
// --- Low-confidence suppression ---
if (params.low_conf_threshold > 0.0f && res.p_prob < params.low_conf_threshold) {
    fprintf(stderr,
        "[debug] low conf %.2f -> skipped: '%s'\n",
        res.p_prob, res.text.c_str());
    continue; // skip low-confidence chunk
}
// --- End suppression ---
```

#### 3. Optional: debug placeholder

If you want visible placeholders for analysis (not for final output):

```cpp
if (params.low_conf_threshold > 0.0f && res.p_prob < params.low_conf_threshold) {
    printf("[GARBLED]\n");
    continue;
}
```

#### 4. Document in `README.md`

Add one line under **Advanced Parameters**:

> `--lowconf-threshold <float>` — suppress output segments with average confidence below this value (e.g., `0.35`).

---

### 🧪 Test Plan

1. Run `whisper-stream` on a known noisy clip with and without `--lowconf-threshold 0.35`.
2. Verify that low-confidence “thank you” / “you” artifacts are skipped.
3. Observe `[debug] low conf` messages in stderr.
4. Ensure genuine speech (p_prob > threshold) is unaffected.

---

### 🧭 Example Run

```bash
./build/bin/whisper-stream \
  --eos-config eos_defaults.conf \
  --lowconf-threshold 0.35 \
  -m models/ggml-large-v3.bin \
  -f /dev/stdin
```

---

### 🏁 Success Criteria

* No spurious “thank you” or “you” outputs during silence/noise.
* Configurable suppression threshold.
* Backward compatible: if `--lowconf-threshold` is unset or `0`, all output passes through unchanged.

---

would you like me to include a short code-context diff snippet (showing where to insert in the current whisper-stream main loop)?
