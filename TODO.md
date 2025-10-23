## 🧠 Tasklist — Auto Language Re-Evaluation in `whisper-stream`

### 🎯 Goal

Enable **on-the-fly language detection and switching** in `whisper-stream`, without restarting the process or reloading the model.
The system should periodically re-evaluate the input language from recent audio, switch when confidence is high, and use a safe fallback language before the first detection.

---

### ⚙️ CLI Additions

Add these new parameters to `examples/stream/stream.cpp`:

```bash
--auto-lang-reeval <seconds>     # periodically re-evaluate language (0 = off)
--auto-lang-threshold <float>    # minimum confidence to trigger switch (default 0.75)
--auto-lang-fallback <string>    # fallback language before first detection (default "en")
```

Default values:

```cpp
float auto_lang_reeval   = 0.0f;    // disabled
float auto_lang_threshold = 0.75f;
std::string auto_lang_fallback = "en";
```

Parse these options like existing flags and store in `whisper_params`.

---

### 🧩 Core Logic

#### 1. **Initialization**

* If `--language auto` is active, set:

  ```cpp
  current_lang = auto_lang_fallback;
  bool has_detected_lang = false;
  ```
* Whisper runs in fallback language until a confident detection occurs.

#### 2. **Re-evaluation scheduler**

* Maintain a rolling timer or frame counter:

    * When `auto_lang_reeval > 0`, check elapsed time since last re-eval.
    * Every N seconds (e.g., 5s), trigger detection:

      ```cpp
      float prob_new = 0.0f;
      int lang_new_id = whisper_lang_auto_detect(ctx, recent_pcm.data(),
                                                 recent_pcm.size(), &prob_new);
      const char *lang_new = whisper_lang_str(lang_new_id);
      ```

#### 3. **Decision logic**

```cpp
if (prob_new >= auto_lang_threshold && lang_new != current_lang) {
    fprintf(stderr,
      "[auto-lang] Detected switch: %s → %s (p=%.2f)\n",
      current_lang.c_str(), lang_new, prob_new);

    current_lang = lang_new;
    has_detected_lang = true;

    whisper_full_reset_state(ctx);  // reset without unloading
    // optionally reinitialize params.language = current_lang
}
```

If `prob_new < threshold`:

* Keep using `current_lang`.
* Do not reset or reinitialize.

If detection fails repeatedly and `has_detected_lang == false`:

* Continue using `auto_lang_fallback` until success.

#### 4. **Buffered audio replay**

* Maintain a small circular PCM buffer (5–10s).
* When a language switch occurs:

    * Flush decoder queue,
    * Replay buffer into Whisper with new language:

      ```cpp
      whisper_full(ctx, params, buffer_audio, num_samples);
      ```

This allows context continuity across switches.

---

### 🧠 Data Structures

* `std::vector<float> recent_pcm` — ring buffer holding the last few seconds of audio.
* `std::chrono::steady_clock::time_point last_lang_eval` — tracks when last detection occurred.
* `std::string current_lang` — active decoding language.
* `bool has_detected_lang` — tracks whether an initial detection succeeded.

---

### 🪶 Logging

Use stderr debug output (like existing EOS logs):

```
[auto-lang] re-eval triggered after 5.0s
[auto-lang] Detected switch: en → de (p=0.93)
[auto-lang] Keeping en (p=0.42 below threshold)
```

---

### 🧩 Integration Notes

* Place re-eval logic near where you handle chunk processing in the main decode loop.
* Avoid blocking decode — run detection inline but non-destructively on the recent PCM copy.
* Ensure GPU model context (`ctx`) stays active; don’t unload or reload the model.
* Maintain thread safety — detection and decode must not share mutable buffers simultaneously.
* Optionally add `--debug-auto-lang` flag to force verbose mode.

---

### ✅ Expected Behavior

| Scenario                       | Behavior                                                       |
| ------------------------------ | -------------------------------------------------------------- |
| Startup with `--language auto` | Uses fallback language until first confident detection         |
| Confidence ≥ threshold         | Switch to new language, reset state, replay buffer             |
| Confidence < threshold         | Keep current language                                          |
| Silence or mixed speech        | No frequent switching (cooldown = `auto-lang-reeval` interval) |
| No audio yet                   | Decode with fallback until detection available                 |

---

### 🔬 Test Plan

1. Feed alternating English/German samples into stdin.
2. Run with:

   ```
   ./whisper-stream --stdin --language auto \
       --auto-lang-reeval 5 --auto-lang-threshold 0.75 --auto-lang-fallback en
   ```
3. Observe:

    * Smooth transitions (no restart)
    * Debug logs showing re-eval intervals
    * Correct detection when language changes
    * Minor accuracy dip (~1s) around switches, then recovery

---
