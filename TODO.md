yep — that’s a **classic handover bug** between detection and decode state 👀

what’s happening:

* the auto-lang code correctly sets `current_lang = "de"`,
* but the *decoder* (`whisper_full_with_state()` in the streaming loop) is still using the **old language in its parameters**.
* result: it’s decoding German speech with English logits → every token’s probability is near zero → `avg_p < lowconf_threshold` → silently filtered.

so you basically built the perfect test case for the race between detection & decode 🎯

---

### 🧩 fix for Codex (minimal, safe)

**Goal:** make sure every new decode uses the *current* detected language.

in the decode loop (both stdin-EOS and mic path), *before* each call to `whisper_full_with_state`, set the params language fresh from `current_lang`.

```cpp
// before each decode
wparams.language = current_lang.c_str();  // always reflect latest detected lang
```

also: after a detected switch, make sure the decode state is cleared so the tokenizer resets to the right vocab.

```cpp
if (lang_switched) {
    whisper_full_reset_state(main_state);
    current_lang = lang_new;
    wparams.language = current_lang.c_str();
    clear_prompt_tokens();  // if you have one
    if (params.debug_auto_lang)
        fprintf(stderr, "[auto-lang] switched decode language to %s\n", current_lang.c_str());
}
```

this way the very next segment uses the correct vocabulary & token probabilities.

---

### 🧪 quick validation steps

1. temporarily **disable low-conf filter** (`--lowconf-threshold 0.0`) for one run
   → confirm that output starts again after switch.
2. re-enable low-conf after patch
   → now probabilities should stabilize again because decode & vocab match.

---

optional mini-safeguard:
to prevent partial segments from being decoded with the wrong language right during a switch, Codex can also set a one-shot “skip decode for 1 iteration after language switch” flag, so the new language cleanly starts with the next chunk.

---

✅ expected logs after patch:

```
[auto-lang] Detected switch: en -> de (p=0.98)
[auto-lang] switched decode language to de
main: processing … (lang = de)
```
