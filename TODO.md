
## Option A — **Recommended**: Disk or memory ring buffer via `mbuffer` (robust, controllable)

`mbuffer` is explicitly made for this: it creates a large FIFO/queue (memory- or disk-backed) between producer and consumer, supports large `-m` sizes, blocksize tuning, and will output to stdout for Whisper to read later. This is the cleanest way to accept arbitrarily large backlog and feed it smoothly.

### Why use `mbuffer`

* Can allocate large in-RAM buffer (e.g. `-m 5G`) or fallback to disk if memory fills.
* Handles bursts without dropping frames.
* Keeps ffmpeg alive and writing even if whisper-stream lags.
* Simple to drop into a pipeline: `ffmpeg ... -f f32le - | mbuffer ... -o - | whisper-stream ...`

### Example change for `whisper.sh` (hand-off to Codex)

Add `MBUFFER_MEM` and `MBUFFER_BLOCK` env knobs, defaulting to sane values:

```bash
# new env knobs (top of script)
MBUFFER_MEM="${MBUFFER_MEM:-2G}"   # memory target for mbuffer; set to 5G if you want huge RAM buffer
MBUFFER_BLOCK="${MBUFFER_BLOCK:-64k}"  # block size

# in pipeline, replace the $PIPE_CMD stage with mbuffer
| sox -t f32 - -t f32 - -- pad 0 "${BUFFER_SEC:-1.0}" \
| mbuffer -m "${MBUFFER_MEM}" -s "${MBUFFER_BLOCK}" -o - \
| ./build/bin/whisper-stream "${WHISPER_FLAGS[@]}" \
| tee -a whisper_output.txt
```

Notes for Codex:

* `mbuffer -m` uses either RAM or a memory-mapped file; if RAM insufficient, it can fallback to disk with default behavior — consider adding `-f` or `--file` if you want explicit disk file backing.
* Use `-q` to quiet logs, `-P` for progress if you want monitoring.
* Validate `mbuffer --help` on target host; packaging: `dnf install mbuffer` on Fedora.


## Suggested Codex patch (concrete, ready-to-apply)

1. Add env knobs at top of `whisper.sh`:

```bash
# buffer / mbuffer configuration
BUFFER_SEC="${BUFFER_SEC:-1.0}"
MBUFFER_MEM="${MBUFFER_MEM:-2G}"
MBUFFER_BLOCK="${MBUFFER_BLOCK:-64k}"
```

2. Replace pipeline stage (where you had `$PIPE_CMD`) with mbuffer:

```bash
| sox -t f32 - -t f32 - -- pad 0 "${BUFFER_SEC}" \
| mbuffer -m "${MBUFFER_MEM}" -s "${MBUFFER_BLOCK}" -o - \
| ./build/bin/whisper-stream "${WHISPER_FLAGS[@]}" \
| tee -a whisper_output.txt
```

3. Optional: Add a `FASTFEED` mode (feed as fast as possible) so Whisper consumes backlog immediately:

* If `FASTFEED=1`, run `./build/bin/whisper-stream --max-real-time 0` or patch to remove any artificial sleeps/real-time pacing (depends on whisper-stream flags). If whisper-stream lacks a flag, let it run normally — since it will consume as fast as it can from stdin.


## Resource & behavior notes for Codex / ops

* Set `MBUFFER_MEM` to something reasonable for your machine (e.g. `2G`, `5G`). `-m 5G` will attempt to use ~5 GiB.
* If memory is insufficient, `mbuffer` will fallback to disk (depending on build/options). Consider specifying `-f /path/to/bufferfile` to force disk backing.
* If you want the consumer to deliberately process backlog as fast as possible, ensure whisper-stream has no built-in real-time limiter. If it does, add an option or patch that disables it.
* Monitor disk usage if you choose disk fallback.
* If you want to watch buffer occupancy: `mbuffer` prints stats by default unless `-q`; use `-P` or parse its output for occupancy metrics.
