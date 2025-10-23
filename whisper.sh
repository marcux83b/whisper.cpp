#!/usr/bin/env bash
#
# whisper.sh — streaming helper
#
# Requirements:
# - ffmpeg, sox, mbuffer
#   Install mbuffer:
#     - Debian/Ubuntu:   sudo apt-get install mbuffer
#     - Fedora:          sudo dnf install mbuffer
#     - macOS (Homebrew): brew install mbuffer
#
# Useful env toggles:
# - DEBUG_EOS=true     # enable EOS debug logs in whisper-stream
# - MONITOR=1          # insert pv to monitor throughput (debug only)
# - FFMPEG_DEBUG=1     # ffmpeg verbose + timestamp debug
# - BUFFER_SEC=1.0     # seconds of end padding via sox (default 1.0)
# - MBUFFER_MEM=2G     # mbuffer memory target (e.g., 2G, 5G)
# - MBUFFER_BLOCK=64k  # mbuffer block size
#
## Usage
#  ./whisper.sh -o ./whisper_output.txt
#  (the -o/--output argument is mandatory; stdout is reserved for diagnostics)
set -euo pipefail

# === CONFIG ===
export VK_ICD_FILENAMES="/usr/share/vulkan/icd.d/intel_icd.x86_64.json"
export DRI_PRIME=0
export GGML_VULKAN_DEBUG=0

MIC_SRC="alsa_input.pci-0000_00_1f.3-platform-skl_hda_dsp_generic.HiFi__Mic1__source"
SPEAKER_SRC="alsa_output.pci-0000_00_1f.3-platform-skl_hda_dsp_generic.HiFi__Speaker__sink.monitor"
MODEL="models/ggml-large-v3.bin"
EOS_CONF="./eos_defaults.conf"
RNN="$HOME/rnnoise-models/std.rnnn"   # try std.rnnn if cb feels too aggressive

# env toggle: DEBUG_EOS=true ./whisper.sh
DEBUG_EOS="${DEBUG_EOS:-false}"

# buffer / mbuffer configuration
BUFFER_SEC="${BUFFER_SEC:-1.0}"
MBUFFER_MEM="${MBUFFER_MEM:-2G}"
MBUFFER_BLOCK="${MBUFFER_BLOCK:-64k}"

# === REQUIRED ARGS ===
OUT_FILE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -o|--output)
      OUT_FILE="${2:-}"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 -o OUTPUT_FILE" >&2
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: $0 -o OUTPUT_FILE" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$OUT_FILE" ]]; then
  echo "error: -o OUTPUT_FILE is required" >&2
  echo "Usage: $0 -o OUTPUT_FILE" >&2
  exit 1
fi

# === BUILD ARG LIST ===
WHISPER_FLAGS=(
  -m "$MODEL"
  --stdin --stdin-format f32le
  --eos-config "$EOS_CONF"
  --step 1000 --length 4000 --keep 0
  --lowconf-threshold 0.35
  --language auto
  --auto-lang-reeval 3
  --auto-lang-window-sec 4
  --auto-lang-threshold 0.75
  --auto-lang-fallback en
  --debug-auto-lang
)

if [[ "$DEBUG_EOS" == "true" ]]; then
  WHISPER_FLAGS+=(--debug-eos)
fi

# === PIPELINE (RNNoise tuned mix & sync buffers) ===

# Optional diagnostics:
# - Set MONITOR=1 to insert pv and observe throughput (debug only)
# - Set FFMPEG_DEBUG=1 to enable verbose ffmpeg logs with timestamp debug
MONITOR="${MONITOR:-0}"
FFMPEG_DEBUG="${FFMPEG_DEBUG:-0}"

if [[ "$MONITOR" = "1" ]]; then
  PIPE_CMD="pv -rab"
else
  PIPE_CMD="cat"
fi

if [[ "$FFMPEG_DEBUG" = "1" ]]; then
  FFDBG_OPTS=( -loglevel verbose -debug_ts )
else
  FFDBG_OPTS=( -loglevel warning )
fi

ffmpeg -hide_banner -nostats "${FFDBG_OPTS[@]}" \
  -re -use_wallclock_as_timestamps 1 \
  -f pulse -thread_queue_size 256 -i "$SPEAKER_SRC" \
  -f pulse -thread_queue_size 256 -i "$MIC_SRC" \
  -filter_complex "[1:a]aformat=sample_fmts=flt:channel_layouts=mono,aresample=16000, \
                      arnndn=m=$RNN:mix=0.90,highpass=f=80,lowpass=f=3400[mic]; \
                   [0:a]aformat=sample_fmts=flt:channel_layouts=mono,aresample=16000[speaker]; \
                   [mic][speaker]amix=inputs=2:duration=shortest:dropout_transition=2, \
                      highpass=f=120,lowpass=f=3500,compand=attacks=0.2:decays=0.4:points=-90/-900|-70/-70|-40/-20|0/0, \
                       volume=1.9,aresample=resampler=soxr:async=0:first_pts=0,asetpts=N/SR/TB" \
  -ac 1 -ar 16000 -f f32le - \
| sox -t f32 - -t f32 - -- pad 0 "${BUFFER_SEC}" \
| mbuffer -m "${MBUFFER_MEM}" -s "${MBUFFER_BLOCK}" -o - \
| ./build/bin/whisper-stream "${WHISPER_FLAGS[@]}" >> "$OUT_FILE"
