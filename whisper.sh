#!/usr/bin/env bash
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

# === BUILD ARG LIST ===
WHISPER_FLAGS=(
  -m "$MODEL"
  --stdin --stdin-format f32le
  --eos-config "$EOS_CONF"
  --step 1000 --length 4000 --keep 0
)

if [[ "$DEBUG_EOS" == "true" ]]; then
  WHISPER_FLAGS+=(--debug-eos)
fi

# === PIPELINE ===
ffmpeg -hide_banner -nostats -loglevel warning \
  -f pulse -thread_queue_size 512 -i "$SPEAKER_SRC" \
  -f pulse -thread_queue_size 512 -i "$MIC_SRC" \
  -filter_complex "[1:a]aformat=sample_fmts=flt:channel_layouts=mono,aresample=16000,arnndn=m=$RNN:mix=0.95,highpass=f=80,lowpass=f=3400[mic]; \
                   [0:a]aformat=sample_fmts=flt:channel_layouts=mono,aresample=16000[speaker]; \
                   [mic][speaker]amix=inputs=2:duration=shortest:dropout_transition=2,highpass=f=120,lowpass=f=3500,volume=1.8,aresample=resampler=soxr:async=1:first_pts=0,asetpts=N/SR/TB" \
  -ac 1 -ar 16000 -f f32le - \
| ./build/bin/whisper-stream "${WHISPER_FLAGS[@]}" \
| tee -a whisper_output.txt
