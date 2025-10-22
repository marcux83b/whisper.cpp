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

# === PIPELINE (v68c with RNNoise tuned mix & sync buffers) ===
ffmpeg -hide_banner -nostats -loglevel warning \
  -f pulse -thread_queue_size 1024 -i "$MIC_SRC" \
  -filter_complex "aformat=sample_fmts=flt:channel_layouts=mono,aresample=16000, \
                   arnndn=m=$RNN:mix=0.90,highpass=f=80,lowpass=f=3400, \
                   compand=attacks=0.2:decays=0.4:points=-90/-900|-70/-70|-40/-20|0/0, \
                   volume=1.9,aresample=resampler=soxr:async=1:first_pts=0,asetpts=N/SR/TB" \
  -ac 1 -ar 16000 -f f32le - \
| ./build/bin/whisper-stream \
     -m "$MODEL" \
     --stdin --stdin-format f32le \
     --eos-config ./eos_defaults.conf \
     --debug-eos
