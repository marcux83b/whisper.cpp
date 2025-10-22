#!/bin/bash
set -euo pipefail

# === PREP ===
mux_err() { while IFS= read -r line; do printf "[ERR] %s\n" "$line" >&3; done; }
mux_out() { while IFS= read -r line; do printf "[OUT] %s\n" "$line" >&3; done; }

MODEL="./models/ggml-large-v3.bin"
EOS_CONF="./eos_defaults.conf"
RNN="$HOME/rnnoise-models/std.rnnn"

exec 3> stream_long_v68d_rnnoise.mux

# === PIPELINE ===
ffmpeg -hide_banner -nostats -loglevel warning \
  -f f32le -ar 16000 -ac 1 -i mic_test_long_gain_68.raw \
  -filter_complex "aformat=sample_fmts=flt:channel_layouts=mono,aresample=16000,
                   arnndn=m=$RNN:mix=0.90,highpass=f=80,lowpass=f=3400,
                   compand=attacks=0.2:decays=0.4:points=-90/-900|-70/-70|-40/-20|0/0,
                   volume=1.9,aresample=resampler=soxr:async=1:first_pts=0,asetpts=N/SR/TB" \
  -ac 1 -ar 16000 -f f32le - \
| stdbuf -oL -eL ./build/bin/whisper-stream \
      -m "$MODEL" \
      --stdin --stdin-format f32le \
      --eos-config "$EOS_CONF" \
      --debug-eos \
      2> >(mux_err) \
| mux_out

exec 3>&-
