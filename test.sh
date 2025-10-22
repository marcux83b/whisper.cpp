#!/bin/bash
set -euo pipefail

mux_err() { while IFS= read -r line; do printf "[ERR] %s\n" "$line" >&3; done; }
mux_out() { while IFS= read -r line; do printf "[OUT] %s\n" "$line" >&3; done; }

exec 3> stream_long_v68b.mux
cat mic_test_long_gain_68.raw \
| stdbuf -oL -eL ./build/bin/whisper-stream \
    -m ./models/ggml-large-v3.bin \
    --stdin --stdin-format f32le \
    --eos-config ./eos_defaults.conf \
    --debug-eos \
    2> >(mux_err) \
| mux_out
exec 3>&-
