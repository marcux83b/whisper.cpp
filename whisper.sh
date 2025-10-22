#!/usr/bin/env bash
export VK_ICD_FILENAMES="/usr/share/vulkan/icd.d/intel_icd.x86_64.json"
export DRI_PRIME=0
export GGML_VULKAN_DEBUG=0

MODEL="models/ggml-large-v3.bin"
#v1  -filter_complex "[0:a][1:a]amix=inputs=2:duration=shortest:dropout_transition=2,aresample=resampler=soxr:async=1:first_pts=0" \
#v2  -filter_complex "[0:a][1:a]amix=inputs=2:duration=shortest:dropout_transition=2,agate=threshold=-35dB:ratio=5:attack=5:release=500:makeup=0,aresample=resampler=soxr:async=1:first_pts=0" \
#v3  -filter_complex "[0:a][1:a]amix=inputs=2:duration=shortest:dropout_transition=2,agate=threshold=-45dB:ratio=2:attack=10:release=200,aresample=resampler=soxr:async=1:first_pts=0" \
#v4  -filter_complex "[0:a][1:a]amix=inputs=2:duration=shortest:dropout_transition=2,silenceremove=stop_periods=-1:stop_threshold=-40dB:stop_silence=2,aresample=resampler=soxr:async=1:first_pts=0,asetpts=N/SR/TB" \
#ffmpeg -hide_banner -nostats -loglevel warning \
#  -f pulse -thread_queue_size 512 \
#  -i alsa_output.pci-0000_00_1f.3-platform-skl_hda_dsp_generic.HiFi__Speaker__sink.monitor \
#  -f pulse -thread_queue_size 512 \
#  -i alsa_input.pci-0000_00_1f.3-platform-skl_hda_dsp_generic.HiFi__Mic1__source \
#  -filter_complex "[0:a][1:a]amix=inputs=2:duration=shortest:dropout_transition=2,highpass=f=200,lowpass=f=3000,compand=attacks=0.2:decays=0.3:points=-90/-900|-70/-70|-40/-20|0/0,volume=2.0,silenceremove=stop_periods=-1:stop_threshold=-35dB:detection=peak:stop_silence=1,aresample=resampler=soxr:async=1:first_pts=0,asetpts=N/SR/TB" \
#  -ac 1 -ar 16000 -f f32le - \
#| ./build/bin/whisper-stream \
#    -m "$MODEL" \
#    --stdin --stdin-format f32le \
#| tee -a whisper_output.txt

ffmpeg -hide_banner -nostats -loglevel warning \
  -f pulse -thread_queue_size 512 -i alsa_output.pci-0000_00_1f.3-platform-skl_hda_dsp_generic.HiFi__Speaker__sink.monitor \
  -f pulse -thread_queue_size 512 -i alsa_input.pci-0000_00_1f.3-platform-skl_hda_dsp_generic.HiFi__Mic1__source \
  -filter_complex "[0:a][1:a]amix=inputs=2:duration=shortest:dropout_transition=2,highpass=f=120,lowpass=f=3500,compand=attacks=0.2:decays=0.3:points=-90/-900|-70/-70|-40/-20|0/0,volume=2.0,silenceremove=stop_periods=-1:stop_threshold=-30dB:detection=peak:stop_silence=0.7,aresample=resampler=soxr:async=1:first_pts=0,asetpts=N/SR/TB" \
  -ac 1 -ar 16000 -f f32le - \
| ./build/bin/whisper-stream -m "$MODEL" --stdin --stdin-format f32le --step 5000 --length 15000 --keep 400 \
| tee -a whisper_output.txt
