// Real-time speech recognition of input from a microphone
//
// A very quick-n-dirty implementation serving mainly as a proof of concept.
//
#include "common-sdl.h"
#include "common.h"
#include "common-whisper.h"
#include "whisper.h"

#include <chrono>
#include <cstdio>
#include <fstream>
#include <string>
#include <thread>
#include <vector>
#include <cmath>
#include <array>
#include <functional>
#include <sstream>
#include <algorithm>
#include <cctype>
#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

// command-line parameters
struct whisper_params {
    int32_t n_threads  = std::min(4, (int32_t) std::thread::hardware_concurrency());
    int32_t step_ms    = 3000;
    int32_t length_ms  = 10000;
    int32_t keep_ms    = 200;
    int32_t capture_id = -1;
    int32_t max_tokens = 32;
    int32_t audio_ctx  = 0;
    int32_t beam_size  = -1;

    float vad_thold    = 0.6f;
    float freq_thold   = 100.0f;

    bool translate     = false;
    bool no_fallback   = false;
    bool print_special = false;
    bool no_context    = true;
    bool no_timestamps = false;
    bool tinydiarize   = false;
    bool save_audio    = false; // save audio to wav file
    bool use_gpu       = true;
    bool flash_attn    = true;
    bool use_stdin     = false;           // use stdin instead of microphone
    std::string stdin_format = "f32le";   // stdin audio format: f32le or s16le

    std::string language  = "en";
    std::string model     = "models/ggml-base.en.bin";
    std::string fname_out;

    // low-confidence suppression (avg token prob threshold); 0.0 disables
    float low_conf_threshold = 0.0f;
};

// end-of-speech (EOS) detection parameters for stdin streaming
struct eos_params {
    float on  = 0.0085f;   // RMS threshold to enter speech (~ -44 dBFS)
    float off = 0.0045f;   // RMS threshold to leave speech (~ -49 dBFS)
    int   hang_ms = 200;   // keep 'speech' this long after last loud frame
    int   preroll_ms = 200;      // prepend audio at speech start
    int   flush_zero_ms = 1000;  // flush if this many ms of zeros
    int   min_chunk_ms  = 600;   // ignore blips shorter than this
    bool  debug = false;         // log EOS decisions
};

void whisper_print_usage(int argc, char ** argv, const whisper_params & params);

static bool load_eos_config_file(const std::string & path, eos_params & eos) {
    std::ifstream fin(path);
    if (!fin.is_open()) {
        fprintf(stderr, "warning: failed to open EOS config '%s'\n", path.c_str());
        return false;
    }
    std::string line;
    int count = 0;
    while (std::getline(fin, line)) {
        // strip comments
        const size_t hash = line.find('#');
        if (hash != std::string::npos) line = line.substr(0, hash);
        // trim
        auto ltrim = [](std::string & s){ s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch){ return !std::isspace(ch); })); };
        auto rtrim = [](std::string & s){ s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch){ return !std::isspace(ch); }).base(), s.end()); };
        ltrim(line); rtrim(line);
        if (line.empty()) continue;
        const size_t eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string key = line.substr(0, eq);
        std::string val = line.substr(eq+1);
        ltrim(key); rtrim(key); ltrim(val); rtrim(val);
        if (key == "eos-on") eos.on = std::stof(val);
        else if (key == "eos-off") eos.off = std::stof(val);
        else if (key == "eos-hang-ms") eos.hang_ms = std::stoi(val);
        else if (key == "eos-preroll-ms") eos.preroll_ms = std::stoi(val);
        else if (key == "eos-flush-zero-ms") eos.flush_zero_ms = std::stoi(val);
        else if (key == "eos-min-chunk-ms") eos.min_chunk_ms = std::stoi(val);
        else if (key == "debug-eos") {
            if (val == "1" || val == "true" || val == "yes" ) eos.debug = true; else eos.debug = false;
        } else {
            // ignore unknown keys
            continue;
        }
        ++count;
    }
    fprintf(stderr, "loaded EOS config '%s' (%d entries)\n", path.c_str(), count);
    return true;
}

static bool whisper_params_parse(int argc, char ** argv, whisper_params & params, eos_params & eos) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            whisper_print_usage(argc, argv, params);
            exit(0);
        }
        else if (arg == "-t"    || arg == "--threads")       { params.n_threads     = std::stoi(argv[++i]); }
        else if (                  arg == "--step")          { params.step_ms       = std::stoi(argv[++i]); }
        else if (                  arg == "--length")        { params.length_ms     = std::stoi(argv[++i]); }
        else if (                  arg == "--keep")          { params.keep_ms       = std::stoi(argv[++i]); }
        else if (arg == "-c"    || arg == "--capture")       { params.capture_id    = std::stoi(argv[++i]); }
        else if (arg == "-mt"   || arg == "--max-tokens")    { params.max_tokens    = std::stoi(argv[++i]); }
        else if (arg == "-ac"   || arg == "--audio-ctx")     { params.audio_ctx     = std::stoi(argv[++i]); }
        else if (arg == "-bs"   || arg == "--beam-size")     { params.beam_size     = std::stoi(argv[++i]); }
        else if (arg == "-vth"  || arg == "--vad-thold")     { params.vad_thold     = std::stof(argv[++i]); }
        else if (arg == "-fth"  || arg == "--freq-thold")    { params.freq_thold    = std::stof(argv[++i]); }
        else if (arg == "-tr"   || arg == "--translate")     { params.translate     = true; }
        else if (arg == "-nf"   || arg == "--no-fallback")   { params.no_fallback   = true; }
        else if (arg == "-ps"   || arg == "--print-special") { params.print_special = true; }
        else if (arg == "-kc"   || arg == "--keep-context")  { params.no_context    = false; }
        else if (arg == "-l"    || arg == "--language")      { params.language      = argv[++i]; }
        else if (arg == "-m"    || arg == "--model")         { params.model         = argv[++i]; }
        else if (arg == "-f"    || arg == "--file")          { params.fname_out     = argv[++i]; }
        else if (arg == "-tdrz" || arg == "--tinydiarize")   { params.tinydiarize   = true; }
        else if (arg == "-sa"   || arg == "--save-audio")    { params.save_audio    = true; }
        else if (arg == "-ng"   || arg == "--no-gpu")        { params.use_gpu       = false; }
        else if (arg == "-fa"   || arg == "--flash-attn")    { params.flash_attn    = true; }
        else if (arg == "-nfa"  || arg == "--no-flash-attn") { params.flash_attn    = false; }
        else if (                  arg == "--stdin")          { params.use_stdin     = true; }
        else if (                  arg == "--stdin-format")   { params.stdin_format  = argv[++i]; }
        else if (                  arg == "--lowconf-threshold") { params.low_conf_threshold = std::stof(argv[++i]); }
        else if (                  arg == "--eos-config")     { load_eos_config_file(argv[++i], eos); }
        // EOS flags (stdin streaming)
        else if (                  arg == "--eos-on")         { eos.on              = std::stof(argv[++i]); }
        else if (                  arg == "--eos-off")        { eos.off             = std::stof(argv[++i]); }
        else if (                  arg == "--eos-hang-ms")    { eos.hang_ms         = std::stoi(argv[++i]); }
        else if (                  arg == "--eos-preroll-ms") { eos.preroll_ms      = std::stoi(argv[++i]); }
        else if (                  arg == "--eos-flush-zero-ms") { eos.flush_zero_ms = std::stoi(argv[++i]); }
        else if (                  arg == "--eos-min-chunk-ms")  { eos.min_chunk_ms  = std::stoi(argv[++i]); }
        else if (                  arg == "--debug-eos")      { eos.debug           = true; }

        else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            whisper_print_usage(argc, argv, params);
            exit(0);
        }
    }

    return true;
}

void whisper_print_usage(int /*argc*/, char ** argv, const whisper_params & params) {
    fprintf(stderr, "\n");
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h,       --help          [default] show this help message and exit\n");
    fprintf(stderr, "  -t N,     --threads N     [%-7d] number of threads to use during computation\n",    params.n_threads);
    fprintf(stderr, "            --step N        [%-7d] audio step size in milliseconds\n",                params.step_ms);
    fprintf(stderr, "            --length N      [%-7d] audio length in milliseconds\n",                   params.length_ms);
    fprintf(stderr, "            --keep N        [%-7d] audio to keep from previous step in ms\n",         params.keep_ms);
    fprintf(stderr, "  -c ID,    --capture ID    [%-7d] capture device ID\n",                              params.capture_id);
    fprintf(stderr, "  -mt N,    --max-tokens N  [%-7d] maximum number of tokens per audio chunk\n",       params.max_tokens);
    fprintf(stderr, "  -ac N,    --audio-ctx N   [%-7d] audio context size (0 - all)\n",                   params.audio_ctx);
    fprintf(stderr, "  -bs N,    --beam-size N   [%-7d] beam size for beam search\n",                      params.beam_size);
    fprintf(stderr, "  -vth N,   --vad-thold N   [%-7.2f] voice activity detection threshold\n",           params.vad_thold);
    fprintf(stderr, "  -fth N,   --freq-thold N  [%-7.2f] high-pass frequency cutoff\n",                   params.freq_thold);
    fprintf(stderr, "  -tr,      --translate     [%-7s] translate from source language to english\n",      params.translate ? "true" : "false");
    fprintf(stderr, "  -nf,      --no-fallback   [%-7s] do not use temperature fallback while decoding\n", params.no_fallback ? "true" : "false");
    fprintf(stderr, "  -ps,      --print-special [%-7s] print special tokens\n",                           params.print_special ? "true" : "false");
    fprintf(stderr, "  -kc,      --keep-context  [%-7s] keep context between audio chunks\n",              params.no_context ? "false" : "true");
    fprintf(stderr, "  -l LANG,  --language LANG [%-7s] spoken language\n",                                params.language.c_str());
    fprintf(stderr, "  -m FNAME, --model FNAME   [%-7s] model path\n",                                     params.model.c_str());
    fprintf(stderr, "  -f FNAME, --file FNAME    [%-7s] text output file name\n",                          params.fname_out.c_str());
    fprintf(stderr, "  -tdrz,    --tinydiarize   [%-7s] enable tinydiarize (requires a tdrz model)\n",     params.tinydiarize ? "true" : "false");
    fprintf(stderr, "  -sa,      --save-audio    [%-7s] save the recorded audio to a file\n",              params.save_audio ? "true" : "false");
    fprintf(stderr, "  -ng,      --no-gpu        [%-7s] disable GPU inference\n",                          params.use_gpu ? "false" : "true");
    fprintf(stderr, "  -fa,      --flash-attn    [%-7s] enable flash attention during inference\n",        params.flash_attn ? "true" : "false");
    fprintf(stderr, "  -nfa,     --no-flash-attn [%-7s] disable flash attention during inference\n",       params.flash_attn ? "false" : "true");
    fprintf(stderr, "            --stdin         [%-7s] read PCM audio from stdin (no device)\n",           params.use_stdin ? "true" : "false");
    fprintf(stderr, "            --stdin-format  [%-7s] stdin PCM format: f32le or s16le\n",           params.stdin_format.c_str());
    fprintf(stderr, "            --eos-config F  [    none] load EOS params from key=value config file\n");
    fprintf(stderr, "            --eos-on F      [    0.0085] RMS to enter speech (stdin EOS)\n");
    fprintf(stderr, "            --eos-off F     [    0.0045] RMS to leave speech (stdin EOS)\n");
    fprintf(stderr, "            --eos-hang-ms N [        200] Hang time after last loud frame\n");
    fprintf(stderr, "            --eos-preroll-ms N[       200] Prepend this much audio at start\n");
    fprintf(stderr, "            --eos-flush-zero-ms N[    1000] Flush if this much zero-run\n");
    fprintf(stderr, "            --eos-min-chunk-ms N[      600] Ignore blips shorter than this\n");
    fprintf(stderr, "            --debug-eos     [   optional] Log EOS decisions to stderr\n");
    fprintf(stderr, "            --lowconf-threshold F [    0.00] Suppress segments with avg token prob below F (e.g., 0.35)\n");
    fprintf(stderr, "\n");
}

// ===== EOS helpers (20 ms frames @ 16k) =====
namespace eos_helpers {
    constexpr int SR = 16000;
    constexpr int FRAME_SAMPLES = SR * 20 / 1000; // 320

    inline float frame_rms(const float* p, int n) {
        double acc = 0.0;
        for (int i = 0; i < n; ++i) { double x = p[i]; acc += x*x; }
        return (float) std::sqrt(acc / n);
    }
    inline bool is_zero_frame(const float* p, int n) {
        for (int i = 0; i < n; ++i) if (std::fabs(p[i]) > 1e-7f) return false;
        return true;
    }
    inline int ms_to_frames(int ms) { return std::max(1, ms / 20); }

    struct float_ring {
        std::vector<float> buf; size_t w = 0; bool full = false;
        explicit float_ring(size_t cap) : buf(cap) {}
        void push(const float* p, size_t n) {
            for (size_t i=0;i<n;++i) { buf[w] = p[i]; w=(w+1)%buf.size(); if (w==0) full=true; }
        }
        void dump_to(std::vector<float>& out) const {
            if (!full && w==0) return;
            if (!full) { out.insert(out.end(), buf.begin(), buf.begin()+w); }
            else {
                out.insert(out.end(), buf.begin()+w, buf.end());
                out.insert(out.end(), buf.begin(), buf.begin()+w);
            }
        }
        void clear() { w=0; full=false; }
    };

    enum class State { IDLE, VOICE, HANG, ARMED };
    struct state_t {
        State st = State::IDLE;
        int hang = 0;
        int zero_run = 0;
        int frames_in_chunk = 0;
    };
}

int main(int argc, char ** argv) {
    ggml_backend_load_all();

    whisper_params params;
    eos_params eos;

    if (whisper_params_parse(argc, argv, params, eos) == false) {
        return 1;
    }

    // detect if stdout is a TTY (interactive console)
    bool stdout_is_tty = true;
#ifdef _WIN32
    stdout_is_tty = _isatty(_fileno(stdout)) != 0;
#else
    stdout_is_tty = isatty(fileno(stdout)) != 0;
#endif

    params.keep_ms   = std::min(params.keep_ms,   params.step_ms);
    params.length_ms = std::max(params.length_ms, params.step_ms);

    const int n_samples_step = (1e-3*params.step_ms  )*WHISPER_SAMPLE_RATE;
    const int n_samples_len  = (1e-3*params.length_ms)*WHISPER_SAMPLE_RATE;
    const int n_samples_keep = (1e-3*params.keep_ms  )*WHISPER_SAMPLE_RATE;
    const int n_samples_30s  = (1e-3*30000.0         )*WHISPER_SAMPLE_RATE;

    const bool use_vad = n_samples_step <= 0; // sliding window mode uses VAD

    const int n_new_line = !use_vad ? std::max(1, params.length_ms / params.step_ms - 1) : 1; // number of steps to print new line

    params.no_timestamps  = !use_vad;
    params.no_context    |= use_vad;
    params.max_tokens     = 0;

    // init audio

    audio_async audio(params.length_ms);
    if (!params.use_stdin) {
        if (!audio.init(params.capture_id, WHISPER_SAMPLE_RATE)) {
            fprintf(stderr, "%s: audio.init() failed!\n", __func__);
            return 1;
        }
        audio.resume();
    }

    // whisper init
    if (params.language != "auto" && whisper_lang_id(params.language.c_str()) == -1){
        fprintf(stderr, "error: unknown language '%s'\n", params.language.c_str());
        whisper_print_usage(argc, argv, params);
        exit(0);
    }

    struct whisper_context_params cparams = whisper_context_default_params();

    cparams.use_gpu    = params.use_gpu;
    cparams.flash_attn = params.flash_attn;

    struct whisper_context * ctx = whisper_init_from_file_with_params(params.model.c_str(), cparams);
    if (ctx == nullptr) {
        fprintf(stderr, "error: failed to initialize whisper context\n");
        return 2;
    }

    std::vector<float> pcmf32    (n_samples_30s, 0.0f);
    std::vector<float> pcmf32_old;
    std::vector<float> pcmf32_new(n_samples_30s, 0.0f);

    std::vector<whisper_token> prompt_tokens;

    // print some info about the processing
    {
        fprintf(stderr, "\n");
        if (!whisper_is_multilingual(ctx)) {
            if (params.language != "en" || params.translate) {
                params.language = "en";
                params.translate = false;
                fprintf(stderr, "%s: WARNING: model is not multilingual, ignoring language and translation options\n", __func__);
            }
        }
        fprintf(stderr, "%s: processing %d samples (step = %.1f sec / len = %.1f sec / keep = %.1f sec), %d threads, lang = %s, task = %s, timestamps = %d ...\n",
                __func__,
                n_samples_step,
                float(n_samples_step)/WHISPER_SAMPLE_RATE,
                float(n_samples_len )/WHISPER_SAMPLE_RATE,
                float(n_samples_keep)/WHISPER_SAMPLE_RATE,
                params.n_threads,
                params.language.c_str(),
                params.translate ? "translate" : "transcribe",
                params.no_timestamps ? 0 : 1);

        if (!use_vad) {
            fprintf(stderr, "%s: n_new_line = %d, no_context = %d\n", __func__, n_new_line, params.no_context);
        } else {
            fprintf(stderr, "%s: using VAD, will transcribe on speech activity\n", __func__);
        }

        fprintf(stderr, "\n");
    }

    int n_iter = 0;

    bool is_running = true;

    std::ofstream fout;
    if (params.fname_out.length() > 0) {
        fout.open(params.fname_out);
        if (!fout.is_open()) {
            fprintf(stderr, "%s: failed to open output file '%s'!\n", __func__, params.fname_out.c_str());
            return 1;
        }
    }

    wav_writer wavWriter;
    // save wav file
    if (params.save_audio) {
        // Get current date/time for filename
        time_t now = time(0);
        char buffer[80];
        strftime(buffer, sizeof(buffer), "%Y%m%d%H%M%S", localtime(&now));
        std::string filename = std::string(buffer) + ".wav";

        wavWriter.open(filename, WHISPER_SAMPLE_RATE, 16, 1);
    }
    if (params.use_stdin) {
        printf("[Reading audio from stdin. Expected %s PCM at %d Hz, mono]\n", params.stdin_format.c_str(), WHISPER_SAMPLE_RATE);
    } else {
        printf("[Start speaking]\n");
    }
    fflush(stdout);

    auto t_last  = std::chrono::high_resolution_clock::now();
    const auto t_start = t_last;

    // If stdin mode, run EOS-driven frame loop
    if (params.use_stdin) {
        // stdin reader lambdas
        using namespace eos_helpers;

        std::function<bool(float*,int)> read_frame;
        if (params.stdin_format == std::string("s16le")) {
            read_frame = [&](float* out, int n){
                std::vector<int16_t> tmp(n);
                size_t n_read = fread(tmp.data(), sizeof(int16_t), n, stdin);
                if (n_read < (size_t)n) {
                    if (feof(stdin)) return false;
                    // partial frame: pad zeros
                    for (size_t i = 0; i < n_read; ++i) out[i] = tmp[i] / 32768.0f;
                    for (int i = (int)n_read; i < n; ++i) out[i] = 0.0f;
                    return true;
                }
                for (int i = 0; i < n; ++i) out[i] = tmp[i] / 32768.0f;
                return true;
            };
        } else { // f32le
            read_frame = [&](float* out, int n){
                size_t n_read = fread(out, sizeof(float), n, stdin);
                if (n_read < (size_t)n) {
                    if (feof(stdin)) return false;
                    for (int i = (int)n_read; i < n; ++i) out[i] = 0.0f;
                    return true;
                }
                return true;
            };
        }

        // decode wrapper
        int n_iter_eos = 0;
        auto decode_chunk = [&](const std::vector<float>& buf){
            if (buf.empty()) return;
            whisper_full_params wparams = whisper_full_default_params(params.beam_size > 1 ? WHISPER_SAMPLING_BEAM_SEARCH : WHISPER_SAMPLING_GREEDY);
            wparams.print_progress   = false;
            wparams.print_special    = params.print_special;
            wparams.print_realtime   = false;
            wparams.print_timestamps = !params.no_timestamps;
            wparams.translate        = params.translate;
            wparams.single_segment   = true; // same as non-VAD streaming
            wparams.max_tokens       = params.max_tokens;
            wparams.language         = params.language.c_str();
            wparams.n_threads        = params.n_threads;
            wparams.beam_search.beam_size = params.beam_size;
            wparams.audio_ctx        = params.audio_ctx;
            wparams.tdrz_enable      = params.tinydiarize;
            wparams.temperature_inc  = params.no_fallback ? 0.0f : wparams.temperature_inc;
            wparams.prompt_tokens    = params.no_context ? nullptr : prompt_tokens.data();
            wparams.prompt_n_tokens  = params.no_context ? 0       : prompt_tokens.size();

            if (whisper_full(ctx, wparams, buf.data(), buf.size()) != 0) {
                fprintf(stderr, "%s: failed to process audio (stdin/EOS)\n", argv[0]);
                return;
            }

            // printing similar to non-VAD path
            if (stdout_is_tty) {
                printf("\33[2K\r");
                printf("%s", std::string(100, ' ').c_str());
                printf("\33[2K\r");
            } else {
                printf("\n");
            }

            const int n_segments = whisper_full_n_segments(ctx);
            for (int i = 0; i < n_segments; ++i) {
                const char * text = whisper_full_get_segment_text(ctx, i);

                // low-confidence suppression: compute average token probability for this segment
                if (params.low_conf_threshold > 0.0f) {
                    int token_count = whisper_full_n_tokens(ctx, i);
                    if (token_count > 0) {
                        double sum_p = 0.0;
                        for (int j = 0; j < token_count; ++j) {
                            sum_p += whisper_full_get_token_p(ctx, i, j);
                        }
                        float avg_p = (float)(sum_p / token_count);
                        if (avg_p < params.low_conf_threshold) {
                            fprintf(stderr, "[debug] low conf %.2f -> skipped: '%s'\n", avg_p, text);
                            continue;
                        }
                    }
                }
                if (params.no_timestamps) {
                    printf("%s", text);
                    fflush(stdout);
                    if (params.fname_out.length() > 0) {
                        fout << text;
                    }
                } else {
                    const int64_t t0s = whisper_full_get_segment_t0(ctx, i);
                    const int64_t t1s = whisper_full_get_segment_t1(ctx, i);
                    std::string output = "[" + to_timestamp(t0s, false) + " --> " + to_timestamp(t1s, false) + "]  " + text;
                    output += "\n";
                    printf("%s", output.c_str());
                    fflush(stdout);
                    if (params.fname_out.length() > 0) {
                        fout << output;
                    }
                }
            }
            if (params.fname_out.length() > 0) {
                fout << std::endl;
            }
            ++n_iter_eos;

            // update prompt tokens to keep context across chunks
            if (!params.no_context) {
                prompt_tokens.clear();
                const int n_segments2 = whisper_full_n_segments(ctx);
                for (int i = 0; i < n_segments2; ++i) {
                    const int token_count = whisper_full_n_tokens(ctx, i);
                    for (int j = 0; j < token_count; ++j) {
                        prompt_tokens.push_back(whisper_full_get_token_id(ctx, i, j));
                    }
                }
            }
        };

        auto dbg = [&](const char* m){ if (eos.debug) fprintf(stderr, "%s\n", m); };

        eos_helpers::float_ring preroll(eos_helpers::ms_to_frames(eos.preroll_ms) * eos_helpers::FRAME_SAMPLES);
        std::vector<float> chunk; chunk.reserve(8*eos_helpers::FRAME_SAMPLES);
        eos_helpers::state_t S;
        std::array<float, eos_helpers::FRAME_SAMPLES> fr;
        const int HANG_FR   = eos_helpers::ms_to_frames(eos.hang_ms);
        const int FLUSHZ_FR = eos_helpers::ms_to_frames(eos.flush_zero_ms);
        const int MIN_FR    = eos_helpers::ms_to_frames(eos.min_chunk_ms);

        auto flush_now = [&](){
            if ((int)chunk.size() >= MIN_FR * eos_helpers::FRAME_SAMPLES) {
                dbg("EOS: FLUSH");
                decode_chunk(chunk);
            } else {
                dbg("EOS: DROP tiny");
            }
            chunk.clear(); S.frames_in_chunk = 0; preroll.clear();
            S.st = eos_helpers::State::IDLE;
        };

        // Main frame loop
        while (true) {
            if (!read_frame(fr.data(), eos_helpers::FRAME_SAMPLES)) {
                // EOF: flush any pending
                if (!chunk.empty()) flush_now();
                break;
            }

            if (params.save_audio) {
                wavWriter.write(fr.data(), fr.size());
            }

            const bool zero = eos_helpers::is_zero_frame(fr.data(), eos_helpers::FRAME_SAMPLES);
            S.zero_run = zero ? (S.zero_run + 1) : 0;
            const float rms = eos_helpers::frame_rms(fr.data(), eos_helpers::FRAME_SAMPLES);
            const bool on  = (rms >= eos.on);
            const bool off = (rms <  eos.off);

            // preroll always records
            preroll.push(fr.data(), eos_helpers::FRAME_SAMPLES);

            switch (S.st) {
                case eos_helpers::State::IDLE:
                    if (on && !zero) {
                        preroll.dump_to(chunk);
                        chunk.insert(chunk.end(), fr.begin(), fr.end());
                        S.frames_in_chunk = (int)(chunk.size()/eos_helpers::FRAME_SAMPLES);
                        S.hang = HANG_FR;
                        S.st = eos_helpers::State::VOICE;
                        dbg("EOS: IDLE->VOICE");
                    }
                    break;
                case eos_helpers::State::VOICE:
                    chunk.insert(chunk.end(), fr.begin(), fr.end());
                    ++S.frames_in_chunk;
                    if (on && !zero) {
                        S.hang = HANG_FR;
                    } else {
                        S.st = eos_helpers::State::HANG; dbg("EOS: VOICE->HANG");
                    }
                    break;
                case eos_helpers::State::HANG:
                    chunk.insert(chunk.end(), fr.begin(), fr.end());
                    ++S.frames_in_chunk;
                    if (on && !zero) {
                        S.st = eos_helpers::State::VOICE; S.hang = HANG_FR; dbg("EOS: HANG->VOICE");
                    } else {
                        if (--S.hang <= 0) { S.st = eos_helpers::State::ARMED; dbg("EOS: HANG->ARMED"); }
                    }
                    break;
                case eos_helpers::State::ARMED:
                    if (S.zero_run >= FLUSHZ_FR || off) {
                        flush_now();
                    } else if (on && !zero) {
                        S.st = eos_helpers::State::VOICE; S.hang = HANG_FR;
                        chunk.insert(chunk.end(), fr.begin(), fr.end());
                        ++S.frames_in_chunk;
                        dbg("EOS: ARMED->VOICE");
                    }
                    break;
            }

            // mid-slice long utterances using --length/--keep
            if (S.st != eos_helpers::State::IDLE) {
                const int cur_ms = S.frames_in_chunk * 20;
                if (cur_ms >= params.length_ms) {
                    // decode current chunk but keep the last --keep ms
                    decode_chunk(chunk);
                    const int keep_samps = (int)((WHISPER_SAMPLE_RATE * params.keep_ms) / 1000);
                    if (keep_samps > 0 && (int)chunk.size() > keep_samps) {
                        std::vector<float> kept(chunk.end() - keep_samps, chunk.end());
                        chunk.swap(kept);
                    } else {
                        chunk.clear();
                    }
                    S.frames_in_chunk = (int)(chunk.size()/eos_helpers::FRAME_SAMPLES);
                    preroll.clear(); // avoid duplicating preroll mid-utterance
                    dbg("EOS: MID-SLICE");
                }
            }
        }

        // stdin mode done
        audio.pause();
        whisper_print_timings(ctx);
        whisper_free(ctx);
        return 0;
    }

    // main audio loop (microphone / SDL)
    while (is_running) {
        // stdin path handled earlier
        if (params.save_audio) {
            wavWriter.write(pcmf32_new.data(), pcmf32_new.size());
        }
        // handle Ctrl + C
        is_running = sdl_poll_events();

        if (!is_running) {
            break;
        }

        // process new audio

        if (!use_vad) {
            while (true) {
                // handle Ctrl + C
                is_running = sdl_poll_events();
                if (!is_running) {
                    break;
                }
                if (params.use_stdin && audio.is_stdin_eof()) {
                    // break out of inner loop and outer loop will catch EOF too
                    is_running = false;
                    break;
                }
                audio.get(params.step_ms, pcmf32_new);

                if ((int) pcmf32_new.size() > 2*n_samples_step) {
                    fprintf(stderr, "\n\n%s: WARNING: cannot process audio fast enough, dropping audio ...\n\n", __func__);
                    audio.clear();
                    continue;
                }

                if ((int) pcmf32_new.size() >= n_samples_step) {
                    audio.clear();
                    break;
                }

                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }

            const int n_samples_new = pcmf32_new.size();

            // take up to params.length_ms audio from previous iteration
            const int n_samples_take = std::min((int) pcmf32_old.size(), std::max(0, n_samples_keep + n_samples_len - n_samples_new));

            //printf("processing: take = %d, new = %d, old = %d\n", n_samples_take, n_samples_new, (int) pcmf32_old.size());

            pcmf32.resize(n_samples_new + n_samples_take);

            for (int i = 0; i < n_samples_take; i++) {
                pcmf32[i] = pcmf32_old[pcmf32_old.size() - n_samples_take + i];
            }

            memcpy(pcmf32.data() + n_samples_take, pcmf32_new.data(), n_samples_new*sizeof(float));

            pcmf32_old = pcmf32;
        } else {
            const auto t_now  = std::chrono::high_resolution_clock::now();
            const auto t_diff = std::chrono::duration_cast<std::chrono::milliseconds>(t_now - t_last).count();

            if (t_diff < 2000) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));

                continue;
            }

            audio.get(2000, pcmf32_new);

            if (::vad_simple(pcmf32_new, WHISPER_SAMPLE_RATE, 1000, params.vad_thold, params.freq_thold, false)) {
                audio.get(params.length_ms, pcmf32);
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));

                continue;
            }

            t_last = t_now;
        }

        // run the inference
        {
            whisper_full_params wparams = whisper_full_default_params(params.beam_size > 1 ? WHISPER_SAMPLING_BEAM_SEARCH : WHISPER_SAMPLING_GREEDY);

            wparams.print_progress   = false;
            wparams.print_special    = params.print_special;
            wparams.print_realtime   = false;
            wparams.print_timestamps = !params.no_timestamps;
            wparams.translate        = params.translate;
            wparams.single_segment   = !use_vad;
            wparams.max_tokens       = params.max_tokens;
            wparams.language         = params.language.c_str();
            wparams.n_threads        = params.n_threads;
            wparams.beam_search.beam_size = params.beam_size;

            wparams.audio_ctx        = params.audio_ctx;

            wparams.tdrz_enable      = params.tinydiarize; // [TDRZ]

            // disable temperature fallback
            //wparams.temperature_inc  = -1.0f;
            wparams.temperature_inc  = params.no_fallback ? 0.0f : wparams.temperature_inc;

            wparams.prompt_tokens    = params.no_context ? nullptr : prompt_tokens.data();
            wparams.prompt_n_tokens  = params.no_context ? 0       : prompt_tokens.size();

            if (whisper_full(ctx, wparams, pcmf32.data(), pcmf32.size()) != 0) {
                fprintf(stderr, "%s: failed to process audio\n", argv[0]);
                return 6;
            }

            // print result;
            {
                if (!use_vad) {
                    if (stdout_is_tty) {
                        printf("\33[2K\r");
                        // print long empty line to clear the previous line
                        printf("%s", std::string(100, ' ').c_str());
                        printf("\33[2K\r");
                    } else {
                        // when stdout is not a TTY, do not overwrite; add a newline separator
                        printf("\n");
                    }
                } else {
                    const int64_t t1 = (t_last - t_start).count()/1000000;
                    const int64_t t0 = std::max(0.0, t1 - pcmf32.size()*1000.0/WHISPER_SAMPLE_RATE);

                    printf("\n");
                    printf("### Transcription %d START | t0 = %d ms | t1 = %d ms\n", n_iter, (int) t0, (int) t1);
                    printf("\n");
                }

                const int n_segments = whisper_full_n_segments(ctx);
                for (int i = 0; i < n_segments; ++i) {
                    const char * text = whisper_full_get_segment_text(ctx, i);

                    // low-confidence suppression: compute average token probability for this segment
                    if (params.low_conf_threshold > 0.0f) {
                        int token_count = whisper_full_n_tokens(ctx, i);
                        if (token_count > 0) {
                            double sum_p = 0.0;
                            for (int j = 0; j < token_count; ++j) {
                                sum_p += whisper_full_get_token_p(ctx, i, j);
                            }
                            float avg_p = (float)(sum_p / token_count);
                            if (avg_p < params.low_conf_threshold) {
                                fprintf(stderr, "[debug] low conf %.2f -> skipped: '%s'\n", avg_p, text);
                                continue;
                            }
                        }
                    }

                    if (params.no_timestamps) {
                        printf("%s", text);
                        fflush(stdout);

                        if (params.fname_out.length() > 0) {
                            fout << text;
                        }
                    } else {
                        const int64_t t0 = whisper_full_get_segment_t0(ctx, i);
                        const int64_t t1 = whisper_full_get_segment_t1(ctx, i);

                        std::string output = "[" + to_timestamp(t0, false) + " --> " + to_timestamp(t1, false) + "]  " + text;

                        if (whisper_full_get_segment_speaker_turn_next(ctx, i)) {
                            output += " [SPEAKER_TURN]";
                        }

                        output += "\n";

                        printf("%s", output.c_str());
                        fflush(stdout);

                        if (params.fname_out.length() > 0) {
                            fout << output;
                        }
                    }
                }

                if (params.fname_out.length() > 0) {
                    fout << std::endl;
                }

                if (use_vad) {
                    printf("\n");
                    printf("### Transcription %d END\n", n_iter);
                }
            }

            ++n_iter;

            if (!use_vad && (n_iter % n_new_line) == 0) {
                printf("\n");

                // keep part of the audio for next iteration to try to mitigate word boundary issues
                pcmf32_old = std::vector<float>(pcmf32.end() - n_samples_keep, pcmf32.end());

                // Add tokens of the last full length segment as the prompt
                if (!params.no_context) {
                    prompt_tokens.clear();

                    const int n_segments = whisper_full_n_segments(ctx);
                    for (int i = 0; i < n_segments; ++i) {
                        const int token_count = whisper_full_n_tokens(ctx, i);
                        for (int j = 0; j < token_count; ++j) {
                            prompt_tokens.push_back(whisper_full_get_token_id(ctx, i, j));
                        }
                    }
                }
            }

            // In VAD mode, avoid reprocessing the same audio on the next cycle
            if (use_vad) {
                audio.clear();
            }
            fflush(stdout);
        }
    }

    audio.pause();

    whisper_print_timings(ctx);
    whisper_free(ctx);

    return 0;
}
