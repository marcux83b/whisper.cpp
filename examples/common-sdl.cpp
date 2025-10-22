#include "common-sdl.h"

#include <cstdio>
#include <cstring>
#include <algorithm>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

audio_async::audio_async(int len_ms) {
    m_len_ms = len_ms;

    m_running = false;
}

audio_async::~audio_async() {
    if (m_dev_id_in) {
        SDL_CloseAudioDevice(m_dev_id_in);
    }
}

bool audio_async::init(int capture_id, int sample_rate) {
    SDL_LogSetPriority(SDL_LOG_CATEGORY_APPLICATION, SDL_LOG_PRIORITY_INFO);

    if (SDL_Init(SDL_INIT_AUDIO) < 0) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "Couldn't initialize SDL: %s\n", SDL_GetError());
        return false;
    }

    SDL_SetHintWithPriority(SDL_HINT_AUDIO_RESAMPLING_MODE, "medium", SDL_HINT_OVERRIDE);

    {
        int nDevices = SDL_GetNumAudioDevices(SDL_TRUE);
        fprintf(stderr, "%s: found %d capture devices:\n", __func__, nDevices);
        for (int i = 0; i < nDevices; i++) {
            fprintf(stderr, "%s:    - Capture device #%d: '%s'\n", __func__, i, SDL_GetAudioDeviceName(i, SDL_TRUE));
        }
    }

    SDL_AudioSpec capture_spec_requested;
    SDL_AudioSpec capture_spec_obtained;

    SDL_zero(capture_spec_requested);
    SDL_zero(capture_spec_obtained);

    capture_spec_requested.freq     = sample_rate;
    capture_spec_requested.format   = AUDIO_F32;
    capture_spec_requested.channels = 1;
    capture_spec_requested.samples  = 1024;
    capture_spec_requested.callback = [](void * userdata, uint8_t * stream, int len) {
        audio_async * audio = (audio_async *) userdata;
        audio->callback(stream, len);
    };
    capture_spec_requested.userdata = this;

    if (capture_id >= 0) {
        fprintf(stderr, "%s: attempt to open capture device %d : '%s' ...\n", __func__, capture_id, SDL_GetAudioDeviceName(capture_id, SDL_TRUE));
        m_dev_id_in = SDL_OpenAudioDevice(SDL_GetAudioDeviceName(capture_id, SDL_TRUE), SDL_TRUE, &capture_spec_requested, &capture_spec_obtained, 0);
    } else {
        fprintf(stderr, "%s: attempt to open default capture device ...\n", __func__);
        m_dev_id_in = SDL_OpenAudioDevice(nullptr, SDL_TRUE, &capture_spec_requested, &capture_spec_obtained, 0);
    }

    if (!m_dev_id_in) {
        fprintf(stderr, "%s: couldn't open an audio device for capture: %s!\n", __func__, SDL_GetError());
        m_dev_id_in = 0;

        return false;
    } else {
        fprintf(stderr, "%s: obtained spec for input device (SDL Id = %d):\n", __func__, m_dev_id_in);
        fprintf(stderr, "%s:     - sample rate:       %d\n",                   __func__, capture_spec_obtained.freq);
        fprintf(stderr, "%s:     - format:            %d (required: %d)\n",    __func__, capture_spec_obtained.format,
                capture_spec_requested.format);
        fprintf(stderr, "%s:     - channels:          %d (required: %d)\n",    __func__, capture_spec_obtained.channels,
                capture_spec_requested.channels);
        fprintf(stderr, "%s:     - samples per frame: %d\n",                   __func__, capture_spec_obtained.samples);
    }

    m_sample_rate = capture_spec_obtained.freq;

    m_audio.resize((m_sample_rate*m_len_ms)/1000);

    return true;
}

bool audio_async::init_stdin(int sample_rate, const std::string & format) {
    m_sample_rate  = sample_rate;
    m_audio.resize((m_sample_rate*m_len_ms)/1000);
    m_input_mode   = MODE_STDIN;
    m_stdin_eof    = false;
    m_stdin_format = format;

#ifdef _WIN32
    _setmode(_fileno(stdin), _O_BINARY);
#endif

    fprintf(stderr, "%s: initialized stdin audio capture, sample rate = %d, format = %s\n", __func__, m_sample_rate, m_stdin_format.c_str());
    return true;
}

bool audio_async::resume() {
    if (m_input_mode == MODE_STDIN) {
        if (m_running) {
            fprintf(stderr, "%s: already running!\n", __func__);
            return false;
        }
        m_running = true;
        return true;
    }

    if (!m_dev_id_in) {
        fprintf(stderr, "%s: no audio device to resume!\n", __func__);
        return false;
    }

    if (m_running) {
        fprintf(stderr, "%s: already running!\n", __func__);
        return false;
    }

    SDL_PauseAudioDevice(m_dev_id_in, 0);

    m_running = true;

    return true;
}

bool audio_async::pause() {
    if (m_input_mode == MODE_STDIN) {
        if (!m_running) {
            fprintf(stderr, "%s: already paused!\n", __func__);
            return false;
        }
        m_running = false;
        return true;
    }

    if (!m_dev_id_in) {
        fprintf(stderr, "%s: no audio device to pause!\n", __func__);
        return false;
    }

    if (!m_running) {
        fprintf(stderr, "%s: already paused!\n", __func__);
        return false;
    }

    SDL_PauseAudioDevice(m_dev_id_in, 1);

    m_running = false;

    return true;
}

bool audio_async::clear() {
    if (m_input_mode == MODE_STDIN) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_audio_pos = 0;
        m_audio_len = 0;
        return true;
    }

    if (!m_dev_id_in) {
        fprintf(stderr, "%s: no audio device to clear!\n", __func__);
        return false;
    }

    if (!m_running) {
        fprintf(stderr, "%s: not running!\n", __func__);
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(m_mutex);

        m_audio_pos = 0;
        m_audio_len = 0;
    }

    return true;
}

// callback to be called by SDL
void audio_async::callback(uint8_t * stream, int len) {
    if (!m_running) {
        return;
    }

    size_t n_samples = len / sizeof(float);

    if (n_samples > m_audio.size()) {
        n_samples = m_audio.size();

        stream += (len - (n_samples * sizeof(float)));
    }

    //fprintf(stderr, "%s: %zu samples, pos %zu, len %zu\n", __func__, n_samples, m_audio_pos, m_audio_len);

    {
        std::lock_guard<std::mutex> lock(m_mutex);

        if (m_audio_pos + n_samples > m_audio.size()) {
            const size_t n0 = m_audio.size() - m_audio_pos;

            memcpy(&m_audio[m_audio_pos], stream, n0 * sizeof(float));
            memcpy(&m_audio[0], stream + n0 * sizeof(float), (n_samples - n0) * sizeof(float));
        } else {
            memcpy(&m_audio[m_audio_pos], stream, n_samples * sizeof(float));
        }
        m_audio_pos = (m_audio_pos + n_samples) % m_audio.size();
        m_audio_len = std::min(m_audio_len + n_samples, m_audio.size());
    }
}

void audio_async::get(int ms, std::vector<float> & result) {
    if (m_input_mode == MODE_STDIN && m_running) {
        // For stdin mode, try to read more data before returning
        read_from_stdin();
    }

    if (m_input_mode != MODE_STDIN && !m_dev_id_in) {
        fprintf(stderr, "%s: no audio device to get audio from!\n", __func__);
        return;
    }

    if (!m_running) {
        fprintf(stderr, "%s: not running!\n", __func__);
        return;
    }

    result.clear();

    {
        std::lock_guard<std::mutex> lock(m_mutex);

        if (ms <= 0) {
            ms = m_len_ms;
        }

        size_t n_samples = (m_sample_rate * ms) / 1000;
        if (n_samples > m_audio_len) {
            n_samples = m_audio_len;
        }

        result.resize(n_samples);

        int s0 = m_audio_pos - n_samples;
        if (s0 < 0) {
            s0 += m_audio.size();
        }

        if (s0 + n_samples > m_audio.size()) {
            const size_t n0 = m_audio.size() - s0;

            memcpy(result.data(), &m_audio[s0], n0 * sizeof(float));
            memcpy(&result[n0], &m_audio[0], (n_samples - n0) * sizeof(float));
        } else {
            memcpy(result.data(), &m_audio[s0], n_samples * sizeof(float));
        }
    }
}

bool sdl_poll_events() {
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
        switch (event.type) {
            case SDL_QUIT:
                {
                    return false;
                }
            default:
                break;
        }
    }

    return true;
}

bool audio_async::read_from_stdin() {
    if (m_stdin_eof) {
        return false;
    }

    // Number of samples to read (1/10th of a second of audio)
    const size_t n_samples_to_read = std::max(1, m_sample_rate / 10);

    size_t n_read = 0;
    std::vector<float> buf;
    buf.resize(n_samples_to_read);

    if (m_stdin_format == "s16le") {
        std::vector<int16_t> tmp(n_samples_to_read);
        n_read = fread(tmp.data(), sizeof(int16_t), n_samples_to_read, stdin);
        if (n_read == 0) {
            if (feof(stdin)) {
                fprintf(stderr, "%s: reached end of stdin (s16le)\n", __func__);
                m_stdin_eof = true;
                return false;
            }
            return true; // no data yet
        }
        for (size_t i = 0; i < n_read; ++i) {
            buf[i] = tmp[i] / 32768.0f;
        }
    } else { // f32le (default)
        n_read = fread(buf.data(), sizeof(float), n_samples_to_read, stdin);
        if (n_read == 0) {
            if (feof(stdin)) {
                fprintf(stderr, "%s: reached end of stdin (f32le)\n", __func__);
                m_stdin_eof = true;
                return false;
            }
            return true; // no data yet
        }
    }

    // Store the data in the circular buffer
    {
        std::lock_guard<std::mutex> lock(m_mutex);

        const size_t cap = m_audio.size();
        if (cap == 0) return false;

        size_t remaining = n_read;
        size_t offset    = 0;
        while (remaining > 0) {
            const size_t to_end = cap - m_audio_pos;
            const size_t n0 = std::min(remaining, to_end);
            memcpy(&m_audio[m_audio_pos], buf.data() + offset, n0 * sizeof(float));
            m_audio_pos = (m_audio_pos + n0) % cap;
            m_audio_len = std::min(m_audio_len + n0, cap);
            remaining -= n0;
            offset    += n0;
        }
    }

    return true;
}
