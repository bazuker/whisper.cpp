#include "common-sdl.h"
#include "common.h"
#include "common-whisper.h"
#include "whisper.h"

#include <chrono>
#include <deque>
#include <algorithm>
#include <cctype>
#include <regex>
#include <cassert>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>
#include <fstream>
#include <iostream>
#include "json.hpp"
#include <curl/curl.h>
#include <atomic>
#include <mutex>
#include "httplib.h"

using json = nlohmann::json;

std::deque<std::string> transcriptions_seen;  // Stack to store the last transcriptions

// --- GLOBALS FOR NEW HTTP-TRIGGERED TRANSCRIPTION ---
std::atomic<bool> g_transcribe_mode{false};
std::mutex g_buffer_mutex;
std::deque<float> g_audio_buffer;  // Circular buffer to hold last 2 seconds of audio

// Global pointer for the HTTP server
std::shared_ptr<httplib::Server> g_server = nullptr;

// Define the number of samples corresponding to 2 seconds.
// (Assumes WHISPER_SAMPLE_RATE is defined in common-whisper.h)
const int n_samples_2sec = 2 * WHISPER_SAMPLE_RATE;  // 2 seconds worth of samples

// HTTP server function running in a separate thread
void http_server_thread() {
    // Create the server and store it in the global pointer
    g_server = std::make_shared<httplib::Server>();
    g_server->Get("/transcribe", [](const httplib::Request &req, httplib::Response &res) {
        // When the /transcribe endpoint is hit, enable transcription.
        g_transcribe_mode = true;
        res.set_content("Transcription enabled", "text/plain");
    });
    // Listen on port 8080 on all interfaces.
    g_server->listen("0.0.0.0", 8080);
}

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
    bool flash_attn    = false;

    std::string language  = "en";
    std::string model     = "models/ggml-base.en.bin";
    std::string fname_out;
    std::string server_url = "";
};

void whisper_print_usage(int argc, char ** argv, const whisper_params & params);

static bool whisper_params_parse(int argc, char ** argv, whisper_params & params) {
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
        else if (arg == "-u"    || arg == "--url")           { params.server_url    = argv[++i];}

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
    fprintf(stderr, "  -fa,      --flash-attn    [%-7s] flash attention during inference\n",               params.flash_attn ? "true" : "false");
    fprintf(stderr, "  -u URL,    --url URL      [%-7s] send transcriptions to this URL\n", params.server_url.c_str());
    fprintf(stderr, "\n");
}

// Callback function to capture server response
size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* out) {
    size_t total_size = size * nmemb;
    out->append((char*)contents, total_size);  // Append response data to the string
    return total_size;
}

void send_transcription(const std::string& server_url, const std::string& transcription, double timestamp_start, double timestamp_end) {
    CURLcode res;
    std::string response_data;
    long http_code = 0;

    // Initialize cURL
    CURL* curl = curl_easy_init();
    if (!curl) {
        std::cerr << "Curl initialization failed!" << std::endl;
        return;
    }

    // Prepare JSON data with transcription, timestamp_start, and timestamp_end
    json post_data = {
        {"text", transcription},
        {"timestamp_start", timestamp_start},
        {"timestamp_end", timestamp_end}
    };
    std::string json_data = post_data.dump();

    // Set up headers (using a curl_slist)
    struct curl_slist *headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");

    // Set the URL and other options
    res = curl_easy_setopt(curl, CURLOPT_URL, server_url.c_str());
    if (res != CURLE_OK) {
        std::cerr << "curl_easy_setopt() failed for URL: " << curl_easy_strerror(res) << std::endl;
        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);
        return;
    }

    res = curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_data.c_str());
    if (res != CURLE_OK) {
        std::cerr << "curl_easy_setopt() failed for POSTFIELDS: " << curl_easy_strerror(res) << std::endl;
        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);
        return;
    }

    res = curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    if (res != CURLE_OK) {
        std::cerr << "curl_easy_setopt() failed for HTTPHEADER: " << curl_easy_strerror(res) << std::endl;
        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);
        return;
    }

    // Set the write callback to capture the response
    res = curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    if (res != CURLE_OK) {
        std::cerr << "curl_easy_setopt() failed for WRITEFUNCTION: " << curl_easy_strerror(res) << std::endl;
        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);
        return;
    }

    // Pass the string to store the response in
    res = curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_data);
    if (res != CURLE_OK) {
        std::cerr << "curl_easy_setopt() failed for WRITEDATA: " << curl_easy_strerror(res) << std::endl;
        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);
        return;
    }

    // Perform the request
    res = curl_easy_perform(curl);

    // Check the HTTP status code
    if (res != CURLE_OK) {
        std::cerr << "CURL request failed: " << curl_easy_strerror(res) << std::endl;
    } else {
        // Get the HTTP response code
        res = curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
        if (res != CURLE_OK) {
            std::cerr << "Failed to get response code: " << curl_easy_strerror(res) << std::endl;
        }

        // Print the response only if the status code is not 200 OK
        if (http_code != 200) {
            std::cout << "Response from server (HTTP " << http_code << "): " << response_data << std::endl;
        }
    }

    // Clean up
    curl_slist_free_all(headers);  // Free the header list
    curl_easy_cleanup(curl);  // Clean up cURL handle
}

std::string normalize_text(const std::string& text) {
    std::string normalized = text;

    // Convert to lowercase
    std::transform(normalized.begin(), normalized.end(), normalized.begin(), ::tolower);

    // Remove all symbols (punctuation like ,.;!?)
    normalized = std::regex_replace(normalized, std::regex(R"([^\w\s])"), "");

    return normalized;
}

void add_transcription(const std::string& raw_text, const std::string& fname_out, const std::string& server_url, double timestamp_start, double timestamp_end) {
    std::string text = normalize_text(raw_text);  // Normalize text for duplicate detection

    // Check if this normalized transcription is already in the last 5
    for (const auto& prev_text : transcriptions_seen) {
        if (prev_text == text) {
            //std::cerr << "Duplicate transcription detected, skipping: " << raw_text << std::endl;
            return;  // Avoid writing duplicate transcriptions
        }
    }

    transcriptions_seen.push_back(text);

    if (transcriptions_seen.size() > 3) {
        transcriptions_seen.pop_front();
    }

    // Save to file if specified
    if (!fname_out.empty()) {
        json jsonOutput = {
            {"timestamp_start", timestamp_start},
            {"timestamp_end", timestamp_end},
            {"text", raw_text}  // Store the original transcription, not the normalized one
        };
        std::ofstream fout(fname_out, std::ios::app);
        fout << jsonOutput.dump() << std::endl;
    }

    // Send to server if URL is provided
    if (!server_url.empty()) {
        send_transcription(server_url, raw_text, timestamp_start, timestamp_end);
    }
}

int main(int argc, char ** argv) {
    whisper_params params;

    if (whisper_params_parse(argc, argv, params) == false) {
        return 1;
    }

    curl_global_init(CURL_GLOBAL_DEFAULT);

    params.keep_ms   = std::min(params.keep_ms,   params.step_ms);
    params.length_ms = std::max(params.length_ms, params.step_ms);

    const int n_samples_step = (1e-3*params.step_ms  )*WHISPER_SAMPLE_RATE;
    const int n_samples_len  = (1e-3*params.length_ms)*WHISPER_SAMPLE_RATE;
    const int n_samples_keep = (1e-3*params.keep_ms  )*WHISPER_SAMPLE_RATE;

    const bool use_vad = n_samples_step <= 0; // sliding window mode uses VAD

    const int n_new_line = !use_vad ? std::max(1, params.length_ms / params.step_ms - 1) : 1; // number of steps to print new line

    params.no_timestamps  = !use_vad;
    params.no_context    |= use_vad;
    params.max_tokens     = 0;

    // init audio
    audio_async audio(params.length_ms);
    if (!audio.init(params.capture_id, WHISPER_SAMPLE_RATE)) {
        fprintf(stderr, "%s: audio.init() failed!\n", __func__);
        return 1;
    }

    audio.resume();

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

    std::vector<float> pcmf32    (n_samples_len, 0.0f);
    std::vector<float> pcmf32_new(n_samples_len, 0.0f);

    std::vector<whisper_token> prompt_tokens;

    // Start the HTTP server in a separate thread.
    std::thread server_thread(http_server_thread);

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
        //strftime(buffer, sizeof(buffer), "%Y%m%d%H%M%S", localtime(&now));
        //std::string filename = std::string(buffer) + ".wav";
        std::string filename = "save.wav";

        wavWriter.open(filename, WHISPER_SAMPLE_RATE, 16, 1);
    }
    printf("[Start speaking]\n");
    fflush(stdout);

    whisper_full_params wparams = whisper_full_default_params(params.beam_size > 1 ? WHISPER_SAMPLING_BEAM_SEARCH : WHISPER_SAMPLING_GREEDY);

    wparams.print_progress   = false;
    wparams.print_special    = params.print_special;
    wparams.print_realtime   = false;
    wparams.print_timestamps = !params.no_timestamps;
    wparams.translate        = params.translate;
    wparams.single_segment   = true;
    wparams.max_tokens       = params.max_tokens;
    wparams.language         = params.language.c_str();
    wparams.n_threads        = params.n_threads;
    wparams.beam_search.beam_size = params.beam_size;

    wparams.audio_ctx        = params.audio_ctx;
    wparams.tdrz_enable      = params.tinydiarize;
    wparams.temperature_inc  = params.no_fallback ? 0.0f : wparams.temperature_inc;
    wparams.prompt_tokens    = nullptr;
    wparams.prompt_n_tokens  = 0;

    // main audio loop
    while (is_running) {
        is_running = sdl_poll_events();
        if (!is_running) {
            break;
        }

        // Lock and update buffer
//        if (!g_transcribe_mode) {
//            audio.get(200, pcmf32_new);
//            {
//                std::lock_guard<std::mutex> lock(g_buffer_mutex);
//                g_audio_buffer.insert(g_audio_buffer.end(), pcmf32_new.begin(), pcmf32_new.end());
//                while (g_audio_buffer.size() > n_samples_2sec) {
//                    g_audio_buffer.pop_front();
//                }
//            }
//
//            std::this_thread::sleep_for(std::chrono::milliseconds(50));
//            continue;
//        }
        if (!g_transcribe_mode) {
            //audio.get(3000, pcmf32_new);
            // printf("[new %ld]\n", pcmf32_new.size());
            // fflush(stdout);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        auto transcription_started = std::chrono::high_resolution_clock::now();

        printf("[transcription started %ld]\n", pcmf32.size());
        fflush(stdout);

        pcmf32_new.clear();

        bool speech_detected = false;
        const int chunk_ms = 2000;
        const int chunk_samples = (WHISPER_SAMPLE_RATE * chunk_ms) / 1000;
        const int max_recording_samples = WHISPER_SAMPLE_RATE * 10;

        auto t_last = std::chrono::high_resolution_clock::now();

        while (g_transcribe_mode && is_running) {
            is_running = sdl_poll_events();
            if (!is_running) break;

            auto t_now = std::chrono::high_resolution_clock::now();
            auto t_diff = std::chrono::duration_cast<std::chrono::milliseconds>(t_now - t_last).count();
            auto t_diff_total = std::chrono::duration_cast<std::chrono::milliseconds>(t_now - transcription_started).count();

            if (t_diff_total > n_samples_len) {
                g_transcribe_mode = false;
                printf("[audio too long]\n");
                break;
            }

            if (t_diff < chunk_ms) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }

            audio.get(chunk_ms, pcmf32_new);
            // Wait until the buffer has enough audio
            if (pcmf32_new.size() < chunk_samples) {
                printf("[waiting for more audio, have %zu samples]\n", pcmf32_new.size());
                fflush(stdout);
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }

            printf("[chunk collected: %zu samples]\n", pcmf32_new.size());
            fflush(stdout);

            // Now pcmf32_new has enough audio; perform VAD
            if (::vad_simple(pcmf32_new, WHISPER_SAMPLE_RATE, 700, params.vad_thold, params.freq_thold, false)) {
                printf("[speech detected]\n");
                fflush(stdout);
                speech_detected = true;
            } else if (speech_detected) {
                auto t_now = std::chrono::high_resolution_clock::now();
                auto transcription_end = std::chrono::duration_cast<std::chrono::milliseconds>(t_now - transcription_started).count();
                audio.get(transcription_end + 2000, pcmf32);
                fflush(stdout);
                g_transcribe_mode = false;
                printf("[speech ended]\n");
                printf("[grabbed %ld ms]\n", t_diff + 2000);
                break;
            }

            // Buffer limit handling
            if (pcmf32.size() > max_recording_samples) {
                printf("[erasing buffer %ld > %d]\n", pcmf32.size(), max_recording_samples);
                fflush(stdout);
                pcmf32.erase(pcmf32.begin(), pcmf32.begin() + pcmf32_new.size());
            }

            t_last = t_now;
        }

        // Transcribe pcmf32 buffer
        printf("[transcription finished]\n");
        printf("[transcription contains %zu samples]\n", pcmf32.size());
        fflush(stdout);

        if (params.save_audio) {
            wavWriter.write(pcmf32.data(), pcmf32.size());
        }

        if (whisper_full(ctx, wparams, pcmf32.data(), pcmf32.size()) != 0) {
            fprintf(stderr, "%s: failed to process audio\n", argv[0]);
            return 6;
        }

        const int n_segments = whisper_full_n_segments(ctx);
        printf("[segments found %d]", n_segments);
        for (int i = 0; i < n_segments; ++i) {
            std::string text = whisper_full_get_segment_text(ctx, i);
            int64_t t0 = whisper_full_get_segment_t0(ctx, i);
            int64_t t1 = whisper_full_get_segment_t1(ctx, i);

            double timestamp_start = t0 / 100.0;
            double timestamp_end = t1 / 100.0;

            printf("[transcribed]\n");
            printf(text.c_str());
            printf("\n");
            fflush(stdout);

            add_transcription(text, params.fname_out, params.server_url, timestamp_start, timestamp_end);
            //wavWriter.write(pcmf32.data(), pcmf32.size());
        }

        pcmf32.clear();  // Clear buffer after transcription
        audio.clear();
    }


    audio.pause();

    whisper_print_timings(ctx);
    whisper_free(ctx);
    curl_global_cleanup();

    // Signal the HTTP server to stop.
    if (g_server) {
        g_server->stop();
    }
    // Join the HTTP server thread before exit.
    if (server_thread.joinable()) {
        server_thread.join();
    }

    return 0;
}