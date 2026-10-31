// p100e-server: server_context / server_routes (declared in the fork's tools/server/server-context.h) on the p100e
// engine. This file replaces the fork's server-context.cpp (98ab7e5, MIT): the route handlers (request parsing via
// oaicompat_chat_params_parse / server_schema, response formatting via server-task.cpp, streaming) are the fork's code
// with only the engine-specific parts changed; the slot loop is new and drives server/engine.h:
//   prompt:   the task's tokens in n_ubatch chunks (one chunk per loop iteration, so cancel / other tasks are seen
//             between chunks), the last one returning the next-token logits (or the GPU argmax for greedy requests)
//   generate: one engine decode step over [sampled token, MTP drafts] per iteration; the host sampler (srv_sampler,
//             llama.cpp's common_sampler chain) samples row by row and stops at the first sample that differs from
//             the draft (llama.cpp's sample-and-accept rule, exact for any sampler); keep() rolls back the rest
//   tokens:   process_token / stop strings / limits / partial + final responses are the fork's logic
//   images:   --mmproj loads the vision encoder (src/vision, p100e-vision.dll: the fork's mtmd / clip on one GPU). The
//             HTTP thread only parses the request (the fork's oaicompat_chat_params_parse: data URIs -> files, image
//             parts -> the media marker); the task carries the rendered prompt + files (server_task::cli_prompt /
//             cli_files, cli stays false) and the image_url.detail values. The task loop encodes when the task gets a
//             slot (launch_slot_with_task: serialized with the engine's GPU work; repeated images come from the
//             encoder's embedding cache) and prefills text + image rows through engine::prefill_mm, images whole.
// Not supported yet (errors, not silent): audio / video input, embeddings / rerank, LoRA, slot save / restore,
// n_cmpl > 1, prompt caching between requests (every request starts from an empty sequence).
#include "server-context.h"
#include "server-chat.h"
#include "server-common.h"
#include "server-http.h"
#include "server-task.h"
#include "server-queue.h"
#include "server-schema.h"
#include "server-stream.h"

#include "build-info.h"
#include "common.h"
#include "llama.h"
#include "gguf.h"
#include "log.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include "engine.h"
#include "server_opts.h"
#include "srv_sampler.h"
#include "live_status.h"
#include "ngram_shadow.h"
#ifdef P100E_HAVE_VISION
#include "vision.h"
#endif

#include <algorithm>
#include <atomic>
#include <map>
#include <mutex>
#include <cinttypes>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <exception>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <utility>

constexpr int HTTP_POLLING_SECONDS = 1;

// exit code after a fatal engine error (p100e::engine_fatal_error, engine.h: a P2P allreduce timed out and the engine
// is unusable): the supervisor restarts the process
constexpr int P100E_EXIT_ENGINE_FATAL = 3;

// image_url.detail of every image of the request being handled on this HTTP thread, in file order (set by the chat
// completions route right before handle_completions_impl, which takes it; the parse drops the field)
static thread_local std::vector<std::string> p100e_tls_image_detail;

// per-position MTP draft counters (/metrics p100e:spec_drafted_posN_total / spec_accepted_posN_total), whole process
constexpr int P100E_POS_N = 8;
static std::atomic<uint64_t> p100e_pos_drafted[P100E_POS_N];
static std::atomic<uint64_t> p100e_pos_accepted[P100E_POS_N];

// /metrics: the prompt cache's counters, written by the task loop when it answers a metrics task (Prometheus text)
// n-gram shadow telemetry (ngram_shadow.h): request class hint per completion id (set on the HTTP thread, read at release)
static thread_local std::string p100e_tls_class_hint;
static std::mutex p100e_ng_hint_mx;
static std::map<std::string, std::string> p100e_ng_hint;

static std::mutex p100e_cache_metrics_mx;
static std::string p100e_cache_metrics;

// the image_url parts' detail values (OpenAI chat format), in the order oaicompat_chat_params_parse collects the files
static std::vector<std::string> p100e_image_details(const json & body) {
    std::vector<std::string> out;
    if (!body.contains("messages") || !body.at("messages").is_array()) {
        return out;
    }
    for (const auto & msg : body.at("messages")) {
        if (!msg.is_object() || !msg.contains("content") || !msg.at("content").is_array()) {
            continue;
        }
        for (const auto & p : msg.at("content")) {
            if (p.is_object() && json_value(p, "type", std::string()) == "image_url") {
                const json iu = json_value(p, "image_url", json::object());
                out.push_back(iu.is_object() ? json_value(iu, "detail", std::string()) : std::string());
            }
        }
    }
    return out;
}

// the same for a /v1/responses body (input[].content[] input_image parts carry "detail" beside "image_url"; the fork's
// conversion to chat format drops it). Empty when the count differs from the converted body's image parts (then
// every image gets the default cap, as before).
static std::vector<std::string> p100e_responses_image_details(const json & body, const json & converted) {
    std::vector<std::string> out;
    if (!body.contains("input") || !body.at("input").is_array()) {
        return out;
    }
    for (const auto & item : body.at("input")) {
        if (!item.is_object() || !item.contains("content") || !item.at("content").is_array()) {
            continue;
        }
        for (const auto & p : item.at("content")) {
            if (p.is_object() && json_value(p, "type", std::string()) == "input_image") {
                out.push_back(p.contains("detail") && p.at("detail").is_string() ? p.at("detail").get<std::string>() : std::string());
            }
        }
    }
    if (out.size() != p100e_image_details(converted).size()) {
        out.clear();
    }
    return out;
}

// --p100e-no-remote-images: refuse media parts (chat format, after the responses / anthropic conversion) whose url is
// fetched by the server (http / https: the fork's handle_media downloads any URL, a blind SSRF from the LAN); data: URIs
// and raw base64 still work
static void p100e_check_remote_media(const json & body) {
    if (!g_p100e_opts.no_remote_images || !body.contains("messages") || !body.at("messages").is_array()) {
        return;
    }
    for (const auto & msg : body.at("messages")) {
        if (!msg.is_object() || !msg.contains("content") || !msg.at("content").is_array()) {
            continue;
        }
        for (const auto & p : msg.at("content")) {
            if (!p.is_object()) {
                continue;
            }
            const std::string type = json_value(p, "type", std::string());
            if (type != "image_url" && type != "input_audio" && type != "input_video") {
                continue;
            }
            const json m = p.contains(type) && p.at(type).is_object() ? p.at(type) : json::object();
            for (const char * k : { "url", "data" }) {
                if (m.contains(k) && m.at(k).is_string() && string_starts_with(m.at(k).get<std::string>(), "http")) {
                    throw std::invalid_argument("remote media URLs are disabled on this server (--p100e-no-remote-images): "
                                                "send the image as a data: URI");
                }
            }
        }
    }
}

// log a fatal engine error and exit at once. The engine stays unusable, so failing the request and carrying on would
// only fail every later request too; in-flight requests see their connections drop. std::_Exit: no static
// destructors / CUDA teardown while HTTP threads and the broken engine are still around (the log worker is drained
// first so the line reaches the log file).
[[noreturn]] static void p100e_engine_fatal(const char * what) {
    LOG_ERR("srv  p100e: FATAL engine error, exiting with code %d so the supervisor restarts the server: %s\n",
            P100E_EXIT_ENGINE_FATAL, what);
    common_log_pause(common_log_main());
    fprintf(stderr, "p100e-server: FATAL engine error (exit %d): %s\n", P100E_EXIT_ENGINE_FATAL, what);
    fflush(nullptr);
    std::_Exit(P100E_EXIT_ENGINE_FATAL);
}

// GGUF metadata of a vocab-only model: libllama skips the hyperparameters then (n_ctx_train, n_embd and the file type
// read 0 / "all F32"), but the key / value table is loaded, so read them from there
static int64_t p100e_meta_int(const llama_model * model, const std::string & key, int64_t def) {
    char buf[64] = {0};
    if (llama_model_meta_val_str(model, key.c_str(), buf, sizeof(buf)) > 0) {
        char * end = nullptr;
        const long long v = strtoll(buf, &end, 10);
        if (end && end != buf) {
            return v;
        }
    }
    return def;
}

static std::string p100e_meta_arch(const llama_model * model) {
    char buf[64] = {0};
    return llama_model_meta_val_str(model, "general.architecture", buf, sizeof(buf)) > 0 ? std::string(buf) : std::string();
}

// the vision encoder will be the p100e-native one (src/nvit, vision_opts::native = -1): P100E_NVIT not "0" and the
// mmproj's projector is the one nvit implements. Decided before the engine loads, because the engine's KV pool keeps
// the encoder's VRAM free: mtmd needs ~1 GB on one GPU, nvit ~30 MB per GPU plus scratch borrowed from the prefill ring.
static bool p100e_expect_native_vision(const std::string & mmproj) {
    const char * env = getenv("P100E_NVIT");
    if (env && atoi(env) == 0) return false;
    gguf_init_params gp = { /*no_alloc =*/ true, /*ctx =*/ nullptr };
    gguf_context * g = gguf_init_from_file(mmproj.c_str(), gp);
    if (!g) return false;
    const int64_t k = gguf_find_key(g, "clip.projector_type");
    const bool ok = k >= 0 && gguf_get_kv_type(g, k) == GGUF_TYPE_STRING &&
                    std::string(gguf_get_val_str(g, k)) == "qwen3vl_merger";
    gguf_free(g);
    return ok;
}

enum slot_state {
    SLOT_STATE_IDLE,
    SLOT_STATE_STARTED,            // task assigned, prompt not started
    SLOT_STATE_PROCESSING_PROMPT,
    SLOT_STATE_GENERATING,
};

static const char * slot_state_name(int s) {
    static const char * names[] = {"idle", "started", "processing_prompt", "generating"};
    return s >= 0 && s < 4 ? names[s] : "?";
}

#ifndef P100E_GIT_COMMIT
#define P100E_GIT_COMMIT "unknown"
#endif
// /props build_info: this tree's commit, then the fork's (libllama / server code) build
static std::string p100e_build_info() {
    return std::string("p100e-") + P100E_GIT_COMMIT + " (fork " + llama_build_info() + ")";
}

struct server_slot {
    int id = 0;
    int n_ctx = 0;
    int seq = -1;   // engine sequence while processing

    std::unique_ptr<const server_task> task;
    std::unique_ptr<const server_task> task_prev;   // for /slots after the task finished

    int64_t t_last_used = -1;
    int32_t n_predict_max = -1;

    size_t last_nl_pos = 0;
    std::string generated_text;
    llama_tokens generated_tokens;   // return_tokens
    llama_tokens emitted;            // every emitted token (--p100e-print-ids)
    size_t n_sent_text = 0;
    std::vector<completion_token_output> generated_token_probs;

    bool has_next_token = true;
    bool has_new_line   = false;
    bool truncated      = false;
    stop_type stop = STOP_TYPE_NONE;
    std::string stopping_word;

    slot_state state = SLOT_STATE_IDLE;

    llama_tokens prompt;      // text tokens of the task
    std::vector<p100e::prompt_piece> mm;   // multimodal prompt (text and image pieces); empty: text only (prompt)
    int n_mm_rows = 0;        // rows of mm (text tokens + image rows)
    size_t mm_next = 0;       // next piece of mm to feed
    int mm_off = 0;           // tokens of mm[mm_next] fed already (text pieces are split at n_ubatch)
    std::vector<uint64_t> mm_keys;   // prompt-cache key per row of mm (vision mm_row_keys: image rows by image hash)
    int n_prompt_done = 0;    // prompt tokens (rows) fed to the engine
    int n_fed = 0;            // tokens in the engine state (prompt + accepted generated tokens)

    srv_sampler_ptr smpl;
    p100e_ng::Shadow ng;                      // n-gram shadow telemetry (pure observation)
    llama_token sampled = LLAMA_TOKEN_NULL;   // emitted, not fed yet
    llama_tokens drafts;                      // MTP drafts for the next step
    std::vector<float> draft_p;               // engine draft_probs of `drafts` (cumulative), empty if unknown
    int n_draft_trimmed = 0;                  // drafts made but cut by the step's row cap (never verified)
    bool draft_pending = false;               // finish_step done: drafts come from the step's draft_batch
    bool all_greedy = true;                   // every step took the GPU argmax (--p100e-refcheck applies)

    server_slot_stats stats;
    std::vector<uint64_t> n_accepted_per_pos;

    int64_t t_print_last = 0;
    int32_t n_gen_last = 0;

    // the last finished task's numbers: an idle slot's /slots entry keeps describing it until the next task
    struct {
        server_slot_stats stats;
        int n_prompt_rows = 0;
        int n_fed = 0;
        int32_t n_remain = -1;
        bool has_new_line = false;
        int64_t t_end = 0;
    } prev;

    void reset() {
        if (task) {
            prev = {stats, prompt_rows(), n_fed, n_remaining(), has_new_line, ggml_time_us()};
        }
        last_nl_pos    = 0;
        generated_text = "";
        has_next_token = true;
        has_new_line   = false;
        truncated      = false;
        stop           = STOP_TYPE_NONE;
        stopping_word  = "";
        n_sent_text    = 0;
        generated_tokens.clear();
        generated_token_probs.clear();
        emitted.clear();
        prompt.clear();
        mm.clear();
        n_mm_rows = 0;
        mm_next = 0;
        mm_off = 0;
        mm_keys.clear();
        n_prompt_done = 0;
        n_fed = 0;
        sampled = LLAMA_TOKEN_NULL;
        drafts.clear();
        draft_p.clear();
        n_draft_trimmed = 0;
        all_greedy = true;
        smpl.reset();
        ng.clear();
        task_prev = std::move(task);
        task.reset();
        stats = {};
        n_accepted_per_pos.clear();
        n_predict_max = -1;
    }

    // KV rows of the prompt (text tokens + image rows)
    int prompt_rows() const {
        return mm.empty() ? (int) prompt.size() : n_mm_rows;
    }

    int32_t n_remaining() const {
        return n_predict_max == -1 ? -1 : n_predict_max - (int32_t) stats.n_gen;
    }

    bool has_budget() const {
        return n_predict_max == -1 || n_remaining() > 0;
    }

    bool is_processing() const {
        return state != SLOT_STATE_IDLE;
    }

    void add_token(const completion_token_output & token) {
        generated_token_probs.push_back(token);
    }

    size_t find_stopping_strings(const std::string & text, const size_t last_token_size, bool is_full_stop) {
        size_t stop_pos = std::string::npos;

        for (const std::string & word : task->params.antiprompt) {
            size_t pos;

            if (is_full_stop) {
                const size_t tmp      = word.size() + last_token_size;
                const size_t from_pos = text.size() > tmp ? text.size() - tmp : 0;

                pos = text.find(word, from_pos);
            } else {
                pos = string_find_partial_stop(text, word);
            }

            if (pos != std::string::npos && (stop_pos == std::string::npos || pos < stop_pos)) {
                if (is_full_stop) {
                    stop           = STOP_TYPE_WORD;
                    stopping_word  = word;
                    has_next_token = false;
                }
                stop_pos = pos;
            }
        }

        return stop_pos;
    }

    void print_timings_tg() {
        if (stats.n_gen < 100) {
            return;
        }

        const int64_t t_now = ggml_time_us();

        if (t_now - t_print_last < 3*1000*1000) {
            return;
        }

        const double n_gen_second     = stats.n_gen_tps();
        const double n_gen_second_win = 1e6 / (t_now - t_print_last) * (stats.n_gen - n_gen_last);

        t_print_last = t_now;
        n_gen_last = stats.n_gen;

        SLT_INF(*this, "n_gen = %6d, tg = %6.2f t/s, tg_3s = %6.2f t/s\n", (int) stats.n_gen, n_gen_second, n_gen_second_win);
    }

    void print_timings_pp() const {
        const double t_prompt_total = (ggml_time_us() - stats.t_start) / 1000.0;

        if (t_prompt_total < 3000.0) {
            return;
        }

        const double n_prompt_second = t_prompt_total > 0 ? 1e3 / t_prompt_total * stats.n_prompt_processed : 0.0;
        const double f_progress = task->n_tokens() > 0 ? (double) n_prompt_done / task->n_tokens() : 0.0;

        SLT_INF(*this, "prompt processing, n_tokens = %6d, progress = %.2f, t = %6.2f s / %.2f tokens per second\n",
                (int) stats.n_prompt_processed, f_progress, t_prompt_total / 1e3, n_prompt_second);
    }

    // monitors parse these lines: keep the format (and the function name,
    // which the SLT_* macros print as "print_timing")
    void print_timings() const {
        const double t_prompt_total = stats.t_prompt_ms();
        const double t_gen_total    = stats.t_gen_ms();

        const double t_prompt        = stats.t_prompt_per_token_ms();
        const double n_prompt_second = stats.n_prompt_tps();

        const double t_gen        = stats.t_gen_per_token_ms();
        const double n_gen_second = stats.n_gen_tps();

        SLT_INF(*this,
                "prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
                t_prompt_total, (int) stats.n_prompt_processed, t_prompt, n_prompt_second);

        SLT_INF(*this,
                "       eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
                t_gen_total, (int) stats.n_gen, t_gen, n_gen_second);

        SLT_INF(*this,
                "      total time = %10.2f ms / %5d tokens\n",
                t_prompt_total + t_gen_total, (int) (stats.n_prompt_processed + stats.n_gen));

        const int32_t n_draft_total       = stats.n_draft_tokens;
        const int32_t n_draft_accepted    = stats.n_draft_accepted;
        const int32_t n_draft_verif_steps = stats.n_draft_verif_steps;

        if (n_draft_total > 0) {
            const float  draft_ratio  = (float) n_draft_accepted / n_draft_total;
            const double mean_acc_len = n_draft_verif_steps > 0 ? 1.0 + (double) n_draft_accepted / (double) n_draft_verif_steps : 1.0;

            SLT_INF(*this,
                    "draft acceptance = %0.5f (%5d accepted / %5d generated), mean len = %5.2f\n",
                    draft_ratio, n_draft_accepted, n_draft_total, mean_acc_len);
        }
        if (n_draft_trimmed > 0) {   // "generated" above = verified drafts; these were made but never verified
            SLT_INF(*this, "draft stats: verified %d accepted %d trimmed by row cap %d (made %d)\n",
                    n_draft_total, n_draft_accepted, n_draft_trimmed, n_draft_total + n_draft_trimmed);
        }
    }

    // the /slots entry (the fork's fields) plus p100e live fields. n_prompt_tokens = the whole prompt (text tokens + image
    // rows, known when the task gets the slot), n_prompt_tokens_processed = rows prefilled so far (cached rows
    // excluded), next_token[0].n_decoded = tokens generated so far. An idle slot reports its last task (has_next_token
    // false). Built by the task loop (live status snapshot, live_status.h).
    json to_json(const llama_vocab * vocab, bool only_metrics, bool speculative) const {
        json res = {
            {"id",            id},
            {"n_ctx",         n_ctx},
            {"speculative",   speculative},
            {"is_processing", is_processing()},
            {"state",         slot_state_name(state)},
        };

        const bool live = task != nullptr;
        const auto & ptask = live ? task : task_prev;
        const server_slot_stats & st = live ? stats : prev.stats;

        if (ptask) {
            res["id_task"] = ptask->id;
        }
        res["n_prompt_tokens"]           = (int32_t) (live ? prompt_rows() : prev.n_prompt_rows);
        res["n_prompt_tokens_processed"] = st.n_prompt_processed;
        res["n_prompt_tokens_cache"]     = st.n_prompt_cached;
        if (ptask) {
            res["params"] = ptask->params.to_json(only_metrics);
        }
        res["next_token"] = json::array({
            {
                {"has_next_token", live ? has_next_token : false},
                {"has_new_line",   live ? has_new_line : prev.has_new_line},
                {"n_remain",       live ? n_remaining() : prev.n_remain},
                {"n_decoded",      st.n_gen},
            }
        });

        // timestamps in unix ms (0: not reached yet)
        const double t_prompt_ms = st.t_prompt_ms();
        const double t_gen_ms    = st.t_gen_us() / 1e3;
        res["p100e"] = {
            {"n_prompt_done",    live ? n_prompt_done : prev.n_prompt_rows},   // prompt rows in the engine incl. cached
            {"n_ctx_used",       live ? n_fed : prev.n_fed},
            {"t_start_ms",       p100e_unix_ms(st.t_start)},
            {"t_prompt_last_ms", p100e_unix_ms(st.t_prompt_last)},
            {"t_gen_last_ms",    p100e_unix_ms(st.t_gen_last)},
            {"t_end_ms",         live ? 0 : p100e_unix_ms(prev.t_end)},
            {"t_prompt_ms",      t_prompt_ms},
            {"t_gen_ms",         t_gen_ms},
            {"prompt_tps",       t_prompt_ms > 0 ? 1e3 * st.n_prompt_processed / t_prompt_ms : 0.0},
            {"gen_tps",          t_gen_ms > 0 ? 1e3 * st.n_gen / t_gen_ms : 0.0},
            {"n_draft_tokens",   st.n_draft_tokens},
            {"n_draft_accepted", st.n_draft_accepted},
        };

        if (!only_metrics && ptask) {
            res["prompt"] = common_detokenize(vocab, ptask->tokens.get_text_tokens(), true);
            res["generated"] = generated_text;
        }

        return res;
    }
};

//
// server_context_impl
//

struct server_context_impl {
    friend struct server_context;

public:
    // tokenizer / GGUF metadata only (vocab_only load): never runs a forward
    llama_model * model_tgt = nullptr;

    mtmd_context * mctx = nullptr;   // never set: images go through the p100e vision encoder (venc)
#ifdef P100E_HAVE_VISION
    std::unique_ptr<p100e::vision_encoder> venc;   // --mmproj
#endif
    bool has_vision() const {
#ifdef P100E_HAVE_VISION
        return venc != nullptr;
#else
        return false;
#endif
    }
    mtmd_helper_init_opt init_opt = mtmd_helper_init_opt_default();
    const llama_vocab * vocab = nullptr;

    server_queue    queue_tasks;
    server_response queue_results;

    server_chat_params chat_params;

    server_state_callback_t callback_state = [](server_state, json) -> void {};

    server_context_impl() = default;

    ~server_context_impl() {
#ifdef P100E_HAVE_VISION
        venc.reset();
#endif
        eng.reset();
        model_holder.reset();
    }

    server_metrics get_metrics() const {
        return metrics;
    }

    void reset_metrics_bucket() {
        metrics.reset_bucket();
    }

private:
    common_params params_base;
    llama_model_ptr model_holder;
    std::unique_ptr<p100e::engine> eng;

    int n_ctx_slot_v = 0;
    std::vector<server_slot> slots;
    server_metrics metrics;

    json json_ui_settings = json::object();

    std::string model_name;
    std::set<std::string> model_aliases;
    std::set<std::string> model_tags;

    uint64_t n_params = 0, model_size = 0;
    int n_ctx_train = 0, n_embd = 0;
    std::string ftype_name;
    int n_completed = 0;   // completion tasks finished (--p100e-max-requests)

    bool can_speculate() const { return eng && eng->info().n_draft_max > 0; }

    bool load_model(common_params & params) {
        params_base = params;

        SRV_INF("loading model '%s'\n", params.model.path.c_str());

        // tokenizer + metadata: libllama vocab-only (no tensors, no backend buffers)
        {
            llama_model_params mp = llama_model_default_params();
            mp.vocab_only = true;
            llama_model * m = llama_model_load_from_file(params.model.path.c_str(), mp);
            if (m == nullptr) {
                SRV_ERR("failed to load the vocabulary of '%s'\n", params.model.path.c_str());
                return false;
            }
            model_holder.reset(m);
            model_tgt = m;
        }
        vocab = llama_model_get_vocab(model_tgt);

        // the model's recommended sampling defaults (general.sampling.*), EOG biases (common_init_result)
        srv_init_sampler_from_model(model_tgt, params_base.sampling);
        if (params_base.sampling.ignore_eos && llama_vocab_eos(vocab) == LLAMA_TOKEN_NULL) {
            SRV_WRN("%s", "vocab does not have an EOS token, ignoring --ignore-eos\n");
            params_base.sampling.ignore_eos = false;
        }
        params_base.sampling.logit_bias_eog.clear();
        for (llama_token i = 0; i < llama_vocab_n_tokens(vocab); i++) {
            if (llama_vocab_is_eog(vocab, i)) {
                params_base.sampling.logit_bias_eog.push_back({i, -INFINITY});
            }
        }
        if (params_base.sampling.ignore_eos) {
            params_base.sampling.logit_bias.insert(params_base.sampling.logit_bias.end(),
                params_base.sampling.logit_bias_eog.begin(), params_base.sampling.logit_bias_eog.end());
        }

        {
            const std::string arch = p100e_meta_arch(model_tgt);
            n_ctx_train = (int) p100e_meta_int(model_tgt, arch + ".context_length", llama_model_n_ctx_train(model_tgt));
            n_embd      = (int) p100e_meta_int(model_tgt, arch + ".embedding_length", llama_model_n_embd(model_tgt));
            const int64_t ft = p100e_meta_int(model_tgt, "general.file_type", -1);
            ftype_name = ft >= 0 ? llama_ftype_name((llama_ftype) ft) : llama_ftype_name(llama_model_ftype(model_tgt));
            // tensor statistics: libllama counts them from the GGUF tensor table even for a vocab-only load
            n_params   = llama_model_n_params(model_tgt);
            model_size = llama_model_size(model_tgt);
        }

        // per-slot context, as llama-server (non-unified KV): -c is the total over -np slots, 0 = the training context
        const int n_par = std::max(1, params_base.n_parallel);
        params_base.n_parallel = n_par;
        const int n_ctx_total = params_base.n_ctx > 0 ? params_base.n_ctx : n_ctx_train;
        n_ctx_slot_v = std::min(n_ctx_total / n_par, n_ctx_train);
        if (n_ctx_total / n_par > n_ctx_train) {
            SRV_WRN("the slot context (%d) exceeds the training context of the model (%d) - capping\n",
                    n_ctx_total / n_par, n_ctx_train);
        }

        const auto & o = g_p100e_opts;
        bool expect_native_vision = false;   // the KV pool was sized for the native vision encoder (see below)
        if (o.no_engine) {
            SRV_WRN("%s", "--p100e-no-engine: text endpoints only, completions return an error\n");
        } else {
            p100e::engine_tp_params ep;
            ep.model_path = params_base.model.path;
            ep.devices    = o.devices;
            ep.split      = o.split;
            ep.n_ctx      = n_ctx_slot_v;
            ep.n_ubatch   = params_base.n_ubatch;
            ep.mtp_k      = o.mtp_k;
            ep.mtp_tau    = o.mtp_tau;
            ep.n_seq      = n_par;
            ep.cache      = o.cache;
            ep.cache_resident = o.cache_resident;
            ep.host_budget_mb = o.host_budget_mb;
            ep.ckpt_budget_mb = o.ckpt_budget_mb;
            if (o.pool_tokens != 0) {
                ep.pool_tokens = o.pool_tokens;
            }
            ep.margin_mb = o.margin_mb;
            // native vision encoder: ~30 MB per GPU + scratch lent from the prefill ring, so keep only a little VRAM
            // free on every GPU instead of mtmd's 1024 MiB on the last one (the pool's 512 MiB margin stays)
            if (!params_base.mmproj.path.empty() && params_base.mmproj_use_gpu &&
                p100e_expect_native_vision(params_base.mmproj.path)) {
                ep.reserve_mb.assign(o.devices.size(), 128);
                expect_native_vision = true;
            }
            {   // message starts: the prompt cache's checkpoint positions
                const llama_tokens ims = common_tokenize(vocab, "<|im_start|>", false, true);
                if (ims.size() == 1) {
                    ep.split_tokens.push_back(ims[0]);
                } else {
                    SRV_WRN("%s", "no <|im_start|> token: prompt-cache checkpoints only at the end of a turn\n");
                }
            }
            if (!o.kv_type.empty()) {
                ep.kv_type = o.kv_type;
            } else {
                switch (params_base.cache_type_k) {
                    case GGML_TYPE_Q4_0: ep.kv_type = "q4_0"; break;
                    case GGML_TYPE_Q8_0: ep.kv_type = "q8_0"; break;
                    default:             ep.kv_type = "q8_0"; break;   // p100e default (llama.cpp's is f16)
                }
            }
            const std::string arch_e = p100e_meta_arch(model_tgt);
            const bool use_fn = o.engine.empty() ? arch_e == "qwen4exp" : o.engine == "fn";
            if (use_fn && n_par != 1) {
                SRV_ERR("the Flash-Next engine serves one slot: run with -np 1 (got %d)\n", n_par);
                return false;
            }
            try {
                const int64_t t0 = ggml_time_us();
                if (use_fn) {
                    p100e::engine_fn_params fp;
                    fp.model_path   = params_base.model.path;
                    fp.devices      = o.devices;
                    fp.n_ctx        = n_ctx_slot_v;
                    fp.max_t        = o.fn_max_t > 0 ? o.fn_max_t : params_base.n_ubatch;
                    fp.hot_per_rank = o.fn_hot;
                    fp.mtp_k        = o.mtp_k;
                    fp.pipe_nodes   = o.fn_pipe_nodes;
                    fp.pipe_layers  = o.fn_pipe_layers;
                    fp.rank_slots   = o.fn_slots;
                    eng = p100e::make_engine_fn(fp);
                } else
                eng = p100e::make_engine_tp(ep);
                SRV_INF("engine: %s, loaded in %.1f s\n", eng->info().desc.c_str(), (ggml_time_us() - t0) / 1e6);
                eng->print_mem("after load");
                if (n_params == 0) {
                    n_params   = eng->info().n_params;
                    model_size = eng->info().model_bytes;
                }
                if (eng->info().n_vocab != llama_vocab_n_tokens(vocab)) {
                    SRV_ERR("engine vocab %d != tokenizer vocab %d\n", eng->info().n_vocab, llama_vocab_n_tokens(vocab));
                    return false;
                }
            } catch (const std::exception & e) {
                SRV_ERR("failed to load the engine: %s\n", e.what());
                return false;
            }
        }

        if (!params_base.mmproj.path.empty()) {
#ifdef P100E_HAVE_VISION
            // after the engine (tp_model enables P2P first; the encoder's ggml-cuda must not, vision.h)
            p100e::vision_opts vo;
            vo.mmproj     = params_base.mmproj.path;
            vo.text_model = params_base.model.path;
            vo.device     = !params_base.mmproj_use_gpu ? -1
                          : o.vision_device >= 0 ? o.vision_device
                          : (o.no_engine || o.devices.empty()) ? -1 : o.devices.back();
            vo.min_tokens = params_base.image_min_tokens > 0 ? params_base.image_min_tokens : 1024;
            vo.max_tokens = params_base.image_max_tokens > 0 ? params_base.image_max_tokens : 2048;   // default cap 2048
            vo.warmup     = params_base.warmup && vo.device >= 0;   // reserves the GPU compute buffer once
            if (!o.no_engine && !o.devices.empty()) {
                // nvit: the engine's GPUs (sequence-parallel over both); it splits over 1 or 2, so with 4 engine GPUs the
                // first two (its scratch is lent per device from those ranks' prefill rings)
                vo.native_devices.assign(o.devices.begin(),
                                         o.devices.begin() + (std::ptrdiff_t) std::min<size_t>(2, o.devices.size()));
            }
            if (params_base.cpuparams.n_threads > 0) {
                vo.n_threads = params_base.cpuparams.n_threads;
            }
            try {
                const int64_t t0 = ggml_time_us();
                venc = std::make_unique<p100e::vision_encoder>(vo);
                SRV_INF("vision: encoder on %s, %s, image tokens %d..%d, n_embd %d, DLLs %s, loaded in %.1f s%s\n",
                        vo.device >= 0 ? ("CUDA" + std::to_string(vo.device)).c_str() : "CPU",
                        params_base.mmproj.path.c_str(), venc->min_tokens(), venc->max_tokens(), venc->n_embd(),
                        venc->dll_dir().c_str(), (ggml_time_us() - t0) / 1e6,
                        venc->cpu_fallback() ? " - WARNING: part of the encoder falls back to the CPU" : "");
                for (const auto & w : venc->warnings()) {
                    SRV_WRN("vision: %s\n", w.c_str());
                }
                SRV_INF("vision: %s encoder\n", venc->native() ? "p100e-native (nvit, weights in host RAM)" : "mtmd");
                if (expect_native_vision && !venc->native()) {
                    SRV_WRN("%s", "vision: the native encoder did not start, mtmd runs instead, but the KV pool was sized "
                                  "for the native one (128 MiB reserve per GPU): large images may run out of VRAM\n");
                }
                if (eng) {
                    if (venc->n_embd() != eng->info().n_embd) {
                        SRV_ERR("vision: the mmproj's embedding width %d != the model's %d\n", venc->n_embd(), eng->info().n_embd);
                        return false;
                    }
                    // native encoder: activations in the engine's prefill ring while it is idle (as p100e-run does;
                    // P100E_NVIT_BORROW=0: own scratch). Busy ring -> the encoder's own scratch for that encode.
                    const char * nb = getenv("P100E_NVIT_BORROW");
                    if (venc->native() && eng->can_lend() && !(nb && atoi(nb) == 0)) {
                        p100e::engine * e = eng.get();
                        venc->set_scratch_source([e]() {
                            std::vector<p100e::vision_encoder::scratch_region> v;
                            for (const auto & s : e->lend_scratch()) v.push_back({s.dev, s.p, s.bytes, s.lease});
                            return v;
                        });
                        SRV_INF("%s", "vision: native encoder borrows the prefill ring as scratch\n");
                    }
                    eng->print_mem("after the vision encoder");
                }
            } catch (const std::exception & e) {
                SRV_ERR("failed to load the vision encoder: %s\n", e.what());
                return false;
            }
#else
            SRV_ERR("%s", "--mmproj: this p100e-server build has no vision encoder (P100E_VISION off)\n");
            return false;
#endif
        }

        SRV_INF("initializing slots, n_slots = %d, n_ctx_slot = %d, engine sequences = %d\n",
                n_par, n_ctx_slot_v, eng ? eng->info().n_seq_max : 0);

        slots.clear();
        for (int i = 0; i < n_par; i++) {
            slots.emplace_back();
            slots.back().id = i;
            slots.back().n_ctx = n_ctx_slot_v;
        }

        if (!params_base.model_alias.empty()) {
            model_name = *params_base.model_alias.begin();
        } else if (!params_base.model.get_name().empty()) {
            model_name = params_base.model.get_name();
        } else {
            model_name = std::filesystem::path(params_base.model.path).filename().string();
        }
        model_aliases = params_base.model_alias;
        model_tags    = params_base.model_tags;

        params = params_base;

        return init();
    }

    bool init() {
        queue_tasks.on_new_task([this](server_task && task, bool is_yielding) {
            const bool status = task.type == SERVER_TASK_TYPE_METRICS || task.type == SERVER_TASK_TYPE_SLOT_GET ||
                                task.type == SERVER_TASK_TYPE_NEXT_RESPONSE;
            const bool r = process_single_task(std::move(task), is_yielding);
            if (!status) {
                publish_live(true);   // e.g. a new task's whole prompt size, before its first prompt chunk runs
            }
            return r;
        });
        queue_tasks.on_update_slots([this]() {
            update_slots();
            publish_live(false);
        });

        metrics.init();

        {
            const std::string & cfg = params_base.ui_config_json;
            if (!cfg.empty()) {
                try {
                    json_ui_settings = json::parse(cfg);
                } catch (const std::exception & e) {
                    SRV_ERR("%s: failed to parse UI config: %s\n", __func__, e.what());
                    return false;
                }
            }
        }

        // chat template params (fork server-context.cpp init())
        {
            common_chat_templates_ptr chat_templates;
            bool enable_thinking = false;

            try {
                chat_templates = common_chat_templates_init(model_tgt, params_base.chat_template);

                SRV_TRC("%s: chat template, example_format: '%s'\n", __func__,
                    common_chat_format_example(chat_templates.get(), params_base.use_jinja, params_base.default_template_kwargs).c_str());

                const bool template_supports_thinking = params_base.use_jinja && common_chat_templates_support_enable_thinking(chat_templates.get());
                enable_thinking = params_base.enable_reasoning != 0 && template_supports_thinking;
                SRV_TRC("%s: chat template, thinking = %d\n", __func__, enable_thinking);
            } catch (const std::exception & e) {
                SRV_ERR("%s: chat template parsing error: %s\n", __func__, e.what());
                return false;
            }

            chat_params = {
                /* use_jinja             */ params_base.use_jinja,
                /* prefill_assistant     */ params_base.prefill_assistant,
                /* reasoning_format      */ params_base.reasoning_format,
                /* chat_template_kwargs  */ params_base.default_template_kwargs,
                /* tmpls                 */ std::move(chat_templates),
                /* allow_image           */ has_vision(),
                /* allow_audio           */ false,
                /* allow_video           */ false,
                /* enable_thinking       */ enable_thinking,
                /* reasoning_budget      */ params_base.sampling.reasoning_budget_tokens,
                /* reasoning_budget_msg  */ params_base.sampling.reasoning_budget_message,
                /* media_path            */ params_base.media_path,
                /* force_pure_content    */ params_base.force_pure_content_parser
            };

            {
                auto caps = common_chat_templates_get_caps(chat_params.tmpls.get());
                auto it = params_base.default_template_kwargs.find("preserve_reasoning");
                bool supported = caps.at("supports_preserve_reasoning");
                bool specified = params_base.preserve_reasoning_specified;
                bool enabled = it != params_base.default_template_kwargs.end() && it->second == "true";
                if (supported && !specified) {
                    SRV_WRN("%s", "chat template supports preserving reasoning, it is enabled by default (may use more tokens, disable via --no-reasoning-preserve)\n");
                }
                if (supported && !enabled) {
                    SRV_INF("%s", "chat template supports preserving reasoning, consider enabling it via --reasoning-preserve\n");
                }
                if (!supported && specified && enabled) {
                    SRV_WRN("%s", "chat template does NOT support preserving reasoning, --reasoning-preserve has no effect\n");
                }
            }
        }

        return true;
    }

    // slots holding (or about to open) an engine sequence
    int n_seq_busy() const {
        int n = 0;
        for (const auto & s : slots) n += s.is_processing();
        return n;
    }

    server_slot * get_slot_by_id(int id_slot) {
        id_slot = id_slot % slots.size();
        for (server_slot & slot : slots) {
            if (slot.id == id_slot) {
                return &slot;
            }
        }
        return nullptr;
    }

    server_slot * get_slot_by_cmpl_id(const std::string & cmpl_id) {
        if (cmpl_id.empty()) {
            return nullptr;
        }
        for (server_slot & slot : slots) {
            if (slot.is_processing() && slot.task && slot.task->params.oaicompat_cmpl_id == cmpl_id) {
                return &slot;
            }
        }
        return nullptr;
    }

    // an idle slot (requested id, else least recently used); nullptr also while the engine has no free sequence
    server_slot * get_available_slot(const server_task & task) {
        if (!eng || n_seq_busy() >= eng->info().n_seq_max) {
            return nullptr;
        }
        server_slot * ret = nullptr;
        if (task.id_slot != -1) {
            ret = get_slot_by_id(task.id_slot);
            if (ret && ret->is_processing()) {
                return nullptr;
            }
            if (ret) {
                SLT_INF(*ret, "selected slot by id (%d)\n", task.id_slot);
            }
            return ret;
        }
        int64_t t_last = -1;
        for (server_slot & slot : slots) {
            if (slot.is_processing()) {
                continue;
            }
            if (!ret || slot.t_last_used <= t_last) {
                t_last = slot.t_last_used;
                ret = &slot;
            }
        }
        if (ret != nullptr) {
            SLT_INF(*ret, "selected slot by LRU, t_last = %" PRId64 "\n", t_last);
        }
        return ret;
    }

    // a task with images (handle_completions_impl: cli_prompt + cli_files, the details in slot_action.filename): encode
    // them (vision encoder: cache first) and build the prompt pieces; task.tokens becomes the row list (image rows
    // LLAMA_TOKEN_NULL, as llama-server's server_tokens), so n_tokens() / the context check / usage count image rows
    bool build_mm_prompt(server_task & task, std::vector<p100e::prompt_piece> & mm, int & n_rows,
                         std::vector<uint64_t> & keys) {
#ifdef P100E_HAVE_VISION
        if (!venc) {
            send_error(task, "image input needs --mmproj", ERROR_TYPE_NOT_SUPPORTED);
            return false;
        }
        std::vector<std::string> detail = string_split<std::string>(task.slot_action.filename, ',');
        std::vector<p100e::image_input> ins(task.cli_files.size());
        for (size_t i = 0; i < ins.size(); ++i) {
            ins[i].bytes.assign(task.cli_files[i].begin(), task.cli_files[i].end());
            ins[i].max_tokens = venc->detail_max_tokens(i < detail.size() ? detail[i] : std::string());
        }
        std::vector<p100e::mm_piece> pieces;
        const int64_t t0 = ggml_time_us();
        try {
            pieces = venc->build(task.cli_prompt, ins, get_media_marker());
        } catch (const std::exception & e) {
            send_error(task, std::string("image input: ") + e.what(), ERROR_TYPE_INVALID_REQUEST);
            return false;
        }
        llama_tokens rows;
        mm.clear();
        int k = 0;
        for (const auto & p : pieces) {
            p100e::prompt_piece q;
            if (p.image) {
                const auto & e = *p.image;
                q.embd   = std::shared_ptr<const float>(p.image, e.data.data());   // keeps the cache entry alive
                q.nx     = e.nx;
                q.ny     = e.ny;
                q.n_embd = e.n_embd;
                rows.insert(rows.end(), (size_t) e.n_tok(), LLAMA_TOKEN_NULL);
                SRV_INF("image %d: %dx%d px -> %dx%d = %d tokens (cap %d), %s in %.0f ms\n", k, e.src_w, e.src_h, e.nx,
                        e.ny, e.n_tok(), ins[k].max_tokens, p.t.cached ? "cached" : "encoded", p.t.total_ms());
                ++k;
            } else {
                q.tokens = p.tokens;
                rows.insert(rows.end(), p.tokens.begin(), p.tokens.end());
            }
            mm.push_back(std::move(q));
        }
        SRV_INF("multimodal prompt: %zu image(s), %zu rows, built in %.2f s\n", ins.size(), rows.size(),
                (ggml_time_us() - t0) / 1e6);
        n_rows = (int) rows.size();
        keys = p100e::mm_row_keys(pieces);
        task.tokens = server_tokens(rows, false);
        task.cli_files.clear();
        task.cli_prompt.clear();
        return true;
#else
        (void) mm; (void) n_rows; (void) keys;
        send_error(task, "image input is not supported by this p100e-server build", ERROR_TYPE_NOT_SUPPORTED);
        return false;
#endif
    }

    bool launch_slot_with_task(server_slot & slot, server_task && task) {
        const int n_vocab = llama_vocab_n_tokens(vocab);
        std::vector<p100e::prompt_piece> mm;
        int n_mm_rows = 0;
        std::vector<uint64_t> mm_keys;
        if (!task.cli_files.empty() && !build_mm_prompt(task, mm, n_mm_rows, mm_keys)) {
            return false;
        }
        llama_tokens toks = task.tokens.get_text_tokens();
        if (mm.empty() && toks.size() != task.tokens.size()) {
            send_error(task, "multimodal input is not supported yet by p100e-server", ERROR_TYPE_NOT_SUPPORTED);
            return false;
        }
        for (llama_token t : toks) {
            if (t < 0 || t >= n_vocab) {
                send_error(task, "Prompt contains invalid tokens", ERROR_TYPE_INVALID_REQUEST);
                return false;
            }
        }
        if (!task.params.lora.empty()) {
            send_error(task, "LoRA adapters are not supported by p100e-server", ERROR_TYPE_NOT_SUPPORTED);
            return false;
        }

        if (task.need_sampling()) {
            try {
                slot.smpl.reset(srv_sampler_init(model_tgt, task.params.sampling));
            } catch (std::exception & e) {
                std::string err_msg = std::string("Failed to initialize samplers: ") + e.what();
                send_error(task, err_msg, ERROR_TYPE_INVALID_REQUEST);
                return false;
            }
            SLT_TRC(slot, "sampler chain: %s\n", srv_sampler_print(slot.smpl.get()).c_str());
        }

        slot.n_predict_max = task.params.n_predict != -1 ? task.params.n_predict : params_base.n_predict;
        slot.prompt = std::move(toks);
        slot.mm = std::move(mm);
        slot.n_mm_rows = n_mm_rows;
        slot.mm_next = 0;
        slot.mm_off = 0;
        slot.mm_keys = std::move(mm_keys);
        slot.task = std::make_unique<const server_task>(std::move(task));
        slot.state = SLOT_STATE_STARTED;

        SLT_INF(slot, "processing task, is_child = %d\n", slot.task->is_child());
        return true;
    }

    void release(server_slot & slot) {
        if (!slot.is_processing()) {
            return;
        }
        SLT_INF(slot, "stop processing: n_tokens = %d, truncated = %d\n", slot.n_fed, slot.truncated);
        if (g_p100e_opts.print_ids && slot.task) {
            std::string s;
            for (llama_token t : slot.prompt) s += std::to_string(t) + ",";
            if (!s.empty()) s.pop_back();
            SLT_INF(slot, "prompt ids: %s\n", s.c_str());
            s.clear();
            for (llama_token t : slot.emitted) s += " " + std::to_string(t);
            SLT_INF(slot, "generated ids:%s\n", s.c_str());
        }
        ng_report(slot);
        slot.t_last_used = ggml_time_us();
        slot.state = SLOT_STATE_IDLE;
        if (slot.stats.n_gen > 0) {
            metrics_on_prediction(slot);
        }
        if (slot.seq >= 0 && eng) {
            if (g_p100e_opts.refcheck && !slot.emitted.empty() && slot.n_prompt_done == slot.prompt_rows()) {
                refcheck(slot);
            }
            eng->seq_close(slot.seq);
        }
        slot.seq = -1;
        const bool was_completion = slot.task && slot.task->need_sampling();
        slot.reset();
        queue_tasks.pop_deferred_task(slot.id);
        if (was_completion && g_p100e_opts.max_requests > 0 && ++n_completed >= g_p100e_opts.max_requests) {
            SRV_INF("--p100e-max-requests %d reached, terminating\n", g_p100e_opts.max_requests);
            queue_tasks.terminate();
        }
    }

    // --p100e-refcheck: rerun a finished task as plain decoding (prompt forward, then one forward per token with host
    // logits, no MTP) through a fresh sampler built from the task's parameters, and compare the ids. Runs for every
    // task whose sampling is reproducible: temperature <= 0 (greedy, also with a tool-call grammar or a reasoning
    // budget) or a fixed seed. Checks that the server's path - GPU argmax, MTP verify + rollback, sample-and-match,
    // greedy steps cut by the reasoning budget, ubatch chunking - emits exactly what plain decoding emits. With a
    // fixed seed this is the exactness test of MTP under sampling: sample-and-match draws one random number per emitted
    // token from the same distribution as plain decoding (the verify rows equal the T = 1 rows bit for bit), so the
    // sequences must be identical, not just equally distributed.
    void refcheck(server_slot & slot) {
        const auto & sp_task = slot.task->params.sampling;
        if (!(sp_task.temp <= 0.0f || sp_task.seed != LLAMA_DEFAULT_SEED)) {
            SLT_INF(slot, "%s", "REFCHECK skipped: sampled with a random seed\n");
            return;
        }
        const int n = (int) slot.emitted.size();
        llama_tokens ref;
        try {
            common_params_sampling sp = sp_task;
            srv_sampler_ptr smpl(srv_sampler_init(model_tgt, sp));
            srv_sampler_reset(smpl.get());
            for (llama_token t : slot.prompt) {
                srv_sampler_accept(smpl.get(), t, false);
            }
            eng->seq_keep(slot.seq, 0);
            p100e::decode_res out;
            const int U = eng->info().n_ubatch;
            if (!slot.mm.empty()) {
                eng->prefill_mm(slot.seq, slot.mm, true, out);   // one call: the engine cuts its own ubatch chunks
            }
            for (int i = 0; slot.mm.empty() && i < (int) slot.prompt.size(); i += U) {
                const int c = std::min(U, (int) slot.prompt.size() - i);
                eng->prefill(slot.seq, llama_tokens(slot.prompt.begin() + i, slot.prompt.begin() + i + c), true, out);
            }
            llama_token t = srv_sampler_sample(smpl.get(), out.logits.data());
            srv_sampler_accept(smpl.get(), t, true);
            ref.push_back(t);
            std::vector<p100e::decode_res> outs;
            while ((int) ref.size() < n) {
                p100e::decode_req rq;
                rq.seq = slot.seq;
                rq.tokens = { ref.back() };
                rq.logits = true;
                eng->decode({ rq }, outs);
                eng->keep(slot.seq, 1);
                t = srv_sampler_sample(smpl.get(), outs[0].logits.data());
                srv_sampler_accept(smpl.get(), t, true);
                ref.push_back(t);
            }
        } catch (const p100e::engine_fatal_error & e) {
            p100e_engine_fatal(e.what());
        } catch (const std::exception & e) {
            SLT_ERR(slot, "REFCHECK failed: %s\n", e.what());
            return;
        }
        const char * kind = sp_task.temp <= 0.0f ? "greedy" : "sampled";
        int first_diff = -1;
        for (int i = 0; i < n; ++i) {
            if (ref[i] != slot.emitted[i]) {
                first_diff = i;
                break;
            }
        }
        if (first_diff < 0) {
            SLT_INF(slot, "REFCHECK IDENTICAL: %d generated tokens == plain decoding (%s%s, drafts %d / accepted %d, prompt %d tokens)\n",
                    n, kind, slot.all_greedy ? ", GPU argmax path" : "", (int) slot.stats.n_draft_tokens,
                    (int) slot.stats.n_draft_accepted, slot.prompt_rows());
        } else {
            SLT_INF(slot, "REFCHECK MISMATCH at token %d of %d: server %d, plain decoding %d (%s, prompt %d tokens)\n",
                    first_diff, n, slot.emitted[first_diff], ref[first_diff], kind, slot.prompt_rows());
        }
    }

    bool process_token(completion_token_output & result, server_slot & slot) {
        const std::string token_str = result.text_to_send;

        slot.generated_text += token_str;
        slot.emitted.push_back(result.tok);
        if (slot.task->params.return_tokens) {
            slot.generated_tokens.push_back(result.tok);
        }
        slot.has_next_token = true;

        bool incomplete = validate_utf8(slot.generated_text) < slot.generated_text.size();

        if (!incomplete) {
            size_t pos = std::min(slot.n_sent_text, slot.generated_text.size());

            const std::string str_test = slot.generated_text.substr(pos);
            bool send_text = true;

            size_t stop_pos = slot.find_stopping_strings(str_test, token_str.size(), true);
            if (stop_pos != std::string::npos) {
                slot.generated_text.erase(
                    slot.generated_text.begin() + pos + stop_pos,
                    slot.generated_text.end());
                pos = std::min(slot.n_sent_text, slot.generated_text.size());
            } else if (slot.has_next_token && !llama_vocab_is_eog(vocab, result.tok) ) {
                stop_pos = slot.find_stopping_strings(str_test, token_str.size(), false);
                send_text = stop_pos == std::string::npos;
            }

            if (send_text) {
                result.text_to_send = slot.generated_text.substr(pos, std::string::npos);
                slot.n_sent_text += result.text_to_send.size();
            } else {
                result.text_to_send = "";
            }

            slot.add_token(result);
            if (slot.task->params.stream) {
                send_partial_response(slot, result, false);
            }
        }

        if (incomplete) {
            slot.has_next_token = true;
        }

        // no context shift: stop before the context is full
        if (slot.n_fed + 1 >= slot.n_ctx) {
            slot.truncated      = true;
            slot.stop           = STOP_TYPE_LIMIT;
            slot.has_next_token = false;

            SLT_DBG(slot, "stopped due to running out of context capacity, n_tokens = %d, task.n_tokens = %d, n_gen = %d, n_ctx = %d\n",
                    slot.n_fed, slot.task->n_tokens(), (int) slot.stats.n_gen, slot.n_ctx);
        }

        if (slot.stats.n_gen > 0 && slot.has_next_token && !slot.has_budget()) {
            slot.stop           = STOP_TYPE_LIMIT;
            slot.has_next_token = false;

            SLT_DBG(slot, "stopped by limit, n_gen = %d, n_predict = %d\n", (int) slot.stats.n_gen, slot.task->params.n_predict);
        }

        if (slot.has_new_line) {
            if (slot.task->params.n_indent > 0) {
                if (slot.last_nl_pos > 0) {
                    size_t pos = slot.last_nl_pos;

                    int n_indent = 0;
                    while (pos < slot.generated_text.size() && (slot.generated_text[pos] == ' ' || slot.generated_text[pos] == '\t')) {
                        n_indent++;
                        pos++;
                    }

                    if (pos < slot.generated_text.size() && n_indent < slot.task->params.n_indent) {
                        slot.stop           = STOP_TYPE_LIMIT;
                        slot.has_next_token = false;

                        slot.generated_text.erase(pos, std::string::npos);

                        SLT_DBG(slot, "stopped by indentation limit, n_gen = %d, n_indent = %d\n", (int) slot.stats.n_gen, n_indent);
                    }
                }

                {
                    const size_t pos = slot.generated_text.find('\n', slot.last_nl_pos);

                    if (pos != std::string::npos) {
                        slot.last_nl_pos = pos + 1;
                    }
                }
            }
        }

        if (result.text_to_send.find('\n') != std::string::npos) {
            slot.has_new_line = true;

            if (slot.task->params.t_max_predict_ms > 0 && slot.stats.t_gen_ms() > slot.task->params.t_max_predict_ms) {
                slot.stop           = STOP_TYPE_LIMIT;
                slot.has_next_token = false;

                SLT_DBG(slot, "stopped by time limit, n_gen = %d, t_max_predict_ms = %d ms\n", (int) slot.stats.n_gen, (int) slot.task->params.t_max_predict_ms);
            }
        }

        if (llama_vocab_is_eog(vocab, result.tok)) {
            slot.stop           = STOP_TYPE_EOS;
            slot.has_next_token = false;

            SLT_DBG(slot, "%s", "stopped by EOS\n");
        }

        SLT_DBG(slot, "n_gen = %d, n_remaining = %d, next token: %5d '%s'\n", (int) slot.stats.n_gen, slot.n_remaining(), result.tok, token_str.c_str());

        return slot.has_next_token;
    }

    // n_probs > 0: token probabilities from the logits row (pre-sampling) or the sampler's candidates (post-sampling)
    void populate_token_probs(const server_slot & slot, completion_token_output & result, bool post_sampling,
                              bool special, const float * logits) const {
        const size_t n_probs_request = slot.task->params.sampling.n_probs;

        if (post_sampling) {
            const auto * cur_p = srv_sampler_get_candidates(slot.smpl.get(), true);
            const size_t max_probs = cur_p->size;
            const size_t n_probs = std::min(max_probs, n_probs_request);

            for (size_t i = 0; i < max_probs; i++) {
                if (cur_p->data[i].id == result.tok) {
                    result.prob = cur_p->data[i].p;
                    break;
                }
            }

            result.probs.reserve(n_probs);
            for (size_t i = 0; i < n_probs; i++) {
                if (cur_p->data[i].p == 0.0) {
                    break;
                }
                result.probs.push_back({
                    cur_p->data[i].id,
                    common_token_to_piece(vocab, cur_p->data[i].id, special),
                    cur_p->data[i].p
                });
            }
        } else if (logits != nullptr) {
            const int n_vocab = llama_vocab_n_tokens(vocab);
            std::vector<llama_token_data> cur(n_vocab);
            for (llama_token t = 0; t < n_vocab; t++) {
                cur[t] = llama_token_data{t, logits[t], 0.0f};
            }
            const size_t n_top = std::min((size_t) n_vocab, n_probs_request);
            std::partial_sort(cur.begin(), cur.begin() + n_top, cur.end(),
                [](const llama_token_data & a, const llama_token_data & b) { return a.logit > b.logit; });
            const float max_l = cur[0].logit;
            float cum_sum = 0.0f;
            for (auto & t : cur) {
                t.p = expf(t.logit - max_l);
                cum_sum += t.p;
            }
            for (auto & t : cur) {
                t.p /= cum_sum;
            }
            for (const auto & t : cur) {
                if (t.id == result.tok) {
                    result.prob = t.p;
                    break;
                }
            }
            result.probs.reserve(n_top);
            for (size_t i = 0; i < n_top; i++) {
                result.probs.push_back({ cur[i].id, common_token_to_piece(vocab, cur[i].id, special), cur[i].p });
            }
        }
    }

    void send_error(const server_task & task, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER) {
        send_error(task.id, error, type);
    }

    void send_error(const server_slot & slot, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER) {
        send_error(slot.task->id, error, type, slot.task->n_tokens(), slot.n_ctx);
    }

    void send_error(const int id_task, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER, const int32_t n_prompt_tokens = 0, const int32_t n_ctx = 0) {
        SRV_ERR("task id = %d, error: %s\n", id_task, error.c_str());

        auto res = std::make_unique<server_task_result_error>();
        res->id              = id_task;
        res->err_type        = type;
        res->err_msg         = error;
        res->n_prompt_tokens = n_prompt_tokens;
        res->n_ctx           = n_ctx;

        queue_results.send(std::move(res));
    }

    void send_partial_response(server_slot & slot, const completion_token_output & tkn, bool is_progress, bool is_begin = false) {
        auto res = std::make_unique<server_task_result_cmpl_partial>();

        res->id    = slot.task->id;
        res->index = slot.task->index;

        if (is_progress) {
            res->is_progress        = true;
            res->progress.total     = slot.task->n_tokens();
            res->progress.cache     = slot.stats.n_prompt_cached;
            res->progress.processed = slot.n_prompt_done;
            res->progress.time_ms   = slot.stats.t_elapsed_us() / 1000;
        }
        if (is_begin) {
            res->is_begin = true;
        } else {
            res->content = tkn.text_to_send;
            res->tokens  = { tkn.tok };
        }

        res->n_decoded             = slot.stats.n_gen;
        res->n_prompt_tokens       = slot.task->n_tokens();
        res->n_prompt_tokens_cache = slot.stats.n_prompt_cached;
        res->post_sampling_probs   = slot.task->params.post_sampling_probs;

        res->verbose           = slot.task->params.verbose;
        res->res_type          = slot.task->params.res_type;
        res->oaicompat_model   = slot.task->params.oaicompat_model;
        res->oaicompat_cmpl_id = slot.task->params.oaicompat_cmpl_id;

        if (slot.task->params.sampling.n_probs > 0) {
            res->prob_output = tkn;
        }

        if (slot.stop != STOP_TYPE_NONE || slot.task->params.timings_per_token) {
            res->stats = slot.stats;
        }

        queue_results.send(std::move(res));
    }

    void send_final_response(server_slot & slot) {
        auto res = std::make_unique<server_task_result_cmpl_final>();

        res->id      = slot.task->id;
        res->id_slot = slot.id;

        res->index = slot.task->index;

        if (slot.task->params.stream) {
            res->content     = "";
            res->tokens      = llama_tokens{};
        } else {
            res->content     = std::move(slot.generated_text);
            res->tokens      = std::move(slot.generated_tokens);
        }
        res->stats           = slot.stats;
        // only the native (non-OAI) format prints the prompt; skip detokenizing a long prompt otherwise
        if (slot.task->params.res_type == TASK_RESPONSE_TYPE_NONE) {
            res->prompt = common_detokenize(vocab, slot.prompt, true);
        }
        res->response_fields = slot.task->params.response_fields;

        res->truncated             = slot.truncated;
        res->n_decoded             = slot.stats.n_gen;
        res->n_prompt_tokens       = slot.task->n_tokens();
        res->n_prompt_tokens_cache = slot.stats.n_prompt_cached;
        res->n_tokens_cached       = slot.n_fed;
        res->has_new_line          = slot.has_new_line;
        res->stopping_word         = slot.stopping_word;
        res->stop                  = slot.stop;
        res->post_sampling_probs   = slot.task->params.post_sampling_probs;

        res->verbose           = slot.task->params.verbose;
        res->stream            = slot.task->params.stream;
        res->include_usage     = slot.task->params.include_usage;
        res->res_type          = slot.task->params.res_type;
        res->oaicompat_model   = slot.task->params.oaicompat_model;
        res->oaicompat_cmpl_id = slot.task->params.oaicompat_cmpl_id;

        if (slot.task->params.sampling.n_probs > 0) {
            if (!slot.task->params.stream && slot.stop == STOP_TYPE_WORD) {
                const llama_tokens stop_word_toks = common_tokenize(vocab, slot.stopping_word, false);

                size_t safe_offset = std::min(slot.generated_token_probs.size(), stop_word_toks.size());
                res->probs_output = std::vector<completion_token_output>(
                        slot.generated_token_probs.begin(),
                        slot.generated_token_probs.end() - safe_offset);
            } else {
                res->probs_output = std::vector<completion_token_output>(
                        slot.generated_token_probs.begin(),
                        slot.generated_token_probs.end());
            }
        }

        res->generation_params = slot.task->params;

        queue_results.send(std::move(res));
    }

    bool process_single_task(server_task && task, bool is_yielding) {
        if (is_yielding && task.type != SERVER_TASK_TYPE_METRICS && task.type != SERVER_TASK_TYPE_SLOT_GET) {
            return false;
        }

        switch (task.type) {
            case SERVER_TASK_TYPE_COMPLETION:
            case SERVER_TASK_TYPE_INFILL:
                {
                    if (task.cli) {
                        send_error(task, "CLI input is not supported", ERROR_TYPE_NOT_SUPPORTED);
                        break;
                    }
                    if (!eng) {
                        send_error(task, "this p100e-server instance has no engine (--p100e-no-engine)", ERROR_TYPE_UNAVAILABLE);
                        break;
                    }
                    if (task.is_parent()) {
                        send_error(task, "n_cmpl > 1 is not supported by p100e-server", ERROR_TYPE_NOT_SUPPORTED);
                        break;
                    }
                    server_slot * slot = get_available_slot(task);
                    if (slot == nullptr) {
                        SRV_DBG("no slot is available, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }
                    const int id_task = task.id;
                    if (!launch_slot_with_task(*slot, std::move(task))) {
                        SRV_ERR("failed to launch slot with task, id_task = %d\n", id_task);
                        // the slot stays idle and release() never runs for it: hand the slot to the next queued task
                        // here, or the queue waits until some other request ends (the task may itself have been the
                        // one release() popped; with one engine sequence every concurrent request is queued)
                        queue_tasks.pop_deferred_task(slot->id);
                        break;
                    }
                } break;
            case SERVER_TASK_TYPE_EMBEDDING:
            case SERVER_TASK_TYPE_RERANK:
                {
                    send_error(task, "embeddings / rerank are not supported by p100e-server", ERROR_TYPE_NOT_SUPPORTED);
                } break;
            case SERVER_TASK_TYPE_CANCEL:
                {
                    for (auto & slot : slots) {
                        if (slot.task && slot.task->id == task.id_target) {
                            SLT_INF(slot, "%s", "cancelled (client disconnected)\n");
                            release(slot);
                            break;
                        }
                    }
                } break;
            case SERVER_TASK_TYPE_CONTROL:
                {
                    auto res = std::make_unique<server_task_result_control>();
                    res->id = task.id;

                    server_slot * slot = get_slot_by_cmpl_id(task.params.control_cmpl_id);
                    if (slot == nullptr) {
                        res->success = false;
                        res->message = "no active completion for this id";
                        queue_results.send(std::move(res));
                        break;
                    }

                    if (task.params.control_action == "reasoning_end") {
                        if (!slot->task->params.sampling.reasoning_control) {
                            res->success = false;
                            res->message = "reasoning control not enabled for this completion";
                            queue_results.send(std::move(res));
                            break;
                        }
                        // false unless the budget is counting (inside <think>): before it opened, after it closed,
                        // already forcing, or a template without reasoning markers
                        res->success = srv_sampler_reasoning_budget_force(slot->smpl.get());
                        if (!res->success) {
                            res->message = "reasoning is not in progress for this completion (not inside the thinking block)";
                        }
                    } else {
                        res->success = false;
                        res->message = "unknown control action";
                    }

                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_NEXT_RESPONSE:
                {
                } break;
            case SERVER_TASK_TYPE_METRICS:
                {
                    int n_processing_slots = 0;
                    for (server_slot & slot : slots) {
                        if (slot.is_processing()) {
                            n_processing_slots++;
                        }
                    }

                    auto res = std::make_unique<server_task_result_metrics>();
                    res->id                  = task.id;
                    res->n_processing_slots  = n_processing_slots;
                    res->n_tasks_deferred    = queue_tasks.queue_tasks_deferred_size();
                    res->metrics             = metrics;
                    if (eng) {
                        std::lock_guard<std::mutex> lk(p100e_cache_metrics_mx);
                        p100e_cache_metrics = p100e_cache_text();
                    }

                    if (task.metrics_reset_bucket) {
                        metrics.reset_bucket();
                    }
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_GET:
                {
                    json slots_data = json::array();

                    int n_idle_slots = 0;

                    for (server_slot & slot : slots) {
                        if (!slot.is_processing()) {
                            n_idle_slots++;
                        }
                        slots_data.push_back(slot.to_json(vocab, true, can_speculate()));
                    }

                    auto res = std::make_unique<server_task_result_slots>();
                    res->id           = task.id;
                    res->slots_data   = std::move(slots_data);
                    res->n_idle_slots = n_idle_slots;

                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_SAVE:
            case SERVER_TASK_TYPE_SLOT_RESTORE:
            case SERVER_TASK_TYPE_SLOT_ERASE:
                {
                    send_error(task, "slot save / restore / erase is not supported by p100e-server", ERROR_TYPE_NOT_SUPPORTED);
                } break;
            case SERVER_TASK_TYPE_GET_LORA:
                {
                    auto res = std::make_unique<server_task_result_get_lora>();
                    res->id = task.id;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SET_LORA:
                {
                    send_error(task, "LoRA adapters are not supported by p100e-server", ERROR_TYPE_NOT_SUPPORTED);
                } break;
        }

        return true;
    }

    template <typename F>
    void guarded(server_slot & slot, F && f) {
        try {
            f();
        } catch (const p100e::engine_fatal_error & e) {
            p100e_engine_fatal(e.what());
        } catch (const std::exception & e) {
            SLT_ERR(slot, "got exception: %s\n", e.what());
            send_error(slot, std::string("got exception: ") + e.what(), ERROR_TYPE_SERVER);
            release(slot);
        }
    }

    // max drafts for the slot's next step: room in the context and in the token budget (llama.cpp get_n_draft_max)
    int n_draft_max(const server_slot & slot) const {
        if (!can_speculate() || slot.task->params.sampling.n_probs > 0) {
            return 0;   // probabilities are reported per sampled token, not for drafts (as llama.cpp)
        }
        int k = eng->info().n_draft_max;
        if (g_p100e_opts.draft_rows > 0) {   // several slots generating: share the draft row budget (sched::k_eff)
            const int n = n_generating();
            if (n > 1) k = std::min(k, std::max(1, (std::max(0, g_p100e_opts.draft_rows - n) + n - 1) / n));
        }
        if (g_p100e_opts.mtp_multi >= 0) {   // several slots generating: at most mtp_multi drafts each
            if (n_generating() > 1) k = std::min(k, g_p100e_opts.mtp_multi);
        }
        k = std::min(k, slot.n_ctx - slot.n_fed - 2);
        if (slot.n_remaining() > 0) {
            k = std::min(k, slot.n_remaining() - 1);
        }
        return std::max(0, k);
    }

    int n_generating() const {
        int n = 0;
        for (const auto & o : slots) n += o.state == SLOT_STATE_GENERATING;
        return n;
    }

    bool use_gpu_argmax(const server_slot & slot) const {
        return g_p100e_opts.greedy_gpu && srv_sampler_is_greedy(slot.smpl.get());
    }

    bool accept_special_token(const server_slot & slot, llama_token token) const {
        return params_base.special ||
            slot.task->params.sampling.preserved_tokens.find(token) != slot.task->params.sampling.preserved_tokens.end();
    }

    // emit one sampled token: stats, text, stop conditions; false = the slot was released
    bool emit_token(server_slot & slot, llama_token id, const float * logits_row) {
        slot.stats.n_gen += 1;
        if (slot.stats.n_gen == 1) {
            slot.stats.update_prompt_last();
            slot.t_print_last = ggml_time_us();
            slot.n_gen_last = 0;
        }
        slot.stats.update_gen_last();

        completion_token_output result;
        result.tok          = id;
        result.text_to_send = common_token_to_piece(vocab, result.tok, accept_special_token(slot, result.tok));
        result.prob         = 1.0f;

        if (slot.task->params.sampling.n_probs > 0) {
            populate_token_probs(slot, result, slot.task->params.post_sampling_probs, params_base.special, logits_row);
        }

        if (!process_token(result, slot)) {
            slot.print_timings();
            send_final_response(slot);
            release(slot);
            return false;
        }
        return true;
    }

    void start_slot(server_slot & slot) {
        slot.stats.update_prompt_start();
        slot.state = SLOT_STATE_PROCESSING_PROMPT;

        SLT_TRC(slot, "new prompt, n_ctx_slot = %d, task.n_tokens = %d\n", slot.n_ctx, slot.task->n_tokens());

        if (slot.prompt_rows() == 0) {
            SLT_WRN(slot, "%s", "empty prompt - releasing slot\n");
            slot.print_timings();
            send_final_response(slot);
            release(slot);
            return;
        }
        if (slot.task->n_tokens() >= slot.n_ctx) {
            send_error(slot,
                       string_format("request (%d tokens) exceeds the available context size (%d "
                                     "tokens), try increasing it",
                                     slot.task->n_tokens(), slot.n_ctx),
                       ERROR_TYPE_EXCEED_CONTEXT_SIZE);
            release(slot);
            return;
        }

        // prompt cache: the engine starts the sequence from the longest cached prefix with a known state (keys per row:
        // token ids, image rows by image hash); the rows before n_cached are not fed again
        std::vector<uint64_t> keys;
        if (slot.mm.empty()) {
            keys.reserve(slot.prompt.size());
            for (llama_token t : slot.prompt) keys.push_back((uint64_t) (uint32_t) t);
        } else {
            keys = slot.mm_keys;
        }
        int n_cached = 0;
        const bool use_cache = slot.task->params.cache_prompt && (int) keys.size() == slot.prompt_rows();
        slot.seq = use_cache ? eng->seq_open_cached(keys, n_cached) : eng->seq_open();
        if (slot.seq < 0) {
            throw std::runtime_error("engine has no free sequence");
        }
        if (!use_cache) {
            eng->seq_keep(slot.seq, 0);
            n_cached = 0;
        }
        n_cached = std::max(0, std::min(n_cached, slot.prompt_rows() - 1));
        slot.n_fed = n_cached;
        slot.n_prompt_done = n_cached;
        if (!slot.mm.empty()) {   // skip the cached rows of the pieces (never inside an image)
            int left = n_cached;
            while (left > 0 && slot.mm_next < slot.mm.size()) {
                const p100e::prompt_piece & pc = slot.mm[slot.mm_next];
                const int r = pc.n_rows() - slot.mm_off;
                if (left >= r) {
                    left -= r;
                    slot.mm_next++;
                    slot.mm_off = 0;
                } else {
                    if (pc.embd) {
                        throw std::runtime_error("prompt cache: reuse point inside an image");
                    }
                    slot.mm_off += left;
                    left = 0;
                }
            }
        }
        slot.stats.n_prompt_cached    = n_cached;
        slot.stats.n_prompt_processed = 0;
        metrics.add_prompt_cached(n_cached);
        if (n_cached > 0) {
            SLT_INF(slot, "prompt cache: reusing %d of %d rows\n", n_cached, slot.prompt_rows());
        }

        if (slot.task->params.stream) {
            if (slot.task->params.return_progress) {
                send_partial_response(slot, {}, true);
            } else {
                send_partial_response(slot, {}, false, true);
            }
        }
    }

    // the next pieces of a multimodal prompt for one prefill_mm call: up to n_ubatch rows, text split where needed, an
    // image always whole (one that does not fit the rest of this group starts the next; one longer than n_ubatch is a
    // group of its own and the engine chunks it); n = the group's rows
    std::vector<p100e::prompt_piece> next_mm_group(server_slot & slot, int & n) const {
        const int U = eng->info().n_ubatch;
        std::vector<p100e::prompt_piece> grp;
        n = 0;
        while (slot.mm_next < slot.mm.size() && n < U) {
            const p100e::prompt_piece & pc = slot.mm[slot.mm_next];
            if (pc.embd) {
                if (n > 0 && n + pc.n_rows() > U) {
                    break;
                }
                grp.push_back(pc);
                n += pc.n_rows();
                slot.mm_next++;
                continue;
            }
            const int take = std::min((int) pc.tokens.size() - slot.mm_off, U - n);
            p100e::prompt_piece q;
            q.tokens.assign(pc.tokens.begin() + slot.mm_off, pc.tokens.begin() + slot.mm_off + take);
            grp.push_back(std::move(q));
            n += take;
            slot.mm_off += take;
            if (slot.mm_off == (int) pc.tokens.size()) {
                slot.mm_next++;
                slot.mm_off = 0;
            }
        }
        return grp;
    }

    // one prompt chunk; the last one samples the first token
    void prompt_step(server_slot & slot) {
        const int P = slot.prompt_rows();
        int n = 0;
        std::vector<p100e::prompt_piece> grp;
        if (slot.mm.empty()) {
            n = std::min(eng->info().n_ubatch, P - slot.n_prompt_done);
        } else {
            grp = next_mm_group(slot, n);
        }
        const bool is_last = slot.n_prompt_done + n == P;

        if (is_last) {
            // llama.cpp init_sampler(): the prompt tokens enter the sampler history (penalties), not the grammar
            srv_sampler_reset(slot.smpl.get());
            srv_sampler_accept_prompt(slot.smpl.get(), slot.prompt);
        }
        const bool want_logits = is_last && !use_gpu_argmax(slot);
        if (want_logits) {
            slot.all_greedy = false;
        }

        p100e::decode_res out;
        const int64_t t0 = ggml_time_us();
        if (slot.mm.empty()) {
            const llama_tokens chunk(slot.prompt.begin() + slot.n_prompt_done, slot.prompt.begin() + slot.n_prompt_done + n);
            eng->prefill(slot.seq, chunk, want_logits, out);
        } else {
            eng->prefill_mm(slot.seq, grp, want_logits, out);
            static bool mem_logged = false;   // GPU headroom once the encoder and an image prefill have run
            if (is_last && !mem_logged) {
                mem_logged = true;
                eng->print_mem("after the first image prompt");
            }
        }
        metrics.add_prompt(n, ggml_time_us() - t0);
        slot.n_prompt_done += n;
        slot.n_fed += n;
        metrics_post_forward();
        slot.stats.n_prompt_processed += n;
        slot.stats.set_prompt_last(ggml_time_us());

        if (slot.task->params.stream && slot.task->params.return_progress) {
            send_partial_response(slot, {}, true);
        }
        slot.print_timings_pp();

        if (!is_last) {
            return;
        }

        slot.state = SLOT_STATE_GENERATING;
        slot.stats.n_gen = 0;

        llama_token id;
        if (want_logits) {
            id = srv_sampler_sample(slot.smpl.get(), out.logits.data());
        } else {
            id = out.argmax.at(0);
        }
        srv_sampler_accept(slot.smpl.get(), id, true);

        if (!emit_token(slot, id, want_logits ? out.logits.data() : nullptr)) {
            return;
        }

        slot.sampled = id;
        const int k = n_draft_max(slot);
        slot.drafts = eng->draft(slot.seq, id, k);
        slot.draft_p = eng->draft_probs(slot.seq);
        slot.stats.n_draft_tokens += slot.drafts.size();
    }

    // n-gram shadow: what a gated suffix-match drafter would propose after the slot's history (host only, no engine call)
    void ng_observe(server_slot & slot, int n_acc) {
        if (!p100e_ng::enabled()) return;
        if (!slot.ng.started()) {
            std::vector<int32_t> h;
            if (slot.mm.empty()) {
                h.assign(slot.prompt.begin(), slot.prompt.end());
            } else {
                int32_t img = -1;
                for (const auto & pc : slot.mm) {
                    if (pc.embd) { for (int i = 0; i < pc.n_rows(); ++i) h.push_back(img--); }   // unique, never matches
                    else h.insert(h.end(), pc.tokens.begin(), pc.tokens.end());
                }
            }
            slot.ng.start(std::move(h));
        }
        slot.ng.step(slot.emitted);
        slot.ng.note_real(n_acc);
    }

    // one log line per request + the process totals
    void ng_report(server_slot & slot) {
        if (!p100e_ng::enabled() || !slot.ng.started()) return;
        p100e_ng::LvlAgg a[p100e_ng::N_LVL];
        slot.ng.resolve(slot.emitted, a);
        std::string hint;
        if (slot.task) {
            std::lock_guard<std::mutex> lk(p100e_ng_hint_mx);
            auto it = p100e_ng_hint.find(slot.task->params.oaicompat_cmpl_id);
            if (it != p100e_ng_hint.end()) { hint = it->second; p100e_ng_hint.erase(it); }
        }
        std::string line;
        char b[256];
        snprintf(b, sizeof b, "NGSHADOW hint=%s prompt=%d gen=%d steps=%d", hint.empty() ? "-" : hint.c_str(),
                 slot.prompt_rows(), (int) slot.emitted.size(), slot.ng.steps());
        line = b;
        for (int l = 0; l < p100e_ng::N_LVL; ++l) {
            snprintf(b, sizeof b, " m%d=%llu,%llu,%llu,%llu,", p100e_ng::LVL[l], (unsigned long long) a[l].n,
                     (unsigned long long) a[l].prop, (unsigned long long) a[l].match, (unsigned long long) a[l].real);
            line += b;
            for (int k = 0; k <= p100e_ng::L_PROP; ++k) { line += (k ? ":" : "") + std::to_string((unsigned long long) a[l].k[k]); }
        }
        SLT_INF(slot, "%s\n", line.c_str());
    }

    // after the engine's decode step: sample-and-match over the rows, keep the accepted rows, emit, draft again
    void finish_step(server_slot & slot, const p100e::decode_req & rq, const p100e::decode_res & out) {
        const int n_vocab = llama_vocab_n_tokens(vocab);
        const int T = (int) rq.tokens.size();
        const llama_tokens drafts = slot.drafts;   // rq.tokens[1..]
        std::vector<llama_token> acc;              // emitted tokens, in order
        std::vector<const float *> rows;           // logits row each emitted token was sampled from (n_probs)
        int n_keep = 0;                            // rows that stay in the engine state
        bool cut = false;                          // a greedy step met a row it cannot sample from argmax
        for (int i = 0; i < T; ++i) {
            llama_token s;
            const float * row = nullptr;
            if (rq.logits) {
                row = out.logits.data() + (size_t) i * n_vocab;
                s = srv_sampler_sample(slot.smpl.get(), row);
            } else {
                if (i > 0 && !srv_sampler_is_greedy(slot.smpl.get())) {
                    cut = true;   // e.g. the reasoning budget started forcing: row i needs its logits
                    break;
                }
                s = out.argmax.at(i);
            }
            srv_sampler_accept(slot.smpl.get(), s, true);
            acc.push_back(s);
            rows.push_back(row);
            n_keep = i + 1;
            // llama.cpp #28049: the accepted run ends at the first end-of-generation token
            if (llama_vocab_is_eog(vocab, s)) {
                break;
            }
            if (i < (int) drafts.size() && s == drafts[i]) {
                continue;   // draft i accepted: row i + 1 is its continuation
            }
            break;
        }
        // rows 0 .. n_keep - 1 stay in the state; acc.back() (sampled from row n_keep - 1) is the next step's first row.
        // A cut keeps the same rule: acc.back() == drafts[n_keep - 1] is then row n_keep's token, which is dropped
        // with the rows after it and fed again next step.
        GGML_ASSERT(n_keep >= 1 && (int) acc.size() == n_keep);
        if (cut) {
            SLT_DBG(slot, "greedy step cut at row %d (sampler no longer greedy)\n", n_keep);
        }
        int n_acc_drafts = 0;
        while (n_acc_drafts < (int) acc.size() && n_acc_drafts < (int) drafts.size() && acc[n_acc_drafts] == drafts[n_acc_drafts]) {
            n_acc_drafts++;
        }
        ng_observe(slot, (int) acc.size());
        eng->keep(slot.seq, n_keep);
        const int n_fed_before = slot.n_fed;   // n_fed_before + n_keep after the emit loop below

        if (!drafts.empty()) {
            slot.stats.n_draft_accepted += n_acc_drafts;
            slot.stats.n_draft_verif_steps += 1;
            if (slot.n_accepted_per_pos.empty()) {
                slot.n_accepted_per_pos.resize(eng->info().n_draft_max, 0);
            }
            for (int i = 0; i < n_acc_drafts && i < (int) slot.n_accepted_per_pos.size(); ++i) {
                slot.n_accepted_per_pos[i]++;
            }
            // process-wide per-position counters for /metrics: drafted = how often draft position i was offered, so
            // accepted[i] / drafted[i] is the exact per-position rate (the stop at confidence < tau makes drafted fall)
            for (int i = 0; i < (int) drafts.size() && i < P100E_POS_N; ++i) {
                p100e_pos_drafted[i].fetch_add(1, std::memory_order_relaxed);
                if (i < n_acc_drafts) {
                    p100e_pos_accepted[i].fetch_add(1, std::memory_order_relaxed);
                }
            }
        }

        // emit token by token with the state as plain decoding would have it: acc[i] was sampled from row i, so the
        // state holds the rows 0..i (n_fed_before + i + 1) when process_token checks the context limit - the cut then
        // falls on the same token whether or not drafts were accepted
        for (size_t i = 0; i < acc.size(); ++i) {
            slot.n_fed = n_fed_before + (int) i + 1;
            if (!emit_token(slot, acc[i], rows[i])) {
                return;   // released
            }
        }

        slot.sampled = acc.back();
        slot.drafts.clear();
        slot.draft_p.clear();
        slot.draft_pending = true;   // drafted with the other slots of this step (update_slots: one draft_batch)
    }

    // /metrics p100e:* prompt cache lines
    std::string p100e_cache_text() const {
        if (!eng) {
            return {};
        }
        const auto c = eng->cstats();
        std::string t;
        p100e_prom(t, "prompt_cache_hits_total", "Requests that reused a cached prefix.", (double) c.hits);
        p100e_prom(t, "prompt_cache_misses_total", "Requests without a reusable prefix.", (double) c.misses);
        p100e_prom(t, "prompt_cache_rows_reused_total", "Prompt rows taken from the cache.", (double) c.rows_reused);
        p100e_prom(t, "prompt_cache_entries", "Idle cached sequences (GPU or host RAM).", (double) c.entries);
        p100e_prom(t, "prompt_cache_offloaded", "Cached sequences in pinned host RAM.", (double) c.offloaded);
        p100e_prom(t, "prompt_cache_checkpoints", "Delta-net state checkpoints held.", (double) c.ckpts);
        p100e_prom(t, "prompt_cache_host_bytes", "Pinned host RAM of offloaded sequences and checkpoints.", (double) c.host_bytes);
        p100e_prom(t, "kv_pool_tokens", "KV pool positions (all sequences).", (double) c.pool_tokens);
        p100e_prom(t, "kv_pool_free_tokens", "Free KV pool positions.", (double) c.free_tokens);
        for (int i = 0; i < P100E_POS_N; ++i) {   // exact per-position MTP acceptance: accepted / drafted
            const std::string s = std::to_string(i + 1);
            p100e_prom(t, "spec_drafted_pos" + s + "_total", "Verify steps in which draft position was offered.", (double) p100e_pos_drafted[i].load(), "counter");
            p100e_prom(t, "spec_accepted_pos" + s + "_total", "Verify steps in which draft position was accepted.", (double) p100e_pos_accepted[i].load(), "counter");
        }
        if (p100e_ng::enabled()) {   // n-gram shadow telemetry totals
            auto & T = p100e_ng::totals();
            p100e_prom(t, "ngshadow_requests_total", "Requests observed by the n-gram shadow.", (double) T.requests.load(), "counter");
            p100e_prom(t, "ngshadow_steps_total", "Decode steps observed by the n-gram shadow.", (double) T.steps.load(), "counter");
            p100e_prom(t, "ngshadow_gen_tokens_total", "Generated tokens of the observed requests.", (double) T.gen.load(), "counter");
            p100e_prom(t, "ngdraft_steps_total", "Steps whose drafts came from the n-gram matcher (P100E_NGRAM_DRAFT).", (double) T.live_steps.load(), "counter");
            for (int l = 0; l < p100e_ng::N_LVL; ++l) {
                const std::string s = "m" + std::to_string(p100e_ng::LVL[l]);
                p100e_prom(t, "ngshadow_" + s + "_steps_total", "Steps with a suffix match at this gate level.", (double) T.n[l].load(), "counter");
                p100e_prom(t, "ngshadow_" + s + "_proposed_total", "Tokens the n-gram drafter would have proposed (L=7).", (double) T.prop[l].load(), "counter");
                p100e_prom(t, "ngshadow_" + s + "_matched_total", "Proposed tokens that matched the real next tokens (prefix).", (double) T.match[l].load(), "counter");
                p100e_prom(t, "ngshadow_" + s + "_real_tokens_total", "Tokens the real MTP step emitted on those steps.", (double) T.real[l].load(), "counter");
            }
        }
        return t;
    }

    // metrics history (P100E_METRICS_LOG=<csv path>, unset = off): one row when the server starts, one a minute while a
    // slot is busy and one when the last busy slot finishes, so the cumulative counters (per-position MTP accepts,
    // tokens, prompt cache) outlive the process; a row with event "start" is a restart (the counters begin at zero)
    int64_t hist_last_us   = -1;
    int     hist_busy_prev = 0;
    void hist_row(const char * ev) {
        static const char * path = getenv("P100E_METRICS_LOG");
        FILE * f = fopen(path, "ab");
        if (!f) {
            return;
        }
        fseek(f, 0, SEEK_END);
        if (ftell(f) == 0) {
            fputs("time,event,drafts,draft_tokens,accepted_tokens,"
                  "drafted_pos1,accepted_pos1,drafted_pos2,accepted_pos2,drafted_pos3,accepted_pos3,"
                  "tokens_predicted,predict_s,prompt_tokens,prompt_s,prompt_cached,"
                  "cache_hits,cache_misses,cache_rows_reused,cache_entries,cache_ckpts,ngdraft_steps,slots_busy\n", f);
        }
        char ts[32];
        const time_t tt = time(nullptr);
        struct tm tmv;
#ifdef _WIN32
        localtime_s(&tmv, &tt);
#else
        localtime_r(&tt, &tmv);
#endif
        strftime(ts, sizeof(ts), "%Y-%m-%dT%H:%M:%S", &tmv);
        p100e::engine::cache_stats c{};
        if (eng) {
            c = eng->cstats();
        }
        int busy = 0;
        for (const server_slot & slot : slots) busy += slot.is_processing();
        const uint64_t ngd = p100e_ng::enabled() ? (uint64_t) p100e_ng::totals().live_steps.load() : 0;
        fprintf(f, "%s,%s,%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",", ts, ev, metrics.n_draft_verif_steps, metrics.n_draft_tokens, metrics.n_draft_accepted);
        for (int i = 0; i < 3; ++i) {
            fprintf(f, "%" PRIu64 ",%" PRIu64 ",", (uint64_t) p100e_pos_drafted[i].load(), (uint64_t) p100e_pos_accepted[i].load());
        }
        fprintf(f, "%" PRIu64 ",%.3f,%" PRIu64 ",%.3f,%" PRIu64 ",", (uint64_t) metrics.predict.count, metrics.predict.time / 1e6,
                (uint64_t) metrics.prompt.count, metrics.prompt.time / 1e6, metrics.n_prompt_cached);
        fprintf(f, "%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%d\n",
                (uint64_t) c.hits, (uint64_t) c.misses, (uint64_t) c.rows_reused, (uint64_t) c.entries, (uint64_t) c.ckpts, ngd, busy);
        fclose(f);
    }
    void hist_tick(int64_t now, int busy) {
        static const bool on = [] { const char * v = getenv("P100E_METRICS_LOG"); return v && *v; }();
        if (!on) {
            return;
        }
        if (hist_last_us < 0) {
            hist_row("start");
            hist_last_us = now;
        } else if (busy == 0 && hist_busy_prev > 0) {
            hist_row("idle");
            hist_last_us = now;
        } else if (busy > 0 && now - hist_last_us >= 60000000) {
            hist_row("tick");
            hist_last_us = now;
        }
        hist_busy_prev = busy;
    }

    // live status snapshot for /slots and /metrics (live_status.h): after every loop iteration and every task
    // force = false (the per-step call): at most one snapshot per 100 ms, plus one whenever the number of busy slots
    // changes (a request starting or finishing), so the last step of a request is always published and /slots is
    // never left stale; slot.to_json + the cache text are ~100 heap allocations that a 9 ms decode step need not pay
    int64_t live_t_us = 0;
    int     live_busy = -1;
    void publish_live(bool force) {
        int busy = 0;
        for (const server_slot & slot : slots) busy += slot.is_processing();
        const int64_t now = ggml_time_us();
        hist_tick(now, busy);
        if (!force && busy == live_busy && now - live_t_us < 100000) return;
        live_t_us = now;
        live_busy = busy;
        json sl = json::array();
        int n_idle = 0;
        for (const server_slot & slot : slots) {
            n_idle += !slot.is_processing();
            sl.push_back(slot.to_json(vocab, true, can_speculate()));
        }
        p100e_live_publish(std::move(sl), n_idle, (int) slots.size() - n_idle,
                           (int) queue_tasks.queue_tasks_deferred_size(), metrics, p100e_cache_text());
    }

    static int ng_gate() {   // P100E_NGRAM_DRAFT: 0 = off (default), else the minimum suffix-match length (8 / 12 / 24)
        static const int g = [] { const char * v = getenv("P100E_NGRAM_DRAFT"); return v ? atoi(v) : 0; }();
        return g;
    }

    // the draft phase of every slot that finished this step: one engine draft_batch (engine_tp: one batched draft-head
    // pass for all of them; each slot's drafts are those it would get alone)
    void draft_step(const std::vector<server_slot *> & gen) {
        std::vector<server_slot *> ds;
        std::vector<p100e::engine::draft_req> dr;
        std::vector<std::vector<int32_t>> ngp;   // per ds entry: the n-gram proposal that replaces the head's drafts (or empty)
        for (auto * s : gen) {
            if (!s->draft_pending) continue;
            s->draft_pending = false;
            if (s->state != SLOT_STATE_GENERATING) continue;
            p100e::engine::draft_req q;
            q.seq = s->seq;
            q.next = s->sampled;
            q.k_max = n_draft_max(*s);
            // P100E_NGRAM_DRAFT=<min match length>: a gated suffix match replaces the chained draft-head runs (the first head
            // pass still runs: it catches the head's cache up). Drafts only change speed; the verify step decides every token.
            ngp.emplace_back();
            if (ng_gate() > 0 && q.k_max > 0 && p100e_ng::enabled() && s->ng.started()) {
                int32_t buf[3];
                const int np = s->ng.propose(s->emitted, buf, std::min(q.k_max, 3), ng_gate());
                if (np > 0) { ngp.back().assign(buf, buf + np); q.k_max = 0; }
            }
            ds.push_back(s);
            dr.push_back(q);
        }
        if (ds.empty()) return;
        std::vector<std::vector<int32_t>> outs;
        for (;;) {
            try {
                eng->draft_batch(dr, outs);
                break;
            } catch (const p100e::engine_fatal_error & e) {
                p100e_engine_fatal(e.what());
            } catch (const p100e::engine_seq_error & e) {
                // one sequence has no room and no draft ran: fail that slot alone, draft the others
                size_t w = ds.size();
                for (size_t i = 0; i < ds.size(); ++i) {
                    if (ds[i]->seq == e.seq) {
                        w = i;
                    }
                }
                if (w == ds.size()) {
                    SRV_ERR("draft failed: %s\n", e.what());
                    for (auto * s : ds) {
                        send_error(*s, std::string("draft failed: ") + e.what(), ERROR_TYPE_SERVER);
                        release(*s);
                    }
                    return;
                }
                SLT_ERR(*ds[w], "draft failed: %s\n", e.what());
                send_error(*ds[w], std::string("draft failed: ") + e.what(), ERROR_TYPE_SERVER);
                release(*ds[w]);
                ds.erase(ds.begin() + w);
                dr.erase(dr.begin() + w);
                ngp.erase(ngp.begin() + w);
                if (ds.empty()) {
                    return;
                }
            } catch (const std::exception & e) {
                SRV_ERR("draft failed: %s\n", e.what());
                for (auto * s : ds) {
                    send_error(*s, std::string("draft failed: ") + e.what(), ERROR_TYPE_SERVER);
                    release(*s);
                }
                return;
            }
        }
        for (size_t i = 0; i < ds.size(); ++i) {
            server_slot & slot = *ds[i];
            if (!ngp[i].empty()) {
                slot.drafts.assign(ngp[i].begin(), ngp[i].end());
                slot.draft_p.clear();   // unknown: the row trim treats it as p = 1
                p100e_ng::totals().live_steps++;
            } else {
                slot.drafts.assign(outs[i].begin(), outs[i].end());
                slot.draft_p = eng->draft_probs(slot.seq);
            }
            slot.stats.n_draft_tokens += slot.drafts.size();
            slot.print_timings_tg();
        }
    }

    size_t trim_rr = 0;   // round-robin start of the row trim
    std::vector<p100e::decode_res> dec_outs;   // update_slots' decode results (buffers reused step to step)

    void update_slots() {
        bool all_idle = true;
        for (auto & slot : slots) {
            if (slot.is_processing()) {
                all_idle = false;
                break;
            }
        }
        if (all_idle) {
            return;
        }
        {
            server_task task(SERVER_TASK_TYPE_NEXT_RESPONSE);
            task.id = queue_tasks.get_new_id();
            queue_tasks.post(std::move(task));
        }

        // new tasks: open a sequence, signal the stream start
        for (auto & slot : slots) {
            if (slot.state == SLOT_STATE_STARTED) {
                guarded(slot, [&] { start_slot(slot); });
            }
        }

        // one prompt chunk (the first slot with a pending prompt), so cancels and new tasks are seen between chunks.
        // Decode share: while other slots generate, a chunk waits until they have had decode_share of the wall time
        // since the previous chunk (at least one step), so a long prefill no longer drops them to one step per
        // 2048-row chunk (~0.5-3 t/s on the P100). Only the order of chunks and steps changes, never their content.
        bool generating = false;
        for (const auto & slot : slots) {
            generating = generating || slot.state == SLOT_STATE_GENERATING;
        }
        // Which prompt: the one with the fewest rows left (P100E_PROMPT_ORDER=first: the lowest slot, the old order),
        // so a short tool-result turn is not queued behind another chat's whole 50K-row prefill (live log task 6037:
        // 73 rows took 41.9 s behind slot 0's 20.9K prompt).
        if (!generating || (dec_since_chunk > 0 && dec_owed <= 0.0)) {
            server_slot * ps = nullptr;
            for (auto & slot : slots) {
                if (slot.state != SLOT_STATE_PROCESSING_PROMPT) {
                    continue;
                }
                if (!ps || (prompt_srf() && slot.prompt_rows() - slot.n_prompt_done < ps->prompt_rows() - ps->n_prompt_done)) {
                    ps = &slot;
                }
            }
            if (ps) {
                const int64_t t0 = ggml_time_us();
                guarded(*ps, [&] { prompt_step(*ps); });
                const double sh = decode_share();
                dec_owed = generating && sh > 0.0 ? (ggml_time_us() - t0) / 1e6 * sh / (1.0 - sh) : 0.0;
                dec_since_chunk = 0;
            }
        }

        // one decode step for every generating slot (a batched engine runs them in one forward)
        std::vector<server_slot *> gen;
        std::vector<p100e::decode_req> reqs;
        for (auto & slot : slots) {
            if (slot.state != SLOT_STATE_GENERATING) {
                continue;
            }
            p100e::decode_req rq;
            rq.seq = slot.seq;
            rq.tokens.push_back(slot.sampled);
            rq.tokens.insert(rq.tokens.end(), slot.drafts.begin(), slot.drafts.end());
            rq.logits = !use_gpu_argmax(slot);
            if (rq.logits) {
                slot.all_greedy = false;
            }
            gen.push_back(&slot);
            reqs.push_back(std::move(rq));
        }
        if (gen.empty()) {
            return;
        }
        {   // the engine's batched step takes at most n_rows_max rows: trim the longest draft lists first
            // (several slots and --p100e-draft-rows N: at most max(slots, N) rows, ties trimmed round robin)
            int rows = 0;
            for (const auto & rq : reqs) rows += (int) rq.tokens.size();
            int cap = eng->info().n_rows_max;
            const int ns = (int) reqs.size();
            if (ns > 1 && g_p100e_opts.draft_rows > 0) cap = std::min(cap, std::max(ns, g_p100e_opts.draft_rows));
            ++trim_rr;
            // P100E_DRAFT_TRIM=conf: cut the last draft with the lowest cumulative draft-head probability first (the
            // drafts least likely to be accepted: most expected accepted drafts per verified row). Default (unset):
            // the longest list first, ties round robin. Only which drafts get verified changes, never the tokens.
            static const bool trim_conf = [] { const char * v = getenv("P100E_DRAFT_TRIM"); return v && !strcmp(v, "conf"); }();
            auto lastp = [&](size_t i) {   // p of slot i's last draft (unknown: 1)
                const server_slot & sl = *gen[i];
                const size_t nd = sl.drafts.size();
                return nd > 0 && sl.draft_p.size() >= nd ? sl.draft_p[nd - 1] : 1.0f;
            };
            while (rows > cap) {
                size_t w = trim_rr % reqs.size();
                if (trim_conf) {
                    bool have = false;
                    for (size_t c = 0; c < reqs.size(); ++c) {
                        const size_t i = (trim_rr + c) % reqs.size();
                        if (reqs[i].tokens.size() <= 1) continue;
                        if (!have || lastp(i) < lastp(w)) { w = i; have = true; }
                    }
                } else {
                    for (size_t c = 1; c < reqs.size(); ++c) {
                        const size_t i = (trim_rr + c) % reqs.size();
                        if (reqs[i].tokens.size() > reqs[w].tokens.size()) w = i;
                    }
                }
                if (reqs[w].tokens.size() <= 1) break;
                reqs[w].tokens.pop_back();
                gen[w]->drafts.pop_back();
                if (gen[w]->draft_p.size() > gen[w]->drafts.size()) gen[w]->draft_p.resize(gen[w]->drafts.size());
                gen[w]->stats.n_draft_tokens -= 1;   // counted when drafted, never verified
                gen[w]->n_draft_trimmed += 1;
                rows--;
            }
        }
        std::vector<p100e::decode_res> & outs = dec_outs;   // kept across steps: the engine reuses its logits buffers
        const int64_t t_dec0 = ggml_time_us();
        for (;;) {
            try {
                eng->decode(reqs, outs);
                break;
            } catch (const p100e::engine_fatal_error & e) {
                p100e_engine_fatal(e.what());
            } catch (const p100e::engine_seq_error & e) {
                // one sequence has no room (KV pool full / context full) and nothing ran: fail that slot alone, the
                // step is repeated for the others (its release closes the sequence, which can then make room)
                size_t w = gen.size();
                for (size_t i = 0; i < gen.size(); ++i) {
                    if (gen[i]->seq == e.seq) {
                        w = i;
                    }
                }
                if (w == gen.size()) {
                    SRV_ERR("decode failed: %s\n", e.what());
                    for (auto * s : gen) {
                        send_error(*s, std::string("decode failed: ") + e.what(), ERROR_TYPE_SERVER);
                        release(*s);
                    }
                    return;
                }
                SLT_ERR(*gen[w], "decode failed: %s\n", e.what());
                send_error(*gen[w], std::string("decode failed: ") + e.what(), ERROR_TYPE_SERVER);
                release(*gen[w]);
                gen.erase(gen.begin() + w);
                reqs.erase(reqs.begin() + w);
                if (gen.empty()) {
                    return;
                }
            } catch (const std::exception & e) {
                SRV_ERR("decode failed: %s\n", e.what());
                for (auto * s : gen) {
                    send_error(*s, std::string("decode failed: ") + e.what(), ERROR_TYPE_SERVER);
                    release(*s);
                }
                return;
            }
        }
        metrics_post_forward(gen, reqs);
        for (size_t i = 0; i < gen.size(); ++i) {
            server_slot & slot = *gen[i];
            if (slot.state != SLOT_STATE_GENERATING) {
                continue;
            }
            guarded(slot, [&] { finish_step(slot, reqs[i], outs[i]); });
        }
        draft_step(gen);
        dec_owed -= (ggml_time_us() - t_dec0) / 1e6;   // the step incl. its keep / draft work
        dec_since_chunk++;
    }

    // decode-share state of update_slots (see there)
    double dec_owed = 0.0;
    int dec_since_chunk = 1;
    static bool prompt_srf() {
        static const bool srf = !getenv("P100E_PROMPT_ORDER") || strcmp(getenv("P100E_PROMPT_ORDER"), "first") != 0;
        return srf;
    }
    static double decode_share() {
        static const double sh = [] {
            double v = g_p100e_opts.decode_share;
            if (const char * e = getenv("P100E_DECODE_SHARE")) v = atof(e);
            return std::min(0.95, std::max(0.0, v));
        }();
        return sh;
    }

    server_response_reader get_response_reader() {
        return server_response_reader(queue_tasks, queue_results, HTTP_POLLING_SECONDS);
    }

    // /metrics after one engine forward - a prompt chunk or a decode step (llama-server's metrics_post_decode counts
    // one llama_decode the same way): n_decode, the processing slots during it (n_busy_slots_per_decode), the most
    // tokens a slot holds (n_tokens_max; a decode step's rows count until keep() resolves them, as the fork's
    // slot.prompt holds the drafts during the verify decode)
    void metrics_post_forward(const std::vector<server_slot *> & gen = {}, const std::vector<p100e::decode_req> & reqs = {}) {
        metrics.n_decode++;
        for (const server_slot & slot : slots) {
            if (slot.is_processing()) {
                metrics.n_busy_slots++;
            }
            int n_tokens = slot.n_fed;
            for (size_t i = 0; i < gen.size(); ++i) {
                if (gen[i] == &slot) {
                    n_tokens += (int) reqs[i].tokens.size();
                }
            }
            metrics.n_tokens_max = std::max(metrics.n_tokens_max, (uint64_t) n_tokens);
        }
    }

    void metrics_on_prediction(const server_slot & slot) {
        const uint64_t t_us    = slot.stats.t_gen_us();
        const uint64_t n       = slot.stats.n_gen;
        const uint64_t n_steps = slot.stats.n_gen_steps();

        metrics.predict       .add(n, n_steps, t_us);
        metrics.predict_bucket.add(n, n_steps, t_us);

        metrics.n_draft_tokens      += slot.stats.n_draft_tokens;
        metrics.n_draft_accepted    += slot.stats.n_draft_accepted;
        metrics.n_draft_verif_steps += slot.stats.n_draft_verif_steps;

        auto & dst = metrics.n_accepted_per_pos;
        const auto & src = slot.n_accepted_per_pos;
        if (dst.size() < src.size()) {
            dst.resize(src.size(), 0);
        }
        for (size_t i = 0; i < src.size(); i++) {
            dst[i] += src[i];
        }
    }
};

//
// server_context (public API)
//

server_context::server_context() : impl(new server_context_impl()) {}
server_context::~server_context() = default;

bool server_context::load_model(common_params & params) {
    return impl->load_model(params);
}

void server_context::start_loop() {
    impl->queue_tasks.start_loop(-1);   // no sleeping: the engine keeps its weights on the GPUs
}

void server_context::terminate() {
    impl->queue_tasks.terminate();
}

llama_context * server_context::get_llama_context() const {
    return nullptr;
}

server_response_reader server_context::get_response_reader() {
    return impl->get_response_reader();
}

server_context_meta server_context::get_meta() const {
    auto bos_id = llama_vocab_bos(impl->vocab);
    auto eos_id = llama_vocab_eos(impl->vocab);
    auto bos_token_str = bos_id != LLAMA_TOKEN_NULL ? common_token_to_piece(impl->vocab, bos_id, true) : "";
    auto eos_token_str = eos_id != LLAMA_TOKEN_NULL ? common_token_to_piece(impl->vocab, eos_id, true) : "";


    return server_context_meta {
        /* build_info             */ p100e_build_info(),
        /* model_name             */ impl->model_name,
        /* model_aliases          */ impl->model_aliases,
        /* model_tags             */ impl->model_tags,
        /* model_path             */ impl->params_base.model.path,
        /* has_mtmd               */ impl->mctx != nullptr || impl->has_vision(),
        /* has_inp_image          */ impl->chat_params.allow_image,
        /* has_inp_audio          */ impl->chat_params.allow_audio,
        /* has_inp_video          */ impl->chat_params.allow_video,
        /* json_ui_settings       */ impl->json_ui_settings,
        /* slot_n_ctx             */ impl->n_ctx_slot_v,
        /* pooling_type           */ LLAMA_POOLING_TYPE_NONE,

        /* chat_params            */ impl->chat_params,
        /* chat_template_caps     */ common_chat_templates_get_caps(impl->chat_params.tmpls.get()),

        /* bos_token_str          */ bos_token_str,
        /* eos_token_str          */ eos_token_str,
        /* fim_pre_token          */ llama_vocab_fim_pre(impl->vocab),
        /* fim_sub_token          */ llama_vocab_fim_suf(impl->vocab),
        /* fim_mid_token          */ llama_vocab_fim_mid(impl->vocab),
        /* fim_pad_token          */ llama_vocab_fim_pad(impl->vocab),
        /* fim_rep_token          */ llama_vocab_fim_rep(impl->vocab),
        /* fim_sep_token          */ llama_vocab_fim_sep(impl->vocab),

        /* logit_bias_eog         */ impl->params_base.sampling.logit_bias_eog,

        /* model_vocab_type       */ llama_vocab_type(impl->vocab),
        /* model_vocab_n_tokens   */ llama_vocab_n_tokens(impl->vocab),
        /* model_n_ctx_train      */ impl->n_ctx_train,
        /* model_n_embd_inp       */ impl->n_embd,
        /* model_n_params         */ impl->n_params,
        /* model_size             */ impl->model_size,
        /* model_ftype            */ impl->ftype_name,
    };
}

// generator-like API for HTTP response generation
struct server_res_generator : server_res_spipe {
    server_response_reader rd;
    server_res_generator(server_queue & queue_tasks, server_response & queue_results, int sleep_idle_seconds, bool bypass_sleep = false)
            : rd(queue_tasks, queue_results, HTTP_POLLING_SECONDS) {
        bypass_sleep |= sleep_idle_seconds < 0;
        if (!bypass_sleep) {
            queue_tasks.wait_until_no_sleep();
        }
    }
    void ok(const json & response_data) {
        status = 200;
        data = safe_json_to_str(response_data);
    }
    void error(const json & error_data) {
        status = json_value(error_data, "code", 500);
        data = safe_json_to_str({{ "error", error_data }});
    }
};

void server_context::set_state_callback(server_state_callback_t callback) {
    impl->callback_state = std::move(callback);
}

//
// server_routes (fork server-context.cpp, MIT; changes marked p100e)
//

std::unique_ptr<server_res_generator> server_routes::handle_completions_impl(
            const server_http_req & req,
            server_task_type type,
            const json & data,
            const std::vector<raw_buffer> & files,
            task_response_type res_type) {
    GGML_ASSERT(type == SERVER_TASK_TYPE_COMPLETION || type == SERVER_TASK_TYPE_INFILL);

    auto res = create_response();
    auto completion_id = gen_chatcmplid();
    auto & rd = res->rd;
    auto & params = this->params;

    res->set_req(&req);

    int32_t sse_ping_interval = params.sse_ping_interval;

    // p100e: the route's image_url.detail values / n-gram class hint, taken before anything can throw, so a failed
    // request never leaves them on this pooled HTTP thread for the next request
    std::vector<std::string> image_detail = std::move(p100e_tls_image_detail);
    p100e_tls_image_detail.clear();
    const std::string ng_hint = std::move(p100e_tls_class_hint);
    p100e_tls_class_hint.clear();

    try {
        std::vector<server_task> tasks;

        const auto & prompt = data.at("prompt");

        if (!params.path_prompts_log_dir.empty()) {
            const auto file_path = std::filesystem::path(params.path_prompts_log_dir) / string_format("%012" PRId64 ".txt", ggml_time_ms());
            std::ofstream f(file_path);
            if (f) {
                f << (prompt.is_string() ? prompt.get<std::string>().c_str() : prompt.dump(2).c_str());
            } else {
                SRV_ERR("failed to create %s\n", file_path.string().c_str());
            }
        }

        std::vector<server_tokens> inputs;

        if (res_type != TASK_RESPONSE_TYPE_NONE && ctx_server.mctx != nullptr) {
            inputs.push_back(process_mtmd_prompt(ctx_server.mctx, prompt.get<std::string>(), files, ctx_server.init_opt));
        } else {
            inputs = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, true, true, ctx_server.init_opt);
        }

        // p100e: images are encoded in the task loop (launch_slot_with_task), not on this HTTP thread
        if (!files.empty()) {
            if (!ctx_server.has_vision()) {
                throw std::runtime_error("image input is not supported - hint: if this is unexpected, you may need to provide the mmproj");
            }
            if (!prompt.is_string() || inputs.size() != 1) {
                throw std::runtime_error("for mtmd, input prompt must be a string.");
            }
        }

        json delims = json_value(data, "message_delimiters", json::array());
        auto delimiters = common_chat_msg_delimiters_parse(delims);
        delimiters.tokenize(ctx_server.vocab);

        for (size_t i = 0; i < inputs.size(); i++) {
            server_task task = server_task(type);

            task.id = rd.get_new_id();

            task.tokens = std::move(inputs[i]);
            if (!files.empty()) {
                task.cli_prompt = prompt.get<std::string>();
                task.cli_files  = files;
                std::string d;
                for (size_t k = 0; k < image_detail.size(); ++k) {
                    d += (k ? "," : "") + image_detail[k];
                }
                task.slot_action.filename = d;   // unused by completion tasks: the image_url.detail values
            }
            task.params = server_schema::eval_llama_cmpl_schema(
                    ctx_server.vocab,
                    params,
                    meta->logit_bias_eog,
                    data);

            task.params.message_spans = task.tokens.find_message_spans(delimiters);

            task.id_slot = json_value(data, "id_slot", -1);
            sse_ping_interval = task.params.sse_ping_interval;

            task.params.res_type          = res_type;
            task.params.oaicompat_cmpl_id = completion_id;
            if (!ng_hint.empty()) {
                std::lock_guard<std::mutex> lk(p100e_ng_hint_mx);
                if (p100e_ng_hint.size() > 256) p100e_ng_hint.clear();
                p100e_ng_hint[completion_id] = ng_hint;
            }
            task.params.oaicompat_model   = meta->model_name;

            if (task.params.n_cmpl > 1) {
                int n_children = task.params.n_cmpl - 1;
                for (int j = 0; j < n_children; j++) {
                    task.add_child(task.id, rd.get_new_id());
                }
            }

            tasks.push_back(std::move(task));
        }

        rd.post_tasks(std::move(tasks));
    } catch (const std::exception & e) {
        res->error(format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    bool stream = json_value(data, "stream", false);

    if (!stream) {
        auto all_results = rd.wait_for_all(req.should_stop);
        if (all_results.is_terminated) {
            return res;
        } else if (all_results.error) {
            res->error(all_results.error->to_json());
            return res;
        } else {
            json arr = json::array();
            for (auto & res : all_results.results) {
                GGML_ASSERT(dynamic_cast<server_task_result_cmpl_final*>(res.get()) != nullptr);
                arr.push_back(res->to_json());
            }
            GGML_ASSERT(!arr.empty() && "empty results");
            if (arr.size() == 1) {
                res->ok(arr[0]);
            } else if (res_type == TASK_RESPONSE_TYPE_OAI_CHAT || res_type == TASK_RESPONSE_TYPE_OAI_CMPL) {
                json & choices = arr[0]["choices"];
                for (size_t i = 1; i < arr.size(); i++) {
                    choices.push_back(std::move(arr[i]["choices"][0]));
                }
                res->ok(arr[0]);
            } else {
                res->ok(arr);
            }
        }
    } else {
        auto first_result = rd.next(req.should_stop);
        if (first_result == nullptr) {
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (first_result->is_error()) {
            res->error(first_result->to_json());
            return res;
        }

        GGML_ASSERT(
            dynamic_cast<server_task_result_cmpl_partial*>(first_result.get()) != nullptr ||
            dynamic_cast<server_task_result_cmpl_final*>  (first_result.get()) != nullptr
        );

        json first_result_json = first_result->to_json();
        if (first_result_json == nullptr) {
            res->data = "";
        } else if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
            res->data = format_anthropic_sse(first_result_json);
        } else if (res_type == TASK_RESPONSE_TYPE_OAI_RESP) {
            res->data = format_oai_resp_sse(first_result_json);
        } else {
            res->data = format_oai_sse(first_result_json);
        }
        res->status = 200;
        res->content_type = "text/event-stream";
        res->set_next([res_this = res.get(), res_type, sse_ping_interval](std::string & output) -> bool {
            static auto format_error = [](task_response_type res_type, const json & res_json) {
                if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
                    return format_anthropic_sse({
                        {"event", "error"},
                        {"data", res_json},
                    });
                } else {
                    return format_oai_sse(json {{ "error", res_json }});
                }
            };

            auto effective_should_stop = [&res_this]() {
                return res_this->should_stop();
            };

            try {
                if (effective_should_stop()) {
                    return false;
                }

                if (!res_this->data.empty()) {
                    output = std::move(res_this->data);
                    res_this->data.clear();
                    return true;
                }

                server_response_reader & rd = res_this->rd;

                if (!rd.has_next()) {
                    switch (res_type) {
                        case TASK_RESPONSE_TYPE_NONE:
                        case TASK_RESPONSE_TYPE_OAI_RESP:
                        case TASK_RESPONSE_TYPE_ANTHROPIC:
                            output = "";
                            break;

                        default:
                            output = "data: [DONE]\n\n";
                            break;
                    }
                    return false;
                }

                bool timeout = false;
                int64_t start_time = ggml_time_ms();
                auto result = rd.next([&timeout, &start_time, sse_ping_interval, &effective_should_stop]() {
                    if (effective_should_stop()) {
                        return true;
                    } else if (sse_ping_interval > 0 && ggml_time_ms() - start_time > (int64_t)sse_ping_interval * 1000) {
                        timeout = true;
                        return true;
                    }
                    return false;
                });

                if (timeout) {
                    output = ":\n\n";
                    return true;
                }

                if (result == nullptr) {
                    GGML_ASSERT(effective_should_stop());
                    return false;
                }

                if (result->is_error()) {
                    json res_json = result->to_json();
                    output = format_error(res_type, res_json);
                    return false;
                } else {
                    GGML_ASSERT(
                        dynamic_cast<server_task_result_cmpl_partial*>(result.get()) != nullptr
                        || dynamic_cast<server_task_result_cmpl_final*>(result.get()) != nullptr
                    );
                    json res_json = result->to_json();
                    if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
                        output = format_anthropic_sse(res_json);
                    } else if (res_type == TASK_RESPONSE_TYPE_OAI_RESP) {
                        output = format_oai_resp_sse(res_json);
                    } else {
                        output = format_oai_sse(res_json);
                    }
                }

                return true;

            } catch (const std::exception & e) {
                json error_json = format_error_response(e.what(), ERROR_TYPE_SERVER);
                output = format_error(res_type, error_json);
                return false;
            }
        });
    }

    return res;
}

std::unique_ptr<server_res_generator> server_routes::create_response(bool bypass_sleep) {
    return std::make_unique<server_res_generator>(queue_tasks, queue_results, params.sleep_idle_seconds, bypass_sleep);
}

server_routes::server_routes(const common_params & params, server_context & ctx_server)
        : params(params),
          ctx_server(*ctx_server.impl),
          queue_tasks(ctx_server.impl->queue_tasks),
          queue_results(ctx_server.impl->queue_results) {
    init_routes();

    queue_tasks.on_sleeping_state([this](bool is_sleeping) {
        update_cached_responses(is_sleeping);
    });
}

static json get_res_model_info(const server_context_meta & meta) {
    return {
        {"id",       meta.model_name},
        {"aliases",  meta.model_aliases},
        {"tags",     meta.model_tags},
        {"object",   "model"},
        {"created",  std::time(0)},
        {"owned_by", "llamacpp"},
        {"meta",     {
            {"vocab_type",  meta.model_vocab_type},
            {"n_vocab",     meta.model_vocab_n_tokens},
            {"n_ctx",       meta.slot_n_ctx},
            {"n_ctx_train", meta.model_n_ctx_train},
            {"n_embd",      meta.model_n_embd_inp},
            {"n_params",    meta.model_n_params},
            {"size",        meta.model_size},
            {"ftype",       meta.model_ftype},
        }},
    };
}

static json get_res_models(const server_context_meta & meta) {
    return json{
        {"models", json::array({
            {
                {"name",  meta.model_name},
                {"model", meta.model_name},
                {"modified_at", ""},
                {"size", ""},
                {"digest", ""},
                {"type", "model"},
                {"description", ""},
                {"tags", json::array({""})},
                {"capabilities", meta.has_mtmd ? json::array({"completion","multimodal"}) : json::array({"completion"})},
                {"parameters", ""},
                {"details", {
                    {"parent_model", ""},
                    {"format", "gguf"},
                    {"family", ""},
                    {"families", json::array({""})},
                    {"parameter_size", ""},
                    {"quantization_level", ""}
                }}
            }
        })},
        {"object", "list"},
        {"data", json::array({
            get_res_model_info(meta),
        })}
    };
}

static json get_res_props(const server_context_meta & meta, const common_params & params, bool is_sleeping) {
    task_params tparams;
    tparams.sampling = params.sampling;
    json default_generation_settings_for_props = json {
        { "params", tparams.to_json(true) },
        { "n_ctx",  meta.slot_n_ctx },
    };

    std::string tmpl_default = common_chat_templates_source(meta.chat_params.tmpls.get(), "");
    std::string tmpl_tools   = common_chat_templates_source(meta.chat_params.tmpls.get(), "tool_use");

    json props = {
        { "default_generation_settings", default_generation_settings_for_props },
        { "total_slots",                 params.n_parallel },
        { "model_alias",                 meta.model_name },
        { "model_ftype",                 meta.model_ftype },
        { "model_path",                  meta.model_path },
        { "modalities",                  json {
            {"vision", meta.has_inp_image},
            {"video",  meta.has_inp_video},
            {"audio",  meta.has_inp_audio},
        } },
        { "media_marker",                get_media_marker() },
        { "endpoint_slots",              params.endpoint_slots },
        { "endpoint_props",              params.endpoint_props },
        { "endpoint_metrics",            params.endpoint_metrics },
        { "ui",                          params.ui },
        { "ui_settings",                 meta.json_ui_settings },
        { "chat_template",               tmpl_default },
        { "chat_template_caps",          meta.chat_template_caps },
        { "bos_token",                   meta.bos_token_str },
        { "eos_token",                   meta.eos_token_str },
        { "build_info",                  meta.build_info },
        { "is_sleeping",                 is_sleeping },
        { "cors_proxy_enabled",          params.ui_mcp_proxy },
    };
    if (params.use_jinja) {
        if (!tmpl_tools.empty()) {
            props["chat_template_tool_use"] = tmpl_tools;
        }
    }

    return props;
}

json server_routes::get_model_info() const {
    return get_res_model_info(*meta);
}

// p100e: honour the thinking budget. Some proxies put reasoning_budget_tokens at the top level of
// the /v1/messages body; llama.cpp's Anthropic converter rebuilds the body and drops
// it, so the template's budget was never applied. Carry the budget fields over to the converted request, where
// oaicompat_chat_params_parse arms the reasoning-budget sampler with them.
static void p100e_keep_thinking_budget(const json & in, json & out) {
    for (const char * k : {"reasoning_budget_tokens", "thinking_budget_tokens", "reasoning_budget_message", "reasoning_control"}) {
        if (in.contains(k) && !in.at(k).is_null()) {
            out[k] = in.at(k);
        }
    }
}

void server_routes::init_routes() {
    this->get_health = [this](const server_http_req &) {
        auto res = create_response(true);
        res->ok({{"status", "ok"}});
        return res;
    };

    this->get_metrics = [this](const server_http_req & req) {
        auto res = create_response(true);
        if (!params.endpoint_metrics) {
            res->error(format_error_response("This server does not support metrics endpoint. Start it with `--metrics`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }
        {   // p100e: from the task loop's live snapshot, without waiting for the loop
            server_task_result_metrics live;
            std::string extra;
            if (p100e_live_metrics(live, extra)) {
                res->headers["Process-Start-Time-Unix"] = std::to_string(live.metrics.t_start);
                res->content_type = "text/plain; version=0.0.4";
                res->status = 200;
                res->data = live.to_metrics() + extra;
                return res;
            }
        }

        {
            server_task task(SERVER_TASK_TYPE_METRICS);
            task.id = res->rd.get_new_id();
            task.metrics_reset_bucket = true;
            res->rd.post_task(std::move(task), true);
        }

        auto result = res->rd.next(req.should_stop);
        if (!result) {
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        auto res_task = dynamic_cast<server_task_result_metrics*>(result.get());
        GGML_ASSERT(res_task != nullptr);

        res->headers["Process-Start-Time-Unix"] = std::to_string(res_task->metrics.t_start);
        res->content_type = "text/plain; version=0.0.4";
        res->status = 200;
        res->data = res_task->to_metrics();
        {
            std::lock_guard<std::mutex> lk(p100e_cache_metrics_mx);
            res->data += p100e_cache_metrics;
        }

        return res;
    };

    this->get_slots = [this](const server_http_req & req) {
        auto res = create_response();
        if (!params.endpoint_slots) {
            res->error(format_error_response("This server does not support slots endpoint. Start it with `--slots`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }
        {   // p100e: from the task loop's live snapshot, without waiting for the loop
            server_task_result_slots live;
            if (p100e_live_slots(live.slots_data, live.n_idle_slots)) {
                if (!req.get_param("fail_on_no_slot").empty() && live.n_idle_slots == 0) {
                    res->error(format_error_response("no slot available", ERROR_TYPE_UNAVAILABLE));
                    return res;
                }
                res->ok(live.to_json());
                return res;
            }
        }

        {
            server_task task(SERVER_TASK_TYPE_SLOT_GET);
            task.id = res->rd.get_new_id();
            res->rd.post_task(std::move(task), true);
        }

        auto result = res->rd.next(req.should_stop);
        if (!result) {
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        auto * res_task = dynamic_cast<server_task_result_slots*>(result.get());
        GGML_ASSERT(res_task != nullptr);

        if (!req.get_param("fail_on_no_slot").empty()) {
            if (res_task->n_idle_slots == 0) {
                res->error(format_error_response("no slot available", ERROR_TYPE_UNAVAILABLE));
                return res;
            }
        }

        res->ok(res_task->to_json());
        return res;
    };

    this->post_slots = [this](const server_http_req &) {
        auto res = create_response();
        res->error(format_error_response("slot save / restore / erase is not supported by p100e-server", ERROR_TYPE_NOT_SUPPORTED));
        return res;
    };

    this->get_props = [this](const server_http_req &) {
        auto res = create_response(true);
        res->ok(get_res_props(*meta, params, false));
        return res;
    };

    this->post_props = [this](const server_http_req &) {
        auto res = create_response();
        if (!params.endpoint_props) {
            res->error(format_error_response("This server does not support changing global properties. Start it with `--props`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }
        res->ok({{ "success", true }});
        return res;
    };

    this->post_infill = [this](const server_http_req & req) {
        auto res = create_response();
        std::string err;
        if (llama_vocab_fim_pre(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "prefix token is missing. ";
        }
        if (llama_vocab_fim_suf(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "suffix token is missing. ";
        }
        if (llama_vocab_fim_mid(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "middle token is missing. ";
        }
        if (!err.empty()) {
            res->error(format_error_response(string_format("Infill is not supported by this model: %s", err.c_str()), ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        json data = json::parse(req.body);
        if (data.contains("prompt") && !data.at("prompt").is_string()) {
            res->error(format_error_response("\"prompt\" must be a string", ERROR_TYPE_INVALID_REQUEST));
        }

        if (!data.contains("input_prefix")) {
            res->error(format_error_response("\"input_prefix\" is required", ERROR_TYPE_INVALID_REQUEST));
        }

        if (!data.contains("input_suffix")) {
            res->error(format_error_response("\"input_suffix\" is required", ERROR_TYPE_INVALID_REQUEST));
        }

        if (data.contains("input_extra") && !data.at("input_extra").is_array()) {
            res->error(format_error_response("\"input_extra\" must be an array of {\"filename\": string, \"text\": string}", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        json input_extra = json_value(data, "input_extra", json::array());
        for (const auto & chunk : input_extra) {
            if (!chunk.contains("text") || !chunk.at("text").is_string()) {
                res->error(format_error_response("extra_context chunk must contain a \"text\" field with a string value", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
            if (chunk.contains("filename") && !chunk.at("filename").is_string()) {
                res->error(format_error_response("extra_context chunk's \"filename\" field must be a string", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
        }
        data["input_extra"] = input_extra;

        std::string prompt = json_value(data, "prompt", std::string());
        std::vector<server_tokens> tokenized_prompts = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, false, true, ctx_server.init_opt);
        data["prompt"] = format_prompt_infill(
            ctx_server.vocab,
            data.at("input_prefix"),
            data.at("input_suffix"),
            data.at("input_extra"),
            params.n_batch,
            params.n_predict,
            meta->slot_n_ctx,
            params.spm_infill,
            tokenized_prompts[0].get_tokens()
        );

        std::vector<raw_buffer> files;
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_INFILL,
            data,
            files,
            TASK_RESPONSE_TYPE_NONE);
    };

    this->post_completions = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        const json body = json::parse(req.body);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body,
            files,
            TASK_RESPONSE_TYPE_NONE);
    };

    this->post_completions_oai = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        const json body = json::parse(req.body);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body,
            files,
            TASK_RESPONSE_TYPE_OAI_CMPL);
    };

    this->post_chat_completions = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = json::parse(req.body);
        std::vector<std::string> detail = p100e_image_details(body);   // the parse drops image_url.detail
        p100e_check_remote_media(body);
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        p100e_tls_image_detail = std::move(detail);
        if (p100e_ng::enabled()) {   // request class hint for the n-gram shadow log: tools / last message role
            std::string h;
            if (body.contains("tools") && body.at("tools").is_array() && !body.at("tools").empty()) h += "tools" + std::to_string(body.at("tools").size()) + ",";
            if (body.contains("messages") && body.at("messages").is_array() && !body.at("messages").empty()) {
                const json & m = body.at("messages").back();
                h += "last=" + (m.contains("role") && m.at("role").is_string() ? m.at("role").get<std::string>() : std::string("?"));
                h += ",n=" + std::to_string(body.at("messages").size());
            }
            p100e_tls_class_hint = h;
        }
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_CHAT);
    };

    this->post_chat_completions_tok = [this](const server_http_req & req) {
        return handle_count_tokens(ctx_server.vocab, ctx_server.mctx, ctx_server.init_opt, req, TASK_RESPONSE_TYPE_OAI_CHAT);
    };

    this->post_control = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);

        const std::string cmpl_id = json_value(body, "id", std::string());
        const std::string action  = json_value(body, "action", std::string());
        if (cmpl_id.empty()) {
            res->error(format_error_response("missing completion id", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        if (action != "reasoning_end") {
            res->error(format_error_response("unknown control action", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_CONTROL);
            task.id              = rd.get_new_id();
            task.params.control_cmpl_id = cmpl_id;
            task.params.control_action  = action;
            rd.post_task(std::move(task));
        }

        auto result = rd.next(req.should_stop);
        if (!result) {
            GGML_ASSERT(req.should_stop());
            return res;
        }
        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }
        res->ok(result->to_json());
        return res;
    };

    this->post_responses_oai = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        const json in = json::parse(req.body);
        json body = server_chat_convert_responses_to_chatcmpl(in);
        std::vector<std::string> detail = p100e_responses_image_details(in, body);   // the conversion drops "detail"
        p100e_check_remote_media(body);
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        p100e_tls_image_detail = std::move(detail);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_RESP);
    };

    this->post_responses_tok_oai = [this](const server_http_req & req) {
        return handle_count_tokens(ctx_server.vocab, ctx_server.mctx, ctx_server.init_opt, req, TASK_RESPONSE_TYPE_OAI_RESP);
    };

    this->post_transcriptions_oai = [this](const server_http_req &) {
        auto res = create_response();
        res->error(format_error_response("The current model does not support audio input.", ERROR_TYPE_NOT_SUPPORTED));
        return res;
    };

    this->post_anthropic_messages = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        const json in = json::parse(req.body);
        json body = server_chat_convert_anthropic_to_oai(in);
        p100e_keep_thinking_budget(in, body);
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        p100e_check_remote_media(body);
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_ANTHROPIC);
    };

    this->post_anthropic_count_tokens = [this](const server_http_req & req) {
        return handle_count_tokens(ctx_server.vocab, ctx_server.mctx, ctx_server.init_opt, req, TASK_RESPONSE_TYPE_ANTHROPIC);
    };

    this->post_apply_template = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = json::parse(req.body);
        p100e_check_remote_media(body);
        json data = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        res->ok({{ "prompt", std::move(data.at("prompt")) }});
        return res;
    };

    this->get_models = [this](const server_http_req &) {
        auto res = create_response(true);
        res->ok(get_res_models(*meta));
        return res;
    };

    this->post_tokenize = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);
        json tokens_response = json::array();
        if (body.count("content") != 0) {
            const bool add_special = json_value(body, "add_special", false);
            const bool parse_special = json_value(body, "parse_special", true);
            const bool with_pieces = json_value(body, "with_pieces", false);

            llama_tokens tokens = tokenize_mixed(ctx_server.vocab, body.at("content"), add_special, parse_special);

            if (with_pieces) {
                for (const auto& token : tokens) {
                    std::string piece = common_token_to_piece(ctx_server.vocab, token);
                    json piece_json;

                    if (is_valid_utf8(piece)) {
                        piece_json = piece;
                    } else {
                        piece_json = json::array();
                        for (unsigned char c : piece) {
                            piece_json.push_back(static_cast<int>(c));
                        }
                    }

                    tokens_response.push_back({
                        {"id", token},
                        {"piece", piece_json}
                    });
                }
            } else {
                tokens_response = tokens;
            }
        }

        res->ok(json{{"tokens", std::move(tokens_response)}});
        return res;
    };

    this->post_detokenize = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);

        std::string content;
        if (body.count("tokens") != 0) {
            const llama_tokens tokens = body.at("tokens").get<llama_tokens>();
            content = tokens_to_str(ctx_server.vocab, tokens);
        }

        res->ok(json{{"content", std::move(content)}});
        return res;
    };

    auto not_supported = [this](const char * what) {
        return [this, what](const server_http_req &) {
            auto res = create_response();
            res->error(format_error_response(std::string(what) + " is not supported by p100e-server", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        };
    };
    this->post_embeddings     = not_supported("embeddings");
    this->post_embeddings_oai = not_supported("embeddings");
    this->post_rerank         = not_supported("reranking");

    this->get_lora_adapters = [this](const server_http_req &) {
        auto res = create_response();
        res->ok(json::array());
        return res;
    };
    this->post_lora_adapters = not_supported("LoRA adapters");
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_save(const server_http_req &, int) {
    auto res = create_response();
    res->error(format_error_response("not supported", ERROR_TYPE_NOT_SUPPORTED));
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_restore(const server_http_req &, int) {
    auto res = create_response();
    res->error(format_error_response("not supported", ERROR_TYPE_NOT_SUPPORTED));
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_erase(const server_http_req &, int) {
    auto res = create_response();
    res->error(format_error_response("not supported", ERROR_TYPE_NOT_SUPPORTED));
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_embeddings_impl(const server_http_req &, task_response_type) {
    auto res = create_response();
    res->error(format_error_response("not supported", ERROR_TYPE_NOT_SUPPORTED));
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_count_tokens(const llama_vocab * vocab, mtmd_context * mctx, const mtmd_helper_init_opt & init_opt, const server_http_req & req, task_response_type res_type) {
    auto res = create_response();
    std::vector<raw_buffer> files;
    json body = json::parse(req.body);
    bool is_oai = false;

    switch (res_type) {
        case TASK_RESPONSE_TYPE_OAI_CHAT:
            {
                is_oai = true;
            } break;
        case TASK_RESPONSE_TYPE_OAI_RESP:
            {
                is_oai = true;
                body = server_chat_convert_responses_to_chatcmpl(body);
            } break;
        case TASK_RESPONSE_TYPE_ANTHROPIC:
            {
                body = server_chat_convert_anthropic_to_oai(body);
            } break;
        default:
            res->error(format_error_response("invalid res_type", ERROR_TYPE_INVALID_REQUEST));
            return res;
    }

    p100e_check_remote_media(body);
    json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
    json prompt = body_parsed.at("prompt");

    size_t n_tokens;
    if (mctx != nullptr) {
        if (!prompt.is_string()) {
            throw std::runtime_error("for mtmd, input prompt must be a string.");
        }
        n_tokens = process_mtmd_prompt(mctx, prompt.get<std::string>(), files, init_opt, true).size();
#ifdef P100E_HAVE_VISION
    } else if (!files.empty() && ctx_server.venc) {
        // p100e: image rows from the decoded size (or the encoder's cache), no encode on this thread
        if (!prompt.is_string()) {
            throw std::runtime_error("for mtmd, input prompt must be a string.");
        }
        const std::vector<std::string> detail = is_oai ? p100e_image_details(json::parse(req.body)) : std::vector<std::string>();
        std::vector<p100e::image_input> ins(files.size());
        for (size_t i = 0; i < files.size(); ++i) {
            ins[i].bytes.assign(files[i].begin(), files[i].end());
            ins[i].max_tokens = ctx_server.venc->detail_max_tokens(i < detail.size() ? detail[i] : std::string());
        }
        n_tokens = (size_t) ctx_server.venc->count_tokens(prompt.get<std::string>(), ins, get_media_marker());
#endif
    } else {
        n_tokens = tokenize_mixed(vocab, prompt, true, true).size();
    }

    json response = {{"input_tokens", static_cast<int64_t>(n_tokens)}};
    if (is_oai) {
        response["object"] = "response.input_tokens";
    }
    res->ok(response);
    return res;
}

void server_routes::update_cached_responses(bool is_sleeping) {
    std::unique_lock<std::mutex> lock(mutex_cache);
    if (is_sleeping) {
        cached_models  = get_res_models(*meta);
        cached_props   = get_res_props(*meta, params, true);
        cached_metrics = ctx_server.get_metrics();
        should_reset_buckets = false;
    } else if (should_reset_buckets) {
        ctx_server.reset_metrics_bucket();
        should_reset_buckets = false;
    }
}
