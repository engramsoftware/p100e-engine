// p100e-server: llama-server-compatible HTTP server on the p100e CUDA engine.
//
// Command line: llama-server's own arguments (parsed by the fork's common_params_parse, so existing llama-server flags
// keep working: -m, --host, --port, --alias, -c, -np, -ub, --jinja, --chat-template-file, --log-file, ...), plus the
// --p100e-* flags below, which are removed before llama.cpp's parser sees the rest. Flags that select llama.cpp
// backends (--split-mode, -ngl, --flash-attn, --spec-*, --cache-type-v, ...) are accepted and ignored; the engine is
// configured by --p100e-*.
//   --p100e-devices 0,1        GPUs (tensor parallel over 1 to 4; Qwen3.8-27B splits evenly over 1, 2 or 4, over 3
//                              as evenly as its whole heads / 256-row blocks allow)
//   --p100e-split a,b,..       relative share per GPU of --p100e-devices (e.g. 2,1,1: the first GPU holds about half
//                              of every split dimension; mixed GPUs). Default: equal (env P100E_SPLIT when unset).
//                              A share that leaves a GPU without a whole KV head / k-head group / block is refused at load
//   --p100e-kv f16|q8_0|q4_0   attention KV type (default: -ctk when q8_0 / q4_0, else q8_0)
//   --p100e-mtp K              MTP drafts per step, adaptive up to K (0 = off; default 3)
//   --p100e-mtp-tau t          adaptive draft length threshold (default 0.5; 0 = always K)
//   --p100e-engine tp|fn       engine: tp (27B tensor parallel) or fn (Flash-Next: hot experts resident, cold
//                              streamed, QSA, MTP; one slot); default fn when general.architecture is qwen4exp
//   --p100e-fn-hot N           fn: GPU-resident experts per layer per GPU (default 72)
//   --p100e-ram-reserve-mb N   fn: host RAM kept free after load, else the load is refused (default 12288 Windows / 6144 Linux, 0 = off)
//   --p100e-fn-pipe-nodes a,b  fn: hybrid pipeline: this server is node 1 (host:port of node 0, node 1)
//   --p100e-fn-pipe-layers k   fn: node 0 runs layers [0, k) (default auto)
//   --p100e-fn-slots a,b|auto  fn: resident experts per layer per GPU (auto: what fits each GPU)
//   --p100e-fn-max-t N         fn: prefill chunk rows (default 512; 0 = -ub)
//   --p100e-no-engine          no GPU: only the text endpoints (tokenize / templates / count tokens / props)
//   --p100e-print-ids          log the prompt and generated token ids of every task
//   --p100e-refcheck           rerun every greedy task as plain greedy decoding and log IDENTICAL / MISMATCH
//                              (debug only: the replay blocks the server loop; never in production)
//   --p100e-no-greedy-gpu      greedy requests copy the logits and take the host argmax (A/B of the GPU argmax path)
//   --p100e-selftest [builtin|file.json]   serve a scripted request set to itself, print the responses, exit
//   --p100e-max-requests N     exit after N completion tasks
//   --p100e-parse-test f.json  parser goldens (request + raw model output -> parsed message), no engine; exit
//   --p100e-margin-mb N        VRAM per GPU the KV pool leaves free for transient allocations (default 512)
//   --p100e-vision-device N    GPU of the image encoder with --mmproj (default: the last engine GPU; CPU with
//                              --no-mmproj-offload or --p100e-no-engine). Image flags are llama-server's: --mmproj,
//                              --image-min-tokens (default 1024), --image-max-tokens (default 2048: the cap for
//                              image_url.detail "high" / "auto" / unset; "low" = the minimum)
//   --p100e-no-remote-images   refuse http(s) image / media URLs in requests (the server would download them: a
//                              blind SSRF from anyone who can reach the port); data: URIs still work. Default: allowed
// Exposure: listening beyond loopback without --api-key / --api-key-file, or with the fork's default CORS origin "*"
// (any web page in a LAN browser can call the API), is logged as a warning at startup; --cors-origins narrows it.
#include "server-context.h"
#include "server-common.h"
#include "server-http.h"
#include "server-stream.h"

#include "arg.h"
#include "build-info.h"
#include "common.h"
#include "llama.h"
#include "log.h"

#include "selftest.h"

int p100e_parse_test(common_params & params, const std::string & file);   // parse_test.cpp
#include "server_opts.h"

#include <atomic>
#include <clocale>
#include <cstring>
#include <exception>
#include <signal.h>
#include <sstream>
#include <thread>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#endif

p100e_server_opts g_p100e_opts;

static std::function<void(int)> shutdown_handler;
static std::atomic_flag is_terminating = ATOMIC_FLAG_INIT;

static inline void signal_handler(int signal) {
    if (is_terminating.test_and_set()) {
        fprintf(stderr, "Received second interrupt, terminating immediately.\n");
        exit(1);
    }
    shutdown_handler(signal);
}

// handlers never throw: exceptions become error responses (fork server.cpp)
static server_http_context::handler_t ex_wrapper(server_http_context::handler_t func) {
    return [func = std::move(func)](const server_http_req & req) -> server_http_res_ptr {
        std::string message;
        error_type error;
        try {
            return func(req);
        } catch (const std::invalid_argument & e) {
            error = ERROR_TYPE_INVALID_REQUEST;
            message = e.what();
        } catch (const std::exception & e) {
            error = ERROR_TYPE_SERVER;
            message = e.what();
        } catch (...) {
            error = ERROR_TYPE_SERVER;
            message = "unknown error";
        }

        auto res = std::make_unique<server_http_res>();
        res->status = 500;
        try {
            json error_data = format_error_response(message, error);
            res->status = json_value(error_data, "code", 500);
            res->data = safe_json_to_str({{ "error", error_data }});
            SRV_WRN("got exception: %s\n", res->data.c_str());
        } catch (const std::exception & e) {
            SRV_ERR("got another exception: %s | while handling exception: %s\n", e.what(), message.c_str());
            res->data = "Internal Server Error";
        }
        return res;
    };
}

// removes the --p100e-* flags from argv (into g_p100e_opts); false on a bad flag
static bool parse_p100e_args(int argc, char ** argv, std::vector<char *> & rest) {
    auto & o = g_p100e_opts;
    rest.push_back(argv[0]);
    auto need = [&](int & i) -> const char * {
        if (i + 1 >= argc) {
            fprintf(stderr, "error: %s needs a value\n", argv[i]);
            return nullptr;
        }
        return argv[++i];
    };
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a.rfind("--p100e-", 0) != 0) {
            rest.push_back(argv[i]);
            continue;
        }
        const char * v = nullptr;
        if (a == "--p100e-devices") {
            if (!(v = need(i))) return false;
            o.devices.clear();
            std::stringstream ss(v);
            std::string t;
            while (std::getline(ss, t, ',')) {
                if (t.empty()) continue;
                char * end = nullptr;
                const long d = strtol(t.c_str(), &end, 10);
                if (!end || *end || d < 0 || d > 63) {
                    fprintf(stderr, "error: --p100e-devices: '%s' is not a GPU index\n", t.c_str());
                    return false;
                }
                o.devices.push_back((int) d);
            }
            // tensor parallel over 1 to 4 GPUs (Qwen3.8-27B: 3 gets an uneven split; the engine names
            // the dimension when a model cannot give every GPU a share)
            if (o.devices.empty() || o.devices.size() > 4) {
                fprintf(stderr, "error: --p100e-devices needs 1 to 4 GPU indices (got %zu)\n", o.devices.size());
                return false;
            }
        } else if (a == "--p100e-split") {
            // relative weights per GPU (tp_model::opts::split): positive numbers, one per --p100e-devices entry (checked
            // after every flag is read)
            if (!(v = need(i))) return false;
            o.split.clear();
            std::stringstream ss(v);
            std::string t;
            while (std::getline(ss, t, ',')) {
                char * end = nullptr;
                const double w = strtod(t.c_str(), &end);
                if (t.empty() || !end || *end || !(w > 0) || w > 1e9) {
                    fprintf(stderr, "error: --p100e-split: '%s' is not a positive weight (e.g. --p100e-split 2,1,1)\n", t.c_str());
                    return false;
                }
                o.split.push_back(w);
            }
            if (o.split.empty()) {
                fprintf(stderr, "error: --p100e-split needs one weight per GPU (e.g. 2,1,1)\n");
                return false;
            }
        } else if (a == "--p100e-kv") {
            if (!(v = need(i))) return false;
            o.kv_type = v;
        } else if (a == "--p100e-mtp") {
            if (!(v = need(i))) return false;
            o.mtp_k = atoi(v);
        } else if (a == "--p100e-mtp-tau") {
            if (!(v = need(i))) return false;
            o.mtp_tau = (float) atof(v);
        } else if (a == "--p100e-engine") {
            if (!(v = need(i))) return false;
            o.engine = v;
        } else if (a == "--p100e-fn-hot") {
            if (!(v = need(i))) return false;
            o.fn_hot = atoi(v);
        } else if (a == "--p100e-ram-reserve-mb") {   // host RAM the fn engine keeps free after load (src/ram_guard.h; 0 = off)
            if (!(v = need(i))) return false;
            static std::string kv;
            kv = std::string("P100E_RAM_RESERVE_MB=") + v;
#ifdef _WIN32
            _putenv(kv.c_str());
#else
            setenv("P100E_RAM_RESERVE_MB", v, 1);
#endif
        } else if (a == "--p100e-fn-max-t") {
            if (!(v = need(i))) return false;
            o.fn_max_t = atoi(v);
        } else if (a == "--p100e-fn-pipe-nodes") {
            if (!(v = need(i))) return false;
            std::string s = v;
            for (size_t p0 = 0; p0 <= s.size();) { const size_t c = s.find(',', p0); o.fn_pipe_nodes.push_back(s.substr(p0, c == std::string::npos ? std::string::npos : c - p0)); if (c == std::string::npos) break; p0 = c + 1; }
        } else if (a == "--p100e-fn-pipe-layers") {
            if (!(v = need(i))) return false;
            o.fn_pipe_layers = !strcmp(v, "auto") ? -1 : atoi(v);
        } else if (a == "--p100e-fn-slots") {
            if (!(v = need(i))) return false;
            std::string s = v;
            for (size_t p0 = 0; p0 <= s.size();) { const size_t c = s.find(',', p0); const std::string x = s.substr(p0, c == std::string::npos ? std::string::npos : c - p0); o.fn_slots.push_back(x == "auto" ? -1 : atoi(x.c_str())); if (c == std::string::npos) break; p0 = c + 1; }
        } else if (a == "--p100e-no-engine") {
            o.no_engine = true;
        } else if (a == "--p100e-print-ids") {
            o.print_ids = true;
        } else if (a == "--p100e-refcheck") {
            o.refcheck = true;
        } else if (a == "--p100e-no-greedy-gpu") {
            o.greedy_gpu = false;
        } else if (a == "--p100e-selftest") {
            o.selftest = "builtin";
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                o.selftest = argv[++i];
            }
        } else if (a == "--p100e-parse-test") {
            if (!(v = need(i))) return false;
            o.parse_test = v;
        } else if (a == "--p100e-vision-device") {
            if (!(v = need(i))) return false;
            o.vision_device = atoi(v);
        } else if (a == "--p100e-no-remote-images") {
            o.no_remote_images = true;
        } else if (a == "--p100e-max-requests") {
            if (!(v = need(i))) return false;
            o.max_requests = atoi(v);
        } else if (a == "--p100e-mtp-multi") {
            if (!(v = need(i))) return false;
            o.mtp_multi = atoi(v);
        } else if (a == "--p100e-draft-rows") {
            if (!(v = need(i))) return false;
            o.draft_rows = atoi(v);
        } else if (a == "--p100e-cache") {
            if (!(v = need(i))) return false;
            o.cache = atoi(v) != 0;
        } else if (a == "--p100e-cache-resident") {
            if (!(v = need(i))) return false;
            o.cache_resident = atoi(v);
        } else if (a == "--p100e-host-budget-mb") {
            if (!(v = need(i))) return false;
            o.host_budget_mb = (size_t) atoll(v);
        } else if (a == "--p100e-ckpt-budget-mb") {
            if (!(v = need(i))) return false;
            o.ckpt_budget_mb = (size_t) atoll(v);
        } else if (a == "--p100e-decode-share") {
            if (!(v = need(i))) return false;
            o.decode_share = atof(v);
        } else if (a == "--p100e-pool-tokens") {
            if (!(v = need(i))) return false;
            o.pool_tokens = atoll(v);
        } else if (a == "--p100e-margin-mb") {
            if (!(v = need(i))) return false;
            o.margin_mb = atoll(v);
        } else {
            fprintf(stderr, "error: unknown p100e flag %s\n", a.c_str());
            return false;
        }
    }
    if (!o.split.empty() && o.split.size() != o.devices.size()) {
        fprintf(stderr, "error: --p100e-split has %zu weight(s) for %zu GPU(s) (--p100e-devices): give one per GPU\n",
                o.split.size(), o.devices.size());
        return false;
    }
    return true;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    std::vector<char *> rest;
    if (!parse_p100e_args(argc, argv, rest)) {
        return 1;
    }

    common_params params;

    common_init();

    server_stream_session_manager_start();

    if (!common_params_parse((int) rest.size(), rest.data(), params, LLAMA_EXAMPLE_SERVER)) {
        return 1;
    }

    llama_backend_init();

    if (params.model.path.empty()) {
        SRV_ERR("%s", "-m <model.gguf> is required\n");
        return 1;
    }
    if (!g_p100e_opts.parse_test.empty()) {
        return p100e_parse_test(params, g_p100e_opts.parse_test);
    }
    // the engine keeps its weights resident: no sleeping; no router mode
    params.sleep_idle_seconds = -1;
    {   // p100e default: 3 slots (several chats at once) unless -np / --parallel is given
        bool np_given = false;
        for (char * a : rest) {
            const std::string s = a;
            np_given = np_given || s == "-np" || s == "--parallel" || s.rfind("--parallel=", 0) == 0;
        }
        if (!np_given || params.n_parallel < 1) {
            params.n_parallel = np_given ? 1 : 3;
        }
    }

    auto model_name = params.model.get_name();
    if (params.model_alias.empty() && !model_name.empty()) {
        params.model_alias.insert(model_name);
    }

    SRV_INF("p100e-server (fork %s, build %s), engine: devices %zu, MTP %d\n", P100E_FORK_DIR_STR,
            llama_build_info(), g_p100e_opts.devices.size(), g_p100e_opts.mtp_k);
    {   // exposure beyond this machine: warnings only (existing clients keep working)
        const std::string & h = params.hostname;
        const bool local = h == "127.0.0.1" || h == "localhost" || h == "::1" || h.rfind("127.", 0) == 0 ||
                           (h.size() > 5 && h.compare(h.size() - 5, 5, ".sock") == 0);
        if (!local) {
            if (params.api_keys.empty()) {
                SRV_WRN("listening on %s with no API key: anyone who can reach the port can use the model, read /slots "
                        "and /props and end other users' reasoning (--api-key / --api-key-file)\n", h.c_str());
            }
            if (params.cors_origins == "*") {
                SRV_WRN("%s", "CORS allows every origin (*): any web page open in a browser that can reach this server can "
                              "call it (--cors-origins to narrow)\n");
            }
            if (!g_p100e_opts.no_remote_images) {
                SRV_WRN("%s", "http(s) image URLs in requests are downloaded by the server (--p100e-no-remote-images to "
                              "refuse them)\n");
            }
        }
    }

    server_context ctx_server;

    server_http_context ctx_http;
    if (!ctx_http.init(params)) {
        SRV_ERR("%s", "failed to initialize HTTP server\n");
        return 1;
    }

    server_routes routes(params, ctx_server);

    ctx_http.get ("/health",                   ex_wrapper(routes.get_health));
    ctx_http.get ("/v1/health",                ex_wrapper(routes.get_health));
    ctx_http.get ("/metrics",                  ex_wrapper(routes.get_metrics));
    ctx_http.get ("/props",                    ex_wrapper(routes.get_props));
    ctx_http.post("/props",                    ex_wrapper(routes.post_props));
    ctx_http.get ("/models",                   ex_wrapper(routes.get_models));
    ctx_http.get ("/v1/models",                ex_wrapper(routes.get_models));
    ctx_http.post("/completion",               ex_wrapper(routes.post_completions));
    ctx_http.post("/completions",              ex_wrapper(routes.post_completions));
    ctx_http.post("/v1/completions",           ex_wrapper(routes.post_completions_oai));
    ctx_http.post("/chat/completions",         ex_wrapper(routes.post_chat_completions));
    ctx_http.post("/v1/chat/completions",      ex_wrapper(routes.post_chat_completions));
    ctx_http.post("/v1/chat/completions/control", ex_wrapper(routes.post_control));
    ctx_http.post("/v1/responses",             ex_wrapper(routes.post_responses_oai));
    ctx_http.post("/responses",                ex_wrapper(routes.post_responses_oai));
    ctx_http.post("/v1/audio/transcriptions",  ex_wrapper(routes.post_transcriptions_oai));
    ctx_http.post("/audio/transcriptions",     ex_wrapper(routes.post_transcriptions_oai));
    ctx_http.post("/v1/messages",              ex_wrapper(routes.post_anthropic_messages));
    ctx_http.post("/infill",                   ex_wrapper(routes.post_infill));
    ctx_http.post("/embedding",                ex_wrapper(routes.post_embeddings));
    ctx_http.post("/embeddings",               ex_wrapper(routes.post_embeddings));
    ctx_http.post("/v1/embeddings",            ex_wrapper(routes.post_embeddings_oai));
    ctx_http.post("/rerank",                   ex_wrapper(routes.post_rerank));
    ctx_http.post("/reranking",                ex_wrapper(routes.post_rerank));
    ctx_http.post("/v1/rerank",                ex_wrapper(routes.post_rerank));
    ctx_http.post("/v1/reranking",             ex_wrapper(routes.post_rerank));
    ctx_http.post("/tokenize",                 ex_wrapper(routes.post_tokenize));
    ctx_http.post("/detokenize",               ex_wrapper(routes.post_detokenize));
    ctx_http.post("/apply-template",           ex_wrapper(routes.post_apply_template));
    ctx_http.post("/chat/completions/input_tokens",    ex_wrapper(routes.post_chat_completions_tok));
    ctx_http.post("/v1/chat/completions/input_tokens", ex_wrapper(routes.post_chat_completions_tok));
    ctx_http.post("/responses/input_tokens",           ex_wrapper(routes.post_responses_tok_oai));
    ctx_http.post("/v1/responses/input_tokens",        ex_wrapper(routes.post_responses_tok_oai));
    ctx_http.post("/v1/messages/count_tokens",         ex_wrapper(routes.post_anthropic_count_tokens));
    ctx_http.get ("/lora-adapters",            ex_wrapper(routes.get_lora_adapters));
    ctx_http.post("/lora-adapters",            ex_wrapper(routes.post_lora_adapters));
    ctx_http.get ("/slots",                    ex_wrapper(routes.get_slots));
    ctx_http.post("/slots/:id_slot",           ex_wrapper(routes.post_slots));
    ctx_http.get ("/v1/stream",                ex_wrapper(server_stream_make_get_handler()));
    ctx_http.post("/v1/streams/lookup",        ex_wrapper(server_stream_make_lookup_handler()));
    ctx_http.del ("/v1/stream",                ex_wrapper(server_stream_make_delete_handler()));

    auto clean_up = [&ctx_http, &ctx_server]() {
        SRV_INF("%s: cleaning up before exit...\n", __func__);
        server_stream_session_manager_stop();
        ctx_http.stop();
        ctx_server.terminate();
        llama_backend_free();
    };

    // HTTP first, so /health answers 503 "Loading model" while the engine loads
    if (!ctx_http.start()) {
        clean_up();
        SRV_ERR("%s", "exiting due to HTTP server error\n");
        return 1;
    }

    if (!ctx_server.load_model(params)) {
        clean_up();
        if (ctx_http.thread.joinable()) {
            ctx_http.thread.join();
        }
        SRV_ERR("%s", "exiting due to model loading error\n");
        return 1;
    }

    routes.update_meta(ctx_server);
    ctx_http.is_ready.store(true);

    SRV_INF("%s", "model loaded\n");

    shutdown_handler = [&](int) {
        ctx_server.terminate();
    };

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
    struct sigaction sigint_action;
    sigint_action.sa_handler = signal_handler;
    sigemptyset (&sigint_action.sa_mask);
    sigint_action.sa_flags = 0;
    sigaction(SIGINT, &sigint_action, NULL);
    sigaction(SIGTERM, &sigint_action, NULL);
#elif defined (_WIN32)
    auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
        return (ctrl_type == CTRL_C_EVENT) ? (signal_handler(SIGINT), true) : false;
    };
    SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif

    SRV_INF("listening on %s\n", ctx_http.listening_address.c_str());

    std::atomic<int> selftest_rc{0};
    std::thread selftest_thread;
    if (!g_p100e_opts.selftest.empty()) {
        const std::string host = params.hostname == "0.0.0.0" ? "127.0.0.1" : params.hostname;
        const int port = ctx_http.port;
        selftest_thread = std::thread([&, host, port]() {
            selftest_rc = p100e_selftest(host, port, g_p100e_opts.selftest);
            ctx_server.terminate();
        });
    }

    // blocks until terminate()
    ctx_server.start_loop();

    clean_up();
    if (selftest_thread.joinable()) {
        selftest_thread.join();
    }
    if (ctx_http.thread.joinable()) {
        ctx_http.thread.join();
    }

    return selftest_rc.load();
}
