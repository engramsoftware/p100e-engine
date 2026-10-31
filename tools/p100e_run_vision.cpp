// p100e-run image mode (--mmproj) and the --seq runner. See run_vision_usage below.
#include "../src/tp_model.h"
#include "../src/tp_mtp.h"
#ifdef P100E_HAVE_VISION
#include "vision.h"
#endif

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <cstdlib>
// the UCRT call the vision tests use: an empty value removes the variable
static int _putenv_s(const char * k, const char * v) { return *v ? setenv(k, v, 1) : unsetenv(k); }
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using namespace p100e;

static const char * run_vision_usage =
    "p100e-run <model.gguf> --devices 0,1 --mmproj <mmproj.gguf> --image <file> [--image <file> ...]\n"
    "    (--prompt-file <f> | --chat <text> | --chat-file <f>) [--gen N] [--ctx N] [--kv f16|q8_0|q4_0] [--ubatch U]\n"
    "  --prompt-file f   the rendered prompt with one <__media__> marker per image (llama-server /apply-template output\n"
    "                    with its media marker replaced by <__media__>)\n"
    "  --chat text       single user turn, thinking off: <|im_start|>user\\n<images>text<|im_end|>\\n<|im_start|>assistant\\n\n"
    "                    <think>\\n\\n</think>\\n\\n (the template with enable_thinking=false)\n"
    "  --vision-dev N    encoder GPU (default: the last of --devices); --vision-dlls dir; --vision-model f (vocab GGUF,\n"
    "                    default the model)\n"
    "  --image-min-tokens N (1024) --image-max-tokens N (2048) --detail low|high (per-image cap) --image-cap N\n"
    "  --repeat-penalty r (1.0) --repeat-last-n n (64): llama.cpp penalties (prompt text tokens count, image rows not)\n"
    "  --top K           print the top K log-probabilities (log-softmax of the raw logits) of every generated step\n"
    "  --runs N          whole request N times (KV reset each time; images come from the embedding cache after run 1)\n"
    "  --vision-cache-mb N (2048; 0 = off)   --encode-only (no LLM)   --vision-verbose\n"
    "  --no-fold         cuBLAS COMPUTE_16F for the encoder's matmuls instead of the fork's fold GEMM\n"
    "  --model-first     load the model before the encoder (the server's order: GPU peer access and the KV caches\n"
    "                    exist when ggml initializes; run it with GGML_CUDA_P2P=1 in the environment to check the\n"
    "                    encoder's GGML_* pinning). The encoder removes inherited GGML_* variables before its DLLs load\n"
    "                    (P100E_VISION_GGML_ENV=keep keeps them, except GGML_CUDA_P2P)\n"
    "  --encode-bench N  encode every image N more times with the cache cleared (encoder timing spread)\n"
    "  --overlap-bench n decode n text tokens while the encoder runs on another thread (stall check), then without\n"
    "  --mtp K[,K..] [--mtp-adaptive tau]  MTP greedy after the plain run, ids compared (IDENTICAL / MISMATCH)\n"
    "  --turn2 text      a second user turn (text only, greedy) on top of the first turn's context; with --mtp also\n"
    "                    through generate_mtp on the context MTP left (ids compared; turn-2 acceptance printed)\n"
    "  --depth D         prompt after D synthetic KV rows (timing at depth; the output is meaningless)\n"
    "  P100E_MM_PACK=0   one forward per piece instead of one forward_mm over the whole prompt\n";

#ifdef P100E_HAVE_VISION
static std::string read_file(const std::string & p) {
    FILE * f = fopen(p.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open " + p);
    std::string s;
    char buf[1 << 16];
    size_t n;
    while ((n = fread(buf, 1, sizeof buf, f)) > 0) s.append(buf, n);
    fclose(f);
    return s;
}

static double log_sum_exp(const float * l, int n) {
    float mx = -INFINITY;
    for (int i = 0; i < n; ++i) mx = std::max(mx, l[i]);
    double s = 0;
    for (int i = 0; i < n; ++i) s += std::exp((double) l[i] - mx);
    return mx + std::log(s);
}

// llama.cpp penalties sampler (repeat only): every distinct token of the last n accepted tokens
static void apply_repeat_penalty(std::vector<float> & l, const std::vector<int32_t> & hist, int last_n, float pen) {
    if (pen == 1.0f || last_n == 0) return;
    const size_t n = last_n < 0 ? hist.size() : std::min(hist.size(), (size_t) last_n);
    std::vector<int32_t> w(hist.end() - n, hist.end());
    std::sort(w.begin(), w.end());
    w.erase(std::unique(w.begin(), w.end()), w.end());
    for (int32_t id : w) {
        float & x = l[id];
        x = x <= 0 ? x * pen : x / pen;
    }
}
#endif

int run_vision(int argc, char ** argv) {
#ifndef P100E_HAVE_VISION
    (void) argc; (void) argv;
    fprintf(stderr, "p100e-run: built without the vision encoder (cmake/vision.cmake: llama.cpp fork not found)\n");
    return 1;
#else
    using clk = std::chrono::steady_clock;
    auto sec = [](clk::time_point a, clk::time_point b) { return std::chrono::duration<double>(b - a).count(); };
    std::vector<int> devices;
    std::vector<std::string> images;
    std::string prompt_file, chat, chat_file, detail, dlls, vmodel, mmproj, turn2;
    int gen = 64, ctx = 8192, ubatch = 2048, vdev = -2, min_tok = 1024, max_tok = 2048, cap = 0, last_n = 64, top = 0;
    int runs = 1, enc_bench = 0, overlap = 0;
    std::vector<int> mtp_k;      // --mtp K[,K..]: after the plain greedy run, MTP greedy with each K (ids compared)
    float mtp_tau = 0.0f;        // --mtp-adaptive tau
    int depth = 0;               // --depth D: timing at context depth D (synthetic KV rows, tp_model::bench_fill_kv)
    size_t cache_mb = 2048;
    float rpen = 1.0f;
    bool encode_only = false, vverbose = false, fold = true, model_first = false;
    uint32_t kv_type = fast::KV_Q8_0;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
            return argv[++i];
        };
        if (a == "--devices") { std::stringstream ss(next()); std::string t; while (std::getline(ss, t, ',')) devices.push_back(std::stoi(t)); }
        else if (a == "--mmproj") mmproj = next();
        else if (a == "--image") images.push_back(next());
        else if (a == "--prompt-file") prompt_file = next();
        else if (a == "--chat") chat = next();
        else if (a == "--chat-file") chat_file = next();
        else if (a == "--gen") gen = std::stoi(next());
        else if (a == "--ctx") ctx = std::stoi(next());
        else if (a == "--ubatch") ubatch = std::stoi(next());
        else if (a == "--kv") {
            const std::string k = next();
            kv_type = k == "f16" ? fast::KV_F16 : k == "q4_0" ? fast::KV_Q4_0 : fast::KV_Q8_0;
        }
        else if (a == "--vision-dev") vdev = std::stoi(next());
        else if (a == "--vision-dlls") dlls = next();
        else if (a == "--vision-model") vmodel = next();
        else if (a == "--image-min-tokens") min_tok = std::stoi(next());
        else if (a == "--image-max-tokens") max_tok = std::stoi(next());
        else if (a == "--image-cap") cap = std::stoi(next());
        else if (a == "--detail") detail = next();
        else if (a == "--repeat-penalty") rpen = std::stof(next());
        else if (a == "--repeat-last-n") last_n = std::stoi(next());
        else if (a == "--top") top = std::stoi(next());
        else if (a == "--runs") runs = std::stoi(next());
        else if (a == "--vision-cache-mb") cache_mb = (size_t) std::stoll(next());
        else if (a == "--encode-only") encode_only = true;
        else if (a == "--encode-bench") enc_bench = std::stoi(next());
        else if (a == "--overlap-bench") overlap = std::stoi(next());
        else if (a == "--vision-verbose") vverbose = true;
        else if (a == "--no-fold") fold = false;
        else if (a == "--model-first") model_first = true;
        else if (a == "--mtp") { std::stringstream ss(next()); std::string t; while (std::getline(ss, t, ',')) mtp_k.push_back(std::stoi(t)); }
        else if (a == "--mtp-adaptive") mtp_tau = std::stof(next());
        else if (a == "--turn2") turn2 = next();
        else if (a == "--depth") depth = std::stoi(next());
        else { fprintf(stderr, "unknown argument %s\n%s", a.c_str(), run_vision_usage); return 1; }
    }
    if (mmproj.empty() || (images.empty() && chat.empty() && chat_file.empty() && prompt_file.empty())) {
        fprintf(stderr, "%s", run_vision_usage);   // no --image: a text-only prompt through the same path (control runs)
        return 1;
    }
    // P100E_MEM_POLL_MS=n: a thread samples cudaMemGetInfo of every --devices GPU each n ms for the whole run and prints
    // the TRACKED minimum free memory per GPU at the end (transient peaks, e.g. scratch that is freed again, included);
    // MEMPOLL mark lines give the phases on the same clock
    struct mem_poller {
        std::vector<int> devs;
        std::vector<size_t> min_free;
        std::vector<double> t_min;
        std::atomic<bool> stop{false};
        std::thread th;
        clk::time_point t0 = clk::now();
        long samples = 0;
        double now() const { return std::chrono::duration<double>(clk::now() - t0).count(); }
        void start(const std::vector<int> & d, int ms) {
            devs = d;
            min_free.assign(d.size(), SIZE_MAX);
            t_min.assign(d.size(), 0.0);
            th = std::thread([this, ms] {
                while (!stop.load()) {
                    for (size_t i = 0; i < devs.size(); ++i) {
                        size_t fr = 0, tot = 0;
                        if (cudaSetDevice(devs[i]) == cudaSuccess && cudaMemGetInfo(&fr, &tot) == cudaSuccess) {
                            if (fr < min_free[i]) { min_free[i] = fr; t_min[i] = now(); }
                        } else cudaGetLastError();
                    }
                    ++samples;
                    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
                }
            });
        }
        void mark(const char * what) {
            if (!th.joinable()) return;
            printf("MEMPOLL mark t=%.2f s: %s\n", now(), what);
            fflush(stdout);
        }
        ~mem_poller() {
            if (!th.joinable()) return;
            stop = true;
            th.join();
            for (size_t i = 0; i < devs.size(); ++i)
                printf("MEMPOLL GPU %d: tracked minimum free %.0f MiB at t=%.2f s (%ld samples over %.1f s)\n", devs[i],
                       min_free[i] / 1048576.0, t_min[i], samples, now());
            fflush(stdout);
        }
    } mpoll;
    if (const char * e = getenv("P100E_MEM_POLL_MS")) if (atoi(e) > 0 && !devices.empty()) mpoll.start(devices, atoi(e));
    try {
        std::unique_ptr<tp_model> mptr;
        std::unique_ptr<tp_mtp> drafter;
        double t_model = 0;
        auto load_model = [&] {
            if (devices.empty()) throw std::runtime_error("image mode needs --devices");
            const auto a = clk::now();
            mptr = std::make_unique<tp_model>(argv[1], devices, ctx, ubatch, !mtp_k.empty(), kv_type);
            if (!mtp_k.empty()) drafter = std::make_unique<tp_mtp>(*mptr);
            t_model = sec(a, clk::now());
        };
        if (model_first) {
            // the server's order: the model first (peer access enabled, KV caches allocated), then the encoder
            load_model();
            printf("model: load %.2f s, before the encoder (server load order)\n", t_model);
            mptr->print_mem("after model load, before the encoder");
            mpoll.mark("model loaded (before the encoder)");
            fflush(stdout);
        }
        const auto t0 = clk::now();
        vision_opts vo;
        vo.mmproj = mmproj;
        vo.text_model = vmodel.empty() ? argv[1] : vmodel;
        vo.device = vdev != -2 ? vdev : (devices.empty() ? 0 : devices.back());
        vo.min_tokens = min_tok;
        vo.max_tokens = max_tok;
        vo.cache_mb = cache_mb;
        vo.dll_dir = dlls;
        vo.verbose = vverbose;
        vo.fold_gemm = fold;
        size_t vfree0 = 0, vfree1 = 0, vtot = 0;
        if (vo.device >= 0) { cudaSetDevice(vo.device); cudaMemGetInfo(&vfree0, &vtot); }
        vision_encoder enc(vo);
        if (vo.device >= 0) { cudaSetDevice(vo.device); cudaMemGetInfo(&vfree1, &vtot); }
        const auto t1 = clk::now();
        printf("vision: encoder on %s, DLLs %s, load + warmup %.2f s, GPU memory used by the encoder %.0f MiB, n_embd %d, "
               "tokens %d..%d, CPU fallback: %s, GGML_CUDA_GEMM_FOLD=%s\n",
               enc.native() ? "p100e-native (nvit, P100E_NVIT_DEVICES)" : vo.device >= 0 ? ("CUDA" + std::to_string(vo.device)).c_str() : "CPU",
               enc.dll_dir().c_str(), sec(t0, t1), (vfree0 - vfree1) / 1048576.0, enc.n_embd(), enc.min_tokens(),
               enc.max_tokens(), enc.cpu_fallback() ? "YES" : "no",
               getenv("GGML_CUDA_GEMM_FOLD") ? getenv("GGML_CUDA_GEMM_FOLD") : "(unset)");
        for (const auto & w : enc.warnings()) printf("vision warning: %s\n", w.c_str());
        mpoll.mark("encoder loaded");
        fflush(stdout);

        std::vector<image_input> ins;
        for (const auto & f : images) {
            image_input in;
            in.bytes = read_file(f);
            in.max_tokens = cap > 0 ? cap : (!detail.empty() ? enc.detail_max_tokens(detail) : 0);
            ins.push_back(std::move(in));
        }
        std::string prompt;
        if (!prompt_file.empty()) prompt = read_file(prompt_file);
        else {
            if (!chat_file.empty()) chat = read_file(chat_file);
            std::string marks;
            for (size_t i = 0; i < images.size(); ++i) marks += vision_encoder::marker();
            prompt = "<|im_start|>user\n" + marks + chat + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
        }

        auto print_image = [&](size_t k, const mm_piece & p) {
            const auto & e = *p.image;
            printf("IMAGE %zu: %dx%d px -> %dx%d = %d tokens, key %016llx%016llx, %s: hash %.1f ms, decode %.1f ms, "
                   "resize %.1f ms, preprocess %.1f ms, encode %.1f ms, total %.1f ms\n",
                   k, e.src_w, e.src_h, e.nx, e.ny, e.n_tok(), (unsigned long long) e.key[0], (unsigned long long) e.key[1],
                   p.t.cached ? "CACHED" : "encoded", p.t.hash_ms, p.t.decode_ms, p.t.resize_ms, p.t.preprocess_ms,
                   p.t.encode_ms, p.t.total_ms());
        };

        if (encode_only) {
            for (int run = 0; run < std::max(1, runs); ++run) {
                const auto a = clk::now();
                auto pieces = enc.build(prompt, ins);
                printf("RUN %d: build %.3f s\n", run, sec(a, clk::now()));
                size_t k = 0;
                int n_text = 0, n_img = 0;
                for (const auto & p : pieces) {
                    if (p.image) { n_img += p.image->n_tok(); continue; }
                    n_text += (int) p.tokens.size();
                    printf("  text:");
                    for (int32_t id : p.tokens) printf(" %d", id);
                    printf("\n");
                }
                printf("  prompt tokens: %d text + %d image = %d\n", n_text, n_img, n_text + n_img);
                for (const auto & p : pieces) {
                    if (!p.image || p.image->data.empty()) continue;
                    print_image(k++, p);
                    double s = 0, s2 = 0;
                    for (float v : p.image->data) { s += v; s2 += (double) v * v; }
                    printf("  embd sum %.6f rms %.6f first %.6f %.6f %.6f\n", s, std::sqrt(s2 / p.image->data.size()),
                           p.image->data[0], p.image->data[1], p.image->data[2]);
                }
            }
            for (int b = 0; b < enc_bench; ++b) {
                enc.cache_clear();
                for (size_t k = 0; k < ins.size(); ++k) {
                    vision_timing t;
                    auto e = enc.encode(ins[k], &t);
                    printf("ENCODE-BENCH %d image %zu: %d tokens preprocess %.1f ms encode %.1f ms\n", b, k, e->n_tok(),
                           t.preprocess_ms, t.encode_ms);
                }
            }
            return 0;
        }

        if (!mtp_k.empty() && rpen != 1.0f) throw std::runtime_error("--mtp is greedy only (no --repeat-penalty)");
        if (!mptr) load_model();
        tp_model & m = *mptr;
        printf("model: load %.2f s, ubatch %d, ctx %d%s\n", t_model, m.max_batch(), ctx,
               model_first ? " (loaded before the encoder)" : "");
        m.print_mem("after load (encoder resident)");
        mpoll.mark("model and encoder loaded");
        // native encoder: activations in the LLM's prefill ring (idle between forwards; P100E_NVIT_BORROW=0: own scratch)
        std::atomic<int> n_lent{0}, n_refused{0};   // scratch-source answers (printed at the end)
        if (enc.native() && !(getenv("P100E_NVIT_BORROW") && atoi(getenv("P100E_NVIT_BORROW")) == 0)) {
            if (m.can_lend()) {
                // (called on the encoder's thread too, --overlap-bench: tp_model.h idle_scratch threading contract)
                enc.set_scratch_source([&m, &n_lent, &n_refused]() {
                    std::vector<vision_encoder::scratch_region> v;
                    for (const auto & s : m.idle_scratch()) v.push_back({s.dev, s.p, s.bytes, s.lease});
                    ++(v.empty() ? n_refused : n_lent);
                    return v;
                });
                m.idle_scratch();   // the ring the first prefill would allocate anyway (lease dropped at once)
                m.print_mem("with the prefill ring (also the native encoder's scratch)");
            } else {
                printf("native encoder: the LLM's prefill pipeline is unavailable here (nothing to lend): own scratch\n");
            }
        }
        const int V = m.vocab();
        for (int run = 0; run < std::max(1, runs); ++run) {
            m.reset();
            if (depth > 0) m.bench_fill_kv(depth);   // timing only: the output is meaningless after synthetic rows
            const auto a = clk::now();
            auto pieces = enc.build(prompt, ins);
            const auto b = clk::now();
            mpoll.mark(("run " + std::to_string(run) + ": images encoded").c_str());
            std::vector<int32_t> hist;   // penalty history: prompt text tokens (image rows skipped), then generated
            std::vector<float> logits;
            double t_text = 0, t_img = 0;
            int n_text = 0, n_img = 0;
            size_t k = 0;
            if (pieces.empty() || pieces.back().image) throw std::runtime_error("the prompt must end with text");
            // default: one forward_mm over all pieces (text shares the image's prefill batches); P100E_MM_PACK=0: one
            // forward per piece (forward_n / forward_image), timed separately
            static const bool pack = !getenv("P100E_MM_PACK") || atoi(getenv("P100E_MM_PACK")) != 0;
            std::vector<tp_model::mm_in> in;
            for (const auto & p : pieces) {
                tp_model::mm_in q;
                if (p.image) {
                    q.embd = p.image->data.data(); q.nx = p.image->nx; q.ny = p.image->ny; q.n_embd = p.image->n_embd;
                    n_img += p.image->n_tok();
                    print_image(k++, p);
                } else {
                    q.tokens = p.tokens.data(); q.n_tok = (int) p.tokens.size();
                    n_text += (int) p.tokens.size();
                    hist.insert(hist.end(), p.tokens.begin(), p.tokens.end());
                }
                in.push_back(q);
            }
            const auto bb = clk::now();
            if (pack) {
                logits = m.forward_mm(in);
            } else {
                for (const auto & p : pieces) {
                    const auto c = clk::now();
                    if (p.image) {
                        m.forward_image(p.image->data.data(), p.image->nx, p.image->ny, p.image->n_embd);
                        t_img += sec(c, clk::now());
                    }
                    else { logits = m.forward_n(p.tokens, 1); t_text += sec(c, clk::now()); }
                }
            }
            const auto c = clk::now();
            const double tpf = sec(bb, c);
            mpoll.mark(("run " + std::to_string(run) + ": prompt prefilled").c_str());
            printf("RUN %d: prompt %d text + %d image tokens (n_past %d, rope delta %d): images %.3f s (encode / cache), "
                   "prefill %.3f s (%.1f t/s%s), time to first token %.3f s\n",
                   run, n_text, n_img, m.n_past(), m.rope_delta(), sec(a, b), tpf, (n_text + n_img) / tpf,
                   pack ? ", packed" : (", text " + std::to_string(t_text) + " s, image rows " + std::to_string(t_img) + " s").c_str(),
                   sec(a, c));
            std::vector<int32_t> out;
            std::string text;
            const auto d0 = clk::now();
            for (int i = 0; i < gen; ++i) {
                if (top > 0) {
                    const double lse = log_sum_exp(logits.data(), V);
                    std::vector<int> idx(V);
                    std::iota(idx.begin(), idx.end(), 0);
                    std::partial_sort(idx.begin(), idx.begin() + top, idx.end(), [&](int x, int y) { return logits[x] > logits[y]; });
                    printf("TOP %d:", i);
                    for (int j = 0; j < top; ++j) printf(" %d:%.4f", idx[j], logits[idx[j]] - lse);
                    printf("\n");
                }
                apply_repeat_penalty(logits, hist, last_n, rpen);
                int next = 0;
                for (int j = 1; j < V; ++j) if (logits[j] > logits[next]) next = j;
                out.push_back(next);
                hist.push_back(next);
                if (enc.is_eog(next)) break;
                text += enc.piece(next);
                if (i + 1 < gen) logits = m.forward({next});
            }
            const double td = sec(d0, clk::now());
            printf("RUN %d: generated %zu tokens in %.3f s (%.2f t/s)\n", run, out.size(), td,
                   out.size() > 1 ? (out.size() - 1) / td : 0.0);
            printf("generated ids:");
            for (int id : out) printf(" %d", id);
            printf("\n=== OUTPUT BEGIN ===\n%s\n=== OUTPUT END ===\n", text.c_str());
            fflush(stdout);
            // --turn2: a text turn on the same context (image turn, then text turns: the Claude Code pattern). The
            // context holds the prompt and every generated token but the last, so the turn-2 prompt starts with it.
            std::vector<int32_t> t2_prompt, out2;
            if (!turn2.empty() && !out.empty()) {
                t2_prompt.push_back(out.back());
                const auto tt = enc.tokenize(std::string(enc.is_eog(out.back()) ? "" : "<|im_end|>") + "\n<|im_start|>user\n" +
                                             turn2 + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
                t2_prompt.insert(t2_prompt.end(), tt.begin(), tt.end());
                const int np0 = m.n_past();
                const auto q0 = clk::now();
                logits = m.forward_n(t2_prompt, 1);
                const auto q1 = clk::now();
                std::string text2;
                for (int i = 0; i < gen; ++i) {
                    int next = 0;
                    for (int j = 1; j < V; ++j) if (logits[j] > logits[next]) next = j;
                    out2.push_back(next);
                    if (enc.is_eog(next)) break;
                    text2 += enc.piece(next);
                    if (i + 1 < gen) logits = m.forward({next});
                }
                printf("RUN %d TURN 2: %zu prompt tokens at n_past %d (rope delta %d), prefill %.3f s, generated %zu tokens "
                       "in %.3f s\ngenerated ids turn 2:", run, t2_prompt.size(), np0, m.rope_delta(), sec(q0, q1),
                       out2.size(), sec(q1, clk::now()));
                for (int id : out2) printf(" %d", id);
                printf("\n=== OUTPUT 2 BEGIN ===\n%s\n=== OUTPUT 2 END ===\n", text2.c_str());
                fflush(stdout);
            }
            // MTP greedy over the same multimodal prompt: ids must equal the plain greedy ones above
            for (int K : mtp_k) {
                if (out.size() < 2) break;
                m.reset();
                std::vector<mtp_piece> mp;
                for (const auto & p : pieces) {
                    mtp_piece q;
                    if (p.image) { q.embd = p.image->data.data(); q.nx = p.image->nx; q.ny = p.image->ny; q.n_embd = p.image->n_embd; }
                    else q.tokens = p.tokens;
                    mp.push_back(std::move(q));
                }
                mtp_opts o;
                o.k = K; o.tau = mtp_tau;
                mtp_stats st;
                const auto ids = generate_mtp(m, *drafter, mp, (int) out.size(), o, st);
                const bool same = ids == out;
                printf("RUN %d MTP k%s%d%s: %.2f t/s after the prompt (%zu tokens, %.3f s; prompt %.3f s, catch-up %.1f ms), "
                       "steps %d, accepted %d/%d = %.1f%%, %s\n", run, mtp_tau > 0 ? "<=" : "=", K,
                       mtp_tau > 0 ? (" tau " + std::to_string(mtp_tau)).c_str() : "", (out.size() - 1) / st.t_gen,
                       out.size(), st.t_gen, st.t_prompt, 1e3 * st.t_catchup, st.steps, st.accepted, st.drafted,
                       st.drafted ? 100.0 * st.accepted / st.drafted : 0.0, same ? "IDENTICAL" : "MISMATCH");
                if (!same) {
                    printf("  mtp ids:");
                    for (int id : ids) printf(" %d", id);
                    printf("\n");
                }
                fflush(stdout);
                if (!same || out2.size() < 2) continue;
                // turn 2 through generate_mtp on the context the MTP turn 1 left (the draft carry continues)
                std::vector<mtp_piece> mp2(1);
                mp2[0].tokens = t2_prompt;
                mtp_stats st2;
                const auto ids2 = generate_mtp(m, *drafter, mp2, (int) out2.size(), o, st2);
                const bool same2 = ids2 == out2;
                printf("RUN %d MTP k%s%d TURN 2: %.2f t/s after the prompt (%zu tokens), steps %d, accepted %d/%d = %.1f%%, "
                       "%s\n", run, mtp_tau > 0 ? "<=" : "=", K, (out2.size() - 1) / st2.t_gen, out2.size(), st2.steps,
                       st2.accepted, st2.drafted, st2.drafted ? 100.0 * st2.accepted / st2.drafted : 0.0,
                       same2 ? "IDENTICAL" : "MISMATCH");
                if (!same2) {
                    printf("  mtp ids turn 2:");
                    for (int id : ids2) printf(" %d", id);
                    printf("\n");
                }
                fflush(stdout);
            }
            mpoll.mark(("run " + std::to_string(run) + ": done (generation, turn 2, MTP)").c_str());
            m.print_mem(("after run " + std::to_string(run) + " (encoder compute buffers as grown by this run's images)").c_str());
            fflush(stdout);
        }
        const auto st = enc.stats();
        printf("vision cache: %zu entries, %.1f MiB, hits %zu, misses %zu, evictions %zu\n", st.entries, st.bytes / 1048576.0,
               st.hits, st.misses, st.evictions);

        if (overlap > 0) {
            // decode while the encoder runs on another thread (other sessions must keep decoding during an encode)
            struct step { double t0, t1; };   // seconds from the phase start
            auto decode_n = [&](int n, std::vector<step> & st, clk::time_point base) {
                for (int i = 0; i < n; ++i) {
                    const auto q = clk::now();
                    m.forward({(int32_t) (1000 + i)});
                    st.push_back({sec(base, q), sec(base, clk::now())});
                }
            };
            auto stats = [](const std::vector<double> & ms, double & mean, double & med, double & mx) {
                std::vector<double> s = ms;
                std::sort(s.begin(), s.end());
                mean = s.empty() ? 0 : std::accumulate(s.begin(), s.end(), 0.0) / s.size();
                med = s.empty() ? 0 : s[s.size() / 2];
                mx = s.empty() ? 0 : s.back();
            };
            for (int with = 0; with < 2; ++with) {
                m.reset();
                m.forward_n(enc.tokenize("Hello, this is a decode stall check."), 1);
                enc.cache_clear();
                std::vector<step> st;
                double enc_ms = 0, e0 = -1, e1 = -1;
                std::thread th;
                const auto base = clk::now();
                if (with) th = std::thread([&] {
                    e0 = sec(base, clk::now());
                    vision_timing t;
                    enc.encode(ins[0], &t);
                    e1 = sec(base, clk::now());
                    enc_ms = t.encode_ms + t.preprocess_ms;
                });
                decode_n(overlap, st, base);
                if (th.joinable()) th.join();
                std::vector<double> all, during;
                for (const auto & x : st) {
                    all.push_back(1e3 * (x.t1 - x.t0));
                    if (with && x.t1 > e0 && x.t0 < e1) during.push_back(1e3 * (x.t1 - x.t0));
                }
                double mean, med, mx;
                stats(all, mean, med, mx);
                printf("OVERLAP %s encode: %d decode steps mean %.2f ms, median %.2f ms, max %.2f ms%s\n",
                       with ? "with" : "without", overlap, mean, med, mx,
                       with ? (" (encode " + std::to_string((int) enc_ms) + " ms)").c_str() : "");
                if (with) {
                    stats(during, mean, med, mx);
                    printf("OVERLAP during the encode (%.0f ms window, encoder thread): %zu decode steps mean %.2f ms, "
                           "median %.2f ms, max %.2f ms\n", 1e3 * (e1 - e0), during.size(), mean, med, mx);
                }
            }
            printf("OVERLAP scratch source so far: %d lent, %d refused\n", n_lent.load(), n_refused.load());
            fflush(stdout);

            // a prefill against an encode on the real lend gate (tp_model.h idle_scratch contract), both orders: a
            // prefill that starts while the encoder holds the ring waits for it; an encode that starts while a prefill
            // runs gets nothing lent and uses its own scratch for that encode only (freed after it). The prefill's
            // logits must equal the reference (no encoder) bit for bit, and a later encode must borrow again.
            if (enc.native() && m.can_lend()) {
                const int PT = std::min(m.max_batch(), 1536);
                std::vector<int32_t> pt(PT);
                for (int i = 0; i < PT; ++i) pt[i] = (int32_t) (1000 + (i * 7919) % 60000);
                auto hash = [](const std::vector<float> & l) {
                    uint64_t h = 1469598103934665603ull;
                    const unsigned char * bp = reinterpret_cast<const unsigned char *>(l.data());
                    for (size_t i = 0; i < l.size() * sizeof(float); ++i) { h ^= bp[i]; h *= 1099511628211ull; }
                    return h;
                };
                m.reset();
                auto q0 = clk::now();
                const uint64_t ref = hash(m.forward_n(pt, 1));
                const double t_ref = sec(q0, clk::now());
                printf("COLLIDE reference: %d-token prefill %.3f s, logits hash %016llx\n", PT, t_ref, (unsigned long long) ref);
                m.print_mem("COLLIDE before (no own encoder scratch expected)");
                // order 0: encode first, the prefill waits; 1: prefill first, own scratch; 2: prefill first and the own
                // scratch "does not fit" (P100E_NVIT_TEST_OOM=1): the encode waits for the lender, then borrows
                for (int order = 0; order < 3; ++order) {
                    m.reset();
                    enc.cache_clear();
                    const int lent0 = n_lent.load(), ref0 = n_refused.load();
                    double enc_s = 0;
                    std::string err;
                    std::thread th([&] {
                        if (order >= 1) std::this_thread::sleep_for(std::chrono::milliseconds(300));   // inside the prefill
                        const auto e = clk::now();
                        try { vision_timing t; enc.encode(ins[0], &t); } catch (const std::exception & ex) { err = ex.what(); }
                        enc_s = sec(e, clk::now());
                    });
                    if (order == 2) _putenv_s("P100E_NVIT_TEST_OOM", "1");
                    if (order == 0) {   // the encoder has its lease before the prefill starts
                        const auto w = clk::now();
                        while (n_lent.load() == lent0 && n_refused.load() == ref0 && sec(w, clk::now()) < 10)
                            std::this_thread::sleep_for(std::chrono::microseconds(200));
                    }
                    q0 = clk::now();
                    const uint64_t h = hash(m.forward_n(pt, 1));
                    const double t_pf = sec(q0, clk::now());
                    th.join();
                    if (order == 2) _putenv_s("P100E_NVIT_TEST_OOM", "");
                    printf("COLLIDE %s: prefill %.3f s (reference %.3f s), encode %.3f s, lent %d refused %d, logits %s%s%s\n",
                           order == 0 ? "encode first (the prefill waits for the lease)"
                           : order == 1 ? "prefill first (the encode gets no lease: own scratch)"
                                        : "prefill first, own scratch does not fit (the encode waits, then borrows)",
                           t_pf, t_ref, enc_s, n_lent.load() - lent0, n_refused.load() - ref0,
                           h == ref ? "IDENTICAL" : "DIFFERENT", err.empty() ? "" : ", encode error: ", err.c_str());
                    m.print_mem(order == 0 ? "COLLIDE after encode-first" : "COLLIDE after prefill-first (own scratch freed?)");
                    fflush(stdout);
                }
                enc.cache_clear();
                const int lent0 = n_lent.load();
                {
                    vision_timing t;
                    enc.encode(ins[0], &t);
                }
                printf("COLLIDE afterwards: a lone encode borrowed again: %s\n", n_lent.load() > lent0 ? "yes" : "NO");
                m.print_mem("COLLIDE end");
                fflush(stdout);
            }
        }
    } catch (const std::exception & e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
#endif
}

// p100e-run --seq <args...> ::: <args...> ::: ...: run this executable once per group, in order (one command for several
// runs). A group may start with VAR=value words (environment of that child only). Exit code: the first non-zero.
int run_seq(int argc, char ** argv) {
#ifdef _WIN32
    char self[MAX_PATH * 2];
    GetModuleFileNameA(nullptr, self, sizeof self);
    std::vector<std::vector<std::string>> groups(1);
    for (int i = 2; i < argc; ++i) {
        if (!strcmp(argv[i], ":::")) { groups.emplace_back(); continue; }
        groups.back().push_back(argv[i]);
    }
    int rc_all = 0;
    for (size_t g = 0; g < groups.size(); ++g) {
        std::vector<std::pair<std::string, std::string>> env;
        // a group starting with "!" runs the rest through cmd.exe /c instead of this executable (helper scripts, e.g.
        // a llama-server reference in the same command)
        const bool shell = !groups[g].empty() && groups[g][0] == "!";
        if (shell) groups[g].erase(groups[g].begin());
        std::string cmd = shell ? std::string("cmd.exe /c") : std::string("\"") + self + "\"";
        bool args = false;
        for (const auto & a : groups[g]) {
            const size_t eq = a.find('=');
            if (!args && eq != std::string::npos && a[0] != '-' && a.find_first_of("\\/:") > eq) {
                env.emplace_back(a.substr(0, eq), a.substr(eq + 1));
                continue;
            }
            args = true;
            if (a.find_first_of(" \t\"") != std::string::npos) {
                std::string q = "\"";
                for (char ch : a) { if (ch == '"') q += '\\'; q += ch; }
                cmd += " " + q + "\"";
            } else cmd += " " + a;
        }
        printf("=== seq %zu/%zu:%s %s\n", g + 1, groups.size(), [&] {
            std::string e;
            for (auto & kv : env) e += " " + kv.first + "=" + kv.second;
            return e;
        }().c_str(), cmd.c_str());
        fflush(stdout);
        fflush(stderr);
        std::vector<std::pair<std::string, std::string>> saved;
        for (auto & kv : env) {
            const char * old = getenv(kv.first.c_str());
            saved.emplace_back(kv.first, old ? old : "");
            SetEnvironmentVariableA(kv.first.c_str(), kv.second.c_str());
        }
        STARTUPINFOA si{};
        si.cb = sizeof si;
        PROCESS_INFORMATION pi{};
        std::vector<char> cl(cmd.begin(), cmd.end());
        cl.push_back('\0');
        DWORD rc = 1;
        if (CreateProcessA(nullptr, cl.data(), nullptr, nullptr, TRUE, 0, nullptr, nullptr, &si, &pi)) {
            WaitForSingleObject(pi.hProcess, INFINITE);
            GetExitCodeProcess(pi.hProcess, &rc);
            CloseHandle(pi.hProcess);
            CloseHandle(pi.hThread);
        } else {
            fprintf(stderr, "seq: CreateProcess failed (%lu)\n", GetLastError());
        }
        for (auto & kv : saved) SetEnvironmentVariableA(kv.first.c_str(), kv.second.empty() ? nullptr : kv.second.c_str());
        printf("=== seq %zu exit %lu\n", g + 1, rc);
        fflush(stdout);
        if (rc && !rc_all) rc_all = (int) rc;
    }
    return rc_all;
#else
    (void) argc; (void) argv;
    fprintf(stderr, "--seq: Windows only\n");
    return 1;
#endif
}
