// p100e-run <model.gguf> --tokens 1,2,3 [--gen N] [--ctx 4096] [--debug] [--logits out.f32] [--devices 0,1 (tensor parallel, resident weights)]
//   [--split a,b,..]: tensor-parallel weight per --devices GPU (e.g. 2,1,1; sets P100E_SPLIT for every tp_model of the
//   run; default equal, as even as the model allows: 3 GPUs get an uneven split)
//   [--mtp K[,K2..]]: MTP speculative greedy decoding (needs --devices). --tokens may hold several prompts
//   separated by ';'. For each prompt: plain greedy --gen N (baseline), then MTP with each K; prints acceptance,
//   t/s and whether the ids are identical. With --bench pp tg: tg tokens after the first prompt (or pp synthetic).
//   [--draft-vocab N[,N2..]]: draft head scores the N most frequent tokens (0 = full head; one pass per value)
//   [--mtp-adaptive tau[,tau2..]] [--mtp-kmin m]: draft up to K, verify draft j only while the product of the draft
//   head's top-1 probabilities >= tau (0 = fixed K). Each config prints per-prompt lines and a SUMMARY line.
//   Without --mtp-adaptive every K is fixed (the gates); the library default mode (mtp_opts{}, src/tp_mtp.h) is
//   `--mtp 3 --mtp-adaptive 0.5`. P100E_MTP_CHECK=1: bitwise logit check of every used verify row (counts per run).
//   MTP prompts longer than --ubatch run as ubatch chunks, each followed by the draft head's KV-only catch-up
//   (P100E_MTP_KV_ONLY=0: the old full catch-up, prompt <= max batch - 1); P100E_MTP_PROF=1 prints prompt / catch-up times.
//   [--kv f16|q8_0|q4_0]: attention KV cache type (tensor parallel path; default q8_0)
//   [--ubatch U]: max tokens per forward (prompts / --bench pp longer than U run as U-token chunks; default: whole prompt)
//   [--bench-depth D1,D2,..]: with --bench pp tg (and --devices), for each depth in increasing order: fill the context up
//   to D (not timed), then time a pp-token prefill and tg decode steps (like llama-bench -d). The fill is real forwards
//   of synthetic tokens up to --bench-real-max N (default: all), synthetic KV rows (tp_model::bench_fill_kv) above.
//   [--load-mtp]: load the MTP block and the draft buffers (memory as with MTP) without MTP decoding
//   [--tokens-file f [--tokens-max N]]: the prompt ids from a file (comma / space separated), optionally truncated
//   [--force a,b,c]: teacher forcing - feed these ids as the generated tokens (instead of the argmax), --gen = their count
//   P100E_TOPK=N: print the top N logits (id:value) of the prompt and every generated step
//   [--ab "A=1,B=2/A=0/..." [--ab-reps N]] (with --devices): in-process A/B of env switches that the engine reads per
//   forward (the prefill switches P100E_PF_*, P100E_UNPACK). Arms separated by '/', VAR=value pairs by ',' (an empty arm
//   = defaults; every variable named in any arm is unset before an arm's own are set).
//     with --bench pp tg [--bench-depth D..]: per depth, N reps of every arm interleaved (order reversed on odd reps);
//       depth 0 runs reset() first, so the pp logits must be bit-identical across reps of an arm; prints per arm
//       mean / min / max t/s and whether its logits hash equals arm 0's ('logits': the pp prefill's; 'tg logits': the
//       last tg decode step's, hashed after the timed loop - they carry every decode step's KV rows / recurrent state,
//       so a decode-only switch shows there). Depths > 0 rewind n_past (timing only, no hashes).
//     with a prompt (--tokens / --tokens-file) and --gen G: per arm reset + prompt + G greedy steps teacher-forced on
//       arm 0's ids; per arm vs arm 0: logits hash equality, KLD(arm0 || arm), top-1 agreement, |d| over arm 0's top-10.
//     with --mtp K[,K2..] [--mtp-adaptive tau..] and a prompt + --gen G (or --bench pp tg: a synthetic pp-token prompt,
//       tg tokens): per rep every arm (interleaved as above): plain greedy, then MTP per config; per arm and config
//       mean / min / max t/s, acceptance, runs identical to the arm's plain greedy, ids vs arm 0, and the draft graphs
//       captured / dropped for a kernel switch (tp_model::reload_env -> tp_mtp re-captures its draft graphs).
//     Every switch of a kernel baked into the CUDA graphs (P100E_MVWRP_*, P100E_MERGE_INPROJ, P100E_GDN_FUSE) drops and
//     re-captures the graphs, so the first step after a switch runs eager + capture inside the timed region (all such arms alike).
//   --bench warm-up: before the first timed measurement of every --bench mode, untimed passes (prefill + up to 4 decode
//     steps, then reset(); --ab: one per arm first; --mtp: also one short MTP generation) until P100E_BENCH_WARMUP
//     seconds (default 3, 0 = off; --ab keeps its one pass per arm) - lazy allocations, graph captures and GPU clocks
//     are out of the timed region.
//   [--prompts-file f]: several prompts, one per line (ids comma / space separated, brackets ignored)
//   [--wire-scan] (with --devices): every prompt (--tokens a,b;c,d or --prompts-file) from an empty context with
//     P100E_PF_WIRE_STATS=1: a PFWIRE line per prefill, a PFSCAN line per prompt (largest |row-parallel partial| on the
//     prefill exchange, its margin to the f16 limit 65504, saturation) and the corpus maximum. P100E_PF_WIRE16=0 gives
//     the exact f32 partials (the values the f16 wire would round); with the f16 wire a saturating prompt is reported
//     (tp_model::last_prefill_saturated) and the instance continues on the f32 wire. Prompts of <= 8 tokens use no
//     prefill pipeline and are skipped.
//   [--kld-base f [--kld-chunks N] [--kld-dump out.bin] [--kld-threads N]]: KL divergence against a llama-perplexity
//   base file (--kl-divergence-base), reproducing llama-perplexity --kl-divergence exactly (src/kld.h): every chunk
//   from an empty context, rows n_ctx/2 .. n_ctx-2 scored, same statistics and log lines. Needs --devices; prompt
//   chunks of --ubatch (default n_ctx). --kv may list several types (f16,q8_0,q4_0): one arm (tp_model) per type.
//   --kld-dump: per-token float32 (KLD, p_diff) pairs like LLAMA_PPL_KLD_DUMP; with
//   several --kv types one file per arm, the type inserted before the extension (out.q8_0.bin).
//   P100E_KLD_HEAD=gemm: LM head of the scored rows through the prefill GEMM (default: decode GEMV, 8 rows per call)
//   [--net-nodes host:port,host:port[,..] --net-node i [--net-logits]]: tensor parallel with ranks on
//   several machines. Every node runs the same command (same model file contents, prompt, --split, --ctx, --gen, ...;
//   checked at connect, a mismatch is refused) with --devices = ITS OWN GPUs and --net-node = its index in the list;
//   global ranks in node order (node 0's devices first). --split a,b,c = weights of all global ranks. Node i listens on
//   its own port for the later nodes and connects to the earlier ones. Decode steps exchange only the greedy id's
//   argmax pairs unless --net-logits (every step's full logits on every node: P100E_PRINT_LOGITS hashes). Prints NET
//   prefill / decode lines (tok/s, the share of wall time in network collectives, bytes). Plain prompt + --gen and
//   --bench pp tg only (no --mtp, sessions, --ab, KLD, vision). Env: P100E_NET_NODES / P100E_NET_NODE,
//   P100E_NET_CONNECT_S (wait for the peers, default 600), P100E_NET_TIMEOUT_S (a stalled collective, default 60).
//   [--pipe-layers k|auto | --pipe-split a,b] with --net-nodes of 2 nodes: hybrid pipeline instead of
//   tensor parallel over the nodes. Node 0 runs layers 0..k-1 with its own tensor split over ITS --devices (--split
//   weighs its GPUs only, node local), node 1 runs layers k..L-1 + the LM head over its --devices. Per forward node 0
//   sends its T residual rows (T x E f32) and node 1 returns the greedy ids (or the logits: --net-logits / the
//   prompt). --pipe-split a,b = --pipe-layers a, checked a + b = L. auto: the most layers node 1 holds. The byte plan of
//   both nodes is printed at load (a node that does not fit is refused). Same run limits as --net-nodes.
// Image mode: --mmproj f --image f [...] (--chat text | --prompt-file f) ...: tools/p100e_run_vision.cpp (usage there)
// p100e-run --seq <args> ::: [VAR=v ...] <args> ::: ...: run this executable once per group (several runs in one command)
// Runs the prompt tokens as one batch, prints the top-10 next tokens, then greedy-decodes N tokens; a prompt that
// saturates the f16 prefill wire is redone on the f32 wire (as a server must; tp_model::last_prefill_saturated).
//   [--sessions n1[,n2..] [--sessions-sync] [--sessions-pool N|auto] [--reserve-mb a,b] [--max-active N] [--draft-rows N] [--decode-per-chunk N]]
//   [--offload-test n1] [--tokens-files f1,f2,..]: multi-session engine tests (tools/sessions_mode.inc)
//   [--cache-test --tokens-file doc [--cache-a N --cache-b N --cache-turns K --cache-turn0 N --cache-step N --cache-gen N]
//    [--cache-no-disk] [--cache-no-variant] [--ckpt-every N] [--min-split N] [--cache-inexact]]: prompt cache tests
//   [--sched-test [--sched-pool N]]: scheduler regressions (tools/cache_mode.inc; own engine, pool N tokens < --ctx)
//   [--fair-test N [--fair-shares 0,0.5]]: decode t/s of a decoding session while an N-token prompt prefills, per
//    sched_opts::decode_share (tools/cache_mode.inc); [--decode-share x] for --sessions
//   cache test extras: [--cache-dir dir] (session files) [--cache-dturns K --cache-dgen N] (decoded-token reuse replay)
// Runs the prompt tokens as one batch, prints the top-10 next tokens, then greedy-decodes N tokens.
#include "../src/model.h"
#include "../src/tp_model.h"
#include "../src/tp_mtp.h"
#include "../src/kld.h"

#include <functional>
#include <future>
#include <memory>
#include <thread>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <climits>
#include <cstdint>
#include <cmath>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace p100e;

// ---- --ab: in-process A/B of env switches ----
struct ab_arm { std::string label; std::vector<std::pair<std::string, std::string>> vars; };

static std::vector<ab_arm> parse_arms(const std::string & s) {
    std::vector<ab_arm> arms;
    std::stringstream as(s);
    std::string a;
    while (std::getline(as, a, '/')) {
        ab_arm arm;
        arm.label = a.empty() ? "(defaults)" : a;
        std::stringstream vs(a);
        std::string kv;
        while (std::getline(vs, kv, ',')) {
            const size_t e = kv.find('=');
            if (kv.empty()) continue;
            if (e == std::string::npos) throw std::runtime_error("--ab: expected VAR=value, got " + kv);
            arm.vars.push_back({kv.substr(0, e), kv.substr(e + 1)});
        }
        arms.push_back(arm);
    }
    if (arms.empty()) throw std::runtime_error("--ab: no arms");
    return arms;
}

static void set_env(const std::string & k, const std::string * v) {
#ifdef _WIN32
    _putenv_s(k.c_str(), v ? v->c_str() : "");
#else
    if (v) setenv(k.c_str(), v->c_str(), 1); else unsetenv(k.c_str());
#endif
}

static void apply_arm(const std::vector<ab_arm> & arms, size_t i) {
    std::set<std::string> all;
    for (auto & a : arms) for (auto & kv : a.vars) all.insert(kv.first);
    for (auto & k : all) set_env(k, nullptr);
    for (auto & kv : arms[i].vars) set_env(kv.first, &kv.second);
}

static uint64_t hash_logits(const float * l, size_t n) {
    uint64_t h = 1469598103934665603ull;
    const unsigned char * b = reinterpret_cast<const unsigned char *>(l);
    for (size_t i = 0; i < n * sizeof(float); ++i) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}

// synthetic --bench prompt / fill ids: (i * mul + add) mod min(200000, vocab). The 248K-vocabulary models keep the
// historical ids (mod 200000); a small test vocabulary (tools/mk_tiny_qwen35.py, 4096) stays in range instead of
// reading embedding rows past the table (a crash).
static int32_t synth_id(size_t i, size_t mul, size_t add, int vocab) {
    return (int32_t) ((i * mul + add) % (size_t) std::max(1, std::min(200000, vocab)));
}

// synthetic --bench decode ids: the historical 1000 + i, wrapped to the vocabulary (a no-op on the real models) so a
// small test vocabulary or a long tg run does not read embedding rows past the table
static int32_t tg_id(int i, int vocab) {
    return (int32_t) ((1000 + (size_t) i) % (size_t) std::max(1, vocab));
}

// Untimed warm-up before the first timed --bench measurement: pass(i) (a prefill + a few decode steps) runs at least
// min_passes times and then until P100E_BENCH_WARMUP seconds (default 3; 0 = only the min_passes) have passed. It takes
// the lazy allocations (prefill pipeline, staging buffers), the CUDA graph captures and the GPU clock ramp out of the
// timed region: an idle-clock watcher can keep the P100 clocks locked low while idle and unlock them only
// when it sees load (1 s poll); the first pp2048 of a run started from idle read 413 vs 530 t/s.
// --net-nodes: a fixed number of warm-up passes (a time limit would give each node its own count: lockstep lost)
static int g_warmup_fixed = -1;

static void bench_warmup(const char * tag, int min_passes, const std::function<void(int)> & pass) {
    if (g_warmup_fixed >= 0) {
        for (int n = 0; n < g_warmup_fixed; ++n) pass(n);
        printf("%s warm-up: %d untimed pass(es) (fixed: --net-nodes)\n", tag, g_warmup_fixed);
        fflush(stdout);
        return;
    }
    const char * e = getenv("P100E_BENCH_WARMUP");
    const double min_s = e && *e ? atof(e) : 3.0;
    const auto t0 = std::chrono::steady_clock::now();
    auto secs = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };
    int n = 0;
    while (n < min_passes || (min_s > 0 && n < 1000 && secs() < min_s)) pass(n++);
    if (n) printf("%s warm-up: %d untimed pass(es), %.2f s (P100E_BENCH_WARMUP %g s)\n", tag, n, secs(), min_s);
    fflush(stdout);
}

// KLD(p || q) of softmax(a) and softmax(b), natural log
static double kld(const std::vector<float> & a, const std::vector<float> & b) {
    double ma = -INFINITY, mb = -INFINITY;
    for (size_t i = 0; i < a.size(); ++i) { ma = std::max(ma, (double) a[i]); mb = std::max(mb, (double) b[i]); }
    double za = 0, zb = 0;
    for (size_t i = 0; i < a.size(); ++i) { za += std::exp(a[i] - ma); zb += std::exp(b[i] - mb); }
    const double la = std::log(za), lb = std::log(zb);
    double k = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double lpa = a[i] - ma - la, lpb = b[i] - mb - lb;
        k += std::exp(lpa) * (lpa - lpb);
    }
    return k;
}
#include "sessions_mode.inc"
#include "cache_mode.inc"

// P100E_PRINT_LOGITS=1: FNV-1a hash of the raw logit bits per step, to compare runs bit-exactly
// --net-nodes: one phase's network share. "net" = wall time inside the TCP collectives (incl. waiting for the slower
// node), bytes = payload sent + received by this node
static void net_report(const char * phase, int tokens, double sec, const net_stats & a, const net_stats & b) {
    const double ns = b.sec - a.sec;
    printf("NET %s: %d tokens in %.3f s = %.2f tok/s; network %.1f ms (%.1f%% of the time) in %llu collectives, "
           "%.2f MB out + %.2f MB in (%.1f KB per token)\n", phase, tokens, sec, sec > 0 ? tokens / sec : 0.0, 1e3 * ns,
           sec > 0 ? 100.0 * ns / sec : 0.0, (unsigned long long) (b.calls - a.calls), (b.bytes_out - a.bytes_out) / 1e6,
           (b.bytes_in - a.bytes_in) / 1e6, tokens ? (double) (b.bytes_out - a.bytes_out + b.bytes_in - a.bytes_in) / 1e3 / tokens : 0.0);
    fflush(stdout);
}

// P100E_LOGITS_DUMP=file: every step's logits (prompt, then each gen step) appended to file as raw f32 (cross-run
// max |diff| / top-1 comparisons, e.g. the hybrid pipeline vs one process)
static void print_logit_hash(const char * tag, int step, const std::vector<float> & l) {
    static const char * dump = getenv("P100E_LOGITS_DUMP");
    if (dump && *dump && !l.empty()) {
        static FILE * df = fopen(dump, "wb");
        if (df) { fwrite(l.data(), 4, l.size(), df); fflush(df); }
    }
    static const bool on = getenv("P100E_PRINT_LOGITS") && atoi(getenv("P100E_PRINT_LOGITS")) != 0;
    if (!on) return;
    uint64_t h = 1469598103934665603ull;
    const unsigned char * b = reinterpret_cast<const unsigned char *>(l.data());
    for (size_t i = 0; i < l.size() * sizeof(float); ++i) { h ^= b[i]; h *= 1099511628211ull; }
    printf("logits %s %d: hash %016llx\n", tag, step, (unsigned long long) h);
}

static std::vector<int> topk(const std::vector<float> & l, int k) {
    std::vector<int> idx(l.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int a, int b) { return l[a] > l[b]; });
    idx.resize(k);
    return idx;
}

int run_vision(int argc, char ** argv);   // tools/p100e_run_vision.cpp: image mode (--mmproj)
int run_seq(int argc, char ** argv);      // --seq a b ::: c d: several runs of this executable in one invocation
static const char * kv_name(uint32_t t) {
    return t == fast::KV_F16 ? "f16" : t == fast::KV_Q4_0 ? "q4_0" : "q8_0";
}

// KL divergence vs a llama-perplexity base file for one KV cache type (see the header comment)
static int run_kld(const char * model_path, const std::vector<int> & devices, int ctx, int ubatch,
                   const std::vector<uint32_t> & kvs, const char * base_path, int max_chunks, const char * dump,
                   int n_threads) {
    using clk = std::chrono::steady_clock;
    auto secs = [](clk::time_point a, clk::time_point b) { return std::chrono::duration<double>(b - a).count(); };
    if (devices.empty()) throw std::runtime_error("--kld-base needs --devices");
    kld_base_file kf(base_path);
    const int n_ctx = (int) kf.n_ctx, n_vocab = kf.n_vocab, first = kf.first, ns = kf.n_scored;
    const int n_chunk = max_chunks > 0 ? std::min(max_chunks, kf.n_chunk) : kf.n_chunk;
    const int mb = ubatch > 0 ? std::min(ubatch, n_ctx) : n_ctx;
    if (n_threads <= 0) n_threads = (int) std::max(1u, std::thread::hardware_concurrency());
    printf("kld: base %s: n_ctx %d, n_vocab %d, %d chunks (evaluating %d), %d scored rows per chunk (positions %d..%d), "
           "ubatch %d, %d CPU threads\n", base_path, n_ctx, n_vocab, kf.n_chunk, n_chunk, ns, first, n_ctx - 2, mb, n_threads);
    // logits of one chunk's scored rows (pinned: the D2H copies of both GPUs run async)
    float * lg = nullptr;
    std::vector<float> lg_pageable;
    if (cudaMallocHost(&lg, (size_t) ns * n_vocab * 4) != cudaSuccess) {
        cudaGetLastError();
        lg_pageable.resize((size_t) ns * n_vocab);
        lg = lg_pageable.data();
        printf("kld: pinned host buffer failed, using pageable memory\n");
    }
    std::vector<uint16_t> base[2];
    for (uint32_t kv : kvs) {
        const auto t0 = clk::now();
        printf("\n==== KLD arm: KV %s ====\n", kv_name(kv));
        std::unique_ptr<tp_model> mt = std::make_unique<tp_model>(model_path, devices, std::max(ctx, n_ctx), mb, false, kv);
        if (mt->vocab() != n_vocab)
            throw std::runtime_error("kld: inconsistent vocabulary (" + std::to_string(n_vocab) + " vs " + std::to_string(mt->vocab()) + ")");
        const gguf_value * ab = mt->gguf().get("tokenizer.ggml.add_bos_token");
        const bool add_bos = ab && ab->u != 0;
        const int32_t bos = (int32_t) mt->gguf().get_int("tokenizer.ggml.bos_token_id", -1);
        if (add_bos && bos < 0) throw std::runtime_error("kld: add_bos_token without bos_token_id");
        printf("load %.2f s, KV %s, ctx %d, ubatch %d, add_bos %d (bos %d)\n", secs(t0, clk::now()), kv_name(kv),
               std::max(ctx, n_ctx), mt->max_batch(), (int) add_bos, bos);
        printf("kl_divergence: computing over %d chunks, n_ctx=%d, batch_size=%d, n_seq=1\n", n_chunk, n_ctx, mb);
        fflush(stdout);
        kld_stats st;
        std::vector<float> kld_values((size_t) n_chunk * ns), p_diff_values((size_t) n_chunk * ns);
        double t_fwd = 0, t_head = 0, t_wait = 0, t_kld = 0;
        int n_redo = 0;   // chunks redone because they saturated the f16 prefill wire
        auto fut = std::async(std::launch::async, [&] { kf.read_chunk(0, base[0]); });
        const auto ta = clk::now();
        for (int c = 0; c < n_chunk; ++c) {
            const auto tc = clk::now();
            const int32_t * ctoks = kf.tokens.data() + (size_t) c * n_ctx;
            std::vector<int32_t> tk(ctoks, ctoks + n_ctx);
            if (add_bos) tk[0] = bos;   // perplexity.cpp: BOS replaces the first token of each chunk (batch only)
            std::vector<float> last;   // the forward's own logits of its last row (position n_ctx - 1)
            // a chunk that saturated the f16 prefill wire is redone (the instance is on the f32 wire from then on)
            for (int attempt = 0; attempt < 2; ++attempt) {
                const uint64_t sat0 = mt->prefill_saturations();
                mt->reset();
                int i = 0;
                for (int ch : mt->ubatch_chunks(n_ctx)) {
                    auto a = clk::now();
                    last = mt->forward_n(std::vector<int32_t>(tk.begin() + i, tk.begin() + i + ch), 1, false);
                    auto b = clk::now();
                    t_fwd += secs(a, b);
                    // this forward's rows at scored positions [first, n_ctx - 2]
                    const int lo = std::max(first, i), hi = std::min(i + ch, n_ctx - 1);
                    if (hi > lo) mt->logits_rows(lo - i, hi - lo, lg + (size_t) (lo - first) * n_vocab);
                    t_head += secs(b, clk::now());
                    i += ch;
                }
                if (mt->prefill_saturations() == sat0) break;
                printf("kld: chunk %d saturated the f16 prefill wire (%llu partial groups): redone on the f32 wire\n",
                       c + 1, (unsigned long long) (mt->prefill_saturations() - sat0));
                ++n_redo;
            }
            if (c == 0) {
                // consistency: logits_rows of the last row (one row = the T = 1 head the forward itself used) must
                // equal the forward's returned logits; count non-finite values in the scored rows
                std::vector<float> chk(n_vocab);
                mt->logits_rows(mt->last_tokens() - 1, 1, chk.data());
                double md = 0;
                int nf = 0, nr = 0;
                for (int v = 0; v < n_vocab; ++v) {
                    nf += !std::isfinite(last[v]);
                    nr += !std::isfinite(chk[v]);
                    if (std::isfinite(last[v]) && std::isfinite(chk[v])) md = std::max(md, (double) std::fabs(last[v] - chk[v]));
                }
                size_t ns_bad = 0;
                for (size_t k = 0; k < (size_t) ns * n_vocab; ++k) ns_bad += !std::isfinite(lg[k]);
                printf("kld check (chunk 1): last row: forward vs logits_rows max |diff| %g, non-finite %d / %d; "
                       "scored rows non-finite %zu of %zu\n", md, nf, nr, ns_bad, (size_t) ns * n_vocab);
                if (nf || nr || ns_bad) throw std::runtime_error("kld: non-finite logits, results would be NaN");
            }
            auto a = clk::now();
            fut.get();
            t_wait += secs(a, clk::now());
            if (c + 1 < n_chunk) fut = std::async(std::launch::async, [&, c] { kf.read_chunk(c + 1, base[(c + 1) % 2]); });
            a = clk::now();
            // tokens: the file's (not the BOS-replaced) ids, row r predicts position first + r + 1
            kld_rows(n_vocab, lg, ctoks + first, ns, base[c % 2].data(), st, kld_values.data() + (size_t) c * ns,
                     p_diff_values.data() + (size_t) c * ns, n_threads);
            t_kld += secs(a, clk::now());
            if (c == 0) {
                const double tp = secs(tc, clk::now());
                printf("kl_divergence: %.2f seconds per pass - ETA %.2f minutes\n\n", tp, tp * n_chunk / 60.0);
                kld_print_header(stdout);
            }
            kld_print_chunk(stdout, c + 1, st);
            fflush(stdout);
        }
        printf("\n");
        kld_print_final(stdout, st, kld_values, p_diff_values);
        printf("kld timing (KV %s): %.1f s for %d chunks: forward %.1f s, LM head + copy %.1f s, base read wait %.1f s, "
               "KLD math %.1f s\n", kv_name(kv), secs(ta, clk::now()), n_chunk, t_fwd, t_head, t_wait, t_kld);
        printf("kld: prefill wire: %llu saturated f16 partial groups, %d chunk(s) redone on the f32 wire\n",
               (unsigned long long) mt->prefill_saturations(), n_redo);
        kld_print_summary(stdout, kv_name(kv), st, kld_values);
        if (dump) {
            std::string path = dump;
            if (kvs.size() > 1) {   // one file per arm: out.bin -> out.q8_0.bin
                const size_t sl = path.find_last_of("/\\"), dot = path.find_last_of('.');
                const size_t at = dot != std::string::npos && (sl == std::string::npos || dot > sl) ? dot : path.size();
                path.insert(at, std::string(".") + kv_name(kv));
            }
            FILE * f = fopen(path.c_str(), "wb");
            if (!f) throw std::runtime_error("kld: cannot write " + path);
            for (size_t k = 0; k < kld_values.size(); ++k) {
                const float v[2] = {kld_values[k], p_diff_values[k]};
                fwrite(v, 4, 2, f);
            }
            fclose(f);
            printf("kld: wrote %zu per-token (KLD, p_diff) pairs to %s\n", kld_values.size(), path.c_str());
        }
        fflush(stdout);
    }
    if (lg_pageable.empty()) cudaFreeHost(lg);
    return 0;
}



// --kl-pair (with --kv REF,TEST, --devices, --prompts-file / --tokens-file): direct KL(REF || TEST) of the next-token
// distribution between two KV cache types of the same weights. Every prompt is one chunk run from an empty context in
// one forward; rows len/2 .. len-2 are scored (like the perplexity tool). The REF arm's scored logits are kept in host
// memory (float32), the REF model is freed, the TEST arm is loaded and compared row by row. Prints mean KL(REF||TEST),
// mean KL(TEST||REF), top-1 agreement, 99% KL and the per-chunk means.
static int run_klpair(const char * model_path, const std::vector<int> & devices, int ctx, int ubatch,
                      const std::vector<uint32_t> & kvs, const std::vector<std::vector<int>> & prompts_in) {
    if (devices.empty() || kvs.size() != 2 || prompts_in.empty()) throw std::runtime_error("--kl-pair needs --devices, --kv REF,TEST and prompts");
    std::vector<std::vector<float>> ref(prompts_in.size());
    std::vector<int> first(prompts_in.size()), ns(prompts_in.size());
    int n_vocab = 0;
    std::vector<std::vector<double>> kl_rt(prompts_in.size()), kl_tr(prompts_in.size());
    std::vector<size_t> agree(prompts_in.size(), 0);
    for (int arm = 0; arm < 2; ++arm) {
        std::unique_ptr<tp_model> mt = std::make_unique<tp_model>(model_path, devices, ctx, ubatch > 0 ? ubatch : 2048, false, kvs[arm]);
        n_vocab = mt->vocab();
        printf("klpair arm %d: KV %s loaded\n", arm, kv_name(kvs[arm]));
        mt->print_mem("klpair");
        fflush(stdout);
        for (size_t c = 0; c < prompts_in.size(); ++c) {
            std::vector<int32_t> tk(prompts_in[c].begin(), prompts_in[c].end());
            const int n = (int) tk.size();
            first[c] = n / 2; ns[c] = n - 1 - first[c];
            std::vector<float> lg((size_t) ns[c] * n_vocab);
            for (int attempt = 0; attempt < 2; ++attempt) {
                const uint64_t sat0 = mt->prefill_saturations();
                mt->reset();
                int i = 0;
                for (int ch : mt->ubatch_chunks(n)) {
                    mt->forward_n(std::vector<int32_t>(tk.begin() + i, tk.begin() + i + ch), 1, false);
                    const int lo = std::max(first[c], i), hi = std::min(i + ch, n - 1);
                    if (hi > lo) mt->logits_rows(lo - i, hi - lo, lg.data() + (size_t) (lo - first[c]) * n_vocab);
                    i += ch;
                }
                if (mt->prefill_saturations() == sat0) break;
                printf("klpair: chunk %zu saturated the f16 wire, redone\n", c);
            }
            for (float v : lg) if (!std::isfinite(v)) throw std::runtime_error("klpair: non-finite logits");
            if (arm == 0) { ref[c] = std::move(lg); continue; }
            const int R = ns[c];
            kl_rt[c].assign(R, 0.0); kl_tr[c].assign(R, 0.0);
            std::vector<char> same(R, 0);
            const int NT = 8;
            std::vector<std::future<void>> fs;
            for (int t = 0; t < NT; ++t) fs.push_back(std::async(std::launch::async, [&, t] {
                std::vector<float> a(n_vocab), b(n_vocab);
                for (int r = t; r < R; r += NT) {
                    memcpy(a.data(), ref[c].data() + (size_t) r * n_vocab, (size_t) n_vocab * 4);
                    memcpy(b.data(), lg.data() + (size_t) r * n_vocab, (size_t) n_vocab * 4);
                    kl_rt[c][r] = kld(a, b);
                    kl_tr[c][r] = kld(b, a);
                    same[r] = std::max_element(a.begin(), a.end()) - a.begin() == std::max_element(b.begin(), b.end()) - b.begin();
                }
            }));
            for (auto & f : fs) f.get();
            for (char x : same) agree[c] += x;
        }
    }
    double s_rt = 0, s_tr = 0; size_t tot = 0, ag = 0;
    std::vector<double> all;
    for (size_t c = 0; c < prompts_in.size(); ++c) {
        double a = 0, b = 0;
        for (size_t r = 0; r < kl_rt[c].size(); ++r) { a += kl_rt[c][r]; b += kl_tr[c][r]; all.push_back(kl_rt[c][r]); }
        const size_t R = kl_rt[c].size();
        printf("KLPAIR chunk %zu: %d tokens, %zu scored rows, KL(%s||%s) mean %.6f, KL(%s||%s) mean %.6f, top-1 agree %.2f%%\n", c,
               (int) prompts_in[c].size(), R, kv_name(kvs[0]), kv_name(kvs[1]), a / R, kv_name(kvs[1]), kv_name(kvs[0]), b / R, 100.0 * agree[c] / R);
        s_rt += a; s_tr += b; tot += R; ag += agree[c];
    }
    std::sort(all.begin(), all.end());
    double var = 0; const double m = s_rt / tot;
    for (double v : all) var += (v - m) * (v - m);
    printf("KLPAIR SUMMARY: %zu rows, KL(%s||%s) mean %.6f +- %.6f, median %.6f, 99%% %.6f, max %.4f; KL(%s||%s) mean %.6f; top-1 agree %.2f%%\n",
           tot, kv_name(kvs[0]), kv_name(kvs[1]), m, std::sqrt(var / (tot - 1) / tot), all[all.size() / 2], all[(size_t) (0.99 * all.size())],
           all.back(), kv_name(kvs[1]), kv_name(kvs[0]), s_tr / tot, 100.0 * ag / tot);
    fflush(stdout);
    return 0;
}

// Batch (several checks in one command): p100e-run <args 1> ++ <args 2> ++ ... runs each group as a child
// process of this exe, in order (one engine per process); a group may start with --env K=V[,K2=V2..] (set for that
// child only). Prints "==== BATCH i/n" headers, each child's exit code and a summary; exit 0 when every child exited 0.
static int run_batch(int argc, char ** argv) {
#ifdef _WIN32
    std::vector<std::vector<std::string>> groups(1);
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "++")) groups.emplace_back();
        else groups.back().push_back(argv[i]);
    }
    char exe[MAX_PATH];
    if (!GetModuleFileNameA(nullptr, exe, MAX_PATH)) return 1;
    int n_ok = 0;
    std::string codes;
    using clk = std::chrono::steady_clock;
    for (size_t g = 0; g < groups.size(); ++g) {
        std::vector<std::string> a = groups[g];
        std::vector<std::string> env;
        while (a.size() >= 2 && a[0] == "--env") {
            std::stringstream ss(a[1]);
            std::string kv;
            while (std::getline(ss, kv, ',')) if (kv.find('=') != std::string::npos) env.push_back(kv);
            a.erase(a.begin(), a.begin() + 2);
        }
        std::string line;
        for (const auto & x : a) line += " " + x;
        printf("==== BATCH %zu/%zu:%s\n", g + 1, groups.size(), line.c_str());
        for (const auto & kv : env) printf("     env %s\n", kv.c_str());
        fflush(stdout);
        // the values this process had: put back after the group (an empty _putenv_s value would delete them)
        std::vector<std::pair<std::string, std::string>> saved;
        std::vector<bool> had;
        for (const auto & kv : env) {
            const std::string k = kv.substr(0, kv.find('='));
            const char * o = getenv(k.c_str());
            saved.push_back({k, o ? o : ""});
            had.push_back(o != nullptr);
            _putenv_s(k.c_str(), kv.substr(kv.find('=') + 1).c_str());
        }
        std::vector<std::string> q{cache_quote(exe)};
        for (const auto & x : a) q.push_back(cache_quote(x));
        std::vector<const char *> cv;
        for (const auto & x : q) cv.push_back(x.c_str());
        cv.push_back(nullptr);
        const auto t0 = clk::now();
        const int rc = (int) _spawnv(_P_WAIT, exe, cv.data());
        for (size_t j = saved.size(); j-- > 0;) set_env(saved[j].first, had[j] ? &saved[j].second : nullptr);   // reverse: repeats
        printf("==== BATCH %zu/%zu exit %d (%.1f s)\n", g + 1, groups.size(), rc,
               std::chrono::duration<double>(clk::now() - t0).count());
        fflush(stdout);
        n_ok += rc == 0;
        codes += (codes.empty() ? "" : " ") + std::to_string(rc);
    }
    printf("BATCH: %d/%zu runs exited 0 (exit codes %s)\n", n_ok, groups.size(), codes.c_str());
    return n_ok == (int) groups.size() ? 0 : 2;
#else
    (void) argc; (void) argv;
    fprintf(stderr, "batch mode: Windows only\n");
    return 1;
#endif
}

int main(int argc, char ** argv) {
    for (int i = 1; i < argc; ++i)
        if (!strcmp(argv[i], "--net-nodes"))
            for (int j = 1; j < argc; ++j)
                if (!strcmp(argv[j], "--seq") || !strcmp(argv[j], "++") || !strcmp(argv[j], "--mmproj")) {
                    fprintf(stderr, "error: --net-nodes: %s is not supported over the network yet\n", argv[j]);
                    return 2;
                }
    if (argc >= 2 && !strcmp(argv[1], "--seq")) return run_seq(argc, argv);
    for (int i = 1; i < argc; ++i) if (!strcmp(argv[i], "++")) return run_batch(argc, argv);
    for (int i = 2; i < argc; ++i) if (!strcmp(argv[i], "--mmproj")) return run_vision(argc, argv);
    if (argc < 3) {
        fprintf(stderr, "usage: p100e-run <model.gguf> --tokens a,b,c [--gen N] [--ctx N] [--debug] [--logits file]\n");
        return 1;
    }
    std::vector<int32_t> toks;
    std::vector<std::vector<int32_t>> prompts;
    std::vector<int> mtp_k;
    std::vector<int> draft_vocabs;   // --draft-vocab N[,N..]: draft LM head over the N most frequent tokens (0 = full head)
    std::vector<float> taus;         // --mtp-adaptive tau[,tau..]: adaptive draft length, K = the maximum (0 = fixed K)
    int kmin = 1;                    // --mtp-kmin: adaptive minimum drafts per step
    float tau1 = 0.0f;               // --mtp-tau1 t: k_min = 0, the first draft needs p >= t (else a plain T = 1 step)
    int gen = 0, ctx = 4096;
    bool debug = false;
    const char * logits_out = nullptr;
    std::vector<int> devices;
    int bench_pp = 0, bench_tg = 0;
    uint32_t kv_type = fast::KV_Q8_0;
    std::vector<uint32_t> kv_types;  // --kv list (--kld-base: one arm per type; other modes use the first)
    bool klpair = false;
    const char * kld_base = nullptr, * kld_dump = nullptr;
    int kld_chunks = 0, kld_threads = 0;
    int ubatch = 0, bench_real_max = INT_MAX;
    std::vector<int> bench_depths, forced;
    bool load_mtp = false;   // --load-mtp: load the nextn block + draft buffers without MTP decoding (memory checks)
    std::string ab_spec;     // --ab
    int ab_reps = 3;         // --ab-reps
    bool wire_scan = false;  // --wire-scan
    std::vector<int> sessions;       // --sessions n1[,n2..]
    sessions_args SA;
    bool sessions_mode = false;
    cache_args CA;
    bool cache_mode = false, sched_mode = false, fair_mode = false;
    std::vector<std::string> net_nodes;   // --net-nodes host:port,host:port
    int net_node = -1;                    // --net-node i
    bool net_logits = false;              // --net-logits: full logits every decode step (else greedy ids only)
    int pipe_layers = 0, pipe_total = 0;  // --pipe-layers k|auto / --pipe-split a,b
    for (int i = 0; i < argc; ++i) CA.argv.push_back(argv[i]);
    auto parse_list = [](const char * s) {
        std::vector<int> v;
        std::stringstream ss(s);
        std::string t;
        while (std::getline(ss, t, ',')) if (!t.empty()) v.push_back(std::stoi(t));
        return v;
    };
    for (int i = 2; i < argc; ++i) {
        if (!strcmp(argv[i], "--tokens") && i + 1 < argc) {
            std::stringstream ps(argv[++i]);
            std::string pr;
            while (std::getline(ps, pr, ';')) {
                std::stringstream ss(pr);
                std::string t;
                prompts.emplace_back();
                while (std::getline(ss, t, ',')) prompts.back().push_back(std::stoi(t));
                if (prompts.back().empty()) prompts.pop_back();
            }
            if (!prompts.empty()) toks = prompts[0];
        } else if (!strcmp(argv[i], "--mtp") && i + 1 < argc) {
            std::stringstream ss(argv[++i]);
            std::string t;
            while (std::getline(ss, t, ',')) mtp_k.push_back(std::stoi(t));
        } else if (!strcmp(argv[i], "--draft-vocab") && i + 1 < argc) {
            std::stringstream ss(argv[++i]);
            std::string t;
            while (std::getline(ss, t, ',')) draft_vocabs.push_back(std::stoi(t));
        } else if (!strcmp(argv[i], "--mtp-adaptive") && i + 1 < argc) {
            std::stringstream ss(argv[++i]);
            std::string t;
            while (std::getline(ss, t, ',')) taus.push_back(std::stof(t));
        } else if (!strcmp(argv[i], "--mtp-kmin") && i + 1 < argc) { kmin = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--mtp-tau1") && i + 1 < argc) { tau1 = (float) atof(argv[++i]); kmin = 0;
        } else if (!strcmp(argv[i], "--gen") && i + 1 < argc) gen = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ctx") && i + 1 < argc) ctx = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--debug")) debug = true;
        else if (!strcmp(argv[i], "--logits") && i + 1 < argc) logits_out = argv[++i];
        else if (!strcmp(argv[i], "--bench") && i + 2 < argc) { bench_pp = atoi(argv[++i]); bench_tg = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--kv") && i + 1 < argc) {
            std::stringstream ss(argv[++i]);
            std::string k;
            kv_types.clear();
            while (std::getline(ss, k, ',')) {
                if (k == "f16") kv_types.push_back(fast::KV_F16);
                else if (k == "q8_0") kv_types.push_back(fast::KV_Q8_0);
                else if (k == "q4_0") kv_types.push_back(fast::KV_Q4_0);
                else { fprintf(stderr, "--kv: f16, q8_0 or q4_0 (comma list for --kld-base)\n"); return 1; }
            }
            if (kv_types.empty()) { fprintf(stderr, "--kv: empty list\n"); return 1; }
            kv_type = kv_types[0];
        }
        else if (!strcmp(argv[i], "--kl-pair")) klpair = true;
        else if (!strcmp(argv[i], "--kld-base") && i + 1 < argc) kld_base = argv[++i];
        else if (!strcmp(argv[i], "--kld-chunks") && i + 1 < argc) kld_chunks = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--kld-dump") && i + 1 < argc) kld_dump = argv[++i];
        else if (!strcmp(argv[i], "--kld-threads") && i + 1 < argc) kld_threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ubatch") && i + 1 < argc) ubatch = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--bench-depth") && i + 1 < argc) bench_depths = parse_list(argv[++i]);
        else if (!strcmp(argv[i], "--bench-real-max") && i + 1 < argc) bench_real_max = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--force") && i + 1 < argc) forced = parse_list(argv[++i]);
        else if (!strcmp(argv[i], "--sessions") && i + 1 < argc) { sessions = parse_list(argv[++i]); sessions_mode = true; }
        else if (!strcmp(argv[i], "--sessions-sync")) SA.sync = true;
        else if (!strcmp(argv[i], "--offload-test") && i + 1 < argc) { SA.offload_n = atoi(argv[++i]); sessions_mode = true; }
        else if (!strcmp(argv[i], "--sessions-pool") && i + 1 < argc) {
            ++i;
            SA.pool = !strcmp(argv[i], "auto") ? -1 : atoll(argv[i]);
        }
        else if (!strcmp(argv[i], "--reserve-mb") && i + 1 < argc) {
            SA.reserve_mb.clear();
            for (int v : parse_list(argv[++i])) SA.reserve_mb.push_back((size_t) v);
        }
        else if (!strcmp(argv[i], "--max-active") && i + 1 < argc) SA.max_active = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--cache-test")) cache_mode = true;
        else if (!strcmp(argv[i], "--sched-test")) sched_mode = true;
        else if (!strcmp(argv[i], "--fair-test") && i + 1 < argc) { fair_mode = true; CA.fair_prompt = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--fair-shares") && i + 1 < argc) {
            CA.fair_shares.clear();
            std::stringstream ss(argv[++i]);
            std::string t;
            while (std::getline(ss, t, ',')) if (!t.empty()) CA.fair_shares.push_back(std::stod(t));
        }
        else if (!strcmp(argv[i], "--cache-a") && i + 1 < argc) CA.a = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--cache-b") && i + 1 < argc) CA.b = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--cache-turns") && i + 1 < argc) CA.turns = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--cache-turn0") && i + 1 < argc) CA.turn0 = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--cache-step") && i + 1 < argc) CA.step = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--cache-gen") && i + 1 < argc) CA.cgen = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--cache-no-disk")) CA.disk = false;
        else if (!strcmp(argv[i], "--cache-no-variant")) CA.no_split_variant = false;
        else if (!strcmp(argv[i], "--cache-no-offload")) CA.offload_turns = false;
        else if (!strcmp(argv[i], "--cache-save-ckpts") && i + 1 < argc) CA.save_ckpts = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--cache-dir") && i + 1 < argc) CA.dir = argv[++i];
        else if (!strcmp(argv[i], "--cache-only") && i + 1 < argc) CA.only = argv[++i];
        else if (!strcmp(argv[i], "--cache-dturns") && i + 1 < argc) CA.dturns = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--cache-dgen") && i + 1 < argc) CA.dgen = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ckpt-every") && i + 1 < argc) CA.ckpt_every = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--min-split") && i + 1 < argc) CA.min_split = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--cache-inexact")) CA.inexact = true;
        else if (!strcmp(argv[i], "--sched-pool") && i + 1 < argc) CA.sched_pool = atoll(argv[++i]);
        else if (!strcmp(argv[i], "--cache-child") && i + 2 < argc) { CA.child_sess = argv[++i]; CA.child_expect = argv[++i]; }
        else if (!strcmp(argv[i], "--draft-rows") && i + 1 < argc) SA.draft_rows = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--decode-per-chunk") && i + 1 < argc) SA.decode_per_chunk = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--decode-share") && i + 1 < argc) SA.decode_share = atof(argv[++i]);
        else if (!strcmp(argv[i], "--tokens-files") && i + 1 < argc) {
            // several prompts, one file each (comma-separated file list; ids as for --tokens-file)
            std::stringstream fl(argv[++i]);
            std::string fn;
            prompts.clear();
            while (std::getline(fl, fn, ',')) {
                FILE * f = fopen(fn.c_str(), "rb");
                if (!f) { fprintf(stderr, "cannot open %s\n", fn.c_str()); return 1; }
                std::string txt;
                char buf[65536];
                size_t n;
                while ((n = fread(buf, 1, sizeof buf, f)) > 0) txt.append(buf, n);
                fclose(f);
                for (char & ch : txt) if (ch == '[' || ch == ']' || ch == ',' || ch == '\r' || ch == '\n') ch = ' ';
                std::stringstream ss(txt);
                prompts.emplace_back();
                for (long long v; ss >> v;) prompts.back().push_back((int32_t) v);
            }
            if (!prompts.empty()) toks = prompts[0];
        }
        else if (!strcmp(argv[i], "--load-mtp")) load_mtp = true;
        else if (!strcmp(argv[i], "--ab") && i + 1 < argc) ab_spec = argv[++i];
        else if (!strcmp(argv[i], "--ab-reps") && i + 1 < argc) ab_reps = std::max(1, atoi(argv[++i]));
        else if (!strcmp(argv[i], "--tokens-file") && i + 1 < argc) {
            // one prompt: comma / whitespace separated ids (e.g. llama-tokenize --ids output, brackets ignored)
            FILE * f = fopen(argv[++i], "rb");
            if (!f) { fprintf(stderr, "cannot open %s\n", argv[i]); return 1; }
            std::string txt;
            char buf[65536];
            size_t n;
            while ((n = fread(buf, 1, sizeof buf, f)) > 0) txt.append(buf, n);
            fclose(f);
            for (char & ch : txt) if (ch == '[' || ch == ']' || ch == ',' || ch == '\r' || ch == '\n') ch = ' ';
            std::stringstream ss(txt);
            prompts.assign(1, {});
            for (long long v; ss >> v;) prompts[0].push_back((int32_t) v);
            toks = prompts[0];
        }
        else if (!strcmp(argv[i], "--prompts-file") && i + 1 < argc) {
            FILE * f = fopen(argv[++i], "rb");
            if (!f) { fprintf(stderr, "cannot open %s\n", argv[i]); return 1; }
            std::string txt;
            char buf[65536];
            size_t n;
            while ((n = fread(buf, 1, sizeof buf, f)) > 0) txt.append(buf, n);
            fclose(f);
            prompts.clear();
            std::stringstream ls(txt);
            for (std::string line; std::getline(ls, line);) {
                for (char & ch : line) if (ch == '[' || ch == ']' || ch == ',' || ch == '\r') ch = ' ';
                std::stringstream ss(line);
                std::vector<int32_t> p;
                for (long long v; ss >> v;) p.push_back((int32_t) v);
                if (!p.empty()) prompts.push_back(p);
            }
            if (!prompts.empty()) toks = prompts[0];
        }
        else if (!strcmp(argv[i], "--wire-scan")) wire_scan = true;
        else if (!strcmp(argv[i], "--tokens-max") && i + 1 < argc) {
            const size_t m = (size_t) atoi(argv[++i]);
            for (auto & p : prompts) if (p.size() > m) p.resize(m);
            if (toks.size() > m) toks.resize(m);
        }
        else if (!strcmp(argv[i], "--devices") && i + 1 < argc) {
            std::stringstream ss(argv[++i]);
            std::string t;
            while (std::getline(ss, t, ',')) devices.push_back(std::stoi(t));
        }
        else if (!strcmp(argv[i], "--split") && i + 1 < argc) {
            // tensor-parallel weights: checked here (a clear error before any load), passed as P100E_SPLIT so every
            // tp_model of this run (KLD arms, sessions, benches) uses them
            const std::string s = argv[++i];
            try { (void) tp_parse_split(s); }
            catch (const std::exception & e) { fprintf(stderr, "error: --split: %s\n", e.what()); return 2; }
            set_env("P100E_SPLIT", &s);
        }
        else if (!strcmp(argv[i], "--net-nodes") && i + 1 < argc) {
            std::stringstream ss(argv[++i]);
            std::string t;
            while (std::getline(ss, t, ',')) if (!t.empty()) net_nodes.push_back(t);
        }
        else if (!strcmp(argv[i], "--net-node") && i + 1 < argc) net_node = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--net-logits")) net_logits = true;
        else if (!strcmp(argv[i], "--pipe-layers") && i + 1 < argc) {
            ++i;
            pipe_layers = !strcmp(argv[i], "auto") ? -1 : atoi(argv[i]);
            if (pipe_layers == 0) { fprintf(stderr, "error: --pipe-layers: k >= 1 (node 0 runs layers 0..k-1) or auto\n"); return 2; }
        }
        else if (!strcmp(argv[i], "--pipe-split") && i + 1 < argc) {
            const std::vector<int> v = parse_list(argv[++i]);
            if (v.size() != 2 || v[0] < 1 || v[1] < 1) {
                fprintf(stderr, "error: --pipe-split a,b: two layer counts >= 1 (node 0 runs the first a, node 1 the last b)\n");
                return 2;
            }
            pipe_layers = v[0];
            pipe_total = v[0] + v[1];
        }
    }
    if (pipe_layers != 0 && net_nodes.empty() && !getenv("P100E_NET_NODES")) {
        fprintf(stderr, "error: --pipe-layers / --pipe-split need --net-nodes (two nodes)\n");
        return 2;
    }
    if (net_nodes.empty())
        if (const char * e = getenv("P100E_NET_NODES"); e && *e) {
            std::stringstream ss(e);
            std::string t;
            while (std::getline(ss, t, ',')) if (!t.empty()) net_nodes.push_back(t);
        }
    if (net_node < 0) net_node = getenv("P100E_NET_NODE") ? atoi(getenv("P100E_NET_NODE")) : 0;
    const bool net_on = !net_nodes.empty();
    if (net_on) g_warmup_fixed = 2;
    uint64_t net_args = 1469598103934665603ull;   // the run arguments every node must share (not the model path, --devices, --net-node)
    if (net_on) {
        const char * bad = devices.empty() ? "no --devices (this node's GPUs)" : klpair ? "--kl-pair" : kld_base ? "--kld-base"
                         : (cache_mode || sched_mode || fair_mode || !CA.child_sess.empty()) ? "--cache-test / --sched-test / --fair-test"
                         : sessions_mode ? "--sessions" : !mtp_k.empty() ? "--mtp" : load_mtp ? "--load-mtp"
                         : !ab_spec.empty() ? "--ab" : wire_scan ? "--wire-scan" : nullptr;
        if (bad) { fprintf(stderr, "error: --net-nodes: %s is not supported over the network yet\n", bad); return 2; }
        auto mix = [&](const void * p, size_t n) {
            for (size_t k = 0; k < n; ++k) net_args = (net_args ^ ((const unsigned char *) p)[k]) * 1099511628211ull;
        };
        for (int i = 2; i < argc; ++i) {
            // node-local values: this node's GPUs, its index, and file paths (Linux vs Windows; the token ids they hold
            // are hashed below)
            if ((!strcmp(argv[i], "--devices") || !strcmp(argv[i], "--net-node")) && i + 1 < argc) { ++i; continue; }
            // pipe mode: --split weighs THIS node's GPUs only (node 0: --split 1,1 for its P100 pair,
            // node 1 none): node local. The --pipe-* values stay in the hash (both nodes must pass the same)
            if (pipe_layers != 0 && !strcmp(argv[i], "--split") && i + 1 < argc) { ++i; continue; }
            const bool path = !strcmp(argv[i], "--tokens-file") || !strcmp(argv[i], "--prompts-file") ||
                              !strcmp(argv[i], "--tokens-files") || !strcmp(argv[i], "--logits");
            mix(argv[i], strlen(argv[i]) + 1);
            if (path && i + 1 < argc) ++i;
        }
        for (const auto & p : prompts) { mix(p.data(), p.size() * 4); mix("|", 1); }
        mix(toks.data(), toks.size() * 4);
    }
    try {
        if (klpair) return run_klpair(argv[1], devices, ctx, ubatch, kv_types, prompts);
        if (kld_base) {
            if (kv_types.empty()) kv_types.push_back(kv_type);
            return run_kld(argv[1], devices, ctx, ubatch, kv_types, kld_base, kld_chunks, kld_dump, kld_threads);
        }
        if (cache_mode || sched_mode || fair_mode || !CA.child_sess.empty()) {
            if (devices.empty()) throw std::runtime_error("--cache-test / --sched-test / --fair-test need --devices");
            CA.S = SA;
            CA.S.model = argv[1];
            CA.S.devices = devices;
            CA.S.ctx = ctx;
            CA.S.ubatch = ubatch;
            CA.S.gen = gen > 0 ? gen : 32;
            CA.S.kv_type = kv_type;
            CA.S.prompts = prompts;
            CA.S.mtp_k = mtp_k.empty() ? 0 : mtp_k[0];
            CA.S.tau = taus.empty() ? 0.0f : taus[0];
            CA.S.kmin = kmin;
            if (!CA.child_sess.empty()) return run_cache_child(CA);
            // one engine per process: a second tp_model in one process gives non-finite logits
            if ((int) sched_mode + (int) cache_mode + (int) fair_mode > 1)
                throw std::runtime_error("--sched-test / --cache-test / --fair-test: one per process");
            if (fair_mode) return run_fair_test(CA);
            return sched_mode ? run_sched_test(CA) : run_cache_test(CA);
        }
        if (sessions_mode) {
            if (devices.empty()) throw std::runtime_error("--sessions needs --devices");
            SA.model = argv[1];
            SA.devices = devices;
            SA.ctx = ctx;
            SA.ubatch = ubatch;
            SA.gen = gen > 0 ? gen : 32;
            SA.kv_type = kv_type;
            SA.prompts = prompts;
            SA.counts = sessions.empty() ? std::vector<int>{2} : sessions;
            SA.mtp_k = mtp_k.empty() ? 0 : mtp_k[0];
            SA.tau = taus.empty() ? 0.0f : taus[0];
            SA.kmin = kmin;
            return run_sessions(SA);
        }
        using clk = std::chrono::steady_clock;
        auto t0 = clk::now();
        int maxp = 0;
        for (auto & p : prompts) maxp = std::max<int>(maxp, (int) p.size());
        // MTP: the draft head's prompt catch-up runs P + 1 tokens
        int mb = mtp_k.empty() ? std::max<int>({64, wire_scan ? maxp : (int) toks.size(), bench_pp})
                               : std::max<int>({64, maxp + 1, bench_pp + 1});
        if (ubatch > 0) mb = std::min(mb, std::max(ubatch, 16));
        if (!forced.empty()) gen = (int) forced.size();
        const int topk_n = getenv("P100E_TOPK") ? atoi(getenv("P100E_TOPK")) : 0;
        auto print_top = [&](const char * tag, int step, const std::vector<float> & l) {
            if (topk_n <= 0) return;
            printf("TOP %s %d:", tag, step);
            for (int id : topk(l, topk_n)) printf(" %d:%.4f", id, l[id]);
            printf("\n");
        };
        std::unique_ptr<model> m1;
        std::unique_ptr<tp_model> mt;
        int32_t net_last_id = 0;   // net mode without --net-logits: the greedy id of the last decode step
        std::function<std::vector<float>(const std::vector<int32_t> &)> fwd;
        if (devices.empty()) {
            m1 = std::make_unique<model>(argv[1], ctx, mb);
            m1->set_debug(debug);
            fwd = [&](const std::vector<int32_t> & t) { return m1->forward(t); };
        } else if (!net_on) {
            mt = std::make_unique<tp_model>(argv[1], devices, ctx, mb, !mtp_k.empty() || load_mtp, kv_type);
            fwd = [&](const std::vector<int32_t> & t) { return mt->forward(t); };
        } else {
            // network tensor parallel: this node's GPUs, every node's split; decode steps return the
            // greedy id only (argmax pairs over the net, a few bytes) unless --net-logits (every node's logit slices)
            tp_model::opts o;
            o.n_ctx = ctx; o.max_batch = mb; o.kv_type = kv_type;
            o.net_nodes = net_nodes; o.net_node = net_node; o.net_args_hash = net_args;
            o.pipe_layers = pipe_layers; o.pipe_total = pipe_total;   // hybrid pipeline, 0 = off
            mt = std::make_unique<tp_model>(argv[1], devices, o);
            fwd = [&](const std::vector<int32_t> & t) {
                if (!net_logits && t.size() == 1) { net_last_id = mt->forward_ids(t, 1)[0]; return std::vector<float>(); }
                return mt->forward(t);
            };
        }
        auto t1 = clk::now();
        const int n_vocab = mt ? mt->vocab() : m1->hp().n_vocab;   // synthetic bench ids stay below it (synth_id)
        if (wire_scan) {
            if (!mt) throw std::runtime_error("--wire-scan needs --devices");
            if (prompts.empty()) throw std::runtime_error("--wire-scan needs prompts (--tokens a,b;c,d or --prompts-file)");
            const std::string on = "1";
            set_env("P100E_PF_WIRE_STATS", &on);
            float worst = -1.0f;
            int worst_i = -1, n_sat = 0, n_run = 0;
            for (size_t i = 0; i < prompts.size(); ++i) {
                if ((int) prompts[i].size() <= 8) { printf("PFSCAN prompt %zu: %zu tokens, skipped\n", i, prompts[i].size()); continue; }
                mt->reset();
                mt->forward(prompts[i]);
                const float a = mt->last_prefill_wire_amax();
                const bool sat = mt->last_prefill_saturated();
                ++n_run;
                n_sat += sat;
                if (a > worst || a != a) { worst = a; worst_i = (int) i; }
                printf("PFSCAN prompt %zu: %zu tokens, max|partial| %.6g, f16 margin %.1fx%s\n", i, prompts[i].size(), a,
                       a > 0 ? 65504.0 / a : 0.0, sat ? ", SATURATED the f16 wire" : "");
                fflush(stdout);
            }
            printf("PFSCAN corpus: %d prompts, max|partial| %.6g (prompt %d), min f16 margin %.1fx, %d saturated "
                   "(%llu partial groups)\n", n_run, worst, worst_i, worst > 0 ? 65504.0 / worst : 0.0, n_sat,
                   (unsigned long long) mt->prefill_saturations());
            return 0;
        }
        if (!ab_spec.empty()) {
            if (!mt) throw std::runtime_error("--ab needs --devices");
            const std::vector<ab_arm> arms = parse_arms(ab_spec);
            const size_t NA = arms.size();
            printf("load %.2f s, ubatch %d, %zu arms, %d reps\n", std::chrono::duration<double>(t1 - t0).count(),
                   mt->max_batch(), NA, ab_reps);
            for (size_t a = 0; a < NA; ++a) printf("ARM %zu: %s\n", a, arms[a].label.c_str());
            std::unique_ptr<tp_mtp> draft_bufs;   // --load-mtp: the draft buffers' memory; --mtp: the MTP arms' drafter
            if (load_mtp || !mtp_k.empty()) draft_bufs = std::make_unique<tp_mtp>(*mt);
            const size_t V = (size_t) mt->vocab();
            if (!mtp_k.empty()) {
                // MTP arms: the prompt (--tokens / --tokens-file and --gen, or --bench pp tg: a synthetic pp-token prompt and
                // tg tokens); per arm plain greedy (base), then generate_mtp per config, ids vs plain greedy and arm 0
                tp_mtp & drafter = *draft_bufs;
                std::vector<int32_t> pr = toks;
                int n = gen;
                if (bench_pp > 0 || bench_tg > 0) {
                    if (pr.empty()) {
                        pr.resize(std::max(bench_pp, 1));
                        for (size_t i = 0; i < pr.size(); ++i) pr[i] = synth_id(i, 7919, 13, (int) V);
                    }
                    n = bench_tg;
                }
                if (pr.empty() || n < 2) throw std::runtime_error("--ab --mtp: needs a prompt and --gen >= 2 (or --bench pp tg, tg >= 2)");
                if (taus.empty()) taus.push_back(0.0f);
                struct mcfg { int k; float tau; std::string name; };
                std::vector<mcfg> cfgs;
                for (int K : mtp_k) for (float tau : taus) {
                    char c[64];
                    if (tau > 0 && kmin == 0) snprintf(c, sizeof c, "k<=%d tau=%.3f tau1=%.2f", K, tau, tau1 > 0 ? tau1 : tau);
                    else if (tau > 0) snprintf(c, sizeof c, "k<=%d tau=%.3f", K, tau);
                    else snprintf(c, sizeof c, "k=%d", K);
                    cfgs.push_back({K, tau, c});
                }
                // [arm][0 = plain greedy, 1.. = cfgs]: t/s per rep, acceptance, ids seen, runs identical to the arm's plain greedy
                struct mres { std::vector<double> tps; int acc = 0, dr = 0, steps = 0, ident = 0, runs = 0; std::set<std::vector<int32_t>> ids; };
                std::vector<std::vector<mres>> res(NA, std::vector<mres>(cfgs.size() + 1));
                std::vector<int> capt(NA, 0), drops(NA, 0);   // draft graphs captured / dropped for a kernel switch (timed reps)
                auto run_arm = [&](size_t a, int ng, bool record) {
                    apply_arm(arms, a);
                    mt->reload_env();
                    const int c0 = drafter.graph_captures(), d0 = drafter.graph_epoch_drops();
                    mt->reset();
                    std::vector<float> l = mt->forward(pr);
                    std::vector<int32_t> base;
                    auto b = clk::now();
                    for (int i = 0; i < ng; ++i) {
                        base.push_back(argmax(l.data(), (int) V));
                        if (i + 1 < ng) l = mt->forward({base.back()});
                    }
                    const double tb = std::chrono::duration<double>(clk::now() - b).count();
                    if (record) { res[a][0].tps.push_back((ng - 1) / tb); res[a][0].ids.insert(base); }
                    for (size_t c = 0; c < cfgs.size(); ++c) {
                        mt->reset();
                        mtp_stats st;
                        mtp_opts o;
                        o.k = cfgs[c].k; o.tau = cfgs[c].tau; o.k_min = kmin; o.tau1 = tau1;
                        const std::vector<int32_t> ids = generate_mtp(*mt, drafter, pr, ng, o, st);
                        if (!record) continue;
                        mres & R = res[a][c + 1];
                        R.tps.push_back((ng - 1) / st.t_gen);
                        R.acc += st.accepted; R.dr += st.drafted; R.steps += st.steps;
                        R.ident += ids == base; R.runs++;
                        R.ids.insert(ids);
                    }
                    if (record) { capt[a] += drafter.graph_captures() - c0; drops[a] += drafter.graph_epoch_drops() - d0; }
                };
                bench_warmup("AB", (int) NA, [&](int i) { run_arm((size_t) i % NA, std::min(n, 8), false); });
                for (int rep = 0; rep < ab_reps; ++rep)
                    for (size_t k = 0; k < NA; ++k) run_arm(rep % 2 ? NA - 1 - k : k, n, true);
                int n_ident = 0, n_cmp = 0;
                const std::vector<int32_t> * ref = res[0][0].ids.size() == 1 ? &*res[0][0].ids.begin() : nullptr;
                for (size_t a = 0; a < NA; ++a) {
                    for (size_t c = 0; c <= cfgs.size(); ++c) {
                        const mres & R = res[a][c];
                        double s = 0, mn = 1e30, mx = 0;
                        for (double t : R.tps) { s += t; mn = std::min(mn, t); mx = std::max(mx, t); }
                        const bool stable = R.ids.size() == 1, same0 = stable && ref && *R.ids.begin() == *ref;
                        printf("AB mtp arm %zu %-5s: %.2f t/s (min %.2f max %.2f, n %zu)", a, c ? cfgs[c - 1].name.c_str() : "base",
                               R.tps.empty() ? 0.0 : s / R.tps.size(), mn, mx, R.tps.size());
                        if (c) {
                            printf(", accepted %d/%d = %.1f%%, tokens/step %.2f, %d/%d runs %s", R.acc, R.dr,
                                   R.dr ? 100.0 * R.acc / R.dr : 0.0, R.steps ? (double) R.runs * (n - 1) / R.steps : 0.0,
                                   R.ident, R.runs, R.ident == R.runs ? "IDENTICAL to plain greedy" : "MISMATCH vs plain greedy");
                            n_ident += R.ident; n_cmp += R.runs;
                        }
                        printf(", ids %s%s  [%s]\n", stable ? "stable" : "VARY", a == 0 && c == 0 ? "" : same0 ? ", == arm 0" : ", != arm 0",
                               arms[a].label.c_str());
                    }
                    printf("AB mtp arm %zu: draft graphs captured %d, dropped for a kernel switch %d (timed reps)  [%s]\n", a,
                           capt[a], drops[a], arms[a].label.c_str());
                }
                printf("MTP identity: %d/%d runs identical to plain greedy\n", n_ident, n_cmp);
                fflush(stdout);
                return 0;
            }
            if (bench_pp > 0 || bench_tg > 0) {
                std::vector<int32_t> pp(bench_pp);
                for (size_t i = 0; i < pp.size(); ++i) pp[i] = synth_id(i, 7919, 13, (int) V);
                if (bench_depths.empty()) bench_depths = {0};
                std::sort(bench_depths.begin(), bench_depths.end());
                bool warm = false;
                for (int D : bench_depths) {
                    if (D + bench_pp + bench_tg > ctx) { printf("AB depth %d: beyond --ctx %d, skipped\n", D, ctx); continue; }
                    const int n0 = mt->n_past();
                    if (D > n0) {
                        auto a = clk::now();
                        if (D <= bench_real_max) {
                            std::vector<int32_t> fill(D - n0);
                            for (size_t i = 0; i < fill.size(); ++i) fill[i] = synth_id(n0 + i, 104729, 7, n_vocab);
                            mt->forward(fill);
                        } else {
                            mt->bench_fill_kv(D);
                        }
                        printf("AB depth %d: fill %s %.1f s\n", D, D <= bench_real_max ? "real" : "synth",
                               std::chrono::duration<double>(clk::now() - a).count());
                    } else if (D < n0) {
                        mt->bench_rewind(D);
                    }
                    if (!warm) {   // untimed passes, one per arm first: lazy allocations (prefill pipeline, staging, graphs), clocks
                        bench_warmup("AB", (int) NA, [&](int i) {
                            apply_arm(arms, (size_t) i % NA);
                            mt->reload_env();
                            if (D == 0) mt->reset(); else mt->bench_rewind(D);
                            if (bench_pp > 0) mt->forward(pp);
                            for (int j = 0; j < std::min(bench_tg, 4); ++j) mt->forward({tg_id(j, n_vocab)});
                        });
                        warm = true;
                    }
                    std::vector<std::vector<double>> tpp(NA), ttg(NA);
                    std::vector<std::set<uint64_t>> hs(NA), hst(NA);   // pp logits / last tg step's logits (depth 0)
                    for (int rep = 0; rep < ab_reps; ++rep) {
                        for (size_t k = 0; k < NA; ++k) {
                            const size_t a = rep % 2 ? NA - 1 - k : k;
                            apply_arm(arms, a);
                            mt->reload_env();
                            if (D == 0) mt->reset(); else mt->bench_rewind(D);
                            auto b = clk::now();
                            if (bench_pp > 0) {
                                const std::vector<float> l = mt->forward(pp);
                                tpp[a].push_back(std::chrono::duration<double>(clk::now() - b).count());
                                hs[a].insert(hash_logits(l.data(), V));
                            }
                            // the last step's logits are kept (a move, as the discarded return was a free) and hashed after
                            // the timed loop: they depend on every step's KV rows / recurrent state, so a decode-only
                            // difference shows there
                            std::vector<float> lt;
                            b = clk::now();
                            for (int i = 0; i < bench_tg; ++i) lt = mt->forward({tg_id(i, n_vocab)});
                            if (bench_tg) ttg[a].push_back(std::chrono::duration<double>(clk::now() - b).count());
                            if (D == 0 && bench_tg) hst[a].insert(hash_logits(lt.data(), V));
                        }
                    }
                    for (size_t a = 0; a < NA; ++a) {
                        double s = 0, mn = 1e30, mx = 0;
                        for (double t : tpp[a]) { const double r = bench_pp / t; s += r; mn = std::min(mn, r); mx = std::max(mx, r); }
                        const double mean = tpp[a].empty() ? 0.0 : s / tpp[a].size();
                        double tg = 0, tmn = 1e30, tmx = 0;
                        for (double t : ttg[a]) { const double r = bench_tg / t; tg += r; tmn = std::min(tmn, r); tmx = std::max(tmx, r); }
                        if (!ttg[a].empty()) tg /= ttg[a].size();
                        const bool stable = hs[a].size() == 1;
                        const bool same0 = stable && hs[0].size() == 1 && *hs[a].begin() == *hs[0].begin();
                        printf("AB depth %d arm %zu:", D, a);
                        if (bench_pp > 0) printf(" pp%d mean %.2f t/s (min %.2f max %.2f, n %zu)", bench_pp, mean, mn, mx, tpp[a].size());
                        if (bench_tg) printf("  tg%d %.2f t/s (min %.2f max %.2f)", bench_tg, tg, tmn, tmx);
                        if (D == 0 && bench_pp > 0)
                            printf("  logits %s%s", stable ? "stable" : "VARY", a == 0 ? "" : same0 ? ", == arm 0" : ", != arm 0");
                        if (D == 0 && bench_tg) {
                            const bool st = hst[a].size() == 1;
                            const bool s0 = st && hst[0].size() == 1 && *hst[a].begin() == *hst[0].begin();
                            printf("  tg logits %s%s", st ? "stable" : "VARY", a == 0 ? "" : s0 ? ", == arm 0" : ", != arm 0");
                        }
                        printf("  [%s]\n", arms[a].label.c_str());
                    }
                    fflush(stdout);
                }
                return 0;
            }
            if (toks.empty() || gen < 1) throw std::runtime_error("--ab: needs --bench pp tg, or a prompt and --gen");
            std::vector<int32_t> ids;   // arm 0's greedy ids (forced for the other arms)
            std::vector<std::vector<float>> ref;   // arm 0's logits per step
            for (size_t a = 0; a < NA; ++a) {
                apply_arm(arms, a);
                mt->reload_env();
                mt->reset();
                auto b = clk::now();
                std::vector<float> l = mt->forward(toks);
                const double tp = std::chrono::duration<double>(clk::now() - b).count();
                int same_hash = 0, top1 = 0, steps = 0;
                double kmax = 0, ksum = 0, dmax = 0;
                std::vector<double> ds;
                for (int st = 0; st <= gen; ++st) {
                    if (a == 0) {
                        ref.push_back(l);
                        if (st < gen) ids.push_back(topk(l, 1)[0]);
                    } else {
                        const std::vector<float> & r = ref[st];
                        same_hash += hash_logits(l.data(), V) == hash_logits(r.data(), V);
                        const double k = kld(r, l);
                        kmax = std::max(kmax, k); ksum += k;
                        top1 += topk(l, 1)[0] == topk(r, 1)[0];
                        for (int id : topk(r, 10)) { const double d = std::fabs((double) l[id] - r[id]); ds.push_back(d); dmax = std::max(dmax, d); }
                        ++steps;
                    }
                    if (st < gen) l = mt->forward({ids[st]});
                }
                if (a == 0) {
                    printf("AB numerics arm 0: prompt %zu tokens %.3f s (%.1f t/s), %d steps; ids:", toks.size(), tp, toks.size() / tp, gen);
                    for (int id : ids) printf(" %d", id);
                    printf("  [%s]\n", arms[a].label.c_str());
                } else {
                    std::sort(ds.begin(), ds.end());
                    printf("AB numerics arm %zu: prompt %.3f s (%.1f t/s); %d/%d steps bit-identical to arm 0; KLD mean %.3e max %.3e; "
                           "top-1 same %d/%d; top-10 |d| median %.4f p99 %.4f max %.4f  [%s]\n",
                           a, tp, toks.size() / tp, same_hash, steps, ksum / steps, kmax, top1, steps, ds[ds.size() / 2],
                           ds[std::min(ds.size() - 1, (size_t) (ds.size() * 0.99))], dmax, arms[a].label.c_str());
                }
                fflush(stdout);
            }
            return 0;
        }
        if (!mtp_k.empty()) {
            if (!mt) throw std::runtime_error("--mtp needs --devices");
            tp_mtp drafter(*mt);
            const int V = mt->vocab();
            if (bench_pp > 0 || bench_tg > 0) {
                if (prompts.empty()) {
                    prompts.emplace_back(bench_pp > 0 ? bench_pp : 1);
                    for (size_t i = 0; i < prompts[0].size(); ++i) prompts[0][i] = synth_id(i, 7919, 13, V);
                }
                prompts.resize(1);
                gen = bench_tg;
            }
            if (gen < 2) throw std::runtime_error("--mtp needs --gen >= 2");
            printf("load %.2f s\n", std::chrono::duration<double>(t1 - t0).count());
            if (bench_pp > 0 || bench_tg > 0) {
                // --bench: untimed warm-up of the main model (prompt + decode steps), the first pass also one short MTP
                // generation at the largest k (draft / verify graph captures); then every run starts from reset() as before
                const int kw = *std::max_element(mtp_k.begin(), mtp_k.end());
                bench_warmup("BENCH", 0, [&](int i) {
                    mt->reset();
                    std::vector<float> l = mt->forward(prompts[0]);
                    for (int j = 0; j < std::min(gen, 4); ++j) l = mt->forward({argmax(l.data(), V)});
                    if (i == 0 && kw >= 1 && kw <= 3) {
                        mt->reset();
                        mtp_stats st;
                        generate_mtp(*mt, drafter, prompts[0], std::min(gen, 16), kw, st);
                    }
                });
                mt->reset();
            }
            int n_ident = 0, n_cmp = 0, n_chk_bad = 0;
            // baselines first: plain greedy, one token per forward
            std::vector<std::vector<int32_t>> bases;
            std::vector<std::vector<uint64_t>> ref_hashes;   // P100E_MTP_CHECK: plain greedy's logit hash per token
            const bool check = getenv("P100E_MTP_CHECK") != nullptr;
            for (size_t pi = 0; pi < prompts.size(); ++pi) {
                const auto & pr = prompts[pi];
                mt->reset();
                auto a = clk::now();
                auto l = mt->forward(pr);
                std::vector<uint64_t> hs;
                auto lhash = [&](const std::vector<float> & v) {
                    uint64_t h = 1469598103934665603ull;
                    const unsigned char * b = (const unsigned char *) v.data();
                    for (size_t i = 0; i < v.size() * 4; ++i) { h ^= b[i]; h *= 1099511628211ull; }
                    return h;
                };
                const double tp = std::chrono::duration<double>(clk::now() - a).count();
                std::vector<int32_t> base;
                a = clk::now();
                for (int i = 0; i < gen; ++i) {
                    const int next = argmax(l.data(), V);
                    // only for the check: the byte-wise hash of 248320 logits costs ~1 ms per token and would count
                    // in the baseline's t/s
                    if (check) hs.push_back(lhash(l));
                    base.push_back(next);
                    if (i + 1 < gen) l = mt->forward({next});
                }
                const double tb = std::chrono::duration<double>(clk::now() - a).count();
                printf("prompt %zu (%zu tokens, %.3f s)\n  base     : %.2f t/s (%d tokens, %.3f s after prompt)\n  ids:", pi,
                       pr.size(), tp, (gen - 1) / tb, gen, tb);
                for (int id : base) printf(" %d", id);
                printf("\n");
                fflush(stdout);
                bases.push_back(base);
                ref_hashes.push_back(hs);
            }
            const std::vector<int32_t> ranking = draft_vocabs.empty() ? std::vector<int32_t>{} : drafter.default_ranking();
            if (draft_vocabs.empty()) draft_vocabs.push_back(-1);   // keep the drafter's default (P100E_DRAFT_VOCAB)
            if (taus.empty()) taus.push_back(0.0f);
            const bool prof = getenv("P100E_MTP_PROF") != nullptr;
            for (int dv : draft_vocabs) {
                if (dv >= 0) drafter.set_draft_vocab(ranking, dv);
                printf("draft vocab %d%s\n", drafter.draft_vocab(), drafter.draft_vocab() == V ? " (full head)" : "");
                for (int K : mtp_k) {
                    for (float tau : taus) {
                        double sum_tps = 0, sum_draft = 0, sum_verify = 0, sum_gen = 0, sum_catch = 0, sum_prompt = 0;
                        int s_acc = 0, s_dr = 0, s_steps = 0, s_nd = 0, s_hist[5] = {0, 0, 0, 0, 0}, s_tok = 0;
                        std::vector<mtp_stats::cal_rec> cal;
                        char cfg[64];
                        if (tau > 0 && kmin == 0) snprintf(cfg, sizeof cfg, "k<=%d tau=%.3f tau1=%.2f", K, tau, tau1 > 0 ? tau1 : tau);
                        else if (tau > 0) snprintf(cfg, sizeof cfg, "k<=%d tau=%.3f", K, tau);
                        else snprintf(cfg, sizeof cfg, "k=%d", K);
                        for (size_t pi = 0; pi < prompts.size(); ++pi) {
                            mt->reset();
                            mtp_stats st;
                            mtp_opts o;
                            o.k = K; o.tau = tau; o.k_min = kmin; o.tau1 = tau1;
                            if (check) o.ref_hash = &ref_hashes[pi];
                            auto ids = generate_mtp(*mt, drafter, prompts[pi], gen, o, st);
                            const bool same = ids == bases[pi];
                            n_cmp++; n_ident += same;
                            const double tps = (gen - 1) / st.t_gen;
                            printf("  p%zu mtp %s : %.2f t/s (%d tokens, %.3f s after prompt)  steps %d  accepted %d/%d = %.1f%%  "
                                   "tokens/step %.2f  %s\n", pi, cfg, tps, gen, st.t_gen, st.steps, st.accepted,
                                   st.drafted, st.drafted ? 100.0 * st.accepted / st.drafted : 0.0,
                                   st.steps ? (double) (gen - 1) / st.steps : 0.0, same ? "IDENTICAL" : "MISMATCH");
                            if (prof)
                                printf("    verify %.2f ms/step, draft %.2f ms/run (%d runs), other %.2f ms/step, drafts/step "
                                       "[0]%d [1]%d [2]%d [3]%d; prompt %.3f s, catch-up %.1f ms\n",
                                       1e3 * st.t_verify / std::max(1, st.steps), 1e3 * st.t_draft / std::max(1, st.n_draft),
                                       st.n_draft, 1e3 * (st.t_gen - st.t_verify - st.t_draft) / std::max(1, st.steps),
                                       st.hist[0], st.hist[1], st.hist[2], st.hist[3], st.t_prompt, 1e3 * st.t_catchup);
                            if (o.ref_hash) {
                                printf("    check: %d verify rows differ from plain greedy's logits\n", st.chk_bad);
                                n_chk_bad += st.chk_bad;
                            }
                            if (!same) {
                                printf("  ids:");
                                for (int id : ids) printf(" %d", id);
                                printf("\n");
                                for (int i = 0; i < gen; ++i) if (ids[i] != bases[pi][i]) { printf("  first diff at %d\n", i); break; }
                            }
                            fflush(stdout);
                            sum_tps += tps; sum_draft += st.t_draft; sum_verify += st.t_verify; sum_gen += st.t_gen;
                            sum_catch += st.t_catchup; sum_prompt += st.t_prompt;
                            s_acc += st.accepted; s_dr += st.drafted; s_steps += st.steps; s_nd += st.n_draft; s_tok += gen - 1;
                            for (int h = 0; h < 5; ++h) s_hist[h] += st.hist[h];
                            cal.insert(cal.end(), st.cal.begin(), st.cal.end());
                        }
                        printf("SUMMARY vocab %d %s : mean %.2f t/s, total %.2f t/s, accepted %d/%d = %.1f%%, tokens/step %.2f, "
                               "draft %.3f ms/run (%.2f runs/step), verify %.2f ms/step, drafts/step [0]%d [1]%d [2]%d [3]%d, "
                               "prompt %.3f s + catch-up %.1f ms per prompt\n",
                               drafter.draft_vocab(), cfg, sum_tps / prompts.size(), s_tok / sum_gen, s_acc, s_dr,
                               s_dr ? 100.0 * s_acc / s_dr : 0.0, s_steps ? (double) s_tok / s_steps : 0.0,
                               1e3 * sum_draft / std::max(1, s_nd), (double) s_nd / std::max(1, s_steps),
                               1e3 * sum_verify / std::max(1, s_steps), s_hist[0], s_hist[1], s_hist[2], s_hist[3],
                               sum_prompt / prompts.size(), 1e3 * sum_catch / prompts.size());
                        if (getenv("P100E_MTP_CALIB")) {
                            // acceptance of a draft (all earlier drafts of its step accepted) by the draft head's top-1
                            // probability p, by the running product of p over the step's drafts, and by position
                            static const float edges[] = {0.0f, 0.2f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 0.95f, 0.99f, 1.01f};
                            const int nbin = (int) (sizeof(edges) / sizeof(edges[0])) - 1;
                            for (int what = 0; what < 2; ++what) {
                                for (int j = 0; j <= K; ++j) {   // j = 0: all positions
                                    printf("CALIB %s %s pos %s:", cfg, what ? "cum" : "p  ", j ? std::to_string(j).c_str() : "*");
                                    for (int b = 0; b < nbin; ++b) {
                                        int n = 0, na = 0;
                                        for (const auto & c : cal) {
                                            const float v = what ? c.cum : c.p;
                                            if ((j == 0 || c.j == j) && v >= edges[b] && v < edges[b + 1]) { n++; na += c.acc; }
                                        }
                                        if (n) printf("  [%.2f,%.2f) %d %.0f%%", edges[b], std::min(1.0f, edges[b + 1]), n, 100.0 * na / n);
                                    }
                                    printf("\n");
                                }
                            }
                        }
                        fflush(stdout);
                    }
                }
            }
            printf("MTP identity: %d/%d runs identical to plain greedy\n", n_ident, n_cmp);
            if (check) printf("MTP check: %d verify rows differ (all runs)\n", n_chk_bad);
            return 0;
        }
        if (!bench_depths.empty()) {
            if (!mt) throw std::runtime_error("--bench-depth needs --devices");
            std::sort(bench_depths.begin(), bench_depths.end());
            std::unique_ptr<tp_mtp> draft_bufs;   // --load-mtp: the draft head's buffers count in the memory numbers
            if (load_mtp) draft_bufs = std::make_unique<tp_mtp>(*mt);
            printf("load %.2f s, ubatch %d\n", std::chrono::duration<double>(t1 - t0).count(), mt->max_batch());
            mt->print_mem("after load");
            std::vector<int32_t> pp(bench_pp > 0 ? bench_pp : 1);
            for (size_t i = 0; i < pp.size(); ++i) pp[i] = synth_id(i, 7919, 13, n_vocab);
            bool first = true;
            if ((int) pp.size() + std::min(bench_tg, 4) <= ctx) {
                bench_warmup("BENCH", 0, [&](int) {   // from an empty context; the depths start from reset() below
                    mt->reset();
                    fwd(pp);
                    if (first) { mt->print_mem("after first prefill"); first = false; }
                    for (int i = 0; i < std::min(bench_tg, 4); ++i) fwd({tg_id(i, n_vocab)});
                });
                mt->reset();
            }
            for (int D : bench_depths) {
                if (D + (int) pp.size() + bench_tg > ctx) { printf("BENCH depth %d: beyond --ctx %d, skipped\n", D, ctx); continue; }
                auto a = clk::now();
                const int n0 = mt->n_past();
                if (D > n0) {
                    if (D <= bench_real_max) {
                        std::vector<int32_t> fill(D - n0);
                        for (size_t i = 0; i < fill.size(); ++i) fill[i] = synth_id(n0 + i, 104729, 7, n_vocab);
                        mt->forward(fill);
                    } else {
                        mt->bench_fill_kv(D);
                    }
                }
                const double tfill = std::chrono::duration<double>(clk::now() - a).count();
                const int d0 = mt->n_past();
                a = clk::now();
                fwd(pp);
                const double tpp = std::chrono::duration<double>(clk::now() - a).count();
                if (first) { mt->print_mem("after first prefill"); first = false; }
                a = clk::now();
                for (int i = 0; i < bench_tg; ++i) fwd({tg_id(i, n_vocab)});
                const double ttg = std::chrono::duration<double>(clk::now() - a).count();
                printf("BENCH depth %d (n_past %d, fill %s %d tokens %.1f s): pp%zu %.2f t/s (%.3f s)  tg%d %.2f t/s (%.3f s)\n",
                       D, d0, D > n0 ? (D <= bench_real_max ? "real" : "synth") : "none", std::max(0, D - n0), tfill, pp.size(),
                       pp.size() / tpp, tpp, bench_tg, bench_tg ? bench_tg / ttg : 0.0, ttg);
                fflush(stdout);
            }
            return 0;
        }
        if (bench_pp > 0 || bench_tg > 0) {
            // synthetic benchmark: prefill bench_pp tokens as one batch, then decode bench_tg tokens one by one
            std::vector<int32_t> pp(bench_pp > 0 ? bench_pp : 1);
            for (size_t i = 0; i < pp.size(); ++i) pp[i] = synth_id(i, 7919, 13, n_vocab);
            if (mt && (int) pp.size() + std::min(bench_tg, 4) <= ctx) {   // (the single-GPU reference model has no reset)
                bench_warmup("BENCH", 0, [&](int) {
                    mt->reset();
                    fwd(pp);
                    for (int i = 0; i < std::min(bench_tg, 4); ++i) fwd({tg_id(i, n_vocab)});
                });
                mt->reset();
            }
            net_stats n0 = mt ? mt->net_counters() : net_stats{};
            auto a = clk::now();
            fwd(pp);
            const double tpp = std::chrono::duration<double>(clk::now() - a).count();
            net_stats n1 = mt ? mt->net_counters() : net_stats{};
            a = clk::now();
            for (int i = 0; i < bench_tg; ++i) fwd({tg_id(i, n_vocab)});
            const double ttg = std::chrono::duration<double>(clk::now() - a).count();
            printf("BENCH pp%zu %.2f t/s (%.3f s)  tg%d %.2f t/s (%.3f s)\n", pp.size(), pp.size() / tpp, tpp, bench_tg,
                   bench_tg ? bench_tg / ttg : 0.0, ttg);
            if (net_on) {
                net_report("prefill", (int) pp.size(), tpp, n0, n1);
                net_report("decode", bench_tg, ttg, n1, mt->net_counters());
            }
            return 0;
        }
        const net_stats np0 = mt ? mt->net_counters() : net_stats{};
        auto logits = fwd(toks);
        const net_stats np1 = mt ? mt->net_counters() : net_stats{};
        const double t_prompt = std::chrono::duration<double>(clk::now() - t1).count();
        if (mt && mt->last_prefill_saturated()) {
            // what a server does: the prompt's state carries clamped f16-wire partials; redo it (the instance uses the
            // f32 wire from now on) and never cache the first attempt
            printf("PFWIRE prompt saturated the f16 wire (%llu partial groups so far): redone on the f32 wire\n",
                   (unsigned long long) mt->prefill_saturations());
            mt->reset();
            logits = fwd(toks);
        }
        auto t2 = clk::now();
        printf("load %.2f s, prompt %zu tokens %.2f s\n", std::chrono::duration<double>(t1 - t0).count(), toks.size(),
               std::chrono::duration<double>(t2 - t1).count());
        if (logits_out) {
            FILE * f = fopen(logits_out, "wb");
            if (!f) throw std::runtime_error(std::string("cannot write ") + logits_out);
            fwrite(logits.data(), 4, logits.size(), f);
            fclose(f);
        }
        if (!logits.empty()) {
            print_logit_hash("prompt", 0, logits);
            printf("top10:");
            for (int id : topk(logits, 10)) printf(" %d(%.3f)", id, logits[id]);
            printf("\n");
            print_top("prompt", 0, logits);
        } else {   // net mode without --net-logits and a one-token prompt: the step returned the greedy id only
            printf("top1 (net greedy id): %d\n", net_last_id);
        }
        if (m1) m1->set_debug(false);
        std::vector<int> out;
        const auto tg0 = clk::now();
        for (int i = 0; i < gen; ++i) {
            // net mode without --net-logits: decode steps return no logits, only the greedy id (net_last_id)
            const int next = !forced.empty() ? forced[i] : logits.empty() ? net_last_id : topk(logits, 1)[0];
            out.push_back(next);
            auto a = clk::now();
            logits = fwd({next});
            if (!logits.empty()) {
                print_logit_hash("gen", i, logits);
                print_top("gen", i, logits);
            }

            printf("gen %d: token %d (%.2f s)\n", i, next, std::chrono::duration<double>(clk::now() - a).count());
        }
        if (gen) {
            printf("generated ids:");
            for (int id : out) printf(" %d", id);
            printf("\n");
        }
        if (net_on) {
            net_report("prefill", (int) toks.size(), t_prompt, np0, np1);
            if (gen) net_report("decode", gen, std::chrono::duration<double>(clk::now() - tg0).count(), np1, mt->net_counters());
        }
    } catch (const std::exception & e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
