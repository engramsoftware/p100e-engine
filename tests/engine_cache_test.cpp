// p100e-engine-cache-test : engine_tp's prompt cache on chat-shaped prompts, every output checked
// against the same prompt run with no cache (seq_open + seq_keep(0)), greedy with MTP drafts.
//
//   p100e-engine-cache-test model.gguf [--devices 0] [--ckpt-mb N] [--gen 24] [--turns 8] [--ckpt-tail 4] [--diag]
//
// Prompts are random tokens with a message start token (SPLIT, the engine's split_tokens) only at message starts:
//   cross_chat   three new chats that share a 1500-row "system + tools" prefix, then differ inside the system message
//                (as Claude Code's per-chat environment block does). The second chat finds no checkpoint inside the
//                shared prefix (only message starts get one); with P100E_LCP_CKPT it plans one at the shared prefix's
//                end, so the third chat reuses >= 1500 rows. Expect: chat3 n_cached >= 1500 (LCP on), 0 (off).
//   rerender     one chat for 8 turns; every turn re-renders the previous reply with other tokens (as a client that
//                rewrites tool calls / drops reasoning does) and adds a tool result + a new assistant start. Expect:
//                each turn reuses from the previous reply's start (n_cached >= previous prompt - 4), also with a
//                checkpoint budget of only a few buffers (P100E_CKPT_ROOM recycles the open chat's old checkpoints;
//                without it the budget fills and the reuse point stays stuck at an early checkpoint).
//                A turn whose output differs from the plain cache-off run is checked again against a cache-off run cut at
//                the cached runs' prefill segments (reuse points, ubatch chunks, checkpoint cuts): prefill rounding
//                depends on segment shapes, so only that replay must match bit for bit (turn 8 of the tiny
//                model drifts from the plain run at a 0.11 logit gap; the replay matches all 24 logit rows exactly).
//   --diag       per request: host logits, cached vs cache-off logit differences, the off run's smallest top-2
//                margin, the first differing token, a re-chunked cache-off run and the same-cuts replay.
//   --ckpt-tail N  engine_tp_params::ckpt_tail (0: no tail checkpoints; cached and cache-off then run the same cuts)
// Prints one line per request and PASS / FAIL lines; exit code 0 when all pass. Env switches select the variant.
#include "engine.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace p100e;

static std::vector<int> parse_list(const char * s) {
    std::vector<int> v;
    for (const char * p = s; *p;) { v.push_back(atoi(p)); while (*p && *p != ',') ++p; if (*p) ++p; }
    return v;
}

static const int32_t SPLIT = 7;
static std::mt19937 rng(12345);

static std::vector<int32_t> rnd(int n, int V) {
    std::vector<int32_t> v(n);
    for (auto & t : v) { do t = (int32_t) (rng() % (uint32_t) V); while (t == SPLIT); }
    return v;
}
static void cat(std::vector<int32_t> & a, const std::vector<int32_t> & b) { a.insert(a.end(), b.begin(), b.end()); }

struct result { int n_cached = 0; std::vector<int32_t> toks; std::vector<std::vector<float>> rows; };

static int top1(const std::vector<float> & L) { return (int) (std::max_element(L.begin(), L.end()) - L.begin()); }
// top-1 minus top-2 logit
static float margin(const std::vector<float> & L) {
    const int a = top1(L);
    float b = -INFINITY;
    for (int i = 0; i < (int) L.size(); ++i) if (i != a) b = std::max(b, L[i]);
    return L[a] - b;
}
static float maxdiff(const std::vector<float> & a, const std::vector<float> & b) {
    float d = 0;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) d = std::max(d, std::fabs(a[i] - b[i]));
    return d;
}

// one request: open (cached or not), feed the prompt from the reuse point in ubatch chunks, greedy decode with drafts.
// diag: host logits on every row (argmax on the host), r.rows[i] = the logits that chose r.toks[i].
// bounds (cache off only): sorted rows where a prefill call also ends (other chunk boundaries)
static result run(engine & E, const std::vector<int32_t> & prompt, bool cache, int gen, bool diag = false,
                  const std::vector<int> & bounds = {}) {
    result r;
    std::vector<uint64_t> keys(prompt.begin(), prompt.end());
    for (auto & k : keys) k = (uint64_t) (uint32_t) k;
    int seq;
    if (cache) seq = E.seq_open_cached(keys, r.n_cached);
    else { seq = E.seq_open(); E.seq_keep(seq, 0); }
    if (seq < 0) throw std::runtime_error("no sequence");
    const int U = E.info().n_ubatch;
    decode_res o;
    for (int i = r.n_cached; i < (int) prompt.size();) {
        int c = std::min(U, (int) prompt.size() - i);
        const auto b = std::upper_bound(bounds.begin(), bounds.end(), i);
        if (b != bounds.end()) c = std::min(c, *b - i);
        E.prefill(seq, std::vector<int32_t>(prompt.begin() + i, prompt.begin() + i + c), diag, o);
        i += c;
    }
    const int nv = E.info().n_vocab;
    int32_t next = diag ? top1(o.logits) : o.argmax.at(0);
    if (diag) r.rows.push_back(o.logits);
    r.toks.push_back(next);
    std::vector<int32_t> dr = E.draft(seq, next, 3);
    while ((int) r.toks.size() < gen) {
        decode_req rq;
        rq.seq = seq;
        rq.logits = diag;
        rq.tokens.push_back(next);
        rq.tokens.insert(rq.tokens.end(), dr.begin(), dr.end());
        std::vector<decode_res> out;
        E.decode({rq}, out);
        std::vector<int32_t> g = out[0].argmax;
        std::vector<std::vector<float>> L;
        if (diag) {
            g.clear();
            for (size_t k = 0; k < rq.tokens.size(); ++k) {
                L.emplace_back(out[0].logits.begin() + k * nv, out[0].logits.begin() + (k + 1) * nv);
                g.push_back(top1(L.back()));
            }
        }
        int keep = 1;
        while (keep < (int) rq.tokens.size() && g[keep - 1] == rq.tokens[keep]) ++keep;
        E.keep(seq, keep);
        for (int j = 0; j < keep; ++j) { r.toks.push_back(g[j]); if (diag) r.rows.push_back(L[j]); }
        next = g[keep - 1];
        dr = E.draft(seq, next, 3);
    }
    r.toks.resize(gen);
    if (diag) r.rows.resize(gen);
    E.seq_close(seq);
    return r;
}

static int main_(int argc, char ** argv);
int main(int argc, char ** argv) {
    try {
        return main_(argc, argv);
    } catch (const std::exception & e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}

static int main_(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc < 2) { fprintf(stderr, "usage: %s model.gguf [--devices 0] [--ckpt-mb N] [--gen N]\n", argv[0]); return 2; }
    engine_tp_params p;
    p.model_path = argv[1];
    p.devices = {0};
    p.n_ctx = 8192;
    p.n_ubatch = 512;
    p.mtp_k = 3;
    p.n_seq = 2;
    p.cache = true;
    p.cache_resident = 2;
    p.split_tokens = {SPLIT};
    p.pool_tokens = 4 * 8192;
    p.host_budget_mb = 2048;
    p.ckpt_budget_mb = 1024;
    int gen = 24, turns = 8;
    bool diag = false;   // --diag: logit margins and cached vs cache-off logit differences per request
    for (int i = 2; i < argc; ++i) {
        if (!strcmp(argv[i], "--devices") && i + 1 < argc) p.devices = parse_list(argv[++i]);
        else if (!strcmp(argv[i], "--ckpt-mb") && i + 1 < argc) p.ckpt_budget_mb = (size_t) atoll(argv[++i]);
        else if (!strcmp(argv[i], "--gen") && i + 1 < argc) gen = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--turns") && i + 1 < argc) turns = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--diag")) diag = true;
        else if (!strcmp(argv[i], "--ckpt-tail") && i + 1 < argc) p.ckpt_tail = atoi(argv[++i]);
    }
    std::unique_ptr<engine> E = make_engine_tp(p);
    printf("%s\n", E->info().desc.c_str());
    const int V = E->info().n_vocab;
    const bool lcp_on = !getenv("P100E_LCP_CKPT") || atoi(getenv("P100E_LCP_CKPT")) > 0;
    int fails = 0, n = 0, same = 0;
    // rerender: the rows where the cached runs so far cut their prefill (the reuse point, ubatch chunks from there, the
    // planned checkpoints at the last ckpt_tail message starts). Prefill numerics depend on these segment shapes (a
    // checkpoint cut leaves e.g. a 4-row last segment, the cache-off run one of up to 512 rows), so on random weights
    // a cached run may drift from the plain cache-off run at a close logit pair; a cache-off run cut at the same rows
    // must then match the cached one bit for bit (the cache restored the state exactly).
    std::vector<int> cuts;
    auto check = [&](const char * name, const std::vector<int32_t> & prompt, bool replay = false) {
        const result c = run(*E, prompt, true, gen, diag);
        const result r = run(*E, prompt, false, gen, diag);
        const bool ok = c.toks == r.toks;
        const int U = E->info().n_ubatch, P = (int) prompt.size();
        if (replay) {
            while (!cuts.empty() && cuts.back() > c.n_cached) cuts.pop_back();
            if (cuts.empty() || cuts.back() != c.n_cached) cuts.push_back(c.n_cached);
            std::vector<int> add;
            for (int x = c.n_cached + U; x < P; x += U) add.push_back(x);
            int nt = 0;
            for (int i = P - 1; i > c.n_cached && nt < p.ckpt_tail; --i) if (prompt[i] == SPLIT) { add.push_back(i); nt++; }
            std::sort(add.begin(), add.end());
            add.erase(std::unique(add.begin(), add.end()), add.end());
            cuts.insert(cuts.end(), add.begin(), add.end());
        }
        result q;   // the same-cuts cache-off replay (run when the outputs differ, or always with --diag)
        if (replay && (!ok || diag)) q = run(*E, prompt, false, gen, diag, cuts);
        const bool okq = !q.toks.empty() && q.toks == c.toks;
        n++; same += ok || okq;
        printf("  %-12s prompt %5zu  reused %5d  output %s\n", name, prompt.size(), c.n_cached,
               ok ? "== cache-off" : okq ? "== same-cuts cache-off replay (plain cache-off differs: segment rounding)" : "DIFFERS");
        if (diag) {
            // a cache-off run with other prefill chunk boundaries (first chunk = reuse point mod ubatch, or 100):
            // how far plain summation-order changes move the logits
            const int fc = c.n_cached % U ? c.n_cached % U : 100;
            const result a = run(*E, prompt, false, gen, true, {fc});
            int j = 0, ja = 0;   // first differing token (gen if none)
            while (j < gen && c.toks[j] == r.toks[j]) ++j;
            while (ja < gen && a.toks[ja] == r.toks[ja]) ++ja;
            float dc = 0, da = 0, mmin = INFINITY;
            int jmin = 0;
            for (int i = 0; i < std::min(j + 1, gen); ++i) {   // rows up to and including the first differing one
                dc = std::max(dc, maxdiff(c.rows[i], r.rows[i]));
                if (margin(r.rows[i]) < mmin) { mmin = margin(r.rows[i]); jmin = i; }
            }
            for (int i = 0; i < std::min(ja + 1, gen); ++i) da = std::max(da, maxdiff(a.rows[i], r.rows[i]));
            printf("    diag: |cached-off| prefill row %.3g, max %.3g over rows 0..%d; |rechunked-off| prefill row %.3g, "
                   "max %.3g (first chunk %d, %s); min off margin %.3g at row %d\n",
                   maxdiff(c.rows[0], r.rows[0]), dc, std::min(j, gen - 1), maxdiff(a.rows[0], r.rows[0]), da, fc,
                   ja == gen ? "output == off" : ("output differs at token " + std::to_string(ja)).c_str(), mmin, jmin);
            if (j < gen) {
                const auto & R = r.rows[j];
                const auto & C = c.rows[j];
                printf("    diag: first differing token %d: off %d (logit %.4f), cached %d (logit %.4f); off margin %.4g, "
                       "cached margin %.4g; off logit of cached's token %.4f, cached logit of off's token %.4f\n",
                       j, r.toks[j], R[r.toks[j]], c.toks[j], C[c.toks[j]], margin(R), margin(C), R[c.toks[j]],
                       C[r.toks[j]]);
            }
            if (replay) {
                int jq = 0;
                while (jq < gen && q.toks[jq] == c.toks[jq]) ++jq;
                float dq = 0;
                for (int i = 0; i < std::min(jq + 1, gen); ++i) dq = std::max(dq, maxdiff(q.rows[i], c.rows[i]));
                printf("    diag: cache-off replay with the cached runs' %zu prefill cuts: |replay-cached| prefill row %.3g, "
                       "max %.3g; output %s cached", cuts.size(), maxdiff(q.rows[0], c.rows[0]), dq,
                       jq == gen ? "==" : "differs from");
                if (jq < gen) printf(" at token %d", jq);
                printf(", %s cache-off\n", q.toks == r.toks ? "==" : "differs from");
            }
        }
        if (!ok && !okq) fails++;
        return c;
    };
    // ---- cross_chat ----
    const std::vector<int32_t> shared = rnd(1500, V);
    std::vector<int> reused;
    for (int k = 0; k < 3; ++k) {
        std::vector<int32_t> pr{SPLIT};
        cat(pr, shared);
        cat(pr, rnd(200, V));   // per-chat part of the system message (no message start)
        pr.push_back(SPLIT); cat(pr, rnd(300, V));
        pr.push_back(SPLIT); cat(pr, rnd(3, V));
        reused.push_back(check(("chat" + std::to_string(k + 1)).c_str(), pr).n_cached);
    }
    const bool cross_ok = lcp_on ? reused[2] >= 1500 : true;
    printf("%s cross_chat: chat3 reused %d rows (%s)\n", cross_ok ? "PASS" : "FAIL", reused[2],
           lcp_on ? "expect >= 1500" : "LCP checkpoint off: informational");
    if (!cross_ok) fails++;
    // ---- rerender ----
    std::vector<int32_t> conv{SPLIT};
    cat(conv, rnd(600, V));
    conv.push_back(SPLIT); cat(conv, rnd(400, V));
    conv.push_back(SPLIT); cat(conv, rnd(3, V));
    int stuck = 0, last = 0;
    for (int t = 0; t < turns; ++t) {
        const result c = check(("turn" + std::to_string(t + 1)).c_str(), conv, true);
        // reuse must reach the previous prompt's assistant start (its last 4 rows: SPLIT + 3)
        if (t > 0 && c.n_cached < last - 4) stuck++;
        last = (int) conv.size();
        // next turn: the reply re-rendered (other tokens), a tool result (3 message starts), a new assistant start
        cat(conv, rnd(120, V));
        for (int m = 0; m < 3; ++m) { conv.push_back(SPLIT); cat(conv, rnd(150, V)); }
        conv.push_back(SPLIT); cat(conv, rnd(3, V));
    }
    printf("%s rerender: %d of %d turns reused less than the previous prompt's assistant start\n",
           stuck == 0 ? "PASS" : "FAIL", stuck, turns - 1);
    if (stuck) fails++;
    const auto cs = E->cstats();
    printf("cache: %d entries (%d offloaded), %d checkpoints, hits %lld misses %lld\n", cs.entries, cs.offloaded,
           cs.ckpts, (long long) cs.hits, (long long) cs.misses);
    printf("%s outputs: %d / %d cached runs == cache-off (or its same-cuts replay)\n", same == n ? "PASS" : "FAIL", same, n);
    printf("ENGINE CACHE TEST: %s\n", fails ? "FAILED" : "passed");
    return fails ? 1 : 0;
}
