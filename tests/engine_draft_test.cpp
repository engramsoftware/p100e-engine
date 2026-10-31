// p100e-engine-draft-test: the server engine's batched draft phase (engine::draft_batch, engine_tp:
// one tp_mtp::run_multi pass for every sequence) against each sequence run ALONE with the per-sequence draft().
// For every case (n sequences, per-sequence k_max with n + sum(k) <= 8 rows): every prompt alone (prefill, then
// draft -> decode -> keep loops, greedy), then all n at once (one decode() per step over every sequence, keep() each,
// ONE draft_batch). Checks, BIT-EXACT: every sequence's generated tokens AND its draft list at every step equal the
// alone run's (each run_multi row must be bitwise the single-sequence run's row, so drafts can not differ).
//
//   p100e-engine-draft-test model.gguf [--devices 0,1] [--kv q8_0] [--gen 48] [--tau 0.5]
#include "engine.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace p100e;

struct trace {
    std::vector<int32_t> toks;
    std::vector<std::vector<int32_t>> drafts;   // per step
    std::vector<std::vector<float>> probs;      // engine draft_probs per step (mtp-multi-accept: bitwise as alone)
};

static std::vector<int> parse_list(const char * s) {
    std::vector<int> v;
    for (const char * p = s; *p;) { v.push_back(atoi(p)); while (*p && *p != ',') ++p; if (*p) ++p; }
    return v;
}

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s model.gguf [--devices 0,1] [--kv q8_0] [--gen N] [--tau t]\n", argv[0]); return 2; }
    engine_tp_params p;
    p.model_path = argv[1];
    p.n_ctx = 4096;
    p.n_ubatch = 512;
    p.mtp_k = 3;
    p.n_seq = 4;
    p.cache = false;
    p.pool_tokens = 4 * 4096;
    int gen = 48;
    for (int i = 2; i < argc; ++i) {
        if (!strcmp(argv[i], "--devices") && i + 1 < argc) p.devices = parse_list(argv[++i]);
        else if (!strcmp(argv[i], "--kv") && i + 1 < argc) p.kv_type = argv[++i];
        else if (!strcmp(argv[i], "--gen") && i + 1 < argc) gen = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--tau") && i + 1 < argc) p.mtp_tau = (float) atof(argv[++i]);
    }
    std::unique_ptr<engine> E = make_engine_tp(p);
    const int V = E->info().n_vocab;
    // prompts: different lengths and contents (the test model gate prompts first, then derived ones)
    std::vector<std::vector<int32_t>> prompts = {
        {760, 6511, 314, 9338, 369},
        {727, 73111, 1393, 1590},
        {7734, 264, 2716, 31708, 883, 279, 17415, 13},
        {623, 220, 16, 19, 24, 17, 11, 30773, 73155},
    };
    for (auto & pr : prompts) for (auto & t : pr) t %= V;

    // one run of `which` together (alone = one sequence); ks = k_max per sequence
    auto run = [&](const std::vector<int> & which, const std::vector<int> & ks) {
        const int n = (int) which.size();
        std::vector<trace> tr(n);
        std::vector<int> seq(n), next(n);
        std::vector<std::vector<int32_t>> dr(n);
        std::vector<std::vector<float>> dp(n);
        for (int i = 0; i < n; ++i) {
            seq[i] = E->seq_open();
            if (seq[i] < 0) throw std::runtime_error("no sequence");
            decode_res o;
            E->prefill(seq[i], prompts[which[i]], false, o);
            next[i] = o.argmax.at(0);
            tr[i].toks.push_back(next[i]);
            dr[i] = E->draft(seq[i], next[i], ks[i]);   // first draft after the prompt: per sequence (carry path)
            dp[i] = E->draft_probs(seq[i]);
        }
        std::vector<char> live(n, 1);
        while (true) {
            std::vector<decode_req> reqs;
            std::vector<int> idx;
            for (int i = 0; i < n; ++i) {
                if (!live[i]) continue;
                tr[i].drafts.push_back(dr[i]);
                tr[i].probs.push_back(dp[i]);
                decode_req r;
                r.seq = seq[i];
                r.tokens.push_back(next[i]);
                r.tokens.insert(r.tokens.end(), dr[i].begin(), dr[i].end());
                r.logits = false;
                reqs.push_back(r);
                idx.push_back(i);
            }
            if (reqs.empty()) break;
            std::vector<decode_res> outs;
            E->decode(reqs, outs);
            std::vector<engine::draft_req> dq;
            std::vector<int> di;
            for (size_t q = 0; q < reqs.size(); ++q) {
                const int i = idx[q];
                const auto & g = outs[q].argmax;
                int keep = 1;
                while (keep < (int) reqs[q].tokens.size() && g[keep - 1] == reqs[q].tokens[keep]) ++keep;
                E->keep(seq[i], keep);
                for (int t = 0; t < keep; ++t) tr[i].toks.push_back(g[t]);
                next[i] = g[keep - 1];
                if ((int) tr[i].toks.size() >= gen) { live[i] = 0; continue; }
                dq.push_back({seq[i], next[i], ks[i]});
                di.push_back(i);
            }
            if (dq.empty()) continue;
            std::vector<std::vector<int32_t>> d;
            if (n == 1) d.push_back(E->draft(dq[0].seq, dq[0].next, dq[0].k_max));   // alone: the per-sequence path
            else E->draft_batch(dq, d);
            for (size_t q = 0; q < dq.size(); ++q) { dr[di[q]] = d[q]; dp[di[q]] = E->draft_probs(dq[q].seq); }
        }
        for (int i = 0; i < n; ++i) { tr[i].toks.resize(gen); E->seq_close(seq[i]); }
        return tr;
    };

    struct tcase { std::vector<int> which, ks; };
    const std::vector<tcase> cases = {
        {{0, 1}, {3, 3}},
        {{0, 1, 2}, {1, 1, 1}},
        {{2, 0, 1}, {2, 1, 1}},
        {{0, 1, 2, 3}, {1, 1, 1, 1}},
        {{3, 2}, {1, 3}},
    };
    int fail = 0, checks = 0;
    for (const tcase & c : cases) {
        const int n = (int) c.which.size();
        const auto together = run(c.which, c.ks);
        printf("case n=%d k=", n);
        for (int k : c.ks) printf("%d ", k);
        printf(":");
        for (int i = 0; i < n; ++i) {
            const auto alone = run({c.which[i]}, {c.ks[i]});
            ++checks;
            const bool tok_ok = alone[0].toks == together[i].toks;
            bool dr_ok = alone[0].drafts == together[i].drafts && alone[0].probs == together[i].probs;
            for (size_t j = 0; j < together[i].drafts.size(); ++j)   // one probability per draft
                if (together[i].probs[j].size() != together[i].drafts[j].size()) dr_ok = false;
            int nd = 0, nacc = 0;
            for (const auto & d : together[i].drafts) nd += (int) d.size();
            nacc = (int) together[i].toks.size() - (int) together[i].drafts.size();
            printf("  s%d %s/%s (%zu steps, %d drafts)", c.which[i], tok_ok ? "tokens ok" : "TOKENS DIFFER",
                   dr_ok ? "drafts ok" : "DRAFTS DIFFER", together[i].drafts.size(), nd);
            (void) nacc;
            if (!tok_ok || !dr_ok) ++fail;
        }
        printf("\n");
    }
    printf("ENGINE DRAFT TEST: %d/%d %s\n", checks - fail, checks, fail ? "FAILED" : "passed");
    return fail ? 1 : 0;
}
