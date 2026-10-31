// n-gram SHADOW telemetry (host only, pure observation). At every decode step of a slot it computes what a gated
// suffix-match n-gram drafter WOULD propose from the slot's own prompt + generated history, and after the request
// resolves it against the tokens that really followed. It never touches an engine call, a draft, the sampler or the
// output. Env P100E_NGRAM_SHADOW (default 1; 0 = off).
//
// Drafter model: longest suffix match over the ladder of window sizes 32,24,16,12,8 (latest occurrence of that window
// in the last 64K tokens, found through a lossy rolling-hash table per window, verified token by token), then
// extended backwards (cap 256) to the match length m. Proposal = the L = 7 tokens that followed the earlier occurrence.
// Gate levels m>=8, m>=12, m>=24 are recorded; the real next tokens decide the prefix-match length k = 0..7 of each
// proposal, resolved at the end of the request (so k can exceed the 1..4 tokens the real step emitted).
#pragma once
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace p100e_ng {

constexpr int L_PROP   = 7;
constexpr int WIN_MAX  = 65536;   // search window (tokens behind the end)
constexpr int EXT_MAX  = 256;     // backward extension cap
constexpr int N_WIN    = 5;
constexpr int WINS[N_WIN] = {32, 24, 16, 12, 8};
constexpr int TBL_BITS = 17;
constexpr int N_LVL    = 3;
constexpr int LVL[N_LVL] = {8, 12, 24};

inline bool enabled() {
    static const bool e = [] { const char * v = getenv("P100E_NGRAM_SHADOW"); return !(v && !strcmp(v, "0")); }();
    return e;
}

struct LvlAgg {
    uint64_t n = 0, prop = 0, match = 0, real = 0;   // gated steps, proposed tokens, matching tokens, real tokens emitted on those steps
    uint64_t k[L_PROP + 1] = {};                      // histogram of the prefix-match length
};

struct Totals {   // process lifetime, for /metrics
    std::atomic<uint64_t> requests{0}, steps{0}, gen{0}, live_steps{0};
    std::atomic<uint64_t> n[N_LVL], prop[N_LVL], match[N_LVL], real[N_LVL];
    Totals() { for (int i = 0; i < N_LVL; ++i) { n[i] = prop[i] = match[i] = real[i] = 0; } }
};
inline Totals & totals() { static Totals t; return t; }

struct Pending { int32_t e, src, m, real; };

class Shadow {
public:
    bool started() const { return started_; }

    void clear() {
        started_ = false;
        h_.clear();
        pend_.clear();
        for (auto & t : tbl_) t.clear();   // keep the capacity: start() refills the same 5 x 512 KB on the next prompt
        for (int & r : roll_at_) r = -1;
        indexed_ = 0;
        n_prompt_ = 0;
        steps_ = 0;
    }

    void start(std::vector<int32_t> && prompt) {
        clear();
        started_ = true;
        h_ = std::move(prompt);
        n_prompt_ = (int) h_.size();
        for (auto & t : tbl_) t.assign((size_t) 1 << TBL_BITS, -1);
        indexed_ = std::max(0, (int) h_.size() - 1 - WIN_MAX);
    }

    // called once per decode step with every token emitted so far (the last one is the step's first row, not yet
    // verified); records the n-gram proposal for what follows it
    void step(const std::vector<int32_t> & emitted) {
        for (size_t i = h_.size() - n_prompt_; i < emitted.size(); ++i) h_.push_back(emitted[i]);
        const int n = (int) h_.size();
        if (n < 9) return;
        const int e = n - 1;
        steps_++;
        for (; indexed_ < e; ++indexed_) insert(indexed_);   // all positions but the current end
        last_ = -1;
        for (int w = 0; w < N_WIN; ++w) {
            const int W = WINS[w];
            if (e + 1 < W) continue;
            const int32_t cand = tbl_[w][hash(e, W) & (((uint32_t) 1 << TBL_BITS) - 1)];
            if (cand < 0 || cand >= e || e - cand > WIN_MAX || cand + 1 < W) continue;
            bool ok = true;
            for (int j = 0; j < W; ++j) if (h_[cand - j] != h_[e - j]) { ok = false; break; }
            if (!ok) continue;
            int m = W;
            while (m < EXT_MAX && cand - m >= 0 && h_[cand - m] == h_[e - m]) ++m;
            pend_.push_back({e, cand, m, -1});
            last_ = (int) pend_.size() - 1;
            break;
        }
    }

    // the real tokens the step just emitted (call after step(), same step)
    void note_real(int n_emitted) { if (last_ >= 0) pend_[last_].real = n_emitted; last_ = -1; }

    int steps() const { return steps_; }

    // live drafting (P100E_NGRAM_DRAFT): after the step's tokens were emitted (`emitted` ends with the token the next step starts
    // with), look up the longest suffix match ending at that token; when its length is >= min_m, write up to max_n tokens that
    // followed the earlier occurrence and return their count (0: no gated proposal). Does not touch the telemetry (pend_, steps_).
    int propose(const std::vector<int32_t> & emitted, int32_t * out, int max_n, int min_m) {
        for (size_t i = h_.size() - n_prompt_; i < emitted.size(); ++i) h_.push_back(emitted[i]);
        const int n = (int) h_.size();
        if (n < 9) return 0;
        const int e = n - 1;
        for (; indexed_ < e; ++indexed_) insert(indexed_);
        for (int w = 0; w < N_WIN; ++w) {
            const int W = WINS[w];
            if (e + 1 < W) continue;
            const int32_t cand = tbl_[w][hash(e, W) & (((uint32_t) 1 << TBL_BITS) - 1)];
            if (cand < 0 || cand >= e || e - cand > WIN_MAX || cand + 1 < W) continue;
            bool ok = true;
            for (int j = 0; j < W; ++j) if (h_[cand - j] != h_[e - j]) { ok = false; break; }
            if (!ok) continue;
            int m = W;
            while (m < EXT_MAX && cand - m >= 0 && h_[cand - m] == h_[e - m]) ++m;
            if (m < min_m) return 0;
            const int k = std::min(max_n, e - cand);   // tokens known after the earlier occurrence
            for (int j = 0; j < k; ++j) out[j] = h_[cand + 1 + j];
            return std::max(k, 0);
        }
        return 0;
    }

    // resolve against the final history (prompt + `emitted`); returns per-level aggregates and adds them to totals()
    void resolve(const std::vector<int32_t> & emitted, LvlAgg out[N_LVL]) {
        for (size_t i = h_.size() - n_prompt_; i < emitted.size(); ++i) h_.push_back(emitted[i]);
        const int n = (int) h_.size();
        for (const Pending & p : pend_) {
            const int avail = n - 1 - p.e;   // real tokens known after the proposal point
            const int L = std::min(std::min(L_PROP, p.e - p.src), avail);
            if (L <= 0) continue;
            int k = 0;
            while (k < L && h_[p.src + 1 + k] == h_[p.e + 1 + k]) ++k;
            for (int l = 0; l < N_LVL; ++l) {
                if (p.m < LVL[l]) continue;
                out[l].n++; out[l].prop += L; out[l].match += k; out[l].k[k]++;
                if (p.real >= 0) out[l].real += p.real;
            }
        }
        auto & T = totals();
        T.requests++; T.steps += steps_; T.gen += emitted.size();
        for (int l = 0; l < N_LVL; ++l) {
            T.n[l] += out[l].n; T.prop[l] += out[l].prop; T.match[l] += out[l].match; T.real[l] += out[l].real;
        }
    }

private:
    // window hash = a mix of the polynomial sum h[end-j] * B^j (j < W, mod 2^64), so insert() rolls it forward one
    // position in O(1) per window instead of rehashing W tokens (a 50K prompt indexed in its first step: ~250K
    // multiplies instead of ~4.6M dependent ones on the task loop). Telemetry only; matches are verified token by token.
    static constexpr uint64_t HB = 0x100000001b3ull;
    static uint64_t tok(int32_t t) { return (uint64_t) (uint32_t) t + 1; }
    static uint64_t bpow(int n) { uint64_t p = 1; for (int i = 0; i < n; ++i) p *= HB; return p; }
    uint64_t poly(int end, int W) const {
        uint64_t x = 0;
        for (int j = W - 1; j >= 0; --j) x = x * HB + tok(h_[end - j]);
        return x;
    }
    static uint32_t mix(uint64_t x, int W) {
        x ^= (uint64_t) W * 0x9e3779b97f4a7c15ull;
        x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ull; x ^= x >> 27; x *= 0x94d049bb133111ebull; x ^= x >> 31;
        return (uint32_t) x;
    }
    uint32_t hash(int end, int W) const { return mix(poly(end, W), W); }
    void insert(int pos) {
        for (int w = 0; w < N_WIN; ++w) {
            const int W = WINS[w];
            if (pos + 1 < W) continue;
            uint64_t x;
            if (roll_at_[w] == pos - 1) {   // drop h[pos - W] (weight B^(W-1)), shift, add h[pos]
                if (!bw_[w]) bw_[w] = bpow(W - 1);
                x = (roll_[w] - tok(h_[pos - W]) * bw_[w]) * HB + tok(h_[pos]);
            } else {
                x = poly(pos, W);
            }
            roll_[w] = x;
            roll_at_[w] = pos;
            tbl_[w][mix(x, W) & (((uint32_t) 1 << TBL_BITS) - 1)] = pos;
        }
    }

    bool started_ = false;
    std::vector<int32_t> h_;
    std::vector<Pending> pend_;
    std::vector<int32_t> tbl_[N_WIN];
    uint64_t roll_[N_WIN] = {};      // poly() of the window ending at roll_at_[w] (insert's rolling state)
    int roll_at_[N_WIN] = {-1, -1, -1, -1, -1};
    uint64_t bw_[N_WIN] = {};        // B^(W-1) per window (0: not computed yet)
    int indexed_ = 0, n_prompt_ = 0, steps_ = 0, last_ = -1;
};

}  // namespace p100e_ng
