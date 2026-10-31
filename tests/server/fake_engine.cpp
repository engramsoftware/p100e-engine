// p100e-server-fake: a deterministic CPU stand-in for server/engine.h (no GPU, no weights), linked into
// p100e-server-fake in place of server/engine_tp.cpp, so the slot loop, the task queue and the MTP accept / rollback
// bookkeeping of server/server_context.cpp can be tested on any machine (tests/server/fake_engine_test.py).
//
// The fake model: a fixed cycle of ordinary tokens (the tokenization of FAKE_TEXT). The greedy next token after token t
// is the token after t in the cycle (the first cycle token after any token outside it); its logits row is 0 except
// that token (+30, so sampling picks it too). It never produces end-of-generation, so a request runs until n_predict,
// a stop string or the context limit. MTP drafts are the true continuation, except every P100E_FAKE_DRAFT_MISS-th
// drafted token (env, 0 = never), which is replaced by another cycle token (the chain continues from the wrong token),
// so such a step accepts only part of its drafts.
// Env P100E_FAKE_DELAY_MS: sleep per prefill / decode call (lets a test queue requests behind a running one).
// Env P100E_FAKE_PREFILL_DELAY_MS: sleep per prefill call instead (a prompt chunk costing more than a decode step).
// Env P100E_FAKE_FATAL_AT=N: the N-th decode() call throws engine_fatal_error (what engine_tp throws after a P2P
// allreduce timeout), so the server's fatal-exit path can be tested.
// Env P100E_FAKE_SEQS=N (default 1): N sequences at once (min with -np), decode() takes them all in one call with at
// most 8 rows (as engine_tp); with --p100e-cache 1 (default) closed sequences stay as cache entries and
// seq_open_cached reuses the longest common prefix (the fake can cut anywhere; at most |prompt| - 1 rows).
// The fake checks every call the way the real engine needs it (open sequence, context capacity incl. the next step's
// draft rows, keep range, decode -> keep -> draft order, token range) and throws on a violation, so a slot-loop bug
// surfaces as an error response instead of passing silently.
#include "engine.h"

#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace p100e {
namespace {

const char * FAKE_TEXT = " alpha beta gamma delta epsilon zeta eta theta iota kappa lambda omicron";

int env_int(const char * name, int def) {
    const char * v = std::getenv(name);
    return v && *v ? std::atoi(v) : def;
}

class fake_engine final : public engine {
public:
    fake_engine(const engine_tp_params & p, int n_vocab, std::vector<int32_t> cycle) : cycle_(std::move(cycle)) {
        inf_.n_vocab     = n_vocab;
        inf_.n_ctx       = p.n_ctx;
        inf_.n_ubatch    = p.n_ubatch;
        inf_.n_seq_max   = std::max(1, std::min(env_int("P100E_FAKE_SEQS", 1), p.n_seq));
        inf_.n_rows_max  = inf_.n_seq_max > 1 ? 8 : (1 << 30);
        cache_ = p.cache && inf_.n_seq_max > 1;
        inf_.n_draft_max = std::max(0, p.mtp_k);
        inf_.kv_type     = p.kv_type;
        delay_ms_   = env_int("P100E_FAKE_DELAY_MS", 0);
        prefill_delay_ms_ = env_int("P100E_FAKE_PREFILL_DELAY_MS", -1);   // per prefill call (default: the same)
        draft_miss_ = env_int("P100E_FAKE_DRAFT_MISS", 0);
        fatal_at_   = env_int("P100E_FAKE_FATAL_AT", 0);
        inf_.desc = "FAKE engine (CPU, token cycle of " + std::to_string(cycle_.size()) + ", MTP k <= " +
                    std::to_string(inf_.n_draft_max) + ", delay " + std::to_string(delay_ms_) + " ms, draft miss every " +
                    std::to_string(draft_miss_) + ") - tests only";
        for (size_t i = 0; i < cycle_.size(); ++i) {
            pos_[cycle_[i]] = (int) i;
        }
    }

    const engine_info & info() const override { return inf_; }

    int seq_open() override {
        int n = 0;
        for (const auto & kv : S_) n += kv.second.open;
        if (n >= inf_.n_seq_max) {
            return -1;
        }
        int id = 0;
        while (S_.count(id)) ++id;   // idle cache entries keep their ids
        S_[id].open = true;
        return id;
    }

    int seq_open_cached(const std::vector<uint64_t> & keys, int & n_cached) override {
        n_cached = 0;
        int best = -1, bc = 0;
        if (cache_) {
            for (auto & kv : S_) {
                if (kv.second.open) continue;
                const auto & t = kv.second.toks;
                int l = 0;
                while (l < (int) t.size() && l < (int) keys.size() && (uint64_t) (uint32_t) t[l] == keys[l]) ++l;
                l = std::min(l, (int) keys.size() - 1);
                if (l > bc) { bc = l; best = kv.first; }
            }
        }
        if (best < 0) {
            const int id = seq_open();
            if (id >= 0) seq_keep(id, 0);
            return id;
        }
        int n = 0;
        for (const auto & kv : S_) n += kv.second.open;
        if (n >= inf_.n_seq_max) {
            return -1;
        }
        if (bc < (int) S_[best].toks.size()) {   // a shared prefix: copied (the entry stays), as engine_tp
            int id = 0;
            while (S_.count(id)) ++id;
            S_[id].toks.assign(S_[best].toks.begin(), S_[best].toks.begin() + bc);
            best = id;
        }
        S_[best].open = true;
        S_[best].toks.resize(bc);
        S_[best].pending = 0;
        n_cached = bc;
        hits_++;
        return best;
    }

    cache_stats cstats() const override {
        cache_stats c;
        c.hits = hits_;
        for (const auto & kv : S_) c.entries += !kv.second.open;
        return c;
    }

    void seq_close(int seq) override {
        check_seq(seq);
        S_[seq].open = false;
        S_[seq].pending = 0;
        if (!cache_) {
            S_.erase(seq);
        }
    }

    const std::vector<int32_t> & seq_tokens(int seq) const override {
        check_seq(seq);
        return S_.at(seq).toks;
    }

    bool seq_keep(int seq, int n) override {
        check_seq(seq);
        cur_ = seq;
        check_no_pending("seq_keep");
        if (n < 0 || n > (int) toks_().size()) {
            return false;
        }
        toks_().resize(n);   // the fake can cut anywhere
        return true;
    }

    void prefill(int seq, const std::vector<int32_t> & tokens, bool want_logits, decode_res & out) override {
        check_seq(seq);
        cur_ = seq;
        check_no_pending("prefill");
        if (tokens.empty()) {
            fail("prefill of 0 tokens");
        }
        check_capacity(tokens.size(), "prefill");
        check_tokens(tokens);
        delay(prefill_delay_ms_ >= 0 ? prefill_delay_ms_ : delay_ms_);
        toks_().insert(toks_().end(), tokens.begin(), tokens.end());
        out = decode_res{};
        fill(out, { next_of(tokens.back()) }, want_logits);
        stats_.n_prompt += (int64_t) tokens.size();
    }

    void decode(const std::vector<decode_req> & reqs, std::vector<decode_res> & out) override {
        out.assign(reqs.size(), decode_res{});
        delay();
        if (fatal_at_ > 0 && ++n_decode_calls_ == fatal_at_) {
            throw engine_fatal_error("fake engine: injected fatal error (P100E_FAKE_FATAL_AT=" + std::to_string(fatal_at_) +
                                     ")");
        }
        size_t rows = 0;
        for (const auto & rq : reqs) rows += rq.tokens.size();
        if ((int) rows > inf_.n_rows_max) {
            fail("decode of " + std::to_string(rows) + " rows (max " + std::to_string(inf_.n_rows_max) + ")");
        }
        if ((int) reqs.size() > inf_.n_seq_max) {
            fail("decode of more sequences than open");
        }
        n_batch_max_ = std::max(n_batch_max_, (int) reqs.size());
        for (size_t r = 0; r < reqs.size(); ++r) {
            const decode_req & rq = reqs[r];
            check_seq(rq.seq);
            cur_ = rq.seq;
            check_no_pending("decode");
            if (rq.tokens.empty()) {
                fail("decode of 0 rows");
            }
            check_capacity(rq.tokens.size(), "decode");
            check_tokens(rq.tokens);
            toks_().insert(toks_().end(), rq.tokens.begin(), rq.tokens.end());
            pending_() = (int) rq.tokens.size();
            std::vector<int32_t> next(rq.tokens.size());
            for (size_t i = 0; i < rq.tokens.size(); ++i) {
                next[i] = next_of(rq.tokens[i]);
            }
            fill(out[r], next, rq.logits);
            stats_.n_decode_steps += 1;
            stats_.n_decode_rows  += (int64_t) rq.tokens.size();
        }
    }

    void keep(int seq, int n_keep) override {
        check_seq(seq);
        cur_ = seq;
        if (pending_() == 0) {
            fail("keep() without a decode()");
        }
        if (n_keep < 1 || n_keep > pending_()) {
            fail("keep(" + std::to_string(n_keep) + ") of a " + std::to_string(pending_()) + "-row step");
        }
        toks_().resize(toks_().size() - (size_t) (pending_() - n_keep));
        pending_() = 0;
    }

    std::vector<int32_t> draft(int seq, int32_t next, int k_max) override {
        check_seq(seq);
        cur_ = seq;
        check_no_pending("draft");
        if (k_max < 0) {
            fail("draft(k_max < 0)");
        }
        std::vector<int32_t> d;
        int32_t t = next;
        const int k = std::min(k_max, inf_.n_draft_max);
        for (int j = 0; j < k; ++j) {
            int32_t dt = next_of(t);
            ++n_drafted_;
            if (draft_miss_ > 0 && n_drafted_ % draft_miss_ == 0) {
                dt = next_of(dt);   // a wrong guess: skips one cycle token
            }
            d.push_back(dt);
            t = dt;
        }
        // the next step decodes [next, drafts]: it must fit
        if ((int) (toks_().size() + 1 + d.size()) > inf_.n_ctx) {
            fail("draft of " + std::to_string(d.size()) + " leaves no room for the next step (" +
                 std::to_string(toks_().size()) + " fed, n_ctx " + std::to_string(inf_.n_ctx) + ")");
        }
        stats_.n_drafted += (int64_t) d.size();
        stats_.n_draft_runs += 1;
        return d;
    }

    engine_stats stats() const override { return stats_; }

    void print_mem(const char * tag) const override {
        fprintf(stderr, "fake engine: %s: no device memory (largest decode batch so far: %d sequences)\n", tag, n_batch_max_);
    }

private:
    engine_info inf_;
    std::vector<int32_t> cycle_;
    std::unordered_map<int32_t, int> pos_;
    struct fseq {
        bool open = false;
        std::vector<int32_t> toks;
        int pending = 0;         // rows of the last decode() not yet resolved by keep()
    };
    std::unordered_map<int, fseq> S_;
    int cur_ = 0;                // the sequence of the call being checked
    bool cache_ = false;
    int64_t hits_ = 0;
    int n_batch_max_ = 0;
    std::vector<int32_t> & toks_() { return S_[cur_].toks; }
    const std::vector<int32_t> & toks_() const { return S_.at(cur_).toks; }
    int & pending_() { return S_[cur_].pending; }
    int pending_() const { return S_.at(cur_).pending; }
    int delay_ms_ = 0;
    int prefill_delay_ms_ = -1;
    int draft_miss_ = 0;
    int fatal_at_ = 0;
    int n_decode_calls_ = 0;
    int64_t n_drafted_ = 0;
    engine_stats stats_;

    [[noreturn]] static void fail(const std::string & msg) {
        throw std::runtime_error("fake engine: " + msg);
    }

    void check_seq(int seq) const {
        const auto it = S_.find(seq);
        if (it == S_.end() || !it->second.open) {
            fail("sequence " + std::to_string(seq) + " is not open");
        }
    }

    void check_no_pending(const char * call) const {
        if (pending_() != 0) {
            fail(std::string(call) + "() before keep() of the last decode()");
        }
    }

    void check_capacity(size_t n, const char * call) const {
        if ((int) (toks_().size() + n) > inf_.n_ctx) {
            fail(std::string(call) + " of " + std::to_string(n) + " rows after " + std::to_string(toks_().size()) +
                 " tokens: context full (n_ctx " + std::to_string(inf_.n_ctx) + ")");
        }
    }

    void check_tokens(const std::vector<int32_t> & tokens) const {
        for (int32_t t : tokens) {
            if (t < 0 || t >= inf_.n_vocab) {
                fail("token " + std::to_string(t) + " out of the vocabulary");
            }
        }
    }

    int32_t next_of(int32_t t) const {
        const auto it = pos_.find(t);
        return it == pos_.end() ? cycle_[0] : cycle_[(it->second + 1) % cycle_.size()];
    }

    void fill(decode_res & out, const std::vector<int32_t> & next, bool logits) const {
        if (logits) {
            out.logits.assign(next.size() * (size_t) inf_.n_vocab, 0.0f);
            for (size_t i = 0; i < next.size(); ++i) {
                out.logits[i * (size_t) inf_.n_vocab + (size_t) next[i]] = 30.0f;
            }
        } else {
            out.argmax = next;
        }
    }

    void delay() const { delay(delay_ms_); }
    void delay(int ms) const {
        if (ms > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(ms));
        }
    }
};

} // namespace

// p100e-server-fake's engine: the tokenizer of the GGUF (vocab-only load, for n_vocab and the token cycle), no weights
std::unique_ptr<engine> make_engine_tp(const engine_tp_params & p) {
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;
    llama_model * m = llama_model_load_from_file(p.model_path.c_str(), mp);
    if (m == nullptr) {
        throw std::runtime_error("fake engine: cannot load the vocabulary of " + p.model_path);
    }
    const llama_vocab * vocab = llama_model_get_vocab(m);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    std::vector<llama_token> cycle(256);
    const int n = llama_tokenize(vocab, FAKE_TEXT, (int32_t) strlen(FAKE_TEXT), cycle.data(), (int32_t) cycle.size(),
                                 false, false);
    std::vector<int32_t> uniq;
    for (int i = 0; i < n; ++i) {
        if (std::find(uniq.begin(), uniq.end(), cycle[i]) == uniq.end() && !llama_vocab_is_eog(vocab, cycle[i])) {
            uniq.push_back(cycle[i]);
        }
    }
    llama_model_free(m);
    if (uniq.size() < 4) {
        throw std::runtime_error("fake engine: the token cycle text gave " + std::to_string(uniq.size()) + " tokens");
    }
    return std::make_unique<fake_engine>(p, n_vocab, uniq);
}

} // namespace p100e

// the fake serves both engine kinds (server_context picks by architecture / --p100e-engine)
std::unique_ptr<p100e::engine> p100e::make_engine_fn(const p100e::engine_fn_params & p) {
    using namespace p100e;
    engine_tp_params t;
    t.model_path = p.model_path;
    t.devices = p.devices;
    t.n_ctx = p.n_ctx;
    t.n_ubatch = p.max_t;
    t.mtp_k = p.mtp_k;
    t.n_seq = 1;
    return make_engine_tp(t);
}
