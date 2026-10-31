// engine_fn: server/engine.h on the Flash-Next model (src/fn_model.h): ONE sequence, no prompt cache; images through
// prefill_mm (fn_model::forward_mm).
// The step contract follows fn_model::generate_mtp exactly (so a greedy request through the server emits the same ids
// as p100e-fn-run --drafts k):
//   prefill():  forward_n in max_t chunks + mtp_catchup per chunk (the draft block sees every prompt row)
//   decode():   forward_n([sampled, drafts...], T) -> T rows of logits (the trunk verifies every row)
//   keep(q):    rollback(T, q) (delta-net / PLE snapshots; attention KV is positional)
//   draft():    after a prefill: mtp_run({next}, N, READY); after a keep: catch-up over the kept drafts + next with the
//               verify's hn rows (MAIN), then chained SELF drafts; always k drafts (no draft probabilities)
// seq_keep(n): n = 0 resets the model (the server starts every prompt from an empty state here); n = every fed token
// is a no-op; any other cut is refused (a recurrent state cannot be cut in the middle).
#include "engine.h"
#include "fn_model.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>

namespace p100e {

namespace {

using clk = std::chrono::steady_clock;
static double secs(clk::time_point a) { return std::chrono::duration<double>(clk::now() - a).count(); }

static int32_t argmax_row(const float * l, int n) {
    int best = 0;
    for (int i = 1; i < n; ++i) if (l[i] > l[best]) best = i;
    return best;
}

class engine_fn final : public engine {
public:
    explicit engine_fn(const engine_fn_params & p) : p_(p) {
        if (p.mtp_k < 0 || p.mtp_k > fn_model::MTP_ROWS - 1) throw std::runtime_error("fn: mtp k must be 0..3");
        {
            gguf_file g(p.model_path);
            for (const auto & t : g.tensors) {
                info_.n_params += (uint64_t) t.nelements();
                info_.model_bytes += t.nbytes;
            }
        }
        fn_model::opts o;
        o.n_ctx = p.n_ctx;
        o.max_t = std::max(1, p.max_t);
        o.hot_per_rank = p.hot_per_rank;
        o.mtp = p.mtp_k > 0;
        o.verbose = true;   // load progress + free VRAM per layer in the server log
        o.rank_slots = p.rank_slots;
        if (!p.pipe_nodes.empty()) {   // node 1 drives; node 0 is a p100e-fn-run worker
            o.net_nodes = p.pipe_nodes; o.net_node = 1; o.pipe_layers = p.pipe_layers != 0 ? p.pipe_layers : -1;
            o.mtp = true;   // node 0 always keeps the rollback snapshots (agreed at connect)
        }
        m_ = std::make_unique<fn_model>(p.model_path, p.devices, o);
        if (p.mtp_k > 0 && !m_->has_mtp()) throw std::runtime_error("fn: the model has no MTP (nextn) block; run with MTP off");
        info_.n_vocab = m_->vocab();
        info_.n_embd = m_->n_embd();
        info_.n_ctx = p.n_ctx;
        info_.n_ubatch = o.max_t * 8;   // the server hands prompt pieces of this size: the PLE prefetch runs 8 chunks ahead
        chunk_ = o.max_t;
        info_.n_seq_max = 1;
        info_.n_draft_max = p.mtp_k;
        info_.n_rows_max = fn_model::MTP_ROWS;
        info_.kv_type = getenv("P100E_FN_KV") ? getenv("P100E_FN_KV") : (p.n_ctx > 32768 ? "q8" : "f16");
        info_.desc = "flash-next fn_model " + std::to_string(m_->n_layer()) + " layers, " + std::to_string(p.devices.size()) +
                     " GPU(s), ctx " + std::to_string(p.n_ctx) + ", chunk " + std::to_string(o.max_t) + ", hot/rank " +
                     std::to_string(p.hot_per_rank) + ", mtp " + std::to_string(p.mtp_k) + ", kv " + info_.kv_type;
    }

    const engine_info & info() const override { return info_; }

    int seq_open() override {
        if (open_) return -1;
        open_ = true;
        reset_();
        return 0;
    }
    void seq_close(int seq) override { check(seq); open_ = false; }
    const std::vector<int32_t> & seq_tokens(int seq) const override { check(seq); return toks_; }
    bool seq_keep(int seq, int n) override {
        check(seq);
        if (n == (int) toks_.size()) return true;
        if (n != 0) return false;
        reset_();
        return true;
    }

    void prefill(int seq, const std::vector<int32_t> & tokens, bool want_logits, decode_res & out) override {
        check(seq);
        if (tokens.empty()) throw std::runtime_error("prefill: no input");
        if (last_ == op::decode || last_ == op::keep || last_ == op::draft)
            throw std::runtime_error("prefill after generation is not supported (seq_keep(0) first)");
        if ((int) toks_.size() + (int) tokens.size() > p_.n_ctx) throw std::runtime_error("context full");
        const auto t0 = clk::now();
        if (m_->n_past() == 0 && m_->has_mtp()) m_->mtp_h_zero();
        std::vector<float> L;
        const int chunk = chunk_;
        m_->ple_prefetch(tokens);
        for (size_t c0 = 0; c0 < tokens.size(); c0 += (size_t) chunk) {
            const std::vector<int32_t> ch(tokens.begin() + c0, tokens.begin() + std::min(tokens.size(), c0 + (size_t) chunk));
            const int p0 = m_->n_past();
            L = m_->forward_n(ch, 1);
            if (m_->has_mtp()) m_->mtp_catchup(ch, p0);
        }
        toks_.insert(toks_.end(), tokens.begin(), tokens.end());
        out.argmax.clear();
        out.logits.clear();
        if (want_logits) out.logits = std::move(L);
        else out.argmax.push_back(argmax_row(L.data(), info_.n_vocab));
        last_ = op::prefill;
        step_.clear();
        st_.t_prompt_s += secs(t0);
        st_.n_prompt += (int64_t) tokens.size();
    }

    // text + image pieces in order (fn_model::forward_mm: M-RoPE image positions, image token for the PLE n-gram); the
    // draft block catches up after every chunk with the image rows still staged
    void prefill_mm(int seq, const std::vector<prompt_piece> & pieces, bool want_logits, decode_res & out) override {
        check(seq);
        if (pieces.empty()) throw std::runtime_error("prefill_mm: no input");
        if (last_ == op::decode || last_ == op::keep || last_ == op::draft)
            throw std::runtime_error("prefill after generation is not supported (seq_keep(0) first)");
        int rows = 0;
        std::vector<fn_model::mm_in> in;
        for (const auto & pc : pieces) {
            fn_model::mm_in q;
            if (pc.embd) {
                if (pc.nx <= 0 || pc.ny <= 0 || pc.n_embd != info_.n_embd)
                    throw std::runtime_error("prefill_mm: bad image piece (grid " + std::to_string(pc.nx) + "x" + std::to_string(pc.ny) +
                                             ", width " + std::to_string(pc.n_embd) + ", model " + std::to_string(info_.n_embd) + ")");
                q.embd = pc.embd.get(); q.nx = pc.nx; q.ny = pc.ny; q.n_embd = pc.n_embd;
            } else {
                if (pc.tokens.empty()) throw std::runtime_error("prefill_mm: empty text piece");
                q.tokens = pc.tokens.data(); q.n_tok = (int) pc.tokens.size();
            }
            rows += pc.n_rows();
            in.push_back(q);
        }
        if ((int) toks_.size() + rows > p_.n_ctx) throw std::runtime_error("context full");
        const auto t0 = clk::now();
        if (m_->n_past() == 0 && m_->has_mtp()) m_->mtp_h_zero();
        std::vector<float> L = m_->has_mtp()
            ? m_->forward_mm(in, [&](int p0, const std::vector<int32_t> & ch) { m_->mtp_catchup(ch, p0); })
            : m_->forward_mm(in);
        for (const auto & pc : pieces) {
            if (pc.embd) toks_.insert(toks_.end(), (size_t) pc.n_rows(), -1);
            else toks_.insert(toks_.end(), pc.tokens.begin(), pc.tokens.end());
        }
        out.argmax.clear();
        out.logits.clear();
        if (want_logits) out.logits = std::move(L);
        else out.argmax.push_back(argmax_row(L.data(), info_.n_vocab));
        last_ = op::prefill;
        step_.clear();
        st_.t_prompt_s += secs(t0);
        st_.n_prompt += rows;
    }

    void decode(const std::vector<decode_req> & reqs, std::vector<decode_res> & out) override {
        out.assign(reqs.size(), decode_res{});
        if (reqs.empty()) return;
        if (reqs.size() != 1) throw std::runtime_error("fn: one sequence only");
        const decode_req & r = reqs[0];
        check(r.seq);
        const int T = (int) r.tokens.size();
        if (T < 1 || T > 1 + p_.mtp_k || T > fn_model::MTP_ROWS) throw std::runtime_error("decode: rows must be 1..1+k");
        if (last_ == op::none) throw std::runtime_error("decode before prefill");
        if ((int) toks_.size() + T > p_.n_ctx) throw std::runtime_error("context full");
        const auto t0 = clk::now();
        std::vector<float> L = m_->forward_n(r.tokens, T);   // T rows of logits
        decode_res & o = out[0];
        if (r.logits) {
            o.logits = std::move(L);
        } else {
            for (int t = 0; t < T; ++t) o.argmax.push_back(argmax_row(L.data() + (size_t) t * info_.n_vocab, info_.n_vocab));
        }
        toks_.insert(toks_.end(), r.tokens.begin(), r.tokens.end());
        step_ = r.tokens;
        last_ = op::decode;
        st_.t_decode_s += secs(t0);
        st_.n_decode_steps++;
        st_.n_decode_rows += T;
    }

    void keep(int seq, int n_keep) override {
        check(seq);
        if (last_ != op::decode) throw std::runtime_error("keep() must follow decode()");
        const int T = (int) step_.size();
        if (n_keep < 1 || n_keep > T) throw std::runtime_error("keep: n_keep out of range");
        m_->rollback(T, n_keep);   // n_keep == T: nothing to restore
        toks_.resize(toks_.size() - (size_t) (T - n_keep));
        step_.resize((size_t) n_keep);
        last_ = op::keep;
    }

    std::vector<int32_t> draft(int seq, int32_t next, int k_max) override {
        check(seq);
        std::vector<int32_t> drafts;
        if (!m_->has_mtp() || p_.mtp_k == 0) return drafts;
        if (last_ != op::prefill && last_ != op::keep) throw std::runtime_error("draft() must follow prefill() or keep()");
        const auto t0 = clk::now();
        const int N = (int) toks_.size();   // position of `next`
        int32_t d0;
        if (last_ == op::prefill) {
            d0 = m_->mtp_run({next}, N, fn_model::MTP_H_READY);
        } else {
            std::vector<int32_t> acc(step_.begin() + 1, step_.end());   // the kept drafts
            acc.push_back(next);
            d0 = m_->mtp_run(acc, N - (int) step_.size() + 1, fn_model::MTP_H_MAIN);   // h = the verify's hn rows
        }
        st_.n_draft_runs++;
        const int k = std::min(k_max, p_.mtp_k);
        if (k > 0) {
            drafts.push_back(d0);
            for (int j = 1; j < k; ++j) {
                drafts.push_back(m_->mtp_run({drafts.back()}, N + j, fn_model::MTP_H_SELF));
                st_.n_draft_runs++;
            }
        }
        st_.n_drafted += (int64_t) drafts.size();
        st_.t_draft_s += secs(t0);
        last_ = op::draft;
        return drafts;
    }

    engine_stats stats() const override { return st_; }
    void print_mem(const char * tag) const override {
        for (int d : p_.devices) {
            size_t fr = 0, tot = 0;
            if (cudaSetDevice(d) == cudaSuccess && cudaMemGetInfo(&fr, &tot) == cudaSuccess)
                fprintf(stderr, "engine_fn %s: GPU %d free %zu MiB of %zu MiB\n", tag, d, fr >> 20, tot >> 20);
        }
    }

private:
    enum class op { none, prefill, decode, keep, draft };
    void check(int seq) const { if (seq != 0 || !open_) throw std::runtime_error("fn: bad sequence"); }
    void reset_() {
        m_->reset();
        if (m_->has_mtp()) m_->mtp_h_zero();
        toks_.clear();
        step_.clear();
        last_ = op::none;
    }

    engine_fn_params p_;
    int chunk_ = 512;   // rows per forward (fn_model max_t)
    engine_info info_;
    engine_stats st_;
    std::unique_ptr<fn_model> m_;
    bool open_ = false;
    std::vector<int32_t> toks_, step_;
    op last_ = op::none;
};

} // namespace

std::unique_ptr<engine> make_engine_fn(const engine_fn_params & p) { return std::make_unique<engine_fn>(p); }

} // namespace p100e
