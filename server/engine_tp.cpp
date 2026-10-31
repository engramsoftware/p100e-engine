// engine_tp: server/engine.h on the multi-sequence tensor-parallel engine (tp_model sequences + tp_mtp) with a prompt
// cache.
//
// Sequences: every engine sequence is a tp_model sequence (paged KV pool shared by all, one delta-net state slot each).
// Up to n_seq are open (the server's slots); closed ones stay as idle prompt-cache entries: resident while slots /
// pages allow (cache_resident extra slots), then offloaded to pinned host RAM (tp_model::seq_offload, host budget), and
// dropped least-recently-used last.
//
// Per step the decode loop mirrors generate_mtp (src/tp_mtp.cu), split where the server's sampler decides:
//   prefill = the ubatch chunks of the main model (forward_n / forward_mm), each followed by the draft head's KV-only
//             catch-up (catchup_kv) with the sequence's carry loaded before and saved after (other sequences' runs
//             overwrite the draft head's h rows in between);
//   decode  = ONE tp_model::forward_batch_ids over [last, drafts] of every generating sequence (<= 8 rows, <= 4 per
//             sequence; each row bitwise the row of that sequence's own step), logits copied for the rows that sample;
//   keep    = tp_model::rollback of the sequence (its snapshot range in the batched step);
//   draft   = the draft head's catch-up run over the kept drafts + `next` (h = the sequence's main-model rows of the step)
//             and the chained drafts (H_SELF), adaptive length (mtp_opts) - per sequence, right after its keep.
// Greedy outputs are those of the sequence alone (rows bitwise as alone; MTP drafts only decide how many rows a step
// verifies); sampled outputs keep the exact distribution (the server's sample-and-match).
//
// Prompt cache (seq_open_cached): keys per row (token id, or vision mm_row_keys for image rows). The reuse point of an
// idle entry is the longest prefix that agrees with the new prompt AND has a known delta-net state: the entry's current
// end (decoded tokens included: a next turn that resends the conversation continues in place), or a checkpoint. A
// prompt gets checkpoints (state + draft carry, pinned host RAM, ckpt_budget) at its last ckpt_tail message starts
// (split_tokens, e.g. <|im_start|>): prefill chunks end there, so the next request that shares the conversation up to
// one of them reuses it. Reuse never starts inside an image (message starts are text). A reused prefix is computed
// with other chunk boundaries than a cold run of the same prompt, so outputs can differ within rounding from a cold run.
//
// Failure handling: a prefill that saturated the f16 prefill wire is redone on the f32 wire over the sequence's whole
// history (its checkpoints are dropped); a P2P allreduce timeout makes every call throw engine_fatal_error.
#include "engine.h"

#include "tp_model.h"
#include "tp_mtp.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <stdexcept>

namespace p100e {
namespace {

using clk = std::chrono::steady_clock;

static double secs(clk::time_point a) { return std::chrono::duration<double>(clk::now() - a).count(); }

static uint32_t kv_type_of(const std::string & s) {
    if (s == "f16") return fast::KV_F16;
    if (s == "q8_0") return fast::KV_Q8_0;
    if (s == "q4_0") return fast::KV_Q4_0;
    throw std::runtime_error("kv type must be f16, q8_0 or q4_0 (got '" + s + "')");
}

static constexpr uint64_t IMG_BIT = 1ull << 63;

static int env_int(const char * name, int dflt) {
    const char * v = getenv(name);
    return v && *v ? atoi(v) : dflt;
}

class engine_tp final : public engine {
public:
    explicit engine_tp(const engine_tp_params & p) : p_(p) {
        // latency switches (by environment):
        //   P100E_LCP_CKPT=N     a checkpoint at the longest prefix the prompt shares with ANY cache entry when that is
        //                        >= N rows past the reuse point (a new chat: the end of the shared system + tools
        //                        prefix; a re-rendered turn: where it diverges). 0 = off. Default 1024.
        //   P100E_CKPT_EVERY=N   also a checkpoint every N rows of a prefill (counted from the reuse point), so a long
        //                        prompt is resumable mid-way even when its message starts are all near its end. 0 = off.
        //                        Default 8192 (the sched's ckpt_every).
        //   P100E_CKPT_ROOM=0/1  a checkpoint that finds no room (host budget full of offloaded entries and no idle
        //                        checkpoint to recycle) drops the least recently used offloaded idle entry instead of
        //                        being skipped silently. Default 1.
        //   P100E_TAKEOVER_TAIL=N  reuse inside an idle entry whose rows past the shared prefix are <= N (the same
        //                        conversation re-rendered its last reply / tool step): cut it in place instead of copying
        //                        the prefix into a new sequence (no near-duplicate entry left behind). 0 = always copy
        //                        (the old behaviour). Default 8192.
        //   P100E_TAKEOVER_ANY=0/1  1 (default, the earlier rule): take over any entry within the tail; 0: only one the
        //                        prompt still agrees with past its last message start (the same conversation), so chats
        //                        that only share a long system / tools prefix are copied and keep their own tails.
        lcp_min_ = env_int("P100E_LCP_CKPT", 1024);
        ck_every_ = env_int("P100E_CKPT_EVERY", 8192);
        ck_room_ = env_int("P100E_CKPT_ROOM", 1) != 0;
        takeover_tail_ = env_int("P100E_TAKEOVER_TAIL", 8192);
        takeover_any_ = env_int("P100E_TAKEOVER_ANY", 1) != 0;
        if (p.mtp_k < 0 || p.mtp_k > 3) throw std::runtime_error("mtp k must be 0..3 (verify batch <= 4)");
        if (p.n_ubatch < 16) throw std::runtime_error("ubatch must be >= 16");
        if (p.n_seq < 1 || p.n_seq > tp_model::MAX_ROWS) throw std::runtime_error("parallel sequences must be 1..8");
        const uint32_t kv = kv_type_of(p.kv_type);
        {
            gguf_file g(p.model_path);
            for (const auto & t : g.tensors) {
                info_.n_params += (uint64_t) t.nelements();
                info_.model_bytes += t.nbytes;
            }
        }
        tp_model::opts mo;
        mo.n_ctx = p.n_ctx;
        mo.max_batch = p.n_ubatch;
        mo.load_mtp = p.mtp_k > 0;
        mo.kv_type = kv;
        mo.max_seqs = p.n_seq + (p.cache ? std::max(0, p.cache_resident) : 0);
        mo.pool_tokens = -1;   // deferred: after the drafter / snapshots / prefill buffers
        mo.default_seq = false;
        mo.split = p.split;    // tensor-parallel weights per GPU (--p100e-split; empty: P100E_SPLIT or equal)
        m_ = std::make_unique<tp_model>(p.model_path, p.devices, mo);
        if (p.mtp_k > 0) {
            if (!m_->has_nextn()) throw std::runtime_error("model has no MTP (nextn) block; run with MTP off");
            d_ = std::make_unique<tp_mtp>(*m_);
            mo_.k = p.mtp_k;
            mo_.tau = p.mtp_tau;
            mo_.k_min = 1;
        }
        m_->enable_rollback(tp_model::MAX_ROWS);   // a batched verify step: up to 7 snapshot rows
        if (!getenv("P100E_HOST_BUDGET_MB")) m_->set_host_budget(p.host_budget_mb << 20);
        m_->prefill_setup();
        if (p.margin_mb >= 0) m_->alloc_pool(p.pool_tokens, p.reserve_mb, (size_t) p.margin_mb);
        else m_->alloc_pool(p.pool_tokens, p.reserve_mb);
        ck_bytes_ = m_->state_bytes() * (size_t) m_->n_ranks() + (size_t) m_->n_embd() * 4;
        info_.n_vocab = m_->vocab();
        info_.n_embd = m_->n_embd();
        info_.n_ctx = p.n_ctx;
        info_.n_ubatch = p.n_ubatch;
        info_.n_seq_max = p.n_seq;
        info_.n_draft_max = p.mtp_k;
        info_.n_rows_max = tp_model::MAX_ROWS;
        info_.kv_type = p.kv_type;
        char buf[512];
        snprintf(buf, sizeof buf,
                 "p100e tp_model on %zu GPU(s), %d sequence(s) (+%d cached resident), n_ctx %d per sequence, KV pool %lld "
                 "tokens, ubatch %d, KV %s, MTP %s, prompt cache %s (checkpoints %.0f MiB each, budget %zu MiB, host %zu MiB; "
                 "lcp ckpt %d, ckpt every %d, ckpt room %d, takeover tail %d)",
                 p.devices.size(), p.n_seq, mo.max_seqs - p.n_seq, p.n_ctx, (long long) m_->pool_tokens(), p.n_ubatch,
                 p.kv_type.c_str(),
                 p.mtp_k > 0 ? (std::string("k <= ") + std::to_string(p.mtp_k) + ", tau " + std::to_string(p.mtp_tau)).c_str()
                             : "off",
                 p.cache ? "on" : "off", ck_bytes_ / 1048576.0, p.ckpt_budget_mb, m_->host_budget() >> 20, lcp_min_,
                 ck_every_, (int) ck_room_, takeover_tail_);
        info_.desc = buf;
        ck_topup();   // the first checkpoints' pinned buffers are ready before the first prompt
    }

    ~engine_tp() override {
        ck_thread_stop();
        for (auto & kv : E_) for (auto & c : kv.second.cps) ck_free(c.buf);
        for (char * b : ck_spare_) cudaFreeHost(b);
    }

    const engine_info & info() const override { return info_; }

    int seq_open() override {
        return fatal([&] { return open_fresh(); });
    }

    int seq_open_cached(const std::vector<uint64_t> & keys, int & n_cached) override {
        return fatal([&] { return open_cached(keys, n_cached); });
    }

    void seq_close(int seq) override {
        fatal([&] { close_(seq); });
    }

    const std::vector<int32_t> & seq_tokens(int seq) const override {
        static const std::vector<int32_t> none;
        const auto it = E_.find(seq);
        return it == E_.end() ? none : it->second.toks;
    }

    bool seq_keep(int seq, int n) override {
        return fatal([&] { return seq_keep_(seq, n); });
    }
    void prefill(int seq, const std::vector<int32_t> & tokens, bool want_logits, decode_res & out) override {
        fatal([&] {
            prompt_piece t;
            t.tokens = tokens;
            prefill_(seq, {t}, want_logits, out);
        });
    }
    void prefill_mm(int seq, const std::vector<prompt_piece> & pieces, bool want_logits, decode_res & out) override {
        fatal([&] { prefill_(seq, pieces, want_logits, out); });
    }
    void decode(const std::vector<decode_req> & reqs, std::vector<decode_res> & out) override {
        fatal([&] { decode_(reqs, out); });
    }
    void keep(int seq, int n_keep) override {
        fatal([&] { keep_(seq, n_keep); });
    }
    std::vector<int32_t> draft(int seq, int32_t next, int k_max) override {
        return fatal([&] { return draft_(seq, next, k_max); });
    }
    void draft_batch(const std::vector<draft_req> & reqs, std::vector<std::vector<int32_t>> & out) override {
        fatal([&] { draft_batch_(reqs, out); });
    }
    std::vector<float> draft_probs(int seq) const override {
        auto it = E_.find(seq);
        return it == E_.end() ? std::vector<float>{} : it->second.dp;
    }

    engine_stats stats() const override { return st_; }
    cache_stats cstats() const override {
        cache_stats c = cs_;
        c.entries = c.offloaded = c.ckpts = 0;
        for (const auto & kv : E_) {
            if (kv.second.open) continue;
            c.entries++;
            if (!m_->seq_resident(kv.first)) c.offloaded++;
        }
        for (const auto & kv : E_) c.ckpts += (int) kv.second.cps.size();
        c.host_bytes = m_->host_used();
        c.pool_tokens = m_->pool_tokens();
        c.free_tokens = m_->free_tokens();
        return c;
    }
    void print_mem(const char * tag) const override { m_->print_mem(tag); }
    bool can_lend() const override { return m_->can_lend(); }
    std::vector<lent_mem> lend_scratch() override {
        std::vector<lent_mem> v;
        for (const auto & s : m_->idle_scratch()) v.push_back({s.dev, s.p, s.bytes, s.lease});
        return v;
    }

private:
    enum class op { none, prefill, decode, keep, draft };
    struct ckpt { int pos = 0; char * buf = nullptr; bool carry = false; uint64_t lru = 0; };
    struct ent {
        bool open = false;
        std::vector<uint64_t> want;      // keys of the prompt being fed (seq_open_cached), for the rows' keys
        std::vector<uint64_t> keys;      // keys of the rows fed (size = state length)
        std::vector<int32_t> toks;       // tokens fed (-1: image rows)
        std::vector<prompt_piece> hist;  // pieces fed, for the f32-wire redo (only kept when the sequence has images)
        bool has_img = false;
        std::vector<int> ck_at;          // planned checkpoint positions of the prompt being fed (ascending)
        std::vector<ckpt> cps;           // ascending pos
        std::vector<int32_t> step;       // rows of the last decode (after keep: the kept rows)
        int row0 = 0;                    // its first row in that batched step
        op last = op::none;
        std::vector<float> dp;           // cumulative draft-head p of the latest drafts (draft_probs)
        bool draft_ok = true;            // the draft head has a row for every position
        bool poisoned = false;           // a prefill / decode / keep failed part-way: the state may not match keys,
                                         //   so close_ drops the entry instead of caching it
        uint64_t t_used = 0;
    };

    template <class F>
    auto fatal(F && f) -> decltype(f()) {
        try {
            return f();
        } catch (const engine_fatal_error &) {
            throw;
        } catch (const std::exception & e) {
            if (m_->ar_failed()) {
                throw engine_fatal_error(std::string("P2P allreduce timed out, the engine is unusable until the process "
                                                     "restarts: ") + e.what());
            }
            throw;
        }
    }

    ent & get(int seq) {
        auto it = E_.find(seq);
        if (it == E_.end() || !it->second.open) throw std::runtime_error("engine_tp: invalid sequence " + std::to_string(seq));
        return it->second;
    }

    int n_open() const {
        int n = 0;
        for (const auto & kv : E_) n += kv.second.open;
        return n;
    }

    // ---- cache entries: room for slots / pages ----
    // the least recently used idle entry (resident only: resident = true), or -1
    int lru_idle(bool resident) const {
        int best = -1;
        uint64_t bt = 0;
        for (const auto & kv : E_) {
            if (kv.second.open) continue;
            if (resident && !m_->seq_resident(kv.first)) continue;
            if (best < 0 || kv.second.t_used < bt) { best = kv.first; bt = kv.second.t_used; }
        }
        return best;
    }

    void drop_entry(int id) {
        auto it = E_.find(id);
        if (it == E_.end()) return;
        for (auto & c : it->second.cps) ck_free(c.buf);
        E_.erase(it);
        if (d_) d_->carry_drop(id);
        m_->seq_free(id);
    }

    // an idle resident entry off the GPU: to host RAM when the budget allows, else dropped
    void evict_one(int id) {
        bool off = false;
        if (p_.cache) {
            try {
                m_->seq_offload(id);
                off = true;
            } catch (const std::exception &) {
                // host budget (or pinned allocation): drop the least recently used offloaded entries and retry once
                while (true) {
                    int o = -1;
                    uint64_t bt = 0;
                    for (const auto & kv : E_)
                        if (!kv.second.open && !m_->seq_resident(kv.first) && (o < 0 || kv.second.t_used < bt)) {
                            o = kv.first; bt = kv.second.t_used;
                        }
                    if (o < 0) break;
                    drop_entry(o);
                    try { m_->seq_offload(id); off = true; break; } catch (const std::exception &) {}
                }
            }
        }
        if (!off) drop_entry(id);
    }

    void need_slot() {
        while (m_->free_slots() == 0) {
            const int v = lru_idle(true);
            if (v < 0) throw std::runtime_error("no free sequence slot");
            evict_one(v);
        }
    }

    // pages so that sequence `seq` can grow to n positions (+ the draft head's lookahead); a full pool throws
    // engine_seq_error (nothing of the sequence changed: a batched caller fails it alone)
    void need_pages(int seq, int n) {
        const int ps = m_->page_size();
        n = std::min(m_->n_ctx(), n + 8);
        const int need = (n + ps - 1) / ps - m_->seq_pages(seq);
        while (need > m_->free_pages()) {
            const int v = lru_idle(true);
            if (v < 0) throw engine_seq_error(seq, "KV pool full (" + std::to_string(m_->free_tokens()) + " free positions)");
            evict_one(v);
        }
    }

    int open_fresh() {
        if (n_open() >= p_.n_seq) return -1;
        need_slot();
        const int id = m_->seq_create();
        ent e;
        e.open = true;
        e.draft_ok = d_ != nullptr;
        e.t_used = ++tick_;
        E_[id] = std::move(e);
        if (d_) d_->carry_drop(id);   // a stale carry of an earlier sequence with this id
        return id;
    }

    // planned checkpoints: the last ckpt_tail message starts in (from, P)
    std::vector<int> plan_ckpts(const std::vector<uint64_t> & keys, int from) const {
        std::vector<int> v;
        if (p_.cache && ck_every_ > 0) {   // P100E_CKPT_EVERY: every N rows from the reuse point (outside images)
            for (int x = from + ck_every_; x < (int) keys.size() - 1; x += ck_every_) {
                int y = x;
                while (y > from && (keys[y] & IMG_BIT) && (keys[y - 1] & IMG_BIT)) --y;
                if (y > from && (v.empty() || v.back() != y)) v.push_back(y);
            }
        }
        std::vector<int> t = plan_tail(keys, from);
        v.insert(v.end(), t.begin(), t.end());
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
        return v;
    }

    std::vector<int> plan_tail(const std::vector<uint64_t> & keys, int from) const {
        std::vector<int> v;
        if (!p_.cache || p_.ckpt_tail <= 0 || p_.split_tokens.empty()) return v;
        for (int i = (int) keys.size() - 1; i > from && (int) v.size() < p_.ckpt_tail; --i) {
            if (keys[i] & IMG_BIT) continue;
            const int32_t t = (int32_t) keys[i];
            if (std::find(p_.split_tokens.begin(), p_.split_tokens.end(), t) != p_.split_tokens.end()) v.push_back(i);
        }
        std::reverse(v.begin(), v.end());
        return v;
    }

    // P100E_LCP_CKPT: a planned checkpoint at the longest prefix the prompt shares with any entry (from = the reuse
    // point), never inside an image (backed off to the image's first row) and never at the prompt's last row
    void add_lcp_ckpt(ent & e, const std::vector<uint64_t> & keys, int from, int lcp) const {
        if (!p_.cache || lcp_min_ <= 0) return;
        int x = std::min(lcp, (int) keys.size() - 1);
        while (x > 0 && (keys[x] & IMG_BIT) && (keys[x - 1] & IMG_BIT)) --x;
        if (x - from < lcp_min_) return;
        auto it = std::lower_bound(e.ck_at.begin(), e.ck_at.end(), x);
        if (it == e.ck_at.end() || *it != x) e.ck_at.insert(it, x);
    }

    // one log line when the prompt shares more with the cache than the reuse point (the rows in between are fed again)
    void log_lcp(int c, int lcp, int P, bool takeover) const {
        fprintf(stderr, "engine_tp: prompt cache: reuse %d, shared prefix %d of %d rows (%s)%s\n", c, lcp, P,
                lcp_min_ > 0 && lcp - c >= lcp_min_ ? "checkpoint planned there" : "no checkpoint planned",
                takeover ? ", entry cut in place" : "");
    }

    int open_cached(const std::vector<uint64_t> & keys, int & n_cached) {
        const auto t_open = clk::now();
        const int id = open_cached_(keys, n_cached);
        if (prof_) fprintf(stderr, "PFPROF open_cached: %d of %zu rows reused, %.1f ms\n", n_cached, keys.size(), 1e3 * secs(t_open));
        return id;
    }
    // P100E_PREFILL_PROF=1: per-request prefill breakdown on stderr (segments, forward / MTP catch-up / checkpoint ms)
    bool prof_ = getenv("P100E_PREFILL_PROF") && atoi(getenv("P100E_PREFILL_PROF")) != 0;

    int open_cached_(const std::vector<uint64_t> & keys, int & n_cached) {
        n_cached = 0;
        if (n_open() >= p_.n_seq) return -1;
        const int P = (int) keys.size();
        int best = -1, best_c = 0, best_ck = -1, best_l = 0;
        int lcp = 0;   // the longest prefix shared with any entry (P100E_LCP_CKPT)
        if (p_.cache) {
            for (auto & kv : E_) {
                const ent & e = kv.second;
                if (e.open) continue;
                const int len = (int) e.keys.size();
                int l = 0;
                const int lim = std::min(len, P);
                while (l < lim && e.keys[l] == keys[l]) ++l;
                lcp = std::max(lcp, l);
                int c = 0, ck = -1;
                if (l == len && len <= P - 1) c = len;   // in place (the entry's end, decoded tokens included)
                for (int i = (int) e.cps.size() - 1; i >= 0; --i)
                    if (e.cps[i].pos <= std::min(l, P - 1) && e.cps[i].pos > c) { c = e.cps[i].pos; ck = i; break; }
                if (c > best_c) { best = kv.first; best_c = c; best_ck = ck; best_l = l; }
            }
        }
        if (best < 0) {
            cs_.misses++;
            const int id = open_fresh();
            if (id >= 0) {
                ent & e = E_.at(id);
                e.want = keys;
                e.ck_at = plan_ckpts(keys, 0);
                add_lcp_ckpt(e, keys, 0, lcp);
                if (lcp >= 1024) log_lcp(0, lcp, (int) keys.size(), false);
            }
            return id;
        }
        int id = best;
        bool forked = false;
        try {
            return reuse_(keys, n_cached, best, best_c, best_ck, best_l, lcp, id, forked);
        } catch (...) {
            // a throw while the entry is reused (pages, restore, truncate, checkpoint load) must not leave it open with
            // no slot owning it (leaked until restart): its state may be half-restored, so it is dropped; a fork's
            // source keeps its rows and goes back to idle
            try {
                if (forked) {
                    auto s = E_.find(best);
                    if (s != E_.end()) s->second.open = false;
                    if (E_.count(id)) drop_entry(id); else m_->seq_free(id);
                } else {
                    drop_entry(best);
                }
            } catch (...) {}
            throw;
        }
    }

    // the cache hit path of open_cached_: entry best, reused at best_c (id / forked are set as soon as they change, for
    // the caller's cleanup on a throw)
    int reuse_(const std::vector<uint64_t> & keys, int & n_cached, int best, int best_c, int best_ck, int best_l, int lcp,
               int & id, bool & forked) {
        const int P = (int) keys.size();
        // reuse at c: the entry's own end (c == len: the same conversation goes on, the entry is taken over), or a
        // checkpoint inside it (c < len: another request shares a prefix - its rows are COPIED into a new sequence so the
        // entry stays for its own conversation; taken over only when there is no room for the copy)
        const int c = best_c;
        const int len = (int) E_.at(best).keys.size();
        E_.at(best).open = true;   // not evicted while room is made below
        // the same conversation re-rendered its tail (a reply / tool step): the entry's rows past the shared prefix are
        // few, so it is cut in place - a copy would leave an abandoned near-duplicate filling the host RAM
        // Same conversation = the prompt still agrees with the entry past its last message start; another chat that only
        // shares the system / tools prefix diverges before it and gets a copy, so two such chats no longer cut each
        // other's tails every turn (P100E_TAKEOVER_ANY=0; the default 1 takes over any entry within the tail, the old rule)
        const bool takeover = c < len && takeover_tail_ > 0 && len - best_l <= takeover_tail_ &&
                              (takeover_any_ || best_l > last_msg_start(E_.at(best).keys));
        if (c < len && !takeover) {
            try {
                const int ps = m_->page_size();
                if (!m_->seq_resident(best)) {   // whole entry back (it stays cached), then copied
                    need_slot();
                    make_pages((len + ps - 1) / ps + 1);
                    m_->seq_restore(best);
                }
                need_slot();
                make_pages((c + ps - 1) / ps + 1);
                id = m_->seq_fork(best, c);
                forked = true;
            } catch (const std::exception & ex) {
                fprintf(stderr, "engine_tp: prompt cache: no room to copy %d rows (%s): the entry is taken over\n", c, ex.what());
            }
        }
        ent & src = E_.at(best);
        if (forked) {
            ent ne;
            ne.keys.assign(src.keys.begin(), src.keys.begin() + c);
            ne.toks.assign(src.toks.begin(), src.toks.begin() + c);
            ne.hist = src.hist;
            ne.has_img = src.has_img;
            cut_hist(ne, c);
            ne.draft_ok = src.draft_ok;
            // the checkpoints of the shared prefix move with the copy: the source is most often the abandoned branch
            // of the same conversation (a re-rendered tool loop or reply), dropped first by LRU, and its checkpoints
            // (the previous reply's start) would go with it, so the next turn found no cache point and re-prefilled
            for (auto it = src.cps.begin(); it != src.cps.end();) {
                if (it->pos <= c) { ne.cps.push_back(*it); it = src.cps.erase(it); } else ++it;
            }
            src.open = false;   // the source stays an idle entry
            src.t_used = ++tick_;
            E_[id] = std::move(ne);
        } else if (!m_->seq_resident(best)) {
            need_slot();
            make_pages((c + m_->page_size() - 1) / m_->page_size() + 1);
            m_->seq_restore_prefix(best, c);   // the state too when c == len
        } else if (c < len) {
            m_->seq_truncate(best, c);
        }
        if (best_ck >= 0 && c != len) {   // the delta-net state (+ draft carry) of position c (moved to id on a fork)
            auto & cv = E_.at(id).cps;
            auto kit = std::find_if(cv.begin(), cv.end(), [c](const ckpt & q) { return q.pos == c; });
            if (kit == cv.end()) throw std::runtime_error("engine_tp: prompt cache: checkpoint lost");
            ckpt & k = *kit;
            k.lru = ++tick_;
            const size_t sb = m_->state_bytes();
            std::vector<const char *> sv;
            for (int r = 0; r < m_->n_ranks(); ++r) sv.push_back(k.buf + (size_t) r * sb);
            ck_join();
            m_->state_load(id, sv);
            if (d_) {
                if (k.carry) d_->carry_from_host(id, (const float *) (k.buf + (size_t) m_->n_ranks() * sb));
                else d_->carry_drop(id);
            }
        }
        ent & e = E_.at(id);
        if (!forked) {   // checkpoints beyond c no longer describe this sequence
            while (!e.cps.empty() && e.cps.back().pos > c) { ck_free(e.cps.back().buf); e.cps.pop_back(); }
            e.keys.resize(c);
            e.toks.resize(c);
            cut_hist(e, c);
        }
        e.open = true;
        e.want = keys;
        e.ck_at = plan_ckpts(keys, c);
        add_lcp_ckpt(e, keys, c, lcp);
        if (lcp - c >= 1024) log_lcp(c, lcp, P, takeover);
        e.step.clear();
        e.last = op::none;
        // the draft head has rows for the reused prefix when the entry's draft rows covered it (prefill catch-ups and
        // decode steps always draft); a missing carry only zeroes h for the next catch-up's first row (drafts only)
        e.draft_ok = d_ != nullptr && e.draft_ok;
        e.t_used = ++tick_;
        cs_.hits++;
        cs_.rows_reused += c;
        n_cached = c;
        return id;
    }

    // the row of the last message start (split_tokens) in an entry's keys, -1 when none (no split tokens: any entry
    // within the takeover tail is taken over, as before)
    int last_msg_start(const std::vector<uint64_t> & keys) const {
        if (p_.split_tokens.empty()) return -1;
        for (int i = (int) keys.size() - 1; i >= 0; --i) {
            if (keys[i] & IMG_BIT) continue;
            const int32_t t = (int32_t) keys[i];
            if (std::find(p_.split_tokens.begin(), p_.split_tokens.end(), t) != p_.split_tokens.end()) return i;
        }
        return -1;
    }

    // free pages >= n: idle resident entries leave the GPU (least recently used first)
    void make_pages(int n) {
        while (m_->free_pages() < n) {
            const int v = lru_idle(true);
            if (v < 0) throw std::runtime_error("KV pool full (" + std::to_string(m_->free_tokens()) + " free positions)");
            evict_one(v);
        }
    }

    static void cut_hist(ent & e, int rows) {
        if (!e.has_img) { e.hist.clear(); return; }
        std::vector<prompt_piece> h;
        int n = 0;
        for (const auto & pc : e.hist) {
            if (n >= rows) break;
            if (pc.embd) {
                if (n + pc.n_rows() > rows) break;   // never happens: cuts are outside images
                h.push_back(pc);
                n += pc.n_rows();
            } else {
                const int take = std::min((int) pc.tokens.size(), rows - n);
                prompt_piece q;
                q.tokens.assign(pc.tokens.begin(), pc.tokens.begin() + take);
                h.push_back(std::move(q));
                n += take;
            }
        }
        e.hist = std::move(h);
        e.has_img = false;
        for (const auto & pc : e.hist) e.has_img = e.has_img || pc.embd != nullptr;
    }

    void close_(int seq) {
        auto it = E_.find(seq);
        if (it == E_.end() || !it->second.open) throw std::runtime_error("engine_tp: invalid sequence " + std::to_string(seq));
        ent & e = it->second;
        e.open = false;
        e.t_used = ++tick_;
        e.want.clear();
        e.ck_at.clear();
        bool keep = p_.cache && !e.keys.empty() && !e.poisoned;
        if (keep) {
            try {
                m_->seq_select(seq);
                if (m_->needs_reset()) keep = false;   // a multimodal forward failed part-way
            } catch (const std::exception &) { keep = false; }
        }
        if (!keep) drop_entry(seq);
    }

    bool seq_keep_(int seq, int n) {
        ent & e = get(seq);
        const int len = (int) e.keys.size();
        if (n == len) return true;
        if (n == 0) {
            m_->seq_select(seq);
            m_->reset();   // KV pages, delta-net state, image list / broken flag of the sequence
            if (d_) d_->carry_drop(seq);
            for (auto & c : e.cps) ck_free(c.buf);
            e.cps.clear();
            e.keys.clear();
            e.toks.clear();
            e.hist.clear();
            e.has_img = false;
            e.step.clear();
            e.last = op::none;
            e.draft_ok = d_ != nullptr;
            e.poisoned = false;   // an empty state again
            return true;
        }
        return false;
    }

    // ---- checkpoints ----
    // ---- spare pinned buffers, allocated off the critical path (P100E_PF_CKPT_SPARE, default 2, 0 = off) ----
    // cudaHostAlloc of a ~300 MB checkpoint buffer costs ~12 ms, which
    // take_ckpt paid at every cut. A helper thread allocates ahead; the budget is charged on the engine thread when the
    // allocation is requested (ck_n_ / host_extra_add), so ck_alloc's checks stay consistent. Buffers only move between
    // threads through ck_ready_ (mutex); ck_spare_ stays engine-thread only.
    static int ck_spare_target() {
        static const int v = getenv("P100E_PF_CKPT_SPARE") ? atoi(getenv("P100E_PF_CKPT_SPARE")) : 2;
        return std::max(0, std::min(v, 8));
    }
    void ck_drain_ready() {
        std::vector<char *> got;
        {
            std::lock_guard<std::mutex> lk(ck_mu_);
            got.swap(ck_ready_);
        }
        for (char * b : got) {
            if (b) { ck_spare_.push_back(b); continue; }
            --ck_n_;   // a failed allocation: undo the charge made at request time
            m_->host_extra_add(-(long long) ck_bytes_);
        }
    }
    void ck_topup() {   // engine thread: request allocations until spare + pending reach the target
        const int target = ck_spare_target();
        if (target <= 0) return;
        ck_drain_ready();
        int pending;
        {
            std::lock_guard<std::mutex> lk(ck_mu_);
            pending = ck_pending_;
        }
        int want = 0;
        while ((int) ck_spare_.size() + pending + want < target &&
               (size_t) (ck_n_ + want + 1) * ck_bytes_ <= (p_.ckpt_budget_mb << 20) &&
               m_->host_used() + (size_t) (want + 1) * ck_bytes_ <= m_->host_budget()) {
            ++ck_n_;   // charged now; the buffer arrives through ck_ready_
            m_->host_extra_add((long long) ck_bytes_);
            ++want;
        }
        if (!want) return;
        {
            std::lock_guard<std::mutex> lk(ck_mu_);
            ck_pending_ += want;
            if (!ck_thread_.joinable()) ck_thread_ = std::thread([this] { ck_thread_main(); });
        }
        ck_cv_.notify_one();
    }
    void ck_thread_main() {
        std::unique_lock<std::mutex> lk(ck_mu_);
        for (;;) {
            ck_cv_.wait(lk, [this] { return ck_quit_ || ck_pending_ > 0; });
            if (ck_quit_) return;
            lk.unlock();
            char * b = nullptr;
            if (cudaHostAlloc((void **) &b, ck_bytes_, cudaHostAllocPortable) != cudaSuccess) { cudaGetLastError(); b = nullptr; }
            lk.lock();
            --ck_pending_;
            ck_ready_.push_back(b);
        }
    }
    void ck_thread_stop() {
        {
            std::lock_guard<std::mutex> lk(ck_mu_);
            ck_quit_ = true;
        }
        ck_cv_.notify_one();
        if (ck_thread_.joinable()) ck_thread_.join();
        ck_drain_ready();
    }

    char * ck_alloc() {
        ck_drain_ready();
        if (!ck_spare_.empty()) { char * b = ck_spare_.back(); ck_spare_.pop_back(); return b; }
        if ((ck_n_ + 1) * ck_bytes_ > (p_.ckpt_budget_mb << 20) || m_->host_used() + ck_bytes_ > m_->host_budget()) {
            // the least recently used checkpoint of an idle entry, else none
            ent * be = nullptr;
            size_t bi = 0;
            for (auto & kv : E_) {
                if (kv.second.open) continue;
                for (size_t i = 0; i < kv.second.cps.size(); ++i)
                    if (!be || kv.second.cps[i].lru < be->cps[bi].lru) { be = &kv.second; bi = i; }
            }
            if (!be) {
                // no idle checkpoint to recycle: the host budget is held by offloaded entries. P100E_CKPT_ROOM drops the
                // least recently used offloaded one (a checkpoint of the running prompt is worth more than an old
                // entry's rows); otherwise the checkpoint is skipped (logged, the next request re-reads from further back)
                int o = -1;
                uint64_t bt = 0;
                if (ck_room_ && (ck_n_ + 1) * ck_bytes_ <= (p_.ckpt_budget_mb << 20))
                    for (const auto & kv : E_)
                        if (!kv.second.open && !m_->seq_resident(kv.first) && (o < 0 || kv.second.t_used < bt)) {
                            o = kv.first; bt = kv.second.t_used;
                        }
                if (o >= 0) {
                    fprintf(stderr, "engine_tp: prompt cache: host budget full, dropping offloaded entry %d (%zu rows) "
                                    "for a checkpoint\n", o, E_.at(o).keys.size());
                    drop_entry(o);
                    return ck_alloc();
                }
                // the checkpoint budget is held by open sequences (a long chat accumulates checkpoints every turn):
                // recycle the least recently used one of them (a loaded checkpoint is never needed again by its load)
                if (ck_room_)
                    for (auto & kv : E_)
                        for (size_t i = 0; i < kv.second.cps.size(); ++i)
                            if (!be || kv.second.cps[i].lru < be->cps[bi].lru) { be = &kv.second; bi = i; }
            }
            if (!be) {
                if (ck_fail_++ % 32 == 0)
                    fprintf(stderr, "engine_tp: prompt cache: no room for a checkpoint (%d buffers, host %zu of %zu MiB), "
                                    "skipped (%llu so far)\n", ck_n_, m_->host_used() >> 20, m_->host_budget() >> 20,
                            (unsigned long long) ck_fail_);
                return nullptr;
            }
            char * b = be->cps[bi].buf;
            be->cps.erase(be->cps.begin() + bi);
            ck_join();   // the evicted buffer may be the pending async save
            return b;   // still counted in ck_n_
        }
        char * b = nullptr;
        if (cudaHostAlloc((void **) &b, ck_bytes_, cudaHostAllocPortable) != cudaSuccess) {
            cudaGetLastError();
            return nullptr;
        }
        ++ck_n_;
        m_->host_extra_add((long long) ck_bytes_);
        return b;
    }
    // P100E_PF_CKPT_ASYNC=0: take_ckpt waits for its D2H copy (state_save) instead of overlapping the next segment
    static bool ck_async() {
        static const bool v = !getenv("P100E_PF_CKPT_ASYNC") || atoi(getenv("P100E_PF_CKPT_ASYNC")) != 0;
        return v;
    }
    void ck_join() { m_->state_save_join(); }   // a pending async checkpoint copy: wait before its buffer is read / freed
    void ck_free(char * b) {
        if (!b) return;
        ck_join();
        if ((int) ck_spare_.size() < std::max(2, ck_spare_target())) { ck_spare_.push_back(b); return; }
        cudaFreeHost(b);
        --ck_n_;
        m_->host_extra_add(-(long long) ck_bytes_);
    }
    void take_ckpt(int seq, ent & e) {
        const int pos = (int) e.keys.size();
        for (const auto & c : e.cps) if (c.pos == pos) return;
        const auto ta = clk::now();
        char * b = ck_alloc();
        if (!b) return;
        const double a_ms = 1e3 * secs(ta);
        const size_t sb = m_->state_bytes();
        std::vector<char *> dst;
        for (int r = 0; r < m_->n_ranks(); ++r) dst.push_back(b + (size_t) r * sb);
        const auto ts = clk::now();
        if (ck_async()) m_->state_save_async(seq, dst);   // joined by ck_join (end of prefill_, ck_free, open_cached)
        else m_->state_save(seq, dst);
        const double s_ms = 1e3 * secs(ts);
        const auto tc = clk::now();
        static const bool ck_check = getenv("P100E_PF_CKPT_CHECK") && atoi(getenv("P100E_PF_CKPT_CHECK")) != 0;
        if (ck_check) {   // verification: the synchronous copy of the same state must be byte-identical
            std::vector<char> ref((size_t) m_->n_ranks() * sb);
            std::vector<char *> rd;
            for (int r = 0; r < m_->n_ranks(); ++r) rd.push_back(ref.data() + (size_t) r * sb);
            m_->state_save(seq, rd);
            ck_join();
            const bool same = memcmp(ref.data(), b, ref.size()) == 0;
            fprintf(stderr, "CKCHECK pos %d: async checkpoint %s sync copy (%zu bytes)\n", pos, same ? "==" : "DIFFERS FROM", ref.size());
        }
        ckpt c;
        c.pos = pos;
        c.buf = b;
        c.carry = d_ && d_->carry_to_host(seq, (float *) (b + (size_t) m_->n_ranks() * sb));
        if (prof_) fprintf(stderr, "PFPROF ckpt parts: alloc %.1f, save %.1f, carry %.1f ms\n", a_ms, s_ms, 1e3 * secs(tc));
        ck_topup();   // the next checkpoint's buffer is allocated while this prompt / its reply runs
        c.lru = ++tick_;
        auto it = std::lower_bound(e.cps.begin(), e.cps.end(), pos, [](const ckpt & a, int p) { return a.pos < p; });
        e.cps.insert(it, c);
    }

    // ---- prefill ----
    std::vector<float> run_text(int seq, const std::vector<int32_t> & tokens, int P0) {
        if (d_) d_->carry_load(seq);   // h of position P0 - 1 (zero for a fresh sequence)
        std::vector<float> L;
        int c0 = 0;
        for (int c : m_->ubatch_chunks((int) tokens.size(), 1)) {
            const std::vector<int32_t> ch(tokens.begin() + c0, tokens.begin() + c0 + c);
            const auto tf = clk::now();
            L = m_->forward_n(ch, 1);
            const double f_ms = 1e3 * secs(tf);
            const auto tc = clk::now();
            if (d_) d_->catchup_kv(ch, P0 + c0);
            if (prof_) fprintf(stderr, "PFPROF seg T=%d at %d: forward %.1f ms, mtp catch-up %.1f ms\n", c, P0 + c0, f_ms, 1e3 * secs(tc));
            c0 += c;
        }
        if (d_) d_->carry_save(seq);
        return L;
    }

    std::vector<float> run_pieces(int seq, const std::vector<prompt_piece> & pieces, int P0) {
        bool img = false;
        for (const auto & pc : pieces) img = img || pc.embd != nullptr;
        if (!img) {
            std::vector<int32_t> t;
            for (const auto & pc : pieces) t.insert(t.end(), pc.tokens.begin(), pc.tokens.end());
            return run_text(seq, t, P0);
        }
        if (d_) d_->carry_load(seq);
        std::vector<tp_model::mm_in> in;
        for (const auto & pc : pieces) {
            tp_model::mm_in q;
            if (pc.embd) { q.embd = pc.embd.get(); q.nx = pc.nx; q.ny = pc.ny; q.n_embd = pc.n_embd; }
            else { q.tokens = pc.tokens.data(); q.n_tok = (int) pc.tokens.size(); }
            in.push_back(q);
        }
        std::vector<float> L = d_ ? m_->forward_mm(in, [&](int p0, const std::vector<int32_t> & ch) { d_->catchup_kv(ch, p0); })
                                  : m_->forward_mm(in);
        if (d_) d_->carry_save(seq);
        return L;
    }

    static void append_pieces(std::vector<prompt_piece> & dst, const std::vector<prompt_piece> & more) {
        for (const auto & pc : more) {
            if (!pc.embd && !dst.empty() && !dst.back().embd) {
                dst.back().tokens.insert(dst.back().tokens.end(), pc.tokens.begin(), pc.tokens.end());
            } else {
                dst.push_back(pc);
            }
        }
    }

    // record rows fed: keys from the planned prompt where they agree with what was fed, else token ids / fresh keys
    void record(ent & e, const std::vector<prompt_piece> & pieces) {
        for (const auto & pc : pieces) {
            if (pc.embd) {
                for (int i = 0; i < pc.n_rows(); ++i) {
                    const size_t at = e.keys.size();
                    e.keys.push_back(at < e.want.size() && (e.want[at] & IMG_BIT) ? e.want[at] : (IMG_BIT | ++img_uniq_));
                    e.toks.push_back(-1);
                }
            } else {
                for (int32_t t : pc.tokens) {
                    e.keys.push_back((uint64_t) (uint32_t) t);
                    e.toks.push_back(t);
                }
            }
        }
    }

    // pieces [rows a, rows b) of a piece list (text cut anywhere, images whole: callers cut outside images)
    static std::vector<prompt_piece> slice(const std::vector<prompt_piece> & v, int a, int b) {
        std::vector<prompt_piece> out;
        int n = 0;
        for (const auto & pc : v) {
            const int r = pc.n_rows();
            const int lo = std::max(a, n), hi = std::min(b, n + r);
            if (lo < hi) {
                if (pc.embd) out.push_back(pc);
                else {
                    prompt_piece q;
                    q.tokens.assign(pc.tokens.begin() + (lo - n), pc.tokens.begin() + (hi - n));
                    out.push_back(std::move(q));
                }
            }
            n += r;
        }
        return out;
    }

    void prefill_(int seq, const std::vector<prompt_piece> & pieces, bool want_logits, decode_res & out) {
        ent & e = get(seq);
        if (pieces.empty()) throw std::runtime_error("prefill: no input");
        if (e.last == op::decode || e.last == op::keep || e.last == op::draft)
            throw std::runtime_error("prefill after generation is not supported (seq_keep(0) first)");
        int rows = 0;
        bool img = false;
        for (const auto & pc : pieces) {
            if (pc.embd && (pc.nx <= 0 || pc.ny <= 0 || pc.n_embd != info_.n_embd))
                throw std::runtime_error("prefill_mm: bad image piece (grid " + std::to_string(pc.nx) + "x" +
                                         std::to_string(pc.ny) + ", width " + std::to_string(pc.n_embd) + ", model " +
                                         std::to_string(info_.n_embd) + ")");
            if (!pc.embd && pc.tokens.empty()) throw std::runtime_error("prefill: empty text piece");
            rows += pc.n_rows();
            img = img || pc.embd != nullptr;
        }
        const int P0 = (int) e.keys.size();
        if (P0 + rows > p_.n_ctx) throw std::runtime_error("context full");
        const auto t0 = clk::now();
        need_pages(seq, P0 + rows);
        m_->seq_select(seq);
        const uint64_t sat0 = m_->prefill_saturations();
        // segments ending at the planned checkpoints inside this call (never inside an image: message starts are text)
        std::vector<int> cuts;
        for (int c : e.ck_at) if (c > P0 && c < P0 + rows) cuts.push_back(c - P0);
        cuts.push_back(rows);
        if (img && !e.has_img) {   // the history so far is text: kept as pieces from now on (f32-wire redo)
            e.hist.clear();
            if (!e.toks.empty()) {
                prompt_piece t;
                t.tokens = e.toks;
                e.hist.push_back(std::move(t));
            }
        }
        std::vector<float> L;
        try {
            int a = 0;
            for (int b : cuts) {
                const std::vector<prompt_piece> seg = slice(pieces, a, b);
                L = run_pieces(seq, seg, P0 + a);
                record(e, seg);
                if (e.has_img || img) append_pieces(e.hist, seg);
                e.has_img = e.has_img || img;
                if (b < rows && m_->prefill_saturations() == sat0) {
                    const auto tk = clk::now();
                    take_ckpt(seq, e);
                    if (prof_) fprintf(stderr, "PFPROF ckpt at %d: %.1f ms\n", P0 + b, 1e3 * secs(tk));
                }
                a = b;
            }
            // a planned checkpoint exactly at this call's end: the server feeds ubatch-sized calls counted from the reuse
            // point, so every P100E_CKPT_EVERY position (a multiple of the ubatch past it) and any message start / LCP
            // point on a chunk boundary lands here, where the in-call cut above (c < P0 + rows) never sees it
            if (m_->prefill_saturations() == sat0 && std::binary_search(e.ck_at.begin(), e.ck_at.end(), P0 + rows))
                take_ckpt(seq, e);
            ck_join();
            if (m_->prefill_saturations() != sat0) L = redo_exact(seq, e, pieces, P0, m_->prefill_saturations() - sat0);
        } catch (...) {
            // a segment (or the f32 redo) failed part-way: the state may be past the recorded keys, or clamped under
            // the full history's keys - the entry must not serve a later request from the cache
            e.poisoned = true;
            throw;
        }
        if (!e.has_img) e.hist.clear();
        out.argmax.clear();
        out.logits.clear();
        if (want_logits) out.logits = std::move(L);
        else out.argmax.push_back(argmax(L.data(), info_.n_vocab));
        e.last = op::prefill;
        e.step.clear();
        st_.t_prompt_s += secs(t0);
        st_.n_prompt += rows;
        if (prof_) fprintf(stderr, "PFPROF prefill_ call: %d rows at %d, %zu segment(s), total %.1f ms\n", rows, P0, cuts.size(), 1e3 * secs(t0));
    }

    // the f16 prefill wire saturated: the sequence's whole history again from an empty state on the f32 wire (the
    // instance stays on the f32 wire); checkpoints of the clamped run are dropped
    std::vector<float> redo_exact(int seq, ent & e, const std::vector<prompt_piece> & more, int P0, uint64_t n_sat) {
        std::vector<prompt_piece> all;
        if (e.has_img) {
            all = e.hist;   // includes `more` already
        } else {
            prompt_piece t;
            t.tokens.assign(e.toks.begin(), e.toks.end());
            all.push_back(t);
        }
        (void) more; (void) P0;
        int rows = 0;
        for (const auto & pc : all) rows += pc.n_rows();
        fprintf(stderr, "engine_tp: the f16 prefill wire saturated (%llu partial group(s)); redoing the %d-row %sprompt "
                        "on the f32 wire\n", (unsigned long long) n_sat, rows, e.has_img ? "multimodal " : "");
        for (auto & c : e.cps) ck_free(c.buf);
        e.cps.clear();
        m_->seq_select(seq);
        m_->reset();
        if (d_) d_->carry_drop(seq);
        const uint64_t sat1 = m_->prefill_saturations();
        std::vector<float> L = run_pieces(seq, all, 0);
        if (m_->prefill_saturations() != sat1) throw std::runtime_error("prefill: saturated again on the f32 wire");
        st_.n_prefill_redo++;
        return L;
    }

    // ---- decode ----
    void decode_(const std::vector<decode_req> & reqs, std::vector<decode_res> & out) {
        // results reuse the caller's buffers of the previous step (no fresh multi-MB logits vectors per step)
        out.resize(reqs.size());
        for (decode_res & o : out) { o.argmax.clear(); o.logits.clear(); }
        if (reqs.empty()) return;
        int rows = 0;
        std::set<int> seen;
        for (const decode_req & r : reqs) {
            ent & e = get(r.seq);
            const int T = (int) r.tokens.size();
            if (T < 1 || T > 1 + p_.mtp_k || T > 4) throw std::runtime_error("decode: rows must be 1..1+k");
            if (e.last == op::none) throw std::runtime_error("decode before prefill");
            if (!seen.insert(r.seq).second) throw std::runtime_error("decode: a sequence twice in one step");
            if ((int) e.keys.size() + T > p_.n_ctx) throw engine_seq_error(r.seq, "context full");
            rows += T;
        }
        if (rows > tp_model::MAX_ROWS) throw std::runtime_error("decode: more than 8 rows in one step");
        // pages of every sequence before anything runs: a sequence the pool has no room for fails alone
        // (engine_seq_error), the step is repeated for the others
        for (const decode_req & r : reqs) need_pages(r.seq, (int) get(r.seq).keys.size() + (int) r.tokens.size());
        const auto t0 = clk::now();
        std::vector<tp_model::seq_rows> b;
        std::vector<int> lrows;
        int row = 0;
        bool keep = false;
        for (const decode_req & r : reqs) {
            ent & e = get(r.seq);
            if (d_ && e.last != op::draft) e.draft_ok = false;
            b.push_back({r.seq, r.tokens});
            e.row0 = row;
            if (r.logits) for (int t = 0; t < (int) r.tokens.size(); ++t) lrows.push_back(row + t);
            keep = keep || r.tokens.size() > 1;
            row += (int) r.tokens.size();
        }
        std::vector<float> & L = dec_L_;   // kept across steps: assign() below reuses its capacity
        std::vector<int32_t> g;
        try {
            g = m_->forward_batch_ids(b, keep, lrows.empty() ? nullptr : &lrows, &L);
        } catch (...) {
            // the forward may have advanced the KV / delta-net state of any of these sequences past their keys:
            // none of them may be kept as a cache entry
            for (const decode_req & r : reqs) get(r.seq).poisoned = true;
            throw;
        }
        int li = 0;
        for (size_t i = 0; i < reqs.size(); ++i) {
            const decode_req & r = reqs[i];
            ent & e = get(r.seq);
            const int T = (int) r.tokens.size();
            decode_res & o = out[i];
            if (r.logits) {
                if (li == 0 && (size_t) T * info_.n_vocab == L.size()) {
                    o.logits.swap(L);   // the only logits rows: no copy; L keeps o's old buffer for the next step
                } else {
                    o.logits.assign(L.begin() + (size_t) li * info_.n_vocab, L.begin() + (size_t) (li + T) * info_.n_vocab);
                }
                li += T;
            } else {
                o.argmax.assign(g.begin() + e.row0, g.begin() + e.row0 + T);
            }
            for (int32_t t : r.tokens) { e.keys.push_back((uint64_t) (uint32_t) t); e.toks.push_back(t); }
            e.step = r.tokens;
            e.last = op::decode;
        }
        st_.t_decode_s += secs(t0);
        st_.n_decode_steps++;
        st_.n_decode_rows += rows;
    }

    void keep_(int seq, int n_keep) {
        ent & e = get(seq);
        if (e.last != op::decode) throw std::runtime_error("keep() must follow decode()");
        const int T = (int) e.step.size();
        if (n_keep < 1 || n_keep > T) throw std::runtime_error("keep: n_keep out of range");
        if (n_keep < T) {
            try {
                m_->seq_select(seq);
                m_->rollback(T, n_keep);
            } catch (...) {
                e.poisoned = true;   // the state holds rows the keys are about to drop
                throw;
            }
            e.keys.resize(e.keys.size() - (size_t) (T - n_keep));
            e.toks.resize(e.toks.size() - (size_t) (T - n_keep));
            e.step.resize((size_t) n_keep);
        }
        if (e.has_img) {   // decoded rows are history too (a later reuse of this entry cuts hist by rows)
            prompt_piece t;
            t.tokens = e.step;
            append_pieces(e.hist, {t});
        }
        e.last = op::keep;
    }

    std::vector<int32_t> draft_(int seq, int32_t next, int k_max) {
        ent & e = get(seq);
        std::vector<int32_t> drafts;
        e.dp.clear();
        if (!d_) return drafts;
        if (e.last != op::prefill && e.last != op::keep) throw std::runtime_error("draft() must follow prefill() or keep()");
        if (!e.draft_ok) {
            e.last = op::draft;
            return drafts;
        }
        const auto t0 = clk::now();
        const int N = (int) e.keys.size();   // position of `next`
        m_->seq_select(seq);
        need_pages(seq, N + 1 + mo_.k);
        int32_t d0;
        if (e.last == op::prefill) {
            d_->carry_load(seq);   // h input row 0 = h_{N-1}, the catch-up's carry
            d0 = d_->run({next}, N, tp_mtp::H_READY);
        } else {
            std::vector<int32_t> acc(e.step.begin() + 1, e.step.end());
            acc.push_back(next);
            d_->h_from_main(0, e.row0, (int) acc.size());   // this sequence's rows of the batched step
            d_->carry_save_row(seq, (int) acc.size() - 1);  // h of `next`'s predecessor (a later prefill continues here)
            d0 = d_->run(acc, N - (int) e.step.size() + 1, tp_mtp::H_READY);
        }
        st_.n_draft_runs++;
        const int k = std::min(k_max, mo_.k);
        if (k > 0) {
            double cum = d_->last_p();
            drafts.push_back(d0);
            e.dp.push_back((float) cum);
            for (int j = 1; j < k; ++j) {
                const int32_t dj = d_->run({drafts.back()}, N + j, tp_mtp::H_SELF);
                st_.n_draft_runs++;
                cum *= d_->last_p();
                if (mo_.tau > 0.0f && j >= mo_.k_min && cum < mo_.tau) break;
                drafts.push_back(dj);
                e.dp.push_back((float) cum);
            }
        }
        st_.n_drafted += (int64_t) drafts.size();
        st_.t_draft_s += secs(t0);
        e.last = op::draft;
        return drafts;
    }

    // draft_() of several sequences after the same decode step, as ONE draft-head pass (tp_mtp::run_multi) over every
    // sequence's catch-up rows (kept drafts + next, h = its main-model rows of the step), then one run_multi per chained
    // round over the sequences still drafting (the last one alone: the single-sequence run) - sched::draft_batched with
    // draft_()'s policy. Every run_multi row is bitwise the single-sequence run's row, so out[i] == draft_(reqs[i]).
    // Sequences that cannot join (first draft after a prefill, draft head out of sync) and P100E_DRAFT_BATCH=0: draft_().
    void draft_batch_(const std::vector<draft_req> & reqs, std::vector<std::vector<int32_t>> & out) {
        out.assign(reqs.size(), {});
        // the pages of every sequence that will draft (the need_pages calls of draft_ and of the batched pass below)
        // before any draft runs: one the pool has no room for fails alone (engine_seq_error) with nothing changed
        if (d_)
            for (const draft_req & rq : reqs) {
                ent & e = get(rq.seq);
                if ((e.last == op::prefill || e.last == op::keep) && e.draft_ok) need_pages(rq.seq, (int) e.keys.size() + 1 + mo_.k);
            }
        static const bool on = !getenv("P100E_DRAFT_BATCH") || atoi(getenv("P100E_DRAFT_BATCH")) != 0;
        std::vector<int> bi;   // indices of reqs that join the batched pass
        if (d_ && on && d_->fused()) {
            std::set<int> seen;
            for (size_t i = 0; i < reqs.size(); ++i) {
                ent & e = get(reqs[i].seq);
                if (!seen.insert(reqs[i].seq).second) throw std::runtime_error("draft_batch: a sequence twice");
                if (e.last == op::keep && e.draft_ok) bi.push_back((int) i);
            }
        }
        if (bi.size() < 2) bi.clear();
        std::vector<char> in_b(reqs.size(), 0);
        for (int i : bi) in_b[i] = 1;
        for (size_t i = 0; i < reqs.size(); ++i)
            if (!in_b[i]) out[i] = draft_(reqs[i].seq, reqs[i].next, reqs[i].k_max);
        if (bi.empty()) return;

        const auto t0 = clk::now();
        const int n = (int) bi.size();
        std::vector<tp_model::seq_rows> rows;
        std::vector<int> p0s, N(n);
        int r = 0;
        for (int q = 0; q < n; ++q) {
            const draft_req & rq = reqs[bi[q]];
            ent & e = get(rq.seq);
            N[q] = (int) e.keys.size();   // position of `next`
            need_pages(rq.seq, N[q] + 1 + mo_.k);
            std::vector<int32_t> acc(e.step.begin() + 1, e.step.end());
            acc.push_back(rq.next);
            d_->h_from_main(r, e.row0, (int) acc.size());          // this sequence's rows of the batched step
            d_->carry_save_row(rq.seq, r + (int) acc.size() - 1);  // h of `next`'s predecessor
            p0s.push_back(N[q] - (int) e.step.size() + 1);
            r += (int) acc.size();
            rows.push_back({rq.seq, std::move(acc)});
        }
        const std::vector<int32_t> picks = d_->run_multi(rows, p0s);
        st_.n_draft_runs += n;
        struct cs { double cum; int k; bool active, single; int midx; };
        std::vector<cs> st(n);
        int kmax = 0;
        for (int q = 0; q < n; ++q) {
            const int k = std::min(reqs[bi[q]].k_max, mo_.k);
            st[q] = {d_->multi_p(q), k, k > 1, false, q};
            get(reqs[bi[q]].seq).dp.clear();
            if (k > 0) { out[bi[q]].push_back(picks[q]); get(reqs[bi[q]].seq).dp.push_back((float) st[q].cum); }
            kmax = std::max(kmax, k);
        }
        for (int j = 1; j < kmax; ++j) {
            std::vector<int> act;
            for (int q = 0; q < n; ++q) if (st[q].active && j < st[q].k) act.push_back(q);
            if (act.empty()) break;
            std::vector<int32_t> pk;
            if (act.size() == 1) {
                cs & s = st[act[0]];
                const int seq = reqs[bi[act[0]]].seq;
                m_->seq_select(seq);
                if (!s.single) d_->h_from_multi(0, s.midx);
                pk.push_back(d_->run({out[bi[act[0]]].back()}, N[act[0]] + j, s.single ? tp_mtp::H_SELF : tp_mtp::H_READY));
                s.single = true;
            } else {
                std::vector<tp_model::seq_rows> cr;
                std::vector<int> cp;
                for (size_t m = 0; m < act.size(); ++m) {
                    d_->h_from_multi((int) m, st[act[m]].midx);
                    cr.push_back({reqs[bi[act[m]]].seq, {out[bi[act[m]]].back()}});
                    cp.push_back(N[act[m]] + j);
                }
                pk = d_->run_multi(cr, cp);
            }
            for (size_t m = 0; m < act.size(); ++m) {
                cs & s = st[act[m]];
                st_.n_draft_runs++;
                const float p = act.size() == 1 ? d_->last_p() : d_->multi_p((int) m);
                if (act.size() > 1) s.midx = (int) m;
                s.cum *= p;
                if (j + 1 >= s.k) s.active = false;
                if (mo_.tau > 0.0f && j >= mo_.k_min && s.cum < mo_.tau) { s.active = false; continue; }
                out[bi[act[m]]].push_back(pk[m]);
                get(reqs[bi[act[m]]].seq).dp.push_back((float) s.cum);
            }
        }
        for (int q = 0; q < n; ++q) {
            st_.n_drafted += (int64_t) out[bi[q]].size();
            get(reqs[bi[q]].seq).last = op::draft;
        }
        st_.t_draft_s += secs(t0);
    }

    engine_tp_params p_;
    engine_info info_;
    std::unique_ptr<tp_model> m_;
    std::unique_ptr<tp_mtp> d_;
    mtp_opts mo_;
    std::map<int, ent> E_;           // open sequences and idle cache entries, by tp_model sequence id
    uint64_t tick_ = 0;
    uint64_t img_uniq_ = 0;
    size_t ck_bytes_ = 0;
    int ck_n_ = 0;                   // pinned checkpoint buffers (live + spare)
    std::vector<char *> ck_spare_;
    std::vector<char *> ck_ready_;   // allocated by ck_thread_ (nullptr = failed), under ck_mu_
    std::mutex ck_mu_;
    std::condition_variable ck_cv_;
    std::thread ck_thread_;
    int ck_pending_ = 0;             // allocations requested, not yet in ck_ready_ (under ck_mu_)
    bool ck_quit_ = false;
    cache_stats cs_;
    engine_stats st_;
    int lcp_min_ = 1024;             // P100E_LCP_CKPT
    int ck_every_ = 8192;            // P100E_CKPT_EVERY
    bool ck_room_ = true;            // P100E_CKPT_ROOM
    int takeover_tail_ = 8192;       // P100E_TAKEOVER_TAIL
    bool takeover_any_ = true;       // P100E_TAKEOVER_ANY
    uint64_t ck_fail_ = 0;
    std::vector<float> dec_L_;       // decode_'s logits rows, reused across steps
};

} // namespace

std::unique_ptr<engine> make_engine_tp(const engine_tp_params & p) {
    return std::make_unique<engine_tp>(p);
}

} // namespace p100e
