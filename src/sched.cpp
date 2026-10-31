#include "p100e_sched.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <io.h>
#include <windows.h>
#else
#include <unistd.h>
#endif

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace p100e {

static double clk_s() {
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double>(clk::now().time_since_epoch()).count();
}

sched::sched(tp_model & m, tp_mtp * d, const sched_opts & o) : m_(m), d_(d), o_(o) {
    t0_ = clk_s();
    o_.max_rows = std::max(1, std::min(o_.max_rows, (int) tp_model::MAX_ROWS));
    if (o_.mo.k < 1 || o_.mo.k > 3) throw std::runtime_error("sched: MTP k must be 1..3");
    if (const char * e = getenv("P100E_CKPT_BUDGET_MB")) o_.ckpt_budget = (size_t) atoll(e) << 20;
    o_.ckpt_spare = std::max(0, o_.ckpt_spare);
    if (const char * e = getenv("P100E_CACHE")) o_.cache = atoi(e) != 0;
    if (const char * e = getenv("P100E_CACHE_EXACT")) o_.exact = atoi(e) != 0;
    if (const char * e = getenv("P100E_DECODE_SHARE")) o_.decode_share = atof(e);
    o_.decode_share = std::min(0.95, std::max(0.0, o_.decode_share));
    if (d_) {
        // rollback snapshot slots for the drafts of one step: k for a single job, else as many as the draft row budget
        // leaves (one row per job). Allocated now (before graphs / an auto-sized pool).
        const int cap = std::min(o_.max_rows, o_.draft_rows);
        const int snaps = o_.max_active <= 1 ? o_.mo.k : std::min(cap - 1, o_.mo.k * o_.max_active);
        m_.enable_rollback(1 + std::max(o_.mo.k, snaps));
    }
    ck_bytes_ = m_.state_bytes() * m_.n_ranks() + (size_t) m_.n_embd() * 4;
}

sched::~sched() {
    try {
        std::vector<int> seqs;
        for (auto & kv : ents_) seqs.push_back(kv.first);
        for (int s : seqs) drop_entry(s);
    } catch (...) {
    }
    for (char * p : ck_free_) cudaFreeHost(p);
    m_.host_extra_add(-(long long) (ck_alloc_ * ck_bytes_));
}

double sched::now() const { return clk_s() - t0_; }

int sched::n_ckpts() const {
    int n = 0;
    for (const auto & kv : ents_) n += (int) kv.second.cps.size();
    return n;
}

std::vector<int> sched::clean_splits(const std::vector<int> & sp, int n) const {
    std::vector<int> v;
    for (int x : sp) if (x > 0 && x < n) v.push_back(x);
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
    return v;
}

// The partition rule: from boundary b, the segment ends at the first split position >= b + min_split (or at n); the
// segment runs as tp_model::ubatch_chunks (max_batch chunks, the remainder last). Sequential from 0, so the boundaries
// below any position depend only on the split positions below it (and a run that passes through boundary c continues
// exactly as one that starts at c).
std::vector<int> sched::partition(const std::vector<int> & sp, int n, int from, std::vector<bool> * expl) const {
    std::vector<int> b{from};
    if (expl) expl->assign(1, false);
    size_t k = 0;
    int pos = from;
    const int ms = std::max(1, o_.min_split);
    while (pos < n) {
        while (k < sp.size() && sp[k] < pos + ms) ++k;
        const bool is_split = k < sp.size() && sp[k] < n;
        const int end = is_split ? sp[k] : n;
        for (int c : m_.ubatch_chunks(end - pos, 1)) {
            pos += c;
            b.push_back(pos);
            if (expl) expl->push_back(is_split && pos == end);
        }
    }
    return b;
}

int sched::submit(const std::vector<int32_t> & prompt, int n_gen, bool mtp, bool keep_seq) {
    request r;
    r.prompt = prompt;
    r.n_gen = n_gen;
    r.mtp = mtp;
    r.keep = keep_seq;
    return submit(r);
}

int sched::submit(const request & r) {
    if (r.prompt.empty() || r.n_gen < 1) throw std::runtime_error("sched: empty prompt or n_gen < 1");
    if (r.mtp && !d_) throw std::runtime_error("sched: MTP job without a draft head");
    const int total = (int) r.prompt.size() + r.n_gen + o_.reserve_extra;
    if (total > m_.n_ctx()) throw std::runtime_error("sched: job longer than the context");
    job j;
    j.id = (int) jobs_.size();
    j.prompt = r.prompt;
    j.n_gen = r.n_gen;
    j.mtp = r.mtp;
    j.keep_seq = r.keep;
    j.cache = r.cache && o_.cache;
    j.splits = clean_splits(r.splits, (int) r.prompt.size());
    j.stop = r.stop;
    j.t_submit = now();
    j.reserved_pages = pages_for(total);
    // a job larger than the whole pool would wait forever
    if (m_.paged() && m_.pool_ready() && j.reserved_pages > m_.pool_pages())
        throw std::runtime_error("sched: job needs " + std::to_string(j.reserved_pages) + " KV pages, the pool holds " +
                                 std::to_string(m_.pool_pages()));
    jobs_.push_back(j);
    live_.push_back(j.id);
    queue_.push_back(j.id);
    adm_dirty_ = true;
    return j.id;
}

int sched::submit_continue(int done_id, int n_gen, bool mtp) {
    const job & o = jobs_.at(done_id);
    request r;
    r.n_gen = n_gen;
    r.mtp = mtp;
    r.keep = o.keep_seq;
    r.cache = o.cache;
    r.stop = o.stop;
    return submit_continue(done_id, r);
}

int sched::submit_continue(int done_id, const request & r) {
    job & o = jobs_.at(done_id);
    if (o.st != job::DONE || o.seq < 0 || !ents_.count(o.seq) || ents_.at(o.seq).done_job != done_id ||
        ents_.at(o.seq).owner >= 0)
        throw std::runtime_error("sched: continue needs a finished job that still owns its sequence");
    if (r.mtp && !d_) throw std::runtime_error("sched: MTP job without a draft head");
    if (r.n_gen < 1) throw std::runtime_error("sched: n_gen < 1");
    if (o.out.empty() && r.prompt.empty()) throw std::runtime_error("sched: continue needs a generated token or new input");
    const int len = m_.seq_len(o.seq);
    job j;
    j.id = (int) jobs_.size();
    j.n_gen = r.n_gen;
    j.mtp = r.mtp;
    j.keep_seq = r.keep_set ? r.keep : (r.keep || o.keep_seq);   // inherits the finished job's keep unless set
    j.cache = r.cache && o_.cache;
    j.stop = r.stop;
    j.cont_of = done_id;
    int total = len + r.n_gen + o_.reserve_extra;
    if (!r.prompt.empty()) {
        // the whole sequence: its tokens, the last generated token (at position len, never fed), the new input
        j.append = true;
        j.prompt = m_.seq_tokens(o.seq);
        if (!o.out.empty()) j.prompt.push_back(o.out.back());
        j.prompt.insert(j.prompt.end(), r.prompt.begin(), r.prompt.end());
        j.splits = clean_splits(r.splits, (int) j.prompt.size());
        total = (int) j.prompt.size() + r.n_gen + o_.reserve_extra;
    }
    if (total > m_.n_ctx()) throw std::runtime_error("sched: continuation longer than the context");
    j.t_submit = now();
    j.reserved_pages = pages_for(total);
    if (m_.paged() && m_.pool_ready() && j.reserved_pages > m_.pool_pages())
        throw std::runtime_error("sched: continuation needs more KV pages than the pool holds");
    ents_.at(o.seq).owner = j.id;   // reserved for the continuation: not matched, evicted or released meanwhile
    jobs_.push_back(j);
    live_.push_back(j.id);
    queue_.push_back(j.id);
    adm_dirty_ = true;
    return j.id;
}

void sched::unlive(int id) {
    const auto it = std::find(live_.begin(), live_.end(), id);
    if (it != live_.end()) live_.erase(it);
    const auto q = std::find(queue_.begin(), queue_.end(), id);
    if (q != queue_.end()) queue_.erase(q);
}

void sched::cancel(int id, bool drop) {
    job & j = jobs_.at(id);
    if (j.st == job::DONE || j.st == job::FAILED) {
        if (drop && j.st == job::DONE) release(id);
        return;
    }
    adm_dirty_ = true;
    j.cancelled = true;
    if (j.st == job::QUEUED) {   // never admitted (a continuation: its session stays with the finished job)
        fail(j, "cancelled");
        return;
    }
    // running or preempted (its sequence is j.seq, resident or offloaded); a queued resume is dropped
    const auto q = std::find(queue_.begin(), queue_.end(), id);
    if (q != queue_.end()) queue_.erase(q);
    if (j.st == job::DECODE) {   // a step boundary: the sequence holds every emitted token but the last
        finish(j);
        if (drop) release(id);
        return;
    }
    // PREFILL: the prefilled part stays an anonymous cache entry (cache / keep), or goes
    unlive(id);
    j.st = job::FAILED;
    j.err = "cancelled";
    j.t_done = now();
    entry & e = ents_.at(j.seq);
    e.owner = -1;
    // an append continuation: back to the finished job it continues, at the turn's end (the checkpoint taken at
    // admission, or the state in place when nothing was prefilled yet), with that job's keep / cache flags
    if (!drop && j.append && j.cont_of >= 0 && jobs_[j.cont_of].st == job::DONE && jobs_[j.cont_of].seq < 0) {
        const int c = j.n_cached;
        const int np = m_.seq_len(j.seq);
        int ck = -1;
        for (size_t i = 0; i < e.cps.size(); ++i) if (e.cps[i].pos == c) ck = (int) i;
        // a preempted (offloaded) prefill: back on the GPU with only the pages of 0..c-1 first (seq_truncate and
        // state_load need a resident sequence); no free slot / pages for that: the session cannot go back (below)
        bool ok = np == c || ck >= 0;
        if (ok && np != c && !m_.seq_resident(j.seq)) {
            try {
                m_.seq_restore_prefix(j.seq, c);
            } catch (const std::exception & ex) {
                fprintf(stderr, "sched: cancel of job %d: cannot restore its sequence to give it back (%s)\n", id, ex.what());
                ok = false;
            }
        }
        if (ok) {
            if (np != c) {
                if (m_.seq_len(j.seq) != c) m_.seq_truncate(j.seq, c);   // restored above: already 0..c-1
                const ckpt_t & cp = e.cps.at(ck);
                const size_t sb = m_.state_bytes();
                std::vector<const char *> src;
                for (int r = 0; r < m_.n_ranks(); ++r) src.push_back(cp.buf + (size_t) r * sb);
                m_.state_load(j.seq, src);
                if (d_) {
                    if (cp.carry) { d_->carry_from_host(j.seq, (const float *) (cp.buf + (size_t) m_.n_ranks() * sb)); e.carry_at = c; }
                    else { d_->carry_drop(j.seq); e.carry_at = -1; }
                }
                while (e.bnd.size() > 1 && e.bnd.back() > c) e.bnd.pop_back();
                e.n_pf = e.bnd.back();
                e.n_draft = std::min(e.n_draft, c);
                ck_trim(e, c);   // after the state operations: a throw above leaves the checkpoints in place
            }
            job & o = jobs_[j.cont_of];
            e.done_job = o.id;
            e.pinned = o.keep_seq;
            e.cached = o.cache;
            e.t_used = now();
            o.seq = j.seq;
            o.offloaded = !m_.seq_resident(o.seq);   // np == c of a preempted prefill: it stays in host RAM
            j.seq = -1;
            j.offloaded = false;
            return;
        }
    }
    if (!drop && (j.cache || j.keep_seq) && m_.seq_len(j.seq) > 0) {
        e.done_job = -1;
        e.pinned = false;
        e.cached = true;
        e.t_used = now();
    } else {
        drop_entry(j.seq);
    }
    j.seq = -1;
    j.offloaded = false;
}

void sched::forget(int id) {
    job & j = jobs_.at(id);
    if (j.st != job::DONE && j.st != job::FAILED) throw std::runtime_error("sched: forget of a job that is not finished");
    if (j.seq >= 0 && ents_.count(j.seq) && ents_.at(j.seq).done_job == id && ents_.at(j.seq).owner >= 0)
        throw std::runtime_error("sched: forget of a job whose next turn is queued");
    if (j.seq >= 0 && ents_.count(j.seq) && ents_.at(j.seq).done_job == id && ents_.at(j.seq).owner < 0) {
        entry & e = ents_.at(j.seq);
        e.done_job = -1;
        e.pinned = false;
        if (!e.cached) drop_entry(j.seq);
    }
    j.seq = -1;
    j.offloaded = false;
    std::vector<int32_t>().swap(j.prompt);
    std::vector<int32_t>().swap(j.out);
    std::vector<int32_t>().swap(j.drafts);
    std::vector<int32_t>().swap(j.stop);
    std::vector<int>().swap(j.splits);
    std::vector<int>().swap(j.chunks);
    std::vector<int>().swap(j.ck);
    std::vector<float>().swap(j.L);
    adm_dirty_ = true;
}

void sched::release(int done_id) {
    adm_dirty_ = true;
    job & j = jobs_.at(done_id);
    if (j.st != job::DONE && j.st != job::FAILED) throw std::runtime_error("sched: release of a job that is not finished");
    if (j.seq >= 0 && ents_.count(j.seq) && ents_.at(j.seq).done_job == done_id) {
        entry & e = ents_.at(j.seq);
        // a continuation still queued on the session: it fails, then the session goes
        if (e.owner >= 0) {
            job & c = jobs_[e.owner];
            if (c.st == job::QUEUED) fail(c, "session released");
            e.owner = -1;
        }
        drop_entry(j.seq);
    }
    j.seq = -1;
}

void sched::drop_entry(int seq) {
    auto it = ents_.find(seq);
    if (it == ents_.end()) return;
    entry & e = it->second;
    ck_trim(e, -1);
    if (d_) d_->carry_drop(seq);
    if (e.done_job >= 0 && jobs_[e.done_job].seq == seq) { jobs_[e.done_job].seq = -1; jobs_[e.done_job].offloaded = false; }
    m_.seq_free(seq);
    ents_.erase(it);
}

void sched::fail(job & j, const std::string & why) {
    adm_dirty_ = true;
    unlive(j.id);
    j.st = job::FAILED;
    j.err = why;
    j.t_done = now();
    if (j.cont_of >= 0 && jobs_[j.cont_of].seq >= 0 && ents_.count(jobs_[j.cont_of].seq) &&
        ents_.at(jobs_[j.cont_of].seq).owner == j.id)
        ents_.at(jobs_[j.cont_of].seq).owner = -1;   // the session stays with its finished job
    // a job that holds its own sequence (a preempted job whose resume can never be admitted) gives it
    // back - its pinned host copy and sequence id would leak otherwise
    if (j.seq >= 0 && ents_.count(j.seq) && ents_.at(j.seq).owner == j.id) {
        ents_.at(j.seq).owner = -1;
        drop_entry(j.seq);
    }
    j.seq = -1;
    j.offloaded = false;
}

// ---- checkpoints ----

char * sched::ck_steal() {
    entry * be = nullptr;
    size_t bi = 0;
    for (auto & kv : ents_)
        for (size_t i = 0; i < kv.second.cps.size(); ++i)
            if (!be || kv.second.cps[i].lru < be->cps[bi].lru) { be = &kv.second; bi = i; }
    if (!be) return nullptr;
    char * b = be->cps[bi].buf;
    be->cps.erase(be->cps.begin() + bi);
    --ck_n_;
    ++ckpt_evictions;
    return b;
}

// A checkpoint buffer: a spare one; else, over the checkpoint cap, the least recently used checkpoint's; else a new
// pinned one if the model's host budget (shared with offloaded sequences) has room - offloaded cache-only entries
// borrowing the checkpoints' share are dropped (LRU) to make it - and otherwise the LRU checkpoint's.
char * sched::ck_get() {
    for (;;) {
        if (!ck_free_.empty()) {
            char * b = ck_free_.back();
            ck_free_.pop_back();
            return b;
        }
        if ((ck_n_ + 1) * ck_bytes_ > o_.ckpt_budget) return ck_steal();
        if (m_.host_used() + ck_bytes_ <= m_.host_budget()) break;
        if (!drop_offloaded(ck_keep_, -1)) return ck_steal();   // (its checkpoints may come back as spares: loop)
    }
    char * b = nullptr;
    if (cudaHostAlloc((void **) &b, ck_bytes_, cudaHostAllocPortable) != cudaSuccess) {
        cudaGetLastError();
        return ck_steal();
    }
    ++ck_alloc_;
    m_.host_extra_add((long long) ck_bytes_);
    return b;
}

void sched::ck_put(char * b) {
    if (!b) return;
    if ((int) ck_free_.size() < std::max(0, o_.ckpt_spare)) {
        ck_free_.push_back(b);
        return;
    }
    cudaFreeHost(b);   // the free list shrinks: pinned memory goes back to the shared budget
    --ck_alloc_;
    m_.host_extra_add(-(long long) ck_bytes_);
}

void sched::ck_trim(entry & e, int c) {
    while (!e.cps.empty() && e.cps.back().pos > c) {
        ck_put(e.cps.back().buf);
        e.cps.pop_back();
        --ck_n_;
    }
}

// the entry's delta-net state (position pos = its n_past) + the draft head's carry -> a checkpoint
void sched::take_ckpt(entry & e, int pos, bool carry) {
    for (const ckpt_t & c : e.cps) if (c.pos == pos) return;
    char * b = ck_get();
    if (!b) return;
    const size_t sb = m_.state_bytes();
    std::vector<char *> dst;
    for (int r = 0; r < m_.n_ranks(); ++r) dst.push_back(b + (size_t) r * sb);
    m_.state_save(e.seq, dst);
    ckpt_t c;
    c.pos = pos;
    c.buf = b;
    c.carry = carry && d_ && d_->carry_to_host(e.seq, (float *) (b + (size_t) m_.n_ranks() * sb));
    c.lru = ++lru_;
    auto it = std::lower_bound(e.cps.begin(), e.cps.end(), pos, [](const ckpt_t & a, int p) { return a.pos < p; });
    e.cps.insert(it, c);
    ++ck_n_;
}

// ---- admission ----

// The best cache entry for job j: idle, matchable entries; the reuse point c is the largest position <= min(LCP,
// |prompt| - 1) with a known delta-net state (a checkpoint, or the entry's current state) that the exactness rule allows.
sched::plan sched::lookup(const job & j) const {
    plan best;
    const int P = (int) j.prompt.size();
    std::vector<int> B;
    if (o_.exact) B = partition(j.splits, P, 0);
    double best_t = -1;
    for (const auto & kv : ents_) {
        const entry & e = kv.second;
        if (e.owner >= 0 || e.pinned || !e.cached) continue;
        if (o_.exact && !e.exact) continue;   // loaded from a file with other numerics (sched::load)
        const std::vector<int32_t> & t = m_.seq_tokens(e.seq);
        const size_t n = std::min(t.size(), (size_t) P);
        const int d = (int) (std::mismatch(t.begin(), t.begin() + n, j.prompt.begin()).first - t.begin());
        const int lim = std::min(d, P - 1);
        if (lim < best.c || lim <= 0) continue;
        const int np = m_.seq_len(e.seq);
        const bool offl = !m_.seq_resident(e.seq);
        int c = 0, ck = -1;
        auto ck_at = [&](int pos) {
            for (int i = (int) e.cps.size() - 1; i >= 0; --i) if (e.cps[i].pos == pos) return i;
            return -1;
        };
        if (o_.exact) {
            // positions that are boundaries of both partitions, with the same boundaries below them
            size_t k = 0;
            while (k < e.bnd.size() && k < B.size() && e.bnd[k] == B[k] && e.bnd[k] <= lim) ++k;
            for (int i = (int) k - 1; i >= 1; --i) {
                const int p = e.bnd[i];
                const bool cur = p == np && np == e.n_pf;
                const int ci = cur ? -1 : ck_at(p);
                if (cur || ci >= 0) { c = p; ck = ci; break; }
            }
        } else {
            if (np <= lim) c = np;   // the current state (decoded tokens too)
            for (int i = (int) e.cps.size() - 1; i >= 0; --i)
                if (e.cps[i].pos <= lim && e.cps[i].pos > c) { c = e.cps[i].pos; ck = i; break; }
        }
        if (c <= 0) continue;
        // larger reuse wins; ties: resident over offloaded, then the most recently used
        const bool better = c > best.c || (c == best.c && ((!offl && best.offl) || (offl == best.offl && e.t_used > best_t)));
        if (!better) continue;
        best.seq = e.seq;
        best.c = c;
        best.ck = ck;
        best.offl = offl;
        best_t = e.t_used;
    }
    return best;
}

sched::plan sched::make_plan(const job & j) const {
    plan p;
    if (j.offloaded && (j.st == job::PREFILL || j.st == job::DECODE)) {   // resume
        p.seq = j.seq;
        p.offl = true;
        p.slot = 1;
        p.pages = std::max(j.reserved_pages, m_.seq_pages(j.seq));
        return p;
    }
    if (j.cont_of >= 0) {   // continuation of a finished job's sequence
        p.seq = jobs_[j.cont_of].seq;
        if (p.seq < 0 || !ents_.count(p.seq) || ents_.at(p.seq).owner != j.id) {   // the session is gone
            p.bad = true;
            return p;
        }
        p.c = m_.seq_len(p.seq);
        p.offl = !m_.seq_resident(p.seq);
        p.slot = p.offl ? 1 : 0;
        p.pages = p.offl ? j.reserved_pages : std::max(0, j.reserved_pages - m_.seq_pages(p.seq));
        return p;
    }
    if (j.cache) p = lookup(j);
    if (p.seq < 0) {
        p.slot = 1;
        p.pages = j.reserved_pages;
        return p;
    }
    const int kept = pages_for(p.c);
    if (p.offl) {
        p.slot = 1;
        p.pages = j.reserved_pages;
    } else {
        p.pages = std::max(0, j.reserved_pages - kept);
        p.freed = std::max(0, m_.seq_pages(p.seq) - kept);
    }
    return p;
}

// make room: the least recently used idle resident entry other than `keep` goes to host RAM (offloaded LRU entries that
// are only cache are dropped first when the host budget is short); a cache entry that cannot be offloaded is dropped.
// Held sessions (keep) are only ever offloaded. false: nothing could be evicted.
bool sched::evict_idle(int keep) {
    // candidates: idle entries, and entries reserved by a continuation that is still queued (admission restores them)
    auto waiting = [&](const entry & e) { return e.owner >= 0 && jobs_[e.owner].st == job::QUEUED; };
    std::vector<std::pair<double, int>> cand;
    for (const auto & kv : ents_)
        if ((kv.second.owner < 0 || waiting(kv.second)) && kv.first != keep && m_.seq_resident(kv.first))
            cand.push_back({kv.second.t_used, kv.first});
    std::sort(cand.begin(), cand.end());
    for (const auto & [t, s] : cand) {
        (void) t;
        entry & e = ents_.at(s);
        bool done = false;
        if (m_.paged()) {
            const size_t bytes = ((size_t) m_.seq_pages(s) * m_.page_bytes() + m_.state_bytes()) * m_.n_ranks();
            // host room (offloaded sequences + checkpoints share the budget): drop offloaded cache-only entries, least
            // recently used first
            while (m_.host_used() + bytes > m_.host_budget())
                if (!drop_offloaded(keep, s)) break;
            if (m_.host_used() + bytes <= m_.host_budget()) {
                try {
                    const double t0 = now();
                    m_.seq_offload(s);   // a failed pinned allocation must not escape step()
                    done = true;
                    ++evictions;
                    if (e.done_job >= 0) { jobs_[e.done_job].offloaded = true; jobs_[e.done_job].t_offload += now() - t0; }
                } catch (const std::exception &) {
                }
            }
        }
        if (!done && !e.pinned && e.owner < 0) {
            drop_entry(s);
            ++drops;
            done = true;
        }
        if (done) return true;
    }
    return false;
}

bool sched::drop_offloaded(int keep1, int keep2) {
    int v = -1;
    for (const auto & kv : ents_)
        if (kv.second.owner < 0 && !kv.second.pinned && kv.first != keep1 && kv.first != keep2 &&
            !m_.seq_resident(kv.first) && (v < 0 || kv.second.t_used < ents_.at(v).t_used))
            v = kv.first;
    if (v < 0) return false;
    drop_entry(v);
    ++drops;
    return true;
}

// FIFO admission. The page check counts what every running job may still grow into (its reservation minus the pages it
// holds), so resumed and continued jobs cannot starve running ones.
void sched::admit() {
    // nothing changed since the last attempt (no submit / finish / release / resume): the queue front still waits.
    // Skipping the retry keeps the cache lookup (an LCP over every entry) off the per-step path.
    if (!adm_dirty_) return;
    adm_dirty_ = false;
    for (int guard = 0; !queue_.empty() && guard < 100000; ++guard) {
        job & j = jobs_[queue_.front()];
        int active = 0, growth = 0;
        for (int id : live_) {
            const job & a = jobs_[id];
            if ((a.st != job::PREFILL && a.st != job::DECODE) || a.offloaded) continue;
            ++active;
            growth += std::max(0, a.reserved_pages - m_.seq_pages(a.seq));
        }
        if (active >= o_.max_active) return;
        const plan p = make_plan(j);
        if (p.bad) {
            queue_.pop_front();
            fail(j, "the session it continues is gone");
            continue;
        }
        const bool fits = m_.free_slots() >= p.slot && (!m_.paged() || m_.free_pages() + p.freed - growth >= p.pages);
        if (!fits) {
            if (evict_idle(p.seq)) continue;
            if (active == 0) {   // nothing will free up: it can never run
                queue_.pop_front();
                fail(j, "does not fit: needs " + std::to_string(p.pages) + " KV pages and " + std::to_string(p.slot) +
                            " slot(s), " + std::to_string(m_.free_pages()) + " pages / " + std::to_string(m_.free_slots()) +
                            " slots free (held sessions that cannot be offloaded keep the rest)");
                continue;
            }
            return;
        }
        queue_.pop_front();
        start(j, p);
    }
}

void sched::start(job & j, const plan & p) {
    const double t = now();
    if (j.offloaded && (j.st == job::PREFILL || j.st == job::DECODE)) {   // resume
        m_.seq_restore(j.seq);
        j.offloaded = false;
        j.t_restore += now() - t;
        return;
    }
    if (j.cont_of >= 0) {   // continuation
        job & o = jobs_[j.cont_of];
        entry & e = ents_.at(p.seq);
        if (p.offl) {
            m_.seq_restore(p.seq);
            o.offloaded = false;
            o.t_restore += now() - t;
        }
        e.owner = j.id;
        e.done_job = -1;
        e.pinned = j.keep_seq;
        e.cached = j.cache;
        o.seq = -1;   // the sequence moves to the new job
        j.seq = p.seq;
        if (j.append) {
            // new input: prefill it from the sequence's length (decoded tokens before it: the partition from there; the
            // entry's prefill boundaries stay where they are). The draft head's rows continue with its carry when the
            // carry belongs to this position, else from a zero h (only the drafts' quality can change).
            j.dkv = d_ && (j.mtp || (o_.draft_kv && (j.cache || j.keep_seq)));
            if (d_ && e.carry_at != p.c) {
                d_->carry_drop(p.seq);
                e.carry_at = -1;
            }
            j.n_cached = p.c;
            j.t_restore += p.offl ? now() - t : 0.0;
            // a checkpoint at the turn's end, so cancel() during this prefill can give the session back to job o
            take_ckpt(e, p.c, e.carry_at == p.c);
            j.st = job::PREFILL;
            plan_chunks(j, e, p.c, false);
            j.t_admit = now();
            return;
        }
        j.st = job::DECODE;
        j.last = o.out.back();
        j.N = m_.seq_len(p.seq);   // the position of `last` (every earlier token is in the sequence)
        j.t_admit = j.t_first = now();
        j.t_restore += p.offl ? now() - t : 0.0;
        j.draft_ok = j.mtp && e.n_draft >= j.N;
        // MTP: the turn ended with the draft head's carry for `last` (h of the position before it) saved, so the first
        // step drafts already (a run of `last` at N, then the chain)
        if (j.mtp && d_ && e.carry_at == j.N && d_->has_carry(j.seq) && !gen_done(j) && j.n_gen > 1) {
            const double c0 = now();
            m_.seq_select(j.seq);
            d_->carry_load(j.seq);
            const int32_t d0 = d_->run({j.last}, j.N, tp_mtp::H_READY);
            ++j.draft_runs;
            j.t_catchup += now() - c0;
            chain(j, d0);
        }
        return;
    }
    // new job: a fresh sequence, or a cache entry truncated to the reuse point
    // the draft head's KV rows: MTP jobs, and (draft_kv) every job whose sequence stays in the cache
    j.dkv = d_ && (j.mtp || (o_.draft_kv && j.cache));
    entry * e = nullptr;
    if (p.seq < 0) {
        const int s = m_.seq_create();
        entry ne;
        ne.seq = s;
        ents_[s] = ne;
        e = &ents_.at(s);
        if (d_) d_->carry_drop(s);   // a stale carry of an earlier sequence with this id
        e->carry_at = 0;             // no carry = zero h for position 0 (carry_load zeroes)
    } else {
        e = &ents_.at(p.seq);
        const int np = m_.seq_len(p.seq);
        const bool in_place = p.c == np;   // the state in the slot / host copy is the state at c
        ck_trim(*e, p.c);
        if (p.offl) m_.seq_restore_prefix(p.seq, p.c);   // only the pages of 0..c-1 (+ the state when it is c's)
        else m_.seq_truncate(p.seq, p.c);
        if (!in_place) {
            const ckpt_t & c = e->cps.at(p.ck);
            const size_t sb = m_.state_bytes();
            std::vector<const char *> src;
            for (int r = 0; r < m_.n_ranks(); ++r) src.push_back(c.buf + (size_t) r * sb);
            m_.state_load(p.seq, src);
            e->cps.at(p.ck).lru = ++lru_;
            if (d_) {
                if (c.carry) { d_->carry_from_host(p.seq, (const float *) (c.buf + (size_t) m_.n_ranks() * sb)); e->carry_at = p.c; }
                else { d_->carry_drop(p.seq); e->carry_at = -1; }
            }
        } else if (e->carry_at != p.c && d_) {
            d_->carry_drop(p.seq);
            e->carry_at = -1;
        }
        // the boundaries only ever record prefill chunks as they were computed: cut back to c; when c is not one of
        // them (non-exact reuse after decoded tokens, or at a state that is not a chunk end) they stay frozen at the
        // last one below c (prefill_chunk extends them only from n_pf), so an exact lookup never compares a position
        // whose KV came from decode steps
        while (e->bnd.size() > 1 && e->bnd.back() > p.c) e->bnd.pop_back();
        e->n_pf = e->bnd.back();
        e->n_draft = std::min(e->n_draft, p.c);
        if (e->done_job >= 0 && jobs_[e->done_job].seq == p.seq) {   // the entry moves on to this job
            jobs_[e->done_job].seq = -1;
            jobs_[e->done_job].offloaded = false;
        }
        e->done_job = -1;
        j.n_cached = p.c;
        j.reuse_offloaded = p.offl;
        j.t_reuse = now() - t;
    }
    e->owner = j.id;
    e->pinned = j.keep_seq;
    e->cached = j.cache;
    j.seq = e->seq;
    j.st = job::PREFILL;
    plan_chunks(j, *e, p.c, true);
    j.t_admit = now();
}

// the chunks of prompt[c..] and where the prefill takes checkpoints. cold_tail: the tail of the request's cold
// partition when c is one of its boundaries, so every chunk from c on is the cold run's (a partition
// restarted at c drops a split position within min_split after c, e.g. cold 8192 | 8230 | end vs restarted
// 8192 | end); otherwise (non-exact reuse at another position, new input after decoded tokens) the partition from c.
void sched::plan_chunks(job & j, const entry & e, int c, bool cold_tail) {
    std::vector<bool> ex;
    const int P = (int) j.prompt.size();
    std::vector<int> b;
    if (cold_tail) {
        b = partition(j.splits, P, 0, &ex);
        const auto it = std::lower_bound(b.begin(), b.end(), c);
        if (it != b.end() && *it == c) {
            const size_t i0 = (size_t) (it - b.begin());
            b.erase(b.begin(), b.begin() + i0);
            ex.erase(ex.begin(), ex.begin() + i0);
        } else {
            b.clear();
        }
    }
    if (b.empty()) b = partition(j.splits, P, c, &ex);
    j.chunks.clear();
    for (size_t i = 1; i < b.size(); ++i) j.chunks.push_back(b[i] - b[i - 1]);
    j.ci = 0;
    j.c0 = c;
    j.n_prefill = P - c;
    // checkpoint positions: the last ckpt_tail split positions, every ckpt_every positions, the prompt end
    std::vector<int> tail;
    for (size_t i = 1; i < b.size(); ++i) if (ex[i]) tail.push_back(b[i]);
    if ((int) tail.size() > o_.ckpt_tail) tail.erase(tail.begin(), tail.end() - std::max(0, o_.ckpt_tail));
    int last = e.cps.empty() ? 0 : e.cps.back().pos;
    j.ck.clear();
    for (size_t i = 1; i < b.size() && j.cache; ++i) {   // checkpoints only serve the cache
        const bool want = b[i] == P || std::binary_search(tail.begin(), tail.end(), b[i]) ||
                          (o_.ckpt_every > 0 && b[i] - last >= o_.ckpt_every);
        if (want) { j.ck.push_back(b[i]); last = b[i]; }
    }
}

// ---- prefill / decode ----

bool sched::prefill_chunk(job & j) {
    const double a = now();
    const int c = j.chunks[j.ci];
    const std::vector<int32_t> ch(j.prompt.begin() + j.c0, j.prompt.begin() + j.c0 + c);
    m_.seq_select(j.seq);
    j.L = m_.forward_n(ch, 1);
    const double b = now();
    j.t_prefill += b - a;
    entry & e = ents_.at(j.seq);
    if (j.dkv) {
        // the draft head's KV-only rows of this chunk; h input row 0 carries over from the previous chunk (zero for the
        // first, the checkpoint's after a cache hit) - saved per sequence, other jobs' draft runs use the buffer between
        d_->carry_load(j.seq);
        d_->catchup_kv(ch, j.c0);
        d_->carry_save(j.seq);
        j.t_catchup += now() - b;
        if (e.n_draft == j.c0) e.n_draft = j.c0 + c;
        e.carry_at = j.c0 + c;
    } else {
        e.carry_at = -1;
    }
    // the entry's prefill boundaries grow only while the prefill continues them (see start())
    if (e.n_pf == j.c0) {
        e.bnd.push_back(j.c0 + c);
        e.n_pf = j.c0 + c;
    }
    j.c0 += c;
    ++j.ci;
    if (std::binary_search(j.ck.begin(), j.ck.end(), j.c0)) {
        const double t = now();
        take_ckpt(e, j.c0, j.dkv);
        j.t_ckpt += now() - t;
        ++j.ckpts;
    }
    t_prefill += now() - a;
    if (j.ci < j.chunks.size()) return false;
    // prompt done: the first generated token, and with MTP the first draft-head run (its full row) + the drafts
    j.last = argmax(j.L.data(), m_.vocab());
    if (j.trace) j.trace->push_back(j.L);
    if (j.force) {
        j.own.push_back(j.last);
        if (!j.force->empty()) j.last = (*j.force)[0];
    }
    j.L.clear();
    j.L.shrink_to_fit();
    j.out.push_back(j.last);
    if (is_stop(j, j.last)) j.stopped = true;
    j.N = (int) j.prompt.size();
    j.t_first = now();
    j.draft_ok = j.mtp && e.n_draft >= j.N;   // the draft rows below `last` are all valid (its own comes with its run)
    if (j.mtp && !gen_done(j)) {
        const double c0 = now();
        d_->carry_load(j.seq);
        const int32_t d0 = d_->run({j.last}, j.N, tp_mtp::H_READY);
        ++j.draft_runs;
        j.t_catchup += now() - c0;
        chain(j, d0);
    }
    j.st = job::DECODE;
    if (gen_done(j)) finish(j);
    return true;
}

// drafts per job for the next step: the draft row budget (draft_rows) shared by the decoding jobs, at least the first
// draft (it comes with the catch-up run for free), at most k
int sched::k_eff() const {
    int n = 0;
    for (int id : live_) if (jobs_[id].st == job::DECODE && !jobs_[id].offloaded) ++n;
    n = std::max(1, n);
    const int share = (std::max(0, o_.draft_rows - n) + n - 1) / n;
    return std::max(1, std::min(o_.mo.k, share));
}

// generate_mtp's draft policy (tp_mtp.cpp): chain up to k drafts; adaptive (tau > 0): draft j is kept while the product
// of the draft head's top-1 probabilities stays >= tau (at least k_min drafts)
void sched::chain(job & j, int32_t d0) {
    const mtp_opts & o = o_.mo;
    const bool adaptive = o.tau > 0.0f;
    const int k = k_eff();
    const int k_min = adaptive ? std::max(0, std::min(o.k_min, k)) : k;
    j.drafts.clear();
    double cum = d_->last_p();
    if (0 >= k_min && cum < (o.tau1 > 0 ? o.tau1 : o.tau)) return;
    j.drafts.push_back(d0);
    for (int jj = 1; jj < k; ++jj) {
        const double a = now();
        const int32_t dj = d_->run({j.drafts.back()}, j.N + jj, tp_mtp::H_SELF);
        t_drafts += now() - a;
        ++j.draft_runs;
        cum *= d_->last_p();
        if (jj >= k_min && cum < o.tau) break;
        j.drafts.push_back(dj);
    }
    apply_oracle(j);
}

void sched::apply_oracle(job & j) {
    if (!j.oracle) return;
    for (size_t i = 0; i < j.drafts.size(); ++i) {
        const size_t at = j.out.size() + i;
        if (at < j.oracle->size()) j.drafts[i] = (*j.oracle)[at];
    }
    if (!j.drafts.empty() && j.steps % 3 == 2) j.drafts.back() ^= 1;
}

// The draft phase of several MTP jobs at once (tp_mtp::run_multi): one run for every job's catch-up rows (accepted
// drafts + the new token, h = its verify rows of the main model), then the chained drafts of the jobs still drafting,
// one row each, per round - the policy of chain() per job, so each job gets the drafts it would get alone.
// mi: indices into dec / acc / row0 / na of the jobs.
void sched::draft_batched(const std::vector<int> & dec, const std::vector<int> & mi,
                          const std::vector<std::vector<int32_t>> & acc, const std::vector<int> & row0,
                          const std::vector<int> & na) {
    const mtp_opts & o = o_.mo;
    const bool adaptive = o.tau > 0.0f;
    const int k = k_eff();
    const int k_min = adaptive ? std::max(0, std::min(o.k_min, k)) : k;
    double a = now();
    std::vector<tp_model::seq_rows> rows;
    std::vector<int> p0s;
    int r = 0;
    for (int i : mi) {
        job & j = jobs_[dec[i]];
        d_->h_from_main(r, row0[i], (int) acc[i].size());
        if (j.keep_seq || j.cache) d_->carry_save_row(j.seq, r + (int) acc[i].size() - 1);   // see decode_step
        rows.push_back({j.seq, acc[i]});
        p0s.push_back(j.N + 1);
        r += (int) acc[i].size();
    }
    std::vector<int32_t> picks = d_->run_multi(rows, p0s);
    t_drafts += now() - a;
    struct cs { int i; double cum; bool active, single; int midx; };
    std::vector<cs> st;
    for (size_t q = 0; q < mi.size(); ++q) {
        job & j = jobs_[dec[mi[q]]];
        ++j.draft_runs;
        j.N += na[mi[q]] + 1;
        j.last = acc[mi[q]].back();
        if (j.keep_seq || j.cache) {
            entry & e = ents_.at(j.seq);
            e.carry_at = j.N;
            if (j.draft_ok) e.n_draft = j.N;
        }
        j.drafts.clear();
        const double cum = d_->multi_p((int) q);
        if (0 >= k_min && cum < (o.tau1 > 0 ? o.tau1 : o.tau)) { st.push_back({mi[q], cum, false, false, (int) q}); continue; }
        j.drafts.push_back(picks[q]);
        st.push_back({mi[q], cum, k > 1, false, (int) q});
    }
    for (int jj = 1; jj < k; ++jj) {
        std::vector<int> act;
        for (size_t q = 0; q < st.size(); ++q) if (st[q].active) act.push_back((int) q);
        if (act.empty()) break;
        a = now();
        if (act.size() == 1) {
            // one job left: the single-sequence chained run (h = its normed hidden: from the last run_multi, or after a
            // single run its own h input row 0)
            cs & s = st[act[0]];
            job & j = jobs_[dec[s.i]];
            m_.seq_select(j.seq);
            if (!s.single) d_->h_from_multi(0, s.midx);
            const int32_t dj = d_->run({j.drafts.back()}, j.N + jj, s.single ? tp_mtp::H_SELF : tp_mtp::H_READY);
            s.single = true;
            ++j.draft_runs;
            s.cum *= d_->last_p();
            if (jj >= k_min && s.cum < o.tau) s.active = false;
            else j.drafts.push_back(dj);
        } else {
            std::vector<tp_model::seq_rows> cr;
            std::vector<int> cp;
            for (size_t m = 0; m < act.size(); ++m) {
                cs & s = st[act[m]];
                job & j = jobs_[dec[s.i]];
                d_->h_from_multi((int) m, s.midx);
                cr.push_back({j.seq, {j.drafts.back()}});
                cp.push_back(j.N + jj);
            }
            const std::vector<int32_t> pk = d_->run_multi(cr, cp);
            for (size_t m = 0; m < act.size(); ++m) {
                cs & s = st[act[m]];
                job & j = jobs_[dec[s.i]];
                s.midx = (int) m;
                ++j.draft_runs;
                s.cum *= d_->multi_p((int) m);
                if (jj >= k_min && s.cum < o.tau) s.active = false;
                else j.drafts.push_back(pk[m]);
            }
        }
        t_drafts += now() - a;
    }
    for (const cs & s : st) apply_oracle(jobs_[dec[s.i]]);
}

void sched::finish(job & j) {
    adm_dirty_ = true;
    unlive(j.id);
    j.st = job::DONE;
    j.t_done = now();
    entry & e = ents_.at(j.seq);
    e.owner = -1;
    e.t_used = j.t_done;
    if (!j.cache && !j.keep_seq) {
        drop_entry(j.seq);
        j.seq = -1;
        return;
    }
    e.done_job = j.id;
    e.pinned = j.keep_seq;
    e.cached = j.cache;
}

void sched::set_trace(int id, std::vector<std::vector<float>> * trace, const std::vector<int32_t> * force) {
    job & j = jobs_.at(id);
    if (j.mtp && (trace || force)) throw std::runtime_error("sched: trace / force need a plain job");
    j.trace = trace;
    j.force = force;
}

void sched::trace_step(job & j) {
    const double a = now();
    if (j.t_dec0 < 0) j.t_dec0 = a;
    m_.seq_select(j.seq);
    std::vector<float> L = m_.forward_n({j.last}, 1);
    int32_t t = argmax(L.data(), m_.vocab());
    if (j.trace) j.trace->push_back(std::move(L));
    if (j.force) {
        j.own.push_back(t);
        if (j.out.size() < j.force->size()) t = (*j.force)[j.out.size()];
    }
    ++j.N;
    j.last = t;
    j.out.push_back(t);
    ++j.steps;
    if (is_stop(j, t)) j.stopped = true;
    t_decode += now() - a;
    if (gen_done(j)) finish(j);
}

void sched::decode_step() {
    std::vector<int> dec;
    for (int id : live_) if (jobs_[id].st == job::DECODE && !jobs_[id].offloaded) dec.push_back(id);
    // test hook: traced / forced jobs step alone (plain T = 1 forwards with their logits)
    for (size_t i = 0; i < dec.size();) {
        job & j = jobs_[dec[i]];
        if (j.trace || j.force) { trace_step(j); dec.erase(dec.begin() + i); } else ++i;
    }
    if (dec.empty()) return;
    // at most max_rows jobs per step, round robin when there are more
    if ((int) dec.size() > o_.max_rows) {
        std::rotate(dec.begin(), dec.begin() + (rr_ % dec.size()), dec.end());
        dec.resize(o_.max_rows);
    }
    ++rr_;
    // rows: every job's last token, then drafts one at a time round robin while the draft row budget (draft_rows: the
    // GEMVs read the weights once per 4 columns), the row limit and the snapshot slots allow
    const int n = (int) dec.size();
    std::vector<int> nd(n, 0);
    int rows = n, snaps = 0;
    const int rows_cap = std::min(o_.max_rows, std::max(n, o_.draft_rows));
    for (bool more = true; more;) {
        more = false;
        for (int i = (rr_ + 0) % n, c = 0; c < n; ++c, i = (i + 1) % n) {
            const job & j = jobs_[dec[i]];
            if (nd[i] < (int) j.drafts.size() && nd[i] < 3 && rows < rows_cap && snaps < m_.n_snap()) {
                ++nd[i]; ++rows; ++snaps; more = true;
            }
        }
    }
    std::vector<tp_model::seq_rows> b(n);
    const double t_step = now();
    for (int i = 0; i < n; ++i) {
        job & j = jobs_[dec[i]];
        if (j.t_dec0 < 0) j.t_dec0 = t_step;
        for (int32_t dt : j.drafts) { j.draft_hash ^= (uint32_t) dt; j.draft_hash *= 1099511628211ull; }
        j.draft_hash ^= 0xffu; j.draft_hash *= 1099511628211ull;
        b[i].seq = j.seq;
        b[i].tokens.push_back(j.last);
        b[i].tokens.insert(b[i].tokens.end(), j.drafts.begin(), j.drafts.begin() + nd[i]);
    }
    double a = now();
    const std::vector<int32_t> g = m_.forward_batch_ids(b, snaps > 0);
    t_decode += now() - a;
    ++decode_steps;
    rows_hist[rows]++;
    // accept, emit, rollback
    std::vector<tp_model::rb_req> rb;
    std::vector<std::vector<int32_t>> acc(n);
    std::vector<int> row0(n), na(n), qv(n);
    int r = 0;
    for (int i = 0; i < n; ++i) {
        job & j = jobs_[dec[i]];
        const int T = (int) b[i].tokens.size();
        row0[i] = r;
        int ac = 0;
        while (ac < T - 1 && g[r + ac] == b[i].tokens[1 + ac]) ++ac;
        const int32_t nt = g[r + ac];
        na[i] = ac;
        if (j.steps == 0) j.first_rows = T;
        j.steps++;
        j.drafted += T - 1;
        j.accepted += ac;
        acc[i].assign(b[i].tokens.begin() + 1, b[i].tokens.begin() + 1 + ac);
        acc[i].push_back(nt);
        // keep the state consistent with the returned tokens: on the last step emit only what is still needed (n_gen,
        // or up to and including a stop token)
        int q = std::min((int) acc[i].size(), j.n_gen - (int) j.out.size());
        for (int k = 0; k < q; ++k)
            if (is_stop(j, acc[i][k])) { q = k + 1; j.stopped = true; break; }
        qv[i] = q;
        if (q < T) rb.push_back({j.seq, T, q});
        for (int k = 0; k < q; ++k) j.out.push_back(acc[i][k]);
        r += T;
    }
    a = now();
    m_.rollback_batch(rb);
    t_rollback += now() - a;
    // per job: done, or the next drafts (MTP) / next input (plain). Several MTP jobs: one batched draft phase
    // (P100E_DRAFT_BATCH=0: one job after the other)
    static const bool batch_drafts = !getenv("P100E_DRAFT_BATCH") || atoi(getenv("P100E_DRAFT_BATCH")) != 0;
    std::vector<int> mi;
    for (int i = 0; i < n; ++i) {
        job & j = jobs_[dec[i]];
        if (gen_done(j)) continue;
        if (j.mtp) mi.push_back(i);
    }
    const bool batched = batch_drafts && mi.size() >= 2 && d_ && d_->fused();
    if (batched) draft_batched(dec, mi, acc, row0, na);
    for (int i = 0; i < n; ++i) {
        job & j = jobs_[dec[i]];
        if (gen_done(j)) {
            // a kept MTP sequence: the draft head's rows of the emitted tokens (positions N+1..N+q) and its carry for the
            // last token (h of position N+q-1), so the next turn (submit_continue) drafts from its first step and its draft
            // KV has no holes. Only the draft head's own KV rows and carry change.
            if (j.mtp && d_ && (j.keep_seq || j.cache)) {
                const double c = now();
                const int q = qv[i];
                m_.seq_select(j.seq);
                d_->h_from_main(0, row0[i], q);
                d_->run(std::vector<int32_t>(acc[i].begin(), acc[i].begin() + q), j.N + 1, tp_mtp::H_READY);
                ++j.draft_runs;
                d_->h_from_main(0, row0[i] + q - 1, 1);
                d_->carry_save(j.seq);
                entry & e = ents_.at(j.seq);
                if (j.draft_ok) e.n_draft = m_.seq_len(j.seq);   // every decoded position has its draft row
                e.carry_at = m_.seq_len(j.seq);
                t_drafts += now() - c;
            }
            finish(j);
            continue;
        }
        const int32_t nt = acc[i].back();
        if (!j.mtp) { j.last = nt; ++j.N; continue; }
        if (batched) continue;
        // draft head: catch-up rows for the accepted drafts + the full row of nt (positions N+1..N+a+1), h = the main
        // model's hn rows of this job in the step just run
        const double c = now();
        m_.seq_select(j.seq);
        d_->h_from_main(0, row0[i], (int) acc[i].size());
        // the carry of the new last token (h of the position before it): the entry stays consistent at every step
        // boundary, so cancel() can end the turn here and the next one drafts from its first step
        if (j.keep_seq || j.cache) d_->carry_save_row(j.seq, (int) acc[i].size() - 1);
        const int32_t d0 = d_->run(acc[i], j.N + 1, tp_mtp::H_READY);
        ++j.draft_runs;
        t_drafts += now() - c;
        j.N += na[i] + 1;
        j.last = nt;
        if (j.keep_seq || j.cache) {
            entry & e = ents_.at(j.seq);
            e.carry_at = j.N;
            if (j.draft_ok) e.n_draft = j.N;
        }
        chain(j, d0);
    }
}

bool sched::step() {
    admit();
    bool work = false;
    bool decoding = false;
    for (int id : live_) if (jobs_[id].st == job::DECODE && !jobs_[id].offloaded) decoding = true;
    // prefill: one chunk of the oldest job still prefilling. While jobs are decoding, its turn comes after
    // decode_per_chunk decode steps and once they have had their decode_share of the time (dec_owed_ <= 0).
    // prefill_first: every admitted job's whole prompt, and no decode step while any job is still prefilling.
    const bool chunk_turn = o_.prefill_first || !decoding ||
                            (dec_since_chunk_ >= std::max(1, o_.decode_per_chunk) && dec_owed_ <= 0);
    const std::vector<int> live = live_;   // prefill_chunk may finish a job (unlive)
    for (int id : live) {
        job & j = jobs_[id];
        if (j.st != job::PREFILL || j.offloaded) continue;
        work = true;
        if (!chunk_turn) break;
        if (o_.prefill_first) { while (j.st == job::PREFILL) prefill_chunk(j); continue; }
        const double a = now();
        prefill_chunk(j);
        const double sh = o_.decode_share;
        dec_owed_ = sh > 0 ? (now() - a) * sh / (1.0 - sh) : 0.0;
        dec_since_chunk_ = 0;
        break;
    }
    if (o_.prefill_first) {
        admit();
        for (int id : live_) if (jobs_[id].st == job::PREFILL && !jobs_[id].offloaded) return true;
    }
    // decode: one batched step over every decoding job
    bool dec = false;
    for (int id : live_) if (jobs_[id].st == job::DECODE && !jobs_[id].offloaded) dec = true;
    if (dec) {
        const double a = now();
        decode_step();
        dec_owed_ -= now() - a;
        t_dec_wall += now() - a;
        work = true;
        ++dec_since_chunk_;
    }
    return work;
}

void sched::run() {
    while (step()) {}
}

void sched::preempt(int id) {
    job & j = jobs_.at(id);
    if (j.st != job::PREFILL && j.st != job::DECODE) throw std::runtime_error("sched: preempt of a job that is not running");
    if (j.offloaded) return;
    const double a = now();
    m_.seq_offload(j.seq);
    j.offloaded = true;
    j.t_offload += now() - a;
    adm_dirty_ = true;   // its slot and pages are free now
}

void sched::resume(int id) {
    job & j = jobs_.at(id);
    if (!j.offloaded) return;
    if (std::find(queue_.begin(), queue_.end(), id) != queue_.end()) return;
    queue_.push_front(id);   // ahead of new jobs; admitted when its pages and a slot are free
    adm_dirty_ = true;
}

// ---- sessions on disk ----
// file: "P100EPC2" | i32 version 2 | i32 last token, n_pf, n_draft, carry_at, has_carry, keep, n_bnd, exact |
//       u64 numerics hash (tp_model::numerics_hash of the writer) | bnd[n_bnd] | carry (n_embd floats when has_carry) |
//       tp_model::seq_write (its own fingerprint: model content + KV layout) | i32 n_ckpt | per checkpoint: i32 pos, carry
//       flag + ckpt_bytes() bytes ([state GPU 0][state GPU 1][carry])
static const char kSessMagic[8] = {'P', '1', '0', '0', 'E', 'P', 'C', '2'};

void sched::save(int done_id, const std::string & path, int max_ckpts) {
    const job & j = jobs_.at(done_id);
    if (j.st != job::DONE || j.seq < 0 || !ents_.count(j.seq) || ents_.at(j.seq).done_job != done_id)
        throw std::runtime_error("sched::save: needs a finished job that still owns its sequence");
    const entry & e = ents_.at(j.seq);
    const std::string tmp = path + ".tmp";
    FILE * f = fopen(tmp.c_str(), "wb");
    if (!f) throw std::runtime_error("sched::save: cannot open " + tmp);
    try {
        auto wr = [&](const void * p, size_t n) {
            if (n && fwrite(p, 1, n, f) != n) throw std::runtime_error("sched::save: write failed (disk full?)");
        };
        std::vector<float> carry(m_.n_embd(), 0.0f);
        const int has_carry = d_ && d_->carry_to_host(j.seq, carry.data()) ? 1 : 0;
        const int32_t ver = 2;
        const int32_t h[8] = {j.out.empty() ? -1 : j.out.back(), e.n_pf, e.n_draft, e.carry_at, has_carry,
                              j.keep_seq ? 1 : 0, (int32_t) e.bnd.size(), e.exact ? 1 : 0};
        const uint64_t nh = m_.numerics_hash();
        wr(kSessMagic, 8);
        wr(&ver, 4);
        wr(h, sizeof h);
        wr(&nh, 8);
        wr(e.bnd.data(), e.bnd.size() * 4);
        if (has_carry) wr(carry.data(), carry.size() * 4);
        m_.seq_write(j.seq, f);
        const int ncp = (int) e.cps.size();
        const int32_t nck = max_ckpts < 0 ? ncp : std::min(ncp, max_ckpts);
        wr(&nck, 4);
        for (int i = ncp - nck; i < ncp; ++i) {   // the newest (highest positions)
            const int32_t ph[2] = {e.cps[i].pos, e.cps[i].carry ? 1 : 0};
            wr(ph, sizeof ph);
            wr(e.cps[i].buf, ck_bytes_);
        }
        // durable before the rename: a crash must leave either the old file or the complete new one
        if (fflush(f) != 0) throw std::runtime_error("sched::save: flush failed");
#ifdef _WIN32
        if (_commit(_fileno(f)) != 0) throw std::runtime_error("sched::save: commit to disk failed");
#else
        if (fsync(fileno(f)) != 0) throw std::runtime_error("sched::save: fsync failed");
#endif
        if (fclose(f) != 0) { f = nullptr; throw std::runtime_error("sched::save: close failed"); }
        f = nullptr;
    } catch (...) {
        if (f) fclose(f);
        remove(tmp.c_str());
        throw;
    }
    // atomic replace: the old file stays until the new one takes its name
#ifdef _WIN32
    if (!MoveFileExA(tmp.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        const unsigned long err = GetLastError();
        remove(tmp.c_str());
        throw std::runtime_error("sched::save: replacing " + path + " failed (Windows error " + std::to_string(err) + ")");
    }
#else
    if (rename(tmp.c_str(), path.c_str()) != 0) {
        remove(tmp.c_str());
        throw std::runtime_error("sched::save: rename to " + path + " failed");
    }
#endif
}

int sched::load(const std::string & path) {
    adm_dirty_ = true;
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("sched::load: cannot open " + path);
    int seq = -1;
    try {
        auto rd = [&](void * p, size_t n) {
            if (n && fread(p, 1, n, f) != n) throw std::runtime_error("sched::load: short file");
        };
        auto bad = [&](const std::string & what) { throw std::runtime_error("sched::load: bad header (" + what + ")"); };
        char mg[8];
        rd(mg, 8);
        if (!memcmp(mg, "P100EPC1", 8)) throw std::runtime_error("sched::load: a version 1 session file: not loadable");
        if (memcmp(mg, kSessMagic, 8)) throw std::runtime_error("sched::load: not a p100e session file");
        int32_t ver = 0, h[8];
        uint64_t nh = 0;
        rd(&ver, 4);
        if (ver != 2) throw std::runtime_error("sched::load: unsupported session file version");
        rd(h, sizeof h);
        rd(&nh, 8);
        const int32_t last = h[0], n_pf = h[1], n_draft = h[2], carry_at = h[3], has_carry = h[4], keep = h[5],
                      n_bnd = h[6], exact = h[7];
        if (n_bnd < 1 || n_bnd > (1 << 22)) bad("boundary count");
        if (last < -1 || last >= m_.vocab()) bad("last token");
        if ((has_carry | keep | exact) & ~1) bad("flags");
        std::vector<int> bnd(n_bnd);
        rd(bnd.data(), bnd.size() * 4);
        // the boundaries: 0, strictly ascending, ending at n_pf
        if (bnd[0] != 0 || bnd.back() != n_pf) bad("boundaries must run from 0 to n_pf");
        for (size_t i = 1; i < bnd.size(); ++i) if (bnd[i] <= bnd[i - 1]) bad("boundaries not ascending");
        std::vector<float> carry(m_.n_embd(), 0.0f);
        if (has_carry) rd(carry.data(), carry.size() * 4);
        seq = m_.seq_read(f);   // checks the model / KV fingerprint
        const int n_past = m_.seq_len(seq);
        if (n_pf > n_past || n_draft < 0 || n_draft > n_past || carry_at < -1 || carry_at > n_past)
            bad("n_pf / n_draft / carry_at beyond the sequence's " + std::to_string(n_past) + " positions");
        if (last < 0 && n_past == 0) bad("empty session");
        entry e;
        e.seq = seq;
        e.bnd = bnd;
        e.n_pf = n_pf;
        e.n_draft = n_draft;
        e.carry_at = (has_carry && d_) ? carry_at : -1;
        e.pinned = keep != 0;
        e.cached = true;
        // exact reuse only from a file whose prefill numerics are this process's (same build and switches)
        e.exact = exact != 0 && nh == m_.numerics_hash();
        if (exact && !e.exact)
            fprintf(stderr, "sched::load: %s was written with other numerics (build / GPU / P100E_* switches; here: %s): "
                            "its prefix is reused without the exactness guarantee\n", path.c_str(),
                    m_.numerics_env().empty() ? "none set" : m_.numerics_env().c_str());
        e.t_used = now();
        ents_[seq] = e;
        if (d_) {
            if (has_carry) d_->carry_from_host(seq, carry.data());
            else d_->carry_drop(seq);
        }
        int32_t nck = 0;
        rd(&nck, 4);
        ck_keep_ = seq;   // making room for its checkpoints must not drop the entry being loaded
        struct unkeep { int & k; ~unkeep() { k = -1; } } uk{ck_keep_};
        if (nck < 0 || nck > (1 << 16)) bad("checkpoint count");
        int prev = 0;
        for (int i = 0; i < nck; ++i) {
            int32_t ph[2];
            rd(ph, sizeof ph);
            if (ph[0] <= prev || ph[0] > n_past || (ph[1] & ~1)) bad("checkpoint position / flag");
            prev = ph[0];
            char * b = ck_get();
            if (!b) {   // over the checkpoint budget: skip the rest
                std::vector<char> skip(ck_bytes_);
                rd(skip.data(), ck_bytes_);
                continue;
            }
            try { rd(b, ck_bytes_); } catch (...) { ck_put(b); throw; }
            ckpt_t c;
            c.pos = ph[0];
            c.buf = b;
            c.carry = ph[1] != 0 && d_;
            c.lru = ++lru_;
            entry & ee = ents_.at(seq);
            auto it = std::lower_bound(ee.cps.begin(), ee.cps.end(), c.pos, [](const ckpt_t & a, int p) { return a.pos < p; });
            ee.cps.insert(it, c);
            ++ck_n_;
        }
        fclose(f);
        f = nullptr;
        job j;
        j.id = (int) jobs_.size();
        j.st = job::DONE;
        j.seq = seq;
        j.keep_seq = e.pinned;
        j.cache = true;
        if (last >= 0) j.out.push_back(last);
        j.t_submit = j.t_admit = j.t_first = j.t_done = now();
        ents_.at(seq).done_job = j.id;
        jobs_.push_back(j);
        return j.id;
    } catch (...) {
        if (f) fclose(f);
        if (seq >= 0) {
            if (ents_.count(seq)) drop_entry(seq);
            else m_.seq_free(seq);
        }
        throw;
    }
}

} // namespace p100e
