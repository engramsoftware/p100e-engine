// Multi-session scheduler with the prompt cache: greedy generation of several jobs on one
// tp_model, the engine-side core of the server's slot loop (llama-server -np N). Engine only, no HTTP: the server layer
// adds sampling, templates and streaming on top of this API.
//
// One step():
//   0. (decode fairness) while jobs decode, a prefill chunk only runs once the decode steps since the previous chunk
//      took decode_share / (1 - decode_share) of that chunk's time (default share 0.5: half the wall time each).
//   1. admission (FIFO): a queued job starts when a sequence slot and its whole page reservation (prompt + n_gen + drafts,
//      on top of what the running jobs may still grow into) are free; idle sequences (finished jobs kept for the cache or
//      as sessions) are offloaded to pinned host RAM (least recently used first) or dropped to make room. New jobs look up
//      the prompt cache first (below). Continuations (submit_continue) and resumes (resume) go through the same admission.
//   2. prefill: ONE chunk of the oldest job still prefilling, plus the MTP draft head's KV-only catch-up of that chunk.
//      Chunk boundaries: the request's split positions and max_batch chunks between them (partition rule below), so a
//      prompt runs as the same chunks whether the job is alone or not and whether its prefix came from the cache or not.
//      Mixing prefill rows into the decode batch would run them through the decode GEMV instead of the prefill GEMM -
//      different numerics - so prefill interleaves with decode at chunk granularity.
//   3. decode: ONE fused step over every decoding job (tp_model::forward_batch_ids): each job contributes its last token,
//      plus its MTP drafts (<= 8 rows in total, <= 4 per job; drafts beyond the budget are dropped - only acceptance
//      changes, never the tokens). Then per job: accept, rollback (one kernel for all jobs), the draft head's catch-up +
//      next drafts.
// Every job's tokens are bitwise those of the same job run alone, with or without MTP (each row of a batched step is
// computed exactly as in the single-sequence step; tests/fast_sessions_test.cu, p100e-run --sessions).
//
// Prompt cache. A finished job's sequence stays as a cache entry (tokens, KV pages, its
// chunk boundaries, delta-net checkpoints). A new job takes the entry with the longest reusable prefix:
//   - the prefix can only be resumed where the 48 delta-net layers' state is known: a CHECKPOINT (conv + ssm state of
//     every GPU + the draft head's carry, pinned host RAM, LRU under ckpt_budget) or the entry's current state;
//   - checkpoints are taken during prefill at chunk boundaries: the last ckpt_tail split positions of the request (e.g.
//     message starts passed by the server), every ckpt_every positions, and the prompt end;
//   - restore = KV truncated to the checkpoint position c (pages beyond it freed; offloaded entries restore only the
//     pages of 0..c-1), the delta-net state + draft carry loaded from the checkpoint, then only prompt[c..] is prefilled
//     (the draft head's KV rows below c stay valid: they were written by the catch-up of the same chunks).
// Exactness rule (sched_opts::exact): a prompt's chunks are a pure function of the prompt length and its split
// positions (partition(): from each segment start, the next split >= min_split positions on, else max_batch chunks). The
// prefill numerics depend on the chunk (fold-attention key partition, GEMM shapes, the <= 8-token decode path), so the
// cache reuses a prefix only up to a position c that is a boundary of BOTH the cached run's chunks and the new
// request's cold chunks, with the same boundaries below c, and only prefilled positions (never decoded tokens). The
// cached run then prefills the cold partition's tail from c, so it computes every chunk exactly as the cold run of the
// same request: identical logits, identical tokens. An entry's boundaries only ever record prefill chunks as computed
// (they stop growing where decoded tokens or a non-boundary reuse point come in).
// A cold run therefore needs the same split positions as the cached one below c; positions derived from the token
// prefix alone (message starts) satisfy this automatically.
// exact = false (the DEFAULT, like llama-server; P100E_CACHE_EXACT=1 selects exact): reuse up to any checkpoint or
// the entry's current state, incl. decoded tokens, at or below the common prefix - a client that resends the previous
// answer (Claude Code) then prefills only the new tokens, not the answer again; results are within rounding of a cold
// run.
// preempt() / resume(): a job's sequence (KV pages incl. the MTP block, delta-net state) to pinned host RAM and back -
// more sessions than the GPU pool holds; generation continues bit-identically.
// Sessions on disk: save() / load() (tokens, KV pages, delta-net state, draft carry, checkpoints; checked against the
// model / KV layout) - a loaded session is an idle cache entry and a submit_continue handle.
#pragma once

#include "tp_model.h"
#include "tp_mtp.h"

#include <algorithm>
#include <deque>
#include <map>
#include <string>
#include <vector>

namespace p100e {

struct sched_opts {
    int max_active = 3;            // jobs holding a sequence at once (prefilling or decoding)
    int max_rows = tp_model::MAX_ROWS;   // rows per batched decode step
    // rows per step when drafts are added (each job's last token always gets its row): the decode GEMVs read the
    // weights once per 4 columns, so a 5th row costs about a whole T = 1 GEMV pass (P100: T = 2 / 3 / 4 / 8 = 24 / 28 /
    // 33 / ~51 ms per step); drafts per job = this budget shared by the decoding jobs (k_eff), at least 1
    int draft_rows = 4;
    mtp_opts mo;                   // draft policy of MTP jobs (default: adaptive k <= 3, tau 0.5)
    bool prefill_first = false;    // finish every admitted job's prefill before decode steps (benchmarks)
    // decode steps between two prefill chunks while jobs are decoding (at least; 1 = alternate, like llama-server's
    // chunked prefill)
    int decode_per_chunk = 1;
    // share of the wall time that decoding jobs get while another job prefills (P100E_DECODE_SHARE):
    // after a prefill chunk of t seconds the decode steps run for t * share / (1 - share) seconds before the next chunk
    // (and at least decode_per_chunk steps). The chunks themselves never change, so every job's tokens stay identical;
    // only the order of chunks and steps does. 0.5: a decoding session keeps about half its rate during a long prefill
    // and the prompt takes about twice as long while anything decodes; 0 = decode_per_chunk steps per chunk only (one
    // 2048-token chunk ~4.5 s on the P100 per step: ~2-3 t/s for the decoder). Clamped to [0, 0.95].
    double decode_share = 0.5;
    int reserve_extra = 8;         // pages reserved per job: prompt + n_gen + this many positions (drafts)
    // ---- prompt cache ----
    bool cache = true;             // finished jobs' sequences stay as cache entries (requests with cache = true)
    // pinned host RAM for delta-net checkpoints, LRU (P100E_CKPT_BUDGET_MB): a cap within the model's host budget
    // (tp_model::host_budget, shared with offloaded sequences; engine_opts::host_budget_mb)
    size_t ckpt_budget = (size_t) 8 << 30;
    int ckpt_spare = 2;            // freed checkpoint buffers kept for reuse; more are given back (cudaFreeHost)
    int ckpt_every = 8192;         // a checkpoint at the first chunk boundary >= this many positions after the previous one
    int ckpt_tail = 4;             // checkpoints at the last ckpt_tail split positions of a request's prefilled part
    // a split position starts a new segment only when it is >= min_split positions after the previous segment start
    // (0 or the previous split position used); the max_batch chunk ends inside a segment do not count, so a split
    // shortly after one of them still makes a short chunk (max_batch 2048, split at 8230: ... 8192 | 8230 | ...)
    int min_split = 64;
    // reuse only what a cold run reproduces bit for bit (rule above); default false = also reuse decoded tokens.
    // P100E_CACHE_EXACT=1 / 0 overrides.
    bool exact = false;
    bool draft_kv = true;          // with a draft head: every prefill (MTP or not) writes the draft head's KV rows, so any
                                   // cache entry serves MTP requests (KV-only catch-up, ~0.6 % of the prefill time)
};

class sched {
public:
    struct request {
        std::vector<int32_t> prompt;
        int n_gen = 1;
        bool mtp = false;
        bool keep = false;         // a held session: kept after DONE for submit_continue / save, never matched by other
                                   // requests and never dropped (only offloaded); release() frees it
        bool keep_set = false;     // submit_continue: keep as given (else the finished job's keep is inherited when false)
        bool cache = true;         // look up the prompt cache and leave the sequence in it afterwards
        std::vector<int> splits;   // chunk split positions in (0, |prompt|), e.g. message starts (see the exactness rule)
        // generation ends at the first of these tokens (e.g. end-of-turn / EOS): it is the job's last output token, and
        // the sequence holds everything before it, like after n_gen tokens - a stop inside an MTP step keeps only the
        // rows up to it, so the session continues exactly from there (submit_continue)
        std::vector<int32_t> stop;
    };
    struct job {
        int id = -1;
        std::vector<int32_t> prompt;
        int n_gen = 0;
        bool mtp = false;
        bool keep_seq = false;     // request::keep
        bool cache = true;         // request::cache (and sched_opts::cache)
        std::vector<int> splits;
        enum state { QUEUED, PREFILL, DECODE, DONE, FAILED } st = QUEUED;
        std::string err;           // FAILED: why (e.g. larger than the KV pool)
        int seq = -1;              // its sequence (DONE: while the cache entry still belongs to it, else -1)
        bool offloaded = false;    // preempted (resume() re-admits it)
        int cont_of = -1;          // submit_continue: the finished job whose sequence this one continues
        bool append = false;       // a continuation with new input: prompt = the whole sequence (its tokens + the last
                                   // generated token + the input), prefilled from n_cached = the sequence's length
        std::vector<int32_t> stop; // request::stop
        bool stopped = false;      // ended on a stop token
        bool cancelled = false;    // cancel()
        std::vector<int> chunks;   // prompt chunk sizes (from c0 at admission)
        size_t ci = 0;             // next chunk
        int c0 = 0;                // first position of the next chunk
        std::vector<int> ck;       // chunk ends where the prefill takes a checkpoint (ascending)
        bool dkv = false;          // the prefill writes the draft head's KV rows (MTP job, or sched_opts::draft_kv)
        bool draft_ok = false;     // MTP decode on a draft KV without holes (every decoded position gets its row)
        std::vector<float> L;      // logits of the last prompt chunk
        std::vector<int32_t> out;  // generated ids
        int32_t last = -1;         // last generated token (not fed yet)
        int N = 0;                 // its position
        std::vector<int32_t> drafts;
        int reserved_pages = 0;
        // test only (models with useless draft heads, e.g. random weights): the drafts are replaced by these tokens (the
        // job's expected output), the last draft of every third step deliberately wrong - exercises accepted rows,
        // partial acceptance and rollback. Drafts only change acceptance, never the output.
        const std::vector<int32_t> * oracle = nullptr;
        // test only (set_trace, plain jobs): `trace` gets the full logits of every emitted token - each decode step of the
        // job runs alone as a plain T = 1 forward (bitwise the row of a batched step); with `force` the emitted token i is
        // (*force)[i] (teacher forcing) and the model's own greedy pick goes to `own`
        std::vector<std::vector<float>> * trace = nullptr;
        const std::vector<int32_t> * force = nullptr;
        std::vector<int32_t> own;
        // stats (seconds on the steady clock since the scheduler started)
        double t_submit = 0, t_admit = 0, t_first = 0, t_done = 0, t_prefill = 0, t_catchup = 0;
        double t_dec0 = -1;        // start of its first decode step (decode t/s = (out - 1) / (t_done - t_dec0))
        int steps = 0, drafted = 0, accepted = 0, draft_runs = 0;
        int first_rows = 0;        // rows of its first decode step (1 + drafts)
        uint64_t draft_hash = 1469598103934665603ull;   // FNV-1a over every draft proposed (before the row budget)
        double t_offload = 0, t_restore = 0;
        // prompt cache
        int n_cached = 0;          // prompt positions reused from the cache
        int n_prefill = 0;         // prompt positions prefilled
        int ckpts = 0;             // checkpoints taken
        double t_ckpt = 0;         // seconds taking checkpoints
        double t_reuse = 0;        // seconds restoring the reused prefix (pages, truncation, checkpoint load)
        bool reuse_offloaded = false;   // the reused entry came back from host RAM
    };

    sched(tp_model & m, tp_mtp * d, const sched_opts & o);
    ~sched();                      // frees every sequence / checkpoint the scheduler holds (the model outlives it)
    int submit(const request & r);
    int submit(const std::vector<int32_t> & prompt, int n_gen, bool mtp = false, bool keep_seq = false);
    // the next turn of a finished job whose sequence is still its own (a keep job, or a cache entry no other request
    // took): decode n_gen more tokens after its last token. Queued like any job (admission restores it from host RAM
    // when it was offloaded, with the page reservation of the continuation); with MTP the first step already drafts
    // (the draft head's carry was saved when the turn ended). The new job inherits keep, cache and stop.
    int submit_continue(int done_id, int n_gen, bool mtp = false);
    // the next turn WITH NEW INPUT (the multi-turn path of a server: the session keeps its KV and state, only the new
    // tokens are prefilled): r.prompt = the tokens that follow the job's last generated token (e.g. the rest of the
    // end-of-turn markup, the next user message, the assistant prefix); the last generated token (never fed) and
    // r.prompt are prefilled from the sequence's length in the chunks of the partition from there (r.splits: positions in
    // the WHOLE sequence, i.e. history + last token + input, like a cold request's), then r.n_gen tokens are generated.
    // r.mtp, r.keep, r.cache and r.stop apply to this turn and the sequence after it. Empty r.prompt: decode only. The
    // result is within rounding of a cold run of the whole conversation (decoded tokens are reused), not bit-identical.
    int submit_continue(int done_id, const request & r);
    // a finished job's sequence (session / cache entry) back to the pool: pages, slot or host copy, checkpoints. A
    // continuation still queued on it fails ('session released').
    void release(int done_id);
    // stop a job (client gone, stop string, thinking budget):
    //  - QUEUED: taken out of the queue, FAILED 'cancelled' (a continuation's session stays with its finished job);
    //  - DECODE: ends at once like a finished job (DONE, cancelled = true, out = the tokens so far, every one but the last
    //    in the sequence; with MTP the draft head's carry is current): submit_continue (e.g. with '</think>' as input),
    //    save and release work as after a normal finish; the sequence stays per keep / cache;
    //  - PREFILL: FAILED 'cancelled'; the prefilled part stays as an anonymous cache entry when the job had cache or keep
    //    (a later request on the same conversation reuses it), else it is freed.
    // drop = true frees the sequence in every case. Preempted jobs are cancelled the same way (from host RAM).
    void cancel(int id, bool drop = false);
    // the server is done with a finished job: its record shrinks to a stub (prompt / output freed); a sequence it still
    // owns becomes an anonymous cache entry (unpinned; freed when the job had cache = false). Job ids stay valid.
    void forget(int id);
    bool step();                   // one iteration; false when no job is left to run (all done, failed or preempted)
    void run();                    // step() until it returns false
    void preempt(int id);          // offload the job's sequence to pinned host RAM; the job pauses
    void resume(int id);           // re-admit it (when its pages and a slot are free); it continues with the next step
    // sessions on disk: the sequence of a finished job (tokens, KV pages, delta-net state, draft carry, and its newest
    // max_ckpts checkpoints, -1 = all) -> path: written to path.tmp, flushed to disk, then moved over path atomically
    // (MoveFileEx REPLACE_EXISTING | WRITE_THROUGH: a crash leaves the old file or the complete new one). load() checks
    // the model fingerprint (tp_model::seq_read: refused on a mismatch) and the header (boundaries 0 .. n_pf ascending,
    // n_pf / n_draft / carry / checkpoint positions within the sequence), and returns a finished-job handle whose
    // sequence is an idle (offloaded) cache entry: submit_continue(handle, ..) continues it, and requests reuse its prefix
    // - exactly (bit for bit) only when the file's numerics hash is this process's (tp_model::numerics_hash), else
    // only by non-exact lookups.
    void save(int done_id, const std::string & path, int max_ckpts = -1);
    int load(const std::string & path);
    // job records live in a deque: a reference stays valid while other jobs are submitted (until forget(id) shrinks it)
    const job & get(int id) const { return jobs_.at(id); }
    void set_oracle(int id, const std::vector<int32_t> * o) { jobs_.at(id).oracle = o; }   // test only (job::oracle)
    void set_trace(int id, std::vector<std::vector<float>> * trace, const std::vector<int32_t> * force);   // test only
    int n_jobs() const { return (int) jobs_.size(); }
    double now() const;            // seconds since construction
    const sched_opts & opts() const { return o_; }
    // chunk boundaries of a prompt of n tokens with split positions sp (sorted) from position `from` (a boundary):
    // `from`, then every chunk end up to n (the partition rule above; explicit[i]: boundary i is a split position)
    std::vector<int> partition(const std::vector<int> & sp, int n, int from = 0, std::vector<bool> * explicit_ = nullptr) const;
    // cache statistics
    int n_entries() const { return (int) ents_.size(); }
    int n_ckpts() const;
    size_t ckpt_bytes() const { return ck_bytes_; }   // one checkpoint (both GPUs + the draft carry)
    size_t ckpt_host_bytes() const { return ck_alloc_ * ck_bytes_; }   // pinned: live checkpoints + spare buffers
    int evictions = 0, drops = 0, ckpt_evictions = 0;
    // step statistics
    int decode_steps = 0;
    double t_decode = 0, t_prefill = 0, t_drafts = 0, t_rollback = 0;
    double t_dec_wall = 0;         // seconds in whole decode steps (forward, drafts, rollback; step())
    int rows_hist[tp_model::MAX_ROWS + 1] = {};

private:
    struct ckpt_t { int pos = 0; char * buf = nullptr; bool carry = false; uint64_t lru = 0; };
    struct entry {                 // a sequence the scheduler holds
        int seq = -1;
        int owner = -1;            // job running (or queued to continue) on it; -1: idle
        int done_job = -1;         // the finished job it belongs to (submit_continue / save handle)
        bool pinned = false;       // request::keep
        bool cached = true;        // matchable by new requests
        std::vector<int> bnd{0};   // chunk boundaries of the prefilled prefix (ascending, 0 first, the last = n_pf)
        int n_pf = 0;              // positions 0..n_pf-1 were prefilled (decoded tokens come after)
        int n_draft = 0;           // the draft head's KV rows are valid for positions < n_draft
        int carry_at = -1;         // the draft carry continues at this position (= h of position carry_at - 1); -1: none
        std::vector<ckpt_t> cps;   // ascending pos
        double t_used = 0;
        bool exact = true;         // bnd may be compared with this process's cold partitions (false: loaded from a
                                   // session file written with other numerics, sched::load)
    };
    struct plan {
        int seq = -1;              // the entry used (-1: a new sequence)
        int c = 0;                 // positions kept
        int ck = -1;               // checkpoint index to load (-1: the state in place, or none)
        bool offl = false;         // the entry is offloaded (restore)
        int slot = 0, pages = 0, freed = 0;   // slots / pages needed, pages the truncation gives back
        bool bad = false;          // cannot run (a continuation whose session is gone): fail the job
    };
    void admit();
    plan make_plan(const job & j) const;
    plan lookup(const job & j) const;
    void start(job & j, const plan & p);
    bool evict_idle(int keep);     // offload (or drop) the least recently used idle resident entry other than keep
    bool drop_offloaded(int keep1, int keep2);   // drop the least recently used offloaded cache-only entry; false: none
    void plan_chunks(job & j, const entry & e, int c, bool cold_tail);   // chunks + checkpoint positions from c
    bool gen_done(const job & j) const { return j.stopped || (int) j.out.size() >= j.n_gen; }
    bool is_stop(const job & j, int32_t t) const { return std::find(j.stop.begin(), j.stop.end(), t) != j.stop.end(); }
    void unlive(int id);           // a job left QUEUED / PREFILL / DECODE
    void drop_entry(int seq);
    void fail(job & j, const std::string & why);
    bool prefill_chunk(job & j);   // one chunk; true when the prompt is done
    void decode_step();
    void trace_step(job & j);      // test hook (job::trace / force): one plain T = 1 step of j alone
    void chain(job & j, int32_t d0);   // drafts after the draft head's run that picked d0 (generate_mtp's policy)
    void apply_oracle(job & j);
    int k_eff() const;
    void draft_batched(const std::vector<int> & dec, const std::vector<int> & mi, const std::vector<std::vector<int32_t>> & acc,
                       const std::vector<int> & row0, const std::vector<int> & na);
    void finish(job & j);
    void take_ckpt(entry & e, int pos, bool carry);
    char * ck_get();               // a pinned checkpoint buffer within the budgets (LRU eviction); nullptr: none
    char * ck_steal();             // the least recently used checkpoint's buffer; nullptr: no checkpoint
    void ck_put(char * b);         // a buffer back (kept as a spare up to ckpt_spare, else freed)
    void ck_trim(entry & e, int c);   // drop the entry's checkpoints beyond position c
    int pages_for(int tokens) const { return (tokens + m_.page_size() - 1) / m_.page_size(); }
    std::vector<int> clean_splits(const std::vector<int> & sp, int n) const;

    tp_model & m_;
    tp_mtp * d_;
    sched_opts o_;
    std::deque<job> jobs_;         // a deque: references stay valid on push_back (get())
    std::vector<int> live_;        // ids of the jobs not finished (queued, prefilling, decoding, preempted), ascending
    std::deque<int> queue_;
    std::map<int, entry> ents_;    // by sequence id
    int rr_ = 0;                   // round-robin start of the draft budget
    int dec_since_chunk_ = 0;      // decode steps since the last prefill chunk
    double dec_owed_ = 0;          // decode seconds still due before the next prefill chunk (decode_share)
    bool adm_dirty_ = true;        // admission may make progress (set by submit / finish / release / resume / load)
    double t0_ = 0;
    size_t ck_bytes_ = 0;          // one checkpoint: state_bytes() per GPU + the carry (n_embd floats)
    size_t ck_n_ = 0;              // live checkpoints
    size_t ck_alloc_ = 0;          // pinned checkpoint buffers allocated (live + spare), registered as host_extra
    int ck_keep_ = -1;             // an offloaded entry ck_get must not drop for room (the one sched::load fills)
    std::vector<char *> ck_free_;  // recycled pinned checkpoint buffers
    uint64_t lru_ = 0;
};

} // namespace p100e
