// p100e-server <-> inference engine: the narrow interface the HTTP / text layer calls.
//
// Plain C++ only (no CUDA, ggml or llama.cpp types), so the server never includes engine headers and the engine
// never includes llama.cpp headers. Today's implementation is engine_tp (server/engine_tp.cpp: the single-sequence
// tensor-parallel tp_model + MTP draft head); the multi-session engine implements the same class.
//
// Model of a sequence: the engine holds the state (attention KV rows + recurrent delta-net state + MTP draft-head KV)
// of the tokens it was fed for that sequence (seq_tokens). A request
//   1. opens a sequence (or reuses one: seq_keep(n) keeps the first n fed tokens when the engine can cut there),
//   2. prefill()s the rest of the prompt (the caller chunks it by info().n_ubatch so it can cancel / report progress
//      between chunks); the last call returns the next-token distribution of the last prompt token,
//   3. loops decode() -> sampler (the server's) -> keep() -> draft():
//        decode(): rows = [last sampled token, draft_1..draft_k] for every generating sequence; returns per row either
//                  the full logits (host sampling) or only the argmax (greedy fast path, computed on the GPUs)
//        keep():   the sampler accepted the first n_keep rows (1 + accepted drafts); the rest are rolled back
//        draft():  MTP drafts for the step after `next` (the token the sampler emitted after the kept rows)
//   4. closes the sequence (or leaves it open for the next request's seq_keep).
// decode() takes every generating sequence at once: a batched engine runs them in one forward, engine_tp loops.
// Thread safety: all calls from one thread (the server's task loop).
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace p100e {

struct engine_info {
    int n_vocab = 0;
    int n_embd = 0;
    int n_ctx = 0;          // max tokens per sequence (positions)
    int n_ubatch = 0;       // prompt chunk the engine runs in one forward
    int n_seq_max = 1;      // sequences that can be open at once
    int n_draft_max = 0;    // MTP draft tokens per step (0: no draft head)
    int n_rows_max = 1 << 30;   // rows of one decode() call over all sequences (tokens + drafts); the caller trims drafts
    std::string kv_type;    // attention KV cache type (f16 / q8_0 / q4_0)
    std::string desc;       // one line for the startup log
    uint64_t n_params = 0;  // parameter count of the GGUF tensors
    uint64_t model_bytes = 0;   // bytes of the GGUF tensors
};

// the rows one sequence contributes to a decode step
struct decode_req {
    int seq = -1;
    std::vector<int32_t> tokens;   // [last sampled token, draft_1 .. draft_k]
    bool logits = true;            // true: host logits of every row; false: argmax ids only (greedy fast path)
};

struct decode_res {
    std::vector<int32_t> argmax;   // per row (filled when logits == false)
    std::vector<float> logits;     // [rows][n_vocab] (filled when logits == true)
};

// one piece of a multimodal prompt (prefill_mm): a run of text tokens, or an image's embedding rows (the vision
// encoder's output: [nx * ny][n_embd] f32 on the host, raster order). The shared pointer keeps the rows alive for as long
// as the engine may need them again (engine_tp keeps the pieces of the open sequence for the f32-wire redo).
struct prompt_piece {
    std::vector<int32_t> tokens;           // text (empty for an image)
    std::shared_ptr<const float> embd;     // image rows (null for text)
    int nx = 0, ny = 0, n_embd = 0;        // image token grid and row width
    int n_rows() const { return embd ? nx * ny : (int) tokens.size(); }
};

struct engine_stats {
    double t_prompt_s = 0, t_decode_s = 0, t_draft_s = 0;
    int64_t n_prompt = 0, n_decode_steps = 0, n_decode_rows = 0, n_drafted = 0, n_draft_runs = 0;
    int64_t n_prefill_redo = 0;   // prompts redone on the exact (f32) prefill wire after the f16 wire saturated
};

// The engine cannot continue and the process must restart (engine_tp: a P2P allreduce wait timed out -
// tp_model::ar_failed(); the instance stays unusable, every later step would throw). Any engine call may throw it; the
// server logs it and exits non-zero so its supervisor restarts it, instead of failing the request and carrying on.
struct engine_fatal_error : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// One sequence of a multi-sequence call (decode / draft_batch) cannot take its step (engine_tp: no KV page left for it,
// or its context is full). Thrown before the call changed any state: the caller fails that sequence alone and repeats
// the call for the others. Any other exception from those calls concerns the whole step.
struct engine_seq_error : std::runtime_error {
    int seq = -1;
    engine_seq_error(int s, const std::string & what) : std::runtime_error(what), seq(s) {}
};

class engine {
public:
    virtual ~engine() = default;
    virtual const engine_info & info() const = 0;

    // sequences
    virtual int seq_open() = 0;                                    // -1: no free sequence
    virtual void seq_close(int seq) = 0;
    virtual const std::vector<int32_t> & seq_tokens(int seq) const = 0;   // tokens fed so far
    // keep the first n fed tokens; false when the state cannot be cut there (then nothing changed). n = 0 always works.
    virtual bool seq_keep(int seq, int n) = 0;
    // prompt cache: open a sequence for a prompt whose rows have these keys (a text row: its token id; an image row:
    // bit 63 | hash, vision mm_row_keys - different images never match). The engine may start it from the state of an
    // idle cached sequence that shares a prefix (n_cached rows, always < keys.size(); the caller feeds rows from there
    // on), and plans its own checkpoints for this prompt. Default: no cache (seq_open, n_cached = 0).
    virtual int seq_open_cached(const std::vector<uint64_t> & keys, int & n_cached) {
        (void) keys;
        n_cached = 0;
        const int s = seq_open();
        if (s >= 0) seq_keep(s, 0);
        return s;
    }
    // cache statistics for /metrics and the log (zeros without a cache)
    struct cache_stats { int entries = 0, offloaded = 0, ckpts = 0; int64_t hits = 0, misses = 0, rows_reused = 0;
                         size_t host_bytes = 0; int64_t pool_tokens = 0, free_tokens = 0; };
    virtual cache_stats cstats() const { return {}; }

    // append prompt tokens (any count; runs n_ubatch chunks). want_logits: out.logits = [1][n_vocab] of the last row,
    // else out.argmax = {greedy token}. Throws on failure (context full, CUDA error). A prefill whose inexact fast
    // path went out of range (engine_tp: the f16 prefill wire clamped a partial, tp_model::last_prefill_saturated) is
    // redone exactly inside the call - the whole prompt fed so far, from an empty state - so the state a prefill leaves
    // (and anything a prompt cache keeps of it) never holds clamped values.
    virtual void prefill(int seq, const std::vector<int32_t> & tokens, bool want_logits, decode_res & out) = 0;

    // prefill() for a multimodal prompt part: text and image pieces appended in order (Qwen3-VL interleaved M-RoPE: an
    // image advances the rotary position by max(nx, ny), text after it continues from there). Images are always whole
    // (the caller never splits one across calls); the engine cuts the rows into its own ubatch chunks. seq_tokens()
    // holds -1 for every image row. Same contract as prefill() otherwise, including the exact redo after a saturated
    // f16 wire (over every piece fed so far, images included). Engines without image support throw.
    virtual void prefill_mm(int seq, const std::vector<prompt_piece> & pieces, bool want_logits, decode_res & out) {
        (void) seq; (void) pieces; (void) want_logits; (void) out;
        throw std::runtime_error("this engine does not support image input");
    }

    // one step for every request (each sequence at most once); out[i] belongs to reqs[i]. engine_seq_error: that one
    // sequence has no room, nothing was run (out may hold buffers of an earlier step: the caller may keep `out` across
    // steps so its logits buffers are reused)
    virtual void decode(const std::vector<decode_req> & reqs, std::vector<decode_res> & out) = 0;
    // after decode(): keep the first n_keep rows of this sequence's step (1..rows)
    virtual void keep(int seq, int n_keep) = 0;
    // after prefill() or decode() + keep(): MTP drafts for the step that starts with `next` (at most k_max; the engine
    // may propose fewer - adaptive draft length). k_max = 0: no drafts, but the draft head still records `next`'s
    // predecessors so later steps can draft. Always empty without a draft head.
    virtual std::vector<int32_t> draft(int seq, int32_t next, int k_max) = 0;
    // draft() for several sequences of the same step at once (after every one's keep()): out[i] = the drafts of
    // reqs[i], each exactly what draft() alone would return. engine_tp runs one batched draft-head pass for all of them.
    struct draft_req { int seq = -1; int32_t next = 0; int k_max = 0; };
    virtual void draft_batch(const std::vector<draft_req> & reqs, std::vector<std::vector<int32_t>> & out) {
        out.clear();
        for (const draft_req & r : reqs) out.push_back(draft(r.seq, r.next, r.k_max));
    }
    // the draft head's probability of each draft of seq's latest draft()/draft_batch() list (cumulative: p of drafts
    // 0..j all being its picks), same length as that list; empty when unknown. Only ranks drafts (row trim), never tokens.
    virtual std::vector<float> draft_probs(int seq) const { (void) seq; return {}; }

    virtual engine_stats stats() const = 0;
    virtual void print_mem(const char * tag) const = 0;

    // device memory the engine can lend while it is idle (tp_model::idle_scratch: the prefill ring), e.g. as the native
    // vision encoder's scratch. Empty while a prefill runs; `lease` is held until the borrower is done.
    struct lent_mem { int dev = -1; void * p = nullptr; size_t bytes = 0; std::shared_ptr<void> lease; };
    virtual bool can_lend() const { return false; }
    virtual std::vector<lent_mem> lend_scratch() { return {}; }
};

struct engine_tp_params {
    std::string model_path;
    std::vector<int> devices{0, 1};
    std::vector<double> split;      // tensor-parallel weight per GPU (empty: equal, or P100E_SPLIT; tp_model::opts::split)
    int n_ctx = 4096;
    int n_ubatch = 2048;
    std::string kv_type = "q8_0";   // f16 | q8_0 | q4_0
    int mtp_k = 3;                  // max drafts per step (0: no MTP, the draft head is not loaded)
    float mtp_tau = 0.5f;           // adaptive draft length: draft j only while prod p_i >= tau (0: always k)
    int n_seq = 1;                  // sequences generating / prefilling at once (the server's -np)
    bool cache = true;              // prompt cache: closed sequences stay as cache entries (GPU, then host RAM)
    int cache_resident = 4;         // idle cache entries kept on the GPU (delta-net state slots beyond n_seq)
    int ckpt_tail = 4;              // checkpoints at the last ckpt_tail message starts of a prompt (split_tokens)
    std::vector<int32_t> split_tokens;   // a message start (e.g. <|im_start|>): checkpoint positions
    size_t host_budget_mb = 20480;  // pinned host RAM: offloaded cache entries + checkpoints
    size_t ckpt_budget_mb = 8192;   // of which checkpoints at most
    int64_t pool_tokens = -1;       // KV pool positions shared by all sequences; < 0: free VRAM after the reserves
    std::vector<size_t> reserve_mb; // per GPU kept free (empty: 1024 MiB on the last GPU, the vision encoder)
    int64_t margin_mb = -1;         // plus this on every GPU (transient allocations; < 0: tp_model's default 512)
};

// multi-sequence tensor-parallel engine (tp_model sequences + tp_mtp, prompt cache); throws on load failure
std::unique_ptr<engine> make_engine_tp(const engine_tp_params & p);

// Flash-Next (src/fn_model.h): one sequence, hot Q4 experts resident per GPU, the rest streamed from host RAM, QSA
// sparse attention, MTP drafts. Env switches (P100E_FN_*) as p100e-fn-run; kv type by P100E_FN_KV (q8 above 32K).
struct engine_fn_params {
    std::string model_path;
    std::vector<int> devices{0, 1};
    int n_ctx = 4096;
    int max_t = 512;                // prefill chunk (rows per forward; activation buffers ~1 MB per row)
    int hot_per_rank = 72;          // GPU-resident experts per layer per GPU (-1: the file's hot set split evenly)
    int mtp_k = 3;                  // drafts per step (0: the draft block is not loaded)
    // hybrid pipeline: this server is node 1 (the driver: layers [k, L) + head + draft block); node 0
    // serves layers [0, k) with `p100e-fn-run <model> --net-nodes a,b --net-node 0 --pipe-layers k ...`
    std::vector<std::string> pipe_nodes;   // "host:port" of node 0 and node 1 (empty: one process)
    int pipe_layers = 0;                   // k, -1 = auto
    std::vector<int> rank_slots;           // resident expert slots per local GPU (-1 = auto: fill the GPU); empty = hot_per_rank
};
std::unique_ptr<engine> make_engine_fn(const engine_fn_params & p);

} // namespace p100e
