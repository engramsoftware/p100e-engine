// Tensor-parallel Qwen3.8 on N GPUs (built for 2x P100). Weights resident on the GPUs, sliced at load:
//  - column-parallel (split output rows): attn_q/k/v (by head), attn_qkv/attn_gate/ssm_beta/ssm_alpha (by
//    delta-net head group), ffn_gate/up (by row), output (by vocab row)
//  - row-parallel (split input columns, partial sums + all-reduce): attn_output, ssm_out, ffn_down
// The residual stream is replicated on every GPU. Two all-reduces per layer.
// Rank counts: 1, 2 and 4 GPUs split every dimension of Qwen3.8-27B evenly (4: 6 q heads + 1 KV
// head, 4 of 16 delta-net k heads, FFN 4352, vocab 62080 per GPU). R > 2: every rank reads all R partials (event path:
// one copy per peer; P2P path: one flag word per sender) and adds them in rank order, so the replicas stay identical;
// R <= 2 runs the same kernels and launches as before.
// Uneven shares (src/tp_split.h): every rank holds whole units of each dimension (KV-head groups,
// delta-net k-head groups, 256-row FFN / vocab blocks) from one per-rank range table (split()), so 3 GPUs work (27B
// with equal weights: KV heads 2/1/1, k heads 4/8/4, FFN 23/20/25 x 256 rows, vocab 400/235/335 x 256 rows: the
// 256-row blocks even out the parameters the whole heads leave uneven) and mixed GPUs can take unequal shares
// (opts::split / P100E_SPLIT=a,b,c relative weights). Each rank's buffers, delta-net state and KV pages are sized by
// its own share (rank_state Hl .. page_bytes); the residual stream, the allreduce buffers and the logit rows stay
// [E] / [VOCAB] per row. Equal weights on a dimension that divides by R give exactly dim / R: R = 1, 2, 4 keep the old
// slices, kernels and launches bit for bit.
// Ranks on several machines (opts::net_nodes): SPMD lockstep. Every node runs the same program with the
// same arguments and knows the global rank count R_ and the split, but holds rank_state / weights only for ITS ranks
// (rk_ = the local ranks, global ranks r0_ .. r0_ + rk_.size() - 1; global rank order = node order). The cross-node
// traffic is data only (src/net_link.h): each allreduce's partials (one message per node with all its ranks'
// partials, D2H -> TCP -> H2D into the peer slots, then the same add kernels over all R_ partials in global rank order:
// every replica bit-identical, equal to a one-process run of the same split with the event allreduce), the prefill
// pipeline's per-chunk exchange (f16 wire as in one process), the logit slices / argmax pairs and a few host verdicts.
// Off in net mode: CUDA graphs, the P2P allreduce, MTP, sessions to host / disk, logits_rows, forward_batch_ids.
// Without net_nodes nothing changes (R_ = rk_.size(), r0_ = 0, the same kernels and launches).
// Hybrid pipeline (opts::pipe_layers with net_nodes of 2 nodes): the tensor split stays INSIDE each
// node and whole layers are handed between them. Node 0 runs decoder layers [0, k) over its own GPUs exactly as a
// one-process run of R = its device count (P2P allreduce, fused decode, CUDA graphs, prefill pipeline), node 1 runs
// layers [k, L) + output norm + LM head over its GPUs (typically one: R = 1, the single-GPU paths). Each node loads,
// caches (KV, delta-net state) and splits only its own layers; nothing of the tensor-parallel net mode runs (net_ stays
// empty: R_ / r0_ / split_ are local). Per forward of T rows the link carries node 0's final residual rows S.x [T][E]
// f32 (exact, no f16) to node 1 and the result back (the greedy ids or the logit rows). Both nodes still run the same
// program with the same arguments (SPMD): the token embedding happens on node 0, every result is returned on both.
// Off in pipe mode: MTP, sessions to host / disk, logits_rows, forward_batch_ids, forward_mm (images).
#pragma once

#include "gguf.h"
#include "lend_gate.h"
#include "net_link.h"
#include "pdl/pdl_link.h"   // net_transport
#include "tp_split.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <atomic>
#include <cstdio>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "fast/matvec.cuh"
#include "fast/attn.cuh"
#include "fast/ssm.cuh"
#include "fast/mrope.cuh"
#include "mm_track.h"

namespace p100e {

namespace pdl { class gpu_link; }   // src/pdl/pdl_gpu.cuh (P100E_NET_GPULINK; tp_model.cu only)

struct pf_rank;   // prefill pipeline resources (streams, f16 weight ring, events), tp_model.cu

struct dev_weight {
    uint32_t type = 0;
    int64_t n_in = 0, n_out = 0;   // local shape
    size_t row_bytes = 0;
    void * d = nullptr;
    fast::packed_weight pk;        // fast decode layout (may be repacked in place: then d is NOT raw GGUF bytes)
    bool packed = false;           // true when d holds a repacked (non-GGUF) layout
    bool owns = true;              // false: a view into a concatenated allocation (freed through its owner)
};

// one decode matvec launch: a weight, or several weights with the same input concatenated row-wise in one
// allocation (one kernel over all their rows; outputs land back to back at y)
struct mv_launch {
    fast::packed_weight pk;
    float * y = nullptr;
    std::vector<std::pair<int, int64_t>> parts;   // load_set: (index in the set, rows) of each part, in row order
    std::vector<fast::mv_out> outs;               // T > 1: per-part outputs [T][rows] (empty: y)
};

// per-layer pointers for the fused T = 1 decode step (resolved once at load, no name lookups per token)
struct layer_plan {
    bool attn = false;
    const float *attn_norm = nullptr, *post_norm = nullptr, *q_norm = nullptr, *k_norm = nullptr;
    fast::GdnWeights gw{};
    std::vector<mv_launch> in_attn;           // projections of the attention / delta-net input (read cur)
    std::vector<mv_launch> in_ffn;            // ffn_gate, ffn_up
    // T > 1 rows (MTP verify): one launch per weight (no concatenation, whose outputs would interleave per row),
    // outputs [T][n_out] in the per-op buffers (attention q/k/v -> a/e/g; delta net qkv/z/beta/alpha -> a/b/d/e;
    // ffn gate/up -> a/b)
    std::vector<mv_launch> in_attn_t, in_ffn_t;
    float *qkv = nullptr, *z = nullptr, *beta = nullptr, *alpha = nullptr;   // delta-net projection outputs
    float *q = nullptr, *k = nullptr, *v = nullptr;                           // attention projection outputs
    float *fg = nullptr, *fu = nullptr;                                       // ffn gate / up outputs
    fast::packed_weight o, down;              // attn_output or ssm_out; ffn_down
};

struct rank_state {
    int dev = 0;
    // this rank's share of the split dimensions (tp_model::split(): the ranges; equal shares = the old dim / R)
    int Hl = 0, HKVl = 0;                     // attention q heads, KV heads
    int HKl = 0, HVl = 0, Cl = 0;             // delta-net k heads, v heads, conv channels (2 HKl DK + HVl DV)
    int FFl = 0;                              // FFN rows
    int VL = 0;                               // vocab rows (S.logits / S.lrows row stride)
    int64_t v0 = 0;                           // token id of its first vocab row (argmax / host logits offset)
    size_t csz = 0, ssz = 0;                  // conv / ssm state floats per delta-net layer (slot and snapshot stride)
    size_t pblk = 0, page_bytes = 0;          // KV pool: one page of one layer's K (or V); one page of every KV layer, K + V
    cudaStream_t s = nullptr;
    cublasHandle_t cb = nullptr;
    std::map<std::string, dev_weight> w;      // quantized weights (local slices)
    std::map<std::string, float *> f;         // small f32 tensors (norms, conv, biases)
    float * w_scratch = nullptr;              // f32 dequant scratch
    void * mv_scratch = nullptr;              // fast::matvec activation scratch
    unsigned int * gdn_cnt = nullptr;         // fast::gdn_layer fuse_cnt (P100E_GDN_FUSE)
    float *x, *cur, *part, *peer, *a, *b, *c, *d, *e, *g;
    int32_t * pos = nullptr;                  // positions [max_batch] (a view into desc)
    int32_t * rpos = nullptr;                 // [3][max_batch] M-RoPE positions (t, h, w) of the staged rows (images in context)
    std::vector<float *> kc, vc;              // reference path (P100E_FAST=0): f32 caches, one sequence
    std::vector<float *> conv_state, ssm_state;   // delta-net state of the ACTIVE sequence per layer (views into the
                                                  // slot pools; host-pointer paths: prefill, per-op, catch-up)
    std::vector<fast::kv_cache> kv;          // per layer: KV cache base (paged: the pool layer's K / V, pt = ptab)
    // ---- sequences ----
    char * kv_pool = nullptr;                 // paged KV pool [K | V][kv layer][page][page size][row]
    int32_t * ptab = nullptr;                 // page tables [max_slots][max_pages] (device)
    int32_t * desc = nullptr;                 // step descriptor (device): rtab [MAX_ROWS] | gseq [MAX_ROWS][4] | pos [max_batch]
    int32_t * rtab = nullptr, * gseq = nullptr;   // views into desc: per-row page-table offset, per-sequence GDN rows
    float * conv_pool = nullptr, * ssm_pool = nullptr;   // delta-net states [slot][gdn layer][...]
    float * conv_snp = nullptr, * ssm_snp = nullptr;     // rollback snapshots [gdn layer][snapshot slot][...]
    float * logits = nullptr;                 // [8][VL]
    float * hn = nullptr;                     // [max_batch][E] output_norm(x) of the last forward (h_nextn for MTP)
    void * amax = nullptr;                    // [8] (value, index) argmax of logit rows (forward_ids, MTP draft head)
    std::vector<float *> conv_snap, ssm_snap; // rollback snapshots per layer (views into conv_snp / ssm_snp: slot i = state
                                              // after token i of a keep_states batch)
    cudaStream_t rb_s = nullptr;              // rollback copy stream (overlaps the MTP draft head; joined by the next forward)
    cudaEvent_t ev_rb_in = nullptr, ev_rb = nullptr;
    char * ck_stage = nullptr;                // state_save_async: device staging copy of the state [conv | ssm]
    cudaStream_t ck_s = nullptr;              // its D2H stream (overlaps the next forward); ev_ck_in: S.s -> ck_s,
    cudaEvent_t ev_ck_in = nullptr, ev_ck = nullptr;   // ev_ck: the D2H done (state_save_join / the next staging copy)
    cudaEvent_t ev_part = nullptr;            // partial-sum ready
    cudaEvent_t ev_copied = nullptr;          // my partial arrived on the peer
    cudaStream_t s2 = nullptr;                // P100E_Q8_FORK: side stream of the small q8 in-projection
    cudaEvent_t ev_q8f = nullptr, ev_q8j = nullptr;   // P100E_Q8_FORK: fork / join events
    cudaEvent_t ev_join = nullptr;            // CUDA graph: this rank's stream joins rank 0's (capture / launch)
    void * aq = nullptr;                      // a16-quantized activation shared by the matvecs of one input (fused decode)
    float * part2 = nullptr;                  // P2P allreduce: second partial buffer (part / part2 alternate)
    int * flag = nullptr;                     // P2P allreduce: [0] sequence number written by the peer, [1] wait
                                              // timeouts, [2] this GPU's own signal counter (fast::p2p_signal_ctr)
    std::vector<dev_weight> cat;              // owners of concatenated weight allocations
    std::vector<layer_plan> lp;               // fused decode plan per layer
    pf_rank * pf = nullptr;                   // prefill pipeline (created on first use)
    float * lrows = nullptr;                  // logits_rows: [64][VL] LM-head output (lazy)
};

class tp_model {
public:
    // Construction options. The engine keeps several sequences: each one = a page table into the shared
    // paged KV pool (all attention layers + the MTP block, pages of 2^page_shift positions), a delta-net state slot and its
    // own position count. The single-sequence API (forward_n / forward_ids / rollback / reset / n_past) acts on the ACTIVE
    // sequence (seq_select); forward_batch_ids runs rows of several sequences in one fused step.
    struct opts {
        int n_ctx = 4096;               // per-sequence context cap (positions)
        int max_batch = 512;            // ubatch: longer prompts run as consecutive forwards of <= max_batch tokens
        bool load_mtp = false;
        uint32_t kv_type = fast::KV_Q8_0;
        int max_seqs = 1;               // GPU-resident sequences (delta-net state slots, page-table rows)
        int64_t pool_tokens = 0;        // KV pool: > 0 tokens; 0 = n_ctx * max_seqs; < 0 = deferred: alloc_pool() later
                                        // (auto = as large as free VRAM allows, after the drafter / prefill buffers exist)
        bool default_seq = true;        // create + select sequence 0 at construction (the single-sequence tools)
        int page_shift = 8;             // pages of 256 positions (P100E_PAGE_SHIFT overrides; 5..12)
        // tensor-parallel shares: a relative weight per GPU in device order (e.g. {2, 1, 1}); empty =
        // P100E_SPLIT=a,b,c when set, else equal weights (as even as the model's units allow; src/tp_split.h)
        std::vector<double> split;
        // network tensor parallel: "host:port" per node (global rank order = node order: node 0's
        // devices are ranks 0.., then node 1's ...); empty = one process. devices = THIS node's GPUs only; split = the
        // weights of all ranks of all nodes. Every node must pass the same values (checked at connect, refused on a
        // mismatch: protocol, model, rank count, split, n_ctx, max_batch, KV type, net_args_hash, control env switches).
        std::vector<std::string> net_nodes;
        int net_node = 0;                 // this node's index in net_nodes
        uint64_t net_args_hash = 0;       // the caller's run arguments (e.g. p100e-run's argv minus node-local ones)
        int net_connect_s = 600;          // wait this long for every node to come up (P100E_NET_CONNECT_S)
        int net_timeout_s = 60;           // a collective with no progress this long fails (P100E_NET_TIMEOUT_S)
        // hybrid pipeline (needs net_nodes with exactly 2 nodes): node 0 runs layers [0, pipe_layers)
        // with its own tensor split over its devices (split / P100E_SPLIT = ITS weights only), node 1 runs the rest +
        // the LM head. 0 = off (net_nodes then means tensor parallel over the nodes); -1 = auto: the most layers node 1
        // holds with its KV pool and a margin (both nodes' free VRAM, exchanged at connect). Checked at load against
        // both nodes' free VRAM (the byte plan is printed; a node that does not fit is refused).
        int pipe_layers = 0;
        int pipe_total = 0;               // --pipe-split a,b: a + b, must equal the model's layer count (0 = not given)
    };
    // kv_type: attention KV cache type of the fast path (fast::KV_Q8_0 default / KV_F16 / KV_Q4_0). max_batch = the ubatch:
    // longer prompts run as consecutive forwards of <= max_batch tokens (forward_n). One sequence, pool = n_ctx positions.
    tp_model(const std::string & path, const std::vector<int> & devices, int n_ctx, int max_batch, bool load_mtp = false,
             uint32_t kv_type = fast::KV_Q8_0);
    tp_model(const std::string & path, const std::vector<int> & devices, const opts & o);
    ~tp_model();

    static constexpr int MAX_ROWS = 8;   // rows of one batched step (all sequences); <= 4 per sequence
    // numerics version of this build: bump it with any change that alters the bits a prefill or decode step writes into
    // the KV cache or the delta-net state (kernels, accumulation order, chunk handling). Part of numerics_hash().
    static constexpr uint32_t NUMERICS_VERSION = 4;   // 2: f16 prefill wire default off; 3: default on again;
    // 4: P100E_Q6_PAD=1 (ffn_down q6r3) and P100E_MV_Q6_H16=2 (scale-fold Q6_K tail) defaults

    // ---- sequences ----
    int seq_create();                    // empty sequence (zero state) on a free slot; throws when no slot is free
    void seq_free(int id);               // pages and slot back to the pool (offloaded: host copy freed)
    void seq_select(int id);             // the active sequence (must be resident)
    int seq_active() const { return active_; }
    int seq_len(int id) const;           // positions processed
    bool seq_resident(int id) const;
    int seq_pages(int id) const;         // pool pages held (resident) / saved (offloaded)
    // idle sequence -> pinned host RAM (KV pages incl. the MTP block, delta-net + conv state); frees its pages and slot
    void seq_offload(int id);
    void seq_restore(int id);            // back on the GPU (any free slot / pages; the result is bit-identical)
    size_t host_bytes() const { return host_bytes_; }   // pinned host memory held by offloaded sequences
    // pinned host RAM budget of the prompt cache (default 20 GiB, P100E_HOST_BUDGET_MB): offloaded sequences AND the
    // pinned memory registered with host_extra_add (the scheduler's delta-net checkpoints) together; seq_offload /
    // seq_read throw when a sequence would exceed it
    void set_host_budget(size_t bytes) { host_budget_ = bytes; }
    size_t host_budget() const { return host_budget_; }
    void host_extra_add(long long bytes) { const long long v = (long long) host_extra_ + bytes; host_extra_ = v > 0 ? (size_t) v : 0; }
    size_t host_extra() const { return host_extra_; }
    size_t host_used() const { return host_bytes_ + host_extra_; }
    size_t seq_host_bytes(int id) const;
    // ---- prompt cache primitives ----
    // the tokens the sequence has processed (position p = toks[p]; maintained by every forward, rollback and truncation)
    const std::vector<int32_t> & seq_tokens(int id) const { return seqs_.at(id).toks; }
    // resident sequence: keep positions 0..n-1 (n <= seq_len), pages beyond them back to the pool. The delta-net state
    // is NOT touched: the caller loads the state of position n (state_load) unless n == seq_len.
    void seq_truncate(int id, int n);
    // offloaded sequence: back on the GPU with only positions 0..n-1 (n <= seq_len; a free slot + their pages); the host
    // copy is freed. The delta-net state is NOT restored (it belongs to seq_len): the caller loads position n's (state_load)
    void seq_restore_prefix(int id, int n);
    // resident sequence src: a NEW sequence (free slot + pages) holding a copy of its positions 0..n-1 (KV pages incl. the
    // MTP block, tokens, image list cut at n). The delta-net state is zero: the caller loads position n's (state_load).
    // Throws when no slot / not enough free pages (nothing changed then).
    int seq_fork(int src, int n);
    // delta-net + conv state per GPU buffer: the largest rank's (an uneven split's rank r fills the first state_bytes(r)
    // bytes of its buffer; state_save zeroes the rest)
    size_t state_bytes() const {
        size_t b = 0;
        for (int r = 0; r < (int) rk_.size(); ++r) b = b > state_bytes(r) ? b : state_bytes(r);
        return b;
    }
    size_t state_bytes(int r) const { return (size_t) n_gdn_ * (rk_[r].csz + rk_[r].ssz) * 4; }
    // the resident sequence's delta-net + conv state <-> host (dst / src[rank], state_bytes() each, pinned for speed;
    // [conv | ssm], the offload layout). Synchronous (every rank's stream idle on return).
    void state_save(int id, const std::vector<char *> & dst);
    // state_save without the wait: the state is staged on the device in stream order (exactly what state_save would copy),
    // the D2H runs on a side stream while the caller's next forwards run. dst (pinned) is valid after state_save_join(),
    // which every caller that reads, frees or reuses dst must call first. One pending save per model.
    void state_save_async(int id, const std::vector<char *> & dst);
    void state_save_join();
    void state_load(int id, const std::vector<const char *> & src);
    void state_zero(int id);
    // disk (persistence across restarts): the sequence (tokens, KV pages incl. the MTP block, delta-net state; resident or
    // offloaded) -> f, with a fingerprint of the model / KV layout. seq_read creates an OFFLOADED sequence from f (pinned
    // host RAM, counts against the host budget; seq_restore / seq_restore_prefix bring it to the GPU) after checking the
    // fingerprint (throws on a mismatch or a short file).
    // The fingerprint (hard identity, a mismatch is refused): model file size, GGUF metadata hash, tensor table hash,
    // tensor content sample hash, dimensions, KV type / page layout, GPU count (an uneven tensor-parallel split folds
    // its range table into that field: a file from another split is refused), state / page bytes.
    void seq_write(int id, FILE * f);
    int seq_read(FILE * f);
    // numerics identity: NUMERICS_VERSION, max_batch, every GPU's compute capability, the CUDA runtime and
    // cuBLAS versions, and the P100E_* environment switches that can change numerics (all of them except the ones listed
    // in tp_model.cu as numerics-neutral: debug output, budgets, cache policy, stream priorities, graphs ...). A session
    // file from a build or environment with another hash is still valid KV, but not bit-identical to this process's cold
    // runs: the scheduler loads it without exact reuse (sched::load).
    uint64_t numerics_hash() const { return numerics_; }
    const std::string & numerics_env() const { return numerics_env_; }   // the P100E_* switches hashed
    // ---- KV pool ----
    // tokens > 0: that many positions (rounded up to pages); tokens < 0: all free VRAM except reserve_mb[rank] and
    // margin_mb on each GPU (the smaller GPU decides). Drops captured graphs. Prints the capacity.
    void alloc_pool(int64_t tokens, const std::vector<size_t> & reserve_mb = {}, size_t margin_mb = 512);
    bool pool_ready() const { return pool_ok_; }
    bool paged() const { return paged_; }
    int page_size() const { return 1 << psh_; }
    int64_t pool_tokens() const { return (int64_t) n_pages_ << psh_; }
    int64_t free_tokens() const { return (int64_t) free_pages_.size() << psh_; }
    int free_pages() const { return (int) free_pages_.size(); }
    int pool_pages() const { return n_pages_; }
    int max_slots() const { return max_slots_; }
    int free_slots() const { return (int) free_slots_.size(); }
    size_t page_bytes() const { return page_bytes_max_; }   // per GPU (uneven split: the largest rank's)
    int n_ctx() const { return n_ctx_; }
    // pages so that the active sequence holds positions 0..n-1 (the MTP draft head writes ahead of n_past)
    void reserve_positions(int n) { ensure_pages(active_, n); }
    // ---- batched decode (T = sum of rows <= MAX_ROWS, 1..4 rows per sequence, distinct resident sequences) ----
    struct seq_rows { int seq; std::vector<int32_t> tokens; };
    // one fused step over every sequence's rows (row t of sequence s at its position n_past(s) + t); returns the greedy
    // token of every row in order; each sequence advances by its row count. keep_states: rollback snapshots (MTP verify;
    // needs sum(rows - 1) <= n_snap()). Each row is bitwise the row of the same sequence's single-sequence step.
    // logit_rows / logits (optional): the host logits of those rows of the step ([n][VOCAB], rows in step order).
    // logit_rows without logits: the rows stay in the pinned buffer, read through batch_logits() (no host copy)
    std::vector<int32_t> forward_batch_ids(const std::vector<seq_rows> & b, bool keep_states,
                                           const std::vector<int> * logit_rows = nullptr, std::vector<float> * logits = nullptr);
    // row j of the last forward_batch_ids' logit_rows ([VOCAB] f32, pinned): valid until the next forward of any kind
    const float * batch_logits(int j) const { return logits_h_ + (size_t) j * VOCAB_; }
    struct rb_req { int seq, T, n_keep; };   // after forward_batch_ids(keep): keep the first n_keep of the seq's T rows
    void rollback_batch(const std::vector<rb_req> & list);
    int n_snap() const { return n_snap_; }
    void prefill_setup();                // allocate the prefill pipeline now (memory planning before alloc_pool(-1))
    std::vector<float> forward(const std::vector<int32_t> & tokens);
    // forward returning logits of the last n_logits rows ([n_logits][VOCAB], n_logits <= 8). keep_states: also
    // save the recurrent (conv/ssm) state after every token but the last, so rollback() can drop trailing tokens.
    // More than max_batch tokens (no keep_states): ubatch chunks of max_batch tokens in order (ubatch_chunks), logits of
    // the last chunk (only its rows are in hn afterwards; generate_mtp runs the chunks itself, each followed by the draft
    // head's KV-only catch-up of its rows).
    std::vector<float> forward_n(const std::vector<int32_t> & tokens, int n_logits, bool keep_states = false);
    // forward_n's chunk sizes for a T-token prompt (one chunk when T <= max_batch)
    std::vector<int> ubatch_chunks(int T, int n_logits = 1) const;
    // same forward (never the prefill pipeline), but returns the greedy token (argmax, first index on ties) of each of the
    // last n_rows rows, computed on the GPUs: no logits copy, no host argmax (the MTP verify)
    std::vector<int32_t> forward_ids(const std::vector<int32_t> & tokens, int n_rows, bool keep_states = false);
    // after forward_n(T tokens, keep_states=true): keep only the first n_keep tokens (1..T). With images in the context
    // rope_delta() is recomputed from the image list (the delta after the last image left); a cut inside an image throws
    // before anything changes.
    void rollback(int T, int n_keep);
    void enable_rollback(int max_tokens);       // allocate state snapshots for verify batches of <= max_tokens
    void reset();                               // empty context (n_past 0, zero recurrent state)
    // benchmark only: pretend positions n_past..n-1 were processed - the attention KV caches get finite pseudo-random
    // rows there (recurrent states unchanged) and n_past becomes n. Attention cost does not depend on the values.
    void bench_fill_kv(int n);
    // benchmark only (p100e-run --ab at a depth): n_past back to n <= n_past (the KV rows past n are overwritten by the
    // next forward; the recurrent state is NOT rewound, so only timings are meaningful afterwards)
    void bench_rewind(int n);
    // re-read the env switches that may change between forwards in one process (p100e-run --ab): P100E_AR_FENCE
    // (the P2P allreduce signal fence, a device flag word read by the graphed kernels), P100E_AR_SIG_FUSE, and the
    // kernel choices baked into captured graphs (P100E_MVWRP_*, P100E_MERGE_INPROJ, P100E_GDN_FUSE): a change of those
    // drops this model's step graphs and bumps kern_epoch(), which makes a tp_mtp on this model drop its draft graphs
    // before its next run. Streams are synced first.
    void reload_env();
    // number of reload_env calls that changed a kernel choice baked into captured graphs (0 after construction)
    uint64_t kern_epoch() const { return kern_epoch_; }
    void print_mem(const char * tag) const;     // free / total MiB per GPU (cudaMemGetInfo)
    // f16 prefill wire (P100E_PF_WIRE16, default on with 2 GPUs; =0 = exact f32 wire): true when the most recent forward() / forward_n() /
    // forward_mm() / forward_image() call (all of its ubatch chunks) ran a prefill whose row-parallel partials did not fit f16 and were clamped to
    // +-65504. That call's logits and everything it left behind (residual stream, KV cache rows, recurrent state) carry
    // the clamped values: the caller discards them - reset() / rewind and redo the prompt (every later forward of this
    // instance uses the f32 wire already) - and never caches that state (prompt cache, sessions).
    bool last_prefill_saturated() const { return pf_sat_last_; }
    // saturated f16-wire partials (groups of 4 values) since construction; monotonic, so a caller that runs a prompt as
    // several forward calls compares it before and after
    uint64_t prefill_saturations() const { return pf_sat_total_; }
    // P100E_PF_WIRE_STATS=1 (read per forward): the largest |partial| the prefills of the most recent forward() /
    // forward_n() / forward_mm() call exchanged (both GPUs, every row-parallel GEMM, every ubatch chunk; with the f16 wire the rounded
    // value, 65504 when clamped); -1 when nothing was measured. Each prefill also prints a PFWIRE line (value, layer /
    // GEMM, margin to 65504). p100e-run --wire-scan runs a prompt corpus through it.
    float last_prefill_wire_amax() const { return pf_wire_amax_; }
    // true once a P2P allreduce wait of the fused decode timed out (fast/fused.cu p2p_wait_thread: the host-mapped
    // status line; reads host memory only, no sync). The step that saw it throws (check_ar), and so does every later
    // step: neither reset() nor anything else clears the status line or the device timeout count, and the peer's
    // sequence numbers are no longer in step. After a timeout the instance is unusable: the process must restart (a
    // server treats the error as fatal). A timeout during an MTP draft run surfaces at the next forward's check.
    bool ar_failed() const;
    // after a forward: the logits of its rows [row0, row0 + n) -> host [n][VOCAB] (pinned memory recommended), for
    // KL-divergence scoring (forward / forward_n return only the last rows). output_norm + LM head per row on the
    // forward's final residual stream S.x: the decode GEMV in groups of 8 rows (matmul with T <= 8, the kernel that
    // computes the prompt's last-row logits); P100E_KLD_HEAD=gemm: the prefill GEMM (f16 products) in groups of 64.
    // Scratch: S.cur rows (free between forwards) and S.lrows. Leaves every other path and buffer untouched.
    void logits_rows(int row0, int n, float * host);
    int last_tokens() const { return last_T_; }  // rows of the last forward (its S.x rows)
    const gguf_file & gguf() const { return g_; }
    // device memory idle between forwards, per rank in device order: the prefill pipeline's f16 weight ring
    // (P100E_PF_RING_MB, default 640 MB per GPU; allocated here if no prefill ran yet; forward_prefill resets it at the
    // end, nothing is cached across forwards). Lent to the vision encoder (nvit) for its activations.
    // Threading contract (src/lend_gate.h): tp_model is otherwise single-threaded (one thread runs the forwards), but
    // idle_scratch() may be called from ONE other thread (the encoder's) while that thread decodes:
    //  - while a prefill runs (forward_prefill; forward_mm / chunked forward_n for their whole chunk loop, so a borrower
    //    cannot slip in between chunks) or waits to start, idle_scratch() returns {} and the borrower uses its own memory;
    //  - otherwise the returned entries carry a lease (idle_mem::lease, shared by all entries): the memory is the
    //    borrower's until the last copy of the lease is destroyed, and a prefill that starts meanwhile waits for that
    //    (decode, MTP verify and the classic path never touch the ring and do not wait);
    //  - one lease at a time; the thread that holds a lease must drop it before it runs a prefill itself (it would wait
    //    for itself); the leases must not outlive the model.
    // Empty when the prefill pipeline is unusable. Text numerics are unaffected (the gate only orders the uses).
    struct idle_mem { int dev; void * p; size_t bytes; std::shared_ptr<void> lease; };
    std::vector<idle_mem> idle_scratch();
    // false: the prefill pipeline is unusable on this instance (e.g. one GPU of a 27B: FFN slice 17408 > 9216; the
    // largest rank's FFN share counts), so
    // idle_scratch() is always empty; a borrower then keeps its own memory instead of asking at every use
    bool can_lend() const { return prefill_ok(); }
    uint32_t kv_type() const { return kv_type_; }
    int max_batch() const { return max_batch_; }
    int n_past() const { return seqs_.at(active_).n_past; }

    // ---- images, per sequence (the ACTIVE one; rope_delta / image rows / broken flag live in seq_t) (Qwen3-VL interleaved M-RoPE, llama.cpp mtmd semantics) ----
    // A prompt piece: n_tok text tokens, or an image's embedding rows embd [nx * ny][n_embd] f32 (host; raster order, from
    // vision_encoder). Image token i placed at rotary position P (the next text position) gets t = P, h = P + i / nx,
    // w = P + i % nx; afterwards the rotary position advances by max(nx, ny) (not nx * ny) while the KV rows advance by
    // nx * ny, so rope_delta() grows by nx * ny - max(nx, ny). Text rows (prefill, decode, MTP verify / draft head) are
    // roped at row - rope_delta(); while rope_delta() is 0 and nothing multimodal is staged, every path runs exactly the
    // text-only kernels. Attention stays causal by KV row (= llama.cpp's mask for images: causal in raster order within
    // an image). Every image's KV rows and the delta after it are kept (image_rows(), mm_track.h): rollback() recomputes
    // rope_delta from them and refuses a cut inside an image; reset() clears both.
    // n_embd: the width of the embedding rows (vision_encoder::n_embd()); forward_mm throws unless it equals n_embd()
    // (an mmproj of another model would otherwise be read past its rows or as garbage)
    struct mm_in { const int32_t * tokens = nullptr; int n_tok = 0; const float * embd = nullptr; int nx = 0, ny = 0;
                   int n_embd = 0; };
    // forward over the pieces appended to the context: ONE row list cut into ubatch chunks, so text next to an image
    // shares its prefill batches (no separate small forwards for "<|vision_start|>" / the question after the image).
    // Returns the last row's logits. after_chunk(p0, chunk tokens): after each chunk's forward (KV rows p0.., hn rows
    // 0..), with the chunk's rows still staged so set_pos / embed inside it see them (the MTP draft head's KV-only
    // catch-up; image rows have token 0 in the vector). rope_delta() and image_rows() are updated after every chunk, so
    // when after_chunk throws they describe exactly the rows in the context. When a chunk's forward itself throws (its
    // rows half written) or the context would be left inside an image, the context is marked broken (needs_reset()):
    // every forward and rollback then throws until reset().
    std::vector<float> forward_mm(const std::vector<mm_in> & pieces,
                                  const std::function<void(int, const std::vector<int32_t> &)> & after_chunk = nullptr);
    // one image (forward_mm with a single piece); after_chunk(p0, T)
    void forward_image(const float * embd, int nx, int ny, int n_embd,
                       const std::function<void(int, int)> & after_chunk = nullptr);
    int rope_delta() const { return sq().rope_delta; }        // KV rows minus rotary positions of the next text token
    int rope_pos() const { return sq().n_past - sq().rope_delta; } // rotary position of the next text token
    const mm_track & image_rows() const { return sq().mmt; }  // KV rows of the images in the context, delta after each
    bool needs_reset() const { return sq().mm_broken; }       // a multimodal forward failed part-way (see forward_mm)
    int vocab() const { return VOCAB_; }
    int n_embd() const { return E_; }
    int n_ranks() const { return (int) rk_.size(); }   // this process's ranks (net mode: the local ones)
    // ---- network tensor parallel ----
    int n_ranks_global() const { return R_; }          // every node's ranks (= n_ranks() without net_nodes)
    int rank0() const { return r0_; }                  // global index of local rank 0
    bool net() const { return (bool) net_; }           // tensor parallel over the nodes (not pipe mode)
    // the link of either mode (net: tensor parallel over the nodes; pipe: the layer handoff), nullptr in one process
    const net_transport * link() const { return net_ ? net_.get() : pnet_.get(); }
    int net_node() const { return link() ? link()->self() : 0; }
    int net_nodes() const { return link() ? link()->n_nodes() : 1; }
    // monotonic since construction; with the GPU link (P100E_NET_GPULINK) its all-reduces count as collectives and
    // bytes, but not as time (they never block the host: the GPU waits, see net_ar_pre)
    net_stats net_counters() const;
    // ---- hybrid pipeline ----
    bool pipe() const { return (bool) pnet_; }
    int layer_begin() const { return lb_; }            // the decoder layers this process runs: [layer_begin, layer_end)
    int layer_end() const { return le_; }
    int n_layers() const { return L_; }
    const tp_split & split() const { return split_; }   // every rank's ranges of the split dimensions
    bool has_nextn() const { return n_nextn_ > 0; }
    size_t weight_bytes(int rank) const;

private:
    friend class tp_mtp;
    void init(const std::string & path, const std::vector<int> & devices, const opts & o);
    // ---- sequences ----
    struct seq_t {
        bool used = false;
        int slot = -1;                    // delta-net state slot / page-table row (-1: offloaded)
        int n_past = 0;
        std::vector<int32_t> toks;        // processed tokens (size n_past; bench_fill_kv rows: -1)
        std::vector<int32_t> pages;       // pool pages, position p in pages[p >> psh_]
        int snap0 = 0;                    // first snapshot slot of its rows in the last keep_states step
        std::vector<char *> host;         // offloaded: pinned copy per rank ([pages][page_bytes] | conv | ssm)
        int host_pages = 0;
        size_t host_bytes = 0;
        // images (forward_mm): KV rows minus rotary position of the next text token, the images' KV rows + delta after
        // each (rollback / truncate recompute rope_delta from it), and a failed multimodal forward (only reset() recovers)
        int rope_delta = 0;
        mm_track mmt;
        bool mm_broken = false;
    };
    std::vector<seq_t> seqs_;
    int active_ = -1;
    seq_t & sq() { return seqs_.at(active_); }
    const seq_t & sq() const { return seqs_.at(active_); }
    bool paged_ = true;                         // P100E_PAGED=0: contiguous per-layer caches (one sequence, no batching)
    bool pool_ok_ = false;
    int psh_ = 8, max_slots_ = 1, max_pages_ = 0, n_pages_ = 0;
    std::vector<int> free_pages_, free_slots_;
    int n_kvl_ = 0, n_gdn_ = 0;                 // KV layers in a page (attention + MTP block), delta-net layers
    std::vector<int> kvl_of_, gdn_of_;          // layer -> its KV layer / delta-net index (-1: none)
    size_t page_bytes_max_ = 0;                 // the largest rank's KV page (per rank: rank_state pblk / page_bytes)
    int32_t * ptab_h_ = nullptr;                // pinned host mirror of the page tables
    int32_t * desc_h_ = nullptr;                // pinned staging of the step descriptor (pos_h_ = desc_h_ + DESC_POS)
    size_t host_bytes_ = 0;
    size_t host_budget_ = (size_t) 20 << 30;
    size_t host_extra_ = 0;
    static constexpr int DESC_RT = 0, DESC_GS = MAX_ROWS, DESC_POS = MAX_ROWS * 5;
    fast::kv_cache kv_seq(int r, int il) const;    // layer il of the active sequence (host-p0 calls: prefill, per-op)
    fast::kv_cache kv_rows(int r, int il) const;   // layer il with per-row page tables (fused steps: S.rtab)
    void ensure_pages(int id, int n);           // pages for positions 0..n-1 of sequence id (+ table upload)
    void restore_impl(int id, int np, bool state);   // offloaded -> resident: the first np saved pages (+ the state)
    void pages_out(int id, int r, int p0, int n, char * dst);   // rank r's resident pages p0.. -> host, async on S.s
    std::vector<uint64_t> fingerprint() const { return fp_; }   // model / KV layout identity of a sequence file
    void identity_init();                      // fp_ / numerics_ (end of init: the weight pages are still cached)
    std::vector<uint64_t> fp_;
    uint64_t numerics_ = 0;
    std::string numerics_env_;
    void bind_states();                         // S.conv_state / S.ssm_state = the active sequence's slot
    void sync_all();                            // every rank's streams idle (incl. a pending rollback)
    fast::gdn_multi gdn_desc(int r, int il, int T, int nseq, bool snap) const;
    void rollback_slots(const std::vector<std::pair<int, int>> & sl);   // (state slot, snapshot slot) pairs
    // one decoder block on S.x (il == L_ is the MTP block); kv_only: just the K / V cache rows (attention layers)
    void layer(int il, int p0, int T, bool keep_states, bool kv_only = false);
    void embed(const std::vector<int32_t> & tokens);        // token_embd rows -> S.x
    // stage + upload the step descriptor of the active sequence: positions p0..p0+T-1, its page table for every row,
    // one delta-net sequence (rows 0..T-1, its slot, snapshots from slot 0)
    void set_pos(int p0, int T);
    struct col_range { int64_t c0, c1; };      // input-column range (elements)
    void load_rows(int r, const std::string & name, const std::vector<std::pair<int64_t, int64_t>> & rows,
                   const std::vector<col_range> & cols);
    void load_small(int r, const std::string & name, const std::vector<std::pair<int64_t, int64_t>> & rows);
    // row_bytes = the row stride of h (the raw row bytes, or more after pad_slice)
    struct host_slice { uint32_t type = 0; int64_t n_in = 0, n_out = 0; size_t row_bytes = 0; std::vector<uint8_t> h; };
    // re-lays hs out with the row stride the fast packer wants (fast::pack_row_stride: Q6_K rows padded for q6r3,
    // P100E_Q6_PAD), zero pad bytes; no-op without the fast path or when the stride already matches
    void pad_slice(host_slice & hs) const;
    struct set_part { std::string name; std::vector<std::pair<int64_t, int64_t>> rows; std::vector<col_range> cols; };
    host_slice slice(const std::string & name, const std::vector<std::pair<int64_t, int64_t>> & rows,
                     const std::vector<col_range> & cols);
    void upload(int r, const std::string & name, host_slice && hs);
    // loads weights that read the same input; same-type parts are concatenated into one allocation. Returns one
    // launch per allocation, outputs assigned back to back from *y_off (in floats from rk_[r].a); out[i] = part i's output
    std::vector<mv_launch> load_set(int r, const std::vector<set_part> & parts, size_t & y_off, std::vector<float *> & out);
    void matmul(int r, const std::string & name, const float * x, float * y, int T);
    void norm(const float * x, const float * w, float * y, int64_t rows, int n, cudaStream_t s);
    void allreduce_add(int T);                 // x += sum over ranks of part
    void allreduce_send(int T);                // every rank's part -> the peer's peer[ar_cur_] (no add)
    const float * peer_buf(int r) const { return rk_[r].peer + (size_t) ar_cur_ * peer_stride_; }
    // R ranks: S.peer = [parity][sender slot][max_batch][E], slot of sender q at receiver r = q < r ? q : q - 1 (R = 2:
    // slot 0, i.e. peer_buf). Rank q's partial as rank r sees it after the event allreduce (q == r: its own part).
    // r = local rank, q = GLOBAL rank (net mode: the slots cover every node's ranks; one process: global == local)
    const float * peer_from(int r, int q) const {
        const int R = R_, g = r0_ + r;
        return rk_[r].peer + ((size_t) ar_cur_ * (R - 1) + (q < g ? q : q - 1)) * peer_stride_;
    }
    // (net pre-add, P100E_NET_PREADD: global ranks below pre_k_ arrived already folded into S.x; their place in the fold
    // is taken by -0.0f, the exact identity of float addition. pre_k_ = 0 in one process.)
    const float * part_of(int r, int q) const { return q < pre_k_ ? pad_[r] : q == r0_ + r ? rk_[r].part : peer_from(r, q); }
    // ---- network tensor parallel ----
    std::unique_ptr<net_transport> net_;       // P100E_NET=tcp (src/pdl/pdl_link.h)
    int R_ = 1, r0_ = 0;                       // global rank count, global index of local rank 0
    std::vector<int> node_r0_, node_nr_;       // per node: its first global rank, its rank count
    char * net_tx_ = nullptr, * net_rx_ = nullptr;   // pinned staging: [local rank][slot] / [global rank][slot]
    size_t net_slot_ = 0;                      // bytes per rank slot (max_batch x E x 4)
    std::vector<char> amax_net_;               // argmax pairs of every global rank [R_][8] (argmax_collect)
    void net_init(const std::vector<int> & devices, const opts & o);   // connect + handshake (init, before the split)
    void net_check(const opts & o, const std::vector<double> & w);     // agreement after the split (init)
    void net_allreduce_send(int T);            // allreduce_send over the network (event path + TCP): pre, wire, post
    // Speed switches, every one bit-exact (identical logits to one process, the same split, the event
    // allreduce):
    //  - P100E_NET_PREADD (default 1): the receivers' sum is the left fold ((x + p0) + p1) + p2 .. in global rank order
    //    (fast::addn_rmsnorm_q, k::add, add_norm_prescale). Node 0 holds ranks 0..k-1, the prefix of that fold, and x is
    //    bit-identical on every node, so node 0 sends s = (x + p0) + .. + p(k-1) (ONE row block instead of k): a receiver
    //    sets x = s and puts -0.0f (x + -0 == x exactly) in place of p0..p(k-1), keeping the same kernels and order. NOT
    //    p0 + p1 alone: float addition is not associative, x + (p0 + p1) != (x + p0) + p1. Decode always (k >= 2);
    //    prefill only on the f32 wire (an f16-wire s would be 4 bytes vs 2 x 2). Node 0's own sum is unchanged.
    //  - P100E_NET_GRAPH (default 1): T = 1 fused decode steps as CUDA graphs per segment between network exchanges
    //    (net_capture / net_launch): 2 L + 1 graphs per step, captured after the first eager step of a (slot, M-RoPE)
    //    shape; between two segments the host syncs, exchanges, and the next graph starts with the H2D of what arrived.
    //  - P100E_NET_SPIN (net_link, default 1): busy-poll receive.
    //  - P100E_NET_GPULINK (default 0; P100E_NET=pdl only): the all-reduces of T <= gl_maxt_
    //    rows skip the host. Each sending rank's publish kernel writes its partial (or node 0's pre-added s) straight
    //    into the CUDA-mapped send ring and rings a bell word; gpu_link's pinned spin thread (P100E_PDL_CPU) sends it
    //    at once; received partials land in the mapped receive ring and each rank's recv_copy kernel spins on the
    //    mapped flag word, then copies them into the same peer slots / S.x the host path's H2D fills. Every add kernel
    //    after it is unchanged (bit-identical by construction: the same bytes in the same places). No D2H, no H2D, no
    //    stream sync and no exchange() per all-reduce: the host enqueues a whole step and the network runs beside it;
    //    with P100E_NET_GRAPH the step is ONE graph (no segment split, no host between the all-reduces).
    bool net_preadd_ = false;                  // P100E_NET_PREADD and node 0 holds >= 2 ranks (same on every node)
    pdl::gpu_link * gl_ = nullptr;             // P100E_NET_GPULINK (effective: agreed by every node)
    int gl_maxt_ = 0;                          // all-reduces of T <= gl_maxt_ rows go through gl_ (the ring slot size)
    uint64_t gl_out_ = 0, gl_in_ = 0, gl_calls_ = 0;   // gl_ traffic as enqueued (payload bytes, all-reduces)
    bool gl_on(int T) const { return gl_ && T <= gl_maxt_; }
    void gl_init();                            // after the pre-add decision (init): agree, rings, spin thread
    int pre_k_ = 0;                            // receivers with pre-add on: node 0's rank count (see part_of)
    std::vector<float *> pad_;                 // receivers: per local rank [max_batch][E] of -0.0f
    float * pre_buf_ = nullptr;                // node 0: s = x + p0 + .. on rank 0's GPU [max_batch][E]
    struct step_graph;                         // (defined with the CUDA graph members below)
    bool net_cap_ = false;                     // capturing net segment graphs (net_allreduce_send splits the capture)
    std::string net_cap_err_;
    std::vector<step_graph> * net_cap_out_ = nullptr;   // the set being captured (forward-declared type: pointer)
    bool net_graph_ = true, net_graph_failed_ = false;
    void net_ar_pre(int T);                    // enqueue: local peer copies, pre-add (node 0), D2H of what this node sends
    void net_ar_wire(int T);                   // host: sync this node's streams, the TCP exchange
    void net_ar_post(int T);                   // enqueue: H2D of the received partials (and s into S.x), local waits
    void net_seg_begin();
    bool net_seg_end(std::vector<step_graph> & out);
    struct net_gset;
    net_gset * net_find(int T);
    bool net_capture(int T);                   // record fused_body(T) as segment graphs (nothing runs)
    void net_launch(const net_gset & G, int T);
    void net_gather_logits(int n_rows);        // logits_h_ rows: the other nodes' vocab slices
    void net_gather_amax(int n_rows);          // amax_h_ -> amax_net_ with every node's pairs
    int64_t net_min(int64_t v);                // the smallest v of all nodes
    void net_refuse(const char * what) const;  // throws "<what>: not supported over the network yet" in net / pipe mode
    // ---- hybrid pipeline (see the top of this file) ----
    std::unique_ptr<net_transport> pnet_;      // the handoff link (P100E_NET=tcp|pdl); empty = not pipe mode
    int lb_ = 0, le_ = 0;                      // decoder layers of this process [lb_, le_) (one process / net: [0, L_))
    bool has_head() const { return le_ == L_; }   // output norm + LM head live here
    bool has_embd() const { return lb_ == 0; }    // the token embedding happens here
    bool own_layer(int il) const { return il >= lb_ && il < le_; }
    char * pipe_h_ = nullptr;                  // pinned staging: [pipe_hdr][max_batch][E] f32 rows / [8][VOCAB] logits
    size_t pipe_cap_ = 0;
    uint64_t pipe_step_ = 0;                   // handoffs so far (both nodes count the same: lockstep check)
    void pipe_init(const std::vector<int> & devices, const opts & o);   // connect, agree, layer range, byte plan
    // node 0: rank 0's S.x rows [T][E] (already D2H'd into pipe_h_ by the caller, streams synced) -> node 1, with
    // node 0's f16-wire saturation count of this forward (node 1 adds it: last_prefill_saturated() agrees)
    void pipe_send_x(int T, int nsat);
    void pipe_recv_x(int T);                   // node 1: the rows -> S.x of every local rank (H2D on S.s, async)
    // node 1 -> node 0: the greedy ids (ids != nullptr: n_rows int32) or the logit rows logits_h_ [n_rows][VOCAB]
    void pipe_result(int n_rows, std::vector<int32_t> * ids);
    void pipe_x_d2h(int T);                    // node 0: rank 0's S.x rows -> pipe_h_ (async on its stream)
    void norm_in_n(int r, const float * w, bool pending, int T, size_t o, float * y);   // norm_in / norm_last for R > 2
    // fused layer loop, T = 1..MAX_ROWS rows of nseq sequences (S.desc), ends with the last FFN partials pending
    void decode_layers(int T, bool keep_states, int nseq = 1);
    // one fused decoder block on S.x (T rows of nseq sequences per the step descriptor, plan S.lp[il]; il == L_: the MTP /
    // nextn block when loaded), starting with the previous partials pending and ending with its FFN partials pending
    void fused_layer(int il, int T, bool keep_states, int nseq = 1);
    // MTP draft head (fused): x[T-1] += pending partials, y = rmsnorm(x[T-1]) * w, S.aq = a16(y) (one row, T = 1 layout)
    void norm_last(int r, const float * w, int T, float * y);
    bool fused_ok() const { return use_fast_ && fuse_ && plan_ok_ && nextn_plan_ok_; }
    float * part_out(int r) const { return p2p_ar_ ? (ar_buf_ ? rk_[r].part2 : rk_[r].part) : rk_[r].part; }
    void ar_post(int T);                       // fused decode: partials written -> P2P signal or copy to the peer
    // P100E_AR_SIG_FUSE=1 (default 0, read at init and by reload_env): ar_post launches no p2p_signal_k; the signal is
    // done by the next wait (fast::add2_rmsnorm_q_sig_wait), one launch per allreduce and GPU instead of two.
    // sig_pend_[r]: rank r's signal of the last ar_post is still owed (sig_par_[r]: the counter word it writes).
    // sig_last_: the flag-block counter word holding the latest sequence number (0 = P2P_FLAG_CTR0, 1 = CTR1,
    // -1 = both); a fused signal reads that word and writes the other one (p2p_signal_ctr uses CTR0: sig_need(0)).
    bool sig_fuse_ = false;
    bool gdn_fuse_ = false;   // P100E_GDN_FUSE (step 6b, default 0): T = 1 gdn_prep + gdn_scan in one launch (S.gdn_cnt)
    bool q8_fork_ = false;    // P100E_Q8_FORK (default 0): the small q8 in-projection on a side stream (graph branch), joined before its consumer
    bool merge_in_ = false;   // P100E_MERGE_INPROJ (step 6a, default 0): fused_layer's mixed-type T = 1 in-projections in one launch
    uint64_t kern_epoch_ = 0; // kern_epoch(): bumped by reload_env when it drops the graphs for a kernel switch
    bool sig_pend_[2] = {false, false};
    int sig_par_[2] = {0, 0};
    int sig_last_ = -1;
    void sig_need(int word);                   // make counter word `word` hold the latest (equalize launch if needed)
    void sig_flush(int r);                     // rank r's owed signal as its own launch
    void norm_in(int r, const float * w, bool pending, int T);   // [x += partials], cur = rmsnorm(x) * w, aq = a16(cur)
    // prefill pipeline (T > 8): token chunks on separate streams, tensor-parallel exchange of one chunk overlapping
    // the GEMMs of the others, weights unpacked to f16 ahead of use on a side stream (tp_model.cu)
    std::vector<float> forward_prefill(const std::vector<int32_t> & tokens);
    void pf_setup();                           // the prefill pipeline's buffers (lazy; prefill_setup() forces it)
    std::vector<float> forward_classic(const std::vector<int32_t> & tokens, int n_logits, bool keep_states,
                                       std::vector<int32_t> * ids = nullptr);
    // argmax of n_rows logit rows [n_rows][VL] of rank r (async on its stream) -> pinned amax_h_; after a sync of every
    // rank, argmax_collect() combines the ranks: global id = the rank's v0 + its index; rank 0's slice first and a later
    // slice wins only with a strictly larger value, so ties keep the lower id
    void argmax_launch(int r, const float * logits, int n_rows);
    std::vector<int32_t> argmax_collect(int n_rows) const;
    bool prefill_ok() const;
    // CUDA graphs of the fused step (P100E_GRAPH): decode_layers + output norm + lm_head (+ hn copy), captured once
    // per (T, keep_states) after the first eager run of that shape and replayed afterwards. Every launch in the body
    // reads its varying inputs from device memory (positions S.pos, the P2P allreduce counters), so a replay equals
    // the eager enqueue. The token-dependent inputs (set_pos, embed) and the result copies stay eager around it.
    // key: rows, sequences, keep_states, M-RoPE (images in a row's sequence), and for one sequence its slot (its state / page-table pointers are baked in)
    struct step_graph { int T = 0; int nseq = 1; int slot = -1; bool keep = false; bool mr = false; cudaGraph_t g = nullptr;
                        cudaGraphExec_t ge = nullptr; size_t nodes = 0; int buf_out = 0, cur_out = 0;
                        bool q8f = false; bool sig = false; int sig_in = -1, sig_out = -1; };   // P100E_AR_SIG_FUSE mode + counter word state
    void fused_body(int T, bool keep_states, int nseq = 1);   // decode_layers + output norm + lm_head (+ hn), rows 0..T-1
    step_graph * find_graph(int T, bool keep, int nseq = 1);
    bool capture_graph(int T, bool keep, int nseq = 1);   // streams idle; false (and eager from then on) when capture fails
    // capture body() (enqueues on every rank's stream) into one multi-device graph (fork from / join into rank 0's
    // stream); G.buf_out / cur_out get the allreduce parity a run of the body leaves. Streams idle. false on failure
    bool capture_body(const std::function<void()> & body, step_graph & G, std::string & err);
    void launch_graph(const step_graph & G);
    void drop_graphs(bool keep_only = false);   // keep_only: just the keep_states graphs (snapshot pointers)
    void dbg_trace(const char * tag, int il, int T);   // P100E_PF_DEBUG: sync + hash of the main buffers per rank

    gguf_file g_;
    int n_ctx_, max_batch_;
    // M-RoPE (forward_mm): the per-sequence state is in seq_t (rope_delta, mmt, mm_broken);
    // mm_ = the rows of the forward being staged (forward_mm's chunk): token or embedding row + rotary position each
    void check_ok() const;                       // throws when the active sequence is mm_broken
    struct mm_row { int32_t tok; const float * embd; int t, h, w; };
    const mm_row * mm_ = nullptr;
    int msec_[3] = {11, 11, 10}, msect_ = 32;    // rope.dimension_sections (pairs): t, h, w; sum incl. the 4th
    int32_t * rpos_h_ = nullptr;                 // pinned [3][max_batch] host copy of S.rpos
    // the step being staged uses M-RoPE positions (S.rpos): set by set_pos / forward_batch_ids / the batched MTP draft
    // when a staged row is an image row or belongs to a sequence with images (rope_delta != 0); text rows then get
    // t = h = w = row - rope_delta of their own sequence (identical to NEOX at that position)
    bool step_mr_ = false;
    bool mrope_now() const { return step_mr_; }
    // rotary rows of a step: row i of the staged step, sequence seq at KV row p (text) -> rpos_h_; returns true if non-NEOX
    bool stage_rpos_text(int i, const seq_t & q, int p);
    void upload_rpos(int T);                     // rpos_h_ rows 0..T-1 -> S.rpos on every rank (S.s order)
    fast::mrope_pos mrope(int r) const;          // S.rpos of rank r when mrope_now(), else {} (plain NEOX)
    void rope_rows(int r, float * x, int T, int H);   // rotary on x [T][H][HD] of rank r: NEOX at S.pos, or M-RoPE
    int last_T_ = 0;                           // tokens of the last forward_classic / forward_prefill (logits_rows)
    uint32_t kv_type_ = fast::KV_Q8_0;

    int n_nextn_ = 0, n_snap_ = 0;
    // rollback (MTP): one gather-copy kernel per GPU for every delta-net layer's state (P100E_RB_KERNEL=0: one
    // cudaMemcpyAsync per state, 192 calls), on a side stream that the next forward joins (P100E_RB_ASYNC=0: in order)
    bool rb_kernel_ = true, rb_async_ = true, rb_pending_ = false, ck_pending_ = false;
    void rb_join();                             // the main streams wait for a pending rollback copy
    int E_, L_, FF_, H_, HKV_, HD_, NROT_, CONV_K_, HK_, HV_, DK_, DV_, VOCAB_;
    tp_split split_;                           // per-rank ranges of the split dimensions (init, src/tp_split.h)
    float eps_, rope_base_;
    int full_attn_interval_;
    std::vector<rank_state> rk_;
    size_t scratch_elems_ = 0;
    bool use_fast_ = true;                     // P100E_FAST=0 disables the fast matvec path
    bool fuse_ = true;                         // P100E_FUSE=0: decode T=1 through the per-op path
    bool concat_ = true;                       // P100E_CONCAT=0: no concatenated weights (one launch per weight)
    bool ar_safe_ = false;                     // P100E_AR_SAFE=1: old allreduce handshake (wait for the peer before copying)
    bool plan_ok_ = false;                     // every decode weight has a fast path (fused decode possible)
    bool nextn_plan_ok_ = false;               // the nextn block has a fused plan (S.lp[L_]) and eh_proj a fast path
    bool p2p_ar_ = true;                       // P100E_P2P_AR=0: event allreduce in the fused decode (default: one-shot P2P, flag + P2P loads)
    bool gdn_batch_ = true;                    // P100E_GDN_BATCH=0: keep_states delta net one token at a time + snapshot copies
    bool concat_t_ = true;                     // P100E_CONCAT_T=0 / P100E_VERIFY_CAT=0: T > 1 fused steps run one in-projection launch per weight
    int ar_buf_ = 0, ar_cur_ = 0;              // allreduce double buffer: next / last written peer buffer
    size_t peer_stride_ = 0;                   // floats per peer buffer
    float * logits_h_ = nullptr;               // pinned host logits (both GPUs' D2H copies run async, overlapped)
    int32_t * pos_h_ = nullptr;                // pinned host positions (= desc_h_ + DESC_POS)
    void * amax_h_ = nullptr;                  // pinned host argmax pairs [R][8]
    int * ar_status_h_ = nullptr;              // P2P allreduce: host-mapped status line per GPU ([R][16]; fast::p2p flag block)
    void check_ar() const;                     // throws when a P2P allreduce wait timed out (called after every step's sync)
    char * emb_h_ = nullptr;                   // pinned host token-embedding rows [max_batch] (prefill upload, lazy)
    mutable std::atomic<int> pf_ok_{-1};       // prefill_ok() verdict for this instance (-1: not checked yet; also read
                                               // by idle_scratch() on the borrower's thread)
    lend_gate ring_gate_;                      // the prefill ring: forward_prefill (owner) vs idle_scratch() (borrower)
    bool pf_wire16_off_ = false;               // the f16 prefill wire saturated a partial once: f32 wire from then on
    bool pf_sat_last_ = false;                 // last_prefill_saturated()
    uint64_t pf_sat_total_ = 0;                // prefill_saturations()
    float pf_wire_amax_ = -1.0f;               // last_prefill_wire_amax()
    std::vector<size_t> mv_scratch_bytes_;
    int graph_maxt_ = 8;                       // P100E_GRAPH: fused steps with T <= this replay a CUDA graph (0 = eager)
    bool graph_failed_ = false;                // a capture failed: eager from then on
    cudaEvent_t ev_fork_ = nullptr;            // CUDA graph: rank 0's stream forks to the others (device 0 event)
    std::vector<step_graph> graphs_;
    // net segment graphs (P100E_NET_GRAPH): one set per (T, slot, M-RoPE, q8 fork)
    struct net_gset {
        int T = 1, slot = -1; bool mr = false, q8f = false; std::vector<step_graph> seg;
        uint64_t gl_out = 0, gl_in = 0, gl_calls = 0;   // gl_ traffic of one replay (counted at capture)
    };
    std::vector<net_gset> net_gsets_;
    std::vector<unsigned long long> whash_total_;   // P100E_WHASH: running fingerprint of every loaded tensor per rank
};

} // namespace p100e
