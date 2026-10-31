// Qwen3.8-Flash-Next (GGUF arch "qwen4exp") on 2 x P100: hyper-connection residual (4 streams), PLE n-gram embedding
// (one layer), gated delta-net layers (sigmoid output gate), gated full attention (QSA budget: dense below it), MoE FFN
// (512 experts top-10 + shared expert) from a hot/cold expert-split file .
//
// v0: correctness first, batched over T rows from the start
// (T = 1 decode, T <= FN_MAX_T prefill chunk: prompt processing matters as much as decode):
//  - dense weights replicated on every GPU (fast::pack_weight + matvec for T <= 8, the fold GEMM above), f32 glue kernels
//  - hot experts split by expert id across the GPUs (rank r owns hot local ids [r*HOTR, (r+1)*HOTR) on its GPU), every
//    other expert of the rank pre-packed in pinned host RAM and staged per use; the rows routed to one expert are gathered
//    and run as one T_e-row call; the FFN partials are summed with one allreduce per layer
//  - attention: dense GQA over an f32 KV cache (exact while n_kv <= indexer top-k + ratio - 1, i.e. the QSA budget
//    selects every cell); the indexer / top-k / gather path comes later
//  - routing on the host (router logits read back per layer)
// The qwen35 engine (tp_model) is untouched.
#pragma once

#include "gguf.h"
#include "fast/matvec.cuh"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <map>
#include <memory>
#include <functional>
#include <chrono>
#include <string>
#include <deque>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include <set>
#include <string>
#include <vector>
#include <memory>

namespace p100e {

class net_transport;   // src/pdl/pdl_link.h

constexpr int FN_MAX_T = 64;        // default rows per forward (prefill chunk); opts::max_t raises it
constexpr int FN_MAX_LOGITS = 16;   // logits rows per forward (n_logits is clamped to this)
constexpr int MTP_ROWS_DEFAULT = 4;   // = fn_model::MTP_ROWS (opts default of rexp_max_t)

class fn_model {
public:
    struct opts {
        int n_ctx = 4096;
        int max_t = FN_MAX_T;       // max rows per forward_n (prefill chunk): host-staged experts are fetched once per
                                    // chunk per layer, so prefill throughput scales with it (activation buffers ~T * 1 MB)
        int hot_per_rank = -1;      // GPU-resident hot experts per layer per rank (-1: the file's hot set split evenly)
        bool verbose = false;
        int layer_limit = -1;
        int stage_mb = 64;          // expert staging arena (2 sets) in MiB (P100E_FN_STAGE_MB); a batch = the experts that fit one set
        int prefetch_min_t = 256;   // chunks of at least this many rows prefetch EVERY host expert of the next layer while the
                                    // current layer computes (two full-layer arenas; P100E_FN_PREFETCH_T, 0 = off)
        bool tp = true;             // dense attention / delta-net blocks split by heads across the ranks (P100E_FN_TP=0: replicated)
        // adaptive resident experts (files with a low-bit copy of EVERY expert, exporter --cold-all): hit counts per expert,
        // between tokens the least-used resident expert gives its slot to the most-used non-resident one (its Q4 bytes
        // are read from the GGUF). P100E_FN_ADAPT=0 off, _EVERY tokens between adapt steps, _MAX swaps per layer per step
        bool adapt = true;
        int adapt_every = 8, adapt_max = 8, adapt_max_prefill = 32;   // swaps per layer per step (decode / after a prefill chunk)
        float adapt_min = 2.0f;                                        // routed rows since the last decay an expert needs to be promoted
        bool pin_host = true;       // host experts in pinned RAM (P100E_FN_PIN=0: pageable, for boxes short on RAM)       // debug: run only the first N layers (smoke tests on small GPUs; logits are meaningless)
        int64_t ram_reserve_mb = -1;   // host RAM left free after load (ram_guard.h): -1 = P100E_RAM_RESERVE_MB or 12288 (Windows) / 6144 MiB, 0 = off
        bool mtp = true;            // load the nextn draft block (blk.<L>) for MTP speculative decoding (P100E_FN_MTP=0 off)
        // per-rank resident expert slots: empty = hot_per_rank on every rank (unchanged default); one
        // value per local rank (uneven GPUs, e.g. 3070 + 1660); a value -1 = auto: what fits that rank's GPU after
        // everything else of the byte plan (see plan_rank)
        std::vector<int> rank_slots;
        // hybrid pipeline over two machines (see the pipe section below): net_nodes = "host:port" of node 0
        // and node 1; node 0 runs layers [0, pipe_layers) + the token embedding (+ the PLE layer) and is a WORKER
        // (pipe_serve()), node 1 runs layers [pipe_layers, L) + output head + the MTP draft block and DRIVES (forward_n,
        // rollback, reset send commands to node 0). 0 = off; -1 = auto (both nodes' free VRAM / RAM exchanged at connect).
        std::vector<std::string> net_nodes;
        int net_node = 0;
        int pipe_layers = 0;
        int pipe_total = 0;          // --pipe-split a,b: a + b must equal the model's layer count (0 = not given)
        uint64_t net_args_hash = 0;  // the caller's run arguments that must agree across the nodes
        int net_connect_s = 600, net_timeout_s = 120;
        bool plan_only = false;      // parse the hyperparameters only (no GPU work): print_plan() for an offline byte plan
        bool mtp_with_limit = false; // debug: keep the draft block (file blk.<L>) under a layer_limit (pipe / rollback tests)
        // remote expert tier (src/fn_rexp.h): node 0 listens on rexp_listen ("host:port", host may be
        // empty / 0.0.0.0) for p100e-fn-expert-server, which holds Q4 experts this node has not resident and computes
        // them per layer per forward (one round trip). Forwards of more than rexp_max_t rows (prefill chunks) stay local.
        // rexp_sum: node 1 returns one weighted sum per row (fewer bytes, not bit-equal to the local path). Empty = off.
        std::string rexp_listen;
        int rexp_max_t = MTP_ROWS_DEFAULT;
        bool rexp_sum = false;
        int rexp_connect_s = 1800, rexp_timeout_s = 120;
    };
    // ---- hybrid pipeline ----
    // Driver / worker instead of SPMD: node 1 (the driver: head + draft block, where sampling and MTP decisions happen)
    // calls the ordinary API; every forward_n sends one command (op, rows, n_past, step, the pending rollback, the
    // tokens) to node 0, which embeds the tokens, runs its layers and returns the wide residual rows [T][HC][E] f32
    // (exact: the same bits a one-process run hands from layer k-1 to layer k). rollback() / reset() are queued and ride
    // on the next command (no extra round trip per MTP step). Node 0 keeps its own copy of the token history (PLE
    // n-grams) and its own delta-net / PLE snapshots, so a rollback is exact on both sides.
    bool pipe() const { return (bool) pnet_; }
    bool pipe_worker() const { return pnet_ && le_ < L_; }
    int layer_begin() const { return lb_; }
    int layer_end() const { return le_; }
    void pipe_serve();          // node 0: run commands until the driver quits (returns) or the link fails (throws)
    void pipe_quit();           // node 1: tell node 0 to return from pipe_serve (also done by the destructor)
    // the byte plan of one node: layers [l0, l1) (+ head + draft block when l1 == L), R ranks with the given free VRAM
    // per rank (bytes; ranks sharing a GPU pass their share), host RAM free; slots per rank (-1 = auto) resolved into
    // `slots_out`. Fits = every rank's total <= its free VRAM and the pinned host RAM + `reserve` <= host_free (reserve
    // < 0: this machine's P100E_RAM_RESERVE_MB / opts::ram_reserve_mb, default 12288 MiB on Windows, 6144 on Linux).
    struct rank_plan { double dense = 0, slots_b = 0, kv = 0, state = 0, head = 0, buf = 0, total = 0, free = 0; int slots = 0, max_slots = 0; };
    struct node_plan { std::vector<rank_plan> rk; double host = 0, host_free = 0, reserve = 0; bool fits = false; double resident = 0; };
    node_plan plan_node(int l0, int l1, bool embed, const std::vector<double> & gpu_free, const std::vector<int> & slots_req,
                        double host_free, bool pipe_mode = false, double reserve = -1) const;
    double ram_reserve_bytes() const;   // RAM left to the OS and other programs (P100E_RAM_RESERVE_MB, opts::ram_reserve_mb)
    void print_plan(const char * tag, int l0, int l1, const node_plan & p) const;
    // offline plan (plan_only instances): best k for the given nodes, both nodes' plans printed; returns k (-1: none)
    int plan_split(const std::vector<double> & free0, double host0, const std::vector<int> & slots0,
                   const std::vector<double> & free1, double host1, const std::vector<int> & slots1, int k_req, bool print,
                   double res0 = -1, double res1 = -1) const;
    // MTP (generate_mtp): the draft block = one full-attention + MoE block fed by (token at p, the trunk's wide residual
    // at p-1) through eh_proj(concat(enorm(embed), hnorm(h))); chained drafts feed the block's own residual. Verification
    // runs [last token + k drafts] through the trunk in one batch (T <= MTP_ROWS), keeps the longest matching prefix and
    // rolls the recurrent state back (delta-net per-row snapshots, PLE history snapshot; attention KV is positional).
    static constexpr int MTP_ROWS = 4;
    struct mtp_stats { int steps = 0, drafted = 0, accepted = 0, n_draft = 0; double t_prompt = 0, t_catchup = 0, t_draft = 0, t_verify = 0; };
    bool has_mtp() const { return mtp_il_ >= 0; }
    // greedy generation with MTP drafts (k drafts per step, 1..MTP_ROWS-1): the prompt in `chunk`-row forwards, then
    // n_gen tokens; bit-equal to plain greedy decoding by construction (the trunk verifies every emitted token)
    std::vector<int32_t> generate_mtp(const std::vector<int32_t> & prompt, int n_gen, int chunk, int k, mtp_stats & st);
    // keep the first n_keep rows of the last T-row forward (1 <= n_keep <= T): states rolled back, n_past reduced
    void rollback(int T, int n_keep);
    enum { MTP_H_MAIN = 0, MTP_H_SELF = 1, MTP_H_READY = 2 };
    // run the draft block on tokens at positions p0..; h rows: MAIN = the last trunk forward's hn rows 0..T-1, SELF = row 0
    // from the block's own last residual, READY = row 0 from the carry. Returns the last row's greedy token.
    int32_t mtp_run(const std::vector<int32_t> & tokens, int p0, int hsrc);
    // prompt catch-up right after forward_n(tokens) at p0: entry p0+i = (tokens[i], h_{p0+i-1}), row 0 from the carry
    void mtp_catchup(const std::vector<int32_t> & tokens, int p0);
    void mtp_h_zero();   // carry = 0 (a fresh context)
    // Multimodal rows (images from the vision encoder: [nx * ny][E] f32 host rows, raster order) between text pieces, as
    // tp_model::forward_mm: Qwen3-VL interleaved M-RoPE (rope.dimension_sections) with image token i of an nx x ny grid
    // placed at P at (t, h, w) = (P, P + i / nx, P + i % nx), the text after it continuing at P + max(nx, ny); the PLE
    // n-gram sees ple.image_token_id for image rows (as the reference's KV cell tokens). Runs max_t-row chunks through
    // forward_n with the chunk's rows staged; after_chunk(p0, chunk tokens) runs after each chunk with the rows still
    // staged (the MTP catch-up reads the image embeddings through it). Returns the last row's logits.
    struct mm_in { const int32_t * tokens = nullptr; int n_tok = 0; const float * embd = nullptr; int nx = 0, ny = 0; int n_embd = 0; };
    std::vector<float> forward_mm(const std::vector<mm_in> & pieces, const std::function<void(int, const std::vector<int32_t> &)> & after_chunk = {});
    int rope_pos() const { return n_past_ - rope_delta_; }   // rotary position of the next text token
    // PLE table prefetch: the per-layer-embedding rows of `tokens` (the rows about to be fed at n_past_..) are asked from
    // the OS on a background thread (PrefetchVirtualMemory batches), so a prompt's disk reads overlap the GPU work of
    // the earlier chunks instead of stalling ple() row by row (2.7 s per 2048 first-seen rows on a SATA disk). Pure
    // scheduling: the bytes read are the same, the output is bit-identical. Called by forward_mm / generate_mtp /
    // engine_fn::prefill with the whole prompt; harmless when the rows are already cached.
    void ple_prefetch(const std::vector<int32_t> & tokens);
private:
    int32_t mtp_block(const std::vector<int32_t> & tokens, int p0, bool want_logits);
public:
    fn_model(const std::string & path, const std::vector<int> & devices, const opts & o);
    ~fn_model();

    // T tokens at positions n_past().. ; returns the logits of the last n_logits rows [n][vocab] (host)
    std::vector<float> forward_n(const std::vector<int32_t> & tokens, int n_logits);
    std::vector<float> forward1(int32_t token);
    int32_t forward1_argmax(int32_t token);   // greedy token (ties: lower index)
    void reset();                             // empty context
    int n_past() const { return n_past_; }
    int vocab() const { return VOCAB_; }
    int n_embd() const { return E_; }
    int n_layer() const { return L_; }

    // debug dump of named activations of the LAST forward (rank 0; set_dump before the call). Names follow the llama.cpp
    // reference's cb() names: model.input_embed, ple_embd, ple_gate-<il>, hc_mixed-<il> (attention-side mixer),
    // hc_mixed_ffn-<il> (FFN-side), attn_output-<il> ([T][HV][DV] delta-net core output on linear layers, [T][E] on
    // attention layers), linear_attn_out-<il>, ffn_moe_out-<il> (rank 0's partial), ffn_out-<il>, l_last-<il>
    // ([T][HC][E]), result_norm, result_output
    void set_dump(bool on) { dump_ = on; }
    const std::map<std::string, std::vector<float>> & dump() const { return dumped_; }
    const gguf_file & gguf() const { return g_; }

private:
    struct dw {   // a GPU weight
        fast::packed_weight pk;
        void * d = nullptr;
        size_t bytes = 0;
        bool owns = true;
        bool raw = false;   // no GEMV kernel for this type / shape: dequantized to f32 per use + cuBLAS (correctness fallback)
    };
    struct expert_set {   // one projection of a set of experts of a layer
        uint32_t type = 0;
        int64_t n_in = 0, n_out = 0;
        size_t row_bytes = 0, bytes = 0;   // per expert
        int n = 0;                           // experts in the set
        char * base = nullptr;               // resident: device [n][bytes]; host: pinned (or pageable) [n][stride] of packed bytes
        size_t stride = 0;                   // bytes between experts: resident = bytes; host = the gate|up|down record of the expert
        bool pinned = false, owner = true;   // host triples share one buffer (the gate set owns it)
        int layout = 0;                      // fast layout of every expert slice
        bool raw = false;                    // see dw::raw
        std::vector<fast::packed_weight> pk; // resident: per local expert (views into base)
    };
    struct layer_w {
        bool attn = false;
        const float *hc_attn_norm = nullptr, *hc_attn_inject = nullptr, *hc_ffn_norm = nullptr, *hc_ffn_inject = nullptr;
        dw hc_attn_down, hc_attn_up, hc_ffn_down, hc_ffn_up;
        dw attn_q, attn_k, attn_v, attn_output;
        const float *attn_q_norm = nullptr, *attn_k_norm = nullptr;
        dw attn_qkv, attn_gate, ssm_beta, ssm_alpha, ssm_out;
        const float *ssm_conv1d = nullptr, *ssm_dt = nullptr, *ssm_a = nullptr, *ssm_norm = nullptr;
        dw router;                                   // ffn_gate_inp F32 [n_expert][E]
        const float * shexp_gate = nullptr;          // ffn_gate_inp_shexp [E]
        dw shexp_gate_w, shexp_up_w, shexp_down_w;   // shared expert
        expert_set hot_gate, hot_up, hot_down;       // this rank's resident hot experts
        expert_set hhot_gate, hhot_up, hhot_down;    // this rank's hot experts without a resident slot (host)
        expert_set cold_gate, cold_up, cold_down;    // this rank's cold experts (host)
        size_t host_bytes = 0, pf_hh_off = 0;        // bytes of this rank's host records (cold + hhot); hhot offset in a prefetch arena
        std::vector<int> hot_map, cold_map;          // router id -> local id in THIS rank's set, -1 = not here
        bool adaptive = false;                       // cold copies of every own expert exist: resident slots can be swapped
        std::vector<int> slot_expert;                // resident slot -> router id
        std::vector<float> cnt;                      // [NEXP] routed rows since the last adapt step (decayed)
        int64_t hits = 0, rows = 0, swaps = 0;       // resident rows / all rows / swaps done (stats)
        std::vector<char> slot_busy;                 // a promotion is in flight for this slot
        std::vector<char> inflight;                  // [NEXP] P100E_FN_ADAPT_ASYNC: queued or landing, not resident yet
        std::map<int, int> hhot_idx;                 // router id -> host-hot local id
        int hhot_map(int e) const { auto it = hhot_idx.find(e); return it == hhot_idx.end() ? -1 : it->second; }
        void hhot_map_set(int e, int j) { hhot_idx[e] = j; }
        dw ple_key, ple_value;
        const float *ple_norm_key = nullptr, *ple_norm_query = nullptr, *ple_norm_conv = nullptr, *ple_conv1d = nullptr;
        dw idx_q, idx_k;                                             // QSA indexer projections [E -> IDX_H*IDX_D], [E -> IDX_D]
        const float *idx_q_norm = nullptr, *idx_k_norm = nullptr;
        dw eh_proj, hc_head_down, hc_head_up;                        // nextn (draft block only)
        const float *enorm = nullptr, *hnorm = nullptr, *hc_head_norm = nullptr;
    };
    struct rank_state {
        int dev = 0;
        int Hr = 0, HKVr = 0, HKr = 0, HVr = 0, CONVr = 0, h0 = 0, hkv0 = 0, hk0 = 0, hv0 = 0;   // this rank's head counts / first heads (tp)
        cudaStream_t s = nullptr;
        cublasHandle_t cb = nullptr;
        cudaEvent_t ev = nullptr, ev2 = nullptr;
        cudaStream_t cs = nullptr;                      // expert staging copies (host -> qstage), overlapped with compute
        cudaStream_t ps = nullptr; cudaEvent_t ps_ev = nullptr;   // P100E_FN_ADAPT_ASYNC: promotion copies + packs (lowest priority)
        cudaEvent_t st_ready[2] = {}, st_free[2] = {};  // staging set k: copies landed / compute done with it
        int32_t * hidx_pin = nullptr; float * hw_pin = nullptr;   // pinned [max_t * NEXPU] gather rows / weights per layer
        std::map<std::string, float *> f;     // small f32 tensors
        std::vector<dw> dense;                // owners
        std::vector<layer_w> lw;
        dw output, output_hc_down, output_hc_up;
        const float * output_hc_norm = nullptr;
        float *res = nullptr;      // hyper-connection residual [T][HC][E]
        float *cur = nullptr;      // mixed stream [T][E]
        float *xn = nullptr;       // hc norm [T][HC*E]
        float *a = nullptr, *b = nullptr, *c = nullptr, *d = nullptr, *e = nullptr, *g = nullptr;   // wide scratch [T][...]
        float *part = nullptr;     // block output / FFN partial [T][E]
        float *peer = nullptr;     // the other rank's partial [T][E]
        float *inject = nullptr;   // [T][HC]
        float *router_out = nullptr;   // [T][n_expert]
        float *logits = nullptr;   // [T][VOCAB]
        float *gath = nullptr, *gout = nullptr, *gw = nullptr;   // MoE: gathered rows [T][E], expert outputs [T][E], weights [T]
        int32_t *gidx = nullptr;   // MoE: token row of each gathered row
        void * gtab = nullptr;     // MoE grouped path: device mv_group tables, 2 sets x 3 projections x [max_t * NEXPU]
        void * tab_pin[2] = {};    // pinned host copies of the two table sets
        void * q2 = nullptr;       // MoE grouped path: a16 of a batch's silu outputs (down-projection input columns)
        void * qa = nullptr;       // MoE grouped path: a16 of the gathered rows [max_t * NEXPU] (gate / up input columns)
        int32_t * rowof = nullptr; int32_t * rowof_pin = nullptr;   // [max_t][NEXPU] gathered-row index per (token, slot), -1 = other rank
        char * pf[2] = {};                       // full-layer prefetch arenas: this rank's cold records then its hhot records
        size_t pf_bytes = 0;                     // arena size = the largest layer's host set
        cudaEvent_t pf_ready[2] = {}, pf_free[2] = {};
        int pf_layer[2] = {-1, -1};              // layer held by each arena (-1 = none)
        struct stage_stat { int64_t fetches = 0; std::set<int> seen; };
        std::map<int, stage_stat> stage_stats;                                // verbose: host-expert use per layer
        int32_t * route_ids = nullptr; float * route_wts = nullptr;           // GPU router output [max_t][NEXPU]
        int32_t * route_ids_pin = nullptr; float * route_wts_pin = nullptr;   // pinned host copies
        void * mv_scratch = nullptr;
        void * gemm_scratch = nullptr; size_t gemm_bytes = 0;
        void * qstage = nullptr;   // raw weight staging (host expert rows, embeddings, small tensors)
        char * hbounce = nullptr;  // pinned host bounce for expert loads read with gguf_file::read_at (one expert slice)
        size_t qstage_bytes = 0;
        float * w_scratch = nullptr; size_t w_scratch_bytes = 0;
        std::vector<void *> kc, vc;              // attention layers: KV [n_ctx][HKV][HD], f32 or f16 (kv_f16_)
        float *kf = nullptr, *vf = nullptr;      // f16 cache: f32 window of rows [0, N) for the dense paths (N <= QSA_W + max_t)
        std::vector<float *> conv_state, ssm_state;   // delta-net layers
        float * ple_hist = nullptr;              // PLE conv history [PLE_HIST + FN_MAX_T][HC*E]
        float * ple_emb = nullptr;               // [T][E]
        int32_t * pos = nullptr;                 // [T] cell index (QSA visibility)
        int32_t * rpos = nullptr;                // [3][max_t] rotary positions (t, h, w) of the rows (text: all = cell - rope_delta)
        // QSA (fn_qsa.cuh): per attention layer the block keys [n_ctx / ratio][IDX_D] and the raw keys of the incomplete
        // tail [ratio][IDX_D]; scratch: raw keys of the chunk [T][IDX_D], indexer queries [T][IDX_H][IDX_D], per-slab
        // head scores [QSA_SL * IDX_H][NB] and combined scores [QSA_SL][NB], selected cells [T][W] + counts [T]
        std::vector<float *> kblk, ktail;
        int32_t * blk_rpos = nullptr;            // [3][NB] rotary positions of each block's first cell (indexer key rope)
        float *qk_raw = nullptr, *qq = nullptr, *qsa_s = nullptr, *qsa_c = nullptr;
        int32_t *sel = nullptr, *seln = nullptr;
        // MTP: hn [max_t][HC*E] = the wide residual after the last trunk layer of the last forward; mhin [max_t][HC*E] the
        // draft block's h input rows; mcarry / mself [HC*E]; mcat [max_t*HC][2E] concat scratch; per delta-net layer
        // MTP_ROWS-1 state snapshots (after row t); ple_snap [PLE_HIST][HC*E]
        float *hn = nullptr, *mhin = nullptr, *mcarry = nullptr, *mself = nullptr, *mcat = nullptr, *ple_snap = nullptr;
        std::vector<float *> conv_snap, ssm_snap;
    };
    void load(const std::vector<int> & devices);
    dw upload_weight(rank_state & S, const gguf_tensor * t, const char * what, int64_t expert = -1, int64_t rows0 = 0, int64_t nrows = -1);
    dw upload_raw(rank_state & S, uint32_t type, const uint8_t * src, int64_t n_in, int64_t n_out, const char * what);
    // host -> device copy; a source inside the file mapping is read with ReadFile through the bounce buffer (no faults)
    void h2d(rank_state & S, void * dst, const uint8_t * src, size_t n);
    // concatenated row ranges / a block-aligned column range of a 2-D tensor (tensor-parallel slices)
    dw upload_rows(rank_state & S, const gguf_tensor * t, const char * what, const std::vector<std::pair<int64_t, int64_t>> & ranges);
    dw upload_cols(rank_state & S, const gguf_tensor * t, const char * what, const std::vector<std::pair<int64_t, int64_t>> & cols);
    const float * small_f32(rank_state & S, const std::string & key, const std::vector<float> & v);   // device copy of host f32 data
    const float * small(rank_state & S, const std::string & name, bool required = true);
    void load_experts(rank_state & S, int r, int il);
    void mv(rank_state & S, const dw & w, const float * x, float * y, int T);
    void gemm_pk(rank_state & S, const fast::packed_weight & pk, bool raw, const float * x, float * y, int T);
    static bool needs_raw(uint32_t type, int64_t n_in);
    void hc_mix(rank_state & S, const float * norm_w, const dw & down, const dw & up, const float * inject_w, float * out, float * inject, int T);
    void hc_combine(rank_state & S, const float * block_out, const float * inject, int T);
    void ple(rank_state & S, int il, const std::vector<int32_t> & tokens);
    void layer_attn(rank_state & S, int il, const float * cur, float * out, int T);
    // one block (hc mixers, attention / delta-net, MoE FFN, allreduces) over the rows in S.res; the trunk and the draft block
    void run_layer(int il, int T, const std::vector<int32_t> & tokens, bool prefetch, std::vector<int> & ids, std::vector<float> & wts);
    void layer_gdn(rank_state & S, int il, const float * cur, float * out, int T);
    void layer_ffn(rank_state & S, int r, int il, const float * cur, float * out, int T, const std::vector<int> & ids, const std::vector<float> & wts);
    void route(int il, int T, std::vector<int> & ids, std::vector<float> & wts);
    // any T: this rank's (expert, token) pairs as grouped launches (one group per pair, batched by a staging arena);
    // returns false when a used expert set's layout has no grouped kernel (the caller then runs per-expert calls)
    // cpu_e (P100E_FN_CPU, [NEXP] flags): those experts' rows are numbered after the GPU rows and not computed; the
    // combine is left to the caller (nr = {GPU rows, all rows}). nullptr: as before.
    bool ffn_grouped(rank_state & S, int il, const float * cur, float * out, int T, const std::map<int, std::vector<std::pair<int, float>>> & by_e,
                     const std::vector<char> * cpu_e = nullptr, int * nr = nullptr);
    void prefetch_layer(rank_state & S, int il);   // copy the layer's whole host set into arena il & 1 (copy stream)
    void adapt(bool big);                           // swap resident experts by hit counts (between forwards)
    // asynchronous promotions: a worker thread reads the expert's Q4 slices from the GGUF into a pinned ring entry; the
    // next adapt() copies landed entries to their slots on the copy stream and packs them (the GPU does it before the
    // next forward's work on the same streams), then updates the maps
    struct promo { int rank, il, slot, e, ring; };
    struct ring_entry {
        char * buf = nullptr; cudaEvent_t done[2] = {nullptr, nullptr}; std::atomic<bool> used{false}; bool landed = false; int landed_rank = 0;
        ring_entry() = default;
        ring_entry(const ring_entry & o) : buf(o.buf), used(o.used.load()), landed(o.landed), landed_rank(o.landed_rank) { done[0] = o.done[0]; done[1] = o.done[1]; }
        ring_entry & operator=(const ring_entry & o) { buf = o.buf; done[0] = o.done[0]; done[1] = o.done[1]; used.store(o.used.load()); landed = o.landed; landed_rank = o.landed_rank; return *this; }
    };
    std::vector<ring_entry> ring_;
    std::deque<promo> promo_todo_, promo_ready_;
    std::mutex promo_mu_; std::condition_variable promo_cv_;
    std::thread promo_thread_; bool promo_stop_ = false;
    size_t promo_bytes_ = 0;                        // bytes of one expert's three slices (gate | up | down, each 256-aligned)
    void promo_worker();
    void promo_start();
    void promo_land(const promo & p);
    // P100E_FN_ADAPT_ASYNC=1 (off by default): promo_land queues the copy + pack on the rank's promotion stream after
    // the compute already queued (the old expert leaves the maps at once, the new one joins them only when its copy is
    // done, polled by promo_finalize at each forward / adapt step), so the compute stream never waits for PCIe copies
    bool adapt_async_ = false;
    std::vector<promo> lands_;                      // landing promotions (async mode)
    void promo_finalize();
    void promote(rank_state & S, int il, int slot, int e);   // expert e's Q4 bytes (GGUF) -> resident slot
    void allreduce(int T);
    void grab(const char * name, int il, int r, const float * x, int64_t n);
    void sync_all();

    gguf_file g_;
    opts o_;
    int E_, L_, VOCAB_, HC_, HCDIM_, HCLR_, H_, HKV_, HD_, NROT_, HK_, HV_, DK_, DV_, CONV_C_, NEXP_, NEXPU_, FFE_, FFS_;
    int PLE_HEADS_ = 0, PLE_DIM_ = 0, PLE_NGRAM_ = 3, PLE_PER_ = 8, PLE_KERN_ = 4, PLE_HIST_ = 0, PLE_LAYER_ = -1;
    int64_t ple_eos_ = 0;
    std::vector<uint64_t> ple_mult_;
    std::vector<int64_t> ple_off_, ple_vocab_;
    float eps_, rope_base_, exp_scale_;
    int full_attn_interval_;
    // QSA: indexer heads / head dim / top-k cells, block ratio (0 = dense), W = top_k + ratio - 1 cells per query,
    // SL = query rows per scoring slab (bounds the [SL * IDX_H][NB] score scratch); P100E_FN_QSA=0 forces dense
    int IDX_H_ = 0, IDX_D_ = 0, IDX_TOPK_ = 0, RATIO_ = 0, QSA_W_ = 0;
    static constexpr int QSA_SL_ = 32;
    static constexpr int QSA_RING_ = 16;   // raw indexer keys kept per layer (>= ratio + MTP_ROWS: a rollback re-pools its block)
    static constexpr size_t HBOUNCE_ = 8u << 20;   // bytes of the per-rank pinned bounce buffer for expert loads
    bool qsa_ = false;
    int kv_mode_ = 1;       // KV cache storage: 0 f32, 1 f16, 2 q8 (P100E_FN_KV=f32|f16|q8; default q8 when n_ctx > 32768 else
                            // f16); the gathered attention reads it directly, the dense paths read an f32 copy of the window
    int mtp_il_ = -1;       // the draft block's layer index (L_) when loaded, else -1
    int NL_ = 0;            // layers with per-layer state: L_ (+ 1 with the draft block)
    int n_past_ = 0;
    std::vector<int32_t> toks_;
    // multimodal state: the staged chunk's rows (forward_mm), per-cell rotary positions, the rotary delta of the text
    // after the images fed so far (cell - rope_delta_), rope sections (pairs) and the PLE image token
    struct mm_row { int32_t tok; const float * embd; int t, h, w; };
    const mm_row * mm_ = nullptr;
    int mm_p0_ = 0, mm_n_ = 0;
    bool mm_seen_ = false;
    int rope_delta_ = 0;
    std::vector<int32_t> rp_[3];
    int32_t img_tok_ = 0;
    int msec_[3] = {11, 11, 10}, msect_ = 32;
    int rot(int cell, int s) const;
    void ple_rowids(const std::vector<int32_t> & tokens, int p0, std::vector<int64_t> & rowid) const;   // n-gram hash rows of tokens at p0..
    std::thread ple_th_;
    // P100E_FN_TIME=1: per-phase wall time of a forward (every rank synchronized at each phase boundary, so only for
    // profiling): printed after each forward_n as 'fn time T=..: phase ms ...'
    bool time_on_ = false;
    std::map<std::string, double> tm_;
    std::chrono::steady_clock::time_point tm_t0_;
    void tm_mark(const char * phase);
    // P100E_FN_TIME=2: no device-wide sync per phase; an event is recorded on rank 0's stream at each mark and the
    // phases are the GPU-timeline gaps between them (host waits show up in the phase that waited), read at the end
    int time_mode_ = 0;
    std::vector<std::pair<std::string, cudaEvent_t>> tm_ev_;
    std::vector<cudaEvent_t> tm_pool_;
    void tm_start();
    void tm_flush();
    // P100E_FN_PROF_HC=1: GPU time of each hc_mix kernel on rank 0 (events between them, read after each forward,
    // averages printed at exit): rmsnorm, down matvec, silu_scale, up matvec, mixed, inject dot
    bool prof_hc_ = false;
    std::vector<std::vector<cudaEvent_t>> hc_ev_;
    size_t hc_used_ = 0;
    double hc_sum_[6] = {}; int64_t hc_n_ = 0;
    void hc_prof_flush();
    int adapt_cap_ = 0;   // P100E_FN_ADAPT_CAP: promotions per rank per adapt step, best gain first (0 = only the per-layer limit)
    void stage_pos(rank_state & S, int p0, int T);
    void stage_embed(rank_state & S, const std::vector<int32_t> & tokens, int p0);
    void rope_rows(rank_state & S, float * x, int T, int H, int D);
    void rope_blocks(rank_state & S, float * kb, int b0, int n);
    std::vector<rank_state> rk_;
    bool dump_ = false;
    std::map<std::string, std::vector<float>> dumped_;
    std::vector<int32_t> hot_count_;
    std::vector<int32_t> rank_order_;   // [L][NEXP] experts best first (expert_split.rank_order), empty if absent
    int since_adapt_ = 0;   // per layer (file)
    // ---- CPU expert pool (P100E_FN_CPU=1; fn_cpu_pool.h): this layer's non-resident experts are
    // computed in host RAM by the CPU pool while the GPUs run the resident ones; the CPU rows join the GPU's combine
    struct cpu_moe;
    std::unique_ptr<cpu_moe> cpu_;
    // ---- remote expert tier: node 1's experts join ffn_grouped's combine like the CPU pool's rows
    struct rexp_moe;
    std::unique_ptr<rexp_moe> rexp_;
    void rexp_connect(int R);                  // constructor, before load(): accept node 1, agree, send its expert set
    void rexp_ready();                         // after load(): wait until node 1 has loaded, report the coverage
    std::vector<char> rexp_resident0(int il, int R) const;   // the experts load_experts() makes resident on any of R ranks (same rules)
    void rexp_begin(int il, int T, const std::vector<int> & ids, const std::vector<float> & wts);   // layer_ffn, rank 0: pick + send
    void rexp_finish(int il, int T);                                                                // layer_ffn, last rank: receive + combine
    bool rexp_held(int il, int e) const;       // node 1 holds expert e of layer il
    bool rexp_covered(int il, int e) const;    // ... or a swap to it is in flight (adapt(): not promoted here)
    void rexp_adapt();                         // adapt(): node 1's set follows this node's routing counts (SWAP)
    void cpu_init();
    void ram_reserve_check(int R);   // ram_guard.h: refuse a load whose host experts would leave less than the reserve
    void cpu_stage_x(rank_state & S, int T);   // route(): rank 0's FFN input rows -> pinned host (D2H on S.s)
    void cpu_begin(int il, int T, const std::vector<int> & ids, const std::vector<float> & wts);   // layer_ffn, rank 0: pick + submit
    void cpu_finish(int il, int T);                                                                // layer_ffn, last rank: wait + combine
    void ffn_by_e(rank_state & S, int il, int T, const std::vector<int> & ids, const std::vector<float> & wts,
                  std::map<int, std::vector<std::pair<int, float>>> & by_e);   // (token, weight) per expert (+ P100E_FN_TAIL_SKIP)
    void ffn_shared(rank_state & S, int r, int il, const float * cur, float * out, int T);   // rank 0: + shared expert
    void ffn_shared_pre(rank_state & S, int il, const float * cur, int T);   // rank 0, remote tier: its matvecs early
    void ffn_shared_post(rank_state & S, int il, float * out, int T);        // ... and its add after the combine
    // ---- hybrid pipeline ----
    int lb_ = 0, le_ = 0;                       // trunk layers of this node [lb_, le_) (one process: [0, L_))
    int pipe_k_ = 0;                            // the agreed split (pipe mode)
    std::unique_ptr<net_transport> pnet_;
    char * pipe_h_ = nullptr;                   // pinned staging: [hdr][max_t][HC*E] f32 rows / a command
    size_t pipe_cap_ = 0;
    uint64_t pipe_step_ = 0;
    int pend_rb_T_ = 0, pend_rb_keep_ = 0;      // driver: a rollback not yet sent to node 0
    bool pend_reset_ = false;                   // driver: a reset not yet sent to node 0
    bool quit_sent_ = false;
    std::vector<int> slots_;                    // resolved resident slots per local rank (adaptive files)
    std::vector<double> plan_tot_, plan_free_;  // one process: the byte plan's total / free VRAM per rank (load() reports the real use)
    bool has_head() const { return le_ == L_; }                 // output head + draft block on this node
    bool has_embed() const { return lb_ == 0; }                 // token embedding on this node
    bool owns(int il) const { return (il >= lb_ && il < le_) || (il == mtp_il_ && has_head()); }
    int mtp_file_il_ = -1;                      // the draft block's blk index in the file (debug layer limit + MTP)
    int file_layer(int il) const { return il == mtp_il_ && mtp_file_il_ >= 0 ? mtp_file_il_ : il; }
    void pipe_init(const std::vector<int> & devices);           // connect, agree, plan, layer range (before load)
    void pipe_cmd(int op, const std::vector<int32_t> & tokens); // driver: one command (+ the pending rollback / reset)
    void pipe_recv_rows(int T);                                 // driver: node 0's residual rows -> S.res of every rank
    void pipe_send_rows(int T);                                 // worker: rank 0's S.res rows -> node 1
    void local_rollback(int T, int n_keep);
    void local_reset();
    static double host_free_bytes();                            // available RAM (P100E_FN_HOST_FREE_MB overrides)
    static std::vector<double> gpu_free_per_rank(const std::vector<int> & devices);   // free VRAM (P100E_FN_GPU_FREE_MB caps)
};

} // namespace p100e
