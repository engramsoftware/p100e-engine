// MTP (multi-token prediction / nextn) speculative decoding for tp_model, greedy.
// The model's nextn block (blk.<L>.nextn.*) is run as a draft head, matching llama.cpp's qwen35 graph_mtp:
//   h_norm = rmsnorm(h, hnorm), e_norm = rmsnorm(embed(tok), enorm), x = eh_proj(concat(e_norm, h_norm)),
//   one full-attention decoder block (own KV cache), then rmsnorm(shared_head_norm) and the shared output head.
// The entry at position p is (token at p, main model's output_norm(x) at p-1); drafts chain on the block's own
// normed hidden. Verification runs [last token + k drafts] through the main model in one batch, keeps the longest
// matching prefix and rolls back the recurrent state of the rejected tokens.
//
// Draft head execution (T <= 4 rows): the fused decode machinery (tp_model::fused_layer on the nextn plan, fused
// norms + P2P allreduce + prequantized matvecs, concatenated q|k|v and gate|up) replayed as one CUDA graph per
// (T, h source); T > 4 and P100E_DRAFT_FUSE=0 use the per-op path. The draft's LM head can score
// a frequency-ranked subset of the vocabulary (set_draft_vocab): the verify always runs the full head, so the output
// stays identical to plain greedy, only the acceptance can change.
// Prompt catch-up (catchup_kv, fork LLAMA_MTP_KV_ONLY): only the last prompt entry's output is ever read (the first
// draft), so every earlier entry just writes its K / V rows (eh_proj + allreduce + attn_norm + wk / wv + k_norm + rope +
// append: ~15 % of the block's GEMM work and no attention); the prompt runs in ubatch chunks, each followed by the
// KV-only rows of its entries, then one full T = 1 run (the fused / graphed draft step) for the first generated token.
// P100E_MTP_KV_ONLY=0: the old catch-up (the whole block over all P + 1 entries in one per-op run, P + 1 <= max_batch).
#pragma once

#include "tp_model.h"

#include <map>

namespace p100e {

class tp_mtp {
public:
    explicit tp_mtp(tp_model & m);
    ~tp_mtp();

    // h input of the next run(): where rows 0..T-1 of the nextn block's hidden input come from
    enum hsrc : int {
        H_READY = 0,   // prepared by the caller (h_zero / h_from_main)
        H_MAIN = 1,    // rows 0..T-1 = the main model's hn rows 0..T-1 of the last forward (catch-up after a verify)
        H_SELF = 2,    // row 0 = the last run()'s final normed hidden (chained draft, T = 1)
    };
    // h input rows for the next run(): copy main-model hn rows [src0, src0+n) into rows [dst0, dst0+n)
    void h_from_main(int dst0, int src0, int n);
    void h_zero(int row);
    // several sequences: the h input row 0 a sequence's next run continues from (the prompt chunk
    // carry, a chain's last hidden) is saved / restored per sequence around other sequences' runs. carry_load of a
    // sequence without a saved carry zeroes row 0 (a prompt's first chunk).
    void carry_save(int seq);
    void carry_save_row(int seq, int row);   // the carry = h input row `row` (after h_from_main: a step's last row)
    void carry_load(int seq);
    void carry_drop(int seq);
    // prompt cache checkpoints / session files: a sequence's saved carry <-> host (n_embd floats,
    // synchronous). carry_to_host returns false (and writes zeros) when the sequence has no carry.
    bool carry_to_host(int seq, float * dst);
    void carry_from_host(int seq, const float * src);
    bool has_carry(int seq) const { return carry_.count(seq) != 0; }
    // several sequences in one draft-head run (batched multi-session MTP): the rows of sequence i
    // (tokens at positions p0[i]..) stacked in order, 2..8 sequences, <= 8 rows; the h input rows are staged by the caller
    // (h_from_main for catch-up rows, h_from_multi for chained rows). Returns the pick of every sequence's last row
    // (multi_p(i) = its probability); each row is bitwise the single-sequence run's row (so the drafts and the acceptance
    // are those of the sequence alone). One fused step, CUDA graph per (rows, sequences).
    std::vector<int32_t> run_multi(const std::vector<tp_model::seq_rows> & b, const std::vector<int> & p0);
    float multi_p(int i) const { return mp_.at(i); }
    void h_from_multi(int dst, int i);   // h input row dst <- the last run_multi's normed hidden of sequence i
    // run the nextn block on tokens at positions p0.. with the h rows from `src`; returns the last row's greedy
    // token over the draft vocabulary (argmax on the GPUs; last_p() / last_margin() describe its confidence)
    int32_t run(const std::vector<int32_t> & tokens, int p0, hsrc src = H_READY);
    // KV-only prompt catch-up of entries p0..p0+T-1 (T = tokens.size() <= max_batch), right after the main model's
    // forward over exactly these tokens: entry p0+i = (tokens[i], h_{p0+i-1}) with h_{p0-1} = h input row 0 (the carry:
    // h_zero(0) before the first chunk) and h_{p0}.. = the main model's hn rows 0..T-2. Writes the nextn KV rows only;
    // afterwards h input row 0 = hn row T-1 (the carry for the next chunk, or the h of the first generated token's run).
    // Ends with a sync of every rank.
    void catchup_kv(const std::vector<int32_t> & tokens, int p0);

    // draft LM head over the first n ids of `ranked` (most frequent first; n rounded down to a multiple of 128 so
    // every GPU gets an aligned slice). n <= 0 or n >= vocab: the full output head.
    void set_draft_vocab(const std::vector<int32_t> & ranked, int n);
    int draft_vocab() const { return n_sub_ > 0 ? n_sub_ : m_.vocab(); }
    // built-in ranking for the model's vocabulary (tools/draft_vocab.py: control/user-defined + byte tokens, then
    // corpus frequency); falls back to token-id order (BPE merge rank) when the GGUF vocab does not match
    std::vector<int32_t> default_ranking(bool * matched = nullptr) const;

    float last_p() const { return last_p_; }            // softmax probability of the pick (over the draft vocabulary)
    float last_margin() const { return last_margin_; }  // top-1 minus top-2 logit
    bool fused() const { return fuse_; }
    bool graphs() const { return graph_; }
    // debug counters (p100e-run --ab --mtp): draft graphs captured since construction, and how many times the
    // captured draft graphs were dropped because tp_model::reload_env changed a kernel choice (kern_epoch)
    int graph_captures() const { return n_capt_; }
    int graph_epoch_drops() const { return n_epoch_drops_; }

private:
    struct dgraph { int T = 0; bool from_main = false; int slot = -1; tp_model::step_graph G; };   // slot: the sequence's
                                                                                                // page table is baked in
    void body(int T, bool from_main);              // fused draft step on the staged inputs (S.x rank 0, S.pos)
    void body_multi(int T, int nseq);              // run_multi's step (descriptor S.desc of every sequence)
    struct mgraph { int T = 0, nseq = 0; tp_model::step_graph G; };
    std::vector<mgraph> mgraphs_;
    std::vector<float *> hmul_;                    // per rank [MAX_ROWS][E]: every sequence's normed last row (run_multi)
    std::vector<float> mp_;
    void run_perop(int T, int p0);                 // per-op path (any T)
    void eh_in(int T);                             // per-op: x = eh_proj(concat(enorm(embed), hnorm(h))) (staged S.x, hin)
    // eh_proj is row-parallel over its 2E input columns: rank r reads tp_split eh = [r, r + 1) * 2E/R when 2E divides
    // by R (R = 2 / 4), 256-column blocks otherwise (R = 3: 14/13/13 on the 27B, rank 1 straddles the halves). An "h
    // rank" reads columns of the hidden half (hin_), the others only the embedding half. R = 2: rank 1, the whole half.
    bool h_rank(int r) const { return m_.split_.rk[r].eh.e() > m_.E_; }
    // every split but the R = 2 halves (halves_): rank r's eh_proj input = its columns of the normed halves as a dense
    // [T][W] block in S.c (the rmsnorm needs the whole half; a straddling rank norms both); fused: + S.aq = a16 of it.
    // Returns S.c
    const float * eh_slice(int r, int T, bool fused);
    bool halves_ = false;   // R = 2: rank 0 the embedding half, rank 1 the hidden half (no slice, the old kernels)
    void head(int r, const float * h, bool prequant);   // LM head (full or subset) on h (one row) + pick/confidence; prequant: S.aq = a16(h)
    int32_t collect();                             // combine the ranks' (max, idx, sum, 2nd) -> token, last_p_
    int32_t collect_at(const void * c, float & p, float & margin) const;   // c = the ranks' entries
    void drop_graphs();
    // the draft graphs bake fused_layer's kernel choices (P100E_MVWRP_*, P100E_MERGE_INPROJ, P100E_GDN_FUSE): when
    // tp_model::reload_env changed one (m_.kern_epoch() moved), drop them so the next eager run re-captures
    void check_epoch();
    uint64_t kern_epoch_ = 0;                        // m_.kern_epoch() the captured draft graphs were made under
    int n_capt_ = 0, n_epoch_drops_ = 0;             // graph_captures(), graph_epoch_drops()

    tp_model & m_;
    std::vector<float *> hin_, hout_;   // per rank: [max_batch][E], [E] (hin is read by the h ranks only)
    void * conf_d_[4] = {};                          // per rank (<= fast::P2P_MAX_RANKS): device (max, idx, sumexp, 2nd) of the head row
    void * conf_h_ = nullptr;                        // pinned host copies: [R] single run, [R + R i + r] run_multi row i
    bool fuse_ = true, graph_ = true;
    std::vector<dgraph> graphs_;
    bool graph_failed_ = false;
    int n_sub_ = 0;                                  // draft vocabulary size (0 = full head)
    std::vector<std::vector<int32_t>> sub_ids_;      // per rank: local row -> token id (subset head)
    float last_p_ = 1.0f, last_margin_ = 0.0f;
    std::map<int, std::vector<float *>> carry_;      // per sequence: saved h input row 0 per rank ([E] on the h ranks)
};

struct mtp_stats {
    int steps = 0;      // verify forwards
    int drafted = 0;    // draft tokens proposed (verified)
    int accepted = 0;   // draft tokens accepted
    double t_gen = 0;   // seconds after the prompt forward (MTP prompt catch-up + all steps)
    double t_prompt = 0;    // seconds in the main model's prompt forward(s)
    double t_catchup = 0;   // seconds in the draft head's prompt catch-up, incl. the first draft (KV-only chunks + the
                            // full run of the first generated token; P100E_MTP_KV_ONLY=0: the one full P + 1 row run)
    double t_verify = 0, t_draft = 0;   // seconds in the verify forwards / in the draft-head runs
    int n_draft = 0;    // draft-head runs
    int hist[5] = {0, 0, 0, 0, 0};      // verify steps by number of drafts (0..4)
    // calibration: every verified draft whose predecessors were all accepted (position j = 1.., draft-head top-1
    // probability p, running product cum, logit margin, accepted)
    struct cal_rec { int j; float p, cum, margin; bool acc; };
    std::vector<cal_rec> cal;
    int chk_prev_T = 0, chk_prev_a = 0;   // debug (ref_hash): the previous step's T and accepted drafts
    int chk_bad = 0;                      // debug (ref_hash): verify rows of this run whose logits differ from plain greedy
};

// Default mode (P100x2, ctx 4096):
// adaptive, up to 3 drafts, tau 0.5. Gate prompts (5 x 61 tokens) mean 93.1 t/s vs fixed k = 2 / 3 90.7 / 89.8 and
// k <= 3 tau 0.7 / 0.9 91.6 / 88.8; `--bench 0 128` 111.1 vs fixed k = 2 / 3 96.6 / 108.5 (tau 0.7: 111.2). With the
// cheaper multi-column verify (T = 2 / 3 / 4: 23.6 / 27.6 / 32.8 ms vs 21 ms plain) a low threshold that keeps most
// third drafts wins.
struct mtp_opts {
    int k = 3;          // drafts per step (adaptive: the maximum)
    float tau = 0.5f;   // adaptive draft length: draft j is verified only while prod_{i<=j} p_i >= tau (0 = fixed k)
    int k_min = 1;      // adaptive: always verify at least this many drafts
    float tau1 = 0.0f;  // adaptive, k_min = 0: the first draft is verified only when its p >= tau1 (0: tau)
    // debug (identity checks): FNV-1a hash of plain greedy's logits for generated token i; when set, the verify returns
    // full logits and every used row is compared bitwise (first mismatches printed)
    const std::vector<uint64_t> * ref_hash = nullptr;
};

int argmax(const float * l, int n);

// greedy generation of n_gen tokens after prompt (draft length per mtp_opts; mtp_opts{} = the default mode above).
// Returns the generated ids.
std::vector<int32_t> generate_mtp(tp_model & m, tp_mtp & d, const std::vector<int32_t> & prompt, int n_gen,
                                  const mtp_opts & o, mtp_stats & st);

// a multimodal prompt piece: text tokens, or an image's embedding rows (embd [nx * ny][n_embd] f32 host, see
// tp_model::forward_image); the draft head's catch-up takes the image rows as its "embedding" input (llama.cpp's
// qwen35 MTP graph reads the batch's input embeddings) at the same M-RoPE positions
struct mtp_piece {
    std::vector<int32_t> tokens;
    const float * embd = nullptr;
    int nx = 0, ny = 0;
    int n_embd = 0;   // width of the embedding rows (tp_model::mm_in::n_embd: must equal the model's)
};
// same, after a prompt of pieces appended to the current context (m.n_past() rows, e.g. 0 after reset()); the last
// piece must be text. Needs the KV-only catch-up (P100E_MTP_KV_ONLY unset or 1). The draft head's carry (h of the row
// before the prompt) is zeroed only on a fresh context (n_past() == 0); on a continued one it is the h of the context's
// last row, which every generate_mtp(pieces) leaves in place when it returns, so a follow-up turn on the same context
// (generate_mtp(pieces) again, nothing else run on m in between) drafts like the first. Only the acceptance depends on
// it: the ids are plain greedy's either way. P100E_MTP_CARRY_ZERO=1 (debug): always zero it (the old behaviour).
std::vector<int32_t> generate_mtp(tp_model & m, tp_mtp & d, const std::vector<mtp_piece> & prompt, int n_gen,
                                  const mtp_opts & o, mtp_stats & st);
// fixed k drafts per step
inline std::vector<int32_t> generate_mtp(tp_model & m, tp_mtp & d, const std::vector<int32_t> & prompt, int n_gen,
                                         int k, mtp_stats & st) {
    mtp_opts o;
    o.k = k;
    o.tau = 0.0f;
    return generate_mtp(m, d, prompt, n_gen, o, st);
}

} // namespace p100e
