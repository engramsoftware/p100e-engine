// Fast decode-path small kernels (standalone, f32 activations). Namespace p100e::fast.
// All row pointers should be 16-byte aligned and n % 4 == 0 for the vectorized path; anything else
// silently takes the (correct, slower) scalar path.
#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "fast/attn.cuh"
#include "fast/mrope.cuh"

namespace p100e::fast {

// y[r] = x[r] * rsqrt(mean(x[r]^2) + eps) * w   (w may be null -> 1). y may alias x.
void rmsnorm(const float * x, const float * w, float * y, int64_t rows, int n, float eps, cudaStream_t s);
// x[r] += r[r]; y[r] = rmsnorm(x[r]) * w.   (x updated in place, y may alias x)
void add_rmsnorm(float * x, const float * r, const float * w, float * y, int T, int n, float eps, cudaStream_t s);
// same as above, additionally writes an f16 copy of y into yh (same [rows][n] layout) for half2 GEMV inputs
void rmsnorm_h(const float * x, const float * w, float * y, __half * yh, int64_t rows, int n, float eps, cudaStream_t s);
void add_rmsnorm_h(float * x, const float * r, const float * w, float * y, __half * yh, int T, int n, float eps, cudaStream_t s);
// y = rmsnorm(x, w) * silu(z) per row (same semantics as k::rmsnorm_gated). y may alias x.
void rmsnorm_gated(const float * x, const float * w, const float * z, float * y, int64_t rows, int n, float eps, cudaStream_t s);

void silu_mul(float * gate, const float * up, int64_t n, cudaStream_t s);   // gate = silu(gate) * up
void sigmoid_mul(float * x, const float * g, int64_t n, cudaStream_t s);    // x *= sigmoid(g)

// ---- decode fusions that also write the a16-quantized activation (qbuf, layout of fast::matvec_prequantized) ----
// x[t] = (x[t] + r1[t]) + r2[t] (each term may be null; the sum order is fixed, so replicas on several GPUs that
// pass the partials in the same order stay bit-identical), y[t] = rmsnorm(x[t]) * w, qbuf = a16(y) when qbuf != null.
// Rows t = 0..T-1 of n. Same arithmetic as k::add (twice) + fast::rmsnorm + fast::matvec_quantize. y must not alias x.
// Returns true when qbuf was written (n % 32 == 0, 16-byte aligned rows); otherwise x and y are still done and the
// caller quantizes y itself.
bool add2_rmsnorm_q(float * x, const float * r1, const float * r2, const float * w, float * y, void * qbuf,
                    int T, int n, float eps, cudaStream_t s);

// One-shot two-GPU allreduce for the decode (no copy, no events): each GPU writes its partial into its own memory,
// then p2p_signal() stores seq into the peer's flag (peer memory, P2P store, after the partial is complete in stream
// order). add2_rmsnorm_q_wait() spins until its own *flag >= seq and reads the peer's partial straight from the peer
// GPU (P2P loads), then does exactly add2_rmsnorm_q (same sum order: pass rank 0's partial as r1 on both GPUs).
// Needs peer access both ways, r1 and r2 both non-null, n % 32 == 0. Returns false (nothing launched) otherwise.
void p2p_signal(int * peer_flag, int seq, cudaStream_t s);
// Flag block of one GPU (>= 64 bytes, zeroed, in that GPU's memory): [0] the peer's sequence number (P2P store by the
// peer), [1] waits that timed out (results then wrong), [2] this GPU's own signal counter (p2p_signal_ctr),
// [P2P_FLAG_TEST_DROP] fault injection: this GPU stores no signal from that sequence number on (0 = off),
// [P2P_FLAG_TIMEOUT_MS] spin timeout in ms (0 = 2000), [P2P_FLAG_FENCE] != 0: __threadfence_system() before each
// signal store (off by default, see p2p_signal_k), [P2P_FLAG_STATUS_PTR .. +1] an int * to a host-mapped status
// line (or null): on a timeout the wait writes {1, seq waited for, peer seq seen, elapsed ms} there.
// Sequence numbers compare wrap-safe (unsigned difference).
enum { P2P_FLAG_TEST_DROP = 3, P2P_FLAG_TIMEOUT_MS = 4, P2P_FLAG_FENCE = 5, P2P_FLAG_STATUS_PTR = 6,
       P2P_FLAG_CTR0 = 2, P2P_FLAG_CTR1 = 8 };
int p2p_timeouts(const int * flag);
bool add2_rmsnorm_q_wait(float * x, const float * r1, const float * r2, const int * flag, int seq, const float * w, float * y,
                         void * qbuf, int T, int n, float eps, cudaStream_t s);
// Same protocol with the sequence number decided on the device (CUDA-graph replays: no host value baked into the
// launches): p2p_signal_ctr() does seq = ++*ctr (ctr: this GPU's own counter, in its memory) and stores seq into the
// peer's flag; add2_rmsnorm_q_wait_ctr() waits until *flag >= *ctr (the value of this GPU's latest signal). Both GPUs
// signal once per allreduce, so their counters stay equal. Do not mix with the host-seq calls on the same flag.
void p2p_signal_ctr(int * peer_flag, int * ctr, cudaStream_t s);
bool add2_rmsnorm_q_wait_ctr(float * x, const float * r1, const float * r2, const int * flag, const int * ctr, const float * w,
                             float * y, void * qbuf, int T, int n, float eps, cudaStream_t s);
// P100E_AR_SIG_FUSE: the signal folded into the wait (one launch per allreduce instead of p2p_signal_ctr + the wait).
// add2_rmsnorm_q_sig_wait() first does this GPU's signal of the allreduce (block 0: seq = flag[ctr word 1 - par] + 1
// -> flag[ctr word par] and *peer_flag), then exactly add2_rmsnorm_q_wait_ctr (waits until *flag >= seq). par = the
// allreduce's parity (0 / 1, alternating per allreduce): the two counter words P2P_FLAG_CTR0 / CTR1 alternate so no
// block of the launch reads the word block 0 writes. Both words must hold the latest sequence number when the mode
// is entered (tp_model::reload_env). p2p_sig_flush(): the signal alone with the same words. Same bits as the
// two-launch path (the arithmetic of the norm is untouched).
bool add2_rmsnorm_q_sig_wait(float * x, const float * r1, const float * r2, int * flag, int * peer_flag, int par,
                             const float * w, float * y, void * qbuf, int T, int n, float eps, cudaStream_t s);
void p2p_sig_flush(int * peer_flag, int * flag, int par, cudaStream_t s);
void p2p_sig_equalize(int * flag, cudaStream_t s);   // both counter words := the later of the two (wrap-safe)

// ---- one-shot allreduce over 3..4 GPUs (tensor parallel with R > 2; R <= 2 keeps the calls above) ----
// One flag word per sender: word P2P_FLAG_PEER0 + q of a GPU's flag block holds the sequence number rank q stored there
// (several peers signal one GPU). p2p_signal_n(): seq = ++*ctr (ctr = this GPU's own counter, word P2P_FLAG_CTR0 of its
// block, as p2p_signal_ctr; fence / fault-injection words of the same block) stored into word P2P_FLAG_PEER0 + self of
// every other rank's block (peer.f[q], null for q == self). addn_rmsnorm_q_wait_ctr(): thread 0 of every block waits
// until each other rank's word in this GPU's block reached *ctr (bounded spin, timeouts as add2_rmsnorm_q_wait_ctr),
// then x[t] = (((x[t] + p[0][t]) + p[1][t]) + p[2][t]) (+ p[3][t]): rank order, the same on every GPU, so the replicas
// stay bit-identical; the same f32 adds as np k::add calls in rank order. y = rmsnorm(x) * w, qbuf = a16(y), with
// add2_rmsnorm_q's block shape (same reduction tree), the sum of squares in a pinned order (fused.cu normn_wait_k).
// parts.p[self] is this GPU's partial, the others are read from the peers' memory (P2P loads). np = 3 or 4, every part
// non-null and 16-byte aligned, n % 32 == 0; false (nothing launched) otherwise. addn_rmsnorm_q(): the same kernel
// without the wait (flag / ctr null: the event allreduce, partials already copied), bitwise the same results.
enum { P2P_FLAG_PEER0 = 16, P2P_MAX_RANKS = 4 };
struct p2p_ptrs { const float * p[P2P_MAX_RANKS]; };
struct p2p_flags { int * f[P2P_MAX_RANKS]; };
void p2p_signal_n(const p2p_flags & peer, int np, int self, int * ctr, cudaStream_t s);
bool addn_rmsnorm_q_wait_ctr(float * x, const p2p_ptrs & parts, int np, int self, const int * flag, const int * ctr,
                             const float * w, float * y, void * qbuf, int T, int n, float eps, cudaStream_t s);
bool addn_rmsnorm_q(float * x, const p2p_ptrs & parts, int np, const float * w, float * y, void * qbuf, int T, int n,
                    float eps, cudaStream_t s);

enum ew_op : int { EW_SILU_MUL = 0, EW_SIGMOID_MUL = 1 };
// y = silu(a) * b (EW_SILU_MUL, = k::silu_mul) or a * sigmoid(b) (EW_SIGMOID_MUL, = k::sigmoid_mul) over [T][n],
// plus qbuf = a16(y). y may alias a. Returns false (nothing launched) unless n % 32 == 0 and a, b, y are 16-byte aligned.
// lda / ldb (0 = n; multiples of 4): row strides of a and b (read in place from a concatenated gate|up output); y is
// dense [T][n] and may alias a only when lda == n.
bool ew_quant(ew_op op, const float * a, const float * b, float * y, void * qbuf, int T, int n, cudaStream_t s,
              int lda = 0, int ldb = 0);

// Attention-layer input prep in one launch (decode, f16 / q8_0 / q4_0 KV cache): per token t
//   qg [T][Hq][2*D] (q | gate per head, the attn_q output) -> gate [T][Hq*D]; q = rope(rmsnorm(q) * q_norm) -> q [T][Hq][D]
//   k [T][Hkv][D] -> rope(rmsnorm(k) * k_norm) -> K cache row p0+t;  v -> V cache row p0+t
// Same arithmetic as k::split_q_gate + fast::rmsnorm + k::rope_neox (NEOX, first n_rot dims) + fast::kv_append.
// D % 128 == 0, f16 / q8_0 / q4_0 cache. Returns false (nothing launched) for an unsupported shape/cache type.
// p0 < 0: the cache row of token t is pos[t], read on the device (CUDA-graph replays; the caller keeps pos[t] < n_ctx).
// ldq / ldk / ldv (0 = dense Hq*2*D / Hkv*D / Hkv*D): row strides of qg, k, v (concatenated projection outputs).
// mr.p != nullptr: interleaved M-RoPE (fast/mrope.cuh) - the rotary positions come from mr (t / h / w streams, row t of
// the batch), the cache rows still from pos / p0 (a separate kernel instantiation; mr.p == nullptr runs the old one).
bool attn_prep(const float * qg, const float * k, const float * v, const float * q_norm, const float * k_norm,
               const int32_t * pos, float * q, float * gate, kv_cache & c, int p0, int T, int Hq, int Hkv, int D,
               int n_rot, float freq_base, float eps, cudaStream_t s, int ldq = 0, int ldk = 0, int ldv = 0,
               const mrope_pos & mr = mrope_pos{});

} // namespace p100e::fast
