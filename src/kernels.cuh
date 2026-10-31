// Reference (correctness-first) kernels for the Qwen3.8 forward pass. All activations f32,
// row-major [rows][cols]. These are the baseline the fast P100 kernels get checked against.
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

#include "fast/mrope.cuh"

namespace p100e::k {

// y[r][:] = x[r][:] * rsqrt(mean(x^2) + eps) * (w ? w[:] : 1); in-place allowed
void rmsnorm(const float * x, const float * w, float * y, int64_t nrows, int ncols, float eps, cudaStream_t s);
// y[r][:] = x[r][:] / max(||x[r]||, eps)
void l2norm(const float * x, float * y, int64_t nrows, int ncols, float eps, cudaStream_t s);
// y = rmsnorm(x, w) * silu(z) per row (gated norm of the delta-net output)
void rmsnorm_gated(const float * x, const float * w, const float * z, float * y, int64_t nrows, int ncols, float eps, cudaStream_t s);

void add(float * x, const float * y, int64_t n, cudaStream_t s);                // x += y
void silu_mul(float * gate, const float * up, int64_t n, cudaStream_t s);       // gate = silu(gate) * up
void sigmoid_mul(float * x, const float * g, int64_t n, cudaStream_t s);        // x *= sigmoid(g)

// qg: [T][H][2*D] (q then gate per head) -> q: [T][H][D], gate: [T][H*D]
void split_q_gate(const float * qg, float * q, float * gate, int T, int H, int D, cudaStream_t s);
// NEOX rotary on the first n_rot dims of each head: x [T][H][D], pos [T]
void rope_neox(float * x, const int32_t * pos, int T, int H, int D, int n_rot, float freq_base, cudaStream_t s);
// interleaved M-RoPE (fast/mrope.cuh): rope_neox with each rotary pair's position picked from m.p's t / h / w streams
// (row t of the batch = stream row t). A separate kernel: rope_neox (text without images) stays as it was.
void rope_mrope(float * x, const fast::mrope_pos & m, int T, int H, int D, int n_rot, float freq_base, cudaStream_t s);

// causal attention with GQA: q [T][Hq][D], caches [n_ctx][Hkv][D] (positions 0..p0+T-1 valid),
// out [T][Hq][D]. Query t sits at position p0 + t.
void attention(const float * q, const float * kc, const float * vc, float * out,
               int T, int p0, int Hq, int Hkv, int D, float scale, cudaStream_t s);
// same contract, online softmax with the keys spread over 16 warps (no per-key block barrier, no score buffer, any
// context). Not bit-identical to attention(). Returns false (nothing launched) unless D % 4 == 0, D <= 512, Hq % Hkv
// == 0 and all pointers are 16-byte aligned. attention() itself routes here when its score buffer would not fit.
bool attention_fast(const float * q, const float * kc, const float * vc, float * out,
                    int T, int p0, int Hq, int Hkv, int D, float scale, cudaStream_t s);

// causal depthwise conv (kernel 4) + silu over channels. x [T][C], state [C][3] (previous 3 inputs,
// oldest first), w [C][4]. Writes y [T][C] and updates state to the last 3 inputs.
void conv1d_silu(const float * x, float * state, const float * w, float * y, int T, int C, cudaStream_t s);

// delta-net gates: beta = sigmoid(b), g = softplus(a + dt_bias) * ssm_a ; a, b: [T][Hv]
void gdn_gates(float * a, float * b, const float * dt_bias, const float * ssm_a, int T, int Hv, cudaStream_t s);

// gated delta rule recurrence. conv [T][C] holds q (Hk*Dk), k (Hk*Dk), v (Hv*Dv) per token; q and k are
// L2-normalized inside. v head h uses q/k head h % Hk. state [Hv][Dv][Dk] (row j = column j of S).
// out [T][Hv][Dv]. Needs Dk == Dv <= 256.
void gdn_recurrent(const float * conv, const float * g, const float * beta, float * state, float * out,
                   int T, int Hk, int Hv, int Dk, int Dv, float eps, cudaStream_t s);

// copy the K/V rows of this batch into the caches at positions p0..p0+T-1
void kv_store(const float * k, const float * v, float * kc, float * vc, int T, int p0, int row, cudaStream_t s);

} // namespace p100e::k
