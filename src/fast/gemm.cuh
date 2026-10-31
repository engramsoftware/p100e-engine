// p100e::fast prefill GEMM for raw GGUF-quantized weights.
//
// y[T][n_out] = x[T][n_in] * W^T, W is n_out rows of n_in raw ggml-quantized values (row-major, no repack).
//
// Default path (gemm_q): weights are dequantized to f16 in scratch (ggml to_fp16, raw layout), activations are
// prescaled per token by a power of two (exact, max |x| lands in [8,16)) and rounded to f16, then a Pascal "fold"
// GEMM runs: half2 HFMA2 products with short f16 chains (256 values of K) folded into f32 accumulators
// (scheme of Kmic-68's gemm-fold, MIT, ported from our fork). Runs at the f16 rate on sm_60 with f32-level
// accumulation error; the remaining error is the f16 rounding of the inputs.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace p100e {
namespace fast {

enum class gemm_mode : int {
    fold      = 0,  // default: dequant f16 + fold GEMM (f16 products, f32 accumulation every 256 K)
    cublas32  = 1,  // dequant f16 + cublasGemmEx COMPUTE_32F (= current tp_model::matmul path)
    cublas16  = 2,  // dequant f16 + cublasGemmEx COMPUTE_16F (whole-K f16 chains; f16 output widened)
    cublas16a6 = 3, // as cublas16 with CUBLAS_GEMM_ALGO6 (blocked chains on P100; DEFAULT where not offered)
    fold32    = 4,  // fold GEMM with 32-value f16 chains (more accurate, more fold work)
};

// Scratch needed for gemm_q. Any size >= gemm_scratch_min_bytes works (weights are then processed in row chunks);
// gemm_scratch_bytes is the size that dequantizes the whole matrix in one chunk.
size_t gemm_scratch_bytes(int64_t n_in, int64_t n_out, int T_max);
size_t gemm_scratch_min_bytes(int64_t n_in, int T_max);

// y[T][n_out] = x[T][n_in] * W^T. T = 1..2048 (meant for T >= 16). n_in % 64 == 0.
// Throws std::runtime_error on unsupported type / shape / too little scratch.
void gemm_q(uint32_t ggml_type, const void * w_raw, int64_t n_in, int64_t n_out, const float * x, float * y, int T,
            void * scratch, size_t scratch_bytes, cudaStream_t s, cublasHandle_t cb);

// same, with an explicit algorithm (for tests / A-B timing)
void gemm_q_mode(gemm_mode mode, uint32_t ggml_type, const void * w_raw, int64_t n_in, int64_t n_out,
                 const float * x, float * y, int T, void * scratch, size_t scratch_bytes, cudaStream_t s,
                 cublasHandle_t cb);

} // namespace fast
} // namespace p100e

#include "fast/matvec.cuh"

namespace p100e {
namespace fast {

// Fold GEMM on a PACKED weight (fast::pack_weight, any layout): the f16 dequant step is fast::packed_to_f16
// (bit-identical to ggml's to-f16 of the raw rows), so the result equals gemm_q on the raw bytes exactly.
// Scratch: same sizes as gemm_q (gemm_scratch_bytes / gemm_scratch_min_bytes; smaller scratch = row chunks).
inline size_t gemm_packed_scratch_bytes(int64_t n_in, int64_t n_out, int T_max) {
    return gemm_scratch_bytes(n_in, n_out, T_max);
}
// the same weights through cuBLAS (f16 inputs, f32 accumulate); not bit-identical to gemm_packed (see gemm.cu)
void gemm_packed_cublas32(const packed_weight & w, const float * x, float * y, int T, void * scratch, size_t scratch_bytes,
                          cudaStream_t s, cublasHandle_t cb);
void gemm_packed(const packed_weight & w, const float * x, float * y, int T, void * scratch, size_t scratch_bytes,
                 cudaStream_t s, cublasHandle_t cb);

// ---- prefill pipeline building blocks (tp_model::forward_prefill) ----
// Activations for the fold GEMM, one row per token: x16[t*ldx + k] = f16(v[t][k] * 2^-e_t), cs[t] = 2^e_t, with
// max|v[t]| * 2^-e_t in [8, 16). Same arithmetic as gemm_packed's prescale, so fold_gemm on these is bit-identical
// to gemm_packed on v. K % 4 == 0, rows 16-byte aligned.
// one_read (fork GGML_CUDA_FOLD_PRESCALE_V): the row is held in registers between the max and the
// scaling (one read of v instead of two; K <= 9216, else the two-read kernel). Same bits either way.
void prescale_rows(const float * v, int64_t ldv, int T, int K, half * x16, int64_t ldx, float * cs, cudaStream_t s,
                   bool one_read = true);
// x[t] = (x[t] + p0[t]) + p1[t] (p0 / p1 may be null; pass rank 0's partial as p0), then v = rmsnorm(x[t]) * w with
// fast::rmsnorm's arithmetic, prescaled into x16 / cs as above. x16 == nullptr: only the add. K % 4 == 0, K <= 8192.
void add_norm_prescale(float * x, const float * p0, const float * p1, const float * w, float eps, int T, int K,
                       half * x16, int64_t ldx, float * cs, cudaStream_t s);
// same with f16 partials (the f16 exchange wire, fold_opts::out16): x[t] = (x[t] + (float) p0[t]) + (float) p1[t]
void add_norm_prescale16(float * x, const half * p0, const half * p1, const float * w, float eps, int T, int K,
                         half * x16, int64_t ldx, float * cs, cudaStream_t s);
// v = silu(g) * u (k::silu_mul arithmetic), prescaled as above. g, u: [T][K]. K % 4 == 0, K <= 9216.
void glu_prescale(const float * g, const float * u, int T, int K, half * x16, int64_t ldx, float * cs, cudaStream_t s);

// One fold launch over up to 4 f16 weights that share the activations: y_s[n*ldy_s + m] = cs[n] * sum_k w_s[m][k] *
// x16[n*ldx + k], m < rows_s, n < T. Same chains and folds as gemm_packed's fold kernel (bit-identical outputs); the
// 128-token tile loop is unrolled by two. K % 64 == 0, rows_s % 4 == 0, ldy_s % 4 == 0, ldx % 8 == 0.
struct fold_seg {
    const half * w = nullptr;
    float * y = nullptr;
    int64_t ldy = 0;
    int rows = 0;
    half * y16 = nullptr;   // fold_opts::out16: the f16 output [n*ldy + m] (y unused)
};
// Launch options (none changes a value except out16):
//  raster (fork GEMM_FOLD_RASTER): blocks walk groups of `raster` row tiles x every token tile, so the
//    blocks resident at once share weight tiles in L2; 0/1 = row tiles fastest (the original order). Same bits.
//  out16 (fork GGML_CUDA_GEMM_FOLD=2): each output is rounded to f16 and stored as half into seg.y16;
//    finite values that would round to +-inf (|v| >= 65520) are saturated to +-65504 and counted in *sat (device int,
//    may be null; one count per group of 4 consecutive rows of a token with any saturated value).
//  sass (fork GGML_CUDA_GEMM_FOLD_SASS): T > 64 launches with f32 outputs run a register-bank-fixed
//    sm_60 cubin of the same kernel (src/fast/gemm_sass_fold8.h, tools/sass/fold_bankfix.py) on cc 6.0 devices
//    (elsewhere, or if the cubin does not load, the compiled kernel). Same bits.
struct fold_opts {
    int raster = 0;
    bool out16 = false;
    int * sat = nullptr;
    bool q4k = false;         // P100E_PF_FUSEDQ: every seg.w is q4kr packed rows (cast to const half *), dequantized in the GEMM's loads (same bits)
    int sass = 0;             // 0 off, 1 = bank-fix variant 0 (accumulators + free registers), 2 = variant 1 (wide), 3 = unpatched cubin (debug)
};
void fold_gemm(const fold_seg * segs, int nseg, const half * x16, int64_t ldx, const float * cs, int T, int K,
               cudaStream_t s, const fold_opts & o = fold_opts());

// *out = max(*out, max |v[i]|), i < n; v is half (f16) or float. *out holds a non-negative float (start it at 0); NaN
// counts above inf. For measurements (the prefill wire statistics, P100E_PF_WIRE_STATS), not on a default path.
void abs_max(const void * v, bool f16, size_t n, float * out, cudaStream_t s);

} // namespace fast
} // namespace p100e
