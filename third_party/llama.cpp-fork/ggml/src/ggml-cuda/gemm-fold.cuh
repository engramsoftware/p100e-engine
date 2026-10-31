#pragma once

// Ported from Kmic-68/llama.cpp fc1c9056f, 4f261d20b, cceb8641e (MIT); pf-gemm additions are marked "pf-gemm".

#include "common.cuh"

// Prefill GEMM for Pascal: fp16 HFMA2 products with short fp16 chains folded into fp32 accumulators.
// Replaces cuBLAS COMPUTE_16F (whole-K fp16 accumulation) for quantized/f16 weights x f32
// activations. Returns false if it does not handle the shape. See gemm-fold.cu.
// pf-gemm: off unless GGML_CUDA_GEMM_FOLD is 1 or 2 (Kmic's default is 2). Repacked q4r weights are read through
// ggml_cuda_q4r_to_f16.
bool ggml_cuda_gemm_fold_try(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                             ggml_tensor * dst);

// True for a matmul the fold path would take but for its small row count: those run in fp32 cuBLAS
// instead (exact, and cheap at that size), not in whole-K fp16.
bool ggml_cuda_gemm_fold_wants_f32(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                                   const ggml_tensor * dst);

// Integrate (REVIEW-ALL B3), for the pf-xchg f16 wire: true when the fold makes a MUL_MAT that would otherwise be a
// cuBLAS COMPUTE_16F product (f16-exact f32 result) write values that are not f16-exact: GGML_CUDA_GEMM_FOLD=1
// (fp32 outputs) on a fold-eligible matmul, or a fold-eligible matmul below GGML_CUDA_GEMM_FOLD_MINROWS rows (runs in
// COMPUTE_32F cuBLAS, see ggml_cuda_gemm_fold_wants_f32). False with the fold off, and for mode 2 at or above
// MINROWS (outputs rounded to f16). Conservative: mode 1 says true even where the fold would decline the shape.
bool ggml_cuda_gemm_fold_out_not_f16(int cc, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst);

// pf-gemm: GGML_CUDA_GEMM_FOLD != 0. Every fold hook outside gemm-fold.cu checks this first, so with the switch
// unset none of them runs any new code.
bool ggml_cuda_gemm_fold_enabled();

// pf-gemm, test only: GGML_PF_TEST_ANY_ARCH=1 lets the Pascal-gated pf paths (fold GEMM, cuBLAS ALGO6, q4r
// repack, P100 prefill routing past MMQ) run on any NVIDIA device with fast fp16, for correctness tests on a
// non-Pascal card. Never set in production. ggml_cuda_pf_test_any_arch() lives in pf-test.cuh (shared).
#include "pf-test.cuh"

// Matmul pairing: the graph loop calls set_partner(next node) before running a MUL_MAT whose next node
// is a MUL_MAT on the same src1. If this node takes the fold path and the partner qualifies, both run
// as one launch (one activation prescale); take_partner_done() then returns true and the loop skips
// the partner. Every output is computed exactly as in two launches. GGML_CUDA_GEMM_FOLD_PAIR=0: off.
void ggml_cuda_gemm_fold_set_partner(ggml_tensor * next);
bool ggml_cuda_gemm_fold_take_partner_done();

// Weight prefetch: called by the chunked exchange before the compute stream waits for the peer's
// copies. Dequantizes, on the compute stream, the fold weights that followed this exchange last time
// (learned per device); the next fold matmul uses them if the weight pointer matches. Same dequant,
// only earlier: it fills the exchange wait. GGML_CUDA_FOLD_PREFETCH=0: off.
// pf-gemm: no caller yet (the chunked exchange, 1f92a1b48, is ported in the integrate phase).
void ggml_cuda_gemm_fold_prefetch(ggml_backend_cuda_context & ctx);

// SwiGLU fusion: for a split SwiGLU whose only consumer is the next node, a MUL_MAT that will take the
// fold path (the caller also checks the dispatcher routes it to cuBLAS f16), glu_ok() is true; the
// graph loop then skips the GLU and calls set_glu(glu), and the fold's prescale computes
// silu(gate)*up itself (same expression as the GLU kernel: exact). glu_pending() after the MUL_MAT
// must be false. GGML_CUDA_FOLD_GLU=0: off.
bool ggml_cuda_gemm_fold_glu_ok(ggml_backend_cuda_context & ctx, const ggml_tensor * glu, const ggml_tensor * mm);
void ggml_cuda_gemm_fold_set_glu(const ggml_tensor * glu);
bool ggml_cuda_gemm_fold_glu_pending();

// pf-gemm: compute the MUL_MAT dst whose SwiGLU (dst->src[1]) was skipped, first thing in its MUL_MAT dispatch.
// true: done by the fold. false: the fold declined, the GLU has now been computed by its own kernel, and dst must
// take the normal route.
bool ggml_cuda_gemm_fold_glu_mul_mat(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
