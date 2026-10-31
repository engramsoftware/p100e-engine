#pragma once

#include "common.cuh"

// Width 1..4 matrix-vector products for Q4_K / Q5_K / IQ4_XS (and repacked Q6_K) on sm_60 (GP100), built on
// HFMA2 with the half2 activations of mmvq-f16-sm60.cu.  See kq-sm60.cu.
//
// Returns true when the product was computed here; false: use the normal path.
bool ggml_cuda_kq_mul_mat(
    ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

// Same for a Q6_K tensor in the repacked layout of q4r-sm60.cu (widths 1..4); false: not handled.
bool ggml_cuda_kq_q6r_mul_mat(
    ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

// Same for a Q4_K tensor in the q4kr layout of q4r-sm60.cu (GGML_KQR_Q4K=1), widths 1..8 (5..8 as two passes);
// false: not handled (the caller then uses the unpacked or to_f16 path).
bool ggml_cuda_kq_q4kr_mul_mat(
    ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

// Same for a Q3_K tensor in the q3r layout of q4r-sm60.cu (GGML_Q3R=1), widths 1..8 (5..8 as two passes).
bool ggml_cuda_kq_q3r_mul_mat(
    ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
