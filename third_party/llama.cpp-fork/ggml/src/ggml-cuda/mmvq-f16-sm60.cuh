#pragma once

#include "common.cuh"

// Q4_1 GEMV specialised for sm_60 (GP100): no DP4A, but full-rate HFMA2.
// Returns false when the shape or type does not qualify; the caller then falls back to plain MMVQ.
bool ggml_cuda_mmvq_f16_sm60_supported(
    const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, const ggml_tensor * dst,
    const void * fusion);

void ggml_cuda_mmvq_f16_sm60(
    ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

// Converts f32 activations (ncols columns, row stride s11 floats) to the half2 layout the sm_60 kernels read:
// aq holds 16 half2 per 32-element block, ads one (amax, sum/amax) half2 per block.
void ggml_cuda_a16_quantize(const float * x, half2 * aq, half2 * ads, int64_t s11, int nblocks,
        int ncols, bool vec4, cudaStream_t stream);

// Activations for src1 in the half2 layout, converted once per distinct src1 per graph and reused by
// sibling matmuls.  false: cache unavailable, prepare them yourself.  GGML_A16_CACHE=0 disables.
bool ggml_cuda_a16_cached(ggml_backend_cuda_context & ctx, const ggml_tensor * src1, int nblocks,
        const half2 ** aq, const half2 ** ads);
