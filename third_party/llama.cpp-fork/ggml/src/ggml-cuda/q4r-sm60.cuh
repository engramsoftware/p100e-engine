#pragma once

#include "common.cuh"

// Q4_0 weights repacked in place for sm_60 (GP100).  See q4r-sm60.cu.

// MUL_MAT entry.  May repack src0 in place on its first use; after that the tensor's bytes are in the
// repacked layout and every other consumer must go through ggml_cuda_q4r_unpacked_scope.
//   NONE:   not involved, use the normal path.
//   DONE:   the product was computed here.
//   UNPACK: src0 is repacked but this shape is not handled; run the normal path inside an unpacked scope.
enum { GGML_CUDA_Q4R_NONE = 0, GGML_CUDA_Q4R_DONE = 1, GGML_CUDA_Q4R_UNPACK = 2 };
int ggml_cuda_q4r_mul_mat(
    ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

bool ggml_cuda_q4r_is_repacked(const ggml_tensor * t);

// While alive, every repacked source of dst points at a temporary copy in the standard Q4_0 layout.
struct ggml_cuda_q4r_unpacked_scope {
    ggml_cuda_q4r_unpacked_scope(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
    ~ggml_cuda_q4r_unpacked_scope();

    ggml_backend_cuda_context & ctx;
    ggml_tensor * srcs[GGML_MAX_SRC] = {};
    void        * saved[GGML_MAX_SRC] = {};
    void        * tmp[GGML_MAX_SRC] = {};
    size_t        tmp_size[GGML_MAX_SRC] = {};
};

// Forget repack state for a freed device range.
void ggml_cuda_q4r_forget_range(const void * base, size_t size);

// GGML_Q6R2=1: repacked Q6_K tensors use the q6r2 ql/qh planes (q4r-sm60.cu), read by kq-sm60.cu.
bool ggml_cuda_q6r2_on();

// Dequantize a repacked src0 (ggml_cuda_q4r_is_repacked) to contiguous row-major f16.
void ggml_cuda_q4r_to_f16(const ggml_tensor * src0, half * dst, cudaStream_t stream);

// pf-gemm (fold GEMM): take the first-sight decision for t now, exactly as ggml_cuda_q4r_mul_mat would at t's own
// MUL_MAT (same checks: env, capture, device, eligibility; repacks in place when it decides PACKED), without
// computing anything. For a MUL_MAT that runs without passing through ggml_cuda_q4r_mul_mat (the fold GEMM's
// paired partner, the MUL_MAT of a skipped SwiGLU). Returns ggml_cuda_q4r_is_repacked(t) afterwards.
bool ggml_cuda_q4r_settle(ggml_backend_cuda_context & ctx, const ggml_tensor * t);
