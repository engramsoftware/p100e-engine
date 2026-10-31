#pragma once

// pf-xchg: tensor-parallel exchange helpers (P100 prefill port).
//
// Ported from Kmic-68/llama.cpp e5c264b71 and dce17bf1b (MIT): ship tensor-parallel partial sums as
// f16 when the producing matmul's compute type makes them f16-exact.  On sm_6x the cuBLAS
// COMPUTE_16F path writes an f16 result that is then widened to f32, so such a partial holds only
// f16-representable values and narrowing it for transport is exact in both directions.
//
// Everything in this file is self-contained (CUDA runtime + cuda_fp16 only) so that
// tests/test-pf-xchg-kernels.cu can compile it on its own and check it on any GPU.
//
// Env switches (all default OFF; unset = stock behaviour):
//   GGML_PF_XCHG=1          duplex copy stream for peer copies + Kmic's race fixes (needs GGML_CUDA_P2P and
//                           peer access both ways; ggml-cuda.cu)
//   GGML_PF_XCHG_F16=1      with GGML_PF_XCHG=1: exact f16 wire for eligible peer copies
//   GGML_PF_AR_F16=1        exact f16 wire on the GGML_CUDA_AR_PASCAL=1 host-staged copy-engine path (allreduce.cu)
//   GGML_PF_XCHG_VERIFY=1   diagnostic: check every f16-eligible exchange for exactness (host sync, slow);
//                           an inexact one is logged and sent as f32
//   GGML_PF_TEST_ANY_ARCH=1 TEST ONLY: lift the sm_6x gate of the f16 producer check, let the exchange run
//                           between virtual devices on one GPU (GGML_CUDA_DEVICES), and allow GGML_PF_AR_F16
//                           on the Volta+ internal AllReduce, so the logic can be exercised on a desktop GPU
//                         =2 TEST ONLY: as 1, and also skip the producer check (any batched MUL_MAT partial is
//                           eligible), so that only the probe / verify gate stands between a non-f16-exact
//                           partial and the wire -- a test of that safety net

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <cstdlib>

#include "pf-test.cuh"

static inline bool ggml_cuda_pf_env_on(const char * name) {
    const char * v = getenv(name);
    return v != nullptr && atoi(v) != 0;
}

static inline bool ggml_cuda_pf_xchg_requested() {
    static const bool v = ggml_cuda_pf_env_on("GGML_PF_XCHG");
    return v;
}

static inline bool ggml_cuda_pf_xchg_f16_requested() {
    static const bool v = ggml_cuda_pf_xchg_requested() && ggml_cuda_pf_env_on("GGML_PF_XCHG_F16");
    return v;
}

static inline bool ggml_cuda_pf_ar_f16_requested() {
    static const bool v = ggml_cuda_pf_env_on("GGML_PF_AR_F16");
    return v;
}

static inline bool ggml_cuda_pf_verify() {
    static const bool v = ggml_cuda_pf_env_on("GGML_PF_XCHG_VERIFY");
    return v;
}

// GGML_PF_TEST_ANY_ARCH helpers (ggml_cuda_pf_test_level / _any_arch / _force_eligible): pf-test.cuh, shared with
// pf-gemm and pf-attn.

// ---------------------------------------------------------------------------------------------------
// Kernels.  Grid-stride, pairs of elements as float2/half2 when both pointers allow it, scalar tail.
// ---------------------------------------------------------------------------------------------------

// y[i] = (half) x[i] (round to nearest even).  When flag != nullptr, *flag is set to 1 if any element
// does not survive the round trip, i.e. (float)(half) x[i] != x[i]; NaN trips it too (NaN != NaN),
// which is the conservative direction.  y may be nullptr: check only (the one-time probe).
static __global__ void k_pf_f32_to_f16(const float * __restrict__ x, half * __restrict__ y, const int64_t n,
        int * __restrict__ flag) {
    const int64_t stride = (int64_t) gridDim.x*blockDim.x;
    bool bad = false;
    for (int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x; i < n; i += stride) {
        const float v = x[i];
        const half  h = __float2half_rn(v);
        if (y != nullptr) {
            y[i] = h;
        }
        bad = bad || __half2float(h) != v;
    }
    if (flag != nullptr && bad) {
        *flag = 1;
    }
}

static __global__ void k_pf_f32_to_f16_x2(const float2 * __restrict__ x, half2 * __restrict__ y, const int64_t n2,
        int * __restrict__ flag) {
    const int64_t stride = (int64_t) gridDim.x*blockDim.x;
    bool bad = false;
    for (int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x; i < n2; i += stride) {
        const float2 v = x[i];
        const half2  h = __floats2half2_rn(v.x, v.y);
        if (y != nullptr) {
            y[i] = h;
        }
        const float2 w = __half22float2(h);
        bad = bad || w.x != v.x || w.y != v.y;
    }
    if (flag != nullptr && bad) {
        *flag = 1;
    }
}

static __global__ void k_pf_f16_to_f32(const half * __restrict__ x, float * __restrict__ y, const int64_t n) {
    const int64_t stride = (int64_t) gridDim.x*blockDim.x;
    for (int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x; i < n; i += stride) {
        y[i] = __half2float(x[i]);
    }
}

static __global__ void k_pf_f16_to_f32_x2(const half2 * __restrict__ x, float2 * __restrict__ y, const int64_t n2) {
    const int64_t stride = (int64_t) gridDim.x*blockDim.x;
    for (int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x; i < n2; i += stride) {
        y[i] = __half22float2(x[i]);
    }
}

// own[i] = own[i] + (float) peer[i]: the butterfly's widen + ADD(own, tmp) in one pass, bit for bit
// (Kmic k_add_f16_into_f32, used by the token-chunked exchange hook).
static __global__ void k_pf_add_f16_into_f32(float * __restrict__ own, const half * __restrict__ peer, const int64_t n) {
    const int64_t stride = (int64_t) gridDim.x*blockDim.x;
    for (int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x; i < n; i += stride) {
        own[i] = own[i] + __half2float(peer[i]);
    }
}

static inline unsigned ggml_cuda_pf_grid(const int64_t n, const int block) {
    int64_t g = (n + block - 1)/block;
    if (g < 1) {
        g = 1;
    }
    if (g > 65535) {
        g = 65535; // grid-stride covers the rest
    }
    return (unsigned) g;
}

static inline bool ggml_cuda_pf_aligned(const void * p, const size_t a) {
    return p == nullptr || ((uintptr_t) p % a) == 0;
}

// Narrow n floats to half; optional exactness flag (device int, caller zeroes it); y may be nullptr.
static inline void ggml_cuda_pf_f32_to_f16(const float * x, half * y, const int64_t n, int * flag, cudaStream_t stream) {
    if (n <= 0) {
        return;
    }
    const int block = 256;
    const int64_t n2 = n/2;
    if (n2 > 0 && ggml_cuda_pf_aligned(x, sizeof(float2)) && ggml_cuda_pf_aligned(y, sizeof(half2))) {
        k_pf_f32_to_f16_x2<<<ggml_cuda_pf_grid(n2, block), block, 0, stream>>>(
            (const float2 *) x, (half2 *) y, n2, flag);
        if (n % 2 != 0) {
            k_pf_f32_to_f16<<<1, 32, 0, stream>>>(x + 2*n2, y ? y + 2*n2 : nullptr, 1, flag);
        }
    } else {
        k_pf_f32_to_f16<<<ggml_cuda_pf_grid(n, block), block, 0, stream>>>(x, y, n, flag);
    }
}

static inline void ggml_cuda_pf_f16_to_f32(const half * x, float * y, const int64_t n, cudaStream_t stream) {
    if (n <= 0) {
        return;
    }
    const int block = 256;
    const int64_t n2 = n/2;
    if (n2 > 0 && ggml_cuda_pf_aligned(x, sizeof(half2)) && ggml_cuda_pf_aligned(y, sizeof(float2))) {
        k_pf_f16_to_f32_x2<<<ggml_cuda_pf_grid(n2, block), block, 0, stream>>>((const half2 *) x, (float2 *) y, n2);
        if (n % 2 != 0) {
            k_pf_f16_to_f32<<<1, 32, 0, stream>>>(x + 2*n2, y + 2*n2, 1);
        }
    } else {
        k_pf_f16_to_f32<<<ggml_cuda_pf_grid(n, block), block, 0, stream>>>(x, y, n);
    }
}

static inline void ggml_cuda_pf_add_f16_into_f32(float * own, const half * peer, const int64_t n, cudaStream_t stream) {
    if (n <= 0) {
        return;
    }
    k_pf_add_f16_into_f32<<<ggml_cuda_pf_grid(n, 256), 256, 0, stream>>>(own, peer, n);
}

// ---------------------------------------------------------------------------------------------------
// Cross-TU entry points (defined in ggml-cuda.cu).
// ---------------------------------------------------------------------------------------------------

// GGML_PF_XCHG=1 is in effect for this pair of CUDA devices: GGML_CUDA_P2P is set and peer access works
// both ways (or, with GGML_PF_TEST_ANY_ARCH=1, the two are virtual devices on one GPU).  Logs once per pair.
bool ggml_cuda_pf_xchg_engaged(int device_a, int device_b);

// ---------------------------------------------------------------------------------------------------
// Token-chunked exchange hook (for the integrate phase; consumer ported from Kmic-68/llama.cpp 1f92a1b48, MIT).
//
// A producer that computes a tensor-parallel partial (the last node of a meta-backend subgraph, a MUL_MAT) in n token
// chunks calls ggml_cuda_pf_xchg_chunks_begin() before the first chunk and ggml_cuda_pf_xchg_chunks_mark() after the
// work of each chunk has been enqueued on ctx.stream().  When both GPUs registered matching chunks and the exchange
// is f16-eligible (GGML_PF_XCHG=1 + GGML_PF_XCHG_F16=1), the next AllReduce of that tensor narrows and sends chunk c
// on the copy stream as soon as its event fires, while the compute stream runs the later chunks.  Same wire and
// same sum as the f16 butterfly; only the overlap differs.  Nothing in this branch registers chunks yet, so the
// consumer is inert.  The marks must not be recorded while ctx.stream() is being captured into a CUDA graph:
// other streams wait on these events.
// ---------------------------------------------------------------------------------------------------
#define GGML_CUDA_PF_XCHG_MAX_CHUNKS 8

struct ggml_backend_cuda_context;
struct ggml_tensor;

// col[0..n]: chunk c covers tokens [col[c], col[c+1]) of t (col[0] == 0, col[n] == t->ne[1]); 2 <= n <= MAX_CHUNKS.
// Returns false (and registers nothing) when the hook is not active or the arguments do not fit.
bool ggml_cuda_pf_xchg_chunks_begin(ggml_backend_cuda_context & ctx, const ggml_tensor * t, const int64_t * col, int n);
void ggml_cuda_pf_xchg_chunks_mark(ggml_backend_cuda_context & ctx, int c);
