// p100e::fast prefill GEMM, see gemm.cuh.
// The fold kernel is a port of gemm_fold_kernel from our llama.cpp fork (ggml-cuda/gemm-fold.cu, ported there from
// Kmic-68/llama.cpp fc1c9056f, MIT): 128x128 tile, 8x8 outputs per thread as half2 (lanes = even/odd k), XOR-swizzled
// smem, register double buffering, f16 chains folded to f32 every FOLD_K2 k2-steps.
#include "gemm.cuh"
#include "fast/unpack.cuh"

#include "convert.cuh"   // third_party/ggml: ggml_get_to_fp16_cuda (raw GGUF layout dequant)
#include "gguf.h"        // get_type_info (block size / bytes)

#include <cuda.h>
#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace p100e {
namespace fast {

namespace {

constexpr int BM = 128;
constexpr int BK2 = 16;        // k2 (half2) steps per smem tile = 32 values of K
constexpr int FOLD_K2_DEFAULT = 128;   // fold every 128 k2 steps = 256 values of K
constexpr int XEXP = 3;        // per-token prescale: max |x| -> [8, 16)

inline size_t align256(size_t v) { return (v + 255) & ~(size_t) 255; }

void ck(cudaError_t e, const char * what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("fast::gemm: ") + what + ": " + cudaGetErrorString(e));
}
void cbk(cublasStatus_t e, const char * what) {
    if (e != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string("fast::gemm: ") + what + " cublas status " + std::to_string((int) e));
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    return v;
}

// one block per token: X16[n] = f16(x[n] * 2^-e), cs[n] = 2^e, e chosen so max|x[n]| * 2^-e is in [8, 16)
__global__ void __launch_bounds__(256) prescale_kernel(const float * __restrict__ X, half * __restrict__ X16,
                                                       float * __restrict__ cs, const int K) {
    const int n = blockIdx.x;
    const float4 * x = (const float4 *) (X + (int64_t) n * K);
    const int K4 = K / 4;
    float m = 0.0f;
    for (int k = threadIdx.x; k < K4; k += 256) {
        const float4 v = x[k];
        m = fmaxf(m, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
    }
    __shared__ float sm[8];
    m = warp_max(m);
    if ((threadIdx.x & 31) == 0) sm[threadIdx.x >> 5] = m;
    __syncthreads();
    if (threadIdx.x < 32) {
        m = threadIdx.x < 8 ? sm[threadIdx.x] : 0.0f;
        m = warp_max(m);
        if (threadIdx.x == 0) sm[0] = m;
    }
    __syncthreads();
    m = sm[0];
    const int e = m > 0.0f && isfinite(m) ? ilogbf(m) - XEXP : 0;
    const float s = ldexpf(1.0f, -e);
    half2 * y = (half2 *) (X16 + (int64_t) n * K);
    for (int k = threadIdx.x; k < K4; k += 256) {
        const float4 v = x[k];
        y[2 * k + 0] = __floats2half2_rn(v.x * s, v.y * s);
        y[2 * k + 1] = __floats2half2_rn(v.z * s, v.w * s);
    }
    if (threadIdx.x == 0) cs[n] = ldexpf(1.0f, e);
}

__device__ __forceinline__ float fold_h(const half2 p) {
    return __half2float(__hadd(__low2half(p), __high2half(p)));
}

// Y[n*sy + m] = cs[n] * sum_k W[m][k] * X[n][k]; W: M x K f16, X: N x K f16 (both K-contiguous). K % 32 == 0.
// Tile BM=128 rows x 16*JN tokens; each thread 8 rows x JN tokens. JN = 8 (128 tokens), 2 or 1 for short batches.
template <int FOLD_K2, int JN>
__global__ void __launch_bounds__(256, 1) fold_kernel(const half * __restrict__ W, const half * __restrict__ X,
                                                      const float * __restrict__ cs, float * __restrict__ Y,
                                                      const int M, const int N, const int K, const int64_t sy) {
    constexpr int BNJ = 16 * JN;
    constexpr int NLB = (BNJ * 4 + 255) / 256;   // uint4 loads of X per thread per tile
    __shared__ __align__(16) uint32_t As[2][BK2][BM];
    __shared__ __align__(16) uint32_t Bs[2][BK2][BNJ];

    const int t = threadIdx.x;
    const int tx = t & 15;
    const int ty = t >> 4;
    const int m0 = blockIdx.x * BM;
    const int n0 = blockIdx.y * BNJ;

    float acc[8][JN];
    half2 h[8][JN];
#pragma unroll
    for (int i = 0; i < 8; i++) {
#pragma unroll
        for (int j = 0; j < JN; j++) {
            acc[i][j] = 0.0f;
            h[i][j] = make_half2(0.0f, 0.0f);
        }
    }

    uint4 ra[2], rb[NLB];
    auto gload = [&](const int k0) {
#pragma unroll
        for (int i = 0; i < 2; i++) {
            const int l = t + 256 * i;
            const int r = l >> 2;
            const int c = l & 3;
            ra[i] = m0 + r < M ? *(const uint4 *) (W + (int64_t) (m0 + r) * K + k0 + c * 8) : make_uint4(0, 0, 0, 0);
        }
#pragma unroll
        for (int i = 0; i < NLB; i++) {
            const int l = t + 256 * i;
            const int r = l >> 2;
            const int c = l & 3;
            rb[i] = l < BNJ * 4 && n0 + r < N ? *(const uint4 *) (X + (int64_t) (n0 + r) * K + k0 + c * 8)
                                              : make_uint4(0, 0, 0, 0);
        }
    };
    auto sstore = [&](const int buf) {
#pragma unroll
        for (int i = 0; i < 2; i++) {
            const int l = t + 256 * i;
            const int r = l >> 2;
            const int c = l & 3;
            const int rs = r ^ (c << 3);   // XOR swizzle: avoids the 4-way store conflict, reads stay vectors
            As[buf][c * 4 + 0][rs] = ra[i].x; As[buf][c * 4 + 1][rs] = ra[i].y;
            As[buf][c * 4 + 2][rs] = ra[i].z; As[buf][c * 4 + 3][rs] = ra[i].w;
        }
#pragma unroll
        for (int i = 0; i < NLB; i++) {
            const int l = t + 256 * i;
            if (l >= BNJ * 4) break;
            const int r = l >> 2;
            const int c = l & 3;
            const int rs = r ^ ((c << 3) & (BNJ - 1));
            Bs[buf][c * 4 + 0][rs] = rb[i].x; Bs[buf][c * 4 + 1][rs] = rb[i].y;
            Bs[buf][c * 4 + 2][rs] = rb[i].z; Bs[buf][c * 4 + 3][rs] = rb[i].w;
        }
    };

    const int nt = K / (2 * BK2);
    gload(0);
    sstore(0);
    __syncthreads();

    for (int it = 0; it < nt; it++) {
        const int buf = it & 1;
        if (it + 1 < nt) gload((it + 1) * 2 * BK2);
        const bool restart = (it * BK2) % FOLD_K2 == 0;
#pragma unroll
        for (int k2 = 0; k2 < BK2; k2++) {
            const int sw = (k2 >> 2) << 3;
            const int swb = sw & (BNJ - 1);
            const uint4 a0 = *(const uint4 *) &As[buf][k2][(ty * 4) ^ sw];
            const uint4 a1 = *(const uint4 *) &As[buf][k2][(64 + ty * 4) ^ sw];
            const uint32_t a[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
            uint32_t b[JN];
            if constexpr (JN == 8) {
                const uint4 b0 = *(const uint4 *) &Bs[buf][k2][(tx * 4) ^ swb];
                const uint4 b1 = *(const uint4 *) &Bs[buf][k2][(64 + tx * 4) ^ swb];
                b[0] = b0.x; b[1] = b0.y; b[2] = b0.z; b[3] = b0.w; b[4] = b1.x; b[5] = b1.y; b[6] = b1.z; b[7] = b1.w;
            } else if constexpr (JN == 2) {
                const uint2 b0 = *(const uint2 *) &Bs[buf][k2][(tx * 2) ^ swb];
                b[0] = b0.x; b[1] = b0.y;
            } else {
                b[0] = Bs[buf][k2][tx ^ swb];
            }
#pragma unroll
            for (int j = 0; j < JN; j++) {
#pragma unroll
                for (int i = 0; i < 8; i++) {
                    const half2 ai = *(const half2 *) &a[i];
                    const half2 bj = *(const half2 *) &b[j];
                    h[i][j] = k2 == 0 && restart ? __hmul2(bj, ai) : __hfma2(bj, ai, h[i][j]);
                }
            }
        }
        if (((it + 1) * BK2) % FOLD_K2 == 0 || it + 1 == nt) {
#pragma unroll
            for (int i = 0; i < 8; i++) {
#pragma unroll
                for (int j = 0; j < JN; j++) acc[i][j] += fold_h(h[i][j]);
            }
        }
        if (it + 1 < nt) sstore(buf ^ 1);
        __syncthreads();
    }

#pragma unroll
    for (int j = 0; j < JN; j++) {
        const int n = n0 + (JN == 8 ? (j < 4 ? tx * 4 + j : 64 + tx * 4 + j - 4) : tx * JN + j);
        if (n >= N) continue;
        const float s = cs[n];
#pragma unroll
        for (int ih = 0; ih < 2; ih++) {
            const int m = m0 + ih * 64 + ty * 4;
            if (m >= M) continue;
            const float4 v = make_float4(acc[ih * 4 + 0][j] * s, acc[ih * 4 + 1][j] * s, acc[ih * 4 + 2][j] * s,
                                         acc[ih * 4 + 3][j] * s);
            *(float4 *) (Y + n * sy + m) = v;
        }
    }
}

template <int FOLD_K2>
void launch_fold(const half * w16, const half * x16, const float * cs, float * y, int64_t nr, int T, int64_t n_in,
                 int64_t sy, cudaStream_t s) {
    const unsigned gm = (unsigned) ((nr + BM - 1) / BM);
    if (T <= 16) {
        fold_kernel<FOLD_K2, 1><<<dim3(gm, (T + 15) / 16), 256, 0, s>>>(w16, x16, cs, y, (int) nr, T, (int) n_in, sy);
    } else if (T <= 64) {
        fold_kernel<FOLD_K2, 2><<<dim3(gm, (T + 31) / 32), 256, 0, s>>>(w16, x16, cs, y, (int) nr, T, (int) n_in, sy);
    } else {
        fold_kernel<FOLD_K2, 8><<<dim3(gm, (T + 127) / 128), 256, 0, s>>>(w16, x16, cs, y, (int) nr, T, (int) n_in, sy);
    }
}

__global__ void widen_kernel(const half * __restrict__ a, float * __restrict__ y, const int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __half2float(a[i]);
}

__global__ void to_half_kernel(const float * __restrict__ a, half * __restrict__ y, const int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2half(a[i]);
}

int64_t row_bytes_of(uint32_t type, int64_t n_in) {
    const qtype_info & ti = get_type_info(type);
    return n_in / ti.blck * (int64_t) ti.size;
}

// ================= prefill pipeline kernels (tp_model::forward_prefill) =================

__device__ __forceinline__ float block_max256(float m) {   // blockDim.x <= 256, result broadcast
    __shared__ float smx[8];
    m = warp_max(m);
    __syncthreads();
    if ((threadIdx.x & 31) == 0) smx[threadIdx.x >> 5] = m;
    __syncthreads();
    const int nw = (blockDim.x + 31) >> 5;
    m = (threadIdx.x & 31) < nw ? smx[threadIdx.x & 31] : 0.0f;
    return warp_max(m);
}

__device__ __forceinline__ float absmax4(float m, const float4 v) {
    return fmaxf(m, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
}

// prescale_rows_kernel with one read of X (fork gemm_fold_prescale_v, GGML_CUDA_FOLD_PRESCALE_V): the
// row stays in registers (NV float4 per thread) between the max and the scaling. The max is order-independent and
// every element is scaled and rounded as in prescale_rows_kernel, so the outputs are the same bits. K4 <= 256 * NV.
template <int NV>
__global__ void __launch_bounds__(256) prescale_rows_v_kernel(const float * __restrict__ X, const int64_t ldv,
                                                              half * __restrict__ X16, const int64_t ldx,
                                                              float * __restrict__ cs, const int K) {
    const int n = blockIdx.x;
    const float4 * x = (const float4 *) (X + (int64_t) n * ldv);
    const int K4 = K / 4;
    float4 r[NV];
    float m = 0.0f;
#pragma unroll
    for (int i = 0; i < NV; i++) {
        const int k = threadIdx.x + 256 * i;
        r[i] = k < K4 ? x[k] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        m = absmax4(m, r[i]);
    }
    m = block_max256(m);
    const int e = m > 0.0f && isfinite(m) ? ilogbf(m) - XEXP : 0;
    const float s = ldexpf(1.0f, -e);
    half2 * y = (half2 *) (X16 + (int64_t) n * ldx);
#pragma unroll
    for (int i = 0; i < NV; i++) {
        const int k = threadIdx.x + 256 * i;
        if (k < K4) {
            y[2 * k + 0] = __floats2half2_rn(r[i].x * s, r[i].y * s);
            y[2 * k + 1] = __floats2half2_rn(r[i].z * s, r[i].w * s);
        }
    }
    if (threadIdx.x == 0) cs[n] = ldexpf(1.0f, e);
}

// prescale_kernel with row strides (identical arithmetic)
__global__ void __launch_bounds__(256) prescale_rows_kernel(const float * __restrict__ X, const int64_t ldv,
                                                            half * __restrict__ X16, const int64_t ldx,
                                                            float * __restrict__ cs, const int K) {
    const int n = blockIdx.x;
    const float4 * x = (const float4 *) (X + (int64_t) n * ldv);
    const int K4 = K / 4;
    float m = 0.0f;
    for (int k = threadIdx.x; k < K4; k += 256) m = absmax4(m, x[k]);
    m = block_max256(m);
    const int e = m > 0.0f && isfinite(m) ? ilogbf(m) - XEXP : 0;
    const float s = ldexpf(1.0f, -e);
    half2 * y = (half2 *) (X16 + (int64_t) n * ldx);
    for (int k = threadIdx.x; k < K4; k += 256) {
        const float4 v = x[k];
        y[2 * k + 0] = __floats2half2_rn(v.x * s, v.y * s);
        y[2 * k + 1] = __floats2half2_rn(v.z * s, v.w * s);
    }
    if (threadIdx.x == 0) cs[n] = ldexpf(1.0f, e);
}

// fast::fused.cu block_sum (same reduction tree, so the norm scale is bit-identical)
__device__ __forceinline__ float warp_sum_f(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}
__device__ __forceinline__ float block_sum_f(float v) {
    __shared__ float sh[32];
    v = warp_sum_f(v);
    if (blockDim.x == 32) return v;
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    __syncthreads();
    if (lane == 0) sh[wid] = v;
    __syncthreads();
    const int nw = blockDim.x >> 5;
    v = lane < nw ? sh[lane] : 0.0f;
    return warp_sum_f(v);
}

// one block per token: x += p0 (+ p1); v = rmsnorm(x) * w exactly as fast::rmsnorm's norm_vec_k<.., NV> (same
// thread mapping / reduction); then prescale v into X16 / cs like prescale_kernel.
__device__ __forceinline__ float4 load_part4(const float * p, const int64_t off, const int i) {
    return reinterpret_cast<const float4 *>(p + off)[i];
}
__device__ __forceinline__ float4 load_part4(const half * p, const int64_t off, const int i) {   // f16 wire: widen
    const uint2 u = reinterpret_cast<const uint2 *>(p + off)[i];
    const float2 a = __half22float2(*reinterpret_cast<const half2 *>(&u.x));
    const float2 b = __half22float2(*reinterpret_cast<const half2 *>(&u.y));
    return make_float4(a.x, a.y, b.x, b.y);
}

// PT = float (f32 partials) or half (the f16 exchange wire; widened exactly, then the same adds)
template <int NV, typename PT>
__global__ void add_norm_prescale_kernel(float * __restrict__ x, const PT * __restrict__ p0,
                                         const PT * __restrict__ p1, const float * __restrict__ w, const int n,
                                         const float eps, half * __restrict__ X16, const int64_t ldx,
                                         float * __restrict__ cs) {
    const int n4 = n >> 2;
    const int64_t off = (int64_t) blockIdx.x * n;
    float4 * x4 = reinterpret_cast<float4 *>(x + off);
    float4 v[NV];
    float ss = 0.0f;
#pragma unroll
    for (int k = 0; k < NV; ++k) {
        const int i = threadIdx.x + k * blockDim.x;
        if (i < n4) {
            float4 a = x4[i];
            if (p0) {
                const float4 b = load_part4(p0, off, i);
                a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
            }
            if (p1) {
                const float4 b = load_part4(p1, off, i);
                a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
            }
            if (p0 || p1) x4[i] = a;
            v[k] = a;
            ss += a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w;
        }
    }
    ss = block_sum_f(ss);
    const float sc = rsqrtf(ss / n + eps);
    const float4 * w4 = reinterpret_cast<const float4 *>(w);
    float m = 0.0f;
#pragma unroll
    for (int k = 0; k < NV; ++k) {
        const int i = threadIdx.x + k * blockDim.x;
        if (i < n4) {
            float4 o = v[k];
            o.x *= sc; o.y *= sc; o.z *= sc; o.w *= sc;
            if (w) { const float4 ww = w4[i]; o.x *= ww.x; o.y *= ww.y; o.z *= ww.z; o.w *= ww.w; }
            v[k] = o;
            m = absmax4(m, o);
        }
    }
    m = block_max256(m);
    const int e = m > 0.0f && isfinite(m) ? ilogbf(m) - XEXP : 0;
    const float s = ldexpf(1.0f, -e);
    half2 * y = (half2 *) (X16 + (int64_t) blockIdx.x * ldx);
#pragma unroll
    for (int k = 0; k < NV; ++k) {
        const int i = threadIdx.x + k * blockDim.x;
        if (i < n4) {
            y[2 * i + 0] = __floats2half2_rn(v[k].x * s, v[k].y * s);
            y[2 * i + 1] = __floats2half2_rn(v[k].z * s, v[k].w * s);
        }
    }
    if (threadIdx.x == 0) cs[blockIdx.x] = ldexpf(1.0f, e);
}

// x = (x + p0) + p1, float4 elementwise (p1 may be null)
template <typename PT>
__global__ void add2_kernel(float4 * __restrict__ x, const PT * __restrict__ p0, const PT * __restrict__ p1,
                            const int64_t n4) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += (int64_t) gridDim.x * blockDim.x) {
        float4 a = x[i];
        const float4 b0 = load_part4(p0 + 4 * i, 0, 0);
        a.x += b0.x; a.y += b0.y; a.z += b0.z; a.w += b0.w;
        if (p1) {
            const float4 c = load_part4(p1 + 4 * i, 0, 0);
            a.x += c.x; a.y += c.y; a.z += c.z; a.w += c.w;
        }
        x[i] = a;
    }
}

__device__ __forceinline__ float silu_k(float v) { return v / (1.0f + expf(-v)); }   // == k::silu

// one block of 256 per token: v = silu(g) * u held in registers (NV float4 per thread), then prescaled
template <int NV>
__global__ void __launch_bounds__(256) glu_prescale_kernel(const float * __restrict__ g, const float * __restrict__ u,
                                                           const int K, half * __restrict__ X16, const int64_t ldx,
                                                           float * __restrict__ cs) {
    const int K4 = K / 4;
    const int64_t off = (int64_t) blockIdx.x * K;
    const float4 * g4 = reinterpret_cast<const float4 *>(g + off);
    const float4 * u4 = reinterpret_cast<const float4 *>(u + off);
    float4 v[NV];
    float m = 0.0f;
#pragma unroll
    for (int k = 0; k < NV; ++k) {
        const int i = threadIdx.x + 256 * k;
        if (i < K4) {
            const float4 a = g4[i], b = u4[i];
            v[k] = make_float4(silu_k(a.x) * b.x, silu_k(a.y) * b.y, silu_k(a.z) * b.z, silu_k(a.w) * b.w);
            m = absmax4(m, v[k]);
        }
    }
    m = block_max256(m);
    const int e = m > 0.0f && isfinite(m) ? ilogbf(m) - XEXP : 0;
    const float s = ldexpf(1.0f, -e);
    half2 * y = (half2 *) (X16 + (int64_t) blockIdx.x * ldx);
#pragma unroll
    for (int k = 0; k < NV; ++k) {
        const int i = threadIdx.x + 256 * k;
        if (i < K4) {
            y[2 * i + 0] = __floats2half2_rn(v[k].x * s, v[k].y * s);
            y[2 * i + 1] = __floats2half2_rn(v[k].z * s, v[k].w * s);
        }
    }
    if (threadIdx.x == 0) cs[blockIdx.x] = ldexpf(1.0f, e);
}

// ---- segmented fold GEMM ----
// The chains / folds of fold_kernel<128, JN> (bit-identical), with the bookkeeping of the fork's gemm_fold_kernel_u2
// (Kmic-68 MIT): load / store offsets computed once and the tile loop unrolled by two (compile-time buffer index).
// Row tiles of up to 4 weights in one grid (blockIdx.x = global row tile); each tile sits wholly in one weight.
struct seg_tab {
    const half * w[4];
    float * y[4];
    half * y16[4];
    long long ldy[4];
    int rows[4];
    int tile0[4];
    int nseg;
};

// NOTE: fold_seg_kernel<8, false> also ships as a register-bank-fixed sm_60 cubin (src/fast/gemm_sass_fold8.h, used
// with fold_opts::sass on cc 6.0 devices). After any change to this kernel (or seg_tab / fold_h / the constants),
// rerun tools/sass/fold_bankfix.py; `fold_bankfix.py --check` tells whether the header is stale.
// OUT16 (fork gemm_fold_kernel<.., out16> and u2's epilogue, GGML_CUDA_GEMM_FOLD=2): outputs rounded to f16 and stored
// as half into y16 (saturated at +-65504, counted in *sat). raster > 1 (fork GEMM_FOLD_RASTER, gemm_fold_kernel_u2's
// mapping): the linear block id walks groups of `raster` row tiles x all token tiles, so co-resident blocks share
// weight tiles in L2; every block still computes one whole (row tile, token tile), so no output changes.
template <int JN, bool OUT16>
__global__ void __launch_bounds__(256, 1) fold_seg_kernel(const seg_tab tab, const half * __restrict__ X,
                                                          const int64_t ldx, const float * __restrict__ cs,
                                                          const int N, const int K, const int raster,
                                                          int * __restrict__ sat) {
    constexpr int BNJ = 16 * JN;
    constexpr int NLB = (BNJ * 4 + 255) / 256;   // uint4 loads of X per thread per tile
    constexpr int TPF = FOLD_K2_DEFAULT / BK2;    // tiles per fold (8)
    __shared__ __align__(16) uint32_t As[2][BK2][BM];
    __shared__ __align__(16) uint32_t Bs[2][BK2][BNJ];

    const int t = threadIdx.x;
    const int tx = t & 15;
    const int ty = t >> 4;
    int bt = blockIdx.x, bn = blockIdx.y;
    if (raster > 1) {
        const int gm = gridDim.x, gn = gridDim.y;
        const int lin = blockIdx.x + blockIdx.y * gm;
        const int grp = lin / (raster * gn);
        const int gsz = min(raster, gm - grp * raster);
        const int rr = lin % (raster * gn);
        bt = grp * raster + rr % gsz;
        bn = rr / gsz;
    }
    const half * W = tab.w[0];
    float * Y = tab.y[0];
    half * Y16 = tab.y16[0];
    int64_t ldy = tab.ldy[0];
    int M = tab.rows[0];
    int tb = 0;
#pragma unroll
    for (int i = 1; i < 4; i++) {
        if (i < tab.nseg && bt >= tab.tile0[i]) {
            W = tab.w[i]; Y = tab.y[i]; Y16 = tab.y16[i]; ldy = tab.ldy[i]; M = tab.rows[i]; tb = tab.tile0[i];
        }
    }
    const int m0 = (bt - tb) * BM;
    const int n0 = bn * BNJ;

    float acc[8][JN];
    half2 h[8][JN];
#pragma unroll
    for (int i = 0; i < 8; i++) {
#pragma unroll
        for (int j = 0; j < JN; j++) {
            acc[i][j] = 0.0f;
            h[i][j] = make_half2(0.0f, 0.0f);
        }
    }

    const uint4 * pa[2];
    bool va[2];
    int soa[2];
#pragma unroll
    for (int i = 0; i < 2; i++) {
        const int l = t + 256 * i, r = l >> 2, c = l & 3;
        va[i] = m0 + r < M;
        pa[i] = (const uint4 *) (W + (int64_t) (va[i] ? m0 + r : 0) * K + c * 8);
        soa[i] = (c * 4) * BM + (r ^ (c << 3));
    }
    const uint4 * pb[NLB];
    bool vb[NLB];
    int sob[NLB];
#pragma unroll
    for (int i = 0; i < NLB; i++) {
        const int l = t + 256 * i, r = l >> 2, c = l & 3;
        vb[i] = l < BNJ * 4 && n0 + r < N;
        pb[i] = (const uint4 *) (X + (int64_t) (vb[i] ? n0 + r : 0) * ldx + c * 8);
        sob[i] = l < BNJ * 4 ? (c * 4) * BNJ + (r ^ ((c << 3) & (BNJ - 1))) : 0;
    }

    uint4 ra[2] = {make_uint4(0, 0, 0, 0), make_uint4(0, 0, 0, 0)};
    uint4 rb[NLB];
#pragma unroll
    for (int i = 0; i < NLB; i++) rb[i] = make_uint4(0, 0, 0, 0);
    auto gload = [&](const int it) {   // tile it = K offset it*32 halves = it*4 uint4
#pragma unroll
        for (int i = 0; i < 2; i++) if (va[i]) ra[i] = pa[i][it * 4];
#pragma unroll
        for (int i = 0; i < NLB; i++) if (vb[i]) rb[i] = pb[i][it * 4];
    };
    auto sstore = [&](const int buf) {
#pragma unroll
        for (int i = 0; i < 2; i++) {
            uint32_t * a = &As[buf][0][0] + soa[i];
            a[0] = ra[i].x; a[BM] = ra[i].y; a[2 * BM] = ra[i].z; a[3 * BM] = ra[i].w;
        }
#pragma unroll
        for (int i = 0; i < NLB; i++) {
            if (t + 256 * i >= BNJ * 4) break;
            uint32_t * b = &Bs[buf][0][0] + sob[i];
            b[0] = rb[i].x; b[BNJ] = rb[i].y; b[2 * BNJ] = rb[i].z; b[3 * BNJ] = rb[i].w;
        }
    };
    auto tile = [&](const int buf, const bool restart) {
#pragma unroll
        for (int k2 = 0; k2 < BK2; k2++) {
            const int sw = (k2 >> 2) << 3;
            const int swb = sw & (BNJ - 1);
            const uint4 a0 = *(const uint4 *) &As[buf][k2][(ty * 4) ^ sw];
            const uint4 a1 = *(const uint4 *) &As[buf][k2][(64 + ty * 4) ^ sw];
            const uint32_t a[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
            uint32_t b[JN];
            if constexpr (JN == 8) {
                const uint4 b0 = *(const uint4 *) &Bs[buf][k2][(tx * 4) ^ swb];
                const uint4 b1 = *(const uint4 *) &Bs[buf][k2][(64 + tx * 4) ^ swb];
                b[0] = b0.x; b[1] = b0.y; b[2] = b0.z; b[3] = b0.w; b[4] = b1.x; b[5] = b1.y; b[6] = b1.z; b[7] = b1.w;
            } else if constexpr (JN == 2) {
                const uint2 b0 = *(const uint2 *) &Bs[buf][k2][(tx * 2) ^ swb];
                b[0] = b0.x; b[1] = b0.y;
            } else {
                b[0] = Bs[buf][k2][tx ^ swb];
            }
#pragma unroll
            for (int j = 0; j < JN; j++) {
#pragma unroll
                for (int i = 0; i < 8; i++) {
                    const half2 ai = *(const half2 *) &a[i];
                    const half2 bj = *(const half2 *) &b[j];
                    h[i][j] = k2 == 0 && restart ? __hmul2(bj, ai) : __hfma2(bj, ai, h[i][j]);
                }
            }
        }
    };
    auto fold = [&]() {
#pragma unroll
        for (int i = 0; i < 8; i++) {
#pragma unroll
            for (int j = 0; j < JN; j++) acc[i][j] += fold_h(h[i][j]);
        }
    };

    const int nt = K / (2 * BK2);   // even: K % 64 == 0
    gload(0);
    sstore(0);
    __syncthreads();
    for (int it = 0; it < nt; it += 2) {
        gload(it + 1);
        tile(0, it % TPF == 0);
        sstore(1);
        __syncthreads();
        if (it + 2 < nt) gload(it + 2);
        tile(1, false);
        if ((it + 2) % TPF == 0 || it + 2 >= nt) fold();
        if (it + 2 < nt) sstore(0);
        __syncthreads();
    }

#pragma unroll
    for (int j = 0; j < JN; j++) {
        const int n = n0 + (JN == 8 ? (j < 4 ? tx * 4 + j : 64 + tx * 4 + j - 4) : tx * JN + j);
        if (n >= N) continue;
        const float s = cs[n];
#pragma unroll
        for (int ih = 0; ih < 2; ih++) {
            const int m = m0 + ih * 64 + ty * 4;
            if (m >= M) continue;
            float4 v = make_float4(acc[ih * 4 + 0][j] * s, acc[ih * 4 + 1][j] * s, acc[ih * 4 + 2][j] * s,
                                   acc[ih * 4 + 3][j] * s);
            if constexpr (OUT16) {
                // round to nearest even as the fork's out16 epilogue; a finite value that would round to +-inf is
                // saturated to +-65504 instead (and counted): an inf would poison the residual stream
                float * pv = &v.x;
                bool big = false;
#pragma unroll
                for (int q = 0; q < 4; q++) {
                    if (fabsf(pv[q]) >= 65520.0f && isfinite(pv[q])) { big = true; pv[q] = copysignf(65504.0f, pv[q]); }
                }
                if (big && sat) atomicAdd(sat, 1);
                half2 hh[2] = {__floats2half2_rn(v.x, v.y), __floats2half2_rn(v.z, v.w)};
                *(uint2 *) (Y16 + n * ldy + m) = *(const uint2 *) hh;
            } else {
                *(float4 *) (Y + n * ldy + m) = v;
            }
        }
    }
}

} // namespace

// fold_seg_kernel<8, false> as a register-bank-fixed sm_60 cubin (tools/sass/fold_bankfix.py). Included here, after the
// kernel: its static_asserts pin seg_tab's layout and the kernel constants the binary was built with, and the CMake
// build runs `fold_bankfix.py --check` before this file compiles (a header built from another kernel text fails the
// build), so a stale binary cannot ship.
#include "fast/gemm_sass_fold8.h"

namespace {

// ---- fused-dequant fold GEMM (P100E_PF_FUSEDQ) ----
// fold_seg_kernel with the q4kr weights read packed: the A side loads, per 32-value K tile, 8 qs bytes of the row
// (one uint2) and, once per 256-value super-block, the 16-byte header, and dequantizes into exactly the halves
// kq_direct_k (matvec_unpack.cu) writes: d1 = dall * sc, m1 = dmin * m, half(d1 * q - m1), the same expressions.
// The shared tiles, the hfma2 chains and the folds are those of fold_seg_kernel, so the outputs are bit-identical to
// packed_to_f16 + fold_gemm. Kept apart from fold_seg_kernel (outside the text fold_bankfix.py hashes).
// tab.w[i] holds the q4kr row data (row stride 144 * K / 256 bytes, 16-aligned) cast to const half *.
// Thread slot (r, c) of the A tile = row r, values 8c..8c+7 of group g: in q4kr byte i of the group holds value i
// (low nibble) and value i + 16 (high nibble), so they are bytes 8(c&1)..+7, nibble c>>1.
__device__ __forceinline__ void q4k_scale_min(const int j, const uint4 hd, float & d1, float & m1) {
    // get_scale_min_k4 over the scale bytes q[0..11] = hd.y, hd.z, hd.w (byte k of word W: (W >> 8k) & 0xFF)
    const int k = j & 3;
    const uint32_t q0 = (hd.y >> (8 * k)) & 0xFF, q1 = (hd.z >> (8 * k)) & 0xFF, q2 = (hd.w >> (8 * k)) & 0xFF;
    uint8_t sc, m;
    if (j < 4) {
        sc = q0 & 63;
        m = q1 & 63;
    } else {
        sc = (q2 & 0xF) | ((q0 >> 6) << 4);
        m = (q2 >> 4) | ((q1 >> 6) << 4);
    }
    const half2 dm = *(const half2 *) &hd.x;
    const float dall = __low2half(dm);
    const float dmin = __high2half(dm);
    d1 = dall * sc;
    m1 = dmin * m;
}

template <int JN, bool OUT16>
__global__ void __launch_bounds__(256, 1) fold_seg_q4k_kernel(const seg_tab tab, const half * __restrict__ X,
                                                              const int64_t ldx, const float * __restrict__ cs,
                                                              const int N, const int K, const int raster,
                                                              int * __restrict__ sat) {
    constexpr int BNJ = 16 * JN;
    constexpr int NLB = (BNJ * 4 + 255) / 256;
    constexpr int TPF = FOLD_K2_DEFAULT / BK2;
    __shared__ __align__(16) uint32_t As[2][BK2][BM];
    __shared__ __align__(16) uint32_t Bs[2][BK2][BNJ];

    const int t = threadIdx.x;
    const int tx = t & 15;
    const int ty = t >> 4;
    int bt = blockIdx.x, bn = blockIdx.y;
    if (raster > 1) {
        const int gm = gridDim.x, gn = gridDim.y;
        const int lin = blockIdx.x + blockIdx.y * gm;
        const int grp = lin / (raster * gn);
        const int gsz = min(raster, gm - grp * raster);
        const int rr = lin % (raster * gn);
        bt = grp * raster + rr % gsz;
        bn = rr / gsz;
    }
    const char * W = (const char *) tab.w[0];
    float * Y = tab.y[0];
    half * Y16 = tab.y16[0];
    int64_t ldy = tab.ldy[0];
    int M = tab.rows[0];
    int tb = 0;
#pragma unroll
    for (int i = 1; i < 4; i++) {
        if (i < tab.nseg && bt >= tab.tile0[i]) {
            W = (const char *) tab.w[i]; Y = tab.y[i]; Y16 = tab.y16[i]; ldy = tab.ldy[i]; M = tab.rows[i]; tb = tab.tile0[i];
        }
    }
    const int m0 = (bt - tb) * BM;
    const int n0 = bn * BNJ;
    const int nb = K >> 8;
    const int64_t rbq = (int64_t) 144 * nb;

    float acc[8][JN];
    half2 h[8][JN];
#pragma unroll
    for (int i = 0; i < 8; i++) {
#pragma unroll
        for (int j = 0; j < JN; j++) {
            acc[i][j] = 0.0f;
            h[i][j] = make_half2(0.0f, 0.0f);
        }
    }

    const uint2 * pq[2];   // qs of the row: tile it at pq[i][2 * it] (16 bytes per group, 8 per slot)
    const uint4 * ph[2];   // header plane of the row: super-block sb at ph[i][sb]
    bool va[2];
    int soa[2], nsh[2];
#pragma unroll
    for (int i = 0; i < 2; i++) {
        const int l = t + 256 * i, r = l >> 2, c = l & 3;
        va[i] = m0 + r < M;
        const char * xr = W + (va[i] ? m0 + r : 0) * rbq;
        pq[i] = (const uint2 *) (xr + 8 * (c & 1));
        ph[i] = (const uint4 *) (xr + 128 * nb);
        nsh[i] = 4 * (c >> 1);
        soa[i] = (c * 4) * BM + (r ^ (c << 3));
    }
    const uint4 * pb[NLB];
    bool vb[NLB];
    int sob[NLB];
#pragma unroll
    for (int i = 0; i < NLB; i++) {
        const int l = t + 256 * i, r = l >> 2, c = l & 3;
        vb[i] = l < BNJ * 4 && n0 + r < N;
        pb[i] = (const uint4 *) (X + (int64_t) (vb[i] ? n0 + r : 0) * ldx + c * 8);
        sob[i] = l < BNJ * 4 ? (c * 4) * BNJ + (r ^ ((c << 3) & (BNJ - 1))) : 0;
    }

    uint2 rq[2] = {make_uint2(0, 0), make_uint2(0, 0)};
    uint4 rh[2] = {make_uint4(0, 0, 0, 0), make_uint4(0, 0, 0, 0)};
    int gq = 0;   // group (it & 7) of the loaded tile
    uint4 rb[NLB];
#pragma unroll
    for (int i = 0; i < NLB; i++) rb[i] = make_uint4(0, 0, 0, 0);
    auto gload = [&](const int it) {   // called for it = 0, 1, 2, ... in order: the header is reloaded at each new sb
        gq = it & 7;
#pragma unroll
        for (int i = 0; i < 2; i++) {
            if (va[i]) {
                rq[i] = pq[i][2 * it];
                if (gq == 0) rh[i] = ph[i][it >> 3];
            }
        }
#pragma unroll
        for (int i = 0; i < NLB; i++) if (vb[i]) rb[i] = pb[i][it * 4];
    };
    auto sstore = [&](const int buf) {
#pragma unroll
        for (int i = 0; i < 2; i++) {
            float d1, m1;
            q4k_scale_min(gq, rh[i], d1, m1);
            const uint32_t qw[2] = {rq[i].x >> nsh[i], rq[i].y >> nsh[i]};
            uint32_t o[4];
#pragma unroll
            for (int p = 0; p < 4; p++) {
                const uint32_t w = qw[p >> 1] >> (16 * (p & 1));
                // float(q) for q < 16 built exactly (2^23 + q minus 2^23) instead of an I2F: the same float value,
                // so kq_direct_k's expression gives the same bits
                const float qlo = __int_as_float(0x4B000000 | (w & 0xF)) - 8388608.0f;
                const float qhi = __int_as_float(0x4B000000 | ((w >> 8) & 0xF)) - 8388608.0f;
                const half lo = ggml_cuda_cast<half>(d1 * qlo - m1);          // kq_direct_k's expression
                const half hi = ggml_cuda_cast<half>(d1 * qhi - m1);
                o[p] = (uint32_t) __half_as_ushort(lo) | ((uint32_t) __half_as_ushort(hi) << 16);
            }
            if (!va[i]) o[0] = o[1] = o[2] = o[3] = 0;   // fold_seg_kernel's A regs of a missing row stay 0
            uint32_t * a = &As[buf][0][0] + soa[i];
            a[0] = o[0]; a[BM] = o[1]; a[2 * BM] = o[2]; a[3 * BM] = o[3];
        }
#pragma unroll
        for (int i = 0; i < NLB; i++) {
            if (t + 256 * i >= BNJ * 4) break;
            uint32_t * b = &Bs[buf][0][0] + sob[i];
            b[0] = rb[i].x; b[BNJ] = rb[i].y; b[2 * BNJ] = rb[i].z; b[3 * BNJ] = rb[i].w;
        }
    };
    auto tile = [&](const int buf, const bool restart) {
#pragma unroll
        for (int k2 = 0; k2 < BK2; k2++) {
            const int sw = (k2 >> 2) << 3;
            const int swb = sw & (BNJ - 1);
            const uint4 a0 = *(const uint4 *) &As[buf][k2][(ty * 4) ^ sw];
            const uint4 a1 = *(const uint4 *) &As[buf][k2][(64 + ty * 4) ^ sw];
            const uint32_t a[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
            uint32_t b[JN];
            if constexpr (JN == 8) {
                const uint4 b0 = *(const uint4 *) &Bs[buf][k2][(tx * 4) ^ swb];
                const uint4 b1 = *(const uint4 *) &Bs[buf][k2][(64 + tx * 4) ^ swb];
                b[0] = b0.x; b[1] = b0.y; b[2] = b0.z; b[3] = b0.w; b[4] = b1.x; b[5] = b1.y; b[6] = b1.z; b[7] = b1.w;
            } else if constexpr (JN == 2) {
                const uint2 b0 = *(const uint2 *) &Bs[buf][k2][(tx * 2) ^ swb];
                b[0] = b0.x; b[1] = b0.y;
            } else {
                b[0] = Bs[buf][k2][tx ^ swb];
            }
#pragma unroll
            for (int j = 0; j < JN; j++) {
#pragma unroll
                for (int i = 0; i < 8; i++) {
                    const half2 ai = *(const half2 *) &a[i];
                    const half2 bj = *(const half2 *) &b[j];
                    h[i][j] = k2 == 0 && restart ? __hmul2(bj, ai) : __hfma2(bj, ai, h[i][j]);
                }
            }
        }
    };
    auto fold = [&]() {
#pragma unroll
        for (int i = 0; i < 8; i++) {
#pragma unroll
            for (int j = 0; j < JN; j++) acc[i][j] += fold_h(h[i][j]);
        }
    };

    const int nt = K / (2 * BK2);
    gload(0);
    sstore(0);
    __syncthreads();
    for (int it = 0; it < nt; it += 2) {
        gload(it + 1);
        tile(0, it % TPF == 0);
        sstore(1);
        __syncthreads();
        if (it + 2 < nt) gload(it + 2);
        tile(1, false);
        if ((it + 2) % TPF == 0 || it + 2 >= nt) fold();
        if (it + 2 < nt) sstore(0);
        __syncthreads();
    }

#pragma unroll
    for (int j = 0; j < JN; j++) {
        const int n = n0 + (JN == 8 ? (j < 4 ? tx * 4 + j : 64 + tx * 4 + j - 4) : tx * JN + j);
        if (n >= N) continue;
        const float s = cs[n];
#pragma unroll
        for (int ih = 0; ih < 2; ih++) {
            const int m = m0 + ih * 64 + ty * 4;
            if (m >= M) continue;
            float4 v = make_float4(acc[ih * 4 + 0][j] * s, acc[ih * 4 + 1][j] * s, acc[ih * 4 + 2][j] * s,
                                   acc[ih * 4 + 3][j] * s);
            if constexpr (OUT16) {
                float * pv = &v.x;
                bool big = false;
#pragma unroll
                for (int q = 0; q < 4; q++) {
                    if (fabsf(pv[q]) >= 65520.0f && isfinite(pv[q])) { big = true; pv[q] = copysignf(65504.0f, pv[q]); }
                }
                if (big && sat) atomicAdd(sat, 1);
                half2 hh[2] = {__floats2half2_rn(v.x, v.y), __floats2half2_rn(v.z, v.w)};
                *(uint2 *) (Y16 + n * ldy + m) = *(const uint2 *) hh;
            } else {
                *(float4 *) (Y + n * ldy + m) = v;
            }
        }
    }
}

} // namespace

size_t gemm_scratch_min_bytes(int64_t n_in, int T_max) {
    return align256((size_t) T_max * 4) + align256((size_t) T_max * n_in * 2) + align256((size_t) BM * n_in * 2);
}

size_t gemm_scratch_bytes(int64_t n_in, int64_t n_out, int T_max) {
    return align256((size_t) T_max * 4) + align256((size_t) T_max * n_in * 2) + align256((size_t) n_out * n_in * 2);
}

void gemm_q_mode(gemm_mode mode, uint32_t type, const void * w_raw, int64_t n_in, int64_t n_out, const float * x,
                 float * y, int T, void * scratch, size_t scratch_bytes, cudaStream_t s, cublasHandle_t cb) {
    if (T < 1) return;
    if (n_in % 64 != 0 || n_out % 4 != 0) throw std::runtime_error("fast::gemm_q: needs n_in % 64 == 0, n_out % 4 == 0");
    to_fp16_cuda_t to16 = ggml_get_to_fp16_cuda((ggml_type) type);
    if (!to16) throw std::runtime_error("fast::gemm_q: no f16 dequant for type " + std::to_string(type));
    const int64_t rb = row_bytes_of(type, n_in);

    char * p = (char *) scratch;
    float * cs = (float *) p;  p += align256((size_t) T * 4);
    half * x16 = (half *) p;   p += align256((size_t) T * n_in * 2);
    const bool c16 = mode == gemm_mode::cublas16 || mode == gemm_mode::cublas16a6;
    half * y16 = nullptr;
    if (c16) { y16 = (half *) p; p += align256((size_t) T * n_out * 2); }
    const size_t used = (size_t) (p - (char *) scratch);
    if (used >= scratch_bytes) throw std::runtime_error("fast::gemm_q: scratch too small");
    int64_t rows = (int64_t) ((scratch_bytes - used) / ((size_t) n_in * 2));
    if (rows < n_out) rows = rows / BM * BM;
    rows = std::min(rows, n_out);
    if (rows < 1) throw std::runtime_error("fast::gemm_q: scratch too small for one row tile");
    half * w16 = (half *) p;

    const bool fold = mode == gemm_mode::fold || mode == gemm_mode::fold32;
    if (fold) {
        prescale_kernel<<<T, 256, 0, s>>>(x, x16, cs, (int) n_in);
    } else {
        const int64_t n = (int64_t) T * n_in;
        to_half_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, s>>>(x, x16, n);
    }
    ck(cudaGetLastError(), "x prep");

    const float one = 1.0f, zero = 0.0f;
    const half hone = __float2half(1.0f), hzero = __float2half(0.0f);
    if (!fold) cbk(cublasSetStream(cb, s), "set stream");
    for (int64_t r0 = 0; r0 < n_out; r0 += rows) {
        const int64_t nr = std::min(rows, n_out - r0);
        to16((const char *) w_raw + r0 * rb, w16, nr * n_in, s);
        switch (mode) {
            case gemm_mode::fold:
            case gemm_mode::fold32: {
                if (mode == gemm_mode::fold) launch_fold<FOLD_K2_DEFAULT>(w16, x16, cs, y + r0, nr, T, n_in, n_out, s);
                else launch_fold<16>(w16, x16, cs, y + r0, nr, T, n_in, n_out, s);
                ck(cudaGetLastError(), "fold kernel");
            } break;
            case gemm_mode::cublas32:
                cbk(cublasGemmEx(cb, CUBLAS_OP_T, CUBLAS_OP_N, (int) nr, T, (int) n_in, &one, w16, CUDA_R_16F,
                                 (int) n_in, x16, CUDA_R_16F, (int) n_in, &zero, y + r0, CUDA_R_32F, (int) n_out,
                                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT), "gemm32");
                break;
            case gemm_mode::cublas16:
            case gemm_mode::cublas16a6: {
                const cublasGemmAlgo_t algo = mode == gemm_mode::cublas16a6 ? CUBLAS_GEMM_ALGO6 : CUBLAS_GEMM_DEFAULT;
                cublasStatus_t st = cublasGemmEx(cb, CUBLAS_OP_T, CUBLAS_OP_N, (int) nr, T, (int) n_in, &hone, w16,
                                 CUDA_R_16F, (int) n_in, x16, CUDA_R_16F, (int) n_in, &hzero, y16 + r0, CUDA_R_16F,
                                 (int) n_out, CUBLAS_COMPUTE_16F, algo);
                if (st != CUBLAS_STATUS_SUCCESS && algo != CUBLAS_GEMM_DEFAULT)   // ALGO6 is not offered on every arch
                    st = cublasGemmEx(cb, CUBLAS_OP_T, CUBLAS_OP_N, (int) nr, T, (int) n_in, &hone, w16, CUDA_R_16F,
                                      (int) n_in, x16, CUDA_R_16F, (int) n_in, &hzero, y16 + r0, CUDA_R_16F,
                                      (int) n_out, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT);
                cbk(st, "gemm16");
            } break;
        }
    }
    if (c16) {
        const int64_t n = (int64_t) T * n_out;
        widen_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, s>>>(y16, y, n);
        ck(cudaGetLastError(), "widen");
    }
}

void gemm_q(uint32_t type, const void * w_raw, int64_t n_in, int64_t n_out, const float * x, float * y, int T,
            void * scratch, size_t scratch_bytes, cudaStream_t s, cublasHandle_t cb) {
    gemm_q_mode(gemm_mode::fold, type, w_raw, n_in, n_out, x, y, T, scratch, scratch_bytes, s, cb);
}

void gemm_packed(const packed_weight & w, const float * x, float * y, int T, void * scratch, size_t scratch_bytes,
                 cudaStream_t s, cublasHandle_t) {
    if (T < 1) return;
    const int64_t n_in = w.n_in, n_out = w.n_out;
    if (n_in % 64 != 0 || n_out % 4 != 0)
        throw std::runtime_error("fast::gemm_packed: needs n_in % 64 == 0, n_out % 4 == 0");
    char * p = (char *) scratch;
    float * cs = (float *) p;  p += align256((size_t) T * 4);
    half * x16 = (half *) p;   p += align256((size_t) T * n_in * 2);
    const size_t used = (size_t) (p - (char *) scratch);
    if (used >= scratch_bytes) throw std::runtime_error("fast::gemm_packed: scratch too small");
    int64_t rows = (int64_t) ((scratch_bytes - used) / ((size_t) n_in * 2));
    if (rows < n_out) rows = rows / BM * BM;
    rows = std::min(rows, n_out);
    if (rows < 1) throw std::runtime_error("fast::gemm_packed: scratch too small for one row tile");
    half * w16 = (half *) p;

    prescale_kernel<<<T, 256, 0, s>>>(x, x16, cs, (int) n_in);
    ck(cudaGetLastError(), "x prep");
    for (int64_t r0 = 0; r0 < n_out; r0 += rows) {
        const int64_t nr = std::min(rows, n_out - r0);
        packed_to_f16(w, r0, nr, w16, s);
        launch_fold<FOLD_K2_DEFAULT>(w16, x16, cs, y + r0, nr, T, n_in, n_out, s);
        ck(cudaGetLastError(), "fold kernel");
    }
}

// packed weight x f32 rows through cuBLAS: the packed rows are unpacked to f16 per row tile (packed_to_f16, as
// gemm_packed), x is converted to f16 (no prescale), the GEMM accumulates in f32 (cublasGemmEx 16F x 16F -> 32F).
// Not bit-identical to the fold path (f16 rounding of x instead of the prescale folds; cuBLAS summation order);
// on the P100 the fold kernel runs at ~3 TFLOPS and this at the f16 GEMM rate, so large prefill chunks use it.
void gemm_packed_cublas32(const packed_weight & w, const float * x, float * y, int T, void * scratch, size_t scratch_bytes,
                          cudaStream_t s, cublasHandle_t cb) {
    if (T < 1) return;
    const int64_t n_in = w.n_in, n_out = w.n_out;
    if (n_in % 64 != 0 || n_out % 4 != 0)
        throw std::runtime_error("fast::gemm_packed_cublas32: needs n_in % 64 == 0, n_out % 4 == 0");
    char * p = (char *) scratch;
    p += align256((size_t) T * 4);   // (the prescale slot of the fold layout stays unused: same scratch plan)
    half * x16 = (half *) p;   p += align256((size_t) T * n_in * 2);
    const size_t used = (size_t) (p - (char *) scratch);
    if (used >= scratch_bytes) throw std::runtime_error("fast::gemm_packed_cublas32: scratch too small");
    int64_t rows = (int64_t) ((scratch_bytes - used) / ((size_t) n_in * 2));
    if (rows < n_out) rows = rows / BM * BM;
    rows = std::min(rows, n_out);
    if (rows < 1) throw std::runtime_error("fast::gemm_packed_cublas32: scratch too small for one row tile");
    half * w16 = (half *) p;
    const int64_t n = (int64_t) T * n_in;
    to_half_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, s>>>(x, x16, n);
    ck(cudaGetLastError(), "x to f16");
    const float one = 1.0f, zero = 0.0f;
    cbk(cublasSetStream(cb, s), "set stream");
    for (int64_t r0 = 0; r0 < n_out; r0 += rows) {
        const int64_t nr = std::min(rows, n_out - r0);
        packed_to_f16(w, r0, nr, w16, s);
        cbk(cublasGemmEx(cb, CUBLAS_OP_T, CUBLAS_OP_N, (int) nr, T, (int) n_in, &one, w16, CUDA_R_16F, (int) n_in, x16,
                         CUDA_R_16F, (int) n_in, &zero, y + r0, CUDA_R_32F, (int) n_out, CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT), "gemm32 packed");
    }
}

// ================= prefill pipeline API =================

void prescale_rows(const float * v, int64_t ldv, int T, int K, half * x16, int64_t ldx, float * cs, cudaStream_t s,
                   bool one_read) {
    if (T < 1) return;
    if (K % 4 || ldv % 4 || ldx % 8) throw std::runtime_error("fast::prescale_rows: needs K % 4 == 0, ldv % 4 == 0, ldx % 8 == 0");
    const int K4 = K / 4;
    if      (!one_read || K4 > 256 * 9) prescale_rows_kernel<<<T, 256, 0, s>>>(v, ldv, x16, ldx, cs, K);
    else if (K4 <= 256 * 1) prescale_rows_v_kernel<1><<<T, 256, 0, s>>>(v, ldv, x16, ldx, cs, K);
    else if (K4 <= 256 * 2) prescale_rows_v_kernel<2><<<T, 256, 0, s>>>(v, ldv, x16, ldx, cs, K);
    else if (K4 <= 256 * 3) prescale_rows_v_kernel<3><<<T, 256, 0, s>>>(v, ldv, x16, ldx, cs, K);
    else if (K4 <= 256 * 4) prescale_rows_v_kernel<4><<<T, 256, 0, s>>>(v, ldv, x16, ldx, cs, K);
    else if (K4 <= 256 * 6) prescale_rows_v_kernel<6><<<T, 256, 0, s>>>(v, ldv, x16, ldx, cs, K);
    else                    prescale_rows_v_kernel<9><<<T, 256, 0, s>>>(v, ldv, x16, ldx, cs, K);
    ck(cudaGetLastError(), "prescale_rows");
}

template <typename PT>
static void add_norm_prescale_impl(float * x, const PT * p0, const PT * p1, const float * w, float eps, int T, int K,
                                   half * x16, int64_t ldx, float * cs, cudaStream_t s) {
    if (T < 1) return;
    if (K % 4) throw std::runtime_error("fast::add_norm_prescale: needs K % 4 == 0");
    const int n4 = K / 4;
    if (!x16) {
        if (!p0) return;
        const int64_t n = (int64_t) T * n4;
        const unsigned nb = (unsigned) std::min<int64_t>(4096, (n + 255) / 256);
        add2_kernel<PT><<<nb, 256, 0, s>>>((float4 *) x, p0, p1, n);
        ck(cudaGetLastError(), "add2");
        return;
    }
    if (ldx % 8) throw std::runtime_error("fast::add_norm_prescale: ldx % 8 != 0");
    // NV and thread count exactly as fast::rmsnorm (fused.cu dispatch / launch_vec), so the reduction tree matches
    auto go = [&](auto nv_tag) {
        constexpr int NV = decltype(nv_tag)::value;
        int thr = (n4 + NV - 1) / NV;
        thr = (thr + 31) & ~31;
        add_norm_prescale_kernel<NV, PT><<<T, thr, 0, s>>>(x, p0, p1, w, K, eps, x16, ldx, cs);
    };
    if      (n4 <= 256)  go(std::integral_constant<int, 1>());
    else if (n4 <= 512)  go(std::integral_constant<int, 2>());
    else if (n4 <= 1024) go(std::integral_constant<int, 4>());
    else if (n4 <= 1280) go(std::integral_constant<int, 5>());
    else if (n4 <= 2048) go(std::integral_constant<int, 8>());
    else throw std::runtime_error("fast::add_norm_prescale: K > 8192");
    ck(cudaGetLastError(), "add_norm_prescale");
}

void add_norm_prescale(float * x, const float * p0, const float * p1, const float * w, float eps, int T, int K,
                       half * x16, int64_t ldx, float * cs, cudaStream_t s) {
    add_norm_prescale_impl<float>(x, p0, p1, w, eps, T, K, x16, ldx, cs, s);
}

void add_norm_prescale16(float * x, const half * p0, const half * p1, const float * w, float eps, int T, int K,
                         half * x16, int64_t ldx, float * cs, cudaStream_t s) {
    add_norm_prescale_impl<half>(x, p0, p1, w, eps, T, K, x16, ldx, cs, s);
}

void glu_prescale(const float * g, const float * u, int T, int K, half * x16, int64_t ldx, float * cs, cudaStream_t s) {
    if (T < 1) return;
    if (K % 4 || ldx % 8) throw std::runtime_error("fast::glu_prescale: needs K % 4 == 0, ldx % 8 == 0");
    const int K4 = K / 4;
    if      (K4 <= 256 * 4) glu_prescale_kernel<4><<<T, 256, 0, s>>>(g, u, K, x16, ldx, cs);
    else if (K4 <= 256 * 6) glu_prescale_kernel<6><<<T, 256, 0, s>>>(g, u, K, x16, ldx, cs);
    else if (K4 <= 256 * 9) glu_prescale_kernel<9><<<T, 256, 0, s>>>(g, u, K, x16, ldx, cs);
    else throw std::runtime_error("fast::glu_prescale: K > 9216");
    ck(cudaGetLastError(), "glu_prescale");
}

// max |v| as the bits of a non-negative float (unsigned order = float order; NaN above inf): one atomicMax per warp
template <typename VT>
static __global__ void __launch_bounds__(256) abs_max_kernel(const VT * __restrict__ v, size_t n, unsigned * out) {
    unsigned m = 0;
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        const unsigned b = __float_as_uint((float) v[i]) & 0x7fffffffu;
        m = b > m ? b : m;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const unsigned u = __shfl_xor_sync(0xffffffff, m, o);
        m = u > m ? u : m;
    }
    if ((threadIdx.x & 31) == 0 && m) atomicMax(out, m);
}

void abs_max(const void * v, bool f16, size_t n, float * out, cudaStream_t s) {
    if (n == 0) return;
    const unsigned nb = (unsigned) std::min<size_t>(1024, (n + 1023) / 1024);
    if (f16) abs_max_kernel<half><<<nb, 256, 0, s>>>((const half *) v, n, (unsigned *) out);
    else abs_max_kernel<float><<<nb, 256, 0, s>>>((const float *) v, n, (unsigned *) out);
    ck(cudaGetLastError(), "abs_max");
}

static constexpr bool same_sizes(const unsigned * a, const unsigned * b, int n) {
    for (int i = 0; i < n; ++i)
        if (a[i] != b[i]) return false;
    return true;
}

// the bank-fixed cubin's kernel for the current device: loaded once per device, only on cc 6.0 (it is an sm_60
// image); nullptr elsewhere or if loading fails (then the compiled kernel runs)
static CUfunction sass_fold8(int variant) {
    static std::mutex mu;
    static int state[64][3] = {};   // 0 not tried, 1 loaded, -1 unavailable
    static CUfunction fn[64][3] = {};
    int dev = 0;
    if (variant < 0 || variant > 2 || cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) return nullptr;
    std::lock_guard<std::mutex> lk(mu);
    int & st = state[dev][variant];
    if (st == 0) {
        st = -1;
        cudaDeviceProp p;
        if (cudaGetDeviceProperties(&p, dev) == cudaSuccess && p.major == 6 && p.minor == 0) {
            cudaFree(nullptr);   // the device's primary context is current on this thread
            CUmodule m;
            if (cuModuleLoadData(&m, gemm_sass_fold8[variant]) == CUDA_SUCCESS &&
                cuModuleGetFunction(&fn[dev][variant], m, gemm_sass_fold8_name) == CUDA_SUCCESS)
                st = 1;
            else
                fprintf(stderr, "fast::fold_gemm: bank-fixed SASS unavailable on device %d, using the compiled kernel\n", dev);
        }
    }
    return st > 0 ? fn[dev][variant] : nullptr;
}

void fold_gemm(const fold_seg * segs, int nseg, const half * x16, int64_t ldx, const float * cs, int T, int K,
               cudaStream_t s, const fold_opts & o) {
    if (T < 1 || nseg < 1) return;
    if (nseg > 4) throw std::runtime_error("fast::fold_gemm: at most 4 segments");
    if (K % 64 || ldx % 8) throw std::runtime_error("fast::fold_gemm: needs K % 64 == 0, ldx % 8 == 0");
    seg_tab tab{};
    int tiles = 0;
    for (int i = 0; i < 4; i++) {
        const fold_seg & g = segs[i < nseg ? i : nseg - 1];
        if (i < nseg && (g.rows % 4 || g.ldy % 4 || g.rows < 1))
            throw std::runtime_error("fast::fold_gemm: needs rows % 4 == 0, ldy % 4 == 0");
        if (i < nseg && (o.out16 ? !g.y16 : !g.y)) throw std::runtime_error("fast::fold_gemm: missing output pointer");
        tab.w[i] = g.w; tab.y[i] = g.y; tab.y16[i] = g.y16; tab.ldy[i] = g.ldy; tab.rows[i] = g.rows;
        tab.tile0[i] = tiles;
        if (i < nseg) tiles += (g.rows + BM - 1) / BM;
    }
    tab.nseg = nseg;
    const int ra = o.raster;
    if (o.q4k) {   // fused dequant: segs[i].w are q4kr rows (fold_seg_q4k_kernel); same bits as unpack + fold
        if (K % 256) throw std::runtime_error("fast::fold_gemm: q4k needs K % 256 == 0");
        auto goq = [&](auto out16) {
            constexpr bool O16 = decltype(out16)::value;
            if (T <= 16)
                fold_seg_q4k_kernel<1, O16><<<dim3(tiles, (T + 15) / 16), 256, 0, s>>>(tab, x16, ldx, cs, T, K, ra, o.sat);
            else if (T <= 64)
                fold_seg_q4k_kernel<2, O16><<<dim3(tiles, (T + 31) / 32), 256, 0, s>>>(tab, x16, ldx, cs, T, K, ra, o.sat);
            else
                fold_seg_q4k_kernel<8, O16><<<dim3(tiles, (T + 127) / 128), 256, 0, s>>>(tab, x16, ldx, cs, T, K, ra, o.sat);
        };
        if (o.out16) goq(std::true_type()); else goq(std::false_type());
        ck(cudaGetLastError(), "fold_gemm q4k");
        return;
    }
    if (o.sass > 0 && !o.out16 && T > 64) {   // the bank-fixed SASS of fold_seg_kernel<8, false> (same bits)
        if (CUfunction f = sass_fold8(o.sass - 1)) {
            const half * X = x16;
            int64_t ldx_ = ldx;
            int N = T, K_ = K, raster = ra;
            int * sat = o.sat;
            void * args[] = {&tab, &X, &ldx_, (void *) &cs, &N, &K_, &raster, &sat};
            // the cubin's parameter list (sizes from its EIATTR_KPARAM_INFO, gemm_sass_fold8.h) must be this one
            constexpr unsigned asz[] = {sizeof tab, sizeof X, sizeof ldx_, sizeof cs, sizeof N, sizeof K_, sizeof raster,
                                        sizeof sat};
            static_assert(sizeof asz / sizeof asz[0] == gemm_sass_fold8_nparam &&
                              same_sizes(asz, gemm_sass_fold8_param_size, gemm_sass_fold8_nparam),
                          "fold_gemm's SASS launch arguments differ from the cubin's parameters: rerun fold_bankfix.py");
            const CUresult r = cuLaunchKernel(f, (unsigned) tiles, (unsigned) ((T + 127) / 128), 1, 256, 1, 1, 0,
                                              (CUstream) s, args, nullptr);
            if (r != CUDA_SUCCESS) throw std::runtime_error("fast::fold_gemm: cuLaunchKernel failed for the SASS kernel");
            return;
        }
    }
    auto go = [&](auto out16) {
        constexpr bool O16 = decltype(out16)::value;
        if (T <= 16)
            fold_seg_kernel<1, O16><<<dim3(tiles, (T + 15) / 16), 256, 0, s>>>(tab, x16, ldx, cs, T, K, ra, o.sat);
        else if (T <= 64)
            fold_seg_kernel<2, O16><<<dim3(tiles, (T + 31) / 32), 256, 0, s>>>(tab, x16, ldx, cs, T, K, ra, o.sat);
        else
            fold_seg_kernel<8, O16><<<dim3(tiles, (T + 127) / 128), 256, 0, s>>>(tab, x16, ldx, cs, T, K, ra, o.sat);
    };
    if (o.out16) go(std::true_type()); else go(std::false_type());
    ck(cudaGetLastError(), "fold_gemm");
}

} // namespace fast
} // namespace p100e
