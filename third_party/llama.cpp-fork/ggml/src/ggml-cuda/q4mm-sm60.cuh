#pragma once

// Fused Q4_0 x F32 -> F32 matrix multiply for wide products (prefill) on sm_60 (GP100).
//
// sm_60 has no dp4a and no tensor cores, so stock llama.cpp sends dense prefill to cuBLAS: every ubatch writes an
// f16 copy of each whole weight matrix to VRAM, converts the activations to f16, runs an f16 GEMM and converts the
// result back.  This kernel unpacks Q4_0 straight into shared memory instead, so the f16 weight copy (and its
// pool buffer) never exists, and multiplies with half2 FMAs (P100: 2x the f32 rate).
//
// Math: the same f16 weight values the dequant path produces (d * (q - 8) rounded to half), the same f16
// activations (f32 -> half), half2 products accumulated in half over one 32-wide K block, then added into f32
// accumulators.  cuBLAS COMPUTE_16F accumulates the whole K in half, so this is at least as accurate.
//
// Both Q4_0 layouts are read: standard 18-byte blocks, and the q4r repack (q4r-sm60.cu: per row, qs of all
// blocks then d of all blocks, same row stride).
//
// GGML_P100_Q4MM=1 enables it on sm_60 devices (default off).  GGML_P100_Q4MM=2 enables it on any NVIDIA device,
// for correctness tests on other cards only.

#include "common.cuh"

#define Q4MM_BM      128   // weight rows per block
#define Q4MM_BN      64    // tokens per block
#define Q4MM_BK      32    // one Q4_0 block
#define Q4MM_TM      8     // weight rows per thread
#define Q4MM_TN      4     // tokens per thread
#define Q4MM_THREADS 256   // (BM/TM) * (BN/TN)

// out[n*ne0 + m] = sum_k W[m, k] * x[n*s11 + k]
//   w        : Q4_0 weights, row stride nb01 bytes, K = ne00 (multiple of 32), M = ne01 rows
//   repacked : false = standard block_q4_0, true = q4r layout
template <bool repacked>
static __global__ void __launch_bounds__(Q4MM_THREADS, 2) q4mm_sm60_kernel(
        const char * __restrict__ w, const float * __restrict__ x, float * __restrict__ dst,
        const int ne00, const int ne01, const int ne11, const int64_t nb01, const int64_t s11, const int64_t ne0) {
#if __CUDA_ARCH__ >= GGML_CUDA_CC_PASCAL
    __shared__ __align__(16) half2 ws[Q4MM_BK][Q4MM_BM];                // weight value duplicated (v, v): 16 KiB
    __shared__ __align__(16) half xs[Q4MM_BK][Q4MM_BN];    // activations, k-major: 4 KiB

    const int tid = threadIdx.x;
    const int tm  = tid % (Q4MM_BM/Q4MM_TM);   // 0..15
    const int tn  = tid / (Q4MM_BM/Q4MM_TM);   // 0..15
    const int m0  = blockIdx.x*Q4MM_BM;
    const int n0  = blockIdx.y*Q4MM_BN;
    const int nb  = ne00/Q4MM_BK;

    // weight loader: thread -> (row, half of the block)
    const int  lw_row  = tid/2;
    const int  lw_half = tid%2;
    const bool lw_ok   = m0 + lw_row < ne01;
    const char * wrow  = w + (int64_t) (lw_ok ? m0 + lw_row : 0)*nb01;

    // activation loader: thread -> (token, 8 consecutive k)
    const int  lx_n  = tid/4;
    const int  lx_k  = (tid%4)*8;
    const bool lx_ok = n0 + lx_n < ne11;
    const float * xrow = x + (int64_t) (lx_ok ? n0 + lx_n : 0)*s11;

    float acc[Q4MM_TM][Q4MM_TN];
#pragma unroll
    for (int i = 0; i < Q4MM_TM; ++i) {
#pragma unroll
        for (int j = 0; j < Q4MM_TN; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    for (int kb = 0; kb < nb; ++kb) {
        // ---- unpack 128 rows x 32 values of Q4_0 into ws
        {
            uint32_t q[2];
            half d;
            if (repacked) {
                const uint2 v = *(const uint2 *) (wrow + (int64_t) kb*16 + lw_half*8);
                q[0] = v.x;
                q[1] = v.y;
                d = ((const half *) (wrow + (int64_t) nb*16))[kb];
            } else {
                const char * blk = wrow + (int64_t) kb*18;   // block_q4_0: half d, uint8 qs[16]; 2-byte aligned
                const uint16_t * qs16 = (const uint16_t *) (blk + 2 + lw_half*8);
                q[0] = qs16[0] | ((uint32_t) qs16[1] << 16);
                q[1] = qs16[2] | ((uint32_t) qs16[3] << 16);
                d = *(const half *) blk;
            }
            if (!lw_ok) {
                d = __float2half(0.0f);
            }
#pragma unroll
            for (int b = 0; b < 8; ++b) {
                const int byte = (q[b/4] >> (8*(b%4))) & 0xFF;
                const half lo = __hmul(d, __int2half_rn((byte & 0x0F) - 8));
                const half hi = __hmul(d, __int2half_rn((byte >>   4) - 8));
                ws[lw_half*8 + b     ][lw_row] = __half2half2(lo);
                ws[lw_half*8 + b + 16][lw_row] = __half2half2(hi);
            }
        }
        // ---- 64 tokens x 32 k of activations into xs (f32 -> half)
        {
            float4 a = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            float4 b = a;
            if (lx_ok) {
                const float4 * src = (const float4 *) (xrow + (int64_t) kb*Q4MM_BK + lx_k);
                a = src[0];
                b = src[1];
            }
            xs[lx_k + 0][lx_n] = __float2half(a.x);
            xs[lx_k + 1][lx_n] = __float2half(a.y);
            xs[lx_k + 2][lx_n] = __float2half(a.z);
            xs[lx_k + 3][lx_n] = __float2half(a.w);
            xs[lx_k + 4][lx_n] = __float2half(b.x);
            xs[lx_k + 5][lx_n] = __float2half(b.y);
            xs[lx_k + 6][lx_n] = __float2half(b.z);
            xs[lx_k + 7][lx_n] = __float2half(b.w);
        }
        __syncthreads();

        half2 hacc[Q4MM_TM][Q4MM_TN/2];
#pragma unroll
        for (int i = 0; i < Q4MM_TM; ++i) {
#pragma unroll
            for (int j = 0; j < Q4MM_TN/2; ++j) {
                hacc[i][j] = __float2half2_rn(0.0f);
            }
        }

#pragma unroll 8
        for (int k = 0; k < Q4MM_BK; ++k) {
            half2 wv[Q4MM_TM];
            const float4 * wp = (const float4 *) &ws[k][tm*Q4MM_TM];
            *(float4 *) &wv[0] = wp[0];
            *(float4 *) &wv[4] = wp[1];
            const float2 xv = *(const float2 *) &xs[k][tn*Q4MM_TN];
            const half2 x01 = *(const half2 *) &xv.x;
            const half2 x23 = *(const half2 *) &xv.y;
#pragma unroll
            for (int i = 0; i < Q4MM_TM; ++i) {
                hacc[i][0] = __hfma2(wv[i], x01, hacc[i][0]);
                hacc[i][1] = __hfma2(wv[i], x23, hacc[i][1]);
            }
        }

#pragma unroll
        for (int i = 0; i < Q4MM_TM; ++i) {
            const float2 f0 = __half22float2(hacc[i][0]);
            const float2 f1 = __half22float2(hacc[i][1]);
            acc[i][0] += f0.x;
            acc[i][1] += f0.y;
            acc[i][2] += f1.x;
            acc[i][3] += f1.y;
        }
        __syncthreads();
    }

#pragma unroll
    for (int j = 0; j < Q4MM_TN; ++j) {
        const int n = n0 + tn*Q4MM_TN + j;
        if (n >= ne11) {
            continue;
        }
#pragma unroll
        for (int i = 0; i < Q4MM_TM; ++i) {
            const int m = m0 + tm*Q4MM_TM + i;
            if (m < ne01) {
                dst[(int64_t) n*ne0 + m] = acc[i][j];
            }
        }
    }
#else
    GGML_UNUSED_VARS(w, x, dst, ne00, ne01, ne11, nb01, s11, ne0);
    NO_DEVICE_CODE;
#endif // __CUDA_ARCH__ >= GGML_CUDA_CC_PASCAL
}

// Device-side entry for tests and the host wrapper.
template <bool repacked>
static void q4mm_sm60_launch(const char * w, const float * x, float * dst, int ne00, int ne01, int ne11,
                             int64_t nb01, int64_t s11, int64_t ne0, cudaStream_t stream) {
    const dim3 grid((ne01 + Q4MM_BM - 1)/Q4MM_BM, (ne11 + Q4MM_BN - 1)/Q4MM_BN, 1);
    q4mm_sm60_kernel<repacked><<<grid, Q4MM_THREADS, 0, stream>>>(w, x, dst, ne00, ne01, ne11, nb01, s11, ne0);
}

// Host API (q4mm-sm60.cu).  Returns true if it handled dst = src0 x src1 (src0 Q4_0, standard or q4r layout).
bool ggml_cuda_q4mm_enabled(int cc);
bool ggml_cuda_q4mm_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                            ggml_tensor * dst, bool repacked);
