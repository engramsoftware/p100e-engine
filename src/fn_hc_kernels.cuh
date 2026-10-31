// Flash-Next hyper-connection mixer matvecs for decode / MTP verify (T <= 4):
//  - hc down: Q8_0 repacked (fast LAYOUT_Q8R: row = [qs 32*nb bytes][d nb halves]), n_in 10240 -> n_out 320. The fast
//    path runs one warp per row: 320 warps for a 3.3 MB weight, latency bound (74 us on a P100). Split-K: one block of
//    KS warps per row, each warp a strided share of the row's 16-value chunks, partials summed in a fixed order.
//  - hc up: f16 [n_out 10240][n_in 320] (narrow rows dequantized by fn_model::upload_raw). One warp per row read 640
//    bytes with little in flight (37 us). Here each warp takes RPW rows at once (uint4 loads of all of them before the
//    math) with x staged in shared memory.
// Both sum in a different order than the fast kernels, so their bits differ (P100E_FN_HC_FAST=1 switches them on).
#pragma once
#include <cuda_fp16.h>
#include <cstdint>

namespace p100e::fnhc {

template <int T, int KS>
__global__ void __launch_bounds__(32 * KS) q8r_splitk_k(const char * __restrict__ w, const float * __restrict__ x, float * __restrict__ y,
                                                       int nb, int n_in, int n_out, size_t row_bytes) {
    const int row = blockIdx.x, lane = threadIdx.x, wp = threadIdx.y;
    const char * r = w + (size_t) row * row_bytes;
    float acc[T];
#pragma unroll
    for (int j = 0; j < T; ++j) acc[j] = 0.0f;
    for (int t = wp * 32 + lane; t < 2 * nb; t += 32 * KS) {   // t = 16-value chunk: block t / 2, half t % 2
        const int b = t >> 1, h = t & 1;
        const uint4 q = *((const uint4 *) (r + 32 * b + 16 * h));
        const float d = __half2float(*((const half *) (r + 32 * nb + 2 * b)));
        const uint32_t qw[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
        for (int j = 0; j < T; ++j) {
            const float4 * xp = (const float4 *) (x + (size_t) j * n_in + 32 * b + 16 * h);
            float s = 0.0f;
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                const float4 xv = xp[c];
                s = fmaf((float) (int8_t) (qw[c]), xv.x, s);
                s = fmaf((float) (int8_t) (qw[c] >> 8), xv.y, s);
                s = fmaf((float) (int8_t) (qw[c] >> 16), xv.z, s);
                s = fmaf((float) (int8_t) (qw[c] >> 24), xv.w, s);
            }
            acc[j] = fmaf(d, s, acc[j]);
        }
    }
    __shared__ float part[KS][T];
#pragma unroll
    for (int j = 0; j < T; ++j) {
        float v = acc[j];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (lane == 0) part[wp][j] = v;
    }
    __syncthreads();
    if (wp == 0 && lane < T) {
        float v = 0.0f;
#pragma unroll
        for (int k = 0; k < KS; ++k) v += part[k][lane];
        y[(size_t) lane * n_out + row] = v;
    }
}

// f16 rows of n_in (<= 512, % 8 == 0) values; RPW rows per warp, 8 warps per block
template <int T, int RPW>
__global__ void __launch_bounds__(256) f16_narrow_k(const half * __restrict__ w, const float * __restrict__ x, float * __restrict__ y, int n_in, int n_out) {
    extern __shared__ float xs[];   // [T][n_in]
    for (int i = threadIdx.x + 32 * threadIdx.y; i < T * n_in; i += 256) xs[i] = x[i];
    __syncthreads();
    const int lane = threadIdx.x;
    const int r0 = (blockIdx.x * 8 + threadIdx.y) * RPW;
    const int nv = n_in / 8;   // uint4 (8 halves) per row
    float acc[RPW][T];
#pragma unroll
    for (int k = 0; k < RPW; ++k)
#pragma unroll
        for (int j = 0; j < T; ++j) acc[k][j] = 0.0f;
    for (int v0 = 0; v0 < nv; v0 += 32) {
        const int v = v0 + lane;
        uint4 q[RPW];
#pragma unroll
        for (int k = 0; k < RPW; ++k) q[k] = (v < nv && r0 + k < n_out) ? ((const uint4 *) (w + (size_t) (r0 + k) * n_in))[v] : make_uint4(0, 0, 0, 0);
        if (v >= nv) continue;
#pragma unroll
        for (int k = 0; k < RPW; ++k) {
            const half2 * h = (const half2 *) &q[k];
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                const float2 f = __half22float2(h[c]);
#pragma unroll
                for (int j = 0; j < T; ++j) {
                    const float * xr = xs + j * n_in + 8 * v + 2 * c;
                    acc[k][j] = fmaf(f.x, xr[0], acc[k][j]);
                    acc[k][j] = fmaf(f.y, xr[1], acc[k][j]);
                }
            }
        }
    }
#pragma unroll
    for (int k = 0; k < RPW; ++k)
#pragma unroll
        for (int j = 0; j < T; ++j) {
            float s = acc[k][j];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
            if (lane == 0 && r0 + k < n_out) y[(size_t) j * n_out + r0 + k] = s;
        }
}

// v2 (the one used): blocks of RB rows x one of KB slices of the row; the slice's activations go to shared memory once
// and serve the RB rows (the per-row kernels re-read all T x n_in activations per row: at T = 4 that is 13x the weight
// bytes); partial sums [KB][T][n_out] then one fixed-order reduce. Deterministic (no atomics).
template <int T, int RB, int KB>
__global__ void __launch_bounds__(32 * RB) q8r_tile_k(const char * __restrict__ w, const float * __restrict__ x, float * __restrict__ part,
                                                    int nb, int n_in, int n_out, size_t row_bytes) {
    extern __shared__ float xs[];   // [T][slice]
    const int kb = blockIdx.y, nbs = nb / KB, c0 = kb * nbs * 32, slice = nbs * 32;   // this block's 32-value blocks
    for (int i = threadIdx.x + 32 * threadIdx.y; i < T * slice; i += 32 * RB) { const int j = i / slice, k = i % slice; xs[i] = x[(size_t) j * n_in + c0 + k]; }
    __syncthreads();
    const int row = blockIdx.x * RB + threadIdx.y, lane = threadIdx.x;
    if (row >= n_out) return;
    const char * r = w + (size_t) row * row_bytes;
    float acc[T];
#pragma unroll
    for (int j = 0; j < T; ++j) acc[j] = 0.0f;
    // lane-interleaved: each pass a warp covers 128 consecutive values (4 per lane: one char4 of weights, one float4
    // of activations per column), so the shared-memory reads are conflict-free and the weight reads coalesced
    for (int k0 = 4 * lane; k0 < slice; k0 += 128) {
        const int b = kb * nbs + (k0 >> 5);
        const char4 q = *((const char4 *) (r + (size_t) kb * nbs * 32 + k0));
        const float d = __half2float(*((const half *) (r + 32 * nb + 2 * b)));
#pragma unroll
        for (int j = 0; j < T; ++j) {
            const float4 xv = *((const float4 *) (xs + j * slice + k0));
            float sm = (float) q.x * xv.x;
            sm = fmaf((float) q.y, xv.y, sm);
            sm = fmaf((float) q.z, xv.z, sm);
            sm = fmaf((float) q.w, xv.w, sm);
            acc[j] = fmaf(d, sm, acc[j]);
        }
    }
#pragma unroll
    for (int j = 0; j < T; ++j) {
        float v = acc[j];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (lane == 0) part[((size_t) kb * T + j) * n_out + row] = v;
    }
}
template <int KB>
__global__ void reduce_parts_k(const float * __restrict__ part, float * __restrict__ y, int n) {   // n = T * n_out
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = 0.0f;
#pragma unroll
    for (int k = 0; k < KB; ++k) v += part[(size_t) k * n + i];
    y[i] = v;
}
constexpr int TILE_RB = 8, TILE_KB = 8;
inline size_t q8r_tile_scratch_bytes(int64_t n_out) { return (size_t) TILE_KB * 4 * n_out * 4; }
// part: scratch of q8r_tile_scratch_bytes(n_out)
inline bool q8r_tile(const void * w, const float * x, float * y, float * part, int T, int64_t n_in, int64_t n_out, cudaStream_t s) {
    const int nb = (int) (n_in / 32);
    if (T < 1 || T > 4 || n_in % 32 != 0 || nb % TILE_KB != 0) return false;
    const size_t rb = (size_t) nb * 34, sh = (size_t) T * (nb / TILE_KB) * 32 * 4;
    const dim3 grid((unsigned) ((n_out + TILE_RB - 1) / TILE_RB), TILE_KB), blk(32, TILE_RB);
    switch (T) {
        case 1: q8r_tile_k<1, TILE_RB, TILE_KB><<<grid, blk, sh, s>>>((const char *) w, x, part, nb, (int) n_in, (int) n_out, rb); break;
        case 2: q8r_tile_k<2, TILE_RB, TILE_KB><<<grid, blk, sh, s>>>((const char *) w, x, part, nb, (int) n_in, (int) n_out, rb); break;
        case 3: q8r_tile_k<3, TILE_RB, TILE_KB><<<grid, blk, sh, s>>>((const char *) w, x, part, nb, (int) n_in, (int) n_out, rb); break;
        default: q8r_tile_k<4, TILE_RB, TILE_KB><<<grid, blk, sh, s>>>((const char *) w, x, part, nb, (int) n_in, (int) n_out, rb); break;
    }
    const int n = T * (int) n_out;
    reduce_parts_k<TILE_KB><<<(n + 255) / 256, 256, 0, s>>>(part, y, n);
    return true;
}

constexpr int KS_DEFAULT = 8, RPW_DEFAULT = 4;

// launchers: false = shape / T not handled (the caller runs fast::matvec)
inline bool q8r_splitk(const void * w, const float * x, float * y, int T, int64_t n_in, int64_t n_out, cudaStream_t s) {
    if (T < 1 || T > 4 || n_in % 32 != 0) return false;
    const int nb = (int) (n_in / 32);
    const size_t rb = (size_t) nb * 34;
    const dim3 grid((unsigned) n_out), blk(32, KS_DEFAULT);
    switch (T) {
        case 1: q8r_splitk_k<1, KS_DEFAULT><<<grid, blk, 0, s>>>((const char *) w, x, y, nb, (int) n_in, (int) n_out, rb); break;
        case 2: q8r_splitk_k<2, KS_DEFAULT><<<grid, blk, 0, s>>>((const char *) w, x, y, nb, (int) n_in, (int) n_out, rb); break;
        case 3: q8r_splitk_k<3, KS_DEFAULT><<<grid, blk, 0, s>>>((const char *) w, x, y, nb, (int) n_in, (int) n_out, rb); break;
        default: q8r_splitk_k<4, KS_DEFAULT><<<grid, blk, 0, s>>>((const char *) w, x, y, nb, (int) n_in, (int) n_out, rb); break;
    }
    return true;
}
inline bool f16_narrow(const void * w, const float * x, float * y, int T, int64_t n_in, int64_t n_out, cudaStream_t s) {
    if (T < 1 || T > 4 || n_in % 8 != 0 || n_in > 512) return false;
    const unsigned rows_per_block = 8 * RPW_DEFAULT;
    const dim3 grid((unsigned) ((n_out + rows_per_block - 1) / rows_per_block)), blk(32, 8);
    const size_t sh = (size_t) T * n_in * 4;
    switch (T) {
        case 1: f16_narrow_k<1, RPW_DEFAULT><<<grid, blk, sh, s>>>((const half *) w, x, y, (int) n_in, (int) n_out); break;
        case 2: f16_narrow_k<2, RPW_DEFAULT><<<grid, blk, sh, s>>>((const half *) w, x, y, (int) n_in, (int) n_out); break;
        case 3: f16_narrow_k<3, RPW_DEFAULT><<<grid, blk, sh, s>>>((const half *) w, x, y, (int) n_in, (int) n_out); break;
        default: f16_narrow_k<4, RPW_DEFAULT><<<grid, blk, sh, s>>>((const half *) w, x, y, (int) n_in, (int) n_out); break;
    }
    return true;
}

} // namespace p100e::fnhc
