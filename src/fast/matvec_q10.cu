// Q1_0 a16 HFMA2 GEMV (T = 1..4 columns): the cold-expert ffn_down of the Flash-Next split files
// Q1_0 (ggml type 41): 128 weights per 18-byte block
// [d f16][16 bytes of sign bits], weight j = bit j set ? +d : -d (dequantize_row_q1_0). So the dot product of a 32-weight
// a16 group is  d * sum_k s_k a_k  with s_k = +-1 read straight from the 4 sign bytes of the group.
//
// Memory-bound by construction: 4.5 bytes of weights per 32 outputs-worth of MACs, so the hot loop is the IQ1_S structure
// (matvec_iq1.cu) with the 2048-entry grid replaced by a 256-entry sign table: one LDS.128 per sign byte gives the byte's
// 8 weights as the four +-1 half2 pairs the a16 slots want (slot 4k = (a[4k], a[4k+2]), 4k+1 = (a[4k+16], a[4k+18]),
// 4k+2 = (a[4k+1], a[4k+3]), 4k+3 = (a[4k+17], a[4k+19])). Byte b of the group's 4 sign bytes covers elements 8b..8b+7:
//     b = 0 -> slots 0,2,4,6    b = 1 -> slots 8,10,12,14    b = 2 -> slots 1,3,5,7    b = 3 -> slots 9,11,13,15
// Per (group, row, column): 4 LDS.128 + 16 HFMA2 (f16 pairs, explicit rounding to f32 per group) + 1 FFMA; the weights
// are read with the 2-byte loads the 18-byte block alignment allows (4 LDG.U16 per group-row: 1 d + 3 sign words... the
// group's 4 sign bytes straddle u16 pairs). Rows stay the untouched GGUF bytes (no repack): the same kernel reads a
// GPU-resident row set or a mapped pinned host row set over PCIe (the cold experts).
// A column's sum does not depend on the number of columns in the launch (T invariance), as for IQ1_S.
#include "matvec_internal.cuh"
#include <algorithm>
#include <cstdlib>

namespace p100e::fast {

#define Q10_WARPS 8

static __device__ __forceinline__ uint32_t q10_ld_u16(const char * p) { return *((const uint16_t *) p); }

template <int NC, int R>
static __global__ void __launch_bounds__(32*Q10_WARPS)
q10_gemv(const char * __restrict__ vx, const half2 * __restrict__ aq, const half2 * __restrict__ ads,
         float * __restrict__ dst, const int ngroups, const int nrows, const int row_bytes,
         const int stride_aq, const int stride_ads, const int stride_dst) {
    // sign table: entry v = the four +-1 half2 pairs of a sign byte v in slot order (bits (0,2) (1,3) (4,6) (5,7))
    __shared__ uint4 lut[256];
    for (int v = threadIdx.x; v < 256; v += 32*Q10_WARPS) {
        auto w2 = [&](const int lo, const int hi) -> uint32_t {
            const uint32_t l = (v >> lo) & 1u, h = (v >> hi) & 1u;
            return 0xBC00BC00u ^ (l << 15) ^ (h << 31);   // 0xBC00 = -1.0h, 0x3C00 = +1.0h
        };
        lut[v] = make_uint4(w2(0, 2), w2(1, 3), w2(4, 6), w2(5, 7));
    }
    __syncthreads();

    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;

    for (int row0 = (blockIdx.x*Q10_WARPS + warp)*R; row0 < nrows; row0 += gridDim.x*Q10_WARPS*R) {
        float sumf[NC][R];
#pragma unroll
        for (int j = 0; j < NC; ++j)
#pragma unroll
            for (int i = 0; i < R; ++i) sumf[j][i] = 0.0f;
        const char * const xbase = vx + (size_t) row0*row_bytes;

        for (int t = lane; t < ngroups; t += 32) {
            const int blk_i = t >> 2, g = t & 3;   // 128-weight block, 32-weight group inside it
            half2 a[NC][16];
            float yd[NC];
#pragma unroll
            for (int j = 0; j < NC; ++j) {
                const int4 * ap = (const int4 *) (aq + (size_t) j*stride_aq + (size_t) t*16);
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    const int4 av = ap[c];
                    a[j][4*c + 0] = *((const half2 *) &av.x);
                    a[j][4*c + 1] = *((const half2 *) &av.y);
                    a[j][4*c + 2] = *((const half2 *) &av.z);
                    a[j][4*c + 3] = *((const half2 *) &av.w);
                }
                yd[j] = __half22float2(ads[(size_t) j*stride_ads + t]).x;
            }

#pragma unroll
            for (int i = 0; i < R; ++i) {
                const char * blk = xbase + (size_t) i*row_bytes + (size_t) blk_i*18;
                const float d = __half2float(*((const half *) blk));
                const uint32_t q = q10_ld_u16(blk + 2 + 4*g) | (q10_ld_u16(blk + 4 + 4*g) << 16);
                half2 w[16];
                {
                    const uint4 e0 = lut[q & 0xFFu], e1 = lut[(q >> 8) & 0xFFu], e2 = lut[(q >> 16) & 0xFFu], e3 = lut[q >> 24];
                    const uint32_t u[16] = {e0.x, e2.x, e0.y, e2.y, e0.z, e2.z, e0.w, e2.w,
                                            e1.x, e3.x, e1.y, e3.y, e1.z, e3.z, e1.w, e3.w};
#pragma unroll
                    for (int c = 0; c < 16; ++c) w[c] = *((const half2 *) &u[c]);
                }
#pragma unroll
                for (int j = 0; j < NC; ++j) {
                    half2 se = make_half2(0.0f, 0.0f), so = make_half2(0.0f, 0.0f);
#pragma unroll
                    for (int c = 0; c < 16; c += 2) {
                        se = __hfma2(w[c    ], a[j][c    ], se);
                        so = __hfma2(w[c + 1], a[j][c + 1], so);
                    }
                    const float2 fe = __half22float2(se), fo = __half22float2(so);
                    // explicit rounding steps: no FMA contraction choice may depend on the column count (T invariance)
                    const float in = __fadd_rn(__fadd_rn(fe.x, fe.y), __fadd_rn(fo.x, fo.y));
                    sumf[j][i] = __fmaf_rn(yd[j], __fmul_rn(d, in), sumf[j][i]);
                }
            }
        }
#pragma unroll
        for (int j = 0; j < NC; ++j)
#pragma unroll
            for (int i = 0; i < R; ++i) {
                const float v = warp_reduce_sum(sumf[j][i]);
                if (lane == 0) dst[(size_t) j*stride_dst + row0 + i] = v;
            }
    }
}

// Grouped launch (MoE "mul_mat_id"): one launch covers G (token, expert) pairs. Group g reads expert rows at
// tab[g].w (any device or mapped-host row set of nrows rows), the a16 column tab[g].col of aq / ads, and writes output
// column tab[g].out of dst. Per-row arithmetic is q10_gemv's with NC = 1, so a group's result is bitwise the single
// (NC = 1) launch on that expert: the grouping only removes 10 x 3 x 48 launches per token.
template <int R>
static __global__ void __launch_bounds__(32*Q10_WARPS)
q10_gemv_grouped(const mv_group * __restrict__ tab, const half2 * __restrict__ aq, const half2 * __restrict__ ads,
                 float * __restrict__ dst, const int ngroups, const int nrows, const int row_bytes,
                 const int stride_aq, const int stride_ads, const int stride_dst) {
    __shared__ uint4 lut[256];
    for (int v = threadIdx.x; v < 256; v += 32*Q10_WARPS) {
        auto w2 = [&](const int lo, const int hi) -> uint32_t {
            const uint32_t l = (v >> lo) & 1u, h = (v >> hi) & 1u;
            return 0xBC00BC00u ^ (l << 15) ^ (h << 31);
        };
        lut[v] = make_uint4(w2(0, 2), w2(1, 3), w2(4, 6), w2(5, 7));
    }
    __syncthreads();
    const mv_group gr = tab[blockIdx.y];
    const char * __restrict__ vx = (const char *) gr.w;
    const half2 * __restrict__ aqc = aq + (size_t) gr.col*stride_aq;
    const half2 * __restrict__ adc = ads + (size_t) gr.col*stride_ads;
    float * __restrict__ dc = dst + (size_t) gr.out*stride_dst;

    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int row0 = (blockIdx.x*Q10_WARPS + warp)*R; row0 < nrows; row0 += gridDim.x*Q10_WARPS*R) {
        float sumf[R];
#pragma unroll
        for (int i = 0; i < R; ++i) sumf[i] = 0.0f;
        const char * const xbase = vx + (size_t) row0*row_bytes;
        for (int t = lane; t < ngroups; t += 32) {
            const int blk_i = t >> 2, g = t & 3;
            half2 a[16];
            {
                const int4 * ap = (const int4 *) (aqc + (size_t) t*16);
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    const int4 av = ap[c];
                    a[4*c + 0] = *((const half2 *) &av.x);
                    a[4*c + 1] = *((const half2 *) &av.y);
                    a[4*c + 2] = *((const half2 *) &av.z);
                    a[4*c + 3] = *((const half2 *) &av.w);
                }
            }
            const float yd = __half22float2(adc[t]).x;
#pragma unroll
            for (int i = 0; i < R; ++i) {
                const char * blk = xbase + (size_t) i*row_bytes + (size_t) blk_i*18;
                const float d = __half2float(*((const half *) blk));
                const uint32_t q = q10_ld_u16(blk + 2 + 4*g) | (q10_ld_u16(blk + 4 + 4*g) << 16);
                half2 w[16];
                {
                    const uint4 e0 = lut[q & 0xFFu], e1 = lut[(q >> 8) & 0xFFu], e2 = lut[(q >> 16) & 0xFFu], e3 = lut[q >> 24];
                    const uint32_t u[16] = {e0.x, e2.x, e0.y, e2.y, e0.z, e2.z, e0.w, e2.w,
                                            e1.x, e3.x, e1.y, e3.y, e1.z, e3.z, e1.w, e3.w};
#pragma unroll
                    for (int c = 0; c < 16; ++c) w[c] = *((const half2 *) &u[c]);
                }
                half2 se = make_half2(0.0f, 0.0f), so = make_half2(0.0f, 0.0f);
#pragma unroll
                for (int c = 0; c < 16; c += 2) {
                    se = __hfma2(w[c    ], a[c    ], se);
                    so = __hfma2(w[c + 1], a[c + 1], so);
                }
                const float2 fe = __half22float2(se), fo = __half22float2(so);
                const float in = __fadd_rn(__fadd_rn(fe.x, fe.y), __fadd_rn(fo.x, fo.y));
                sumf[i] = __fmaf_rn(yd, __fmul_rn(d, in), sumf[i]);
            }
        }
#pragma unroll
        for (int i = 0; i < R; ++i) {
            const float v = warp_reduce_sum(sumf[i]);
            if (lane == 0) dc[row0 + i] = v;
        }
    }
}

static int q10_sm_count() {
    static int cnt[16] = {0};
    int dev = 0;
    cudaGetDevice(&dev);
    dev &= 15;
    if (!cnt[dev]) { int n = 0; cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev); cnt[dev] = n > 0 ? n : 28; }
    return cnt[dev];
}

template <int NC, int R>
static void q10_launch(const char * vx, const a16_args & a, cudaStream_t s) {
    static const int per_sm = getenv("P100E_Q10_BLOCKS_PER_SM") ? std::max(1, atoi(getenv("P100E_Q10_BLOCKS_PER_SM"))) : 2;
    const int per_blk = Q10_WARPS*R;
    const int need = (a.nrows + per_blk - 1)/per_blk;
    const dim3 grid(std::max(1, std::min(need, per_sm*q10_sm_count())));
    q10_gemv<NC, R><<<grid, 32*Q10_WARPS, 0, s>>>(vx, a.aq, a.ads, a.dst, a.ngroups, a.nrows, a.row_bytes,
                                                 a.ngroups*16, a.ngroups, a.nrows);
}

// ncols 1..4; nrows % 4 == 0, row_bytes = 18 * n_in/128, rows 2-byte aligned (raw GGUF blocks)
void launch_q10(const char * vx, const a16_args & a, int ncols, cudaStream_t s) {
    static const int rf = getenv("P100E_Q10_R") ? atoi(getenv("P100E_Q10_R")) : 0;   // rows per warp: 2 / 4 / 8
    int r = rf ? rf : 8;
    if (r >= 8 && a.nrows % 8 != 0) r = 4;
#define C(NC) if (r >= 8) q10_launch<NC, 8>(vx, a, s); else if (r >= 4) q10_launch<NC, 4>(vx, a, s); else q10_launch<NC, 2>(vx, a, s);
    switch (ncols) {
        case 1: C(1) break;
        case 2: C(2) break;
        case 3: C(3) break;
        default: C(4) break;
    }
#undef C
}

// G groups (tab on the device), each a full nrows x n_in GEMV of one a16 column; per_sm blocks per SM are spread over the
// groups so a 10-expert layer still fills the GPU.
void launch_q10_grouped(const mv_group * tab, int G, const a16_args & a, cudaStream_t s) {
    static const int rf = getenv("P100E_Q10_R") ? atoi(getenv("P100E_Q10_R")) : 0;
    int r = rf ? rf : 8;
    if (r >= 8 && a.nrows % 8 != 0) r = 4;
    static const int per_sm = getenv("P100E_Q10_BLOCKS_PER_SM") ? std::max(1, atoi(getenv("P100E_Q10_BLOCKS_PER_SM"))) : 2;
    const int per_blk = Q10_WARPS*r;
    const int need = (a.nrows + per_blk - 1)/per_blk;
    const int bx = std::max(1, std::min(need, std::max(1, per_sm*q10_sm_count()/std::max(1, G))));
    const dim3 grid(bx, G);
    if (r >= 8)      q10_gemv_grouped<8><<<grid, 32*Q10_WARPS, 0, s>>>(tab, a.aq, a.ads, a.dst, a.ngroups, a.nrows, a.row_bytes, a.ngroups*16, a.ngroups, a.nrows);
    else if (r >= 4) q10_gemv_grouped<4><<<grid, 32*Q10_WARPS, 0, s>>>(tab, a.aq, a.ads, a.dst, a.ngroups, a.nrows, a.row_bytes, a.ngroups*16, a.ngroups, a.nrows);
    else             q10_gemv_grouped<2><<<grid, 32*Q10_WARPS, 0, s>>>(tab, a.aq, a.ads, a.dst, a.ngroups, a.nrows, a.row_bytes, a.ngroups*16, a.ngroups, a.nrows);
}

} // namespace p100e::fast
