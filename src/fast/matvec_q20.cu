// Q2_0 a16 HFMA2 GEMV (T = 1..4 columns): the cold-expert ffn_down of the Flash-Next split files
// Q2_0 (ggml type 42): 64 weights per 18-byte block
// [d f16][16 bytes of 2-bit codes], code q in {0,1,2,3} -> weight (q - 1) * d, i.e. {-d, 0, +d, +2d}
// (dequantize_row_q2_0). A 32-weight a16 group is 8 code bytes; one 256-entry table turns a code byte (4 weights) into
// the two (q - 1) half2 pairs the a16 slots want: byte b (elements 4b..4b+3) -> for b < 4 slots 4b = (v[4b], v[4b+2]) and
// 4b+2 = (v[4b+1], v[4b+3]); for b >= 4 (elements 16..31) slots 4(b-4)+1 and 4(b-4)+3. So the group's 16 slots are
// 8 LDS.64 lookups; then 16 HFMA2 per column, explicit f32 rounding per group (T invariance), d applied per group.
// Same structure and exactness story as matvec_q10.cu; rows stay the untouched GGUF bytes (2-byte aligned 18-byte blocks).
#include "matvec_internal.cuh"
#include <algorithm>
#include <cstdlib>

namespace p100e::fast {

#define Q20_WARPS 8

static __device__ __forceinline__ uint32_t q20_ld_u16(const char * p) { return *((const uint16_t *) p); }

template <int NC, int R>
static __global__ void __launch_bounds__(32*Q20_WARPS)
q20_gemv(const char * __restrict__ vx, const half2 * __restrict__ aq, const half2 * __restrict__ ads,
         float * __restrict__ dst, const int ngroups, const int nrows, const int row_bytes,
         const int stride_aq, const int stride_ads, const int stride_dst) {
    // sign table: entry v = the four +-1 half2 pairs of a sign byte v in slot order (bits (0,2) (1,3) (4,6) (5,7))
    // code byte v = 4 codes (bits 0-1, 2-3, 4-5, 6-7) -> pairs (code0, code2), (code1, code3) as (q - 1) in fp16
    __shared__ uint2 lut[256];
    for (int v = threadIdx.x; v < 256; v += 32*Q20_WARPS) {
        auto hv = [&](const int c) -> uint32_t {   // (q - 1) as a half bit pattern: -1, 0, +1, +2
            const uint32_t tab[4] = {0xBC00u, 0x0000u, 0x3C00u, 0x4000u};
            return tab[(v >> (2*c)) & 3];
        };
        lut[v] = make_uint2(hv(0) | (hv(2) << 16), hv(1) | (hv(3) << 16));
    }
    __syncthreads();

    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;

    for (int row0 = (blockIdx.x*Q20_WARPS + warp)*R; row0 < nrows; row0 += gridDim.x*Q20_WARPS*R) {
        float sumf[NC][R];
#pragma unroll
        for (int j = 0; j < NC; ++j)
#pragma unroll
            for (int i = 0; i < R; ++i) sumf[j][i] = 0.0f;
        const char * const xbase = vx + (size_t) row0*row_bytes;

        for (int t = lane; t < ngroups; t += 32) {
            const int blk_i = t >> 1, g = t & 1;   // 64-weight block, 32-weight group inside it
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
                // the group's 8 code bytes: two u32 words at byte offset 2 + 8g (2-byte aligned -> four u16 loads)
                const char * cb = blk + 2 + 8*g;
                const uint32_t qa = q20_ld_u16(cb) | (q20_ld_u16(cb + 2) << 16), qb = q20_ld_u16(cb + 4) | (q20_ld_u16(cb + 6) << 16);
                half2 w[16];
                {
                    // bytes 0..3 (elements 0..15) -> even slots, bytes 4..7 (elements 16..31) -> odd slots
                    const uint2 e0 = lut[qa & 0xFFu], e1 = lut[(qa >> 8) & 0xFFu], e2 = lut[(qa >> 16) & 0xFFu], e3 = lut[qa >> 24];
                    const uint2 f0 = lut[qb & 0xFFu], f1 = lut[(qb >> 8) & 0xFFu], f2 = lut[(qb >> 16) & 0xFFu], f3 = lut[qb >> 24];
                    const uint32_t u[16] = {e0.x, f0.x, e0.y, f0.y, e1.x, f1.x, e1.y, f1.y,
                                            e2.x, f2.x, e2.y, f2.y, e3.x, f3.x, e3.y, f3.y};
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
// column tab[g].out of dst. Per-row arithmetic is q20_gemv's with NC = 1, so a group's result is bitwise the single
// (NC = 1) launch on that expert: the grouping only removes 10 x 3 x 48 launches per token.
template <int R>
static __global__ void __launch_bounds__(32*Q20_WARPS)
q20_gemv_grouped(const mv_group * __restrict__ tab, const half2 * __restrict__ aq, const half2 * __restrict__ ads,
                 float * __restrict__ dst, const int ngroups, const int nrows, const int row_bytes,
                 const int stride_aq, const int stride_ads, const int stride_dst) {
    __shared__ uint2 lut[256];
    for (int v = threadIdx.x; v < 256; v += 32*Q20_WARPS) {
        auto hv = [&](const int c) -> uint32_t {   // (q - 1) as a half bit pattern: -1, 0, +1, +2
            const uint32_t tab4[4] = {0xBC00u, 0x0000u, 0x3C00u, 0x4000u};
            return tab4[(v >> (2*c)) & 3];
        };
        lut[v] = make_uint2(hv(0) | (hv(2) << 16), hv(1) | (hv(3) << 16));
    }
    __syncthreads();
    const mv_group gr = tab[blockIdx.y];
    const char * __restrict__ vx = (const char *) gr.w;
    const half2 * __restrict__ aqc = aq + (size_t) gr.col*stride_aq;
    const half2 * __restrict__ adc = ads + (size_t) gr.col*stride_ads;
    float * __restrict__ dc = dst + (size_t) gr.out*stride_dst;

    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int row0 = (blockIdx.x*Q20_WARPS + warp)*R; row0 < nrows; row0 += gridDim.x*Q20_WARPS*R) {
        float sumf[R];
#pragma unroll
        for (int i = 0; i < R; ++i) sumf[i] = 0.0f;
        const char * const xbase = vx + (size_t) row0*row_bytes;
        for (int t = lane; t < ngroups; t += 32) {
            const int blk_i = t >> 1, g = t & 1;
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
                // the group's 8 code bytes: two u32 words at byte offset 2 + 8g (2-byte aligned -> four u16 loads)
                const char * cb = blk + 2 + 8*g;
                const uint32_t qa = q20_ld_u16(cb) | (q20_ld_u16(cb + 2) << 16), qb = q20_ld_u16(cb + 4) | (q20_ld_u16(cb + 6) << 16);
                half2 w[16];
                {
                    // bytes 0..3 (elements 0..15) -> even slots, bytes 4..7 (elements 16..31) -> odd slots
                    const uint2 e0 = lut[qa & 0xFFu], e1 = lut[(qa >> 8) & 0xFFu], e2 = lut[(qa >> 16) & 0xFFu], e3 = lut[qa >> 24];
                    const uint2 f0 = lut[qb & 0xFFu], f1 = lut[(qb >> 8) & 0xFFu], f2 = lut[(qb >> 16) & 0xFFu], f3 = lut[qb >> 24];
                    const uint32_t u[16] = {e0.x, f0.x, e0.y, f0.y, e1.x, f1.x, e1.y, f1.y,
                                            e2.x, f2.x, e2.y, f2.y, e3.x, f3.x, e3.y, f3.y};
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

static int q20_sm_count() {
    static int cnt[16] = {0};
    int dev = 0;
    cudaGetDevice(&dev);
    dev &= 15;
    if (!cnt[dev]) { int n = 0; cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev); cnt[dev] = n > 0 ? n : 28; }
    return cnt[dev];
}

template <int NC, int R>
static void q20_launch(const char * vx, const a16_args & a, cudaStream_t s) {
    static const int per_sm = getenv("P100E_Q20_BLOCKS_PER_SM") ? std::max(1, atoi(getenv("P100E_Q20_BLOCKS_PER_SM"))) : 2;
    const int per_blk = Q20_WARPS*R;
    const int need = (a.nrows + per_blk - 1)/per_blk;
    const dim3 grid(std::max(1, std::min(need, per_sm*q20_sm_count())));
    q20_gemv<NC, R><<<grid, 32*Q20_WARPS, 0, s>>>(vx, a.aq, a.ads, a.dst, a.ngroups, a.nrows, a.row_bytes,
                                                 a.ngroups*16, a.ngroups, a.nrows);
}

// ncols 1..4; nrows % 4 == 0, row_bytes = 18 * n_in/64, rows 2-byte aligned (raw GGUF blocks)
void launch_q20(const char * vx, const a16_args & a, int ncols, cudaStream_t s) {
    static const int rf = getenv("P100E_Q20_R") ? atoi(getenv("P100E_Q20_R")) : 0;   // rows per warp: 2 / 4 / 8
    int r = rf ? rf : 8;
    if (r >= 8 && a.nrows % 8 != 0) r = 4;
#define C(NC) if (r >= 8) q20_launch<NC, 8>(vx, a, s); else if (r >= 4) q20_launch<NC, 4>(vx, a, s); else q20_launch<NC, 2>(vx, a, s);
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
void launch_q20_grouped(const mv_group * tab, int G, const a16_args & a, cudaStream_t s) {
    static const int rf = getenv("P100E_Q20_R") ? atoi(getenv("P100E_Q20_R")) : 0;
    int r = rf ? rf : 8;
    if (r >= 8 && a.nrows % 8 != 0) r = 4;
    static const int per_sm = getenv("P100E_Q20_BLOCKS_PER_SM") ? std::max(1, atoi(getenv("P100E_Q20_BLOCKS_PER_SM"))) : 2;
    const int per_blk = Q20_WARPS*r;
    const int need = (a.nrows + per_blk - 1)/per_blk;
    const int bx = std::max(1, std::min(need, std::max(1, per_sm*q20_sm_count()/std::max(1, G))));
    const dim3 grid(bx, G);
    if (r >= 8)      q20_gemv_grouped<8><<<grid, 32*Q20_WARPS, 0, s>>>(tab, a.aq, a.ads, a.dst, a.ngroups, a.nrows, a.row_bytes, a.ngroups*16, a.ngroups, a.nrows);
    else if (r >= 4) q20_gemv_grouped<4><<<grid, 32*Q20_WARPS, 0, s>>>(tab, a.aq, a.ads, a.dst, a.ngroups, a.nrows, a.row_bytes, a.ngroups*16, a.ngroups, a.nrows);
    else             q20_gemv_grouped<2><<<grid, 32*Q20_WARPS, 0, s>>>(tab, a.aq, a.ads, a.dst, a.ngroups, a.nrows, a.row_bytes, a.ngroups*16, a.ngroups, a.nrows);
}

} // namespace p100e::fast
