// IQ1_S / IQ1_M a16 HFMA2 GEMV for the MTP draft block (T = 1..4 columns).
//
// Weights stay the untouched GGUF rows (LAYOUT_KQRAW): IQ1_S 50-byte blocks [d][qs 32][qh 8 x u16], IQ1_M 56-byte
// blocks [qs 32][qh 16][scales 4 x u16]. A 32-weight group = 4 grid entries of 8 weights; each entry is an index into
// the 2048-entry ggml grid (values -1/0/+1 after the -1 offset), so the group decode is 4 table lookups.
//
// The grid is expanded once per CUDA block into shared memory as f16: 2048 x 16 bytes (32 KiB), one LDS.128 per entry
// giving the entry's 8 weights already as the four half2 pairs the a16 activation slots want (slot 4k = (a[4k], a[4k+2]),
// slot 4k+1 = (a[4k+16], a[4k+18]), slot 4k+2 = (a[4k+1], a[4k+3]), slot 4k+3 = (a[4k+17], a[4k+19]); see
// quantize_a16 / matvec_kq.cu). Entry l of the group covers elements 8l..8l+7:
//     l = 0 -> slots 0,2,4,6    l = 1 -> slots 8,10,12,14    l = 2 -> slots 1,3,5,7    l = 3 -> slots 9,11,13,15
// so even slots are elements 0..15 and odd slots elements 16..31 (IQ1_M has one scale per 16 elements).
// Block structure: persistent blocks (grid ~ 2 x SMs, the table is filled once per block), 8 warps, each warp R rows at
// a time, lanes strided over the 32-weight groups of a row, the a16 activation loaded once per group and shared by the
// R rows. A column's sum does not depend on the number of columns in the launch (T invariance).
//   value = dl * (g + delta),  g in {-1,0,1},  delta = +-1/8
//   sum_k value_k a_k = dl * sum g_k a_k + dl * delta * sum a_k     (IQ1_S: delta and dl per 32-group, sum a_k = ads.y)
// IQ1_M: dl per 16 elements, delta per 8-element entry: the bias uses the entry sums of a (from the f16 slots).
#include "matvec_internal.cuh"
#include "dequantize.cuh"
#include <algorithm>
#include <cstdlib>

namespace p100e::fast {

#define IQ1_WARPS 8

static __device__ __forceinline__ uint32_t ld_u16(const char * p) { return *((const uint16_t *) p); }

template <ggml_type TY, int NC, int R>
static __global__ void __launch_bounds__(32*IQ1_WARPS)
iq1_gemv(const char * __restrict__ vx, const half2 * __restrict__ aq, const half2 * __restrict__ ads,
         float * __restrict__ dst, const int ngroups, const int nrows, const int row_bytes,
         const int stride_aq, const int stride_ads, const int stride_dst) {
    __shared__ uint4 lut[2048];
    for (int i = threadIdx.x; i < 2048; i += 32*IQ1_WARPS) {
        const uint32_t gq = iq1s_grid_gpu[i];   // byte b: low nibble = weight b, high nibble = weight b + 4 (value + 1)
        auto q = [&](const int j) { return (float) ((int) ((gq >> (8*(j & 3) + 4*(j >> 2))) & 0xFu) - 1); };
        const half2 h0 = __floats2half2_rn(q(0), q(2));
        const half2 h1 = __floats2half2_rn(q(1), q(3));
        const half2 h2 = __floats2half2_rn(q(4), q(6));
        const half2 h3 = __floats2half2_rn(q(5), q(7));
        lut[i] = make_uint4(*((const uint32_t *) &h0), *((const uint32_t *) &h1), *((const uint32_t *) &h2), *((const uint32_t *) &h3));
    }
    __syncthreads();

    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    constexpr int blk_bytes = TY == GGML_TYPE_IQ1_S ? 50 : 56;

    for (int row0 = (blockIdx.x*IQ1_WARPS + warp)*R; row0 < nrows; row0 += gridDim.x*IQ1_WARPS*R) {
        float sumf[NC][R];
#pragma unroll
        for (int j = 0; j < NC; ++j)
#pragma unroll
            for (int i = 0; i < R; ++i) sumf[j][i] = 0.0f;
        const char * const xbase = vx + (size_t) row0*row_bytes;

        for (int t = lane; t < ngroups; t += 32) {
            const int sb = t >> 3, g = t & 7;
            half2 a[NC][16];
            float yd[NC], ys[NC];
            [[maybe_unused]] float es[NC][4];
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
                const float2 ds = __half22float2(ads[(size_t) j*stride_ads + t]);
                yd[j] = ds.x;
                ys[j] = ds.y;
                if constexpr (TY == GGML_TYPE_IQ1_M) {
                    // sums of the 8 elements of each entry (f16 slot adds; they only scale the 1/8 delta)
                    const half2 s0 = __hadd2(__hadd2(a[j][0], a[j][2]),  __hadd2(a[j][4],  a[j][6]));
                    const half2 s1 = __hadd2(__hadd2(a[j][8], a[j][10]), __hadd2(a[j][12], a[j][14]));
                    const half2 s2 = __hadd2(__hadd2(a[j][1], a[j][3]),  __hadd2(a[j][5],  a[j][7]));
                    const half2 s3 = __hadd2(__hadd2(a[j][9], a[j][11]), __hadd2(a[j][13], a[j][15]));
                    const float2 f0 = __half22float2(s0), f1 = __half22float2(s1), f2 = __half22float2(s2), f3 = __half22float2(s3);
                    es[j][0] = f0.x + f0.y; es[j][1] = f1.x + f1.y; es[j][2] = f2.x + f2.y; es[j][3] = f3.x + f3.y;
                }
            }

#pragma unroll
            for (int i = 0; i < R; ++i) {
                const char * blk = xbase + (size_t) i*row_bytes + (size_t) sb*blk_bytes;
                uint32_t idx[4];
                float scale_e, scale_o, bias_s = 0.0f;
                uint32_t sg = 0;   // IQ1_M: delta sign bits of the four entries at bits 3, 7, 11, 15
                if constexpr (TY == GGML_TYPE_IQ1_S) {
                    const float d = __half2float(*((const half *) blk));
                    const uint32_t qs = ld_u16(blk + 2 + 4*g) | (ld_u16(blk + 4 + 4*g) << 16);
                    const uint32_t qh = ld_u16(blk + 34 + 2*g);
                    idx[0] = (qs & 0xFFu)         | ((qh & 7u) << 8);
                    idx[1] = ((qs >> 8) & 0xFFu)  | (((qh >> 3) & 7u) << 8);
                    idx[2] = ((qs >> 16) & 0xFFu) | (((qh >> 6) & 7u) << 8);
                    idx[3] = (qs >> 24)           | (((qh >> 9) & 7u) << 8);
                    scale_e = scale_o = d*(float) (2*((qh >> 12) & 7u) + 1);
                    bias_s = scale_e*((qh & 0x8000u) ? -0.125f : 0.125f);
                } else {
                    const uint32_t qs = *((const uint32_t *) (blk + 4*g));
                    const uint32_t qh = ld_u16(blk + 32 + 2*g);
                    const uint2 sc2 = *((const uint2 *) (blk + 48));
                    const uint32_t s0 = sc2.x & 0xFFFFu, s1 = sc2.x >> 16, s2 = sc2.y & 0xFFFFu, s3 = sc2.y >> 16;
                    const uint16_t dbits = (uint16_t) ((s0 >> 12) | ((s1 >> 8) & 0x00f0u) | ((s2 >> 4) & 0x0f00u) | (s3 & 0xf000u));
                    const float d = __half2float(__ushort_as_half(dbits));
                    const uint32_t sw = g < 2 ? s0 : g < 4 ? s1 : g < 6 ? s2 : s3;
                    const int sh = 6*(g & 1);
                    scale_e = d*(float) (2*((sw >> sh) & 7u) + 1);
                    scale_o = d*(float) (2*((sw >> (sh + 3)) & 7u) + 1);
                    idx[0] = (qs & 0xFFu)         | ((qh & 7u) << 8);
                    idx[1] = ((qs >> 8) & 0xFFu)  | (((qh >> 4) & 7u) << 8);
                    idx[2] = ((qs >> 16) & 0xFFu) | (((qh >> 8) & 7u) << 8);
                    idx[3] = (qs >> 24)           | (((qh >> 12) & 7u) << 8);
                    sg = qh;
                }
                half2 w[16];
                {
                    const uint4 e0 = lut[idx[0]], e1 = lut[idx[1]], e2 = lut[idx[2]], e3 = lut[idx[3]];
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
                    if constexpr (TY == GGML_TYPE_IQ1_S) {
                        // explicit rounding steps: no FMA contraction choice may depend on the column count (T invariance)
                        const float in = __fadd_rn(__fadd_rn(fe.x, fe.y), __fadd_rn(fo.x, fo.y));
                        const float tt = __fmaf_rn(scale_e, in, __fmul_rn(bias_s, ys[j]));
                        sumf[j][i] = __fmaf_rn(yd[j], tt, sumf[j][i]);
                    } else {
                        const float d0 = (sg & 0x0008u) ? -0.125f : 0.125f, d1 = (sg & 0x0080u) ? -0.125f : 0.125f;
                        const float d2 = (sg & 0x0800u) ? -0.125f : 0.125f, d3 = (sg & 0x8000u) ? -0.125f : 0.125f;
                        const float be = __fmaf_rn(d1, es[j][1], __fmul_rn(d0, es[j][0]));
                        const float bo = __fmaf_rn(d3, es[j][3], __fmul_rn(d2, es[j][2]));
                        const float tt = __fmaf_rn(scale_o, __fadd_rn(fo.x, fo.y), __fmul_rn(scale_e, __fadd_rn(fe.x, fe.y)));
                        const float bias = __fmaf_rn(scale_o, bo, __fmul_rn(scale_e, be));
                        sumf[j][i] = __fmaf_rn(yd[j], __fadd_rn(tt, bias), sumf[j][i]);
                    }
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

static int sm_count() {
    static int cnt[16] = {0};
    int dev = 0;
    cudaGetDevice(&dev);
    dev &= 15;
    if (!cnt[dev]) { int n = 0; cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev); cnt[dev] = n > 0 ? n : 28; }
    return cnt[dev];
}

template <ggml_type TY, int NC, int R>
static void iq1_launch(const char * vx, const a16_args & a, cudaStream_t s) {
    static const int per_sm = getenv("P100E_IQ1_BLOCKS_PER_SM") ? std::max(1, atoi(getenv("P100E_IQ1_BLOCKS_PER_SM"))) : 2;
    const int per_blk = IQ1_WARPS*R;
    const int need = (a.nrows + per_blk - 1)/per_blk;
    const dim3 grid(std::max(1, std::min(need, per_sm*sm_count())));
    iq1_gemv<TY, NC, R><<<grid, 32*IQ1_WARPS, 0, s>>>(vx, a.aq, a.ads, a.dst, a.ngroups, a.nrows, a.row_bytes,
                                                     a.ngroups*16, a.ngroups, a.nrows);
}

template <ggml_type TY>
static void iq1_dispatch(const char * vx, const a16_args & a, int ncols, cudaStream_t s) {
    // rows per warp (the a16 activation of a group is loaded once for R rows): P100E_IQ1_R = 2 / 4 / 8, default by columns
    static const int rf = getenv("P100E_IQ1_R") ? atoi(getenv("P100E_IQ1_R")) : 0;
    int r = rf ? rf : 8;
    if (r >= 16 && a.nrows % 16 != 0) r = 8;
    if (r >= 8 && a.nrows % 8 != 0) r = 4;   // nrows % 4 == 0 is guaranteed by pack_weight
#define C(NC) if (r >= 16) iq1_launch<TY, NC, 16>(vx, a, s); else if (r >= 8) iq1_launch<TY, NC, 8>(vx, a, s); else if (r >= 4) iq1_launch<TY, NC, 4>(vx, a, s); else iq1_launch<TY, NC, 2>(vx, a, s);
    switch (ncols) {
        case 1: C(1) break;
        case 2: C(2) break;
        case 3: C(3) break;
        case 4: C(4) break;
        default: break;
    }
#undef C
}

// ncols 1..4; nrows % 4 == 0, row_bytes = 50 / 56 * n_in/256, rows 2 / 8 byte aligned
void launch_iq1s(const char * vx, const a16_args & a, int ncols, cudaStream_t s) { iq1_dispatch<GGML_TYPE_IQ1_S>(vx, a, ncols, s); }
void launch_iq1m(const char * vx, const a16_args & a, int ncols, cudaStream_t s) { iq1_dispatch<GGML_TYPE_IQ1_M>(vx, a, ncols, s); }

} // namespace p100e::fast
