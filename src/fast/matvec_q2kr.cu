#include "matvec_internal.cuh"
#include <algorithm>
#include <cstdlib>

namespace p100e::fast {

#define KQ_ROWS 4
#define KQ_NWARPS 4

// ---------------------------------------------------------------- q2kr: Q2_K in the repacked layout (matvec_pack_body.inc)
//
// Row (LAYOUT_Q2KR, built by q2kr_convert): [group plane: 2 u32 (A, B) per 32 values][header plane: 20 B per sb =
// scales[16], d, dmin].  A lane owns one 32-value group t of a row (one 8-byte load, consecutive lanes = consecutive
// 8-byte chunks: a warp reads 256 contiguous bytes), KQ_ROWS rows per block, the a16 activations of mul_mat_vec_q3r.
//
// Decode is q3r's with no high bit and no bias: slot s = 4k + r (W = A or B shifted by 8(k & 1)): x = (W & 3<<2r) | M_r
// puts q2 at the magic's unit bit, and HSUB2 with M_r (1024, 256, 64, 16) gives the exact weight q2 in [0, 3].  Even slots
// are values 0-15 (scale / min byte 2g), odd slots 16-31 (byte 2g + 1): two fp16 partial sums per row.
// The min term needs the activation sums of the two 16-value halves (ye, yo): they do not depend on the row, so they are
// computed once per (group, column) from the a16 slots and reused by the KQ_ROWS rows.
//   y = yd * sum_g [ d*(sc_e*S_e + sc_o*S_o) - dmin*(m_e*ye + m_o*yo) ]      (S = sum q2*a over the half)
// Numerics differ from the generic f32 GEMV only by the a16 activation rounding and fp16 partial sums.
template <int ncols_dst, int nwarps, int minb = 0>
static __global__ void __launch_bounds__(WARP_SIZE*nwarps, minb > 0 ? minb : 4*(KQ_NWARPS/nwarps))
mul_mat_vec_q2kr(
        const char * __restrict__ vx, const half2 * __restrict__ aq, const half2 * __restrict__ ads,
        float * __restrict__ dst, const int ngroups, const int row_bytes,
        const int stride_col_aq, const int stride_col_ads, const int stride_col_dst) {
#if defined(FP16_AVAILABLE)
    const int nb       = ngroups/8;
    const int row0     = blockIdx.x*KQ_ROWS;
    const int tid      = threadIdx.x + threadIdx.y*WARP_SIZE;
    const int nthreads = WARP_SIZE*nwarps;

    // g = t & 7 is fixed per thread (nthreads % 8 == 0): scale bytes 2g, 2g + 1 are bytes 0-1 (g even) / 2-3 (g odd) of
    // header word g >> 1; each byte is (min << 4) | scale
    const int g     = tid & 7;
    const int widx  = g >> 1;
    const int shift = 16*(g & 1);

    const uint32_t M[4]   = {0x64006400u, 0x5C005C00u, 0x54005400u, 0x4C004C00u}; // 1024, 256, 64, 16
    const half2    sub[4] = {__float2half2_rn(1024.0f), __float2half2_rn(256.0f),
                             __float2half2_rn(  64.0f), __float2half2_rn( 16.0f)};

    float sumf[ncols_dst][KQ_ROWS] = {{0.0f}};
    const char * const xbase = vx + (size_t) row0*row_bytes;
    const int htail = 64*nb;

    for (int t = tid; t < ngroups; t += nthreads) {
        const int sb = t >> 3;

        // weights first: all rows' loads are issued before the activation work
        uint2    gw[KQ_ROWS];
        uint32_t sw[KQ_ROWS], dmw[KQ_ROWS];
#pragma unroll
        for (int i = 0; i < KQ_ROWS; ++i) {
            const char * xr = xbase + i*row_bytes;
            gw[i]  = *((const uint2 *) (xr + 8*t));
            const uint32_t * hw = (const uint32_t *) (xr + htail + 20*sb);
            sw[i]  = hw[widx];
            dmw[i] = hw[4];
        }

        // activation columns in passes of NJ: the a16 registers (16 half2 per column) dominate the register count, and T = 4
        // with all four columns live capped occupancy (165 regs); per-column math is unchanged (T invariance holds)
        constexpr int NJ = ncols_dst == 4 ? 1 : ncols_dst;
#pragma unroll
        for (int jb = 0; jb < ncols_dst; jb += NJ) {
            half2 a[NJ][16];
            float yd[NJ], ye[NJ], yo[NJ];
#pragma unroll
            for (int j = 0; j < NJ; ++j) {
                const int4 * ap = (const int4 *) (aq + (size_t) (jb + j)*stride_col_aq + t*16);
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    const int4 av = ap[c];
                    a[j][4*c + 0] = *((const half2 *) &av.x);
                    a[j][4*c + 1] = *((const half2 *) &av.y);
                    a[j][4*c + 2] = *((const half2 *) &av.z);
                    a[j][4*c + 3] = *((const half2 *) &av.w);
                }
                yd[j] = __low2float(ads[(jb + j)*stride_col_ads + t]);
                half2 pe = __hadd2(__hadd2(a[j][0], a[j][2]), __hadd2(a[j][4], a[j][6]));
                half2 po = __hadd2(__hadd2(a[j][1], a[j][3]), __hadd2(a[j][5], a[j][7]));
                pe = __hadd2(pe, __hadd2(__hadd2(a[j][8],  a[j][10]), __hadd2(a[j][12], a[j][14])));
                po = __hadd2(po, __hadd2(__hadd2(a[j][9],  a[j][11]), __hadd2(a[j][13], a[j][15])));
                const float2 fe = __half22float2(pe), fo = __half22float2(po);
                ye[j] = fe.x + fe.y;
                yo[j] = fo.x + fo.y;
            }

#pragma unroll
            for (int i = 0; i < KQ_ROWS; ++i) {
                const uint32_t s32 = sw[i] >> shift;
                const float sc_e = (float) (s32 & 0xF),         m_e = (float) ((s32 >>  4) & 0xF);
                const float sc_o = (float) ((s32 >> 8) & 0xF),  m_o = (float) ((s32 >> 12) & 0xF);
                const float2 dm = __half22float2(*((const half2 *) &dmw[i]));
                const float d_e = dm.x*sc_e, d_o = dm.x*sc_o;

                half2 se[NJ], so[NJ];
#pragma unroll
                for (int j = 0; j < NJ; ++j) {
                    se[j] = make_half2(0.0f, 0.0f);
                    so[j] = make_half2(0.0f, 0.0f);
                }
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const uint32_t W = (k < 2 ? gw[i].x : gw[i].y) >> (8*(k & 1));
#pragma unroll
                    for (int r = 0; r < 4; ++r) {
                        const uint32_t x = (W & (0x00030003u << (2*r))) | M[r];
                        const half2 w = __hsub2(*((const half2 *) &x), sub[r]);
#pragma unroll
                        for (int j = 0; j < NJ; ++j) {
                            if (r & 1) {
                                so[j] = __hfma2(w, a[j][4*k + r], so[j]);
                            } else {
                                se[j] = __hfma2(w, a[j][4*k + r], se[j]);
                            }
                        }
                    }
                }
#pragma unroll
                for (int j = 0; j < NJ; ++j) {
                    const float2 fe = __half22float2(se[j]);
                    const float2 fo = __half22float2(so[j]);
                    sumf[jb + j][i] += yd[j]*(d_e*(fe.x + fe.y) + d_o*(fo.x + fo.y) - dm.y*(m_e*ye[j] + m_o*yo[j]));
                }
            }
        }
    }

    if constexpr (nwarps == 1) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < KQ_ROWS; ++i) {
                const float v = warp_reduce_sum(sumf[j][i]);
                if (threadIdx.x == i) {
                    dst[j*stride_col_dst + row0 + i] = v;
                }
            }
        }
    } else {
        __shared__ float tmp[nwarps][ncols_dst][KQ_ROWS];
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < KQ_ROWS; ++i) {
                const float v = warp_reduce_sum(sumf[j][i]);
                if (threadIdx.x == 0) {
                    tmp[threadIdx.y][j][i] = v;
                }
            }
        }
        __syncthreads();
        if (threadIdx.y != 0) {
            return;
        }
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
            if (threadIdx.x < KQ_ROWS) {
                float v = 0.0f;
#pragma unroll
                for (int w = 0; w < nwarps; ++w) {
                    v += tmp[w][j][threadIdx.x];
                }
                dst[j*stride_col_dst + row0 + threadIdx.x] = v;
            }
        }
    }
#else
    GGML_UNUSED_VARS(vx, aq, ads, dst, ngroups, row_bytes, stride_col_aq, stride_col_ads, stride_col_dst);
    NO_DEVICE_CODE;
#endif // FP16_AVAILABLE
}

template <int ncols_dst>
static void q2kr_launch_n(const char * vx, const a16_args & a, cudaStream_t stream) {
    const dim3 grid(a.nrows/KQ_ROWS, 1, 1);
    const int sa = a.ngroups*16, sd = a.ngroups, so = a.nrows;
    int nw = 2;
    while (nw > 1 && a.ngroups < WARP_SIZE*nw) nw /= 2;
    if (nw == 1) {
        mul_mat_vec_q2kr<ncols_dst, 1><<<grid, dim3(WARP_SIZE, 1, 1), 0, stream>>>(vx, a.aq, a.ads, a.dst, a.ngroups, a.row_bytes, sa, sd, so);
    } else if (ncols_dst == 4) {
        mul_mat_vec_q2kr<ncols_dst, 2, 6><<<grid, dim3(WARP_SIZE, 2, 1), 0, stream>>>(vx, a.aq, a.ads, a.dst, a.ngroups, a.row_bytes, sa, sd, so);
    } else {
        mul_mat_vec_q2kr<ncols_dst, 2><<<grid, dim3(WARP_SIZE, 2, 1), 0, stream>>>(vx, a.aq, a.ads, a.dst, a.ngroups, a.row_bytes, sa, sd, so);
    }
}
void launch_q2kr(const char * vx, const a16_args & a, int ncols, cudaStream_t s) {
    switch (ncols) {
        case 1: q2kr_launch_n<1>(vx, a, s); break;
        case 2: q2kr_launch_n<2>(vx, a, s); break;
        case 3: q2kr_launch_n<3>(vx, a, s); break;
        case 4: q2kr_launch_n<4>(vx, a, s); break;
        default: break;
    }
}

} // namespace p100e::fast
