#include "matvec_internal.cuh"
#include <algorithm>
#include <cstdlib>

namespace p100e::fast {

#define KQ_ROWS 4
#define KQ_NWARPS 4

// ---------------------------------------------------------------- q3r: Q3_K in the repacked layout of q4r-sm60.cu
//
// Row (GGML_Q3R=1, built by q3r_convert): [group words 3 u32 per 32 values: A, B, H][scales 12 B/sb][d 2 B/sb].
// Same work mapping and a16 activations as mul_mat_vec_q4kr (thread = one 32-value group t of a row, KQ_ROWS rows per
// block).  Slot s = 4k + r: x = (W & 3<<2r) | M_r (W = A or B shifted by 8(k & 1), M_r = 1024, 256, 64, 16 in both
// halves) puts q2 at the magic's unit bit, OR-ing in bit 2r + 2 of the k-th h variant {H, H>>1, rotr8, rotr9} makes
// it M_r + q3 (q3 = q2 + 4h), and HSUB2 with M_r + 4 gives the exact weight q3 - 4 in [-4, 3] (kquant\q3r_model.py).
// Even slots are values 0-15 (scale 2g), odd slots 16-31 (scale 2g + 1), so two fp16 partial sums per row as in kq;
// there is no min term.  Numerics differ from the env-off MMVQ path (a16 instead of q8_1 activations).
template <int ncols_dst, int nwarps, int minb = 0>
static __global__ void __launch_bounds__(WARP_SIZE*nwarps, minb > 0 ? minb : 4*(KQ_NWARPS/nwarps))
mul_mat_vec_q3r(
        const char * __restrict__ vx, const half2 * __restrict__ aq, const half2 * __restrict__ ads,
        float * __restrict__ dst, const int ngroups, const int row_bytes,
        const int stride_col_aq, const int stride_col_ads, const int stride_col_dst) {
#if defined(FP16_AVAILABLE)
    const int nb       = ngroups/8;
    const int row0     = blockIdx.x*KQ_ROWS;
    const int tid      = threadIdx.x + threadIdx.y*WARP_SIZE;
    const int nthreads = WARP_SIZE*nwarps;

    // g = t & 7 is fixed per thread (nthreads % 8 == 0): scale indices 2g, 2g + 1.  Their low nibbles are byte
    // 2(g & 1) (+1) of scale word (g & 3) >> 1, low or high nibble for g < 4 / g >= 4; their 2 high bits are bits
    // 2(g >> 1) of bytes 2(g & 1) (+1) of scale word 2 (dequantize_q3_K's four cases in one expression).
    const int g        = tid & 7;
    const int widx     = (g & 3) >> 1;
    const int lo_shift = 16*(g & 1) + (g >= 4 ? 4 : 0);
    const int hi_shift = 16*(g & 1) + 2*(g >> 1);

    const uint32_t M[4]   = {0x64006400u, 0x5C005C00u, 0x54005400u, 0x4C004C00u}; // 1024, 256, 64, 16
    const half2    sub[4] = {__float2half2_rn(1028.0f), __float2half2_rn(260.0f),
                             __float2half2_rn(  68.0f), __float2half2_rn( 20.0f)};

    float sumf[ncols_dst][KQ_ROWS] = {{0.0f}};
    const char * const xbase = vx + (size_t) row0*row_bytes;

    for (int t = tid; t < ngroups; t += nthreads) {
        const int sb = t >> 3;

        half2 a[ncols_dst][16];
        float yd[ncols_dst];
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
            const int4 * ap = (const int4 *) (aq + (size_t) j*stride_col_aq + t*16);
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                const int4 av = ap[c];
                a[j][4*c + 0] = *((const half2 *) &av.x);
                a[j][4*c + 1] = *((const half2 *) &av.y);
                a[j][4*c + 2] = *((const half2 *) &av.z);
                a[j][4*c + 3] = *((const half2 *) &av.w);
            }
            yd[j] = __low2float(ads[j*stride_col_ads + t]);
        }

#pragma unroll
        for (int i = 0; i < KQ_ROWS; ++i) {
            const char * xr = xbase + i*row_bytes;
            const uint32_t * gw = (const uint32_t *) xr + 3*t;
            const uint32_t A = gw[0], B = gw[1], H = gw[2];
            const uint32_t * sw = (const uint32_t *) (xr + 96*nb) + 3*sb;
            const uint32_t wlo = sw[widx], w2 = sw[2];
            const float d = __half2float(*((const half *) (xr + 108*nb + 2*sb)));

            const int sc0 = (int) (((wlo >>  lo_shift)      & 0xF) | (((w2 >>  hi_shift)      & 3) << 4));
            const int sc1 = (int) (((wlo >> (lo_shift + 8)) & 0xF) | (((w2 >> (hi_shift + 8)) & 3) << 4));
            const float scale_e = d*(sc0 - 32);
            const float scale_o = d*(sc1 - 32);

            const uint32_t V[4] = {H, H >> 1, __funnelshift_r(H, H, 8), __funnelshift_r(H, H, 9)};

            half2 se[ncols_dst], so[ncols_dst];
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                se[j] = make_half2(0.0f, 0.0f);
                so[j] = make_half2(0.0f, 0.0f);
            }
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const uint32_t W = (k < 2 ? A : B) >> (8*(k & 1));
#pragma unroll
                for (int r = 0; r < 4; ++r) {
                    const uint32_t x = (W    & (0x00030003u << (2*r))) | M[r];
                    const uint32_t q = x | (V[k] & (0x00040004u << (2*r)));
                    const half2 w = __hsub2(*((const half2 *) &q), sub[r]);
#pragma unroll
                    for (int j = 0; j < ncols_dst; ++j) {
                        if (r & 1) {
                            so[j] = __hfma2(w, a[j][4*k + r], so[j]);
                        } else {
                            se[j] = __hfma2(w, a[j][4*k + r], se[j]);
                        }
                    }
                }
            }
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                const float2 fe = __half22float2(se[j]);
                const float2 fo = __half22float2(so[j]);
                sumf[j][i] += yd[j]*(scale_e*(fe.x + fe.y) + scale_o*(fo.x + fo.y));
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
static void q3r_launch_n(const char * vx, const a16_args & a, cudaStream_t stream) {
    const dim3 grid(a.nrows/KQ_ROWS, 1, 1);
    const int sa = a.ngroups*16, sd = a.ngroups, so = a.nrows;
    int nw = 2;
    while (nw > 1 && a.ngroups < WARP_SIZE*nw) nw /= 2;
    if (nw == 1) {
        mul_mat_vec_q3r<ncols_dst, 1><<<grid, dim3(WARP_SIZE, 1, 1), 0, stream>>>(vx, a.aq, a.ads, a.dst, a.ngroups, a.row_bytes, sa, sd, so);
    } else if (ncols_dst == 4) {   // fork default for width 4: minb = 6 (removes an 80-byte spill)
        mul_mat_vec_q3r<ncols_dst, 2, 6><<<grid, dim3(WARP_SIZE, 2, 1), 0, stream>>>(vx, a.aq, a.ads, a.dst, a.ngroups, a.row_bytes, sa, sd, so);
    } else {
        mul_mat_vec_q3r<ncols_dst, 2><<<grid, dim3(WARP_SIZE, 2, 1), 0, stream>>>(vx, a.aq, a.ads, a.dst, a.ngroups, a.row_bytes, sa, sd, so);
    }
}
void launch_q3r(const char * vx, const a16_args & a, int ncols, cudaStream_t s) {
    switch (ncols) {
        case 1: q3r_launch_n<1>(vx, a, s); break;
        case 2: q3r_launch_n<2>(vx, a, s); break;
        case 3: q3r_launch_n<3>(vx, a, s); break;
        case 4: q3r_launch_n<4>(vx, a, s); break;
        default: break;
    }
}

} // namespace p100e::fast
