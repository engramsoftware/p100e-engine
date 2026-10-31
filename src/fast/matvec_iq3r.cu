#include "matvec_internal.cuh"
#include "dequantize.cuh"   // third_party/ggml: iq3xxs_grid / iq3s_grid device tables
#include <algorithm>
#include <cstdlib>

namespace p100e::fast {

#define IQ3_ROWS 4

static __device__ __forceinline__ uint32_t iq3_h2u(half2 v) { return *((const uint32_t *) &v); }
static __device__ __forceinline__ half2    iq3_u2h(uint32_t u) { return *((const half2 *) &u); }

// ---------------------------------------------------------------- iq3r: IQ3_XXS / IQ3_S in the plane-split layouts
//
// Rows (matvec_pack_body.inc): IQ3_XXS [qs 64*nb][ss 32*nb][d 2*nb], IQ3_S [qs 64*nb][signs 32*nb][qh 8*nb][scales 4*nb][d 2*nb].
// A lane owns one 32-value group t of a row (8 grid-index bytes, one 4-byte sign / scale word), IQ3_ROWS rows per block, the a16
// activations of mul_mat_vec_q2kr.
//
// Group = 8 grid entries m = 0..7 of 4 values (value = d_g * grid byte * sign).  The a16 slot s = 4k + 2j + off holds the
// activation pair (elements 16*off + 4k + j, +2): the weight pair of slot (k, j, off) is bytes (j, j + 2) of grid entry
// m = 4*off + k.  Decode of a pair: PRMT puts the two grid bytes under the fp16 magic 0x6400 (1024 + b, exact), HSUB2 with 1024
// gives b, an XOR with the sign mask of the pair gives +-b.  The sign masks come from a 16-entry shared table indexed by
// the 4-bit sign nibble of the entry (two masks: j = 0 / j = 1), the grid from a shared copy of the 256 / 512 entry table.
//   IQ3_XXS: sign nibble from ksigns_iq2xs[(aux >> 7*(m>>1)) & 127] (7 bits + parity bit), d_g = d*(0.5 + (aux >> 28))*0.5
//   IQ3_S  : grid index = qs[m] | (qh bit m) << 8, sign nibble = (signs >> 4m) & 15, d_g = d*(1 + 2*scale nibble)
// y = yd * sum_g d_g * (sum over the group of b*sign*a): the group sum is two fp16 partial sums (j = 0, 1), b <= 62 (<= 15 for
// IQ3_S) and |a| <= 1 so no overflow; numerics differ from the generic f32 GEMV only by the a16 activation rounding and fp16
// partial sums.
template <bool S, int ncols_dst, int nwarps, int minb = 0>
static __global__ void __launch_bounds__(WARP_SIZE*nwarps, minb > 0 ? minb : 4*(4/nwarps))
mul_mat_vec_iq3r(
        const char * __restrict__ vx, const half2 * __restrict__ aq, const half2 * __restrict__ ads,
        float * __restrict__ dst, const int ngroups, const int row_bytes,
        const int stride_col_aq, const int stride_col_ads, const int stride_col_dst) {
#if defined(FP16_AVAILABLE)
    constexpr int NG = S ? 512 : 256;
    __shared__ uint32_t sgrid[NG];
    __shared__ uint2    ssgn[16];

    const int nb       = ngroups/8;
    const int row0     = blockIdx.x*IQ3_ROWS;
    const int tid      = threadIdx.x + threadIdx.y*WARP_SIZE;
    const int nthreads = WARP_SIZE*nwarps;

    for (int i = tid; i < NG; i += nthreads) sgrid[i] = S ? iq3s_grid[i] : iq3xxs_grid[i];
    if (tid < 16) {   // fp16 sign masks of the value pairs (c = j, j + 2) of a 4-value nibble
        ssgn[tid] = make_uint2((uint32_t) (((tid & 1) << 15) | ((tid & 4) << 29)), (uint32_t) (((tid & 2) << 14) | ((tid & 8) << 28)));
    }
    __syncthreads();

    const half2 c1024 = __float2half2_rn(1024.0f);
    const char * const xbase = vx + (size_t) row0*row_bytes;

    float sumf[ncols_dst][IQ3_ROWS] = {{0.0f}};

    for (int t = tid; t < ngroups; t += nthreads) {
        const int sb = t >> 3, g = t & 7;

        // weights first: all rows' loads are issued before the activation work
        uint32_t qw[IQ3_ROWS][2], sw[IQ3_ROWS];
        uint32_t aux2[IQ3_ROWS];                     // XXS: unused; S: qh byte | scales byte << 8
        uint32_t dw[IQ3_ROWS];
#pragma unroll
        for (int i = 0; i < IQ3_ROWS; ++i) {
            const char * xr = xbase + (size_t) i*row_bytes;
            qw[i][0] = *((const uint32_t *) (xr + 8*t));
            qw[i][1] = *((const uint32_t *) (xr + 8*t + 4));
            sw[i]    = *((const uint32_t *) (xr + 64*nb + 4*t));
            if (S) {
                aux2[i] = (uint32_t) *((const uint8_t *) (xr + 96*nb + t)) |
                          ((uint32_t) *((const uint8_t *) (xr + 104*nb + 4*sb + (g >> 1))) << 8);
                dw[i]   = *((const uint16_t *) (xr + 108*nb + 2*sb));
            } else {
                aux2[i] = 0;
                dw[i]   = *((const uint16_t *) (xr + 96*nb + 2*sb));
            }
        }

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
        for (int i = 0; i < IQ3_ROWS; ++i) {
            const uint32_t sx = sw[i];
            float dg;
            if (S) {
                const int sc = (aux2[i] >> (8 + 4*(g & 1))) & 0xF;
                dg = __half2float(__ushort_as_half((unsigned short) dw[i]))*(float) (1 + 2*sc);
            } else {
                dg = __half2float(__ushort_as_half((unsigned short) dw[i]))*(0.5f + (float) (sx >> 28))*0.5f;
            }

            half2 se[ncols_dst], so[ncols_dst];
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                se[j] = make_half2(0.0f, 0.0f);
                so[j] = make_half2(0.0f, 0.0f);
            }
#pragma unroll
            for (int m = 0; m < 8; ++m) {
                uint32_t idx = (qw[i][m >> 2] >> (8*(m & 3))) & 0xFFu;
                uint32_t nib;
                if (S) {
                    idx |= ((aux2[i] >> m) & 1u) << 8;
                    nib = (sx >> (4*m)) & 0xFu;
                } else {
                    const uint32_t s7 = (sx >> (7*(m >> 1))) & 127u;
                    const uint32_t s8 = s7 | ((__popc(s7) & 1u) << 7);
                    nib = (s8 >> (4*(m & 1))) & 0xFu;
                }
                const uint32_t G  = sgrid[idx];
                const uint2    sm = ssgn[nib];
                const uint32_t x0 = __byte_perm(G, 0x64646464u, 0x4240);   // grid bytes 0, 2 under 0x64
                const uint32_t x1 = __byte_perm(G, 0x64646464u, 0x4341);   // grid bytes 1, 3
                const half2 w0 = iq3_u2h(iq3_h2u(__hsub2(iq3_u2h(x0), c1024)) ^ sm.x);
                const half2 w1 = iq3_u2h(iq3_h2u(__hsub2(iq3_u2h(x1), c1024)) ^ sm.y);
                const int k = m & 3, off = m >> 2;
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    se[j] = __hfma2(w0, a[j][4*k + off],     se[j]);
                    so[j] = __hfma2(w1, a[j][4*k + 2 + off], so[j]);
                }
            }
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                const float2 fe = __half22float2(se[j]);
                const float2 fo = __half22float2(so[j]);
                sumf[j][i] += yd[j]*dg*((fe.x + fe.y) + (fo.x + fo.y));
            }
        }
    }

    if constexpr (nwarps == 1) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < IQ3_ROWS; ++i) {
                const float v = warp_reduce_sum(sumf[j][i]);
                if (threadIdx.x == i) {
                    dst[j*stride_col_dst + row0 + i] = v;
                }
            }
        }
    } else {
        __shared__ float tmp[nwarps][ncols_dst][IQ3_ROWS];
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < IQ3_ROWS; ++i) {
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
            if (threadIdx.x < IQ3_ROWS) {
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

template <bool S, int ncols_dst>
static void iq3r_launch_n(const char * vx, const a16_args & a, cudaStream_t stream) {
    const dim3 grid(a.nrows/IQ3_ROWS, 1, 1);
    const int sa = a.ngroups*16, sd = a.ngroups, so = a.nrows;
    int nw = 2;
    while (nw > 1 && a.ngroups < WARP_SIZE*nw) nw /= 2;
    // 3 / 4 columns: a[ncols][16] + 4 rows of decode state do not fit 128 registers, so fewer resident blocks (no spills)
    constexpr int mb2 = ncols_dst == 4 ? 5 : ncols_dst == 3 ? 6 : 0;
    constexpr int mb1 = ncols_dst == 4 ? 8 : ncols_dst == 3 ? 12 : 0;
    if (nw == 1) {
        mul_mat_vec_iq3r<S, ncols_dst, 1, mb1><<<grid, dim3(WARP_SIZE, 1, 1), 0, stream>>>(vx, a.aq, a.ads, a.dst, a.ngroups, a.row_bytes, sa, sd, so);
    } else {
        mul_mat_vec_iq3r<S, ncols_dst, 2, mb2><<<grid, dim3(WARP_SIZE, 2, 1), 0, stream>>>(vx, a.aq, a.ads, a.dst, a.ngroups, a.row_bytes, sa, sd, so);
    }
}
template <bool S>
static void iq3r_launch(const char * vx, const a16_args & a, int ncols, cudaStream_t s) {
    switch (ncols) {
        case 1: iq3r_launch_n<S, 1>(vx, a, s); break;
        case 2: iq3r_launch_n<S, 2>(vx, a, s); break;
        case 3: iq3r_launch_n<S, 3>(vx, a, s); break;
        case 4: iq3r_launch_n<S, 4>(vx, a, s); break;
        default: break;
    }
}
void launch_iq3r(bool s_variant, const char * vx, const a16_args & a, int ncols, cudaStream_t s) {
    if (s_variant) iq3r_launch<true>(vx, a, ncols, s); else iq3r_launch<false>(vx, a, ncols, s);
}

} // namespace p100e::fast
