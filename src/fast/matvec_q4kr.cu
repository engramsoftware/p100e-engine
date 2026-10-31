#include "matvec_internal.cuh"
#include <algorithm>
#include <cstdlib>

namespace p100e::fast {

#define KQ_ROWS 4
#define KQ_NWARPS 4

// ---------------------------------------------------------------- q4kr: Q4_K in the repacked layout of q4r-sm60.cu
//
// Row layout (GGML_KQR_Q4K=1; same bytes and row stride 144*nb, built by q4kr_convert in q4r-sm60.cu):
//     [ qs plane : 128*nb bytes ][ header plane : 16*nb bytes ]
// Group g (0..7, 32 values) of super-block sb is the 16 bytes at 16*t, t = 8*sb + g (= the a16 block index): byte i
// holds value i of the group in its low nibble and value i + 16 in its high nibble (the Q4_0 order), so one
// 128-bit load gives exactly the group and q4r's LOP3 expansion produces the a16 slot order directly.  The header of
// sb (d, dmin, scales[12], the original 16 bytes) is at 128*nb + 16*sb.
//
// Work mapping, activation slots, fp16 dot order (se: even slots, so: odd slots) and the fp32 epilogue and reductions
// are those of mul_mat_vec_kq<Q4_K>, and the dequantized half2 values are the same exact integers q - 8, so at the
// same nwarps the output is bit-identical to it.  What changes is the load and decode work per (row, group): one
// 16-byte qs load (kq: two 16-byte loads, half of it discarded) and one 16-byte header load (kq: four 4-byte loads),
// and a branch-free scale/min decode (two PRMT and four LOP3 into an exact half2 (sc, m); kq: a g < 4 branch and
// two I2F).
// minb = min blocks per SM for __launch_bounds__ (0 = the kq default 4*(KQ_NWARPS/nwarps)).  It only changes the
// register cap and therefore ptxas's load scheduling and occupancy, not the FP operations, so every value is
// bit-identical (IMPL-REVIEW-1 N1(a)/N5; selected at run time with GGML_KQR_MINB, see q4kr_launch).
// The kernel body (one block = KQ_ROWS rows at row0 of the row set vx): shared by the plain launch and the grouped
// expert launch below; the arithmetic is the same source in both, so the grouped result is bitwise the plain one.
template <int ncols_dst, int nwarps>
static __device__ __forceinline__ void q4kr_body(
        const char * __restrict__ vx, const half2 * __restrict__ aq, const half2 * __restrict__ ads,
        float * __restrict__ dst, const int ngroups, const int row_bytes,
        const int stride_col_aq, const int stride_col_ads, const int stride_col_dst, const int row0) {
#if defined(FP16_AVAILABLE)
    const int nb       = ngroups/8;
    const int tid      = threadIdx.x + threadIdx.y*WARP_SIZE;
    const int nthreads = WARP_SIZE*nwarps;

    // nthreads is a multiple of 8, so g = t & 7 is the same for every t this thread visits: the scale/min selectors
    // of get_scale_min_k4 (ggml-quants.c) are set once.  With b = g & 3 and s0, s1, s2 = scale bytes 0-3, 4-7, 8-11:
    //   g < 4:  sc = s0[b] & 63,                        m = s1[b] & 63
    //   g >= 4: sc = (s2[b] & 15) | (s0[b] >> 6) << 4,  m = (s2[b] >> 4) | (s1[b] >> 6) << 4
    // Q = (s0[b], s0[b], s1[b], s1[b]) and V = s2[b] in every byte; X gathers sc into the low and m into the high
    // half of a 1024-biased half2, which HSUB2 turns into the exact (sc, m).
    const int      g    = tid & 7;
    const uint32_t selV = (g & 3)*0x1111u;   // every byte <- byte b
    const uint32_t selQ = selV + 0x4400u;    // bytes <- (b, b, 4 + b, 4 + b)
    const bool     ghi  = g >= 4;

    const half2 magic_lo = __float2half2_rn(1032.0f); // 1024 + 8: low-nibble halves 1024 + q  -> q - 8
    const half2 magic_hi = __float2half2_rn(  72.0f); //   64 + 8: high-nibble halves  64 + q  -> q - 8
    const half2 magic_sc = __float2half2_rn(1024.0f);

    float sumf[ncols_dst][KQ_ROWS] = {{0.0f}};
    const char * const xbase = vx + (size_t) row0*row_bytes;
    const int htail = 128*nb;

    for (int t = tid; t < ngroups; t += nthreads) {
        const int sb = t >> 3;

        half2 a[ncols_dst][16];
        float yd[ncols_dst], ys[ncols_dst];
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
            const float2 ds = __half22float2(ads[j*stride_col_ads + t]);
            yd[j] = ds.x;
            ys[j] = ds.y;
        }

#pragma unroll
        for (int i = 0; i < KQ_ROWS; ++i) {
            const char * xr = xbase + i*row_bytes;
            const uint4 q = *((const uint4 *) (xr + 16*t));
            const uint4 h = *((const uint4 *) (xr + htail + 16*sb));

            const uint32_t Q = __byte_perm(h.y, h.z, selQ);
            const uint32_t V = __byte_perm(h.w, h.w, selV);
            const uint32_t X = ghi ? (((Q >> 2) & 0x00300030u) | (V & 0x0000000Fu) | ((V >> 4) & 0x000F0000u) | 0x64006400u)
                                   : ((Q & 0x003F003Fu) | 0x64006400u);
            const float2 scm = __half22float2(__hsub2(*((const half2 *) &X), magic_sc));
            const float2 dm  = __half22float2(*((const half2 *) &h.x));

            // word k of the group: bytes 4k..4k+3 = values 4k..4k+3 (low nibbles) and 4k+16..4k+19 (high nibbles)
            //   slot 4k+0 = (v[4k],    v[4k+2]),  slot 4k+1 = (v[4k+16], v[4k+18]),
            //   slot 4k+2 = (v[4k+1],  v[4k+3]),  slot 4k+3 = (v[4k+17], v[4k+19])  -- kq's w[] order.
            // Each word is consumed right away (fewer live registers); se still sees slots 0, 2, ..., 14 and so
            // slots 1, 3, ..., 15 in kq's order, so every fp16 partial sum is the same.
            half2 se[ncols_dst], so[ncols_dst];
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                se[j] = make_half2(0.0f, 0.0f);
                so[j] = make_half2(0.0f, 0.0f);
            }
            const uint32_t qw[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const uint32_t w32  = qw[k];
                const uint32_t w32s = qw[k] >> 8;
                uint32_t t0, t1, t2, t3;
                asm("lop3.b32 %0, %1, %2, %3, 0xea;" : "=r"(t0) : "r"(w32),  "n"(0x000f000f), "n"(0x64006400));
                asm("lop3.b32 %0, %1, %2, %3, 0xea;" : "=r"(t1) : "r"(w32),  "n"(0x00f000f0), "n"(0x54005400));
                asm("lop3.b32 %0, %1, %2, %3, 0xea;" : "=r"(t2) : "r"(w32s), "n"(0x000f000f), "n"(0x64006400));
                asm("lop3.b32 %0, %1, %2, %3, 0xea;" : "=r"(t3) : "r"(w32s), "n"(0x00f000f0), "n"(0x54005400));
                const half2 w0 = __hsub2(*((const half2 *) &t0), magic_lo);
                const half2 w1 = __hsub2(*((const half2 *) &t1), magic_hi);
                const half2 w2 = __hsub2(*((const half2 *) &t2), magic_lo);
                const half2 w3 = __hsub2(*((const half2 *) &t3), magic_hi);
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    se[j] = __hfma2(w0, a[j][4*k + 0], se[j]);
                    so[j] = __hfma2(w1, a[j][4*k + 1], so[j]);
                    se[j] = __hfma2(w2, a[j][4*k + 2], se[j]);
                    so[j] = __hfma2(w3, a[j][4*k + 3], so[j]);
                }
            }

            // Same float expressions as kq_decode<Q4_K> + mul_mat_vec_kq (sc, m exact either way), so the compiler
            // contracts them the same way (checked in SASS).
            const float sc = scm.x;
            const float m  = scm.y;
            float scale_e, scale_o, bias;
            scale_e = dm.x*sc;
            scale_o = scale_e;
            bias    = 8.0f*scale_e - dm.y*m;
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                const float2 fe = __half22float2(se[j]);
                const float2 fo = __half22float2(so[j]);
                sumf[j][i] += yd[j]*(scale_e*(fe.x + fe.y) + scale_o*(fo.x + fo.y) + bias*ys[j]);
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
    GGML_UNUSED_VARS(vx, aq, ads, dst, ngroups, row_bytes, stride_col_aq, stride_col_ads, stride_col_dst, row0);
    NO_DEVICE_CODE;
#endif // FP16_AVAILABLE
}

template <int ncols_dst, int nwarps, int minb = 0>
static __global__ void __launch_bounds__(WARP_SIZE*nwarps, minb > 0 ? minb : 4*(KQ_NWARPS/nwarps))
mul_mat_vec_q4kr(
        const char * __restrict__ vx, const half2 * __restrict__ aq, const half2 * __restrict__ ads,
        float * __restrict__ dst, const int ngroups, const int row_bytes,
        const int stride_col_aq, const int stride_col_ads, const int stride_col_dst) {
    q4kr_body<ncols_dst, nwarps>(vx, aq, ads, dst, ngroups, row_bytes, stride_col_aq, stride_col_ads, stride_col_dst, blockIdx.x*KQ_ROWS);
}

// grouped expert launch (MoE): blockIdx.y = group g of the device table: expert row set tab[g].w, a16 column tab[g].col,
// output column tab[g].out (matvec_internal.cuh mv_group); one column per group
template <int nwarps, int minb = 0, int NC = 1>
static __global__ void __launch_bounds__(WARP_SIZE*nwarps, minb > 0 ? minb : 4*(KQ_NWARPS/nwarps))
mul_mat_vec_q4kr_grouped(
        const mv_group * __restrict__ tab, const half2 * __restrict__ aq, const half2 * __restrict__ ads,
        float * __restrict__ dst, const int ngroups, const int row_bytes,
        const int stride_col_aq, const int stride_col_ads, const int stride_col_dst) {
    const mv_group gr = tab[blockIdx.y];
    q4kr_body<NC, nwarps>((const char *) gr.w, aq + (size_t) gr.col*stride_col_aq, ads + (size_t) gr.col*stride_col_ads,
                          dst + (size_t) gr.out*stride_col_dst, ngroups, row_bytes, stride_col_aq, stride_col_ads, stride_col_dst,
                          blockIdx.x*KQ_ROWS);
}

template <int ncols_dst>
static void q4kr_launch_n(const char * vx, const a16_args & a, cudaStream_t stream) {
    const dim3 grid(a.nrows/KQ_ROWS, 1, 1);
    const int sa = a.ngroups*16, sd = a.ngroups, so = a.nrows;
    int nw = 2;
    while (nw > 1 && a.ngroups < WARP_SIZE*nw) nw /= 2;
    if (nw == 1) {
        mul_mat_vec_q4kr<ncols_dst, 1><<<grid, dim3(WARP_SIZE, 1, 1), 0, stream>>>(vx, a.aq, a.ads, a.dst, a.ngroups, a.row_bytes, sa, sd, so);
    } else if (ncols_dst == 4) {   // fork default for width 4: launch bounds minb = 6 (no spill)
        mul_mat_vec_q4kr<ncols_dst, 2, 6><<<grid, dim3(WARP_SIZE, 2, 1), 0, stream>>>(vx, a.aq, a.ads, a.dst, a.ngroups, a.row_bytes, sa, sd, so);
    } else {
        mul_mat_vec_q4kr<ncols_dst, 2><<<grid, dim3(WARP_SIZE, 2, 1), 0, stream>>>(vx, a.aq, a.ads, a.dst, a.ngroups, a.row_bytes, sa, sd, so);
    }
}
// G groups in one launch: grid (nrows / KQ_ROWS, G); nwarps as the plain launch picks for this ngroups
// NC consecutive a16 columns per group (an expert's routed rows): nwarps from ngroups only, so every column's bits equal
// the NC = 1 launch (the fork's width-4 minb = 6 launch bound keeps NC = 4 spill-free)
template <int NC>
static void q4kr_grouped_n(const mv_group * tab, int G, const a16_args & a, cudaStream_t s) {
    const dim3 grid(a.nrows/KQ_ROWS, G, 1);
    const int sa = a.ngroups*16, sd = a.ngroups, so = a.nrows;
    int nw = 2;
    while (nw > 1 && a.ngroups < WARP_SIZE*nw) nw /= 2;
    if (nw == 1)           mul_mat_vec_q4kr_grouped<1, 0, NC><<<grid, dim3(WARP_SIZE, 1, 1), 0, s>>>(tab, a.aq, a.ads, a.dst, a.ngroups, a.row_bytes, sa, sd, so);
    else if (NC == 4)      mul_mat_vec_q4kr_grouped<2, 6, NC><<<grid, dim3(WARP_SIZE, 2, 1), 0, s>>>(tab, a.aq, a.ads, a.dst, a.ngroups, a.row_bytes, sa, sd, so);
    else                   mul_mat_vec_q4kr_grouped<2, 0, NC><<<grid, dim3(WARP_SIZE, 2, 1), 0, s>>>(tab, a.aq, a.ads, a.dst, a.ngroups, a.row_bytes, sa, sd, so);
}
void launch_q4kr_grouped(const mv_group * tab, int G, const a16_args & a, cudaStream_t s) { q4kr_grouped_n<1>(tab, G, a, s); }
void launch_q4kr_grouped_nc(const mv_group * tab, int G, const a16_args & a, int nc, cudaStream_t s) {
    switch (nc) {
        case 1: q4kr_grouped_n<1>(tab, G, a, s); break;
        case 2: q4kr_grouped_n<2>(tab, G, a, s); break;
        case 3: q4kr_grouped_n<3>(tab, G, a, s); break;
        default: q4kr_grouped_n<4>(tab, G, a, s); break;
    }
}

void launch_q4kr(const char * vx, const a16_args & a, int ncols, cudaStream_t s) {
    switch (ncols) {
        case 1: q4kr_launch_n<1>(vx, a, s); break;
        case 2: q4kr_launch_n<2>(vx, a, s); break;
        case 3: q4kr_launch_n<3>(vx, a, s); break;
        case 4: q4kr_launch_n<4>(vx, a, s); break;
        default: break;
    }
}

} // namespace p100e::fast
