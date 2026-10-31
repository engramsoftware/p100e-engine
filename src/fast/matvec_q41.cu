// Q4_1 GEMV on the a16 activations: port of mul_mat_vec_q4_1_a16 from our fork's mmvq-f16-sm60.cu
// (patch 12, MIT). Weights are the untouched GGUF block_q4_1 rows (LAYOUT_KQRAW); nibbles are expanded to half2 with
// LOP3 magic constants (fp16 0x6400 = 1024 takes a nibble in mantissa bits 0-3 -> 1024 + q, 0x5400 = 64 takes one in
// bits 4-7 -> 64 + q; subtracting 1024 / 64 is exact), multiplied with the a16 activations by HFMA2, and the block's
// scale and min are applied once per block in float: y += yd * (d * af + m * sum/2), where yd = the activation
// group's amax, af = the fp16 dot product of the half block, sum/2 = half the group's normalized sum (each block is
// split over two threads, kh = 0 / 1).
// One change against the fork: the warp count does not depend on the column count (the fork used 4 warps for one
// column and 2 for more), so column j of a T-column call is bit-identical to the T = 1 call on that column (the MTP
// verify compares verify rows with plain decode bit for bit).
#include "matvec_internal.cuh"
#include <algorithm>

namespace p100e::fast {

#define Q41_ROWS 4
#define Q41_QK 32
#define Q41_NWARPS 4

template <int ncols_dst, int nwarps>
static __global__ void __launch_bounds__(WARP_SIZE*nwarps, (ncols_dst >= 6 ? 2 : 4)*(Q41_NWARPS/nwarps))
mul_mat_vec_q4_1_a16(
        const char * __restrict__ vx, const half2 * __restrict__ aq, const half2 * __restrict__ ads,
        float * __restrict__ dst, const int nblocks, const int row_bytes,
        const int stride_col_aq, const int stride_col_ads, const int stride_col_dst) {
    const int row0     = blockIdx.x*Q41_ROWS;
    const int tid      = threadIdx.x + threadIdx.y*WARP_SIZE;
    const int nthreads = WARP_SIZE*nwarps;

    const half2 magic_lo = __float2half2_rn(1024.0f);
    const half2 magic_hi = __float2half2_rn(  64.0f);

    float sumf[ncols_dst][Q41_ROWS] = {{0.0f}};

    // block_q4_1 is 20 bytes: per-row base pointers plus a byte offset built once from kb (no per-row multiplies)
    const char * const xbase = vx + (size_t) row0*row_bytes;

    for (int t = tid; t < 2*nblocks; t += nthreads) {
        const int kb = t >> 1;
        const int kh = t &  1;
        const int koff = kb*(int) sizeof(block_q4_1);

        half2 a[ncols_dst][8];
        float yd[ncols_dst];
        float ys[ncols_dst];
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
            const half2 * ap = aq + (size_t) j*stride_col_aq + kb*(Q41_QK/2) + kh*8;
#pragma unroll
            for (int c2 = 0; c2 < 2; ++c2) {
                const int4 av = *((const int4 *) (ap + 4*c2));
                a[j][4*c2 + 0] = *((const half2 *) &av.x);
                a[j][4*c2 + 1] = *((const half2 *) &av.y);
                a[j][4*c2 + 2] = *((const half2 *) &av.z);
                a[j][4*c2 + 3] = *((const half2 *) &av.w);
            }
            const half2 ds = ads[j*stride_col_ads + kb];
            yd[j] = __low2float(ds);
            ys[j] = __high2float(ds)*0.5f;
        }

#pragma unroll
        for (int i = 0; i < Q41_ROWS; ++i) {
            const block_q4_1 * b = (const block_q4_1 *) (xbase + (size_t) i*row_bytes + koff);
            const float2 dm = __half22float2(b->dm);
            const int * wq = (const int *) b->qs;

            half2 w[8];
#pragma unroll
            for (int k = 0; k < 2; ++k) {
                const int w32  = wq[2*kh + k];
                const int w32s = w32 >> 8;
                int t0, t1, t2, t3;
                asm("lop3.b32 %0, %1, %2, %3, 0xea;" : "=r"(t0) : "r"(w32),  "n"(0x000f000f), "n"(0x64006400));
                asm("lop3.b32 %0, %1, %2, %3, 0xea;" : "=r"(t1) : "r"(w32),  "n"(0x00f000f0), "n"(0x54005400));
                asm("lop3.b32 %0, %1, %2, %3, 0xea;" : "=r"(t2) : "r"(w32s), "n"(0x000f000f), "n"(0x64006400));
                asm("lop3.b32 %0, %1, %2, %3, 0xea;" : "=r"(t3) : "r"(w32s), "n"(0x00f000f0), "n"(0x54005400));
                w[4*k + 0] = __hsub2(*((const half2 *) &t0), magic_lo);
                w[4*k + 1] = __hsub2(*((const half2 *) &t1), magic_hi);
                w[4*k + 2] = __hsub2(*((const half2 *) &t2), magic_lo);
                w[4*k + 3] = __hsub2(*((const half2 *) &t3), magic_hi);
            }

#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                half2 s = make_half2(0.0f, 0.0f);
#pragma unroll
                for (int c2 = 0; c2 < 2; ++c2) {
#pragma unroll
                    for (int c = 0; c < 4; ++c) {
                        s = __hfma2(w[4*c2 + c], a[j][4*c2 + c], s);
                    }
                }
                const float af = __half2float(__hadd(__low2half(s), __high2half(s)));
                sumf[j][i] += yd[j]*(dm.x*af + dm.y*ys[j]);
            }
        }
    }

    if constexpr (nwarps == 1) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < Q41_ROWS; ++i) {
                const float v = warp_reduce_sum(sumf[j][i]);
                if (threadIdx.x == i) {
                    dst[j*stride_col_dst + row0 + i] = v;
                }
            }
        }
    } else {
        __shared__ float tmp[nwarps][ncols_dst][Q41_ROWS];
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < Q41_ROWS; ++i) {
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
#pragma unroll
            for (int i = 0; i < Q41_ROWS; ++i) {
                if (threadIdx.x == i) {
                    float s = 0.0f;
#pragma unroll
                    for (int w = 0; w < nwarps; ++w) {
                        s += tmp[w][j][i];
                    }
                    dst[j*stride_col_dst + row0 + i] = s;
                }
            }
        }
    }
}

template <int ncols_dst>
static void q41_launch(const char * vx, const half2 * aq, const half2 * ads, float * dst, int nblocks, int nrows,
        int row_bytes, int stride_col_aq, int stride_col_ads, int stride_col_dst, cudaStream_t stream) {
    const dim3 grid(nrows/Q41_ROWS, 1, 1);
    int nw = Q41_NWARPS;   // independent of ncols_dst: T-invariant rows (see the file comment)
    while (nw > 1 && 2*nblocks < WARP_SIZE*nw) {
        nw /= 2;
    }
    switch (nw) {
        case 1:
            mul_mat_vec_q4_1_a16<ncols_dst, 1><<<grid, dim3(WARP_SIZE, 1, 1), 0, stream>>>(
                vx, aq, ads, dst, nblocks, row_bytes, stride_col_aq, stride_col_ads, stride_col_dst);
            break;
        case 2:
            mul_mat_vec_q4_1_a16<ncols_dst, 2><<<grid, dim3(WARP_SIZE, 2, 1), 0, stream>>>(
                vx, aq, ads, dst, nblocks, row_bytes, stride_col_aq, stride_col_ads, stride_col_dst);
            break;
        default:
            mul_mat_vec_q4_1_a16<ncols_dst, 4><<<grid, dim3(WARP_SIZE, 4, 1), 0, stream>>>(
                vx, aq, ads, dst, nblocks, row_bytes, stride_col_aq, stride_col_ads, stride_col_dst);
            break;
    }
}

void launch_q41(const char * vx, const a16_args & a, int ncols, cudaStream_t s) {
    const int sa = a.ngroups*16, sd = a.ngroups, so = a.nrows;
    switch (ncols) {
#define C(N) case N: q41_launch<N>(vx, a.aq, a.ads, a.dst, a.ngroups, a.nrows, a.row_bytes, sa, sd, so, s); break;
        C(1) C(2) C(3) C(4) C(5) C(6) C(7) C(8)
#undef C
        default: break;
    }
}

} // namespace p100e::fast
