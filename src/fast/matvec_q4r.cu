#include "matvec_internal.cuh"
#include <algorithm>
#include <cstdlib>

namespace p100e::fast {

#define Q4R_ROWS 4
#define Q4R_QK 32
#define Q4R_NWARPS 4

// Same activation format and LOP3 half2 expansion as mul_mat_vec_q4_0_a16 in mmvq-f16-sm60.cu; the only
// difference is the weight load: one aligned 8-byte load per thread per row instead of three 4-byte loads
// and two funnel shifts, plus a scale read from the row's scale tail.
// Kernel body (one block = Q4R_ROWS rows at row0 of the row set vx), shared by the plain and the grouped expert launch:
// same source, so the grouped result is bitwise the plain one.
template <int ncols_dst, int nwarps>
static __device__ __forceinline__ void q4r_body(
        const char * __restrict__ vx, const half2 * __restrict__ aq, const half2 * __restrict__ ads,
        float * __restrict__ dst, const int nblocks, const int row_bytes,
        const int stride_col_aq, const int stride_col_ads, const int stride_col_dst, const int row0) {
#if defined(FP16_AVAILABLE)
    const int tid      = threadIdx.x + threadIdx.y*WARP_SIZE;
    const int nthreads = WARP_SIZE*nwarps;

    const half2 magic_lo = __float2half2_rn(1032.0f);
    const half2 magic_hi = __float2half2_rn(  72.0f);

    float sumf[ncols_dst][Q4R_ROWS] = {{0.0f}};

    const char * const xbase = vx + (size_t) row0*row_bytes;
    const int dtail = 16*nblocks;

    for (int t = tid; t < 2*nblocks; t += nthreads) {
        const int kb = t >> 1;
        const int kh = t &  1;

        half2 a[ncols_dst][8];
        float yd[ncols_dst];
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
            const half2 * ap = aq + (size_t) j*stride_col_aq + kb*(Q4R_QK/2) + kh*8;
#pragma unroll
            for (int c2 = 0; c2 < 2; ++c2) {
                const int4 av = *((const int4 *) (ap + 4*c2));
                a[j][4*c2 + 0] = *((const half2 *) &av.x);
                a[j][4*c2 + 1] = *((const half2 *) &av.y);
                a[j][4*c2 + 2] = *((const half2 *) &av.z);
                a[j][4*c2 + 3] = *((const half2 *) &av.w);
            }
            yd[j] = __low2float(ads[j*stride_col_ads + kb]);
        }

#pragma unroll
        for (int i = 0; i < Q4R_ROWS; ++i) {
            const char * xr = xbase + i*row_bytes;
            const uint2 q = *((const uint2 *) (xr + 16*kb + 8*kh));
            const float d = __half2float(*((const half *) (xr + dtail + 2*kb)));

            half2 w[8];
#pragma unroll
            for (int k = 0; k < 2; ++k) {
                const int w32  = k == 0 ? (int) q.x : (int) q.y;
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
                for (int c = 0; c < 8; ++c) {
                    s = __hfma2(w[c], a[j][c], s);
                }
                const float af = __half2float(__hadd(__low2half(s), __high2half(s)));
                sumf[j][i] += yd[j]*d*af;
            }
        }
    }

    if constexpr (nwarps == 1) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < Q4R_ROWS; ++i) {
                const float v = warp_reduce_sum(sumf[j][i]);
                if (threadIdx.x == i) {
                    dst[j*stride_col_dst + row0 + i] = v;
                }
            }
        }
    } else {
        __shared__ float tmp[nwarps][ncols_dst][Q4R_ROWS];
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < Q4R_ROWS; ++i) {
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
            if (threadIdx.x < Q4R_ROWS) {
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
    GGML_UNUSED_VARS(vx, aq, ads, dst, nblocks, row_bytes, stride_col_aq, stride_col_ads, stride_col_dst, row0);
    NO_DEVICE_CODE;
#endif // FP16_AVAILABLE
}

template <int ncols_dst, int nwarps>
static __global__ void __launch_bounds__(WARP_SIZE*nwarps, (ncols_dst >= 6 ? 2 : 4)*(Q4R_NWARPS/nwarps))
mul_mat_vec_q4r(
        const char * __restrict__ vx, const half2 * __restrict__ aq, const half2 * __restrict__ ads,
        float * __restrict__ dst, const int nblocks, const int row_bytes,
        const int stride_col_aq, const int stride_col_ads, const int stride_col_dst) {
    q4r_body<ncols_dst, nwarps>(vx, aq, ads, dst, nblocks, row_bytes, stride_col_aq, stride_col_ads, stride_col_dst, blockIdx.x*Q4R_ROWS);
}

// grouped expert launch (MoE): blockIdx.y = group of the device table (matvec_internal.cuh mv_group), one column per group
template <int nwarps, int NC = 1>
static __global__ void __launch_bounds__(WARP_SIZE*nwarps, (NC >= 6 ? 2 : 4)*(Q4R_NWARPS/nwarps))
mul_mat_vec_q4r_grouped(
        const mv_group * __restrict__ tab, const half2 * __restrict__ aq, const half2 * __restrict__ ads,
        float * __restrict__ dst, const int nblocks, const int row_bytes,
        const int stride_col_aq, const int stride_col_ads, const int stride_col_dst) {
    const mv_group gr = tab[blockIdx.y];
    q4r_body<NC, nwarps>((const char *) gr.w, aq + (size_t) gr.col*stride_col_aq, ads + (size_t) gr.col*stride_col_ads,
                         dst + (size_t) gr.out*stride_col_dst, nblocks, row_bytes, stride_col_aq, stride_col_ads, stride_col_dst,
                         blockIdx.x*Q4R_ROWS);
}

// G groups in one launch: grid (nrows / Q4R_ROWS, G); nwarps as q4r_launch<1> picks
// NC consecutive a16 columns per group: nwarps from ngroups only (as the NC = 1 grouped launch), so a column's bits never
// depend on NC (the plain launch_q4r picks nwarps by NC, which is why its NC = 4 sums differ; the grouped path does not)
template <int NC>
static void q4r_grouped_n(const mv_group * tab, int G, const a16_args & a, cudaStream_t s) {
    const dim3 grid(a.nrows/Q4R_ROWS, G, 1);
    const int sa = a.ngroups*16, sd = a.ngroups, so = a.nrows;
    int nw = 4;
    while (nw > 1 && 2*a.ngroups < WARP_SIZE*nw) nw /= 2;
    if (nw == 1)      mul_mat_vec_q4r_grouped<1, NC><<<grid, dim3(WARP_SIZE, 1, 1), 0, s>>>(tab, a.aq, a.ads, a.dst, a.ngroups, a.row_bytes, sa, sd, so);
    else if (nw == 2) mul_mat_vec_q4r_grouped<2, NC><<<grid, dim3(WARP_SIZE, 2, 1), 0, s>>>(tab, a.aq, a.ads, a.dst, a.ngroups, a.row_bytes, sa, sd, so);
    else              mul_mat_vec_q4r_grouped<4, NC><<<grid, dim3(WARP_SIZE, 4, 1), 0, s>>>(tab, a.aq, a.ads, a.dst, a.ngroups, a.row_bytes, sa, sd, so);
}
void launch_q4r_grouped(const mv_group * tab, int G, const a16_args & a, cudaStream_t s) { q4r_grouped_n<1>(tab, G, a, s); }
void launch_q4r_grouped_nc(const mv_group * tab, int G, const a16_args & a, int nc, cudaStream_t s) {
    switch (nc) {
        case 1: q4r_grouped_n<1>(tab, G, a, s); break;
        case 2: q4r_grouped_n<2>(tab, G, a, s); break;
        case 3: q4r_grouped_n<3>(tab, G, a, s); break;
        default: q4r_grouped_n<4>(tab, G, a, s); break;
    }
}

template <int ncols_dst>
static void q4r_launch(const char * vx, const half2 * aq, const half2 * ads, float * dst, int nblocks, int nrows,
        int row_bytes, int stride_col_aq, int stride_col_ads, int stride_col_dst, cudaStream_t stream) {
    const dim3 grid(nrows/Q4R_ROWS, 1, 1);
    // The split of a row over threads and the cross-warp sum depend on nwarps. Default (P100E_MV_Q4R_TINV unset / 0):
    // 4 warps at T = 1, 2 warps for T >= 2 (columns at T >= 2 differ from
    // T = 1 in the low bits). P100E_MV_Q4R_TINV=1: 4 warps for NC <= 4 too, so the MTP verify columns equal the T = 1
    // call bit for bit, at the same speed (3070: 5120 x 5120 T = 2 / 4 42.6 / 44.5 -> 42.2 / 44.3 us). =2: 4 warps for
    // every NC (T-invariant batched decode too; costs 12-25% at T = 8: 56 -> 63, 162 -> 187, 169 -> 211 us, the 2.5
    // groups per thread of a 5120 row leave threads idle). Read once, at the first launch.
    static const int tinv = getenv("P100E_MV_Q4R_TINV") ? atoi(getenv("P100E_MV_Q4R_TINV")) : 0;
    int nw = ncols_dst == 1 || (tinv >= 1 && ncols_dst <= 4) || tinv >= 2 ? Q4R_NWARPS : 2;
    while (nw > 1 && 2*nblocks < WARP_SIZE*nw) {
        nw /= 2;
    }
    switch (nw) {
        case 1:
            mul_mat_vec_q4r<ncols_dst, 1><<<grid, dim3(WARP_SIZE, 1, 1), 0, stream>>>(
                vx, aq, ads, dst, nblocks, row_bytes, stride_col_aq, stride_col_ads, stride_col_dst);
            break;
        case 2:
            mul_mat_vec_q4r<ncols_dst, 2><<<grid, dim3(WARP_SIZE, 2, 1), 0, stream>>>(
                vx, aq, ads, dst, nblocks, row_bytes, stride_col_aq, stride_col_ads, stride_col_dst);
            break;
        default:
            mul_mat_vec_q4r<ncols_dst, 4><<<grid, dim3(WARP_SIZE, 4, 1), 0, stream>>>(
                vx, aq, ads, dst, nblocks, row_bytes, stride_col_aq, stride_col_ads, stride_col_dst);
            break;
    }
}

void launch_q4r(const char * vx, const a16_args & a, int ncols, cudaStream_t s) {
    const int sa = a.ngroups*16, sd = a.ngroups, so = a.nrows;
    // q4r_launch takes nblocks = n_in/32 (= ngroups)
    switch (ncols) {
#define C(N) case N: q4r_launch<N>(vx, a.aq, a.ads, a.dst, a.ngroups, a.nrows, a.row_bytes, sa, sd, so, s); break;
        C(1) C(2) C(3) C(4) C(5) C(6) C(7) C(8)
#undef C
        default: break;
    }
}

} // namespace p100e::fast
