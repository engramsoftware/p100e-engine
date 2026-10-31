#include "matvec_internal.cuh"
#include <algorithm>
#include <cstdlib>

namespace p100e::fast {

#define KQ_ROWS 4
#define KQ_NWARPS 4

template <ggml_type type> struct kq_traits;
template <> struct kq_traits<GGML_TYPE_Q4_K>   { static constexpr int bytes = sizeof(block_q4_K);   };
template <> struct kq_traits<GGML_TYPE_Q5_K>   { static constexpr int bytes = sizeof(block_q5_K);   };
template <> struct kq_traits<GGML_TYPE_IQ4_XS> { static constexpr int bytes = sizeof(block_iq4_xs); };
template <> struct kq_traits<GGML_TYPE_Q6_K>   { static constexpr int bytes = sizeof(block_q6_K);   };

// Position in the activation block of the half2 pair (4i, 4i+2) of a 32-element group; (4i+1, 4i+3) is +2.
static __device__ __forceinline__ constexpr int kq_idx(const int i) {
    return i < 4 ? 4*i : 4*(i - 4) + 1;
}

static __device__ __forceinline__ uint32_t kq_lop(const uint32_t v) {
    uint32_t r;
    asm("lop3.b32 %0, %1, 0x000f000f, 0x64006400, 0xea;" : "=r"(r) : "r"(v));
    return r;
}

// 32 bytes as 8 words, with the widest loads the alignment allows (VA = 4, 8 or 16 bytes).
template <int VA>
static __device__ __forceinline__ void kq_load32(const char * __restrict__ p, uint32_t (&v)[8]) {
    if constexpr (VA == 16) {
        const uint4 a = ((const uint4 *) p)[0], b = ((const uint4 *) p)[1];
        v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w; v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
    } else if constexpr (VA == 8) {
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint2 a = ((const uint2 *) p)[k];
            v[2*k] = a.x; v[2*k + 1] = a.y;
        }
    } else {
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            v[k] = ((const uint32_t *) p)[k];
        }
    }
}

// Decode group g (0..7) of the super-block at blk into w (activation order) plus the float scale and bias:
//     sum_k w_real[k]*a[k] = scale*sum_k w[k]*a[k] + bias*sum_k a[k]
template <ggml_type type, int VA, bool r2 = false>
static __device__ __forceinline__ void kq_decode(
        const char * __restrict__ xr, const int sb, const int g, const int nb, half2 (&w)[16],
        float & scale_e, float & scale_o, float & bias, const uint32_t * lut) {
    if constexpr (type == GGML_TYPE_Q6_K && r2) {
        // q6r2 planes (q4r-sm60.cu): ql2 = 16 bytes per group, word i holds low4 of values 4i..4i+3 in its low
        // nibbles and of 16+4i..16+4i+3 in its high nibbles; qh2 = 8 bytes per group, byte i holds hi2 of 4i+p at
        // bits 2p.  Rebuilds exactly the t0/t1 words of the q6r path below, so the products are bit-identical.
        const int G = 8*sb + g, n = g >> 2, t = g & 3;
        uint32_t q[4], h[2];
        const char * qp = xr + 16*G;
        if constexpr (VA == 16) {
            const uint4 a = *((const uint4 *) qp);
            q[0] = a.x; q[1] = a.y; q[2] = a.z; q[3] = a.w;
        } else if constexpr (VA == 8) {
            const uint2 a = ((const uint2 *) qp)[0], b = ((const uint2 *) qp)[1];
            q[0] = a.x; q[1] = a.y; q[2] = b.x; q[3] = b.y;
        } else {
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                q[k] = ((const uint32_t *) qp)[k];
            }
        }
        const char * hp = xr + 128*nb + 8*G;
        if constexpr (VA >= 8) {
            const uint2 a = *((const uint2 *) hp);
            h[0] = a.x; h[1] = a.y;
        } else {
            h[0] = ((const uint32_t *) hp)[0]; h[1] = ((const uint32_t *) hp)[1];
        }
        const int8_t * scp = (const int8_t *) (xr + 192*nb + 16*sb + 8*n + 2*t);
        const float d = __half2float(*((const half *) (xr + 208*nb + 2*sb)));
        const half2 magic = __float2half2_rn(1056.0f); // 1024 + 32
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const uint32_t vi = i < 4 ? q[i] : q[i - 4] >> 4;
            const uint32_t hb = (h[i >> 2] >> (8*(i & 3))) & 0xFFu;
            const uint32_t t0 = kq_lop(vi)      | ((hb & 3u) << 4)        | (((hb >> 4) & 3u) << 20);
            const uint32_t t1 = kq_lop(vi >> 8) | (((hb >> 2) & 3u) << 4) | (((hb >> 6) & 3u) << 20);
            w[kq_idx(i)    ] = __hsub2(*((const half2 *) &t0), magic);
            w[kq_idx(i) + 2] = __hsub2(*((const half2 *) &t1), magic);
        }
        scale_e = d*scp[0];
        scale_o = d*scp[1];
        bias    = 0.0f;
    } else if constexpr (type == GGML_TYPE_Q6_K) {
        // Repacked row (q4r-sm60.cu): [ql 128*nb][qh 64*nb][scales 16*nb][d 2*nb].  Group g = 32 elements:
        // n = g/4 selects the 128-element half, t = g%4 the quarter.  Quarter t uses the low (t < 2) or high
        // nibble of ql[64n + 32*(t&1) + l] and bits 2t..2t+1 of qh[32n + l], l = 0..31.
        const int n = g >> 2, t = g & 3;
        uint32_t ql[8], qh[8];
        kq_load32<VA>(xr + 128*sb + 64*n + 32*(t & 1), ql);
        kq_load32<VA>(xr + 128*nb + 64*sb + 32*n, qh);
        const int8_t   * scp = (const int8_t   *) (xr + 192*nb + 16*sb + 8*n + 2*t);
        const float d = __half2float(*((const half *) (xr + 208*nb + 2*sb)));
        const int lshift = 4*(t >> 1), hshift = 2*t;
        const half2 magic = __float2half2_rn(1056.0f); // 1024 + 32
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const uint32_t vi = ql[i] >> lshift;
            const uint32_t hi = qh[i] >> hshift;
            const uint32_t t0 = kq_lop(vi)      | ((hi       & 0x00030003u) << 4);
            const uint32_t t1 = kq_lop(vi >> 8) | (((hi >> 8) & 0x00030003u) << 4);
            w[kq_idx(i)    ] = __hsub2(*((const half2 *) &t0), magic);
            w[kq_idx(i) + 2] = __hsub2(*((const half2 *) &t1), magic);
        }
        scale_e = d*scp[0]; // elements 0..15 of the group sit in the even slots (i < 4)
        scale_o = d*scp[1];
        bias    = 0.0f;
    } else if constexpr (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) {
        const char * blk = xr + sb*kq_traits<type>::bytes;
        constexpr int qoff = type == GGML_TYPE_Q4_K ? 16 : 48;
        const float2 dm = __half22float2(*((const half2 *) blk));
        const uint32_t s0 = *((const uint32_t *) (blk + 4));
        const uint32_t s1 = *((const uint32_t *) (blk + 8));
        const uint32_t s2 = *((const uint32_t *) (blk + 12));
        auto qb = [&](const int k) -> int {
            const uint32_t word = k < 4 ? s0 : (k < 8 ? s1 : s2);
            return (word >> (8*(k & 3))) & 0xFF;
        };
        int sc, m;
        if (g < 4) {
            sc = qb(g) & 63;
            m  = qb(g + 4) & 63;
        } else {
            sc = (qb(g + 4) & 0xF) | ((qb(g - 4) >> 6) << 4);
            m  = (qb(g + 4) >>  4) | ((qb(g)     >> 6) << 4);
        }
        const uint4 * qp = (const uint4 *) (blk + qoff + 32*(g >> 1));
        const uint4 qa = qp[0], qc = qp[1];
        uint32_t v[8] = {qa.x, qa.y, qa.z, qa.w, qc.x, qc.y, qc.z, qc.w};
        const int shift = 4*(g & 1);

        uint32_t h[8];
        if constexpr (type == GGML_TYPE_Q5_K) {
            const uint4 * hp = (const uint4 *) (blk + 16);
            const uint4 ha = hp[0], hc = hp[1];
            h[0] = ha.x; h[1] = ha.y; h[2] = ha.z; h[3] = ha.w; h[4] = hc.x; h[5] = hc.y; h[6] = hc.z; h[7] = hc.w;
        }
        constexpr float centre = type == GGML_TYPE_Q4_K ? 8.0f : 16.0f;
        const half2 magic = __float2half2_rn(1024.0f + centre);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const uint32_t vi = v[i] >> shift;
            uint32_t t0 = kq_lop(vi);
            uint32_t t1 = kq_lop(vi >> 8);
            if constexpr (type == GGML_TYPE_Q5_K) {
                t0 |= ((h[i] >>  g     ) & 0x00010001u) << 4;
                t1 |= ((h[i] >> (g + 8)) & 0x00010001u) << 4;
            }
            w[kq_idx(i)    ] = __hsub2(*((const half2 *) &t0), magic);
            w[kq_idx(i) + 2] = __hsub2(*((const half2 *) &t1), magic);
        }
        scale_e = dm.x*sc;
        scale_o = scale_e;
        bias    = centre*scale_e - dm.y*m;
    } else { // IQ4_XS
        const char * blk = xr + sb*kq_traits<type>::bytes;
        const float    d  = __half2float(*((const half *) blk));
        const uint32_t sh = *((const uint16_t *) (blk + 2));
        const uint32_t sl = *((const uint32_t *) (blk + 4));
        const int ls = ((sl >> (4*g)) & 0xF) | (((sh >> (2*g)) & 3) << 4);
        const uint2 * qp = (const uint2 *) (blk + 8 + 16*g);
        const uint2 qa = qp[0], qc = qp[1];
        const uint32_t v[4] = {qa.x, qa.y, qc.x, qc.y};
        // lut[byte] = (kvalues[byte & 15], kvalues[byte >> 4]) as half2: one lookup per quant byte gives the
        // pair of elements (j, j+16), matching the activations rearranged in mul_mat_vec_kq.
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint32_t x = v[k];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const uint32_t e = lut[(x >> (8*i)) & 0xFFu];
                w[4*k + i] = *((const half2 *) &e);
            }
        }
        scale_e = d*(ls - 32);
        scale_o = scale_e;
        bias    = 0.0f;
    }
}

template <ggml_type type, int ncols_dst, int nwarps, int VA, bool r2 = false>
static __global__ void __launch_bounds__(WARP_SIZE*nwarps, 4*(KQ_NWARPS/nwarps))
mul_mat_vec_kq(
        const char * __restrict__ vx, const half2 * __restrict__ aq, const half2 * __restrict__ ads,
        float * __restrict__ dst, const int ngroups, const int row_bytes,
        const int stride_col_aq, const int stride_col_ads, const int stride_col_dst) {
#if defined(FP16_AVAILABLE)
    const int nb       = ngroups/8;
    const int row0     = blockIdx.x*KQ_ROWS;
    const int tid      = threadIdx.x + threadIdx.y*WARP_SIZE;
    const int nthreads = WARP_SIZE*nwarps;

    __shared__ uint32_t lut[type == GGML_TYPE_IQ4_XS ? 256 : 1];
    if constexpr (type == GGML_TYPE_IQ4_XS) {
        for (int i = tid; i < 256; i += nthreads) {
            const half2 e = make_half2(__int2half_rn(kvalues_iq4nl[i & 15]), __int2half_rn(kvalues_iq4nl[i >> 4]));
            lut[i] = *((const uint32_t *) &e);
        }
        __syncthreads();
    }

    float sumf[ncols_dst][KQ_ROWS] = {{0.0f}};
    const char * const xbase = vx + (size_t) row0*row_bytes;

    for (int t = tid; t < ngroups; t += nthreads) {
        const int sb = t >> 3;
        const int g  = t &  7;

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
            if constexpr (type == GGML_TYPE_IQ4_XS) {
                // a16 slots: [4k] = (a[4k], a[4k+2]), [4k+1] = (a[4k+16], a[4k+18]), [4k+2] = (a[4k+1], a[4k+3]),
                // [4k+3] = (a[4k+17], a[4k+19]).  Regroup into (a[m], a[m+16]) at slot m, once per group and
                // shared by the KQ_ROWS rows.
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const half2 p0 = a[j][4*k], p1 = a[j][4*k + 1], p2 = a[j][4*k + 2], p3 = a[j][4*k + 3];
                    a[j][4*k + 0] = __lows2half2 (p0, p1);
                    a[j][4*k + 1] = __lows2half2 (p2, p3);
                    a[j][4*k + 2] = __highs2half2(p0, p1);
                    a[j][4*k + 3] = __highs2half2(p2, p3);
                }
            }
        }

#pragma unroll
        for (int i = 0; i < KQ_ROWS; ++i) {
            half2 w[16];
            float scale_e, scale_o, bias;
            kq_decode<type, VA, r2>(xbase + i*row_bytes, sb, g, nb, w, scale_e, scale_o, bias, lut);
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                half2 se = make_half2(0.0f, 0.0f);
                half2 so = make_half2(0.0f, 0.0f);
#pragma unroll
                for (int c = 0; c < 16; c += 2) {
                    se = __hfma2(w[c    ], a[j][c    ], se);
                    so = __hfma2(w[c + 1], a[j][c + 1], so);
                }
                const float2 fe = __half22float2(se);
                const float2 fo = __half22float2(so);
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
    GGML_UNUSED_VARS(vx, aq, ads, dst, ngroups, row_bytes, stride_col_aq, stride_col_ads, stride_col_dst);
    NO_DEVICE_CODE;
#endif // FP16_AVAILABLE
}

template <ggml_type type, int ncols_dst, int VA, bool r2 = false>
static void kq_launch(const char * vx, const half2 * aq, const half2 * ads, float * dst, int ngroups, int nrows,
        int row_bytes, int stride_col_aq, int stride_col_ads, int stride_col_dst, cudaStream_t stream) {
    const dim3 grid(nrows/KQ_ROWS, 1, 1);
    const int forced = 0;
    // 2 warps at every width: Qwen3.8-27B on 3070+P100, tg128 (nwarps 4 / 2 / 1):
    //   UD-Q5_K_M 15.92 / 16.53 / 15.67, TurboFC (IQ4_XS) 19.54 / 20.65 / 20.29, Q4_0 23.39 / 23.77 / 23.17 t/s.
    // With 4, IQ4_XS measured slower than upstream MMVQ.
    int nw = forced == 1 || forced == 2 || forced == 4 ? forced : 2;
    while (nw > 1 && ngroups < WARP_SIZE*nw) {
        nw /= 2;
    }
    switch (nw) {
        case 1:
            mul_mat_vec_kq<type, ncols_dst, 1, VA, r2><<<grid, dim3(WARP_SIZE, 1, 1), 0, stream>>>(
                vx, aq, ads, dst, ngroups, row_bytes, stride_col_aq, stride_col_ads, stride_col_dst);
            break;
        case 2:
            mul_mat_vec_kq<type, ncols_dst, 2, VA, r2><<<grid, dim3(WARP_SIZE, 2, 1), 0, stream>>>(
                vx, aq, ads, dst, ngroups, row_bytes, stride_col_aq, stride_col_ads, stride_col_dst);
            break;
        default:
            mul_mat_vec_kq<type, ncols_dst, 4, VA, r2><<<grid, dim3(WARP_SIZE, 4, 1), 0, stream>>>(
                vx, aq, ads, dst, ngroups, row_bytes, stride_col_aq, stride_col_ads, stride_col_dst);
            break;
    }
}

template <ggml_type type, int VA, bool r2 = false>
static void kq_dispatch(const char * vx, const half2 * aq, const half2 * ads, float * dst, int ncols, int ngroups,
        int nrows, int row_bytes, int stride_col_aq, int stride_col_ads, int stride_col_dst, cudaStream_t stream) {
    switch (ncols) {
        case 1: kq_launch<type, 1, VA, r2>(vx, aq, ads, dst, ngroups, nrows, row_bytes, stride_col_aq, stride_col_ads, stride_col_dst, stream); break;
        case 2: kq_launch<type, 2, VA, r2>(vx, aq, ads, dst, ngroups, nrows, row_bytes, stride_col_aq, stride_col_ads, stride_col_dst, stream); break;
        case 3: kq_launch<type, 3, VA, r2>(vx, aq, ads, dst, ngroups, nrows, row_bytes, stride_col_aq, stride_col_ads, stride_col_dst, stream); break;
        case 4: kq_launch<type, 4, VA, r2>(vx, aq, ads, dst, ngroups, nrows, row_bytes, stride_col_aq, stride_col_ads, stride_col_dst, stream); break;
        default: break;
    }
}

void launch_q5k(const char * vx, const a16_args & a, int ncols, cudaStream_t s) {
    kq_dispatch<GGML_TYPE_Q5_K, 16>(vx, a.aq, a.ads, a.dst, ncols, a.ngroups, a.nrows, a.row_bytes, a.ngroups*16, a.ngroups, a.nrows, s);
}
void launch_iq4xs(const char * vx, const a16_args & a, int ncols, cudaStream_t s) {
    kq_dispatch<GGML_TYPE_IQ4_XS, 16>(vx, a.aq, a.ads, a.dst, ncols, a.ngroups, a.nrows, a.row_bytes, a.ngroups*16, a.ngroups, a.nrows, s);
}
void launch_q6r2(const char * vx, const a16_args & a, int ncols, int VA, cudaStream_t s) {
    const int sa = a.ngroups*16, sd = a.ngroups, so = a.nrows;
    if (VA == 16)      kq_dispatch<GGML_TYPE_Q6_K, 16, true>(vx, a.aq, a.ads, a.dst, ncols, a.ngroups, a.nrows, a.row_bytes, sa, sd, so, s);
    else if (VA == 8)  kq_dispatch<GGML_TYPE_Q6_K,  8, true>(vx, a.aq, a.ads, a.dst, ncols, a.ngroups, a.nrows, a.row_bytes, sa, sd, so, s);
    else               kq_dispatch<GGML_TYPE_Q6_K,  4, true>(vx, a.aq, a.ads, a.dst, ncols, a.ngroups, a.nrows, a.row_bytes, sa, sd, so, s);
}

} // namespace p100e::fast
