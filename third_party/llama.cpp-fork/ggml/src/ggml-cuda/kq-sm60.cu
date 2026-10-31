// Q4_K / Q5_K / IQ4_XS matrix-vector products for sm_60 (GP100) on HFMA2.
//
// Same idea as the Q4_0 kernels (mmvq-f16-sm60.cu, q4r-sm60.cu): GP100 has no DP4A, so upstream MMVQ emulates
// int8 dot products with vmad and ends up far from the memory bandwidth it could use.  Token generation on
// P100 with a 7B model quantized purely to each type (llama-bench tg128, bytes read per token x t/s):
//     Q4_0 (after q4r)  93.5 t/s  ~372 GB/s
//     IQ4_XS            74.1      ~279
//     Q5_K              52.5      ~256
//     Q4_K              62.0      ~247
//     Q6_K              37.2      ~216
//
// Here every thread owns one 32-element group of one super-block and expands its quants to exact half2
// values with LOP3 magic constants (K-quants) or a byte-permute table lookup (IQ4_XS), then accumulates
// against the half2 activations of mmvq-f16-sm60.cu with HFMA2.  Scales and mins are applied once per group
// in float.  The activation format pairs elements (m, m+2) of a 32-block; the K-quants' "32 consecutive
// bytes, one nibble plane" groups map onto that with a fixed index permutation (kq_idx below), IQ4_XS has
// the same byte layout as Q4_0.
//
// Numerics: quants are centred before the fp16 dot product (q-8 for Q4_K, q-16 for Q5_K) and the offset is
// folded into the float bias term, so the fp16 partial sums stay small and cancellation against the mins
// happens in float.  Activations are the block-amax-scaled halves, so |a| <= 1.
//
// Widths 1..4; wider products use the normal path.  GGML_KQ=0 disables the whole file.

#include "kq-sm60.cuh"
#include "q4r-sm60.cuh"
#include "mmvq-f16-sm60.cuh"
#include "vecdotq.cuh"

#include <algorithm>
#include <cstdlib>

#define KQ_ROWS   4
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
    static const int forced = getenv("GGML_KQ_NWARPS") ? atoi(getenv("GGML_KQ_NWARPS")) : 0;
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
        default: GGML_ABORT("kq: unsupported ncols");
    }
}

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
template <int ncols_dst, int nwarps, int minb = 0>
static __global__ void __launch_bounds__(WARP_SIZE*nwarps, minb > 0 ? minb : 4*(KQ_NWARPS/nwarps))
mul_mat_vec_q4kr(
        const char * __restrict__ vx, const half2 * __restrict__ aq, const half2 * __restrict__ ads,
        float * __restrict__ dst, const int ngroups, const int row_bytes,
        const int stride_col_aq, const int stride_col_ads, const int stride_col_dst) {
#if defined(FP16_AVAILABLE)
    const int nb       = ngroups/8;
    const int row0     = blockIdx.x*KQ_ROWS;
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
            // contracts them the same way (checked in SASS, see IMPL.md stage P2).
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
    GGML_UNUSED_VARS(vx, aq, ads, dst, ngroups, row_bytes, stride_col_aq, stride_col_ads, stride_col_dst);
    NO_DEVICE_CODE;
#endif // FP16_AVAILABLE
}

// Same grid and nwarps choice as kq_launch (including GGML_KQ_NWARPS), which the bit-identity with kq relies on:
// the warp count decides which thread sums which groups.
template <int ncols_dst>
static void q4kr_launch(const char * vx, const half2 * aq, const half2 * ads, float * dst, int ngroups, int nrows,
        int row_bytes, int stride_col_aq, int stride_col_ads, int stride_col_dst, cudaStream_t stream) {
    const dim3 grid(nrows/KQ_ROWS, 1, 1);
    static const int forced = getenv("GGML_KQ_NWARPS") ? atoi(getenv("GGML_KQ_NWARPS")) : 0;
    int nw = forced == 1 || forced == 2 || forced == 4 ? forced : 2;
    while (nw > 1 && ngroups < WARP_SIZE*nw) {
        nw /= 2;
    }
    // GGML_KQR_MINB (nwarps 2 only): min blocks/SM for the q4kr launch bounds.  Per-width override GGML_KQR_MINB<n>
    // (n = 1..4) wins over GGML_KQR_MINB.  Defaults (sweep): 8 for n = 1..3 (fastest; 6/7 equal
    // within noise, 9-12 slower), 6 for n = 4 (168 regs instead of 128 + a 24-byte spill: +4.7 %).
    static const int minb_all = getenv("GGML_KQR_MINB") ? atoi(getenv("GGML_KQR_MINB")) : 0;
    static const int minb_w   = [] {
        char name[16];
        snprintf(name, sizeof(name), "GGML_KQR_MINB%d", ncols_dst);
        return getenv(name) ? atoi(getenv(name)) : 0;
    }();
    const int minb = minb_w ? minb_w : minb_all ? minb_all : (ncols_dst == 4 ? 6 : 0);
    switch (nw) {
        case 1:
            mul_mat_vec_q4kr<ncols_dst, 1><<<grid, dim3(WARP_SIZE, 1, 1), 0, stream>>>(
                vx, aq, ads, dst, ngroups, row_bytes, stride_col_aq, stride_col_ads, stride_col_dst);
            break;
        case 2:
#define Q4KR_MINB_CASE(mb) \
            case mb: mul_mat_vec_q4kr<ncols_dst, 2, mb><<<grid, dim3(WARP_SIZE, 2, 1), 0, stream>>>( \
                vx, aq, ads, dst, ngroups, row_bytes, stride_col_aq, stride_col_ads, stride_col_dst); break;
            switch (minb) {
                Q4KR_MINB_CASE(6)
                Q4KR_MINB_CASE(7)
                Q4KR_MINB_CASE(9)
                Q4KR_MINB_CASE(10)
                Q4KR_MINB_CASE(12)
                default:
                    mul_mat_vec_q4kr<ncols_dst, 2><<<grid, dim3(WARP_SIZE, 2, 1), 0, stream>>>(
                        vx, aq, ads, dst, ngroups, row_bytes, stride_col_aq, stride_col_ads, stride_col_dst);
                    break;
            }
#undef Q4KR_MINB_CASE
            break;
        default:
            mul_mat_vec_q4kr<ncols_dst, 4><<<grid, dim3(WARP_SIZE, 4, 1), 0, stream>>>(
                vx, aq, ads, dst, ngroups, row_bytes, stride_col_aq, stride_col_ads, stride_col_dst);
            break;
    }
}

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
static void q3r_launch(const char * vx, const half2 * aq, const half2 * ads, float * dst, int ngroups, int nrows,
        int row_bytes, int stride_col_aq, int stride_col_ads, int stride_col_dst, cudaStream_t stream) {
    const dim3 grid(nrows/KQ_ROWS, 1, 1);
    static const int forced = getenv("GGML_Q3R_NWARPS") ? atoi(getenv("GGML_Q3R_NWARPS")) : 0;
    int nw = forced == 1 || forced == 2 || forced == 4 ? forced : 2;
    while (nw > 1 && ngroups < WARP_SIZE*nw) {
        nw /= 2;
    }
    switch (nw) {
        case 1:
            mul_mat_vec_q3r<ncols_dst, 1><<<grid, dim3(WARP_SIZE, 1, 1), 0, stream>>>(
                vx, aq, ads, dst, ngroups, row_bytes, stride_col_aq, stride_col_ads, stride_col_dst);
            break;
        case 2: {
            // GGML_Q3R_MINB / GGML_Q3R_MINB<n>: launch-bounds min blocks (register cap only, same FP ops)
            static const int mb_all = getenv("GGML_Q3R_MINB") ? atoi(getenv("GGML_Q3R_MINB")) : 0;
            static const int mb_w   = [] {
                char name[16];
                snprintf(name, sizeof(name), "GGML_Q3R_MINB%d", ncols_dst);
                return getenv(name) ? atoi(getenv(name)) : 0;
            }();
            // default (sweep): 8 for n = 1..3, 6 for n = 4 (+20 %: removes the 80-byte spill)
            const int mb = mb_w ? mb_w : mb_all ? mb_all : (ncols_dst == 4 ? 6 : 0);
#define Q3R_MINB_CASE(m)             case m: mul_mat_vec_q3r<ncols_dst, 2, m><<<grid, dim3(WARP_SIZE, 2, 1), 0, stream>>>(                 vx, aq, ads, dst, ngroups, row_bytes, stride_col_aq, stride_col_ads, stride_col_dst); break;
            switch (mb) {
                Q3R_MINB_CASE(5)
                Q3R_MINB_CASE(6)
                Q3R_MINB_CASE(7)
                Q3R_MINB_CASE(10)
                default:
                    mul_mat_vec_q3r<ncols_dst, 2><<<grid, dim3(WARP_SIZE, 2, 1), 0, stream>>>(
                        vx, aq, ads, dst, ngroups, row_bytes, stride_col_aq, stride_col_ads, stride_col_dst);
                    break;
            }
#undef Q3R_MINB_CASE
        } break;
        default:
            mul_mat_vec_q3r<ncols_dst, 4><<<grid, dim3(WARP_SIZE, 4, 1), 0, stream>>>(
                vx, aq, ads, dst, ngroups, row_bytes, stride_col_aq, stride_col_ads, stride_col_dst);
            break;
    }
}

bool ggml_cuda_kq_q3r_mul_mat(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    if (src0->type != GGML_TYPE_Q3_K || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
        src1->ne[1] < 1 || src1->ne[1] > 8 || src1->ne[2] != 1 || src1->ne[3] != 1 ||
        src1->nb[0] != sizeof(float) || !ggml_is_contiguous(dst)) {
        return false;
    }
    // q4r_tensor_eligible guarantees these for every packed Q3_K tensor
    GGML_ASSERT(src0->ne[0] % (2*QK_K) == 0 && src0->ne[1] % KQ_ROWS == 0 && src0->nb[1] % 4 == 0);

    cudaStream_t stream = ctx.stream();
    const int ngroups = (int) (src0->ne[0]/32);
    const int64_t ne11 = src1->ne[1];

    ggml_cuda_pool_alloc<half2> aq_own (ctx.pool());
    ggml_cuda_pool_alloc<half2> ads_own(ctx.pool());
    const half2 * aqp  = nullptr;
    const half2 * adsp = nullptr;
    if (!ggml_cuda_a16_cached(ctx, src1, ngroups, &aqp, &adsp)) {
        aq_own.alloc ((size_t) ne11*ngroups*16);
        ads_own.alloc((size_t) ne11*ngroups);
        const int64_t s11 = src1->nb[1]/sizeof(float);
        const bool vec4 = s11 % 4 == 0 && ((uintptr_t) src1->data) % 16 == 0;
        ggml_cuda_a16_quantize((const float *) src1->data, aq_own.get(), ads_own.get(), s11, ngroups, (int) ne11, vec4, stream);
        aqp  = aq_own.get();
        adsp = ads_own.get();
    }

    const char * vx = (const char *) src0->data;
    float * d = (float *) dst->data;
    const int nrows = (int) src0->ne[1];
    const int row_bytes = (int) src0->nb[1];
    const int stride_col_aq  = ngroups*16;
    const int stride_col_ads = ngroups;
    const int stride_col_dst = (int) (dst->nb[1]/sizeof(float));

    for (int c0 = 0; c0 < ne11; c0 += 4) {
        const int nc = (int) std::min<int64_t>(4, ne11 - c0);
        const half2 * aqc  = aqp  + (size_t) c0*stride_col_aq;
        const half2 * adsc = adsp + (size_t) c0*stride_col_ads;
        float       * dc   = d    + (size_t) c0*stride_col_dst;
        switch (nc) {
            case 1: q3r_launch<1>(vx, aqc, adsc, dc, ngroups, nrows, row_bytes, stride_col_aq, stride_col_ads, stride_col_dst, stream); break;
            case 2: q3r_launch<2>(vx, aqc, adsc, dc, ngroups, nrows, row_bytes, stride_col_aq, stride_col_ads, stride_col_dst, stream); break;
            case 3: q3r_launch<3>(vx, aqc, adsc, dc, ngroups, nrows, row_bytes, stride_col_aq, stride_col_ads, stride_col_dst, stream); break;
            default: q3r_launch<4>(vx, aqc, adsc, dc, ngroups, nrows, row_bytes, stride_col_aq, stride_col_ads, stride_col_dst, stream); break;
        }
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}

static void kq_run(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

bool ggml_cuda_kq_q4kr_mul_mat(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    if (src0->type != GGML_TYPE_Q4_K || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
        src1->ne[1] < 1 || src1->ne[1] > 8 || src1->ne[2] != 1 || src1->ne[3] != 1 ||
        src1->nb[0] != sizeof(float) || !ggml_is_contiguous(dst)) {
        return false;
    }
    // q4r_tensor_eligible guarantees these for every packed tensor
    GGML_ASSERT(src0->ne[0] % QK_K == 0 && src0->ne[1] % KQ_ROWS == 0 && src0->nb[1] % 16 == 0);

    cudaStream_t stream = ctx.stream();
    const int ngroups = (int) (src0->ne[0]/32);
    const int64_t ne11 = src1->ne[1];

    ggml_cuda_pool_alloc<half2> aq_own (ctx.pool());
    ggml_cuda_pool_alloc<half2> ads_own(ctx.pool());
    const half2 * aqp  = nullptr;
    const half2 * adsp = nullptr;
    if (!ggml_cuda_a16_cached(ctx, src1, ngroups, &aqp, &adsp)) {
        aq_own.alloc ((size_t) ne11*ngroups*16);
        ads_own.alloc((size_t) ne11*ngroups);
        const int64_t s11 = src1->nb[1]/sizeof(float);
        const bool vec4 = s11 % 4 == 0 && ((uintptr_t) src1->data) % 16 == 0;
        ggml_cuda_a16_quantize((const float *) src1->data, aq_own.get(), ads_own.get(), s11, ngroups, (int) ne11, vec4, stream);
        aqp  = aq_own.get();
        adsp = ads_own.get();
    }

    const char * vx = (const char *) src0->data;
    float * d = (float *) dst->data;
    const int nrows = (int) src0->ne[1];
    const int row_bytes = (int) src0->nb[1];
    const int stride_col_aq  = ngroups*16;
    const int stride_col_ads = ngroups;
    const int stride_col_dst = (int) (dst->nb[1]/sizeof(float));

    // Widths 5..8 (MTP verify with more drafts, prefill tails) as two passes over the weights, columns 0..3 and
    // 4..ne11-1, so a packed tensor never needs the full-tensor unpacked copy for a decode width.  (Not
    // bit-identical to env-off there: with GGML_KQR_Q4K=0 these widths run upstream MMVQ on q8_1 activations.)
    for (int c0 = 0; c0 < ne11; c0 += 4) {
        const int nc = (int) std::min<int64_t>(4, ne11 - c0);
        const half2 * aqc  = aqp  + (size_t) c0*stride_col_aq;
        const half2 * adsc = adsp + (size_t) c0*stride_col_ads;
        float       * dc   = d    + (size_t) c0*stride_col_dst;
        switch (nc) {
            case 1: q4kr_launch<1>(vx, aqc, adsc, dc, ngroups, nrows, row_bytes, stride_col_aq, stride_col_ads, stride_col_dst, stream); break;
            case 2: q4kr_launch<2>(vx, aqc, adsc, dc, ngroups, nrows, row_bytes, stride_col_aq, stride_col_ads, stride_col_dst, stream); break;
            case 3: q4kr_launch<3>(vx, aqc, adsc, dc, ngroups, nrows, row_bytes, stride_col_aq, stride_col_ads, stride_col_dst, stream); break;
            default: q4kr_launch<4>(vx, aqc, adsc, dc, ngroups, nrows, row_bytes, stride_col_aq, stride_col_ads, stride_col_dst, stream); break;
        }
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}

bool ggml_cuda_kq_q6r_mul_mat(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || src1->ne[1] < 1 || src1->ne[1] > 4 ||
        src1->ne[2] != 1 || src1->ne[3] != 1 || src1->nb[0] != sizeof(float) || !ggml_is_contiguous(dst)) {
        return false;
    }
    kq_run(ctx, src0, src1, dst);
    return true;
}

bool ggml_cuda_kq_mul_mat(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const ggml_type type = src0->type;
    if (type != GGML_TYPE_Q4_K && type != GGML_TYPE_Q5_K && type != GGML_TYPE_IQ4_XS) {
        return false;
    }
    static const bool off     = getenv("GGML_KQ")      && atoi(getenv("GGML_KQ"))      == 0;
    static const bool off_iq4 = getenv("GGML_KQ_IQ4")  && atoi(getenv("GGML_KQ_IQ4"))  == 0;
    static const bool off_k   = getenv("GGML_KQ_K")    && atoi(getenv("GGML_KQ_K"))    == 0;
    if (off || (type == GGML_TYPE_IQ4_XS && off_iq4) || (type != GGML_TYPE_IQ4_XS && off_k)) {
        return false;
    }
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || cc < GGML_CUDA_CC_PASCAL || cc >= GGML_CUDA_CC_DP4A) {
        return false;
    }
    if (!ggml_is_contiguous(src0) || src0->ne[2] != 1 || src0->ne[3] != 1 || src0->ne[0] % QK_K != 0 ||
        src0->ne[1] % KQ_ROWS != 0 || ((uintptr_t) src0->data) % 16 != 0 ||
        src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || src1->ne[1] < 1 || src1->ne[1] > 4 ||
        src1->ne[2] != 1 || src1->ne[3] != 1 || src1->nb[0] != sizeof(float) || !ggml_is_contiguous(dst)) {
        return false;
    }
    // Every super-block must start 16-aligned (q4_K 144 B, q5_K 176 B) or 8-aligned (iq4_xs 136 B, whose quants
    // are read with 8-byte loads): true for every block when the row stride and the base are.
    if (src0->nb[1] % 8 != 0) {
        return false;
    }
    kq_run(ctx, src0, src1, dst);
    return true;
}

static void kq_run(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const ggml_type type = src0->type;
    cudaStream_t stream = ctx.stream();
    const int ngroups = (int) (src0->ne[0]/32);
    const int64_t ne11 = src1->ne[1];

    ggml_cuda_pool_alloc<half2> aq_own (ctx.pool());
    ggml_cuda_pool_alloc<half2> ads_own(ctx.pool());
    const half2 * aqp  = nullptr;
    const half2 * adsp = nullptr;
    if (!ggml_cuda_a16_cached(ctx, src1, ngroups, &aqp, &adsp)) {
        aq_own.alloc ((size_t) ne11*ngroups*16);
        ads_own.alloc((size_t) ne11*ngroups);
        const int64_t s11 = src1->nb[1]/sizeof(float);
        const bool vec4 = s11 % 4 == 0 && ((uintptr_t) src1->data) % 16 == 0;
        ggml_cuda_a16_quantize((const float *) src1->data, aq_own.get(), ads_own.get(), s11, ngroups, (int) ne11, vec4, stream);
        aqp  = aq_own.get();
        adsp = ads_own.get();
    }

    const char * vx = (const char *) src0->data;
    float * d = (float *) dst->data;
    const int nrows = (int) src0->ne[1];
    const int row_bytes = (int) src0->nb[1];
    const int stride_col_aq  = ngroups*16;
    const int stride_col_ads = ngroups;
    const int stride_col_dst = (int) (dst->nb[1]/sizeof(float));
    switch (type) {
        case GGML_TYPE_Q4_K:
            kq_dispatch<GGML_TYPE_Q4_K, 16>(vx, aqp, adsp, d, (int) ne11, ngroups, nrows, row_bytes, stride_col_aq, stride_col_ads, stride_col_dst, stream);
            break;
        case GGML_TYPE_Q5_K:
            kq_dispatch<GGML_TYPE_Q5_K, 16>(vx, aqp, adsp, d, (int) ne11, ngroups, nrows, row_bytes, stride_col_aq, stride_col_ads, stride_col_dst, stream);
            break;
        case GGML_TYPE_Q6_K:
            // Rows of a repacked Q6_K tensor are 210*nb bytes; use 16- or 8-byte loads when every row allows.
            if (ggml_cuda_q6r2_on()) {
                // q6r2: 16-byte ql2 groups at 16*G, 8-byte qh2 groups at 128*nb + 8*G
                if (row_bytes % 16 == 0) {
                    kq_dispatch<GGML_TYPE_Q6_K, 16, true>(vx, aqp, adsp, d, (int) ne11, ngroups, nrows, row_bytes, stride_col_aq, stride_col_ads, stride_col_dst, stream);
                } else if (row_bytes % 8 == 0) {
                    kq_dispatch<GGML_TYPE_Q6_K,  8, true>(vx, aqp, adsp, d, (int) ne11, ngroups, nrows, row_bytes, stride_col_aq, stride_col_ads, stride_col_dst, stream);
                } else {
                    kq_dispatch<GGML_TYPE_Q6_K,  4, true>(vx, aqp, adsp, d, (int) ne11, ngroups, nrows, row_bytes, stride_col_aq, stride_col_ads, stride_col_dst, stream);
                }
            } else if (row_bytes % 16 == 0) {
                kq_dispatch<GGML_TYPE_Q6_K, 16>(vx, aqp, adsp, d, (int) ne11, ngroups, nrows, row_bytes, stride_col_aq, stride_col_ads, stride_col_dst, stream);
            } else if (row_bytes % 8 == 0) {
                kq_dispatch<GGML_TYPE_Q6_K,  8>(vx, aqp, adsp, d, (int) ne11, ngroups, nrows, row_bytes, stride_col_aq, stride_col_ads, stride_col_dst, stream);
            } else {
                kq_dispatch<GGML_TYPE_Q6_K,  4>(vx, aqp, adsp, d, (int) ne11, ngroups, nrows, row_bytes, stride_col_aq, stride_col_ads, stride_col_dst, stream);
            }
            break;
        case GGML_TYPE_IQ4_XS:
            kq_dispatch<GGML_TYPE_IQ4_XS, 16>(vx, aqp, adsp, d, (int) ne11, ngroups, nrows, row_bytes, stride_col_aq, stride_col_ads, stride_col_dst, stream);
            break;
        default:
            // the callers only pass the types above; anything else must not be read as IQ4_XS
            GGML_ABORT("kq: unsupported type %s", ggml_type_name(type));
    }
}
