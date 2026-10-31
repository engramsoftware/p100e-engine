#pragma once

#include "common.cuh"
#include "fattn-common.cuh"

// GQA-grouped vector FlashAttention kernel for quantized KV (opt-in, see fattn-vec-gqa.cu for the env vars).
//
// The stock vector kernel (fattn-vec.cuh) launches one block-set per Q head (blockIdx.z = head), so with GQA
// every K/V row is loaded, unpacked and dequantized once per Q head (6x for Qwen3.8-27B: 24 Q / 4 KV heads).
// Here one block handles one K/V head together with ncols2 of its Q heads and ncols1 query tokens, i.e.
// ncols = ncols1*ncols2 columns. Column j is token ic0 + j/ncols2 and Q head head0 + j%ncols2.
// Each K/V row is loaded and unpacked once and reused by all columns. launch_fattn, parallel_blocks and
// flash_attn_combine_results are reused unchanged (blockIdx.z covers gqa_ratio/ncols2 head groups per K/V head).
//
// h2 == false: the stock math (Q as q8_1, ggml_cuda_dp4a = vmad on sm_60); only the K row load/unpack is shared.
// h2 == true (sm_60 only): GP100 has no DP4A but full-rate HFMA2 (see mmvq-f16-sm60.cu for the measured rates):
//   K: the 4/8-bit values become exact half2 via LOP3/PRMT magic constants, are multiplied by d in half once per
//      row (the same values as the f16 conversion used by the tile kernel) and dotted with Q stored as half
//      divided by the column's amax (|Q| <= 1). One half lane sums 4-8 products, the rest is float.
//   V: the same magic gives exact integers that are converted to float and scaled by d in float, i.e. the same
//      values as dequantize_V_q4_0/q8_0<float> without I2F or __vsubss4. VKQ accumulates in float as in the stock
//      kernel.
// Only D == 256, type_K == type_V in {Q4_0, Q8_0}, mask present, no ALiBi, no sinks, no logit softcap (the host
// checks this in ggml_cuda_fattn_vec_gqa_route).

#define FATTN_VEC_GQA_NTHREADS 128

bool ggml_cuda_fattn_vec_gqa_route(int cc, const ggml_tensor * dst, bool gqa_opt_applies);

void ggml_cuda_flash_attn_ext_vec_gqa(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
// (x & mask) | magic, reinterpreted as half2. 0x6400 is 1024.0 in fp16: a nibble in mantissa bits 0-3 gives exactly
// 1024 + q. 0x5400 is 64.0: a nibble in mantissa bits 4-7 gives exactly 64 + q.
template <uint32_t mask, uint32_t magic>
static __device__ __forceinline__ half2 fattn_vec_gqa_lop3_h2(const uint32_t x) {
    uint32_t r;
    asm("lop3.b32 %0, %1, %2, %3, 0xea;" : "=r"(r) : "r"(x), "n"(mask), "n"(magic));
    return *((const half2 *) &r);
}

// Bytes 0 and 2 (sel 0x4240) or 1 and 3 (sel 0x4341) of x into the mantissas of 1024.0 (0x64XX), reinterpreted as half2.
template <uint32_t sel>
static __device__ __forceinline__ half2 fattn_vec_gqa_prmt_h2(const uint32_t x) {
    const uint32_t r = __byte_perm(x, 0x64646464, sel);
    return *((const half2 *) &r);
}

// h2 == true layout: element index of the low half of K_h2[i] / Q_h2[j][i] for this lane, the high half is index + 2.
// q4_0: the lane owns qs ints iq = l*nthreads_KQ + lane_KQ (8 values each, low and high nibbles), e0 = 32*(iq/4) + 4*(iq%4):
//     K_h2[4*l + 0] = (e0+ 0, e0+ 2), K_h2[4*l + 1] = (e0+16, e0+18), K_h2[4*l + 2] = (e0+ 1, e0+ 3), K_h2[4*l + 3] = (e0+17, e0+19)
// q8_0: the lane owns ints k = l*nthreads_KQ + lane_KQ (4 values each), e0 = 4*k:
//     K_h2[2*l + 0] = (e0, e0+2), K_h2[2*l + 1] = (e0+1, e0+3)
template <ggml_type type_K, int nthreads_KQ>
static __device__ __forceinline__ int fattn_vec_gqa_h2_index(const int i, const int lane_KQ) {
    if constexpr (type_K == GGML_TYPE_Q4_0) {
        const int iq = (i/4)*nthreads_KQ + lane_KQ;
        const int s  = i % 4;
        return QK4_0*(iq/(QK4_0/8)) + 4*(iq%(QK4_0/8)) + (s & 1)*(QK4_0/2) + (s >> 1);
    } else {
        static_assert(type_K == GGML_TYPE_Q8_0, "bad type");
        const int k = (i/2)*nthreads_KQ + lane_KQ;
        return 4*k + i % 2;
    }
}

// h2 == true: loads this lane's share of one K row as half2 (values times d), in the order of fattn_vec_gqa_h2_index.
template <ggml_type type_K, int D, int nthreads_KQ>
static __device__ __forceinline__ void fattn_vec_gqa_load_K_h2(const char * __restrict__ K_c, half2 * __restrict__ K_h2) {
    const int lane_KQ = nthreads_KQ == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_KQ;

    if constexpr (type_K == GGML_TYPE_Q4_0) {
        const block_q4_0 * K_q4_0 = (const block_q4_0 *) K_c;
        const half2 magic_lo = __float2half2_rn(1032.0f); // 1024 + 8
        const half2 magic_hi = __float2half2_rn(  72.0f); //   64 + 8

#pragma unroll
        for (int l = 0; l < D/(8*nthreads_KQ); ++l) {
            const int iq = l*nthreads_KQ + lane_KQ;
            const int ib = iq / (QK4_0/8);
            const int iw = iq % (QK4_0/8);

            uint32_t w;
            ggml_cuda_memcpy_1<sizeof(w), 2>(&w, K_q4_0[ib].qs + sizeof(w)*iw);
            const uint32_t ws = w >> 8;
            const half2    d2 = __half2half2(K_q4_0[ib].d);

            // Subtract first (exact), then scale: folding the offset into one HFMA2 cancels (see mmvq-f16-sm60.cu).
            K_h2[4*l + 0] = __hmul2(__hsub2(fattn_vec_gqa_lop3_h2<0x000f000f, 0x64006400>(w),  magic_lo), d2);
            K_h2[4*l + 1] = __hmul2(__hsub2(fattn_vec_gqa_lop3_h2<0x00f000f0, 0x54005400>(w),  magic_hi), d2);
            K_h2[4*l + 2] = __hmul2(__hsub2(fattn_vec_gqa_lop3_h2<0x000f000f, 0x64006400>(ws), magic_lo), d2);
            K_h2[4*l + 3] = __hmul2(__hsub2(fattn_vec_gqa_lop3_h2<0x00f000f0, 0x54005400>(ws), magic_hi), d2);
        }
    } else {
        static_assert(type_K == GGML_TYPE_Q8_0, "bad type");
        const block_q8_0 * K_q8_0 = (const block_q8_0 *) K_c;
        const half2 magic = __float2half2_rn(1152.0f); // 1024 + 128

#pragma unroll
        for (int l = 0; l < D/(4*nthreads_KQ); ++l) {
            const int k  = l*nthreads_KQ + lane_KQ;
            const int ib = k / QI8_0;
            const int iw = k % QI8_0;

            uint32_t v;
            ggml_cuda_memcpy_1<sizeof(v), 2>(&v, K_q8_0[ib].qs + sizeof(v)*iw);
            const uint32_t u  = v ^ 0x80808080; // int8 -> uint8 + 128
            const half2    d2 = __half2half2(K_q8_0[ib].d);

            K_h2[2*l + 0] = __hmul2(__hsub2(fattn_vec_gqa_prmt_h2<0x4240>(u), magic), d2);
            K_h2[2*l + 1] = __hmul2(__hsub2(fattn_vec_gqa_prmt_h2<0x4341>(u), magic), d2);
        }
    }
}

// h2 == true: dequantizes 4 consecutive V values starting at i0 to float, same values as dequantize_V_q4_0/q8_0<float, 4>.
template <ggml_type type_V>
static __device__ __forceinline__ void fattn_vec_gqa_dequantize_V_h2(const void * __restrict__ vx, float2 * __restrict__ dst, const int i0) {
    if constexpr (type_V == GGML_TYPE_Q4_0) {
        const block_q4_0 * x = (const block_q4_0 *) vx;

        const int ib    =  i0          /  QK4_0;
        const int iqs   =  i0          % (QK4_0/2);
        const int shift = (i0 % QK4_0) / (QK4_0/2);

        uint32_t q;
        ggml_cuda_memcpy_1<sizeof(q), 2>(&q, x[ib].qs + iqs);
        q >>= 4*shift;

        const half2  magic = __float2half2_rn(1032.0f);
        const float2 v02   = __half22float2(__hsub2(fattn_vec_gqa_lop3_h2<0x000f000f, 0x64006400>(q),      magic));
        const float2 v13   = __half22float2(__hsub2(fattn_vec_gqa_lop3_h2<0x000f000f, 0x64006400>(q >> 8), magic));
        const float  d     = __half2float(x[ib].d);

        dst[0] = make_float2(d*v02.x, d*v13.x);
        dst[1] = make_float2(d*v02.y, d*v13.y);
    } else {
        static_assert(type_V == GGML_TYPE_Q8_0, "bad type");
        const block_q8_0 * x = (const block_q8_0 *) vx;

        const int ib  = i0 / QK8_0;
        const int iqs = i0 % QK8_0;

        uint32_t q;
        ggml_cuda_memcpy_1<sizeof(q), 2>(&q, x[ib].qs + iqs);
        const uint32_t u = q ^ 0x80808080;

        const half2  magic = __float2half2_rn(1152.0f);
        const float2 v02   = __half22float2(__hsub2(fattn_vec_gqa_prmt_h2<0x4240>(u), magic));
        const float2 v13   = __half22float2(__hsub2(fattn_vec_gqa_prmt_h2<0x4341>(u), magic));
        const float  d     = __half2float(x[ib].d);

        dst[0] = make_float2(d*v02.x, d*v13.x);
        dst[1] = make_float2(d*v02.y, d*v13.y);
    }
}
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

// h2 == false: loads this lane's ints of one K row and the block scales, same layout as vec_dot_fattn_vec_KQ_q4_0/q8_0.
template <ggml_type type_K, int D, int nthreads_KQ>
static __device__ __forceinline__ void fattn_vec_gqa_load_K_q8_1(const char * __restrict__ K_c, int * K_i32, float * K_d) {
    const int lane_KQ = nthreads_KQ == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_KQ;

#pragma unroll
    for (int l = 0; l < int(D/(sizeof(int)*nthreads_KQ)); ++l) {
        const int k_KQ = l*nthreads_KQ + lane_KQ;

        if constexpr (type_K == GGML_TYPE_Q4_0) {
            const block_q4_0 * K_q4_0 = (const block_q4_0 *) K_c;

            const int ib    = k_KQ /  QI8_1;
            const int iqs4  = k_KQ %  QI4_0;
            const int shift = k_KQ & (QI8_1/2);

            int v;
            ggml_cuda_memcpy_1<sizeof(int), 2>(&v, K_q4_0[ib].qs + sizeof(int)*iqs4);
            K_i32[l] = (v >> shift) & 0x0F0F0F0F;
            K_d[l]   = __half2float(K_q4_0[ib].d);
        } else {
            static_assert(type_K == GGML_TYPE_Q8_0, "bad type");
            const block_q8_0 * K_q8_0 = (const block_q8_0 *) K_c;

            const int ib  = k_KQ / QI8_0;
            const int iqs = k_KQ % QI8_0;

            int v;
            ggml_cuda_memcpy_1<sizeof(int), 2>(&v, K_q8_0[ib].qs + 4*iqs);
            K_i32[l] = v;
            K_d[l]   = __half2float(K_q8_0[ib].d);
        }
    }
}

// h2 == false: the stock q8_1 dot product of one column with a K row loaded by fattn_vec_gqa_load_K_q8_1.
template <ggml_type type_K, int nints>
static __device__ __forceinline__ float fattn_vec_gqa_dot_q8_1(
        const int * K_i32, const float * K_d, const int * Q_i32, const float2 * Q_ds) {
    float sum = 0.0f;

#pragma unroll
    for (int l = 0; l < nints; ++l) {
        const int sumi = ggml_cuda_dp4a(K_i32[l], Q_i32[l], 0);

        if constexpr (type_K == GGML_TYPE_Q4_0) {
            sum += K_d[l] * (sumi*Q_ds[l].x - (8/QI8_1)*Q_ds[l].y);
        } else {
            sum += K_d[l]*Q_ds[l].x * ((float) sumi);
        }
    }

    return sum;
}

// Currently llvm with the amdgcn target does not support unrolling loops
// that contain a break that can not be resolved at compile time.
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
#endif // __clang__
template<int D, int ncols1, int ncols2, ggml_type type_KV, bool h2> // D == head size
__launch_bounds__(FATTN_VEC_GQA_NTHREADS, 1)
static __global__ void flash_attn_ext_vec_gqa(
        const char * Q_ptr,
        const char * K_ptr,
        const char * V_ptr,
        const char * mask_ptr,
        const char * sinks_ptr,
        const int  * KV_max_ptr,
        float      * dst_ptr,
        float2     * dst_meta_ptr,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int32_t ne00, const uint3   ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
                            const int32_t ne31, const int32_t ne32, const int32_t ne33,
                            const int32_t nb31, const int32_t nb32, const int64_t nb33) {
    ggml_cuda_pdl_lc();
#if defined(FLASH_ATTN_AVAILABLE) && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#if __CUDA_ARCH__ != GGML_CUDA_CC_PASCAL
    // The host selects the half2 variant only for cc == 600, skip it elsewhere for faster compilation:
    if constexpr (h2) {
        GGML_UNUSED_VARS(Q_ptr, K_ptr, V_ptr, mask_ptr, sinks_ptr, KV_max_ptr, dst_ptr, dst_meta_ptr, scale,
            max_bias, m0, m1, n_head_log2, logit_softcap,
            ne00, ne01, ne02, ne03,
                  nb01, nb02, nb03,
            ne10, ne11, ne12, ne13,
                  nb11, nb12, nb13,
                  nb21, nb22, nb23,
                  ne31, ne32, ne33,
                  nb31, nb32, nb33);
        NO_DEVICE_CODE;
        return;
    }
#endif // __CUDA_ARCH__ != GGML_CUDA_CC_PASCAL

    // Not supported by this kernel, excluded by the host:
    GGML_UNUSED_VARS(sinks_ptr, max_bias, m0, m1, n_head_log2, logit_softcap, ne00, ne03, ne10, ne13, ne31, ne32, nb32);

    const char * GGML_CUDA_RESTRICT Q        = Q_ptr;
    const char * GGML_CUDA_RESTRICT K        = K_ptr;
    const char * GGML_CUDA_RESTRICT V        = V_ptr;
    const char * GGML_CUDA_RESTRICT mask     = mask_ptr;
    const int  * GGML_CUDA_RESTRICT KV_max   = KV_max_ptr;
    float      * GGML_CUDA_RESTRICT dst      = dst_ptr;
    float2     * GGML_CUDA_RESTRICT dst_meta = dst_meta_ptr;

    //In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    static_assert(type_KV == GGML_TYPE_Q4_0 || type_KV == GGML_TYPE_Q8_0, "bad type");

    constexpr int ncols       = ncols1*ncols2;
    constexpr int nthreads    = FATTN_VEC_GQA_NTHREADS;
    constexpr int nwarps      = nthreads / WARP_SIZE;
    constexpr int nthreads_KQ = h2 && ncols <= 6 ? WARP_SIZE/2 : WARP_SIZE; // Lanes per K row.
    constexpr int nthreads_V  = (D/4 < WARP_SIZE ? D/4 : WARP_SIZE);

    static_assert(WARP_SIZE % nthreads_KQ == 0, "bad nthreads_K");
    static_assert(WARP_SIZE % nthreads_V  == 0, "bad nthreads_V");
    static_assert(D % (2*WARP_SIZE) == 0, "D not divisible by 2*WARP_SIZE == 64.");
    static_assert(D % (sizeof(int)*nthreads_KQ) == 0, "bad D");
    static_assert(!h2 || D % (8*nthreads_KQ) == 0, "bad D");

    constexpr int V_rows_per_thread = 4;
    constexpr int V_cols_per_iter   = WARP_SIZE / nthreads_V;

    constexpr int nints = D/(sizeof(int)*nthreads_KQ); // h2 == false: q8_1 ints of Q per lane and column.
    constexpr int nh2   = D/(2*nthreads_KQ);           // h2 == true: half2 of K and Q per lane and column.

    const int lane_KQ = nthreads_KQ == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_KQ;

    // blockIdx.z = (sequence*ne12 + K/V head)*(gqa_ratio/ncols2) + head group, matches launch_fattn (ntiles_z_gqa*ne12*ne13).
    const int ic0       = blockIdx.x*ncols1;   // First token of this block.
    const int gqa_ratio = ne02 / ne12;         // Q heads per K/V head.
    const int ngroups   = gqa_ratio / ncols2;  // The host ensures gqa_ratio % ncols2 == 0.
    const int sequence  = blockIdx.z / (ne12*ngroups);
    const int head_KV   = blockIdx.z / ngroups - sequence*ne12;
    const int head0     = head_KV*gqa_ratio + (blockIdx.z % ngroups)*ncols2; // First Q head of this block.

    Q += nb03*sequence + nb02*head0 + nb01*ic0; // Column j: + nb02*(j % ncols2) + nb01*(j / ncols2).
    K += nb13*sequence + nb12*head_KV;
    V += nb23*sequence + nb22*head_KV;

    const int    stride_mask = nb31 / int(sizeof(half));
    const half * maskh       = (const half *) (mask + nb33*(sequence % ne33) + nb31*ic0);

    const int tid = WARP_SIZE*threadIdx.y + threadIdx.x;
    __builtin_assume(tid < nthreads);

    constexpr int ne_KQ      = ncols*D;
    constexpr int ne_combine = nwarps*V_cols_per_iter*D;
    float2 VKQ[ncols][(D/2)/nthreads_V] = {{{0.0f, 0.0f}}};
    __shared__ float KQ[ne_KQ > ne_combine ? ne_KQ : ne_combine];

    float KQ_max[ncols];
    float KQ_sum[ncols];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        KQ_max[j] = -FLT_MAX/2.0f;
        KQ_sum[j] = 0.0f;
    }

    // h2 == false: Q as q8_1 in registers (stock layout). h2 == true: Q/amax as half2 in the order of fattn_vec_gqa_h2_index.
    int    Q_i32[ncols][h2 ? 1 : nints];
    float2 Q_ds [ncols][h2 ? 1 : nints];
    half2  Q_h2 [ncols][h2 ? nh2 : 1];
    float  Q_scale[ncols]; // h2 == true: amax*scale per column.

    ggml_cuda_pdl_sync();
    if constexpr (!h2) {
#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

            if (j0 + nwarps > ncols && j >= ncols) {
                break;
            }

            // Reuse KQ as temporary storage for converting Q to q8_1:
            int    * tmp_q_i32 = (int    *) &KQ[j*D];
            float2 * tmp_q_ds  = (float2 *) (tmp_q_i32 + D/sizeof(int));

            // Set memory to zero if out of bounds:
            if (ncols1 > 1 && ic0 + j/ncols2 >= int(ne01.z)) {
#pragma unroll
                for (int i0 = 0; i0 < int(D/sizeof(int)); i0 += WARP_SIZE) {
                    const int i = i0 + threadIdx.x;

                    if (i0 + WARP_SIZE <= int(D/sizeof(int)) || i < int(D/sizeof(int))) {
                        tmp_q_i32[i] = 0;
                    }
                }
                if (threadIdx.x < D/QK8_1) {
                    tmp_q_ds[threadIdx.x] = make_float2(0.0f, 0.0f);
                }
            } else {
                const float * Q_f = (const float *) (Q + nb02*(j % ncols2) + nb01*(j / ncols2));
                constexpr int nthreads_quantize = D/sizeof(int) < WARP_SIZE ? D/sizeof(int) : WARP_SIZE;
#pragma unroll
                for (int i0 = 0; i0 < int(D/sizeof(int)); i0 += nthreads_quantize) {
                    quantize_q8_1_to_shared<float2, nthreads_quantize>
                        (Q_f + i0*sizeof(int), scale, tmp_q_i32 + i0, tmp_q_ds + i0/QI8_1);
                }
            }
        }

        __syncthreads();

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            int    * tmp_q_i32 = (int    *) &KQ[j*D];
            float2 * tmp_q_ds  = (float2 *) (tmp_q_i32 + D/sizeof(int));

#pragma unroll
            for (int i0 = 0; i0 < int(D/sizeof(int)); i0 += nthreads_KQ) {
                const int i = i0 + lane_KQ;

                Q_i32[j][i0/nthreads_KQ] = tmp_q_i32[i];
                Q_ds[j][i0/nthreads_KQ]  = tmp_q_ds[i/QI8_1];
            }
        }

        __syncthreads();
        GGML_UNUSED_VARS(Q_h2, Q_scale);
    } else {
        __shared__ float Q_scale_shared[ncols];

#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

            if (j0 + nwarps > ncols && j >= ncols) {
                break;
            }

            // Reuse KQ as temporary storage for Q/amax as half in natural order:
            half2 * tmp_q_h2 = (half2 *) &KQ[j*D];

            if (ncols1 > 1 && ic0 + j/ncols2 >= int(ne01.z)) {
#pragma unroll
                for (int i0 = 0; i0 < D/2; i0 += WARP_SIZE) {
                    tmp_q_h2[i0 + threadIdx.x] = make_half2(0.0f, 0.0f);
                }
                if (threadIdx.x == 0) {
                    Q_scale_shared[j] = 0.0f;
                }
            } else {
                const float2 * Q_f2 = (const float2 *) (Q + nb02*(j % ncols2) + nb01*(j / ncols2));

                float2 tmp[(D/2)/WARP_SIZE];
                float amax = 0.0f;
#pragma unroll
                for (int i0 = 0; i0 < D/2; i0 += WARP_SIZE) {
                    tmp[i0/WARP_SIZE] = Q_f2[i0 + threadIdx.x];
                    amax = fmaxf(amax, fmaxf(fabsf(tmp[i0/WARP_SIZE].x), fabsf(tmp[i0/WARP_SIZE].y)));
                }
                amax = warp_reduce_max(amax);
                const float amax_inv = amax > 0.0f ? 1.0f/amax : 0.0f;

#pragma unroll
                for (int i0 = 0; i0 < D/2; i0 += WARP_SIZE) {
                    tmp_q_h2[i0 + threadIdx.x] = __floats2half2_rn(tmp[i0/WARP_SIZE].x*amax_inv, tmp[i0/WARP_SIZE].y*amax_inv);
                }
                if (threadIdx.x == 0) {
                    Q_scale_shared[j] = amax*scale;
                }
            }
        }

        __syncthreads();

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            const half * tmp_q_h = (const half *) &KQ[j*D];

#pragma unroll
            for (int i = 0; i < nh2; ++i) {
                const int e = fattn_vec_gqa_h2_index<type_KV, nthreads_KQ>(i, lane_KQ);
                Q_h2[j][i] = __halves2half2(tmp_q_h[e], tmp_q_h[e + 2]);
            }
            Q_scale[j] = Q_scale_shared[j];
        }

        __syncthreads();
        GGML_UNUSED_VARS(Q_i32, Q_ds);
    }

    const int k_VKQ_max = KV_max ? KV_max[sequence*gridDim.x + blockIdx.x] : ne11;
    K     += blockIdx.y*nthreads * nb11;
    V     += blockIdx.y*nthreads * nb21;
    maskh += blockIdx.y*nthreads;
    for (int k_VKQ_0 = blockIdx.y*nthreads; k_VKQ_0 < k_VKQ_max; k_VKQ_0 += gridDim.y*nthreads,
             // Increment pointers after each loop:
             K += gridDim.y*nthreads*nb11, V += gridDim.y*nthreads*nb21, maskh += gridDim.y*nthreads) {

        // Calculate KQ tile and keep track of new maximum KQ values:
        float KQ_reg[ncols]; // KQ in registers.

        float KQ_max_new[ncols];
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            KQ_max_new[j] = KQ_max[j];
        }

#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < nthreads_KQ; ++i_KQ_0) {
            const int i_KQ = threadIdx.y*WARP_SIZE + (nthreads_KQ == WARP_SIZE ? 0 : (threadIdx.x & ~(nthreads_KQ-1))) + i_KQ_0;

            // The mask is the same for all heads of a token (no ALiBi, slope == 1):
            float mask_val[ncols1];
#pragma unroll
            for (int j1 = 0; j1 < ncols1; ++j1) {
                mask_val[j1] = mask && (ncols1 == 1 || ic0 + j1 < int(ne01.z)) ? __half2float(maskh[j1*stride_mask + i_KQ]) : 0.0f;
            }

            // Load and unpack the K row once, then the dot product for every column:
            float sum[ncols];
            if constexpr (!h2) {
                int   K_i32[nints];
                float K_d[nints];
                fattn_vec_gqa_load_K_q8_1<type_KV, D, nthreads_KQ>(K + i_KQ*nb11, K_i32, K_d);

#pragma unroll
                for (int j = 0; j < ncols; ++j) {
                    sum[j] = fattn_vec_gqa_dot_q8_1<type_KV, nints>(K_i32, K_d, Q_i32[j], Q_ds[j]);
                }
            } else {
                half2 K_h2[nh2];
                fattn_vec_gqa_load_K_h2<type_KV, D, nthreads_KQ>(K + i_KQ*nb11, K_h2);

#pragma unroll
                for (int j = 0; j < ncols; ++j) {
                    half2 acc = __hmul2(K_h2[0], Q_h2[j][0]);
#pragma unroll
                    for (int i = 1; i < nh2; ++i) {
                        acc = __hfma2(K_h2[i], Q_h2[j][i], acc);
                    }
                    sum[j] = __half2float(__hadd(__low2half(acc), __high2half(acc)));
                }
            }

#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                sum[j] = warp_reduce_sum<nthreads_KQ>(sum[j]);

                if constexpr (h2) {
                    sum[j] *= Q_scale[j];
                }

                sum[j] += mask_val[j/ncols2];

                KQ_max_new[j] = fmaxf(KQ_max_new[j], sum[j] + FATTN_KQ_MAX_OFFSET);

                if (lane_KQ == i_KQ_0) {
                    KQ_reg[j] = sum[j];
                }
            }
        }

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
#pragma unroll
            for (int offset = nthreads_KQ; offset < WARP_SIZE; offset <<= 1) {
                KQ_max_new[j] = fmaxf(KQ_max_new[j], __shfl_xor_sync(0xFFFFFFFF, KQ_max_new[j], offset, WARP_SIZE));
            }
            const float KQ_max_scale = expf(KQ_max[j] - KQ_max_new[j]);
            KQ_max[j] = KQ_max_new[j];

            KQ_reg[j] = expf(KQ_reg[j] - KQ_max[j]);
            KQ_sum[j] = KQ_sum[j]*KQ_max_scale + KQ_reg[j];
            KQ[j*nthreads + tid] = KQ_reg[j];

#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V].x *= KQ_max_scale;
                VKQ[j][i_VKQ_0/nthreads_V].y *= KQ_max_scale;
            }
        }

        __syncwarp();

#pragma unroll
        for (int k0 = 0; k0 < WARP_SIZE; k0 += V_cols_per_iter) {
            const int k = threadIdx.y*WARP_SIZE + k0 + (nthreads_V == WARP_SIZE ? 0 : threadIdx.x / nthreads_V);

            float KQ_k[ncols];
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                KQ_k[j] = KQ[j*nthreads + k];
            }
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
                const int i0 = 2*i_VKQ_0 + (nthreads_V == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_V)*V_rows_per_thread;

                // The V row is dequantized once for all columns:
                float2 tmp[V_rows_per_thread/2];
                if constexpr (h2) {
                    fattn_vec_gqa_dequantize_V_h2<type_KV>(V + k*nb21, tmp, i0);
                } else {
                    constexpr dequantize_V_t dequantize_V = get_dequantize_V<type_KV, float, V_rows_per_thread>();
                    dequantize_V(V + k*nb21, tmp, i0);
                }
#pragma unroll
                for (int i_VKQ_1 = 0; i_VKQ_1 < V_rows_per_thread/2; ++i_VKQ_1) {
#pragma unroll
                    for (int j = 0; j < ncols; ++j) {
                        VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1].x += tmp[i_VKQ_1].x*KQ_k[j];
                        VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1].y += tmp[i_VKQ_1].y*KQ_k[j];
                    }
                }
            }
        }
    }

    __shared__ float KQ_max_shared[ncols][WARP_SIZE];
    __shared__ float KQ_sum_shared[ncols][WARP_SIZE];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        if (threadIdx.y == 0) {
            KQ_max_shared[j][threadIdx.x] = -FLT_MAX/2.0f;
            KQ_sum_shared[j][threadIdx.x] = 0.0f;
        }
    }

    __syncthreads();

#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        if (threadIdx.x == 0) {
            KQ_max_shared[j][threadIdx.y] = KQ_max[j];
        }
    }
    __syncthreads();

#pragma unroll
    for (int j_VKQ = 0; j_VKQ < ncols; ++j_VKQ) {
        // Columns are ordered by token, so all remaining columns are out of bounds too:
        if (ncols1 > 1 && ic0 + j_VKQ/ncols2 >= int(ne01.z)) {
            break;
        }

        float kqmax_new = KQ_max_shared[j_VKQ][threadIdx.x];
        kqmax_new = warp_reduce_max(kqmax_new);
        const float kqmax_scale = expf(KQ_max[j_VKQ] - kqmax_new);
        KQ_max[j_VKQ] = kqmax_new;

        float2 * VKQ_tmp = (float2 *) KQ + threadIdx.y*(V_cols_per_iter*D/2)
            + (nthreads_V == WARP_SIZE ? 0 : threadIdx.x / nthreads_V)*(D/2);

#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
            VKQ[j_VKQ][i_VKQ_0/nthreads_V].x *= kqmax_scale;
            VKQ[j_VKQ][i_VKQ_0/nthreads_V].y *= kqmax_scale;
        }
#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
            const int i_VKQ = i_VKQ_0 + (nthreads_V == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_V)*(V_rows_per_thread/2);

            ggml_cuda_memcpy_1<V_rows_per_thread/2*sizeof(float)>(VKQ_tmp + i_VKQ,                       &VKQ[j_VKQ][i_VKQ_0/nthreads_V]);
            ggml_cuda_memcpy_1<V_rows_per_thread/2*sizeof(float)>(VKQ_tmp + i_VKQ + V_rows_per_thread/4, &VKQ[j_VKQ][i_VKQ_0/nthreads_V + V_rows_per_thread/4]);
        }

        KQ_sum[j_VKQ] *= kqmax_scale;
        KQ_sum[j_VKQ] = warp_reduce_sum(KQ_sum[j_VKQ]);
        if (threadIdx.x == 0) {
            KQ_sum_shared[j_VKQ][threadIdx.y] = KQ_sum[j_VKQ];
        }

        __syncthreads();

        if (nthreads <= D || tid < D) {
            KQ_sum[j_VKQ] = KQ_sum_shared[j_VKQ][threadIdx.x];
            KQ_sum[j_VKQ] = warp_reduce_sum(KQ_sum[j_VKQ]);

#pragma unroll
            for (int i0 = 0; i0 < D; i0 += nthreads) {
                float dst_val = 0;
#pragma unroll
                for (int w = 0; w < nwarps; ++w) {
#pragma unroll
                    for (int v = 0; v < V_cols_per_iter; ++v) {
                        dst_val += float(KQ[w*V_cols_per_iter*D + v*D + i0 + tid]);
                    }
                }
                if (gridDim.y == 1) {
                    dst_val /= KQ_sum[j_VKQ];
                }
                dst[(((sequence*int(ne01.z) + ic0 + j_VKQ/ncols2)*ne02 + head0 + j_VKQ%ncols2)*gridDim.y + blockIdx.y)*D + i0 + tid] = dst_val;
            }
        }

        if (j_VKQ < ncols-1) {
            __syncthreads();
        }

    }

    // Unrolled instead of KQ_max[tid] so the per-column state stays in registers:
    if (gridDim.y != 1) {
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            if (tid == j && (ncols1 == 1 || ic0 + j/ncols2 < int(ne01.z))) {
                dst_meta[((sequence*int(ne01.z) + ic0 + j/ncols2)*ne02 + head0 + j%ncols2)*gridDim.y + blockIdx.y] = make_float2(KQ_max[j], KQ_sum[j]);
            }
        }
    }
#else
    GGML_UNUSED_VARS(Q_ptr, K_ptr, V_ptr, mask_ptr, sinks_ptr, KV_max_ptr, dst_ptr, dst_meta_ptr, scale,
        max_bias, m0, m1, n_head_log2, logit_softcap,
        ne00, ne01, ne02, ne03,
              nb01, nb02, nb03,
        ne10, ne11, ne12, ne13,
              nb11, nb12, nb13,
              nb21, nb22, nb23,
              ne31, ne32, ne33,
              nb31, nb32, nb33);
    NO_DEVICE_CODE;
#endif // defined(FLASH_ATTN_AVAILABLE) && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
}
#ifdef __clang__
#pragma clang diagnostic pop
#endif // __clang__

template <int D, int ncols1, int ncols2, ggml_type type_KV, bool h2>
void ggml_cuda_flash_attn_ext_vec_gqa_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    // Printed once per kernel variant so the route can be verified in logs:
    static bool logged = false;
    if (!logged) {
        logged = true;
        GGML_LOG_INFO("fattn: GQA vec D=%d ncols1=%d ncols2=%d type=%s h2=%d\n", D, ncols1, ncols2, ggml_type_name(type_KV), int(h2));
    }

    constexpr int    nwarps        = FATTN_VEC_GQA_NTHREADS / WARP_SIZE;
    constexpr size_t nbytes_shared = 0;
    constexpr int    nbatch_fa     = FATTN_VEC_GQA_NTHREADS; // KV rows per block and loop iteration.
    fattn_kernel_t fattn_kernel = flash_attn_ext_vec_gqa<D, ncols1, ncols2, type_KV, h2>;
    launch_fattn<D, ncols1, ncols2>(ctx, dst, fattn_kernel, nwarps, nbytes_shared, nbatch_fa, false, false, false, false);
}

#define DECL_FATTN_VEC_GQA_CASE(D, ncols1, ncols2, type_KV, h2)                           \
    template void ggml_cuda_flash_attn_ext_vec_gqa_case                                   \
    <D, ncols1, ncols2, type_KV, h2>(ggml_backend_cuda_context & ctx, ggml_tensor * dst) \

// (ncols1, ncols2) pairs used by ggml_cuda_flash_attn_ext_vec_gqa (fattn-vec-gqa.cu) for gqa_ratio 4, 6 and 8:
#define EXTERN_DECL_FATTN_VEC_GQA_CASES(type_KV, h2)                \
    extern DECL_FATTN_VEC_GQA_CASE(256, 1, 4, type_KV, h2);         \
    extern DECL_FATTN_VEC_GQA_CASE(256, 1, 6, type_KV, h2);         \
    extern DECL_FATTN_VEC_GQA_CASE(256, 1, 8, type_KV, h2);         \
    extern DECL_FATTN_VEC_GQA_CASE(256, 2, 3, type_KV, h2);         \
    extern DECL_FATTN_VEC_GQA_CASE(256, 2, 4, type_KV, h2);         \
    extern DECL_FATTN_VEC_GQA_CASE(256, 4, 2, type_KV, h2);         \

EXTERN_DECL_FATTN_VEC_GQA_CASES(GGML_TYPE_Q4_0, false)
EXTERN_DECL_FATTN_VEC_GQA_CASES(GGML_TYPE_Q4_0, true)
EXTERN_DECL_FATTN_VEC_GQA_CASES(GGML_TYPE_Q8_0, false)
EXTERN_DECL_FATTN_VEC_GQA_CASES(GGML_TYPE_Q8_0, true)
