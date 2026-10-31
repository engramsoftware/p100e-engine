// Q8_0 GEMV (f32 activations, repacked planes) and the generic on-the-fly-dequant GEMV (f32 accumulate).
#include "matvec_internal.cuh"
#include "dequantize.cuh"

namespace p100e::fast {

bool type_info(uint32_t t, int & blck, int & bytes) {
    switch ((ggml_type) t) {
#define TI(T, B, K) case T: blck = K; bytes = (int) sizeof(B); return true;
        TI(GGML_TYPE_Q4_0,    block_q4_0,    32)
        TI(GGML_TYPE_Q4_1,    block_q4_1,    32)
        TI(GGML_TYPE_Q5_0,    block_q5_0,    32)
        TI(GGML_TYPE_Q5_1,    block_q5_1,    32)
        TI(GGML_TYPE_Q8_0,    block_q8_0,    32)
        TI(GGML_TYPE_Q2_K,    block_q2_K,    256)
        TI(GGML_TYPE_Q3_K,    block_q3_K,    256)
        TI(GGML_TYPE_Q4_K,    block_q4_K,    256)
        TI(GGML_TYPE_Q5_K,    block_q5_K,    256)
        TI(GGML_TYPE_Q6_K,    block_q6_K,    256)
        TI(GGML_TYPE_IQ2_XXS, block_iq2_xxs, 256)
        TI(GGML_TYPE_IQ2_XS,  block_iq2_xs,  256)
        TI(GGML_TYPE_IQ2_S,   block_iq2_s,   256)
        TI(GGML_TYPE_IQ3_XXS, block_iq3_xxs, 256)
        TI(GGML_TYPE_IQ3_S,   block_iq3_s,   256)
        TI(GGML_TYPE_IQ1_S,   block_iq1_s,   256)
        TI(GGML_TYPE_IQ1_M,   block_iq1_m,   256)
        TI(GGML_TYPE_IQ4_NL,  block_iq4_nl,  32)
        TI(GGML_TYPE_IQ4_XS,  block_iq4_xs,  256)
        TI(GGML_TYPE_MXFP4,   block_mxfp4,   32)
        TI(GGML_TYPE_Q1_0,    block_q1_0,    128)
        TI(GGML_TYPE_Q2_0,    block_q2_0,    64)
        TI(GGML_TYPE_F32,     float,         1)
        TI(GGML_TYPE_F16,     half,          1)
        TI(GGML_TYPE_BF16,    uint16_t,      1)
#undef TI
        default: return false;
    }
}

// ---------------------------------------------------------------------------------------------------------
// F32 / F16 / BF16 weights (the fork's mmvf, Pascal ne11 <= 8, as a p100e GEMV):
// one warp per output row, 4 rows per block; each lane takes 4 consecutive weights per step (16-byte f32 or 8-byte
// f16 / bf16 loads, widened exactly to f32) against the f32 activations of all T columns, f32 FMA, xor-tree warp sum.
// A column's sum does not depend on T (each column has its own accumulator and the same order), so column j of a
// T-column call equals the T = 1 call. Needs n_in % 4 == 0 and 16-byte (f32) / 8-byte (f16, bf16) aligned rows.
// x that is not 16-byte aligned (VX = false): scalar x loads, one T = 1 launch per column (same sums, no misaligned fault).
// ---------------------------------------------------------------------------------------------------------
template <typename WT> struct float_w;
template <> struct float_w<float> {
    static __device__ __forceinline__ float4 load4(const char * r, int k) { return *((const float4 *) (r + 4*(size_t) k)); }
};
template <> struct float_w<half> {
    static __device__ __forceinline__ float4 load4(const char * r, int k) {
        const uint2 u = *((const uint2 *) (r + 2*(size_t) k));
        const float2 a = __half22float2(*((const half2 *) &u.x)), b = __half22float2(*((const half2 *) &u.y));
        return make_float4(a.x, a.y, b.x, b.y);
    }
};
template <> struct float_w<uint16_t> {   // bf16: the high half of an f32
    static __device__ __forceinline__ float4 load4(const char * r, int k) {
        const uint2 u = *((const uint2 *) (r + 2*(size_t) k));
        return make_float4(__uint_as_float(u.x << 16), __uint_as_float(u.x & 0xffff0000u),
                           __uint_as_float(u.y << 16), __uint_as_float(u.y & 0xffff0000u));
    }
};

// x load of 4 consecutive floats: one float4 (VX) or four scalar loads (any 4-byte aligned x)
template <bool VX> static __device__ __forceinline__ float4 load_x4(const float * p) {
    if constexpr (VX) return *((const float4 *) p);
    else return make_float4(p[0], p[1], p[2], p[3]);
}

template <typename WT, int T, bool VX = true>
static __global__ void __launch_bounds__(128) float_gemv(
        const char * __restrict__ w, const float * __restrict__ x, float * __restrict__ y,
        const int n_in, const int n_out, const size_t row_bytes) {
    const int row = blockIdx.x*4 + threadIdx.y;
    if (row >= n_out) return;
    const int lane = threadIdx.x;
    const char * r = w + (size_t) row*row_bytes;
    float acc[T];
#pragma unroll
    for (int j = 0; j < T; ++j) acc[j] = 0.0f;
    for (int k = 4*lane; k < n_in; k += 128) {
        const float4 wv = float_w<WT>::load4(r, k);
#pragma unroll
        for (int j = 0; j < T; ++j) {
            const float4 xv = load_x4<VX>(x + (size_t) j*n_in + k);
            float s = acc[j];
            s = fmaf(wv.x, xv.x, s);
            s = fmaf(wv.y, xv.y, s);
            s = fmaf(wv.z, xv.z, s);
            s = fmaf(wv.w, xv.w, s);
            acc[j] = s;
        }
    }
#pragma unroll
    for (int j = 0; j < T; ++j) {
        const float v = warp_reduce_sum(acc[j]);
        if (lane == 0) y[(size_t) j*n_out + row] = v;
    }
}

static bool is_float(uint32_t t) { return t == GGML_TYPE_F32 || t == GGML_TYPE_F16 || t == GGML_TYPE_BF16; }

template <typename WT>
static void launch_float_t(const char * w, const float * x, float * y, int T, int64_t n_in, int64_t n_out, size_t row_bytes,
                           cudaStream_t s) {
    const dim3 grid((unsigned) ((n_out + 3)/4)), blk(32, 4);
    if (((uintptr_t) x) % 16 != 0) {   // column j's sum does not depend on T: T = 1 launches give the same bits
        for (int j = 0; j < T; ++j)
            float_gemv<WT, 1, false><<<grid, blk, 0, s>>>(w, x + (size_t) j*n_in, y + (size_t) j*n_out, (int) n_in, (int) n_out, row_bytes);
        return;
    }
    switch (T) {
#define C(N) case N: float_gemv<WT, N><<<grid, blk, 0, s>>>(w, x, y, (int) n_in, (int) n_out, row_bytes); break;
        C(1) C(2) C(3) C(4) C(5) C(6) C(7) C(8)
#undef C
        default: break;
    }
}

// ---------------------------------------------------------------------------------------------------------
// Q8_0, repacked: row = [qs plane 32*nb bytes][d plane nb halves].  One warp per output row, 4 rows per block.
// Each lane takes 16 weights (one uint4) of a 32-weight block; f32 FMA against the f32 activations.
// The T > 1 kernels (matvec_wr*.cu) need a 16-byte aligned x; otherwise matvec() ends here, VX = false (as float_gemv).
// ---------------------------------------------------------------------------------------------------------
template <int T, bool VX = true>
static __global__ void __launch_bounds__(128) q8r_gemv(
        const char * __restrict__ w, const float * __restrict__ x, float * __restrict__ y,
        const int nb, const int n_in, const int n_out, const size_t row_bytes) {
    const int row = blockIdx.x*4 + threadIdx.y;
    if (row >= n_out) return;
    const int lane = threadIdx.x;
    const char * r = w + (size_t) row*row_bytes;
    float acc[T];
#pragma unroll
    for (int j = 0; j < T; ++j) acc[j] = 0.0f;

    for (int t = lane; t < 2*nb; t += 32) {
        const int b = t >> 1, h = t & 1;
        const uint4 q = *((const uint4 *) (r + 32*b + 16*h));
        const float d = __half2float(*((const half *) (r + 32*nb + 2*b)));
        const uint32_t qw[4] = { q.x, q.y, q.z, q.w };
#pragma unroll
        for (int j = 0; j < T; ++j) {
            const float * xp = x + (size_t) j*n_in + 32*b + 16*h;
            float s = 0.0f;
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                const float4 xv = load_x4<VX>(xp + 4*c);
                s = fmaf((float) (int8_t) (qw[c]      ), xv.x, s);
                s = fmaf((float) (int8_t) (qw[c] >>  8), xv.y, s);
                s = fmaf((float) (int8_t) (qw[c] >> 16), xv.z, s);
                s = fmaf((float) (int8_t) (qw[c] >> 24), xv.w, s);
            }
            acc[j] = fmaf(d, s, acc[j]);
        }
    }
#pragma unroll
    for (int j = 0; j < T; ++j) {
        const float v = warp_reduce_sum(acc[j]);
        if (lane == 0) y[(size_t) j*n_out + row] = v;
    }
}

void launch_q8r_f32(const char * w, const float * x, float * y, int T, int64_t n_in, int64_t n_out, size_t row_bytes, cudaStream_t s) {
    const dim3 grid((unsigned) ((n_out + 3)/4)), blk(32, 4);
    const int nb = (int) (n_in/32);
    if (((uintptr_t) x) % 16 != 0) {   // per-column sums: T = 1 launches give the same bits
        for (int j = 0; j < T; ++j)
            q8r_gemv<1, false><<<grid, blk, 0, s>>>(w, x + (size_t) j*n_in, y + (size_t) j*n_out, nb, (int) n_in, (int) n_out, row_bytes);
        return;
    }
    switch (T) {
#define C(N) case N: q8r_gemv<N><<<grid, blk, 0, s>>>(w, x, y, nb, (int) n_in, (int) n_out, row_bytes); break;
        C(1) C(2) C(3) C(4) C(5) C(6) C(7) C(8)
#undef C
        default: break;
    }
}

// ---------------------------------------------------------------------------------------------------------
// generic: one CUDA block per output row.  S slots of TPB threads each dequantize one "unit" (256 weights) into
// shared memory with the ggml device dequantizers (same math as ggml_get_to_fp32_cuda), then every thread
// multiplies its elements with the f32 activations of all T columns.  Accumulation is f32.
// ---------------------------------------------------------------------------------------------------------
template <ggml_type TY> struct gen_traits;
#define GT(TY, TPB, LEG) template <> struct gen_traits<TY> { static constexpr int tpb = TPB; static constexpr int legacy = LEG; };
GT(GGML_TYPE_Q4_0, 32, 1) GT(GGML_TYPE_Q4_1, 32, 1) GT(GGML_TYPE_Q5_0, 32, 1) GT(GGML_TYPE_Q5_1, 32, 1) GT(GGML_TYPE_Q8_0, 32, 2)
GT(GGML_TYPE_Q2_K, 64, 0) GT(GGML_TYPE_Q3_K, 64, 0) GT(GGML_TYPE_Q4_K, 32, 0) GT(GGML_TYPE_Q5_K, 64, 0) GT(GGML_TYPE_Q6_K, 64, 0)
GT(GGML_TYPE_IQ2_XXS, 32, 0) GT(GGML_TYPE_IQ2_XS, 32, 0) GT(GGML_TYPE_IQ2_S, 32, 0) GT(GGML_TYPE_IQ3_XXS, 32, 0)
GT(GGML_TYPE_IQ3_S, 32, 0) GT(GGML_TYPE_IQ1_S, 32, 0) GT(GGML_TYPE_IQ1_M, 32, 0) GT(GGML_TYPE_IQ4_NL, 32, 0)
GT(GGML_TYPE_IQ4_XS, 32, 0) GT(GGML_TYPE_MXFP4, 32, 0) GT(GGML_TYPE_Q1_0, 32, 0) GT(GGML_TYPE_Q2_0, 32, 0)
#undef GT

// legacy blocks (32 weights): 8 blocks per unit, 4 threads per block, 4 dequantizer calls per thread
template <ggml_type TY>
static __device__ __forceinline__ void gen_unit(const char * row, const int64_t u, float * buf, const int tid, const int64_t nbl) {
    if constexpr (gen_traits<TY>::legacy != 0) {
        const int blk = tid >> 2, q = tid & 3;
        const int64_t ib = 8*u + blk;
        if (ib >= nbl) return;
        float * yb = buf + 32*blk;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            float2 v;
            if constexpr (TY == GGML_TYPE_Q4_0) { const int iqs = 4*q + k; dequantize_q4_0(row, ib, iqs, v); yb[iqs] = v.x; yb[iqs + 16] = v.y; }
            if constexpr (TY == GGML_TYPE_Q4_1) { const int iqs = 4*q + k; dequantize_q4_1(row, ib, iqs, v); yb[iqs] = v.x; yb[iqs + 16] = v.y; }
            if constexpr (TY == GGML_TYPE_Q5_0) { const int iqs = 4*q + k; dequantize_q5_0(row, ib, iqs, v); yb[iqs] = v.x; yb[iqs + 16] = v.y; }
            if constexpr (TY == GGML_TYPE_Q5_1) { const int iqs = 4*q + k; dequantize_q5_1(row, ib, iqs, v); yb[iqs] = v.x; yb[iqs + 16] = v.y; }
            if constexpr (TY == GGML_TYPE_Q8_0) { const int iqs = 2*(4*q + k); dequantize_q8_0(row, ib, iqs, v); yb[iqs] = v.x; yb[iqs + 1] = v.y; }
        }
    } else {
        // nbl counts native blocks: 256-blocks (K / IQ types: one per unit) or the sub-unit blocks of IQ4_NL / MXFP4 (32),
        // Q2_0 (64) and Q1_0 (128), so a partial last unit (n_in % 256 != 0) decodes exactly its real blocks
        if constexpr (TY == GGML_TYPE_IQ4_NL || TY == GGML_TYPE_MXFP4) { if (8*u + (tid & 7) >= nbl) return; }   // tid % 8 = its block
        else if constexpr (TY == GGML_TYPE_Q1_0 || TY == GGML_TYPE_Q2_0) { if ((TY == GGML_TYPE_Q1_0 ? 2 : 4)*u >= nbl) return; }
        else if (u >= nbl) return;
        if constexpr (TY == GGML_TYPE_Q2_K)    dequantize_q2_K<float>(row, u, buf, tid);
        if constexpr (TY == GGML_TYPE_Q3_K)    dequantize_q3_K<float>(row, u, buf, tid);
        if constexpr (TY == GGML_TYPE_Q4_K)    dequantize_q4_K<float>(row, u, buf, tid);
        if constexpr (TY == GGML_TYPE_Q5_K)    dequantize_q5_K<float>(row, u, buf, tid);
        if constexpr (TY == GGML_TYPE_Q6_K)    dequantize_q6_K<float>(row, u, buf, tid);
        if constexpr (TY == GGML_TYPE_IQ2_XXS) dequantize_iq2_xxs<float>(row, u, buf, tid);
        if constexpr (TY == GGML_TYPE_IQ2_XS)  dequantize_iq2_xs<float>(row, u, buf, tid);
        if constexpr (TY == GGML_TYPE_IQ2_S)   dequantize_iq2_s<float>(row, u, buf, tid);
        if constexpr (TY == GGML_TYPE_IQ3_XXS) dequantize_iq3_xxs<float>(row, u, buf, tid);
        if constexpr (TY == GGML_TYPE_IQ3_S)   dequantize_iq3_s<float>(row, u, buf, tid);
        if constexpr (TY == GGML_TYPE_IQ1_S)   dequantize_iq1_s<float>(row, u, buf, tid);
        if constexpr (TY == GGML_TYPE_IQ1_M)   dequantize_iq1_m<float>(row, u, buf, tid);
        if constexpr (TY == GGML_TYPE_IQ4_NL)  dequantize_iq4_nl<float>(row, u, buf, tid);
        if constexpr (TY == GGML_TYPE_IQ4_XS)  dequantize_iq4_xs<float>(row, u, buf, tid);
        if constexpr (TY == GGML_TYPE_MXFP4)   dequantize_mxfp4<float>(row, u, buf, tid);
        if constexpr (TY == GGML_TYPE_Q1_0)    dequantize_q1_0_unit<float>(row, u, buf, tid, nbl);   // per-block guard
        if constexpr (TY == GGML_TYPE_Q2_0)    dequantize_q2_0_unit<float>(row, u, buf, tid, nbl);
    }
}

template <ggml_type TY>
static __global__ void __launch_bounds__(128) generic_gemv(
        const char * __restrict__ w, const float * __restrict__ x, float * __restrict__ y,
        const int T, const int64_t n_in, const int64_t n_out, const size_t row_bytes, const int64_t nbl /* native blocks, see gen_unit */) {
    constexpr int TPB = gen_traits<TY>::tpb;
    constexpr int S   = 128/TPB;
    __shared__ float buf[S][256];
    __shared__ float red[4][8];
    const int row = blockIdx.x;
    const char * r = w + (size_t) row*row_bytes;
    const int slot = threadIdx.x / TPB, tid = threadIdx.x % TPB;
    const int64_t nunits = (n_in + 255)/256;

    float acc[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[j] = 0.0f;

    for (int64_t u0 = 0; u0 < nunits; u0 += S) {
        const int64_t u = u0 + slot;
        gen_unit<TY>(r, u, buf[slot], tid, nbl);
        __syncthreads();
        if (u < nunits) {
            const int64_t k0 = u*256;
            const int cnt = (int) min((int64_t) 256, n_in - k0);
            for (int e = tid; e < cnt; e += TPB) {
                const float wv = buf[slot][e];
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    if (j < T) acc[j] = fmaf(wv, x[(size_t) j*n_in + k0 + e], acc[j]);
                }
            }
        }
        __syncthreads();
    }
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const float v = warp_reduce_sum(acc[j]);
        if (lane == 0) red[warp][j] = v;
    }
    __syncthreads();
    if (threadIdx.x < T) {
        y[(size_t) threadIdx.x*n_out + row] = red[0][threadIdx.x] + red[1][threadIdx.x] + red[2][threadIdx.x] + red[3][threadIdx.x];
    }
}

static bool is_legacy(uint32_t t) {
    return t == GGML_TYPE_Q4_0 || t == GGML_TYPE_Q4_1 || t == GGML_TYPE_Q5_0 || t == GGML_TYPE_Q5_1 || t == GGML_TYPE_Q8_0;
}

bool generic_supported(uint32_t type) { int b, y; return type_info(type, b, y); }

// n_in must be a multiple of this (the type's block: generic_gemv handles a partial last 256-weight unit)
int64_t generic_group(uint32_t type) { int b = 256, y = 0; return is_float(type) ? 4 : type_info(type, b, y) ? b : 256; }

void launch_generic(uint32_t type, const char * w, const float * x, float * y, int T, int64_t n_in, int64_t n_out,
                    size_t row_bytes, cudaStream_t s) {
    if (is_float(type)) {
        if (type == GGML_TYPE_F32) launch_float_t<float>(w, x, y, T, n_in, n_out, row_bytes, s);
        else if (type == GGML_TYPE_F16) launch_float_t<half>(w, x, y, T, n_in, n_out, row_bytes, s);
        else launch_float_t<uint16_t>(w, x, y, T, n_in, n_out, row_bytes, s);
        return;
    }
    const dim3 grid((unsigned) n_out);
    // native block count (pack_weight checked n_in % blck == 0). Was n_in/256 for every non-legacy type, which dropped the
    // last partial unit of IQ4_NL / MXFP4 / Q1_0 / Q2_0 rows and summed stale shared memory in its place.
    int blck = 256, bytes = 0;
    type_info(type, blck, bytes);
    const int64_t nbl = n_in/blck;
    switch ((ggml_type) type) {
#define G(TY) case TY: generic_gemv<TY><<<grid, 128, 0, s>>>(w, x, y, T, n_in, n_out, row_bytes, nbl); break;
        G(GGML_TYPE_Q4_0) G(GGML_TYPE_Q4_1) G(GGML_TYPE_Q5_0) G(GGML_TYPE_Q5_1) G(GGML_TYPE_Q8_0)
        G(GGML_TYPE_Q2_K) G(GGML_TYPE_Q3_K) G(GGML_TYPE_Q4_K) G(GGML_TYPE_Q5_K) G(GGML_TYPE_Q6_K)
        G(GGML_TYPE_IQ2_XXS) G(GGML_TYPE_IQ2_XS) G(GGML_TYPE_IQ2_S) G(GGML_TYPE_IQ3_XXS) G(GGML_TYPE_IQ3_S)
        G(GGML_TYPE_IQ1_S) G(GGML_TYPE_IQ1_M) G(GGML_TYPE_IQ4_NL) G(GGML_TYPE_IQ4_XS) G(GGML_TYPE_MXFP4) G(GGML_TYPE_Q1_0) G(GGML_TYPE_Q2_0)
#undef G
        default: break;
    }
}

} // namespace p100e::fast
