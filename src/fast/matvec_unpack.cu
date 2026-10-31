// p100e::fast packed_to_f16, see unpack.cuh.
// One CUDA block per row: the packed row is staged in shared memory, un-permuted to the raw GGUF row in a second
// shared buffer (inverse of the repack kernels in matvec_pack_body.inc / matvec.cu), then dequantized with ggml's own
// device dequant functions (dequantize.cuh) or a verbatim copy of ggml's q4_0 / q8_0 f16 expressions, so the f16
// values are bit-identical to ggml_get_to_fp16_cuda on the raw rows.
#include "fast/unpack.cuh"
#include "matvec_internal.cuh"
#include "dequantize.cuh"   // third_party/ggml: dequantize_q3_K / q4_K / q6_K device functions
#include "convert.cuh"
#include "gguf.h"       // get_type_info

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace p100e::fast {

#include "matvec_pack_body.inc"   // q3r slot helpers (the repack kernels themselves are unused here)

namespace {

constexpr int UNPACK_THREADS = 256;

// ---- packed row (smem) -> raw row (smem); inverse of each pack ----

__device__ void unpack_q4r(const uint16_t * p, uint16_t * r, int nb) {          // Q4_0, nb = n_in/32
    for (int i = threadIdx.x; i < 9*nb; i += blockDim.x) {
        const int b = i / 9, k = i - 9*b;
        r[i] = k == 0 ? p[8*nb + b] : p[8*b + k - 1];
    }
}

__device__ void unpack_q8r(const uint16_t * p, uint16_t * r, int nb) {          // Q8_0, nb = n_in/32
    for (int i = threadIdx.x; i < 17*nb; i += blockDim.x) {
        const int b = i / 17, k = i - 17*b;
        r[i] = k == 0 ? p[16*nb + b] : p[16*b + k - 1];
    }
}

__device__ void unpack_q4kr(const uint32_t * row32, uint32_t * d, int nb) {     // Q4_K, nb = n_in/256
    for (int o = threadIdx.x; o < 36*nb; o += blockDim.x) {
        uint32_t v;
        const int sb = o / 36;
        const int r  = o - 36*sb;
        if (r < 4) {
            v = row32[32*nb + 4*sb + r];
        } else {
            const int qi = r - 4;
            const int j  = qi >> 3;
            const int hh = (qi >> 2) & 1;
            const int w  = qi & 3;
            const uint32_t w0 = row32[32*sb + 8*j     + w];
            const uint32_t w1 = row32[32*sb + 8*j + 4 + w];
            v = hh == 0 ? ((w0 & 0x0F0F0F0Fu) | ((w1 & 0x0F0F0F0Fu) << 4))
                        : (((w0 >> 4) & 0x0F0F0F0Fu) | (w1 & 0xF0F0F0F0u));
        }
        d[o] = v;
    }
}

__device__ void unpack_q3r(const uint8_t * row, uint8_t * d, int nb) {          // Q3_K, nb = n_in/256
    const uint32_t * gw = (const uint32_t *) row;
    for (int o = threadIdx.x; o < 110*nb; o += blockDim.x) {
        uint32_t v = 0;
        const int sb = o/110, r = o - 110*sb;
        if (r < 32) {
            for (int t = 0; t < 8; ++t) {
                int s, half;
                q3r_slot_of(r, s, half);
                const uint32_t H = gw[3*(8*sb + t) + 2];
                v |= ((H >> q3r_hpos(s, half)) & 1u) << t;
            }
        } else if (r < 96) {
            const int q = r - 32, n = q >> 5, val = q & 31;
            for (int j = 0; j < 4; ++j) {
                int s, half;
                q3r_slot_of(val, s, half);
                const uint32_t W = gw[3*(8*sb + 4*n + j) + (s >> 3)];
                v |= ((W >> q3r_qpos(s, half)) & 3u) << (2*j);
            }
        } else if (r < 108) {
            v = row[96*nb + 12*sb + (r - 96)];
        } else {
            v = row[108*nb + 2*sb + (r - 108)];
        }
        d[o] = (uint8_t) v;
    }
}

__device__ void unpack_q2kr(const uint8_t * row, uint8_t * d, int nb) {         // Q2_K, nb = n_in/256
    const uint32_t * gw = (const uint32_t *) row;
    for (int o = threadIdx.x; o < 84*nb; o += blockDim.x) {
        uint32_t v = 0;
        const int sb = o/84, r = o - 84*sb;
        if (r < 16) {
            v = row[64*nb + 20*sb + r];
        } else if (r < 80) {
            const int q = r - 16, n = q >> 5, val = q & 31;
            for (int j = 0; j < 4; ++j) {
                int s, half;
                q3r_slot_of(val, s, half);
                const uint32_t W = gw[2*(8*sb + 4*n + j) + (s >> 3)];
                v |= ((W >> q3r_qpos(s, half)) & 3u) << (2*j);
            }
        } else {
            v = row[64*nb + 20*sb + 16 + (r - 80)];
        }
        d[o] = (uint8_t) v;
    }
}

__device__ void unpack_iq3xxsr(const uint8_t * row, uint8_t * d, int nb) {      // IQ3_XXS, nb = n_in/256
    for (int o = threadIdx.x; o < 98*nb; o += blockDim.x) {
        const int sb = o/98, r = o - 98*sb;
        d[o] = r < 2 ? row[96*nb + 2*sb + r] : r < 66 ? row[64*sb + r - 2] : row[64*nb + 32*sb + r - 66];
    }
}

__device__ void unpack_iq3sr(const uint8_t * row, uint8_t * d, int nb) {        // IQ3_S, nb = n_in/256
    for (int o = threadIdx.x; o < 110*nb; o += blockDim.x) {
        const int sb = o/110, r = o - 110*sb;
        d[o] = r < 2 ? row[108*nb + 2*sb + r] : r < 66 ? row[64*sb + r - 2] : r < 74 ? row[96*nb + 8*sb + r - 66]
             : r < 106 ? row[64*nb + 32*sb + r - 74] : row[104*nb + 4*sb + r - 106];
    }
}

// Q6_K step 1: q6r2 planes -> q6r planes (inverse of q6r2_permute); scales and d planes copied
__device__ void unpack_q6r2_planes(const uint8_t * s, uint8_t * d, int nb) {
    const uint8_t * ql = s;
    const uint8_t * qh = s + 128*nb;
    for (int o = threadIdx.x; o < 128*nb; o += blockDim.x) {
        const int sb = o >> 7, r = o & 127, n = r >> 6, a = (r >> 5) & 1, l = r & 31;
        const int j = l & 15, hi = l >> 4;
        const int G0 = 8*sb + 4*n + a, G2 = G0 + 2;
        const int lo4 = (ql[16*G0 + j] >> (4*hi)) & 0xF;
        const int hi4 = (ql[16*G2 + j] >> (4*hi)) & 0xF;
        d[o] = lo4 | (hi4 << 4);
    }
    for (int o = threadIdx.x; o < 64*nb; o += blockDim.x) {
        const int sb = o >> 6, r = o & 63, n = r >> 5, l = r & 31, i = l >> 2, p = l & 3;
        uint8_t v = 0;
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            v |= ((qh[8*(8*sb + 4*n + t) + i] >> (2*p)) & 3) << (2*t);
        }
        d[128*nb + o] = v;
    }
    for (int o = 192*nb + threadIdx.x; o < 210*nb; o += blockDim.x) d[o] = s[o];
}

// Q6_K step 2: q6r planes -> standard blocks (inverse of q6r_convert)
__device__ void unpack_q6r(const uint16_t * row, uint16_t * d, int nb) {
    for (int i = threadIdx.x; i < 105*nb; i += blockDim.x) {
        const int b = i / 105, k = i - 105*b;
        int src_i;
        if      (k <  64) { src_i = 64*b + k; }
        else if (k <  96) { src_i = 64*nb + 32*b + (k - 64); }
        else if (k < 104) { src_i = 96*nb +  8*b + (k - 96); }
        else              { src_i = 104*nb + b; }
        d[i] = row[src_i];
    }
}

// ---- raw row (smem) -> f16 (global), same arithmetic as ggml's to_fp16 converters ----

__device__ void deq_q4_0(const uint8_t * raw, half * y, int nb) {
    const block_q4_0 * x = (const block_q4_0 *) raw;
    for (int p = threadIdx.x; p < 16*nb; p += blockDim.x) {
        const int b = p >> 4, l = p & 15;
        const float d = __half2float(x[b].d);
        const float dm = -8*d;
        const uint8_t q = x[b].qs[l];
        y[32*b + l +  0] = ggml_cuda_cast<half>(d * (q & 0xF) + dm);
        y[32*b + l + 16] = ggml_cuda_cast<half>(d * (q >>  4) + dm);
    }
}

__device__ void deq_q8_0(const uint8_t * raw, half * y, int nb) {   // ggml dequantize_block_q8_0_f16 (fp16 path)
    const half * h = (const half *) raw;   // block = 17 halves: d, 16 x char2
    half2 * y2 = (half2 *) y;
    for (int p = threadIdx.x; p < 16*nb; p += blockDim.x) {
        const half * b0 = h + 17*(p >> 4);
        const half d = *b0;
        const char2 qs = ((const char2 *) (b0 + 1))[p & 15];
        y2[p] = __hmul2(make_half2(qs.x, qs.y), __half2half2(d));
    }
}

// ggml dequantize_q4_K<half> with the same expressions and casts (bit-identical values); a thread's 4 + 4 halves
// go out as two 8-byte stores instead of eight strided 2-byte stores
__device__ __forceinline__ void deq_q4_K_vec(const void * vx, const int64_t ib, half * yy, const int tid) {
    const block_q4_K * x = (const block_q4_K *) vx;
    const int64_t il = tid / 8;
    const int64_t ir = tid % 8;
    const int64_t is = 2 * il;
    const int64_t n = 4;
    half * y = yy + 64 * il + n * ir;
    const float dall = __low2half(x[ib].dm);
    const float dmin = __high2half(x[ib].dm);
    const uint8_t * q = x[ib].qs + 32 * il + n * ir;
    uint8_t sc, m;
    get_scale_min_k4(is + 0, x[ib].scales, sc, m);
    const float d1 = dall * sc; const float m1 = dmin * m;
    get_scale_min_k4(is + 1, x[ib].scales, sc, m);
    const float d2 = dall * sc; const float m2 = dmin * m;
    union { half h[4]; uint2 u; } lo, hi;
#pragma unroll
    for (int l = 0; l < n; ++l) {
        lo.h[l] = ggml_cuda_cast<half>(d1 * (q[l] & 0xF) - m1);
        hi.h[l] = ggml_cuda_cast<half>(d2 * (q[l] >>  4) - m2);
    }
    *(uint2 *) y = lo.u;
    *(uint2 *) (y + 32) = hi.u;
}

template <int LAYOUT>
__global__ void __launch_bounds__(UNPACK_THREADS) unpack_f16_kernel(const char * __restrict__ w, half * __restrict__ dst,
                                                                    const int64_t row0, const int n_in,
                                                                    const int row_bytes, const int boff) {
    extern __shared__ __align__(16) uint8_t sm[];
    uint8_t * A = sm;
    uint8_t * B = sm + boff;
    const int64_t r = blockIdx.x;
    {
        const uint32_t * s = (const uint32_t *) (w + (row0 + r) * row_bytes);
        for (int i = threadIdx.x; i < row_bytes / 4; i += blockDim.x) ((uint32_t *) A)[i] = s[i];
    }
    __syncthreads();
    half * y = dst + r * n_in;
    const int tid = threadIdx.x;
    if constexpr (LAYOUT == LAYOUT_Q4R) {
        const int nb = n_in / 32;
        unpack_q4r((const uint16_t *) A, (uint16_t *) B, nb);
        __syncthreads();
        deq_q4_0(B, y, nb);
    } else if constexpr (LAYOUT == LAYOUT_Q8R) {
        const int nb = n_in / 32;
        unpack_q8r((const uint16_t *) A, (uint16_t *) B, nb);
        __syncthreads();
        deq_q8_0(B, y, nb);
    } else if constexpr (LAYOUT == LAYOUT_Q4KR) {
        const int nb = n_in / 256;
        unpack_q4kr((const uint32_t *) A, (uint32_t *) B, nb);
        __syncthreads();
        for (int ib = tid / 32; ib < nb; ib += blockDim.x / 32) deq_q4_K_vec(B, ib, y + 256*ib, tid & 31);
    } else if constexpr (LAYOUT == LAYOUT_Q3R) {
        const int nb = n_in / 256;
        unpack_q3r(A, B, nb);
        __syncthreads();
        for (int ib = tid / 64; ib < nb; ib += blockDim.x / 64) dequantize_q3_K(B, ib, y + 256*ib, tid & 63);
    } else if constexpr (LAYOUT == LAYOUT_Q2KR) {
        const int nb = n_in / 256;
        unpack_q2kr(A, B, nb);
        __syncthreads();
        for (int ib = tid / 64; ib < nb; ib += blockDim.x / 64) dequantize_q2_K(B, ib, y + 256*ib, tid & 63);
    } else if constexpr (LAYOUT == LAYOUT_IQ3XXSR) {
        const int nb = n_in / 256;
        unpack_iq3xxsr(A, B, nb);
        __syncthreads();
        for (int ib = tid / 32; ib < nb; ib += blockDim.x / 32) dequantize_iq3_xxs(B, ib, y + 256*ib, tid & 31);
    } else if constexpr (LAYOUT == LAYOUT_IQ3SR) {
        const int nb = n_in / 256;
        unpack_iq3sr(A, B, nb);
        __syncthreads();
        for (int ib = tid / 32; ib < nb; ib += blockDim.x / 32) dequantize_iq3_s(B, ib, y + 256*ib, tid & 31);
    } else if constexpr (LAYOUT == LAYOUT_Q5KR) {
        const int nb = n_in / 256;
        for (int o = tid; o < 44*nb; o += blockDim.x) ((uint32_t *) B)[o] = q5kr_word((const uint32_t *) A, nb, o, false);
        __syncthreads();
        for (int ib = tid / 64; ib < nb; ib += blockDim.x / 64) dequantize_q5_K(B, ib, y + 256*ib, tid & 63);
    } else if constexpr (LAYOUT == LAYOUT_Q6R2 || LAYOUT == LAYOUT_Q6R3) {
        const int nb = n_in / 256;
        if constexpr (LAYOUT == LAYOUT_Q6R3) {   // q6r3 -> q6r2 qh records, in place
            for (int G = tid; G < 8*nb; G += blockDim.x) q6r3_qh_permute(A + 128*nb + 8*G, false);
            __syncthreads();
        }
        unpack_q6r2_planes(A, B, nb);
        __syncthreads();
        unpack_q6r((const uint16_t *) B, (uint16_t *) A, nb);
        __syncthreads();
        for (int ib = tid / 64; ib < nb; ib += blockDim.x / 64) dequantize_q6_K(A, ib, y + 256*ib, tid & 63);
    }
}

template <int LAYOUT>
void launch(const packed_weight & w, int64_t row0, int64_t nrows, half * dst, int row_bytes, cudaStream_t s) {
    const int boff = (row_bytes + 15) & ~15;
    const size_t smem = (size_t) boff + row_bytes;
    if (smem > 48 * 1024 || row_bytes % 4 != 0)
        throw std::runtime_error("fast::packed_to_f16: row too large for shared memory (" + std::to_string(row_bytes) + " B)");
    for (int64_t r = 0; r < nrows; r += 65535 * 1024LL) {   // grid.x limit is 2^31-1; chunk generously
        const int64_t n = std::min<int64_t>(nrows - r, 65535 * 1024LL);
        unpack_f16_kernel<LAYOUT><<<(unsigned) n, UNPACK_THREADS, smem, s>>>((const char *) w.data, dst + r * w.n_in,
                                                                             row0 + r, (int) w.n_in, row_bytes, boff);
    }
}

// =====================================================================================================================
// Direct dequant-to-f16 (fork q4r-sm60.cu q4r_to_f16 / q6r2_to_f16 / q4kr_to_f16 / q3r_to_f16): a flat
// grid over the whole row range, each thread (or warp) reading its group straight from the packed planes with 32-bit
// loads and writing its halves with 8- / 16-byte stores; no shared memory, no un-permute pass. Every element is the
// same expression as ggml's to-f16 dequant of the raw block (the smem kernels above), so the f16 bits are identical
// (tests/fast_unpack_test.cu compares both against ggml on every layout). q6r3, q5kr and q8r are p100e layouts,
// written in the same style. w points at the first row of the range, y at its first output.
// =====================================================================================================================
constexpr int DIRECT_THREADS = 256;

// Q4_0 (q4r): one thread per 4 qs bytes = elements j..j+3 and j+16..j+19 of a block (two 8-byte stores)
__global__ void __launch_bounds__(DIRECT_THREADS) q4r_direct_k(const char * __restrict__ w, half * __restrict__ y,
                                                              const int nb, const int row_bytes, const int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;   // 4-byte group over the range
    if (i >= n) return;
    const int64_t row = i / (4 * nb);
    const int g = (int) (i - row * 4 * nb);
    const int kb = g >> 2;
    const int j = 4 * (g & 3);
    const char * xr = w + row * row_bytes;
    const uint32_t q = *((const uint32_t *) (xr + 4 * g));
    const float d = __half2float(*((const half *) (xr + 16 * nb + 2 * kb)));
    const float dm = -8 * d;
    union { half h[4]; uint2 u; } lo, hi;
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        const uint32_t b = (q >> (8 * l)) & 0xFF;
        lo.h[l] = ggml_cuda_cast<half>(d * (b & 0xF) + dm);   // dequantize_block_q4_0's expression
        hi.h[l] = ggml_cuda_cast<half>(d * (b >> 4) + dm);
    }
    half * yr = y + row * nb * 32 + kb * 32 + j;
    *((uint2 *) (yr)) = lo.u;
    *((uint2 *) (yr + 16)) = hi.u;
}

// Q8_0 (q8r): one thread per 8 values (two 32-bit qs loads, one 16-byte store), dequantize_block_q8_0_f16's HMUL2
__global__ void __launch_bounds__(DIRECT_THREADS) q8r_direct_k(const char * __restrict__ w, half * __restrict__ y,
                                                              const int nb, const int row_bytes, const int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;   // 8-value unit over the range
    if (i >= n) return;
    const int64_t row = i / (4 * nb);
    const int u = (int) (i - row * 4 * nb);
    const int kb = u >> 2;
    const char * xr = w + row * row_bytes;
    const uint2 q = *((const uint2 *) (xr + 8 * u));
    const half d = *((const half *) (xr + 32 * nb + 2 * kb));
    const half2 d2 = __half2half2(d);
    const uint32_t qw[2] = {q.x, q.y};
    union { half2 h[4]; uint4 u; } o;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const uint32_t v = qw[k >> 1] >> (16 * (k & 1));
        const char2 c = make_char2((signed char) (v & 0xFF), (signed char) (v >> 8));
        o.h[k] = __hmul2(make_half2(c.x, c.y), d2);
    }
    *((uint4 *) (y + row * nb * 32 + 8 * u)) = o.u;
}

// get_scale_min_k4 (q = the 12 scale bytes)
__device__ __forceinline__ void kq_scale_min(const int j, const uint8_t * q, uint8_t & d, uint8_t & m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >>  4) | ((q[j - 0] >> 6) << 4);
    }
}

// Q4_K (q4kr) and Q5_K (q5kr): one warp per super-block (8 per CUDA block); lane = u32 w of group g of the qs plane,
// elements 4w..4w+3 (low nibbles) and 4w+16..4w+19 (high nibbles) of the group as two 8-byte stores. The q4kr header
// plane starts at byte 128*nb, q5kr's at 160*nb after its qh plane (one u32 per group, high bit of value v at bit
// q5kr_pos(v)). Expressions of dequantize_q4_K / dequantize_q5_K.
template <bool Q5>
__global__ void __launch_bounds__(DIRECT_THREADS) kq_direct_k(const char * __restrict__ w, half * __restrict__ y,
                                                             const int nb, const int row_bytes, const int nsb) {
    const int gsb = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (gsb >= nsb) return;
    const int row = gsb / nb;
    const int sb = gsb - row * nb;
    const int lane = threadIdx.x & 31;
    const int g = lane >> 2;
    const int wi = lane & 3;
    const char * xr = w + (size_t) row * row_bytes;
    const uint32_t q = *((const uint32_t *) (xr + 128 * sb + 4 * lane));
    const uint8_t * hdr = (const uint8_t *) (xr + (Q5 ? 160 : 128) * nb + 16 * sb);
    const half2 dm = *((const half2 *) hdr);
    const float dall = __low2half(dm);
    const float dmin = __high2half(dm);
    uint8_t sc, m;
    kq_scale_min(g, hdr + 4, sc, m);
    const float d1 = dall * sc; const float m1 = dmin * m;
    uint32_t H = 0;
    if constexpr (Q5) H = *((const uint32_t *) (xr + 128 * nb + 32 * sb + 4 * g));
    union { half h[4]; uint2 u; } lo, hi;
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        const uint8_t ql = (q >> (8 * l)) & 0xFF;
        if constexpr (Q5) {
            const int v = 4 * wi + l;   // value index of the low nibble in the group; the high nibble is v + 16
            const int hl = (H >> q5kr_pos(v)) & 1, hh = (H >> q5kr_pos(v + 16)) & 1;
            lo.h[l] = ggml_cuda_cast<half>(d1 * ((ql & 0xF) + (hl ? 16 : 0)) - m1);
            hi.h[l] = ggml_cuda_cast<half>(d1 * ((ql >>  4) + (hh ? 16 : 0)) - m1);
        } else {
            lo.h[l] = ggml_cuda_cast<half>(d1 * (ql & 0xF) - m1);
            hi.h[l] = ggml_cuda_cast<half>(d1 * (ql >>  4) - m1);
        }
    }
    half * yr = y + (size_t) gsb * 256 + 32 * g + 4 * wi;
    *((uint2 *) (yr)) = lo.u;
    *((uint2 *) (yr + 16)) = hi.u;
}

// Q3_K (q3r): the fork's q3r_to_f16 (same layout as p100e's q3r: q3r_qpos / q3r_hpos / q3r_slot_of). One thread per
// 32-value group: 3 group words + 2 scale words + d, 32 halves out as four 16-byte stores. dequantize_q3_K's arithmetic.
__global__ void __launch_bounds__(DIRECT_THREADS) q3r_direct_k(const char * __restrict__ w, half * __restrict__ y,
                                                              const int nb, const int row_bytes, const int64_t ngroups) {
    const int64_t gt = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (gt >= ngroups) return;
    const int64_t row = gt / (8 * nb);
    const int t = (int) (gt - row * 8 * nb);
    const int sb = t >> 3, g = t & 7;
    const char * xr = w + row * row_bytes;
    const uint32_t * gw = (const uint32_t *) xr + 3 * t;
    const uint32_t A = gw[0], B = gw[1], H = gw[2];
    const uint8_t * sc = (const uint8_t *) (xr + 96 * nb + 12 * sb);
    const float d_all = __half2float(*((const half *) (xr + 108 * nb + 2 * sb)));
    float dl[2];
#pragma unroll
    for (int k = 0; k < 2; ++k) {
        const int is = 2 * g + k;
        const int8_t us = is <  4 ? (sc[is - 0] & 0xF) | (((sc[is + 8] >> 0) & 3) << 4) :
                          is <  8 ? (sc[is - 0] & 0xF) | (((sc[is + 4] >> 2) & 3) << 4) :
                          is < 12 ? (sc[is - 8] >>  4) | (((sc[is + 0] >> 4) & 3) << 4) :
                                    (sc[is - 8] >>  4) | (((sc[is - 4] >> 6) & 3) << 4);
        dl[k] = d_all * (us - 32);
    }
    const uint32_t V[4] = {H, H >> 1, __funnelshift_r(H, H, 8), __funnelshift_r(H, H, 9)};
    __align__(16) half out[32];
#pragma unroll
    for (int s = 0; s < 16; ++s) {
        const int k = s >> 2, r = s & 3;
        const uint32_t W = (k < 2 ? A : B) >> (8 * (k & 1));
        const int lo = 4 * k + 16 * (r & 1) + (r >> 1);
#pragma unroll
        for (int half_i = 0; half_i < 2; ++half_i) {
            const int q2 = (W >> (2 * r + 16 * half_i)) & 3;
            const int h = (V[k] >> (2 * r + 2 + 16 * half_i)) & 1;
            const int v = lo + 2 * half_i;
            out[v] = ggml_cuda_cast<half>(dl[v >> 4] * ((int8_t) q2 - (h ? 0 : 4)));
        }
    }
    uint4 * yo = (uint4 *) (y + (size_t) gt * 32);
#pragma unroll
    for (int c = 0; c < 4; ++c) yo[c] = ((const uint4 *) out)[c];
}

// Q2_K (q2kr): one thread per 32-value group (A, B words, 16 scale bytes of the sb, d / dmin), dequantize_q2_K's arithmetic
// (dall * (sc & 0xF) * q - dmin * (sc >> 4)); 32 halves out as four 16-byte stores.
__global__ void __launch_bounds__(DIRECT_THREADS) q2kr_direct_k(const char * __restrict__ w, half * __restrict__ y,
                                                               const int nb, const int row_bytes, const int64_t ngroups) {
    const int64_t gt = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (gt >= ngroups) return;
    const int64_t row = gt / (8 * nb);
    const int t = (int) (gt - row * 8 * nb);
    const int sb = t >> 3, g = t & 7;
    const char * xr = w + row * row_bytes;
    const uint2 AB = *((const uint2 *) (xr + 8 * t));   // 8-byte aligned (rb % 8 == 0 is required by pack_weight)
    const uint8_t * hd = (const uint8_t *) (xr + 64 * nb + 20 * sb);
    const half2 dm = *((const half2 *) (hd + 16));
    const float dall = __low2half(dm), dmin = __high2half(dm);
    const uint8_t sc[2] = {hd[2 * g], hd[2 * g + 1]};
    __align__(16) half out[32];
#pragma unroll
    for (int s = 0; s < 16; ++s) {
        const int k = s >> 2, r = s & 3;
        const uint32_t W = (k < 2 ? AB.x : AB.y) >> (8 * (k & 1));
        const int lo = 4 * k + 16 * (r & 1) + (r >> 1);
#pragma unroll
        for (int half_i = 0; half_i < 2; ++half_i) {
            const int q2 = (W >> (2 * r + 16 * half_i)) & 3;
            const int v = lo + 2 * half_i;
            const uint8_t sv = sc[v >> 4];
            out[v] = ggml_cuda_cast<half>(dall * (sv & 0xF) * q2 - dmin * (sv >> 4));
        }
    }
    uint4 * yo = (uint4 *) (y + (size_t) gt * 32);
#pragma unroll
    for (int c = 0; c < 4; ++c) yo[c] = ((const uint4 *) out)[c];
}

// IQ3_XXS / IQ3_S (iq3xxsr / iq3sr): one thread per 32-value group; dequantize_iq3_xxs / dequantize_iq3_s arithmetic
// (d * grid byte * +-1, in float, then to half); 32 halves out as four 16-byte stores.
template <bool S>
__global__ void __launch_bounds__(DIRECT_THREADS) iq3r_direct_k(const char * __restrict__ w, half * __restrict__ y,
                                                               const int nb, const int row_bytes, const int64_t ngroups) {
    const int64_t gt = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (gt >= ngroups) return;
    const int64_t row = gt / (8 * nb);
    const int t = (int) (gt - row * 8 * nb);
    const int sb = t >> 3, g = t & 7;
    const char * xr = w + row * row_bytes;
    uint32_t qw[2] = {*((const uint32_t *) (xr + 8 * t)), *((const uint32_t *) (xr + 8 * t + 4))};
    const uint32_t sx = *((const uint32_t *) (xr + 64 * nb + 4 * t));
    float d;
    uint32_t qh = 0;
    if (S) {
        qh = *((const uint8_t *) (xr + 96 * nb + t));
        const int sc = (*((const uint8_t *) (xr + 104 * nb + 4 * sb + (g >> 1))) >> (4 * (g & 1))) & 0xf;
        d = (float) *((const half *) (xr + 108 * nb + 2 * sb)) * (1 + 2 * sc);
    } else {
        d = (float) *((const half *) (xr + 96 * nb + 2 * sb)) * (0.5f + (sx >> 28)) * 0.5f;
    }
    __align__(16) half out[32];
#pragma unroll
    for (int il = 0; il < 4; ++il) {
        uint32_t i1 = (qw[il >> 1] >> (16 * (il & 1))) & 0xff, i2 = (qw[il >> 1] >> (16 * (il & 1) + 8)) & 0xff;
        uint8_t signs;
        if (S) {
            i1 |= (qh << (8 - 2 * il)) & 256;
            i2 |= (qh << (7 - 2 * il)) & 256;
            signs = (sx >> (8 * il)) & 0xff;
        } else {
            signs = ksigns_iq2xs[(sx >> (7 * il)) & 127];
        }
        const uint8_t * grid1 = (const uint8_t *) ((S ? iq3s_grid : iq3xxs_grid) + i1);
        const uint8_t * grid2 = (const uint8_t *) ((S ? iq3s_grid : iq3xxs_grid) + i2);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            out[8 * il + j]     = ggml_cuda_cast<half>(d * grid1[j] * (signs & kmask_iq2xs[j + 0] ? -1.f : 1.f));
            out[8 * il + j + 4] = ggml_cuda_cast<half>(d * grid2[j] * (signs & kmask_iq2xs[j + 4] ? -1.f : 1.f));
        }
    }
    uint4 * yo = (uint4 *) (y + (size_t) gt * 32);
#pragma unroll
    for (int c = 0; c < 4; ++c) yo[c] = ((const uint4 *) out)[c];
}

// Q6_K (q6r2 / q6r3): the fork's q6r2_to_f16, one thread per 32-value group (16 ql bytes, 8 qh bytes as 32-bit loads);
// q6r3 holds the group's hi2 bits at q6r3_pos (matvec_pack_body.inc). dequantize_q6_K's arithmetic (d*sc, times q - 32).
template <bool R3>
__global__ void __launch_bounds__(DIRECT_THREADS) q6r_direct_k(const char * __restrict__ w, half * __restrict__ y,
                                                              const int nb, const int row_bytes, const int64_t ngroups) {
    const int64_t gt = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (gt >= ngroups) return;
    const int row = (int) (gt / (8 * nb));
    const int G = (int) (gt - (int64_t) row * 8 * nb);
    const int sb = G >> 3, g = G & 7, n = g >> 2, t = g & 3;
    const uint8_t * xr = (const uint8_t *) (w + (size_t) row * row_bytes);
    const uint32_t * ql4 = (const uint32_t *) (xr + 16 * G);
    const uint32_t * qh4 = (const uint32_t *) (xr + 128 * nb + 8 * G);
    const uint32_t QL[4] = {ql4[0], ql4[1], ql4[2], ql4[3]};
    const uint32_t QH[2] = {qh4[0], qh4[1]};
    const int8_t * sc = (const int8_t *) (xr + 192 * nb + 16 * sb + 8 * n + 2 * t);
    const float d = __half2float(*((const half *) (xr + 208 * nb + 2 * sb)));
    const float d0 = d * sc[0], d1 = d * sc[1];
    __align__(16) half out[32];
#pragma unroll
    for (int l = 0; l < 32; ++l) {
        const int b = l & 15;
        const int lo = (QL[b >> 2] >> (8 * (b & 3) + 4 * (l >> 4))) & 0xF;
        const int hi = R3 ? (QH[l >> 4] >> q6r3_pos(l & 15)) & 3
                          : (QH[l >> 4] >> (2 * (l & 15))) & 3;
        const int v = (lo | (hi << 4)) - 32;
        out[l] = __float2half_rn((l < 16 ? d0 : d1) * v);
    }
    half * yr = y + (size_t) row * nb * 256 + 128 * n + 32 * t + 256 * sb;
#pragma unroll
    for (int c = 0; c < 4; ++c) ((uint4 *) yr)[c] = ((const uint4 *) out)[c];
}

inline unsigned blocks_for(int64_t n, int per_block) { return (unsigned) ((n + per_block - 1) / per_block); }

// the direct kernel for the layout over rows [row0, row0 + nrows); false when the layout has none (or the alignment
// its loads need is missing): the caller falls back to the shared-memory kernel
bool launch_direct(const packed_weight & w, int64_t row0, int64_t nrows, half * dst, int64_t rb, cudaStream_t s) {
    const char * base = (const char *) w.data + row0 * rb;
    if (((uintptr_t) base) % 4 || rb % 4 || ((uintptr_t) dst) % 16 || w.n_in % 32) return false;
    const int rbi = (int) rb;
    switch (w.layout) {
        case LAYOUT_Q4R: {
            const int nb = (int) (w.n_in / 32);
            const int64_t n = nrows * nb * 4;
            q4r_direct_k<<<blocks_for(n, DIRECT_THREADS), DIRECT_THREADS, 0, s>>>(base, dst, nb, rbi, n);
        } return true;
        case LAYOUT_Q8R: {
            const int nb = (int) (w.n_in / 32);
            const int64_t n = nrows * nb * 4;
            if (rb % 8 || ((uintptr_t) base) % 8) return false;   // 8-byte qs loads
            q8r_direct_k<<<blocks_for(n, DIRECT_THREADS), DIRECT_THREADS, 0, s>>>(base, dst, nb, rbi, n);
        } return true;
        case LAYOUT_Q4KR:
        case LAYOUT_Q5KR: {
            const int nb = (int) (w.n_in / 256);
            const int64_t nsb = nrows * nb;
            if (nsb > INT32_MAX) return false;
            if (w.layout == LAYOUT_Q4KR)
                kq_direct_k<false><<<blocks_for(nsb, 8), DIRECT_THREADS, 0, s>>>(base, dst, nb, rbi, (int) nsb);
            else
                kq_direct_k<true><<<blocks_for(nsb, 8), DIRECT_THREADS, 0, s>>>(base, dst, nb, rbi, (int) nsb);
        } return true;
        case LAYOUT_Q3R: {
            const int nb = (int) (w.n_in / 256);
            const int64_t ng = nrows * nb * 8;
            q3r_direct_k<<<blocks_for(ng, DIRECT_THREADS), DIRECT_THREADS, 0, s>>>(base, dst, nb, rbi, ng);
        } return true;
        case LAYOUT_Q2KR: {
            const int nb = (int) (w.n_in / 256);
            const int64_t ng = nrows * nb * 8;
            if (rb % 8 || ((uintptr_t) base) % 8) return false;   // 8-byte group loads
            q2kr_direct_k<<<blocks_for(ng, DIRECT_THREADS), DIRECT_THREADS, 0, s>>>(base, dst, nb, rbi, ng);
        } return true;
        case LAYOUT_IQ3XXSR:
        case LAYOUT_IQ3SR: {
            const int nb = (int) (w.n_in / 256);
            const int64_t ng = nrows * nb * 8;
            if (w.layout == LAYOUT_IQ3XXSR)
                iq3r_direct_k<false><<<blocks_for(ng, DIRECT_THREADS), DIRECT_THREADS, 0, s>>>(base, dst, nb, rbi, ng);
            else
                iq3r_direct_k<true><<<blocks_for(ng, DIRECT_THREADS), DIRECT_THREADS, 0, s>>>(base, dst, nb, rbi, ng);
        } return true;
        case LAYOUT_Q6R2:
        case LAYOUT_Q6R3: {
            const int nb = (int) (w.n_in / 256);
            const int64_t ng = nrows * nb * 8;
            if (nrows > INT32_MAX) return false;
            if (w.layout == LAYOUT_Q6R2)
                q6r_direct_k<false><<<blocks_for(ng, DIRECT_THREADS), DIRECT_THREADS, 0, s>>>(base, dst, nb, rbi, ng);
            else
                q6r_direct_k<true><<<blocks_for(ng, DIRECT_THREADS), DIRECT_THREADS, 0, s>>>(base, dst, nb, rbi, ng);
        } return true;
        default: return false;
    }
}

} // namespace

int unpack_impl_default() {
    static const int v = getenv("P100E_UNPACK") && *getenv("P100E_UNPACK") ? atoi(getenv("P100E_UNPACK")) : UNPACK_DIRECT;
    return v == UNPACK_SMEM ? UNPACK_SMEM : UNPACK_DIRECT;
}

void packed_to_f16(const packed_weight & w, int64_t row0, int64_t nrows, half * dst, cudaStream_t s) {
    packed_to_f16(w, row0, nrows, dst, s, unpack_impl_default());
}

void packed_to_f16(const packed_weight & w, int64_t row0, int64_t nrows, half * dst, cudaStream_t s, int impl) {
    if (nrows <= 0) return;
    if (row0 < 0 || row0 + nrows > w.n_out) throw std::runtime_error("fast::packed_to_f16: row range out of bounds");
    const qtype_info & ti = get_type_info(w.type);
    if (ti.blck <= 0 || w.n_in % ti.blck != 0) throw std::runtime_error("fast::packed_to_f16: bad type/shape " + std::to_string(w.type));
    const int64_t rb0 = w.n_in / ti.blck * (int64_t) ti.size;
    const int64_t rb = (int64_t) row_stride(w);   // the row stride (> rb0: padded Q6_K rows, pack_row_stride)
    if (rb != rb0 && w.layout != LAYOUT_Q6R2 && w.layout != LAYOUT_Q6R3)
        throw std::runtime_error("fast::packed_to_f16: padded row stride on layout " + std::to_string(w.layout));
    if (w.layout == LAYOUT_IQ2XXSR || w.layout == LAYOUT_IQ2XSR || w.layout == LAYOUT_IQ2SR) {   // one kernel for both impls
        if (!iq2r_to_f16(w.layout, (const char *) w.data + row0 * rb, nrows, w.n_in, rb, dst, s))
            throw std::runtime_error("fast::packed_to_f16: iq2r alignment");
        const cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) throw std::runtime_error(std::string("fast::packed_to_f16 (iq2r): ") + cudaGetErrorString(e));
        return;
    }
    if (impl == UNPACK_DIRECT && w.layout != LAYOUT_RAW && w.layout != LAYOUT_KQRAW && w.layout != LAYOUT_IQ1S && w.layout != LAYOUT_IQ1M && w.layout != LAYOUT_Q10 && w.layout != LAYOUT_Q20 && launch_direct(w, row0, nrows, dst, rb, s)) {
        const cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) throw std::runtime_error(std::string("fast::packed_to_f16 (direct): ") + cudaGetErrorString(e));
        return;
    }
    switch (w.layout) {
        case LAYOUT_RAW:
        case LAYOUT_KQRAW:
        case LAYOUT_Q10:
        case LAYOUT_Q20:
        case LAYOUT_IQ1S:   // untouched GGUF rows: ggml's converter
        case LAYOUT_IQ1M: {
            to_fp16_cuda_t to16 = ggml_get_to_fp16_cuda((ggml_type) w.type);
            if (!to16) throw std::runtime_error("fast::packed_to_f16: no f16 dequant for type " + std::to_string(w.type));
            to16((const char *) w.data + row0 * rb, dst, nrows * w.n_in, s);
        } break;
        case LAYOUT_Q4R:  launch<LAYOUT_Q4R >(w, row0, nrows, dst, (int) rb, s); break;
        case LAYOUT_Q8R:  launch<LAYOUT_Q8R >(w, row0, nrows, dst, (int) rb, s); break;
        case LAYOUT_Q4KR: launch<LAYOUT_Q4KR>(w, row0, nrows, dst, (int) rb, s); break;
        case LAYOUT_Q3R:  launch<LAYOUT_Q3R >(w, row0, nrows, dst, (int) rb, s); break;
        case LAYOUT_Q2KR: launch<LAYOUT_Q2KR>(w, row0, nrows, dst, (int) rb, s); break;
        case LAYOUT_IQ3XXSR: launch<LAYOUT_IQ3XXSR>(w, row0, nrows, dst, (int) rb, s); break;
        case LAYOUT_IQ3SR: launch<LAYOUT_IQ3SR>(w, row0, nrows, dst, (int) rb, s); break;
        case LAYOUT_Q6R2: launch<LAYOUT_Q6R2>(w, row0, nrows, dst, (int) rb, s); break;
        case LAYOUT_Q6R3: launch<LAYOUT_Q6R3>(w, row0, nrows, dst, (int) rb, s); break;
        case LAYOUT_Q5KR: launch<LAYOUT_Q5KR>(w, row0, nrows, dst, (int) rb, s); break;
        default: throw std::runtime_error("fast::packed_to_f16: unknown layout " + std::to_string(w.layout));
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("fast::packed_to_f16: ") + cudaGetErrorString(e));
}

} // namespace p100e::fast
