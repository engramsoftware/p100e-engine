// Decode attention v2 (T = 1..4 query tokens, D = 256, GQA 6, f16 / q8_0 / q4_0 cache), see attn_dec2.cuh.
// Block structure after the llama.cpp fork's GQA vector kernel (ggml-cuda/fattn-vec-gqa.cuh): one block per kv
// head serves its 6 q heads, every K / V row is loaded once for the 6 columns, the 4 warps take 32-key groups with no
// block barrier in the key loop, split-K over a FIXED number of interleaved key-tile streams plus a combine.
// Differences: fp32 math with explicit roundings, our cache layouts, the causal limit from the positions (no mask),
// and one block per query token: the block of token t in a T-token call runs exactly the T = 1 computation at
// position p0 + t (same streams, same order), so MTP verify rows are bitwise equal to plain decode at any depth.
//  QK: a warp stages 8 K rows at a time in shared memory (row stride = 4 mod 32 words, quarter q at an offset = q mod 4
//      words: conflict-free) and lane = (row r = lane & 7, quarter q = lane >> 3) dots its 64 dims with the 6 queries
//      (Q in smem, broadcast reads); two shuffles add the quarters. Lane = key afterwards: no per-key reductions.
//  PV: lane = 8 consecutive dims of the V row (read from global), 6 accumulators each.
//  Dequant with magic numbers (int8 / nibble into a float mantissa, one FADD: no I2F), the block scale folded out of the
//  inner loops (QK: d * sum over the block; PV: (p * d) * q).
#include "fast/attn_dec2.cuh"
#include "fast/kv_page.cuh"

#include <cuda_fp16.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>

#define NEG_INF __int_as_float(0xff800000)

namespace p100e::fast {
namespace {

constexpr int DH = 256, GQA = 6, NT = 128, NW = NT / 32, KB = NT;   // KB keys per block iteration (32 per warp)
constexpr int MAX_T = ATTN_MAX_ROWS, SR = 8;                         // rows per call; SR K rows staged per warp step
constexpr int QSTR = 4 * 68;                                         // floats per Q row in smem: quarter q at q * 68

__device__ __forceinline__ float h2f(uint32_t u16) { return __half2float(__ushort_as_half((unsigned short) u16)); }

// byte i of u (unsigned) -> float exactly: 0x4B0000xx = 2^23 + xx
template <int i> __device__ __forceinline__ float mag8(uint32_t u) {
    return __int_as_float(__byte_perm(u, 0x4B000000u, 0x7540 + i));
}
constexpr float M8 = 8388736.0f;   // 2^23 + 128 (q8: u = q ^ 0x80 = q + 128)
constexpr float M4 = 8388616.0f;   // 2^23 + 8   (q4: nibble - 8)
__device__ __forceinline__ float mag4(uint32_t w, int sh) {
    return __fsub_rn(__int_as_float(((w >> sh) & 15u) | 0x4B000000u), M4);
}

// staged row: words per row (= 4 mod 32) and the word offset of quarter q (= q mod 4)
template <uint32_t KT> struct stg {
    static constexpr int S = KT == KV_F16 ? 132 : KT == KV_Q8_0 ? 68 : 36;
    static constexpr int QB = KT == KV_F16 ? 33 : KT == KV_Q8_0 ? 17 : 9;
    static constexpr int HB = KT == KV_F16 ? DH * 2 : KT == KV_Q8_0 ? (DH / 32) * 34 : (DH / 32) * 18;   // global head row
};

// copy 8 head rows (rows 0..7 from base, those < nk) of one kv head into the warp's staging buffer (base = the first
// row's head; consecutive rows rowb apart: the 32 keys of a warp group never straddle a page)
template <uint32_t KT, bool LEAN>
__device__ __forceinline__ void stage_rows(const char * __restrict__ base, uint32_t rowb, int nk, uint32_t * st, int lane) {
    constexpr int S = stg<KT>::S, QB = stg<KT>::QB;
    // lane = (row r = lane & 7, chunk phase qd = lane >> 3): the lane's row address and smem destination are computed once
    // and the chunks qd + 4 j are immediate offsets (the old flat (row, chunk) index needed a division by the chunk count
    // and 64-bit address math per chunk: ~500 instructions per 8-row step for q8_0, now ~45)
    if constexpr (!LEAN && KT != KV_F16) {   // P100E_DEC2_LEAN=0: the original flat (row, chunk) loops (A/B reference)
        if constexpr (KT == KV_Q8_0) {   // 17 uint4 per row: 16 of quants (chunk c -> quarter c / 4) + the scales
            for (int i = lane; i < SR * 17; i += 32) {
                const int r = i / 17, c = i - r * 17;
                if (r >= nk) continue;
                const uint4 * src = reinterpret_cast<const uint4 *>(base + (uint32_t) r * rowb);
                uint32_t * row = st + r * S;
                if (c == 16) {   // 8 scales = words 0..3: scales 2q, 2q+1 -> word QB * q
                    const uint4 v = src[0];
                    row[0] = v.x; row[QB] = v.y; row[2 * QB] = v.z; row[3 * QB] = v.w;
                } else {
                    const uint4 v = src[1 + c];
                    uint32_t * d = row + (c >> 2) * QB + 1 + (c & 3) * 4;
                    d[0] = v.x; d[1] = v.y; d[2] = v.z; d[3] = v.w;
                }
            }
        } else {   // q4_0: 9 uint4 per row, copied as is
            for (int i = lane; i < SR * 9; i += 32) {
                const int r = i / 9, c = i - r * 9;
                if (r >= nk) continue;
                const uint4 v = reinterpret_cast<const uint4 *>(base + (uint32_t) r * rowb)[c];
                *reinterpret_cast<uint4 *>(st + r * S + c * 4) = v;
            }
        }
        return;
    }
    const int r = lane & 7, qd = lane >> 3;
    if constexpr (KT != KV_F16) { if (r >= nk) return; }
    const uint4 * src = reinterpret_cast<const uint4 *>(base + (uint32_t) r * rowb);
    uint32_t * row = st + r * S;
    if constexpr (KT == KV_F16) {   // f16 keeps the flat loop (8 chunks per lane in flight cost registers: 176 already)
        for (int i = lane; i < SR * 32; i += 32) {
            const int rr = i >> 5, c = i & 31;
            if (rr >= nk) continue;
            const uint4 v = reinterpret_cast<const uint4 *>(base + (uint32_t) rr * rowb)[c];
            uint32_t * d = st + rr * S + (c >> 3) * QB + (c & 7) * 4;
            d[0] = v.x; d[1] = v.y; d[2] = v.z; d[3] = v.w;
        }
    } else if constexpr (KT == KV_Q8_0) {   // 17 uint4 per row: 16 of quants (chunk c -> quarter c / 4) + the scales
        uint4 v[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] = src[1 + qd + 4 * j];
        uint4 sv = make_uint4(0, 0, 0, 0);
        if (qd == 0) sv = src[0];   // 8 scales = words 0..3: scales 2q, 2q+1 -> word QB * q
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            uint32_t * d = row + j * QB + 1 + qd * 4;
            d[0] = v[j].x; d[1] = v[j].y; d[2] = v[j].z; d[3] = v[j].w;
        }
        if (qd == 0) { row[0] = sv.x; row[QB] = sv.y; row[2 * QB] = sv.z; row[3 * QB] = sv.w; }
    } else {   // q4_0: 9 uint4 per row, copied as is (quarter q = blocks 2q, 2q+1 = words 9q .. 9q+8)
        uint4 v[3];
#pragma unroll
        for (int j = 0; j < 2; ++j) v[j] = src[qd + 4 * j];
        if (qd == 0) v[2] = src[8];
#pragma unroll
        for (int j = 0; j < 2; ++j) *reinterpret_cast<uint4 *>(row + (qd + 4 * j) * 4) = v[j];
        if (qd == 0) *reinterpret_cast<uint4 *>(row + 32) = v[2];
    }
}

// lane (row r, quarter qd): partial dots of its 64 dims with the 6 queries (Q in smem, quarter qd at qd * 68)
template <uint32_t KT>
__device__ __forceinline__ void dot_quarter(const uint32_t * __restrict__ row, const float * __restrict__ Qs, int qd,
                                            float (&c)[GQA]) {
    constexpr int QB = stg<KT>::QB;
    const float4 * Q4 = reinterpret_cast<const float4 *>(Qs) + qd * 17;   // + j * QSTR / 4
    if constexpr (KT == KV_F16) {
        const uint32_t * w = row + qd * QB;
#pragma unroll
        for (int j = 0; j < GQA; ++j) c[j] = 0.0f;
#pragma unroll 4
        for (int k = 0; k < 16; ++k) {   // 2 words = 4 dims per step
            const float2 a = __half22float2(*reinterpret_cast<const __half2 *>(&w[2 * k]));
            const float2 b = __half22float2(*reinterpret_cast<const __half2 *>(&w[2 * k + 1]));
#pragma unroll
            for (int j = 0; j < GQA; ++j) {
                const float4 q = Q4[j * (QSTR / 4) + k];
                float t = __fmaf_rn(q.x, a.x, c[j]);
                t = __fmaf_rn(q.y, a.y, t);
                t = __fmaf_rn(q.z, b.x, t);
                c[j] = __fmaf_rn(q.w, b.y, t);
            }
        }
    } else if constexpr (KT == KV_Q8_0) {
        const uint32_t sw = row[qd * QB];
        const uint32_t * w = row + qd * QB + 1;
#pragma unroll
        for (int bl = 0; bl < 2; ++bl) {   // two 32-value blocks
            float t[GQA];
#pragma unroll
            for (int j = 0; j < GQA; ++j) t[j] = 0.0f;
#pragma unroll 4
            for (int k = 0; k < 8; ++k) {
                const uint32_t u = w[bl * 8 + k] ^ 0x80808080u;
                const float v0 = __fsub_rn(mag8<0>(u), M8), v1 = __fsub_rn(mag8<1>(u), M8);
                const float v2 = __fsub_rn(mag8<2>(u), M8), v3 = __fsub_rn(mag8<3>(u), M8);
#pragma unroll
                for (int j = 0; j < GQA; ++j) {
                    const float4 q = Q4[j * (QSTR / 4) + bl * 8 + k];
                    float s = __fmaf_rn(q.x, v0, t[j]);
                    s = __fmaf_rn(q.y, v1, s);
                    s = __fmaf_rn(q.z, v2, s);
                    t[j] = __fmaf_rn(q.w, v3, s);
                }
            }
            const float d = h2f(bl ? sw >> 16 : sw & 0xffffu);
#pragma unroll
            for (int j = 0; j < GQA; ++j) c[j] = bl ? __fmaf_rn(d, t[j], c[j]) : __fmul_rn(d, t[j]);
        }
    } else {   // q4_0 quarter words W0..W8: d0 = W0 lo, qs0 = bytes 2..17, d1 = W4 hi, qs1 = W5..W8
        const uint32_t * W = row + qd * QB;
#pragma unroll
        for (int bl = 0; bl < 2; ++bl) {
            float t[GQA];
#pragma unroll
            for (int j = 0; j < GQA; ++j) t[j] = 0.0f;
#pragma unroll
            for (int k = 0; k < 4; ++k) {   // qs word k: low nibbles = dims 4k..4k+3, high = 16+4k..16+4k+3
                const uint32_t x = bl ? W[5 + k] : __funnelshift_r(W[k], W[k + 1], 16);
                float lo[4], hi[4];
#pragma unroll
                for (int b = 0; b < 4; ++b) { lo[b] = mag4(x, 8 * b); hi[b] = mag4(x, 8 * b + 4); }
#pragma unroll
                for (int j = 0; j < GQA; ++j) {
                    const float4 ql = Q4[j * (QSTR / 4) + bl * 8 + k];
                    const float4 qh = Q4[j * (QSTR / 4) + bl * 8 + 4 + k];
                    float s = __fmaf_rn(ql.x, lo[0], t[j]);
                    s = __fmaf_rn(ql.y, lo[1], s);
                    s = __fmaf_rn(ql.z, lo[2], s);
                    s = __fmaf_rn(ql.w, lo[3], s);
                    s = __fmaf_rn(qh.x, hi[0], s);
                    s = __fmaf_rn(qh.y, hi[1], s);
                    s = __fmaf_rn(qh.z, hi[2], s);
                    t[j] = __fmaf_rn(qh.w, hi[3], s);
                }
            }
            const float d = h2f(bl ? W[4] >> 16 : W[0] & 0xffffu);
#pragma unroll
            for (int j = 0; j < GQA; ++j) c[j] = bl ? __fmaf_rn(d, t[j], c[j]) : __fmul_rn(d, t[j]);
        }
    }
}

// grid (T, Hkv, nsplit), 128 threads. Block (t, hk, sp): token t (position p0 + t, keys 0..p0+t; posr: position
// posr[t] of row t's own sequence), the 6 q heads of kv head hk, key tiles sp, sp + nsplit, ... of KB keys. Writes the
// block's (max, sum, unnormalised output) per head. pg: paged cache addressing (fast/kv_page.cuh), per row.
template <uint32_t KT, bool LEAN>
// Registers (sm_60): q8_0 / q4_0 168 = 3 blocks per SM, so the 3 x SMs / Hkv split streams of one call fit in one wave;
// f16 176 (2 blocks, as before paging). Keep it so: holding the page-table pointer and per-head bases live across the
// key loop pushed q4_0 to 175 registers (2 blocks per SM: tg64 -14 % at 128K); the group address is now rebuilt from the
// kernel parameters and a 32-bit table offset.
__global__ void __launch_bounds__(NT) dec2_k(const float * __restrict__ q, const char * __restrict__ kc,
                                             const char * __restrict__ vc, float * __restrict__ pacc,
                                             float * __restrict__ pml, int p0, const int32_t * __restrict__ p0p,
                                             int Hq, int Hkv, float scale, int nsplit, const kv_pg pg,
                                             const int32_t * __restrict__ posr) {
    constexpr int S = stg<KT>::S, HB = stg<KT>::HB;
    constexpr int STG_WORDS = NW * SR * S;
    constexpr int MERGE_WORDS = 2 * GQA * DH;   // two warps' outputs at a time
    constexpr int BUF_WORDS = STG_WORDS > MERGE_WORDS ? STG_WORDS : MERGE_WORDS;
    __shared__ __align__(16) float Qs[GQA * QSTR];
    __shared__ __align__(16) uint32_t buf[BUF_WORDS];
    __shared__ float sm_m[NW][GQA], sm_l[NW][GQA];

    if (p0p) p0 = *p0p;
    const int t = blockIdx.x, hk = blockIdx.y, sp = blockIdx.z;
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int n = (posr ? posr[t] : p0 + t) + 1;   // keys visible to token t
    // 32-bit address arithmetic below the cache bases (cheaper on sm_60 than 64-bit): a contiguous cache and each paged
    // pool region (one layer's K or V pages) stay below 4 GiB (attn_decode2 / tp_model::alloc_pool check it)
    const uint32_t rowb = (uint32_t) (Hkv * HB);
    // row t's page table as a 32-bit offset into pg.pt (a kernel parameter): fewer live registers than a pointer
    const int toff = pg.pt && pg.rtab ? pg.rtab[t] : 0;
    if (sp * KB >= n) {   // no keys for this stream (short contexts): only the "empty" marker, the combine skips it
        if (threadIdx.x < GQA) {
            const size_t o = ((size_t) t * Hq + hk * GQA + threadIdx.x) * nsplit + sp;
            pml[o * 2] = NEG_INF;
            pml[o * 2 + 1] = 0.0f;
        }
        return;
    }

    for (int i = threadIdx.x; i < GQA * DH; i += NT) {   // Qs[j][quarter * 68 + (d % 64)] = q * scale
        const int j = i >> 8, d = i & 255;
        Qs[j * QSTR + (d >> 6) * 68 + (d & 63)] = __fmul_rn(q[((size_t) t * Hq + hk * GQA + j) * DH + d], scale);
    }
    __syncthreads();

    float m[GQA], l[GQA], acc[GQA][8];
#pragma unroll
    for (int j = 0; j < GQA; ++j) {
        m[j] = NEG_INF;
        l[j] = 0.0f;
#pragma unroll
        for (int d = 0; d < 8; ++d) acc[j][d] = 0.0f;
    }
    uint32_t * st = buf + w * SR * S;
    const int r = lane & 7, qd = lane >> 3;

    // the page of the warp's next 32-key group, loaded one tile ahead (the table load is off the critical path)
    int pnext = 0;
    if (pg.pt && sp * KB + w * 32 < n) pnext = pg.pt[toff + ((sp * KB + w * 32) >> pg.psh)];
    for (int tile = sp; tile * KB < n; tile += nsplit) {
        const int k0 = tile * KB + w * 32;
        if (k0 >= n) continue;   // warp-uniform; no block barrier in the loop
        const int nk = min(32, n - k0);
        // the warp's 32 keys lie in one page (pages are multiples of 32 positions): one table lookup per group
        const int pcur = pnext;
        if (pg.pt && k0 + nsplit * KB < n) pnext = pg.pt[toff + ((k0 + nsplit * KB) >> pg.psh)];
        const uint32_t goff = (uint32_t) (hk * HB) + (pg.pt ? (uint32_t) pcur * (uint32_t) pg.pstride +
                                                                (uint32_t) (k0 & ((1 << pg.psh) - 1)) * rowb
                                                              : (uint32_t) k0 * rowb);
        const char * gk = kc + goff, * gv = vc + goff;
        // ---- QK: 4 steps of 8 staged rows; afterwards lane L holds the scores of key k0 + L ----
        float sc[GQA];
#pragma unroll
        for (int j = 0; j < GQA; ++j) sc[j] = NEG_INF;
        for (int s8 = 0; s8 * SR < nk; ++s8) {
            stage_rows<KT, LEAN>(gk + (uint32_t) (s8 * SR) * rowb, rowb, nk - s8 * SR, st, lane);
            __syncwarp();
            float c[GQA];
            dot_quarter<KT>(st + r * S, Qs, qd, c);
#pragma unroll
            for (int j = 0; j < GQA; ++j) {
                c[j] = __fadd_rn(c[j], __shfl_xor_sync(0xffffffff, c[j], 8));
                c[j] = __fadd_rn(c[j], __shfl_xor_sync(0xffffffff, c[j], 16));
                if ((lane >> 3) == s8 && lane < nk) sc[j] = c[j];
            }
            __syncwarp();   // staging buffer reused by the next step
        }
        // ---- online softmax over the warp's keys ----
        float p[GQA];
#pragma unroll
        for (int j = 0; j < GQA; ++j) {
            float mx = sc[j];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));   // finite: nk >= 1
            const float mnew = fmaxf(m[j], mx);
            const float f = m[j] == NEG_INF ? 0.0f : expf(m[j] - mnew);
            p[j] = lane < nk ? expf(sc[j] - mnew) : 0.0f;
            float ps = p[j];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) ps = __fadd_rn(ps, __shfl_xor_sync(0xffffffff, ps, o));
            l[j] = __fmaf_rn(l[j], f, ps);
#pragma unroll
            for (int d = 0; d < 8; ++d) acc[j][d] = __fmul_rn(acc[j][d], f);
            m[j] = mnew;
        }
        // ---- PV: one V row per step, lane = 8 dims ----
        for (int kk0 = 0; kk0 < nk; kk0 += 4) {
            float v[4][8], dv[4];
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                if (kk0 + u >= nk) break;
                const char * head = gv + (uint32_t) (kk0 + u) * rowb;
                if constexpr (KT == KV_F16) {
                    const uint4 x = *reinterpret_cast<const uint4 *>(head + lane * 16);
                    const uint32_t xs[4] = {x.x, x.y, x.z, x.w};
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        const float2 f2 = __half22float2(*reinterpret_cast<const __half2 *>(&xs[e]));
                        v[u][2 * e] = f2.x;
                        v[u][2 * e + 1] = f2.y;
                    }
                    dv[u] = 1.0f;
                } else if constexpr (KT == KV_Q8_0) {
                    const uint2 x = *reinterpret_cast<const uint2 *>(head + DH / 32 * 2 + lane * 8);
                    const uint32_t a = x.x ^ 0x80808080u, b = x.y ^ 0x80808080u;
                    v[u][0] = __fsub_rn(mag8<0>(a), M8); v[u][1] = __fsub_rn(mag8<1>(a), M8);
                    v[u][2] = __fsub_rn(mag8<2>(a), M8); v[u][3] = __fsub_rn(mag8<3>(a), M8);
                    v[u][4] = __fsub_rn(mag8<0>(b), M8); v[u][5] = __fsub_rn(mag8<1>(b), M8);
                    v[u][6] = __fsub_rn(mag8<2>(b), M8); v[u][7] = __fsub_rn(mag8<3>(b), M8);
                    dv[u] = h2f(reinterpret_cast<const uint16_t *>(head)[lane >> 2]);
                } else {
                    const char * blk = head + (lane >> 2) * 18;
                    const uint16_t * b2 = reinterpret_cast<const uint16_t *>(blk + 2 + (lane & 1) * 8);
                    const uint32_t a = b2[0] | ((uint32_t) b2[1] << 16), b = b2[2] | ((uint32_t) b2[3] << 16);
                    const int sh = (lane & 2) ? 4 : 0;
#pragma unroll
                    for (int e = 0; e < 4; ++e) { v[u][e] = mag4(a, 8 * e + sh); v[u][4 + e] = mag4(b, 8 * e + sh); }
                    dv[u] = h2f(*reinterpret_cast<const uint16_t *>(blk));
                }
            }
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                if (kk0 + u >= nk) break;
#pragma unroll
                for (int j = 0; j < GQA; ++j) {
                    const float pj = __shfl_sync(0xffffffff, p[j], kk0 + u);
                    const float pd = KT == KV_F16 ? pj : __fmul_rn(pj, dv[u]);
#pragma unroll
                    for (int d = 0; d < 8; ++d) acc[j][d] = __fmaf_rn(pd, v[u][d], acc[j][d]);
                }
            }
        }
    }

    // ---- merge the 4 warps in order: ((w0 + w1) + (w2 + w3)) through shared memory ----
    if (lane == 0) {
#pragma unroll
        for (int j = 0; j < GQA; ++j) { sm_m[w][j] = m[j]; sm_l[w][j] = l[j]; }
    }
    __syncthreads();   // staging done everywhere; buf reused
    float * ob = reinterpret_cast<float *>(buf);   // [2][GQA][DH]
    float M[GQA];
#pragma unroll
    for (int j = 0; j < GQA; ++j) {
        M[j] = fmaxf(fmaxf(sm_m[0][j], sm_m[1][j]), fmaxf(sm_m[2][j], sm_m[3][j]));
        const float f = m[j] == NEG_INF ? 0.0f : expf(m[j] - M[j]);
#pragma unroll
        for (int d = 0; d < 8; ++d) acc[j][d] = __fmul_rn(acc[j][d], f);
    }
    if (w & 1) {   // warps 1, 3 hand their scaled outputs to warps 0, 2
#pragma unroll
        for (int j = 0; j < GQA; ++j) {
            float4 * dst = reinterpret_cast<float4 *>(&ob[((w >> 1) * GQA + j) * DH + lane * 8]);
            dst[0] = make_float4(acc[j][0], acc[j][1], acc[j][2], acc[j][3]);
            dst[1] = make_float4(acc[j][4], acc[j][5], acc[j][6], acc[j][7]);
        }
    }
    __syncthreads();
    if (w == 0 || w == 2) {
#pragma unroll
        for (int j = 0; j < GQA; ++j) {
            const float4 * src = reinterpret_cast<const float4 *>(&ob[((w >> 1) * GQA + j) * DH + lane * 8]);
            const float4 a = src[0], b = src[1];
            acc[j][0] = __fadd_rn(acc[j][0], a.x); acc[j][1] = __fadd_rn(acc[j][1], a.y);
            acc[j][2] = __fadd_rn(acc[j][2], a.z); acc[j][3] = __fadd_rn(acc[j][3], a.w);
            acc[j][4] = __fadd_rn(acc[j][4], b.x); acc[j][5] = __fadd_rn(acc[j][5], b.y);
            acc[j][6] = __fadd_rn(acc[j][6], b.z); acc[j][7] = __fadd_rn(acc[j][7], b.w);
        }
    }
    __syncthreads();
    if (w == 2) {
#pragma unroll
        for (int j = 0; j < GQA; ++j) {
            float4 * dst = reinterpret_cast<float4 *>(&ob[j * DH + lane * 8]);
            dst[0] = make_float4(acc[j][0], acc[j][1], acc[j][2], acc[j][3]);
            dst[1] = make_float4(acc[j][4], acc[j][5], acc[j][6], acc[j][7]);
        }
    }
    __syncthreads();
    if (w == 0) {
#pragma unroll
        for (int j = 0; j < GQA; ++j) {
            const float4 * src = reinterpret_cast<const float4 *>(&ob[j * DH + lane * 8]);
            const float4 a = src[0], b = src[1];
            const size_t o = ((size_t) t * Hq + hk * GQA + j) * nsplit + sp;
            float4 * dst = reinterpret_cast<float4 *>(pacc + o * DH + lane * 8);
            dst[0] = make_float4(__fadd_rn(acc[j][0], a.x), __fadd_rn(acc[j][1], a.y), __fadd_rn(acc[j][2], a.z),
                                 __fadd_rn(acc[j][3], a.w));
            dst[1] = make_float4(__fadd_rn(acc[j][4], b.x), __fadd_rn(acc[j][5], b.y), __fadd_rn(acc[j][6], b.z),
                                 __fadd_rn(acc[j][7], b.w));
            if (lane == 0) {
                float L = 0.0f;
                if (M[j] != NEG_INF) {
#pragma unroll
                    for (int ww = 0; ww < NW; ++ww) {
                        const float mw = sm_m[ww][j];
                        L = __fmaf_rn(sm_l[ww][j], mw == NEG_INF ? 0.0f : expf(mw - M[j]), L);
                    }
                }
                pml[o * 2] = M[j];
                pml[o * 2 + 1] = L;
            }
        }
    }
}

// out[t][h][d] = sum_s w_s A_s / sum_s w_s L_s, w_s = exp(M_s - max M), splits in order. grid T * Hq, 256 threads.
// Only the streams that own keys are read (the others hold the empty marker and would be skipped anyway).
__global__ void dec2_combine_k(const float * __restrict__ pacc, const float * __restrict__ pml, float * __restrict__ out,
                               int nsplit, int p0, const int32_t * __restrict__ p0p, int Hq,
                               const int32_t * __restrict__ posr) {
    if (p0p) p0 = *p0p;
    const int t = (int) blockIdx.x / Hq;
    const int n = (posr ? posr[t] : p0 + t) + 1;   // keys of this row's token
    const size_t o0 = (size_t) blockIdx.x * nsplit;   // partials are strided by the full stream count
    nsplit = min(nsplit, (n + KB - 1) / KB);
    const int d = threadIdx.x;
    float M = NEG_INF;
    for (int s = 0; s < nsplit; ++s) M = fmaxf(M, pml[(o0 + s) * 2]);
    float num = 0.0f, den = 0.0f;
    for (int s = 0; s < nsplit; ++s) {
        const float ms = pml[(o0 + s) * 2];
        if (ms == NEG_INF) continue;
        const float f = expf(ms - M);
        den = __fmaf_rn(f, pml[(o0 + s) * 2 + 1], den);
        num = __fmaf_rn(f, pacc[(o0 + s) * DH + d], num);
    }
    out[(size_t) blockIdx.x * DH + d] = __fdiv_rn(num, den);
}

struct Scr { float * acc = nullptr; float * ml = nullptr; size_t cap = 0; int sm = 0; };
int g_sm[16];
// split partials per (device, stream): calls on different streams may overlap (tensor-parallel ranks sharing one GPU in
// local tests, future concurrent streams); a stream keeps its pointers (captured graphs)
struct ScrEntry { int dev = -1; cudaStream_t s = nullptr; Scr sc; };
ScrEntry g_s2[64];
int g_ns2 = 0;
Scr * scr_for(int dev, cudaStream_t s) {
    for (int i = 0; i < g_ns2; ++i) if (g_s2[i].dev == dev && g_s2[i].s == s) return &g_s2[i].sc;
    if (g_ns2 == 64) return nullptr;
    g_s2[g_ns2].dev = dev; g_s2[g_ns2].s = s;
    return &g_s2[g_ns2++].sc;
}

} // namespace

static int g_split_override = 0;
// P100E_DEC2_LEAN (default 1): lean K-row staging for q8_0 / q4_0 (bit-identical to 0, the original staging loops).
// Read on first use and again by attn_dec2_reload_env() (the A/B bench; the choice is baked into captured graphs).
static int g_lean = -1;
bool attn_dec2_reload_env() {
    const char * e = getenv("P100E_DEC2_LEAN");
    const int v = e && *e ? (atoi(e) != 0) : 1;
    const bool ch = g_lean != v;
    g_lean = v;
    return ch;
}
void attn_dec2_set_nsplit(int n) { g_split_override = n; }

int attn_dec2_nsplit(int Hkv) {
    int dev = 0;
    cudaGetDevice(&dev);
    int & sm = g_sm[dev & 15];
    if (!sm) cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, dev);
    static const int env = getenv("P100E_DEC2_SPLIT") ? atoi(getenv("P100E_DEC2_SPLIT")) : 0;
    const int ns = g_split_override > 0 ? g_split_override : env > 0 ? env : 3 * sm / std::max(1, Hkv);
    return std::max(1, std::min(ns, 256));
}

bool attn_decode2(const float * q, const kv_cache & c, float * out, int T, int p0, const int32_t * p0_dev, int Hq,
                  float scale, cudaStream_t s, const int32_t * pos_rows) {
    if (c.D != DH || Hq != GQA * c.Hkv || T < 1 || T > MAX_T || !c.k) return false;
    if (c.pt && (c.psh < 5 || c.psh > 20)) return false;   // a warp's 32-key group must not straddle a page
    if (!c.pt && (size_t) c.n_ctx * kv_row_bytes(c.type, c.Hkv, c.D) >= ((size_t) 1 << 32)) return false;   // 32-bit offsets
    if (c.type != KV_F16 && c.type != KV_Q8_0 && c.type != KV_Q4_0) return false;
    const int nsplit = attn_dec2_nsplit(c.Hkv);
    int dev = 0;
    cudaGetDevice(&dev);
    Scr * scp = scr_for(dev, s);
    if (!scp) { fprintf(stderr, "p100e::fast::attn_decode2: too many streams\n"); return false; }
    Scr & sc = *scp;
    // allocated once at the largest T (pointers stay fixed for captured graphs; first call is an eager warm-up)
    const size_t need = (size_t) MAX_T * Hq * nsplit;
    if (need > sc.cap) {
        if (sc.acc) cudaFree(sc.acc);
        if (sc.ml) cudaFree(sc.ml);
        sc.acc = sc.ml = nullptr;
        sc.cap = 0;
        if (cudaMalloc(&sc.acc, need * DH * sizeof(float)) != cudaSuccess ||
            cudaMalloc(&sc.ml, need * 2 * sizeof(float)) != cudaSuccess) {
            cudaGetLastError();
            fprintf(stderr, "p100e::fast::attn_decode2: scratch allocation failed\n");
            return false;
        }
        sc.cap = need;
    }
    const dim3 grid(T, c.Hkv, nsplit);
    const char * kc = (const char *) c.k, * vc = (const char *) c.v;
    const kv_pg pg = kv_pg_of(c);
    if (g_lean < 0) attn_dec2_reload_env();
    const bool lean = g_lean != 0;
#define DEC2_LAUNCH(KT_) (lean ? dec2_k<KT_, true> : dec2_k<KT_, false>)<<<grid, NT, 0, s>>>(q, kc, vc, sc.acc, sc.ml, p0, p0_dev, Hq, c.Hkv, scale, nsplit, pg, pos_rows)
    if (c.type == KV_F16) DEC2_LAUNCH(KV_F16);
    else if (c.type == KV_Q8_0) DEC2_LAUNCH(KV_Q8_0);
    else DEC2_LAUNCH(KV_Q4_0);
#undef DEC2_LAUNCH
    dec2_combine_k<<<T * Hq, DH, 0, s>>>(sc.acc, sc.ml, out, nsplit, p0, p0_dev, Hq, pos_rows);
    return true;
}

} // namespace p100e::fast
