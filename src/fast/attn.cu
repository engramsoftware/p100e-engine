#include "fast/attn.cuh"
#include "fast/attn_dec2.cuh"
#include "fast/kv_q8.cuh"
#include "fast/kv_page.cuh"

#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define NEG_INF __int_as_float(0xff800000)
namespace p100e::fast {
namespace {

constexpr int DH = 256, GQA = 6, TILE = 32, NTHR = 256, MAX_SPLIT = 64;
constexpr int QSTRIDE = 288;   // floats per q row in smem: 8 groups of 32 dims, each padded to 36 (conflict-free float4)
constexpr int Q8_QOFF = DH / 32 * 2;   // q8_0 head row: byte offset of the int8 quants (after the 8 f16 scales)

// bytes of one (position, kv head) row of the cache
template <uint32_t KT> constexpr size_t head_bytes() {
    return KT == KV_F16 ? (size_t) DH * 2 : KT == KV_Q8_0 ? (size_t) (DH / 32) * 34 : (size_t) (DH / 32) * 18;
}

__device__ __forceinline__ float h2f(uint16_t u) { return __half2float(__ushort_as_half(u)); }

// split-K plan for n visible positions: shared by the host-argument path and the device-p0 path (same plan -> same
// summation order -> bit-identical results). cap = min(MAX_SPLIT, 3 * SMs / Hkv).
// T-invariance (MTP verify rows == plain decode) holds only while the chunk is the same for n = p0 + T and n = p0 + t + 1,
// i.e. n <= 64 * cap (4096 on P100x2); past that the chunk grows with n and a verify batch straddling a chunk change
// differs in the last bits. These v1 kernels run only with P100E_ATTN_DEC=1 (or when attn_decode2 refuses a shape):
// the default decode attention is fast/attn_dec2.cu, T-invariant at any depth for f16 / q8_0 / q4_0.
__host__ __device__ __forceinline__ void split_plan(int n, int cap, int & nsplit, int & chunk) {
    nsplit = (n + 63) / 64;
    if (nsplit > cap) nsplit = cap;
    if (nsplit < 1) nsplit = 1;
    chunk = (n + nsplit - 1) / nsplit;
    chunk = (chunk + TILE - 1) / TILE * TILE;
    nsplit = (n + chunk - 1) / chunk;
}

// One block = (split, kv head). 8-lane groups own one position each for the QK phase (lane = 32 dims);
// the V phase is thread-per-dim with NQ accumulators. Scores/probabilities pass through shared memory.
// Every multiply-add is an explicit __fmaf_rn / __fmul_rn / __fadd_rn (the same chains
// the ones nvcc built for most T = 1 elements): a row's arithmetic must not depend on how the compiler contracts a*b+c
// in each <T> instantiation (row t of a T-row call == the T = 1 call: MTP verify).
template <int T, uint32_t KT>
__global__ void __launch_bounds__(NTHR) attn_part_k(const float * __restrict__ q, const char * __restrict__ kc,
                                                    const char * __restrict__ vc, float * __restrict__ part_acc,
                                                    float * __restrict__ part_ml, int p0, int Hq, int Hkv, float scale,
                                                    int nsplit, int chunk, const int32_t * __restrict__ p0p, int cap) {
    constexpr int NQ = GQA * T;
    if (p0p) {   // device-p0 path: grid sized for the largest n; blocks past this step's split count exit
        p0 = *p0p;
        split_plan(p0 + T, cap, nsplit, chunk);
        if ((int) blockIdx.x >= nsplit) return;
    }
    constexpr int NQP = (NQ + 3) & ~3;
    __shared__ __align__(16) float qs[NQ * QSTRIDE];
    __shared__ float ss[NQ * 33];
    __shared__ __align__(16) float pT[TILE * NQP];
    __shared__ float mrow[NQ], lrow[NQ], fac[NQ];

    const int sp = blockIdx.x, hk = blockIdx.y, tid = threadIdx.x;
    const int n_end = p0 + T;
    const int s0 = sp * chunk, s1 = min(n_end, s0 + chunk);
    const size_t headb = head_bytes<KT>();
    const size_t rowb = (size_t) Hkv * headb;

    for (int i = tid; i < NQ * DH; i += NTHR) {
        const int qi = i >> 8, d = i & 255, t = qi / GQA, h = hk * GQA + qi % GQA;
        qs[qi * QSTRIDE + (d >> 5) * 36 + (d & 31)] = q[((size_t) t * Hq + h) * DH + d] * scale;
    }
    if (tid < NQ) { mrow[tid] = NEG_INF; lrow[tid] = 0.0f; }

    float acc[NQ];
#pragma unroll
    for (int i = 0; i < NQ; ++i) acc[i] = 0.0f;
    const int grp = tid >> 3, sub = tid & 7;

    for (int tile = s0; tile < s1; tile += TILE) {
        __syncthreads();   // q/ml init visible; previous tile's ss/pT consumed
        // ---- QK ----
        float part[NQ];
#pragma unroll
        for (int i = 0; i < NQ; ++i) part[i] = 0.0f;
        const int j = tile + grp;
        if (j < s1) {
            float kv[32];
            if constexpr (KT == KV_F16) {
                const uint4 * p = reinterpret_cast<const uint4 *>(kc + j * rowb + hk * headb + sub * 64);
                uint4 u[4];
#pragma unroll
                for (int i = 0; i < 4; ++i) u[i] = p[i];
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const __half2 * h2 = reinterpret_cast<const __half2 *>(&u[i]);
#pragma unroll
                    for (int m = 0; m < 4; ++m) {
                        const float2 f = __half22float2(h2[m]);
                        kv[i * 8 + m * 2] = f.x; kv[i * 8 + m * 2 + 1] = f.y;
                    }
                }
            } else if constexpr (KT == KV_Q8_0) {
                // lane sub = q8 block sub of the head (32 dims): scale + 32 int8 (two 16-byte loads)
                const char * hb = kc + j * rowb + hk * headb;
                const float d = h2f(reinterpret_cast<const uint16_t *>(hb)[sub]);
                const uint4 * p = reinterpret_cast<const uint4 *>(hb + Q8_QOFF + sub * 32);
                const uint4 u0 = p[0], u1 = p[1];
                const uint32_t w[8] = {u0.x, u0.y, u0.z, u0.w, u1.x, u1.y, u1.z, u1.w};
#pragma unroll
                for (int m = 0; m < 8; ++m)
#pragma unroll
                    for (int b = 0; b < 4; ++b) kv[m * 4 + b] = __fmul_rn(q8_lo(w[m], b), d);
            } else {
                const uint16_t * p = reinterpret_cast<const uint16_t *>(kc + j * rowb + hk * headb + sub * 18);
                const float d = h2f(p[0]);
#pragma unroll
                for (int m = 0; m < 8; ++m) {
                    const uint32_t w = p[1 + m];
                    kv[2 * m]          = (float) ((int) (w & 15) - 8) * d;
                    kv[2 * m + 1]      = (float) ((int) ((w >> 8) & 15) - 8) * d;
                    kv[16 + 2 * m]     = (float) ((int) ((w >> 4) & 15) - 8) * d;
                    kv[16 + 2 * m + 1] = (float) ((int) (w >> 12) - 8) * d;
                }
            }
#pragma unroll
            for (int c = 0; c < 8; ++c) {
#pragma unroll
                for (int qi = 0; qi < NQ; ++qi) {
                    const float4 qv = *reinterpret_cast<const float4 *>(&qs[qi * QSTRIDE + sub * 36 + c * 4]);
                    float d4 = __fmaf_rn(qv.x, kv[c * 4], __fmul_rn(qv.y, kv[c * 4 + 1]));
                    d4 = __fmaf_rn(qv.z, kv[c * 4 + 2], d4);
                    d4 = __fmaf_rn(qv.w, kv[c * 4 + 3], d4);
                    part[qi] = __fadd_rn(part[qi], d4);
                }
            }
        }
#pragma unroll
        for (int qi = 0; qi < NQ; ++qi) {
            float v = part[qi];
            v = __fadd_rn(v, __shfl_xor_sync(0xffffffff, v, 1));
            v = __fadd_rn(v, __shfl_xor_sync(0xffffffff, v, 2));
            v = __fadd_rn(v, __shfl_xor_sync(0xffffffff, v, 4));
            if ((qi & 7) == sub) ss[qi * 33 + grp] = (j < s1 && j <= p0 + qi / GQA) ? v : NEG_INF;
        }
        __syncthreads();
        // ---- online softmax, one thread per query row ----
        if (tid < NQ) {
            float mx = NEG_INF;
#pragma unroll
            for (int k = 0; k < TILE; ++k) mx = fmaxf(mx, ss[tid * 33 + k]);
            const float mold = mrow[tid], mnew = fmaxf(mold, mx);
            const bool dead = mnew == NEG_INF;
            const float f = dead ? 1.0f : expf(mold - mnew);
            float l = 0.0f;
#pragma unroll
            for (int k = 0; k < TILE; ++k) {
                const float p = dead ? 0.0f : expf(ss[tid * 33 + k] - mnew);
                pT[k * NQP + tid] = p;
                l = k == 0 ? __fmaf_rn(lrow[tid], f, p) : __fadd_rn(l, p);
            }
            mrow[tid] = mnew; lrow[tid] = l; fac[tid] = f;
        }
        __syncthreads();
        // ---- PV, thread = dim ----
        float vv[TILE];
#pragma unroll
        for (int k = 0; k < TILE; ++k) {
            const int jj = tile + k;
            vv[k] = 0.0f;
            if (jj < s1) {
                if constexpr (KT == KV_F16) {
                    vv[k] = h2f(*reinterpret_cast<const uint16_t *>(vc + jj * rowb + hk * headb + tid * 2));
                } else if constexpr (KT == KV_Q8_0) {
                    const char * hb = vc + jj * rowb + hk * headb;
                    vv[k] = __fmul_rn((float) reinterpret_cast<const int8_t *>(hb + Q8_QOFF)[tid],
                                      h2f(reinterpret_cast<const uint16_t *>(hb)[tid >> 5]));
                } else {
                    const char * b = vc + jj * rowb + hk * headb + (tid >> 5) * 18;
                    const float d = h2f(*reinterpret_cast<const uint16_t *>(b));
                    const int i = tid & 31;
                    const uint32_t by = (uint8_t) b[2 + (i & 15)];
                    vv[k] = (float) ((int) (i < 16 ? (by & 15) : (by >> 4)) - 8) * d;
                }
            }
        }
#pragma unroll
        for (int k = 0; k < TILE; ++k) {
#pragma unroll
            for (int c = 0; c < NQP / 4; ++c) {
                const float4 p = *reinterpret_cast<const float4 *>(&pT[k * NQP + c * 4]);
                const float pp[4] = {p.x, p.y, p.z, p.w};
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    const int qi = c * 4 + u;
                    if (qi < NQ) acc[qi] = k == 0 ? __fmaf_rn(acc[qi], fac[qi], __fmul_rn(pp[u], vv[0]))
                                                  : __fmaf_rn(pp[u], vv[k], acc[qi]);
                }
            }
        }
    }
    __syncthreads();
    const size_t base = (size_t) (hk * nsplit + sp) * NQ;
#pragma unroll
    for (int qi = 0; qi < NQ; ++qi) part_acc[(base + qi) * DH + tid] = acc[qi];
    if (tid < NQ) { part_ml[(base + tid) * 2] = mrow[tid]; part_ml[(base + tid) * 2 + 1] = lrow[tid]; }
}

template <int T>
__global__ void combine_k(const float * __restrict__ pa, const float * __restrict__ pm, float * __restrict__ out,
                          int Hq, int Hkv, int nsplit, const int32_t * __restrict__ p0p, int cap) {
    constexpr int NQ = GQA * T;
    if (p0p) { int chunk; split_plan(*p0p + T, cap, nsplit, chunk); }
    const int t = blockIdx.x / Hq, h = blockIdx.x % Hq, hk = h / GQA, qi = t * GQA + h % GQA, d = threadIdx.x;
    float M = NEG_INF;
    for (int s = 0; s < nsplit; ++s) M = fmaxf(M, pm[((size_t) (hk * nsplit + s) * NQ + qi) * 2]);
    float num = 0.0f, den = 0.0f;
    for (int s = 0; s < nsplit; ++s) {
        const size_t b = (size_t) (hk * nsplit + s) * NQ + qi;
        const float m = pm[b * 2];
        if (m == NEG_INF) continue;
        const float w = expf(m - M);
        den = __fmaf_rn(w, pm[b * 2 + 1], den);
        num = __fmaf_rn(w, pa[b * DH + d], num);
    }
    out[((size_t) t * Hq + h) * DH + d] = __fdiv_rn(num, den);
}

// p0p == nullptr: host p0/nsplit/chunk, grid = nsplit. p0p != nullptr: p0 and the plan come from *p0p on the device,
// nsplit (grid) is the upper bound for the largest n the caller allows.
template <int T, uint32_t KT>
void launch(const float * q, const kv_cache & c, float * out, int p0, int Hq, float scale, int nsplit, int chunk,
            float * pa, float * pm, const int32_t * p0p, int cap, cudaStream_t s) {
    attn_part_k<T, KT><<<dim3(nsplit, c.Hkv), NTHR, 0, s>>>(q, (const char *) c.k, (const char *) c.v, pa, pm, p0, Hq, c.Hkv, scale,
                                                            nsplit, chunk, p0p, cap);
    combine_k<T><<<T * Hq, DH, 0, s>>>(pa, pm, out, Hq, c.Hkv, nsplit, p0p, cap);
}

struct Scratch { float * acc = nullptr; float * ml = nullptr; size_t cap_acc = 0, cap_ml = 0; int sm = 0; };
Scratch g_scr[16];

// ---- append ----
__global__ void append_f16_k(const float * k, const float * v, char * kc, char * vc, int p0, int Hkv, int D, size_t rowb,
                             const int32_t * p0p, const kv_pg pg) {
    if (p0p) p0 = *p0p;
    const int t = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;   // over Hkv*D
    if (i >= Hkv * D) return;
    const size_t o = kv_off(pg, t, p0 + t, rowb);
    reinterpret_cast<__half *>(kc + o)[i] = __float2half(k[(size_t) t * Hkv * D + i]);
    reinterpret_cast<__half *>(vc + o)[i] = __float2half(v[(size_t) t * Hkv * D + i]);
}

// q8_0: one warp per 32-value block (fast/kv_q8.cuh), 8 blocks per CTA
__global__ void append_q8_k(const float * k, const float * v, char * kc, char * vc, int p0, int Hkv, int D, size_t rowb,
                            const int32_t * p0p, const kv_pg pg) {
    if (p0p) p0 = *p0p;
    const int t = blockIdx.y;
    const int nbh = D / 32;
    const int i = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);   // block index over Hkv * nbh
    if (i >= Hkv * nbh) return;                                          // whole warps leave together
    const int lane = threadIdx.x & 31, h = i / nbh, b = i % nbh;
    const size_t o = kv_off(pg, t, p0 + t, rowb) + (size_t) h * nbh * 34;
    const size_t src = (size_t) t * Hkv * D + (size_t) i * 32 + lane;
    q8_quant_block_warp(k[src], lane, kc + o, b, nbh);
    q8_quant_block_warp(v[src], lane, vc + o, b, nbh);
}

// q4_0: one warp per 32-value block (fast/kv_q8.cuh: ggml quantize_row_q4_0_ref), 8 blocks per CTA
__global__ void append_q4_k(const float * k, const float * v, char * kc, char * vc, int p0, int Hkv, int D, size_t rowb,
                            const int32_t * p0p, const kv_pg pg) {
    if (p0p) p0 = *p0p;
    const int t = blockIdx.y;
    const int i = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);   // block index over Hkv * D / 32
    if (i >= Hkv * D / 32) return;                                       // whole warps leave together
    const int lane = threadIdx.x & 31;
    const size_t o = kv_off(pg, t, p0 + t, rowb) + (size_t) i * 18;
    const size_t src = (size_t) t * Hkv * D + (size_t) i * 32 + lane;
    q4_quant_block_warp(k[src], lane, kc + o);
    q4_quant_block_warp(v[src], lane, vc + o);
}

// ---- prefill: block = (2 query tokens x 6 heads = 12 rows, kv head), 256 threads, 32-key tiles ----
constexpr int PF_TQ = 2, PF_NR = PF_TQ * GQA, PF_KT = 32;

__device__ __forceinline__ float pf_warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    return v;
}
__device__ __forceinline__ float pf_warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}

// row r = tt * 6 + hh (token tt of the tile, q head hk*6+hh). QK: 32 groups of 8 lanes, group = 3 rows x 4 keys,
// lane = dims 4l + 32i; PV: thread = dim, 12 row accumulators. K, then V, of a tile staged in smem as f32.
template <uint32_t KT>
__global__ void __launch_bounds__(NTHR) attn_prefill_k(const float * __restrict__ q, const char * __restrict__ kc,
                                                       const char * __restrict__ vc, float * __restrict__ out, int T,
                                                       int p0, int Hq, int Hkv, float scale, const kv_pg pg) {
    __shared__ __align__(16) float Qs[PF_NR][DH];
    __shared__ __align__(16) float KV[PF_KT][DH];
    __shared__ __align__(16) float Ps[PF_KT][PF_NR];   // probabilities [key][row]
    __shared__ float Ss[PF_NR][PF_KT + 1];             // scores [row][key]
    __shared__ float mrow[PF_NR], lrow[PF_NR], fac[PF_NR];

    const int t0 = blockIdx.x * PF_TQ, hk = blockIdx.y, tid = threadIdx.x;
    constexpr size_t HB = head_bytes<KT>();
    const size_t rowb = (size_t) Hkv * HB;
    const char * kbase = kc + (size_t) hk * HB;
    const char * vbase = vc + (size_t) hk * HB;
    const int32_t * tab = kv_tab(pg, 0);                  // one sequence per call
    const int tl = min(T - 1, t0 + PF_TQ - 1);            // last valid token of the tile
    const int kend = p0 + tl + 1;                         // keys this block needs: [0, kend)

    for (int i = tid; i < PF_NR * DH / 4; i += NTHR) {
        const int r = i / (DH / 4), d4 = i % (DH / 4);
        const int t = min(T - 1, t0 + r / GQA), h = hk * GQA + r % GQA;
        float4 v = reinterpret_cast<const float4 *>(q + ((size_t) t * Hq + h) * DH)[d4];
        v.x *= scale; v.y *= scale; v.z *= scale; v.w *= scale;
        reinterpret_cast<float4 *>(&Qs[r][0])[d4] = v;
    }
    if (tid < PF_NR) { mrow[tid] = NEG_INF; lrow[tid] = 0.0f; }

    float acc[PF_NR];
#pragma unroll
    for (int r = 0; r < PF_NR; ++r) acc[r] = 0.0f;
    const int g = tid >> 3, l = tid & 7, rb = g & 3, kb = g >> 2;

    auto load_tile = [&](const char * base, int k0) {   // KV[j][:] = f32(cache[k0 + j]), zero past kend
#pragma unroll
        for (int i = 0; i < PF_KT * DH / 8 / NTHR; ++i) {
            const int idx = tid + i * NTHR, j = idx >> 5, c = idx & 31;
            float4 lo = make_float4(0.f, 0.f, 0.f, 0.f), hi = lo;
            if (k0 + j < kend) {
                if constexpr (KT == KV_F16) {
                    const uint4 u = *reinterpret_cast<const uint4 *>(base + kv_off_tab(pg, tab, k0 + j, rowb) + c * 16);
                    const __half2 * h2 = reinterpret_cast<const __half2 *>(&u);
                    const float2 a = __half22float2(h2[0]), b = __half22float2(h2[1]);
                    const float2 cc = __half22float2(h2[2]), d = __half22float2(h2[3]);
                    lo = make_float4(a.x, a.y, b.x, b.y);
                    hi = make_float4(cc.x, cc.y, d.x, d.y);
                } else if constexpr (KT == KV_Q8_0) {   // dims c*8 .. c*8+7 = 8 int8 of block c/4
                    const char * hb = base + kv_off_tab(pg, tab, k0 + j, rowb);
                    const float d = h2f(reinterpret_cast<const uint16_t *>(hb)[c >> 2]);
                    const uint2 u = *reinterpret_cast<const uint2 *>(hb + Q8_QOFF + c * 8);
                    lo = make_float4(q8_lo(u.x, 0) * d, q8_lo(u.x, 1) * d, q8_lo(u.x, 2) * d, q8_lo(u.x, 3) * d);
                    hi = make_float4(q8_lo(u.y, 0) * d, q8_lo(u.y, 1) * d, q8_lo(u.y, 2) * d, q8_lo(u.y, 3) * d);
                } else {   // q4_0: dims c*8 .. c*8+7 = values w..w+7 of block c/4 (w = (c&3)*8): low or high nibbles
                    const char * hb = base + kv_off_tab(pg, tab, k0 + j, rowb) + (c >> 2) * 18;
                    const float d = h2f(*reinterpret_cast<const uint16_t *>(hb));
                    const int w = (c & 3) * 8, sh = w < 16 ? 0 : 4;
                    const uint16_t * b2 = reinterpret_cast<const uint16_t *>(hb + 2 + (w & 15));
                    float f[8];
#pragma unroll
                    for (int m = 0; m < 4; ++m) {
                        const uint32_t u = b2[m];
                        f[2 * m]     = (float) ((int) ((u >> sh) & 15) - 8) * d;
                        f[2 * m + 1] = (float) ((int) ((u >> (8 + sh)) & 15) - 8) * d;
                    }
                    lo = make_float4(f[0], f[1], f[2], f[3]);
                    hi = make_float4(f[4], f[5], f[6], f[7]);
                }
            }
            reinterpret_cast<float4 *>(&KV[j][c * 8])[0] = lo;
            reinterpret_cast<float4 *>(&KV[j][c * 8])[1] = hi;
        }
    };

    for (int k0 = 0; k0 < kend; k0 += PF_KT) {
        __syncthreads();                       // Q / m,l init visible; previous tile's V and Ps consumed
        load_tile(kbase, k0);
        __syncthreads();
        // ---- QK: 3 rows x 4 keys per group, 32 dims per lane ----
        float s[3][4];
#pragma unroll
        for (int a = 0; a < 3; ++a)
#pragma unroll
            for (int b = 0; b < 4; ++b) s[a][b] = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int d = 4 * l + 32 * i;
            float4 qv[3], kv[4];
#pragma unroll
            for (int a = 0; a < 3; ++a) qv[a] = *reinterpret_cast<const float4 *>(&Qs[rb * 3 + a][d]);
#pragma unroll
            for (int b = 0; b < 4; ++b) kv[b] = *reinterpret_cast<const float4 *>(&KV[kb * 4 + b][d]);
#pragma unroll
            for (int a = 0; a < 3; ++a)
#pragma unroll
                for (int b = 0; b < 4; ++b)
                    s[a][b] += qv[a].x * kv[b].x + qv[a].y * kv[b].y + qv[a].z * kv[b].z + qv[a].w * kv[b].w;
        }
#pragma unroll
        for (int a = 0; a < 3; ++a)
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                float v = s[a][b];
                v += __shfl_xor_sync(0xffffffff, v, 1);
                v += __shfl_xor_sync(0xffffffff, v, 2);
                v += __shfl_xor_sync(0xffffffff, v, 4);
                if (l == ((a * 4 + b) & 7)) {
                    const int r = rb * 3 + a, j = kb * 4 + b;
                    const int t = min(T - 1, t0 + r / GQA);
                    Ss[r][j] = k0 + j <= p0 + t ? v : NEG_INF;
                }
            }
        __syncthreads();                       // scores ready; K no longer needed
        load_tile(vbase, k0);                  // V into the same buffer (overlaps the softmax)
        // ---- online softmax: warp per row, lane = key ----
        const int w = tid >> 5, lane = tid & 31;
        for (int r = w; r < PF_NR; r += NTHR / 32) {
            const float sv = Ss[r][lane];
            const float mold = mrow[r];
            const float mnew = fmaxf(mold, pf_warp_max(sv));   // finite: key 0 is in the first tile for every row
            const float p = expf(sv - mnew);
            const float sum = pf_warp_sum(p);
            Ps[lane][r] = p;
            if (lane == 0) {
                const float f = expf(mold - mnew);
                fac[r] = f;
                lrow[r] = lrow[r] * f + sum;
                mrow[r] = mnew;
            }
        }
        __syncthreads();                       // V, Ps, fac ready
        // ---- PV: thread = dim ----
#pragma unroll
        for (int r = 0; r < PF_NR; ++r) acc[r] *= fac[r];
        const int jn = min(PF_KT, kend - k0);
        for (int j = 0; j < jn; ++j) {
            const float v = KV[j][tid];
            const float4 pa = *reinterpret_cast<const float4 *>(&Ps[j][0]);
            const float4 pb = *reinterpret_cast<const float4 *>(&Ps[j][4]);
            const float4 pc = *reinterpret_cast<const float4 *>(&Ps[j][8]);
            acc[0] += pa.x * v; acc[1] += pa.y * v; acc[2] += pa.z * v; acc[3] += pa.w * v;
            acc[4] += pb.x * v; acc[5] += pb.y * v; acc[6] += pb.z * v; acc[7] += pb.w * v;
            acc[8] += pc.x * v; acc[9] += pc.y * v; acc[10] += pc.z * v; acc[11] += pc.w * v;
        }
    }
    __syncthreads();
#pragma unroll
    for (int r = 0; r < PF_NR; ++r) {
        const int t = t0 + r / GQA;
        if (t < T) out[((size_t) t * Hq + hk * GQA + r % GQA) * DH + tid] = acc[r] / lrow[r];
    }
}

#include "fast/attn_pf2.inc"

} // namespace

bool attn_prefill(const float * q, const kv_cache & c, float * out, int T, int p0, int Hq, float scale, cudaStream_t s) {
    // P100E_ATTN_PF (read per call): 0 = attn_prefill_k (the original), 1..3 = attn_pf2_k with RT = 4 / 2 / 1, 4 (default) =
    // pick by T. Every variant is bitwise the same result (src/fast/attn_pf2.inc, tests/attn_pf_bench.cu).
    // Choice (P100, 100K keys): T <= 12 RT 1 (x2.9 at T = 9), T <= 48 RT 2 (x2.4 at T = 16),
    // above RT 4 (x1.5-1.9 at T = 48..2048).
    const char * e = getenv("P100E_ATTN_PF");
    int v = e && *e ? atoi(e) : 4;
    if (v == 4) v = T <= 12 ? 3 : T <= 48 ? 2 : 1;
    return attn_prefill_variant(q, c, out, T, p0, Hq, scale, s, v);
}

template <uint32_t KT, int RT>
static void launch_pf2(const float * q, const kv_cache & c, float * out, int T, int p0, int Hq, float scale, cudaStream_t s) {
    const dim3 grid((GQA * T + 3 * RT - 1) / (3 * RT), c.Hkv);
    attn_pf2_k<KT, RT><<<grid, NTHR, 0, s>>>(q, (const char *) c.k, (const char *) c.v, out, T, p0, Hq, c.Hkv, scale, kv_pg_of(c));
}

bool attn_prefill_variant(const float * q, const kv_cache & c, float * out, int T, int p0, int Hq, float scale, cudaStream_t s,
                          int variant) {
    if ((c.type != KV_F16 && c.type != KV_Q8_0 && c.type != KV_Q4_0) || c.D != DH || Hq != GQA * c.Hkv || T < 1 || !c.k)
        return false;
    if (variant >= 1 && variant <= 3 && c.type != KV_Q4_0) {
        if (c.type == KV_F16) {
            if (variant == 1) launch_pf2<KV_F16, 4>(q, c, out, T, p0, Hq, scale, s);
            else if (variant == 2) launch_pf2<KV_F16, 2>(q, c, out, T, p0, Hq, scale, s);
            else launch_pf2<KV_F16, 1>(q, c, out, T, p0, Hq, scale, s);
        } else {
            if (variant == 1) launch_pf2<KV_Q8_0, 4>(q, c, out, T, p0, Hq, scale, s);
            else if (variant == 2) launch_pf2<KV_Q8_0, 2>(q, c, out, T, p0, Hq, scale, s);
            else launch_pf2<KV_Q8_0, 1>(q, c, out, T, p0, Hq, scale, s);
        }
        return cudaPeekAtLastError() == cudaSuccess;
    }
    const dim3 grid((T + PF_TQ - 1) / PF_TQ, c.Hkv);
    const kv_pg pg = kv_pg_of(c);
    if (c.type == KV_F16)
        attn_prefill_k<KV_F16><<<grid, NTHR, 0, s>>>(q, (const char *) c.k, (const char *) c.v, out, T, p0, Hq, c.Hkv, scale, pg);
    else if (c.type == KV_Q8_0)
        attn_prefill_k<KV_Q8_0><<<grid, NTHR, 0, s>>>(q, (const char *) c.k, (const char *) c.v, out, T, p0, Hq, c.Hkv, scale, pg);
    else
        attn_prefill_k<KV_Q4_0><<<grid, NTHR, 0, s>>>(q, (const char *) c.k, (const char *) c.v, out, T, p0, Hq, c.Hkv, scale, pg);
    return cudaPeekAtLastError() == cudaSuccess;
}

size_t kv_row_bytes(uint32_t type, int Hkv, int D) {
    return type == KV_F16 ? (size_t) Hkv * D * 2 : type == KV_Q8_0 ? (size_t) Hkv * (D / 32) * 34 : (size_t) Hkv * (D / 32) * 18;
}

kv_cache kv_alloc(uint32_t type, int n_ctx, int Hkv, int D) {
    kv_cache c; c.type = type; c.n_ctx = n_ctx; c.Hkv = Hkv; c.D = D;
    if (type != KV_F16 && type != KV_Q4_0 && type != KV_Q8_0) return c;
    if (type != KV_F16 && D % 32) return c;
    const size_t bytes = kv_row_bytes(type, Hkv, D) * (size_t) n_ctx;
    if (cudaMalloc(&c.k, bytes) != cudaSuccess || cudaMalloc(&c.v, bytes) != cudaSuccess) {
        cudaGetLastError();
        kv_free(c);
    }
    return c;
}

void kv_free(kv_cache & c) {
    if (c.k) cudaFree(c.k);
    if (c.v) cudaFree(c.v);
    c.k = c.v = nullptr;
}

static void append_impl(kv_cache & c, const float * k, const float * v, int p0, const int32_t * p0p, int T, cudaStream_t s) {
    const size_t rowb = kv_row_bytes(c.type, c.Hkv, c.D);
    const kv_pg pg = kv_pg_of(c);
    if (c.type == KV_F16) {
        const int n = c.Hkv * c.D;
        append_f16_k<<<dim3((n + 255) / 256, T), 256, 0, s>>>(k, v, (char *) c.k, (char *) c.v, p0, c.Hkv, c.D, rowb, p0p, pg);
    } else if (c.type == KV_Q8_0) {
        const int n = c.Hkv * c.D / 32;   // blocks, one warp each
        append_q8_k<<<dim3((n + 7) / 8, T), 256, 0, s>>>(k, v, (char *) c.k, (char *) c.v, p0, c.Hkv, c.D, rowb, p0p, pg);
    } else {
        const int n = c.Hkv * c.D / 32;   // blocks, one warp each
        append_q4_k<<<dim3((n + 7) / 8, T), 256, 0, s>>>(k, v, (char *) c.k, (char *) c.v, p0, c.Hkv, c.D, rowb, p0p, pg);
    }
}

__global__ void fill_random_k(uint16_t * k, uint16_t * v, size_t w0, size_t nw, uint32_t type, int head_words,
                              uint32_t seed) {
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < nw; i += (size_t) gridDim.x * blockDim.x) {
        const size_t w = w0 + i;
        uint32_t h = (uint32_t) w * 2654435761u ^ seed;
        h ^= h >> 16; h *= 0x7feb352d; h ^= h >> 15; h *= 0x846ca68b; h ^= h >> 16;
        const int hw = (int) (w % (size_t) head_words);
        uint16_t a, b;
        if (type == KV_F16) {
            a = __half_as_ushort(__float2half((float) (h & 0xffff) / 32768.0f - 1.0f));
            b = __half_as_ushort(__float2half((float) (h >> 16) / 32768.0f - 1.0f));
        } else if ((type == KV_Q8_0 && hw < DH / 32) || (type == KV_Q4_0 && hw % 9 == 0)) {
            a = b = __half_as_ushort(__float2half(type == KV_Q8_0 ? 1.0f / 127.0f : 0.125f));   // block scale
        } else {
            a = (uint16_t) h; b = (uint16_t) (h >> 16);
        }
        k[w] = a;
        v[w] = b;
    }
}

void kv_fill_random(kv_cache & c, int p0, int p1, uint32_t seed, cudaStream_t s) {
    if (!c.k || p1 <= p0 || c.D != DH) return;
    const size_t rowb = kv_row_bytes(c.type, c.Hkv, c.D);
    if (c.pt) {   // paged: each page's run of positions as a contiguous view (table read back from the device)
        const int ps = 1 << c.psh;
        std::vector<int32_t> tab((size_t) ((p1 - 1) >> c.psh) + 1);
        if (cudaMemcpy(tab.data(), c.pt, tab.size() * 4, cudaMemcpyDeviceToHost) != cudaSuccess) return;
        for (int a = p0; a < p1;) {
            const int pg = a >> c.psh, b = std::min(p1, (pg + 1) * ps);
            kv_cache v = c;
            v.pt = nullptr; v.rtab = nullptr;
            v.k = (char *) c.k + (size_t) tab[pg] * c.pstride;
            v.v = (char *) c.v + (size_t) tab[pg] * c.pstride;
            v.n_ctx = ps;
            kv_fill_random(v, a - pg * ps, b - pg * ps, seed + (uint32_t) pg * 0x632be5abu, s);
            a = b;
        }
        return;
    }
    const int head_words = (int) (rowb / c.Hkv / 2);
    const size_t w0 = (size_t) p0 * rowb / 2, nw = (size_t) (p1 - p0) * rowb / 2;
    fill_random_k<<<1024, 256, 0, s>>>((uint16_t *) c.k, (uint16_t *) c.v, w0, nw, c.type, head_words, seed);
}

void kv_append(kv_cache & c, const float * k, const float * v, int p0, int T, cudaStream_t s) {
    append_impl(c, k, v, p0, nullptr, T, s);
}

void kv_append_dev(kv_cache & c, const float * k, const float * v, const int32_t * p0_dev, int T, cudaStream_t s) {
    append_impl(c, k, v, 0, p0_dev, T, s);
}

// scratch for the split partials, per device, allocated once at the largest size (MAX_SPLIT, T = 4) so the pointer
// never changes afterwards (captured graphs keep it)
static bool attn_scratch(const kv_cache & c, Scratch *& out) {
    int dev = 0; cudaGetDevice(&dev);
    Scratch & sc = g_scr[dev & 15];
    if (!sc.sm) cudaDeviceGetAttribute(&sc.sm, cudaDevAttrMultiProcessorCount, dev);
    const size_t a = (size_t) c.Hkv * MAX_SPLIT * GQA * 4 * DH, m = (size_t) c.Hkv * MAX_SPLIT * GQA * 4 * 2;
    if (a > sc.cap_acc || m > sc.cap_ml) {
        if (sc.acc) cudaFree(sc.acc);
        if (sc.ml) cudaFree(sc.ml);
        sc.acc = sc.ml = nullptr; sc.cap_acc = sc.cap_ml = 0;
        if (cudaMalloc(&sc.acc, a * sizeof(float)) != cudaSuccess || cudaMalloc(&sc.ml, m * sizeof(float)) != cudaSuccess) {
            cudaGetLastError();
            fprintf(stderr, "p100e::fast::attn_decode: scratch allocation failed\n");
            return false;
        }
        sc.cap_acc = a; sc.cap_ml = m;
    }
    out = &sc;
    return true;
}

static bool shape_ok(const kv_cache & c, int T, int Hq) {
    if (c.D != DH || Hq != GQA * c.Hkv || T < 1 || T > 4 || !c.k) {
        fprintf(stderr, "p100e::fast::attn_decode: unsupported shape (D=%d Hq=%d Hkv=%d T=%d)\n", c.D, Hq, c.Hkv, T);
        return false;
    }
    return true;
}

static void dispatch(const float * q, const kv_cache & c, float * out, int T, int p0, int Hq, float scale, int nsplit,
                     int chunk, const Scratch & sc, const int32_t * p0p, int cap, cudaStream_t s) {
    float * pa = sc.acc, * pm = sc.ml;
#define L(TT) if (c.type == KV_F16) launch<TT, KV_F16>(q, c, out, p0, Hq, scale, nsplit, chunk, pa, pm, p0p, cap, s); \
              else if (c.type == KV_Q8_0) launch<TT, KV_Q8_0>(q, c, out, p0, Hq, scale, nsplit, chunk, pa, pm, p0p, cap, s); \
              else launch<TT, KV_Q4_0>(q, c, out, p0, Hq, scale, nsplit, chunk, pa, pm, p0p, cap, s);

    switch (T) { case 1: L(1) break; case 2: L(2) break; case 3: L(3) break; default: L(4) break; }
#undef L
}

// P100E_ATTN_DEC=1: the v1 kernels below; default: fast::attn_decode2 (fast/attn_dec2.cu)
static bool use_v1() {
    static const bool v1 = getenv("P100E_ATTN_DEC") && atoi(getenv("P100E_ATTN_DEC")) == 1;
    return v1;
}

bool attn_decode(const float * q, const kv_cache & c, float * out, int T, int p0, int Hq, float scale, cudaStream_t s) {
    if ((!use_v1() || c.pt) && attn_decode2(q, c, out, T, p0, nullptr, Hq, scale, s)) return true;
    if (c.pt) { fprintf(stderr, "p100e::fast::attn_decode: paged cache needs attn_decode2 (unsupported shape)\n"); return false; }
    Scratch * sc = nullptr;
    if (!shape_ok(c, T, Hq) || !attn_scratch(c, sc)) return false;
    const int cap = std::min(MAX_SPLIT, 3 * sc->sm / c.Hkv);
    int nsplit, chunk;
    split_plan(p0 + T, cap, nsplit, chunk);
    dispatch(q, c, out, T, p0, Hq, scale, nsplit, chunk, *sc, nullptr, cap, s);
    return true;
}

bool attn_decode_rows(const float * q, const kv_cache & c, float * out, int T, const int32_t * pos, int Hq, float scale,
                      cudaStream_t s) {
    return pos && attn_decode2(q, c, out, T, 0, nullptr, Hq, scale, s, pos);
}

bool attn_decode_dev(const float * q, const kv_cache & c, float * out, int T, const int32_t * p0_dev, int max_n, int Hq,
                     float scale, cudaStream_t s) {
    if ((!use_v1() || c.pt) && attn_decode2(q, c, out, T, 0, p0_dev, Hq, scale, s)) return true;
    if (c.pt) { fprintf(stderr, "p100e::fast::attn_decode_dev: paged cache needs attn_decode2 (unsupported shape)\n"); return false; }
    Scratch * sc = nullptr;
    if (!shape_ok(c, T, Hq) || !attn_scratch(c, sc)) return false;
    const int cap = std::min(MAX_SPLIT, 3 * sc->sm / c.Hkv);
    // upper bound of split_plan(n).nsplit over n <= max_n (the plan's final nsplit never exceeds its first guess)
    const int nsplit_max = std::max(1, std::min(cap, (max_n + 63) / 64));
    dispatch(q, c, out, T, 0, Hq, scale, nsplit_max, 0, *sc, p0_dev, cap, s);
    return true;
}

} // namespace p100e::fast
