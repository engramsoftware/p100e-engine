// Fold-GEMM prefill attention, see attn_fold.cuh. The QK and PV kernels follow fa_fold_qk2 / fa_fold_pv2 of the
// Kmic-68 llama.cpp fork (ggml-cuda/fattn-gemm.cu, MIT): 128x128 tiles, 8x8 outputs per thread as half2 (lanes =
// even/odd k), XOR-swizzled smem, P in a blocked [n/128][key/32][128][32] layout so a PV B tile is one 8 KiB block.
#include "fast/attn_fold.cuh"
#include "fast/kv_q8.cuh"
#include "fast/kv_page.cuh"

#include <cuda_fp16.h>
#include <algorithm>
#include <cfloat>
#include <cstdint>
#include <cstdlib>
#include <limits>

namespace p100e::fast {
namespace {

constexpr int DH = 256, GQA = 6;
constexpr int BM = 128, BN = 128, BK2 = 16, TK = 128, MAXTILE = 16;   // chunk <= MAXTILE * TK = 2048 keys
constexpr float LOG2E = 1.44269504088896341f;
#define NEG_INF __int_as_float(0xff800000)

struct geo {
    int nsplit, Lsp, c, nkv;   // splits, keys per split, chunk offset within a split, total keys (p0 + T)
    int p0, T, N, C, nblk;     // first query position, tokens, rows (6T), chunk size, row blocks (N / 128 rounded up)
    int z0;                    // first z of this launch (the z range can be split over two streams)
};
__device__ __forceinline__ int g_off(const geo & g, int z) { return (z % g.nsplit) * g.Lsp + g.c; }
// valid keys of the current chunk of split z
__device__ __forceinline__ int g_nkvc(const geo & g, int z) {
    return max(0, min(min(g.C, g.Lsp - g.c), g.nkv - g_off(g, z)));
}
// keys of the current chunk visible to row n (causal: query t = n % T sees positions <= p0 + t)
__device__ __forceinline__ int g_vis(const geo & g, int z, int n) {
    if (n >= g.N) return 0;
    return max(0, min(g_nkvc(g, z), g.p0 + n % g.T + 1 - g_off(g, z)));
}
__device__ __forceinline__ int frow(int i) { return (i < 4 ? 0 : 64) + (threadIdx.x >> 4) * 4 + (i & 3); }
__device__ __forceinline__ int fcol(int j) { return (j < 4 ? 0 : 64) + (threadIdx.x & 15) * 4 + (j & 3); }

// Q16[hk][n][D] = f16(q[t][hk*6 + hh][:] * qs), n = hh * T + t. grid (T, Hq), 64 threads.
__global__ void q16_k(const float * __restrict__ q, half * __restrict__ Q16, int T, int Hq, float qs) {
    const int t = blockIdx.x, h = blockIdx.y, hk = h / GQA, hh = h % GQA;
    const float4 v = reinterpret_cast<const float4 *>(q + ((size_t) t * Hq + h) * DH)[threadIdx.x];
    half2 * dst = reinterpret_cast<half2 *>(Q16 + ((size_t) hk * GQA * T + (size_t) hh * T + t) * DH) + threadIdx.x * 2;
    dst[0] = __floats2half2_rn(v.x * qs, v.y * qs);
    dst[1] = __floats2half2_rn(v.z * qs, v.w * qs);
}

// One chunk of the cache for every (kv head, split) z: K16[z][C][D] (key rows), Vp[z][C/2][D] half2 (lo = even key,
// hi = odd key: the PV A operand). Keys past the chunk's valid count are zero (their P is zero too, and 0 * garbage
// could be NaN). Thread = (key pair, 32-dim block); grid (C / 64, nz), 256 threads.
template <uint32_t KT>
__global__ void kv16_k(const char * __restrict__ kc, const char * __restrict__ vc, half * __restrict__ K16,
                       half2 * __restrict__ Vp, int Hkv, const geo g, const kv_pg pg) {
    const int z = blockIdx.y + g.z0, hk = z / g.nsplit;
    const int b = threadIdx.x & 7;
    const int kp = blockIdx.x * 32 + (threadIdx.x >> 3);
    if (2 * kp >= g.C) return;
    const int nkv_c = g_nkvc(g, z), off = g_off(g, z);
    constexpr size_t HB = KT == KV_F16 ? (size_t) DH * 2 : KT == KV_Q8_0 ? (size_t) (DH / 32) * 34 : (size_t) (DH / 32) * 18;
    const size_t rowb = (size_t) Hkv * HB;
    float vv[2][32];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        const int key = 2 * kp + r;
        uint4 * k16 = reinterpret_cast<uint4 *>(K16 + ((size_t) z * g.C + key) * DH + b * 32);
        if (key < nkv_c) {
            const size_t ko = kv_off(pg, 0, off + key, rowb);   // one sequence per call
            const char * kh = kc + ko + hk * HB;
            const char * vh = vc + ko + hk * HB;
            if constexpr (KT == KV_F16) {
                const uint4 * ks = reinterpret_cast<const uint4 *>(kh + b * 64);
                const uint4 * vs = reinterpret_cast<const uint4 *>(vh + b * 64);
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    k16[i] = ks[i];
                    const uint4 u = vs[i];
                    const __half2 * h2 = reinterpret_cast<const __half2 *>(&u);
#pragma unroll
                    for (int m = 0; m < 4; ++m) {
                        const float2 f = __half22float2(h2[m]);
                        vv[r][i * 8 + m * 2] = f.x;
                        vv[r][i * 8 + m * 2 + 1] = f.y;
                    }
                }
            } else if constexpr (KT == KV_Q4_0) {   // block b: f16 d, byte i = value i | value i+16 << 4
                const char * kb = kh + b * 18, * vb = vh + b * 18;
                const float dk = __half2float(*reinterpret_cast<const __half *>(kb));
                const float dv = __half2float(*reinterpret_cast<const __half *>(vb));
                half kk[32];
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    const int qk = (uint8_t) kb[2 + i], qv = (uint8_t) vb[2 + i];
                    kk[i]      = __float2half(dk * (float) ((qk & 15) - 8));
                    kk[i + 16] = __float2half(dk * (float) ((qk >> 4) - 8));
                    vv[r][i]      = dv * (float) ((qv & 15) - 8);
                    vv[r][i + 16] = dv * (float) ((qv >> 4) - 8);
                }
#pragma unroll
                for (int i = 0; i < 4; ++i) k16[i] = *reinterpret_cast<const uint4 *>(kk + 8 * i);
            } else {
                const float dk = __half2float(reinterpret_cast<const __half *>(kh)[b]);
                const float dv = __half2float(reinterpret_cast<const __half *>(vh)[b]);
                const uint4 * ks = reinterpret_cast<const uint4 *>(kh + (DH / 32) * 2 + b * 32);
                const uint4 * vs = reinterpret_cast<const uint4 *>(vh + (DH / 32) * 2 + b * 32);
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    const uint4 uk = ks[i], uv = vs[i];
                    const uint32_t wk[4] = {uk.x, uk.y, uk.z, uk.w}, wv[4] = {uv.x, uv.y, uv.z, uv.w};
                    half kk[16];
#pragma unroll
                    for (int m = 0; m < 4; ++m)
#pragma unroll
                        for (int e = 0; e < 4; ++e) {
                            kk[m * 4 + e] = __float2half(q8_lo(wk[m], e) * dk);
                            vv[r][i * 16 + m * 4 + e] = q8_lo(wv[m], e) * dv;
                        }
                    k16[2 * i] = *reinterpret_cast<const uint4 *>(kk);
                    k16[2 * i + 1] = *reinterpret_cast<const uint4 *>(kk + 8);
                }
            }
        } else {
#pragma unroll
            for (int i = 0; i < 4; ++i) k16[i] = make_uint4(0, 0, 0, 0);
#pragma unroll
            for (int j = 0; j < 32; ++j) vv[r][j] = 0.0f;
        }
    }
    uint4 * vp = reinterpret_cast<uint4 *>(Vp + ((size_t) z * (g.C / 2) + kp) * DH + b * 32);
#pragma unroll
    for (int j = 0; j < 32; j += 4) {
        half2 p[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) p[e] = __floats2half2_rn(vv[0][j + e], vv[1][j + e]);
        vp[j / 4] = *reinterpret_cast<const uint4 *>(p);
    }
}

// ex2.approx.ftz.f32 for x <= 0. Same MUFU.EX2 as exp2f for x >= -126; below that exp2f returns a denormal where this
// returns 0, which is 0 in f16 and cannot change the row sum (the tile's max element is exactly 1/8).
__device__ __forceinline__ float fast_ex2(float x) {
    float r;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
}

// QK + softmax tile, two CTAs per SM (fa_fold_qk2): grid (ntile, nblk, nz). S tile = 128 keys (rows of A = K16) x 128
// query rows (B = Q16), k = D; the fp16 chains run over all of D (two lanes of D/2 products each), then one fp32 fold.
// P = 2^(S' - m_t) (S' = 4 log2(e) S, Q carried scale/4; m_t = tile max + 3, so P <= 1/8) in f16 to the blocked
// layout, m_t and the fp32 row sum l_t per (tile, row) to mt / lt [z][C/TK][N].
template <bool OPT>
__global__ void __launch_bounds__(256, 2) qk_k(const half * __restrict__ K16, const half * __restrict__ Q16,
                                               half * __restrict__ P, float * __restrict__ mt, float * __restrict__ lt,
                                               const geo g) {
    union smem_t {
        struct { uint32_t As[BK2][BM]; uint32_t Bs[BK2][BN]; } t;
        float red[16][BN];
        uint2 Ps[BN * 16];
    };
    __shared__ __align__(16) smem_t sm;
    __shared__ float colv[BN];
    __shared__ int colvis[BN];

    const int z = blockIdx.z + g.z0, hk = z / g.nsplit;
    const int m0 = blockIdx.x * TK, n0 = blockIdx.y * BN;
    const int t = threadIdx.x, tx = t & 15, ty = t >> 4;
    const int N = g.N, C = g.C;
    const int64_t pblk = ((int64_t) z * g.nblk + blockIdx.y) * (C / 32);   // first 32-key block of this row block

    int vis = 0;
    if (t < BN) {
        vis = g_vis(g, z, n0 + t);
        colvis[t] = vis;
    }
    // every row sees none of this tile's keys: P = 0, m_t = -inf, l_t = 0 without the GEMM
    if (__syncthreads_and(t >= BN || vis <= m0)) {
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const int idx = t + 256 * k, kb = idx >> 9, r = (idx >> 2) & 127;
            *reinterpret_cast<uint4 *>(P + ((pblk + m0 / 32 + kb) * BN + r) * 32 + (idx & 3) * 8) = make_uint4(0, 0, 0, 0);
        }
        if (t < BN && n0 + t < N) {
            const int64_t o = ((int64_t) z * (C / TK) + blockIdx.x) * N + n0 + t;
            mt[o] = NEG_INF;
            lt[o] = 0.0f;
        }
        return;
    }
    const bool any_mask = __syncthreads_or(t < BN && vis < m0 + TK);

    const half * A = K16 + ((int64_t) z * C + m0) * DH;
    const half * B = Q16 + ((int64_t) hk * N + n0) * DH;
    const int nB = N - n0;

    half2 hh[8][8];
#pragma unroll
    for (int i = 0; i < 8; i++)
#pragma unroll
        for (int j = 0; j < 8; j++) hh[i][j] = make_half2(0.0f, 0.0f);
    uint4 ra[2], rb[2];
    auto gload = [&](const int k0) {
#pragma unroll
        for (int i = 0; i < 2; i++) {
            const int l = t + 256 * i, r = l >> 2, c = l & 3;
            ra[i] = *reinterpret_cast<const uint4 *>(A + (int64_t) r * DH + k0 + c * 8);
            rb[i] = r < nB ? *reinterpret_cast<const uint4 *>(B + (int64_t) r * DH + k0 + c * 8) : make_uint4(0, 0, 0, 0);
        }
    };
    constexpr int nit = DH / (2 * BK2);
    gload(0);
    for (int it = 0; it < nit; it++) {
        if (it > 0) __syncthreads();
#pragma unroll
        for (int i = 0; i < 2; i++) {
            const int l = t + 256 * i, r = l >> 2, c = l & 3, rs = r ^ (c << 3);
            sm.t.As[c * 4 + 0][rs] = ra[i].x; sm.t.As[c * 4 + 1][rs] = ra[i].y;
            sm.t.As[c * 4 + 2][rs] = ra[i].z; sm.t.As[c * 4 + 3][rs] = ra[i].w;
            sm.t.Bs[c * 4 + 0][rs] = rb[i].x; sm.t.Bs[c * 4 + 1][rs] = rb[i].y;
            sm.t.Bs[c * 4 + 2][rs] = rb[i].z; sm.t.Bs[c * 4 + 3][rs] = rb[i].w;
        }
        __syncthreads();
        if (it + 1 < nit) gload((it + 1) * 2 * BK2);
#pragma unroll
        for (int k2 = 0; k2 < BK2; k2++) {
            const int sw = (k2 >> 2) << 3;
            const uint4 a0 = *reinterpret_cast<const uint4 *>(&sm.t.As[k2][(ty * 4) ^ sw]);
            const uint4 a1 = *reinterpret_cast<const uint4 *>(&sm.t.As[k2][(64 + ty * 4) ^ sw]);
            const uint4 b0 = *reinterpret_cast<const uint4 *>(&sm.t.Bs[k2][(tx * 4) ^ sw]);
            const uint4 b1 = *reinterpret_cast<const uint4 *>(&sm.t.Bs[k2][(64 + tx * 4) ^ sw]);
            const uint32_t a[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
            const uint32_t b[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
            for (int j = 0; j < 8; j++)
#pragma unroll
                for (int i = 0; i < 8; i++)
                    hh[i][j] = __hfma2(*reinterpret_cast<const half2 *>(&b[j]), *reinterpret_cast<const half2 *>(&a[i]), hh[i][j]);
        }
    }

    // logits in log2 units; masked keys -inf; column max over the tile's 128 keys
    float acc[8][8], cmax[8];
    auto logits = [&](const bool masked) {
#pragma unroll
        for (int j = 0; j < 8; j++) {
            cmax[j] = -FLT_MAX / 2.0f;
            const int v = masked ? colvis[fcol(j)] : TK + m0;
#pragma unroll
            for (int i = 0; i < 8; i++) {
                float s = (__low2float(hh[i][j]) + __high2float(hh[i][j])) * (4.0f * LOG2E);
                if (masked && m0 + frow(i) >= v) s = NEG_INF;
                acc[i][j] = s;
                cmax[j] = fmaxf(cmax[j], s);
            }
        }
    };
    if (OPT && !any_mask) logits(false); else logits(true);
    __syncthreads();   // tile reads done: red / Ps alias them
#pragma unroll
    for (int j = 0; j < 8; j++) sm.red[ty][fcol(j)] = cmax[j];
    __syncthreads();
    if (t < BN) {
        float m = -FLT_MAX / 2.0f;
#pragma unroll
        for (int r = 0; r < 16; r++) m = fmaxf(m, sm.red[r][t]);
        colv[t] = m + 3.0f;
    }
    __syncthreads();
    float csum[8];
#pragma unroll
    for (int j = 0; j < 8; j++) csum[j] = 0.0f;
    // P in f16, staged through smem in two 64-key halves, written as whole 16-byte runs of the blocked layout
#pragma unroll
    for (int ih = 0; ih < 2; ih++) {
        if (ih > 0) __syncthreads();
#pragma unroll
        for (int j = 0; j < 8; j++) {
            const int c = fcol(j);
            const float m = colv[c];
            half pp[4];
#pragma unroll
            for (int q = 0; q < 4; q++) {
                const float p = OPT ? fast_ex2(acc[ih * 4 + q][j] - m) : exp2f(acc[ih * 4 + q][j] - m);   // -inf -> 0
                pp[q] = __float2half(p);
                csum[j] += p;
            }
            sm.Ps[c * 16 + (ty ^ (((c >> 2) & 7) << 1))] = *reinterpret_cast<const uint2 *>(pp);
        }
        __syncthreads();
        const uint4 * Ps = reinterpret_cast<const uint4 *>(sm.Ps);
#pragma unroll
        for (int k = 0; k < 4; k++) {
            const int idx = t + 256 * k, kbl = idx >> 9, r = (idx >> 2) & 127, q = kbl * 4 + (idx & 3);
            *reinterpret_cast<uint4 *>(P + ((pblk + m0 / 32 + ih * 2 + kbl) * BN + r) * 32 + (idx & 3) * 8) =
                Ps[r * 8 + (q ^ ((r >> 2) & 7))];
        }
    }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < 8; j++) sm.red[ty][fcol(j)] = csum[j];
    __syncthreads();
    if (t < BN && n0 + t < N) {
        float s = 0.0f;
#pragma unroll
        for (int r = 0; r < 16; r++) s += sm.red[r][t];
        const int64_t o = ((int64_t) z * (C / TK) + blockIdx.x) * N + n0 + t;
        mt[o] = colv[t] <= -FLT_MAX / 8.0f ? NEG_INF : colv[t];
        lt[o] = s;
    }
}

// grid (DV / BM, nblk, nz), one CTA per SM (fa_fold_pv2). O_out = O_in * corr + sum_t fac_t * (V P_t): A = V as half2
// key pairs [k2][DV], B = P blocked; the fp16 chains restart at every 128-key tile (HMUL2) and fold into fp32 with the
// tile's factor fac_t = 2^(m_t - M). Prologue per row: M = max(m_in, m_t), corr = 2^(m_in - M), l = l * corr + sum_t
// fac_t l_t (blockIdx.x == 0 writes l and m_out). Only the tiles up to the block's last visible key run.
template <bool OPT>
__global__ void __launch_bounds__(256, 1) pv_k(const half2 * __restrict__ Vp, const half * __restrict__ P,
                                               const float * __restrict__ mt, const float * __restrict__ lt,
                                               const float * __restrict__ m_in, float * __restrict__ m_out,
                                               float * __restrict__ l_state, const float * __restrict__ O_in,
                                               float * __restrict__ O_out, const int ntile, const geo g) {
    __shared__ __align__(16) uint32_t As[2][BK2][BM];
    __shared__ __align__(16) uint32_t Bs[2][BK2][BN];
    __shared__ float corr[BN];
    __shared__ float fac[MAXTILE][BN];
    __shared__ int kmax;

    const int z = blockIdx.z + g.z0;
    const int m0 = blockIdx.x * BM, n0 = blockIdx.y * BN;
    const int t = threadIdx.x, tx = t & 15, ty = t >> 4;
    const int N = g.N, C = g.C;
    constexpr int DV = DH;

    if (t == 0) kmax = 0;
    __syncthreads();
    if (t < BN) {
        const int v = g_vis(g, z, n0 + t);
        if (v > 0) atomicMax(&kmax, v);
    }
    auto prologue = [&]() {
        if (t < BN) {
            const int n = n0 + t;
            float c = 0.0f;
            if (n < N) {
                float mm[MAXTILE];
#pragma unroll
                for (int tt = 0; tt < MAXTILE; tt++) mm[tt] = tt < ntile ? mt[((int64_t) z * (C / TK) + tt) * N + n] : NEG_INF;
                const float mo = m_in[(int64_t) z * N + n];
                float M = mo;
#pragma unroll
                for (int tt = 0; tt < MAXTILE; tt++) M = fmaxf(M, mm[tt]);
                c = mo == NEG_INF ? 0.0f : exp2f(mo - M);
                if (blockIdx.x == 0) {
                    float l = l_state[(int64_t) z * N + n] * c;
                    for (int tt = 0; tt < ntile; tt++)
                        if (mm[tt] != NEG_INF) l += lt[((int64_t) z * (C / TK) + tt) * N + n] * exp2f(mm[tt] - M);
                    l_state[(int64_t) z * N + n] = l;
                    m_out[(int64_t) z * N + n] = M;
                }
#pragma unroll
                for (int tt = 0; tt < MAXTILE; tt++)
                    if (tt < ntile) fac[tt][t] = mm[tt] == NEG_INF ? 0.0f : exp2f(mm[tt] - M);
            } else {
                for (int tt = 0; tt < ntile; tt++) fac[tt][t] = 0.0f;
            }
            corr[t] = c;
        }
    };
    __syncthreads();
    const int ntile_run = max(1, min(ntile, (kmax + TK - 1) / TK));

    float acc[8][8];
    half2 hh[8][8];
#pragma unroll
    for (int i = 0; i < 8; i++)
#pragma unroll
        for (int j = 0; j < 8; j++) {
            acc[i][j] = 0.0f;
            hh[i][j] = make_half2(0.0f, 0.0f);
        }

    const uint32_t * A = reinterpret_cast<const uint32_t *>(Vp + (int64_t) z * (C / 2) * DV + m0);
    const half * B = P + ((int64_t) z * g.nblk + blockIdx.y) * C * BN;
    const uint4 * pa[2];
    const uint4 * pb[2];
    int sa[2], sb[2];
#pragma unroll
    for (int i = 0; i < 2; i++) {
        const int l = t + 256 * i;
        const int ra_ = l >> 5, ca = l & 31;
        pa[i] = reinterpret_cast<const uint4 *>(A + (int64_t) ra_ * DV + ca * 4);
        sa[i] = ra_ * BM + ((ca * 4) ^ ((ra_ >> 2) << 3));
        const int rb_ = l >> 2, cb = l & 3;
        pb[i] = reinterpret_cast<const uint4 *>(B + rb_ * 32 + cb * 8);
        sb[i] = (cb * 4) * BN + (rb_ ^ (cb << 3));
    }
    const int64_t da = (int64_t) BK2 * DV / 4;   // uint4 per tile along A
    constexpr int db = BN * 32 / 8;              // uint4 per tile along B
    int ao[4], bo[4];
#pragma unroll
    for (int q = 0; q < 4; q++) {
        ao[q] = (ty * 4) ^ (q << 3);
        bo[q] = (tx * 4) ^ (q << 3);
    }
    uint4 ra[2], rb[2];
    auto gload = [&](const int it) {
#pragma unroll
        for (int i = 0; i < 2; i++) {
            ra[i] = pa[i][it * da];
            rb[i] = pb[i][it * db];
        }
    };
    auto sstore = [&](const int buf) {
#pragma unroll
        for (int i = 0; i < 2; i++) {
            *reinterpret_cast<uint4 *>(&As[buf][0][0] + sa[i]) = ra[i];
            uint32_t * b = &Bs[buf][0][0] + sb[i];
            b[0] = rb[i].x; b[BN] = rb[i].y; b[2 * BN] = rb[i].z; b[3 * BN] = rb[i].w;
        }
    };
    auto tile = [&](const int buf, const bool restart) {
#pragma unroll
        for (int k2 = 0; k2 < BK2; k2++) {
            const uint4 a0 = *reinterpret_cast<const uint4 *>(&As[buf][k2][ao[k2 >> 2]]);
            const uint4 a1 = *reinterpret_cast<const uint4 *>(&As[buf][k2][ao[k2 >> 2] + 64]);
            const uint4 b0 = *reinterpret_cast<const uint4 *>(&Bs[buf][k2][bo[k2 >> 2]]);
            const uint4 b1 = *reinterpret_cast<const uint4 *>(&Bs[buf][k2][bo[k2 >> 2] + 64]);
            const uint32_t a[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
            const uint32_t b[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
            for (int i = 0; i < 8; i++)
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    const half2 ai = *reinterpret_cast<const half2 *>(&a[i]);
                    const half2 bj = *reinterpret_cast<const half2 *>(&b[j]);
                    hh[i][j] = k2 == 0 && restart ? __hmul2(ai, bj) : __hfma2(ai, bj, hh[i][j]);
                }
        }
    };
    auto fold = [&](const int tt) {
#pragma unroll
        for (int j = 0; j < 8; j++) {
            const float f = fac[tt][fcol(j)];
#pragma unroll
            for (int i = 0; i < 8; i++) acc[i][j] += (__low2float(hh[i][j]) + __high2float(hh[i][j])) * f;
        }
    };

    constexpr int TPF = TK / 2 / BK2;   // smem tiles per fold (128 keys)
    const int ntl = ntile_run * TPF;    // tiles of 32 keys, a multiple of 4
    gload(0);
    prologue();
    sstore(0);
    __syncthreads();
    const int it_pf = max(0, ntl - 8);
    for (int it = 0; it < ntl; it += 2) {
        if (OPT && it == it_pf) {   // pull O_in into L2 under the last tiles (1 CTA/SM: the read is otherwise exposed)
#pragma unroll
            for (int j = 0; j < 8; j++) {
                const int n = min(n0 + fcol(j), N - 1);
                const float * po = O_in + ((int64_t) z * N + n) * DV + m0 + ty * 4;
                asm volatile("prefetch.global.L2 [%0];" ::"l"(po));
                asm volatile("prefetch.global.L2 [%0];" ::"l"(po + 64));
            }
        }
        gload(it + 1);
        tile(0, it % TPF == 0);
        sstore(1);
        __syncthreads();
        if (it + 2 < ntl) gload(it + 2);
        tile(1, false);
        if ((it + 2) % TPF == 0) fold((it + 2) / TPF - 1);
        if (it + 2 < ntl) sstore(0);
        __syncthreads();
    }

    float4 oin[8][2];
#pragma unroll
    for (int j = 0; j < 8; j++) {
        const int n = min(n0 + fcol(j), N - 1);
#pragma unroll
        for (int ih = 0; ih < 2; ih++)
            oin[j][ih] = *reinterpret_cast<const float4 *>(O_in + ((int64_t) z * N + n) * DV + m0 + ih * 64 + ty * 4);
    }
#pragma unroll
    for (int j = 0; j < 8; j++) {
        const int c = fcol(j), n = n0 + c;
        if (n >= N) continue;
        const float cr = corr[c];
#pragma unroll
        for (int ih = 0; ih < 2; ih++) {
            const float4 oi = oin[j][ih];
            float4 v;
            v.x = oi.x * cr + acc[ih * 4 + 0][j];
            v.y = oi.y * cr + acc[ih * 4 + 1][j];
            v.z = oi.z * cr + acc[ih * 4 + 2][j];
            v.w = oi.w * cr + acc[ih * 4 + 3][j];
            *reinterpret_cast<float4 *>(O_out + ((int64_t) z * N + n) * DV + m0 + ih * 64 + ty * 4) = v;
        }
    }
}

// out[t][hk*6 + hh][d] = sum_s w_s O_s[n][d] / sum_s w_s l_s[n], w_s = 2^(m_s - max m), n = hh * T + t. grid (T, Hq), 256.
__global__ void finalize_k(const float * __restrict__ O, const float * __restrict__ m, const float * __restrict__ l,
                           float * __restrict__ out, int T, int Hq, int N, int nsplit) {
    const int t = blockIdx.x, h = blockIdx.y, hk = h / GQA, hh = h % GQA, n = hh * T + t;
    float M = NEG_INF;
    for (int sp = 0; sp < nsplit; sp++) M = fmaxf(M, m[(int64_t) (hk * nsplit + sp) * N + n]);
    float L = 0.0f, acc = 0.0f;
    for (int sp = 0; sp < nsplit; sp++) {
        const int64_t o = (int64_t) (hk * nsplit + sp) * N + n;
        const float w = m[o] == NEG_INF ? 0.0f : exp2f(m[o] - M);
        L += w * l[o];
        acc += w * O[o * DH + threadIdx.x];
    }
    out[((size_t) t * Hq + h) * DH + threadIdx.x] = L > 0.0f ? acc / L : 0.0f;
}

__global__ void fill_k(float * p, float v, int64_t n) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x) p[i] = v;
}

inline size_t al(size_t v) { return (v + 255) & ~(size_t) 255; }

struct layout {
    size_t q16, k16, vp, p, mt, lt, o0, o1, m0, m1, l, total;
};
layout make_layout(int T, int Hkv, int C, int nsplit) {
    const size_t N = (size_t) GQA * T, nz = (size_t) Hkv * nsplit, nblk = (N + BN - 1) / BN;
    layout L{};
    size_t o = 0;
    auto take = [&](size_t bytes) { const size_t r = o; o += al(bytes); return r; };
    L.q16 = take((size_t) Hkv * N * DH * 2);
    L.k16 = take(nz * C * DH * 2);
    L.vp = take(nz * (C / 2) * DH * 4);
    L.p = take(nz * nblk * C * BN * 2);
    L.mt = take(nz * (C / TK) * N * 4);
    L.lt = take(nz * (C / TK) * N * 4);
    L.o0 = take(nz * N * DH * 4);
    L.o1 = take(nz * N * DH * 4);
    L.m0 = take(nz * N * 4);
    L.m1 = take(nz * N * 4);
    L.l = take(nz * N * 4);
    L.total = o;
    return L;
}

int sm_count() {
    static int cnt[16] = {};
    int dev = 0;
    cudaGetDevice(&dev);
    if (!cnt[dev & 15]) cudaDeviceGetAttribute(&cnt[dev & 15], cudaDevAttrMultiProcessorCount, dev);
    return cnt[dev & 15];
}

} // namespace

size_t attn_fold_scratch_bytes(int T, int Hkv, int chunk, int nsplit) {
    return make_layout(T, Hkv, std::max(TK, std::min(chunk, MAXTILE * TK)) / TK * TK, std::max(1, nsplit)).total;
}

bool attn_fold(const float * q, const kv_cache & c, float * out, int T, int p0, int Hq, float scale, void * scratch,
               size_t scratch_bytes, cudaStream_t s, int chunk, int nsplit) {
    if ((c.type != KV_F16 && c.type != KV_Q8_0 && c.type != KV_Q4_0) || c.D != DH || Hq != GQA * c.Hkv || T < 1 || !c.k || p0 < 0 ||
        p0 + T > c.n_ctx || !scratch)
        return false;
    const int Hkv = c.Hkv, N = GQA * T, nkv = p0 + T, nblk = (N + BN - 1) / BN;
    int ns = nsplit;
    if (ns <= 0) {   // PV grid (2 x nblk x Hkv x ns) of >= ~4 waves at one CTA per SM, splits of >= 2048 keys
        ns = (4 * sm_count() + 2 * nblk * Hkv - 1) / (2 * nblk * Hkv);
        ns = std::max(1, std::min({ns, 8, nkv / 2048}));
    }
    ns = std::min(ns, 8);
    const int Lsp = ((nkv + ns - 1) / ns + TK - 1) / TK * TK;
    int C = chunk > 0 ? chunk : MAXTILE * TK;
    C = std::min(C, MAXTILE * TK);
    C = (std::min(C, Lsp) + TK - 1) / TK * TK;
    layout L = make_layout(T, Hkv, C, ns);
    while (L.total > scratch_bytes && C > 4 * TK) {
        C = std::max(4 * TK, C / 2 / TK * TK);
        L = make_layout(T, Hkv, C, ns);
    }
    if (L.total > scratch_bytes) return false;
    char * b = static_cast<char *>(scratch);
    half * Q16 = reinterpret_cast<half *>(b + L.q16);
    half * K16 = reinterpret_cast<half *>(b + L.k16);
    half2 * Vp = reinterpret_cast<half2 *>(b + L.vp);
    half * P = reinterpret_cast<half *>(b + L.p);
    float * mt = reinterpret_cast<float *>(b + L.mt), * lt = reinterpret_cast<float *>(b + L.lt);
    float * O_cur = reinterpret_cast<float *>(b + L.o0), * O_nxt = reinterpret_cast<float *>(b + L.o1);
    float * m_cur = reinterpret_cast<float *>(b + L.m0), * m_nxt = reinterpret_cast<float *>(b + L.m1);
    float * l_st = reinterpret_cast<float *>(b + L.l);
    const int nz = Hkv * ns;
    const char * e1 = getenv("P100E_ATTN_OPT"), * e2 = getenv("P100E_ATTN_2S");
    const bool opt = !(e1 && e1[0] == '0'), two = !(e2 && e2[0] == '0');

    q16_k<<<dim3(T, Hq), DH / 4, 0, s>>>(q, Q16, T, Hq, scale * 0.25f);
    cudaMemsetAsync(O_cur, 0, (size_t) nz * N * DH * 4, s);
    cudaMemsetAsync(l_st, 0, (size_t) nz * N * 4, s);
    fill_k<<<256, 256, 0, s>>>(m_cur, -std::numeric_limits<float>::infinity(), (int64_t) nz * N);
    // z range [zb, ze) of every chunk runs on stream st; the two halves are independent (own K16/Vp/P/O/m/l rows)
    auto run = [&](const int zb, const int ze, cudaStream_t st) {
        float * Oc = O_cur, * On = O_nxt, * mc = m_cur, * mn = m_nxt;
        for (int c0 = 0; c0 < Lsp; c0 += C) {
            geo g{ns, Lsp, c0, nkv, p0, T, N, C, nblk, zb};
            const int ntile = (std::min(C, Lsp - c0) + TK - 1) / TK, nzr = ze - zb;
            const kv_pg pg = kv_pg_of(c);
            if (c.type == KV_F16)
                kv16_k<KV_F16><<<dim3(C / 64, nzr), 256, 0, st>>>((const char *) c.k, (const char *) c.v, K16, Vp, Hkv, g, pg);
            else if (c.type == KV_Q8_0)
                kv16_k<KV_Q8_0><<<dim3(C / 64, nzr), 256, 0, st>>>((const char *) c.k, (const char *) c.v, K16, Vp, Hkv, g, pg);
            else
                kv16_k<KV_Q4_0><<<dim3(C / 64, nzr), 256, 0, st>>>((const char *) c.k, (const char *) c.v, K16, Vp, Hkv, g, pg);
            if (opt) {
                qk_k<true><<<dim3(ntile, nblk, nzr), 256, 0, st>>>(K16, Q16, P, mt, lt, g);
                pv_k<true><<<dim3(DH / BM, nblk, nzr), 256, 0, st>>>(Vp, P, mt, lt, mc, mn, l_st, Oc, On, ntile, g);
            } else {
                qk_k<false><<<dim3(ntile, nblk, nzr), 256, 0, st>>>(K16, Q16, P, mt, lt, g);
                pv_k<false><<<dim3(DH / BM, nblk, nzr), 256, 0, st>>>(Vp, P, mt, lt, mc, mn, l_st, Oc, On, ntile, g);
            }
            std::swap(Oc, On);
            std::swap(mc, mn);
        }
        if (zb == 0) { O_cur = Oc; m_cur = mc; }   // chunk count is the same for both halves
    };
    static cudaStream_t s2[16] = {};
    static cudaEvent_t evf[16] = {}, evj[16] = {};
    int dev = 0;
    cudaGetDevice(&dev);
    dev &= 15;
    if (two && nz >= 2) {
        if (!s2[dev]) {
            cudaStreamCreateWithFlags(&s2[dev], cudaStreamNonBlocking);
            cudaEventCreateWithFlags(&evf[dev], cudaEventDisableTiming);
            cudaEventCreateWithFlags(&evj[dev], cudaEventDisableTiming);
        }
        const int h = nz / 2;
        cudaEventRecord(evf[dev], s);
        cudaStreamWaitEvent(s2[dev], evf[dev], 0);
        run(h, nz, s2[dev]);
        cudaEventRecord(evj[dev], s2[dev]);
        run(0, h, s);
        cudaStreamWaitEvent(s, evj[dev], 0);
    } else {
        run(0, nz, s);
    }
    finalize_k<<<dim3(T, Hq), DH, 0, s>>>(O_cur, m_cur, l_st, out, T, Hq, N, ns);
    return cudaPeekAtLastError() == cudaSuccess;
}

} // namespace p100e::fast
