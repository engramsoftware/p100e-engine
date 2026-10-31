// p100e fast: the warp-rows weight decoders (one 32-value group of one row per call), shared by matvec_wr.cu (T = 1
// kernels, wrc / wrcp) and matvec_wrm.cu (multi-column kernel with the dequant shared by the columns).
//
// Each decoder: ctor(lane) sets per-lane constants (g = t & 7 is lane & 7 for every group t a lane visits);
// load() issues the raw loads of one (row, group); dot() returns that group's fp32 contribution.
// Split form of dot() for several activation columns (matvec_wrm.cu): prep() = the column-independent scales of the
// group, chunk<K>() = the dequantized half2 weights of slots 4K..4K+3, fin() = the fp32 contribution from the two fp16
// partial sums. dot() == fin(se, so, prep(r), yd, ys) with se = the HFMA2 chain over the even slots 0, 2, .., 14 and
// so = the chain over the odd slots 1, 3, .., 15 (both from 0, in slot order): the same values, the same fp16 and fp32
// operations in the same order, so the same bits.
#pragma once
#include "matvec_internal.cuh"

namespace p100e::fast {

// weight load policy: 0 = ld.global.nc (__ldg), 1 = ld.global.cg (L2 only), 2 = ld.global.cs (streaming)
template <int LP, class V>
static __device__ __forceinline__ V wld(const V * p) {
    if constexpr (LP == 1) return __ldcg(p);
    else if constexpr (LP == 2) return __ldcs(p);
    else return __ldg(p);
}

static __device__ __forceinline__ uint32_t lop_lo(const uint32_t v) {   // (v & 0x000f000f) | 0x64006400
    uint32_t r;
    asm("lop3.b32 %0, %1, 0x000f000f, 0x64006400, 0xea;" : "=r"(r) : "r"(v));
    return r;
}
static __device__ __forceinline__ uint32_t lop_hi(const uint32_t v) {   // (v & 0x00f000f0) | 0x54005400
    uint32_t r;
    asm("lop3.b32 %0, %1, 0x00f000f0, 0x54005400, 0xea;" : "=r"(r) : "r"(v));
    return r;
}
static __device__ __forceinline__ half2 as_h2(const uint32_t v) { return *((const half2 *) &v); }

// ------------------------------------------------------------------------------------------------- decoders
// Each decoder: ctor(lane) sets per-lane constants (g = t & 7 is lane & 7 for every group t a lane visits);
// load() issues the raw loads of one (row, group); dot() returns that group's fp32 contribution.

// Q4_K, q4kr layout (see matvec_q4kr.cu)
template <int LP>
struct dec_q4kr {
    template <int L> using rebind = dec_q4kr<L>;   // the same decoder with another weight load policy
    uint32_t selV, selQ;
    bool ghi;
    struct raw { uint4 q, h; };
    static __device__ __forceinline__ void prefetch(const char * xr, const int t, const int nb) {
        asm volatile("prefetch.global.L2 [%0];" :: "l"(xr + 16*t));
        (void) nb;
    }
    __device__ __forceinline__ explicit dec_q4kr(const int lane) {
        const int g = lane & 7;
        selV = (g & 3)*0x1111u;
        selQ = selV + 0x4400u;
        ghi  = g >= 4;
    }
    // keep the per-lane constants in registers (matvec_wrm.cu: stops the compiler from re-deriving them every group)
    __device__ __forceinline__ void pin() {
        int h = ghi;
        asm volatile("" : "+r"(selV), "+r"(selQ), "+r"(h));
        ghi = h != 0;
    }
    __device__ __forceinline__ void load(raw & r, const char * xr, const int t, const int nb) const {
        r.q = wld<LP>((const uint4 *) (xr + 16*t));
        r.h = wld<LP>((const uint4 *) (xr + 128*nb + 16*(t >> 3)));
    }
    __device__ __forceinline__ float dot(const raw & r, const half2 (&a)[16], const float yd, const float ys) const {
        const half2 magic_lo = __float2half2_rn(1032.0f);
        const half2 magic_hi = __float2half2_rn(  72.0f);
        const half2 magic_sc = __float2half2_rn(1024.0f);
        const uint4 h = r.h;
        const uint32_t Q = __byte_perm(h.y, h.z, selQ);
        const uint32_t V = __byte_perm(h.w, h.w, selV);
        const uint32_t X = ghi ? (((Q >> 2) & 0x00300030u) | (V & 0x0000000Fu) | ((V >> 4) & 0x000F0000u) | 0x64006400u)
                               : ((Q & 0x003F003Fu) | 0x64006400u);
        const float2 scm = __half22float2(__hsub2(as_h2(X), magic_sc));
        const float2 dm  = __half22float2(as_h2(h.x));
        half2 se = make_half2(0.0f, 0.0f), so = make_half2(0.0f, 0.0f);
        const uint32_t qw[4] = {r.q.x, r.q.y, r.q.z, r.q.w};
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const half2 w0 = __hsub2(as_h2(lop_lo(qw[k])),      magic_lo);
            const half2 w1 = __hsub2(as_h2(lop_hi(qw[k])),      magic_hi);
            const half2 w2 = __hsub2(as_h2(lop_lo(qw[k] >> 8)), magic_lo);
            const half2 w3 = __hsub2(as_h2(lop_hi(qw[k] >> 8)), magic_hi);
            se = __hfma2(w0, a[4*k + 0], se);
            so = __hfma2(w1, a[4*k + 1], so);
            se = __hfma2(w2, a[4*k + 2], se);
            so = __hfma2(w3, a[4*k + 3], so);
        }
        const float scale_e = dm.x*scm.x;
        const float scale_o = scale_e;
        const float bias    = 8.0f*scale_e - dm.y*scm.y;
        const float2 fe = __half22float2(se);
        const float2 fo = __half22float2(so);
        return yd*(scale_e*(fe.x + fe.y) + scale_o*(fo.x + fo.y) + bias*ys);
    }
    // split form of dot() (see the top of this file)
    struct sc { float e, b; };
    __device__ __forceinline__ sc prep(const raw & r) const {
        const half2 magic_sc = __float2half2_rn(1024.0f);
        const uint4 h = r.h;
        const uint32_t Q = __byte_perm(h.y, h.z, selQ);
        const uint32_t V = __byte_perm(h.w, h.w, selV);
        const uint32_t X = ghi ? (((Q >> 2) & 0x00300030u) | (V & 0x0000000Fu) | ((V >> 4) & 0x000F0000u) | 0x64006400u)
                               : ((Q & 0x003F003Fu) | 0x64006400u);
        const float2 scm = __half22float2(__hsub2(as_h2(X), magic_sc));
        const float2 dm  = __half22float2(as_h2(h.x));
        sc s;
        s.e = dm.x*scm.x;
        s.b = 8.0f*s.e - dm.y*scm.y;
        return s;
    }
    template <int K>
    static __device__ __forceinline__ void chunk(const raw & r, half2 (&w)[4]) {
        const half2 magic_lo = __float2half2_rn(1032.0f);
        const half2 magic_hi = __float2half2_rn(  72.0f);
        const uint32_t q = K == 0 ? r.q.x : K == 1 ? r.q.y : K == 2 ? r.q.z : r.q.w;
        w[0] = __hsub2(as_h2(lop_lo(q)),      magic_lo);
        w[1] = __hsub2(as_h2(lop_hi(q)),      magic_hi);
        w[2] = __hsub2(as_h2(lop_lo(q >> 8)), magic_lo);
        w[3] = __hsub2(as_h2(lop_hi(q >> 8)), magic_hi);
    }
    static __device__ __forceinline__ float fin(const half2 se, const half2 so, const sc & s, const float yd, const float ys) {
        const float scale_e = s.e;
        const float scale_o = scale_e;
        const float2 fe = __half22float2(se);
        const float2 fo = __half22float2(so);
        return yd*(scale_e*(fe.x + fe.y) + scale_o*(fo.x + fo.y) + s.b*ys);
    }
};

// Q3_K, q3r layout (see matvec_q3r.cu)
template <int LP>
struct dec_q3r {
    template <int L> using rebind = dec_q3r<L>;
    int widx, lo_shift, hi_shift;
    struct raw { uint32_t A, B, H, wlo, w2; unsigned short d; };
    static __device__ __forceinline__ void prefetch(const char * xr, const int t, const int nb) {
        asm volatile("prefetch.global.L2 [%0];" :: "l"(xr + 12*t));
        (void) nb;
    }
    __device__ __forceinline__ explicit dec_q3r(const int lane) {
        const int g = lane & 7;
        widx     = (g & 3) >> 1;
        lo_shift = 16*(g & 1) + (g >= 4 ? 4 : 0);
        hi_shift = 16*(g & 1) + 2*(g >> 1);
    }
    __device__ __forceinline__ void pin() { asm volatile("" : "+r"(widx), "+r"(lo_shift), "+r"(hi_shift)); }
    __device__ __forceinline__ void load(raw & r, const char * xr, const int t, const int nb) const {
        const uint32_t * gw = (const uint32_t *) xr + 3*t;
        r.A = wld<LP>(gw); r.B = wld<LP>(gw + 1); r.H = wld<LP>(gw + 2);
        const uint32_t * sw = (const uint32_t *) (xr + 96*nb) + 3*(t >> 3);
        r.wlo = wld<LP>(sw + widx); r.w2 = wld<LP>(sw + 2);
        r.d = wld<LP>((const unsigned short *) (xr + 108*nb + 2*(t >> 3)));
    }
    __device__ __forceinline__ float dot(const raw & r, const half2 (&a)[16], const float yd, const float /*ys*/) const {
        const uint32_t M[4]   = {0x64006400u, 0x5C005C00u, 0x54005400u, 0x4C004C00u};
        const half2    sub[4] = {__float2half2_rn(1028.0f), __float2half2_rn(260.0f),
                                 __float2half2_rn(  68.0f), __float2half2_rn( 20.0f)};
        const uint16_t dbits = r.d;
        const float d = __half2float(*((const half *) &dbits));
        const int sc0 = (int) (((r.wlo >>  lo_shift)      & 0xF) | (((r.w2 >>  hi_shift)      & 3) << 4));
        const int sc1 = (int) (((r.wlo >> (lo_shift + 8)) & 0xF) | (((r.w2 >> (hi_shift + 8)) & 3) << 4));
        const float scale_e = d*(sc0 - 32);
        const float scale_o = d*(sc1 - 32);
        const uint32_t V[4] = {r.H, r.H >> 1, __funnelshift_r(r.H, r.H, 8), __funnelshift_r(r.H, r.H, 9)};
        half2 se = make_half2(0.0f, 0.0f), so = make_half2(0.0f, 0.0f);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint32_t W = (k < 2 ? r.A : r.B) >> (8*(k & 1));
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const uint32_t x  = (W    & (0x00030003u << (2*q))) | M[q];
                const uint32_t qq = x | (V[k] & (0x00040004u << (2*q)));
                const half2 w = __hsub2(as_h2(qq), sub[q]);
                if (q & 1) so = __hfma2(w, a[4*k + q], so);
                else       se = __hfma2(w, a[4*k + q], se);
            }
        }
        const float2 fe = __half22float2(se);
        const float2 fo = __half22float2(so);
        return yd*(scale_e*(fe.x + fe.y) + scale_o*(fo.x + fo.y));
    }
    // split form of dot() (see the top of this file)
    struct sc { float e, o; };
    __device__ __forceinline__ sc prep(const raw & r) const {
        const uint16_t dbits = r.d;
        const float d = __half2float(*((const half *) &dbits));
        const int sc0 = (int) (((r.wlo >>  lo_shift)      & 0xF) | (((r.w2 >>  hi_shift)      & 3) << 4));
        const int sc1 = (int) (((r.wlo >> (lo_shift + 8)) & 0xF) | (((r.w2 >> (hi_shift + 8)) & 3) << 4));
        sc s;
        s.e = d*(sc0 - 32);
        s.o = d*(sc1 - 32);
        return s;
    }
    template <int K>
    static __device__ __forceinline__ void chunk(const raw & r, half2 (&w)[4]) {
        const uint32_t M[4]   = {0x64006400u, 0x5C005C00u, 0x54005400u, 0x4C004C00u};
        const half2    sub[4] = {__float2half2_rn(1028.0f), __float2half2_rn(260.0f),
                                 __float2half2_rn(  68.0f), __float2half2_rn( 20.0f)};
        const uint32_t V = K == 0 ? r.H : K == 1 ? r.H >> 1 : K == 2 ? __funnelshift_r(r.H, r.H, 8) : __funnelshift_r(r.H, r.H, 9);
        const uint32_t W = (K < 2 ? r.A : r.B) >> (8*(K & 1));
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const uint32_t x  = (W & (0x00030003u << (2*q))) | M[q];
            const uint32_t qq = x | (V & (0x00040004u << (2*q)));
            w[q] = __hsub2(as_h2(qq), sub[q]);
        }
    }
    static __device__ __forceinline__ float fin(const half2 se, const half2 so, const sc & s, const float yd, const float /*ys*/) {
        const float2 fe = __half22float2(se);
        const float2 fo = __half22float2(so);
        return yd*(s.e*(fe.x + fe.y) + s.o*(fo.x + fo.y));
    }
};

// Q6_K, q6r2 layout (see kq_decode<Q6_K, VA, true> in matvec_kq.cu).  VA = alignment of the 16-byte ql2 record
// (row stride 210*nb is only 8-aligned when nb % 8 == 4).
template <int LP, int VA>
struct dec_q6r2 {
    template <int L> using rebind = dec_q6r2<L, VA>;
    struct raw { uint32_t q[4]; uint32_t h[2]; unsigned short sc; unsigned short d; };
    static __device__ __forceinline__ void prefetch(const char * xr, const int t, const int nb) {
        asm volatile("prefetch.global.L2 [%0];" :: "l"(xr + 16*t));
        (void) nb;
    }
    __device__ __forceinline__ explicit dec_q6r2(const int) {}
    __device__ __forceinline__ void pin() {}
    __device__ __forceinline__ void load(raw & r, const char * xr, const int t, const int nb) const {
        const char * qp = xr + 16*t;
        if constexpr (VA == 16) {
            const uint4 v = wld<LP>((const uint4 *) qp);
            r.q[0] = v.x; r.q[1] = v.y; r.q[2] = v.z; r.q[3] = v.w;
        } else {
            const uint2 v0 = wld<LP>((const uint2 *) qp), v1 = wld<LP>((const uint2 *) qp + 1);
            r.q[0] = v0.x; r.q[1] = v0.y; r.q[2] = v1.x; r.q[3] = v1.y;
        }
        const uint2 hv = wld<LP>((const uint2 *) (xr + 128*nb + 8*t));
        r.h[0] = hv.x; r.h[1] = hv.y;
        r.sc = wld<LP>((const unsigned short *) (xr + 192*nb + 2*t));   // = 192nb + 16sb + 8n + 2(g & 3)
        r.d  = wld<LP>((const unsigned short *) (xr + 208*nb + 2*(t >> 3)));
    }
    __device__ __forceinline__ float dot(const raw & r, const half2 (&a)[16], const float yd, const float /*ys*/) const {
        const half2 magic = __float2half2_rn(1056.0f);
        half2 w[16];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const uint32_t vi = i < 4 ? r.q[i] : r.q[i - 4] >> 4;
            const uint32_t hb = (r.h[i >> 2] >> (8*(i & 3))) & 0xFFu;
            const uint32_t t0 = lop_lo(vi)      | ((hb & 3u) << 4)        | (((hb >> 4) & 3u) << 20);
            const uint32_t t1 = lop_lo(vi >> 8) | (((hb >> 2) & 3u) << 4) | (((hb >> 6) & 3u) << 20);
            const int ki = i < 4 ? 4*i : 4*(i - 4) + 1;
            w[ki    ] = __hsub2(as_h2(t0), magic);
            w[ki + 2] = __hsub2(as_h2(t1), magic);
        }
        half2 se = make_half2(0.0f, 0.0f), so = make_half2(0.0f, 0.0f);
#pragma unroll
        for (int c = 0; c < 16; c += 2) {
            se = __hfma2(w[c    ], a[c    ], se);
            so = __hfma2(w[c + 1], a[c + 1], so);
        }
        const uint16_t dbits = r.d;
        const float d = __half2float(*((const half *) &dbits));
        const float scale_e = d*(float) (int8_t) (r.sc & 0xFF);
        const float scale_o = d*(float) (int8_t) (r.sc >> 8);
        const float2 fe = __half22float2(se);
        const float2 fo = __half22float2(so);
        return yd*(scale_e*(fe.x + fe.y) + scale_o*(fo.x + fo.y));
    }
    // split form of dot() (see the top of this file): chunk K = dot()'s i = K (slots 4K, 4K + 2) and i = K + 4
    // (slots 4K + 1, 4K + 3)
    struct sc { float e, o; };
    __device__ __forceinline__ sc prep(const raw & r) const {
        const uint16_t dbits = r.d;
        const float d = __half2float(*((const half *) &dbits));
        sc s;
        s.e = d*(float) (int8_t) (r.sc & 0xFF);
        s.o = d*(float) (int8_t) (r.sc >> 8);
        return s;
    }
    template <int K>
    static __device__ __forceinline__ void chunk(const raw & r, half2 (&w)[4]) {
        const half2 magic = __float2half2_rn(1056.0f);
        {
            const uint32_t vi = r.q[K];
            const uint32_t hb = (r.h[0] >> (8*K)) & 0xFFu;
            const uint32_t t0 = lop_lo(vi)      | ((hb & 3u) << 4)        | (((hb >> 4) & 3u) << 20);
            const uint32_t t1 = lop_lo(vi >> 8) | (((hb >> 2) & 3u) << 4) | (((hb >> 6) & 3u) << 20);
            w[0] = __hsub2(as_h2(t0), magic);
            w[2] = __hsub2(as_h2(t1), magic);
        }
        {
            const uint32_t vi = r.q[K] >> 4;
            const uint32_t hb = (r.h[1] >> (8*K)) & 0xFFu;
            const uint32_t t0 = lop_lo(vi)      | ((hb & 3u) << 4)        | (((hb >> 4) & 3u) << 20);
            const uint32_t t1 = lop_lo(vi >> 8) | (((hb >> 2) & 3u) << 4) | (((hb >> 6) & 3u) << 20);
            w[1] = __hsub2(as_h2(t0), magic);
            w[3] = __hsub2(as_h2(t1), magic);
        }
    }
    static __device__ __forceinline__ float fin(const half2 se, const half2 so, const sc & s, const float yd, const float /*ys*/) {
        const float2 fe = __half22float2(se);
        const float2 fo = __half22float2(so);
        return yd*(s.e*(fe.x + fe.y) + s.o*(fo.x + fo.y));
    }
};

// Q6_K, q6r3 layout (matvec_pack_body.inc): q6r2 planes, qh record = (HA, HB) re-ordered so every half2 needs one
// shift and one LOP3.  Same weights, slots and fp16 order as dec_q6r2 (so as kq_decode<Q6_K, r2>): bit-identical.
template <int SH>
static __device__ __forceinline__ uint32_t shl_s(const uint32_t v) {   // v << SH, negative = right shift
    if constexpr (SH >= 0) return v << SH; else return v >> (-SH);
}
static __device__ __forceinline__ uint32_t or_and(const uint32_t a, const uint32_t b, const uint32_t m) {   // a | (b & m)
    uint32_t r;
    asm("lop3.b32 %0, %1, %2, %3, 0xf8;" : "=r"(r) : "r"(a), "r"(b), "r"(m));
    return r;
}

template <int LP, int VA>
struct dec_q6r3 {
    template <int L> using rebind = dec_q6r3<L, VA>;
    struct raw { uint32_t q[4]; uint32_t ha, hb; unsigned short sc; unsigned short d; };
    static __device__ __forceinline__ void prefetch(const char * xr, const int t, const int nb) {
        asm volatile("prefetch.global.L2 [%0];" :: "l"(xr + 16*t));
        (void) nb;
    }
    __device__ __forceinline__ explicit dec_q6r3(const int) {}
    __device__ __forceinline__ void pin() {}
    __device__ __forceinline__ void load(raw & r, const char * xr, const int t, const int nb) const {
        const char * qp = xr + 16*t;
        if constexpr (VA == 16) {
            const uint4 v = wld<LP>((const uint4 *) qp);
            r.q[0] = v.x; r.q[1] = v.y; r.q[2] = v.z; r.q[3] = v.w;
        } else {
            const uint2 v0 = wld<LP>((const uint2 *) qp), v1 = wld<LP>((const uint2 *) qp + 1);
            r.q[0] = v0.x; r.q[1] = v0.y; r.q[2] = v1.x; r.q[3] = v1.y;
        }
        const uint2 hv = wld<LP>((const uint2 *) (xr + 128*nb + 8*t));
        r.ha = hv.x; r.hb = hv.y;
        r.sc = wld<LP>((const unsigned short *) (xr + 192*nb + 2*t));
        r.d  = wld<LP>((const unsigned short *) (xr + 208*nb + 2*(t >> 3)));
    }
    template <int K>
    static __device__ __forceinline__ void word(const uint32_t w32, const uint32_t ha, const uint32_t hb,
                                                const half2 (&a)[16], half2 & se, half2 & so) {
        const half2 magic_lo = __float2half2_rn(1056.0f);   // 1024 + 32
        const half2 magic_hi = __float2half2_rn(  96.0f);   //   64 + 32
        const uint32_t w32s = w32 >> 8;
        const uint32_t t0 = or_and(lop_lo(w32),  shl_s<4 - 4*K>(ha), 0x00300030u);
        const uint32_t t2 = or_and(lop_lo(w32s), shl_s<2 - 4*K>(ha), 0x00300030u);
        const uint32_t t1 = or_and(lop_hi(w32),  shl_s<8 - 4*K>(hb), 0x03000300u);
        const uint32_t t3 = or_and(lop_hi(w32s), shl_s<6 - 4*K>(hb), 0x03000300u);
        se = __hfma2(__hsub2(as_h2(t0), magic_lo), a[4*K + 0], se);
        so = __hfma2(__hsub2(as_h2(t1), magic_hi), a[4*K + 1], so);
        se = __hfma2(__hsub2(as_h2(t2), magic_lo), a[4*K + 2], se);
        so = __hfma2(__hsub2(as_h2(t3), magic_hi), a[4*K + 3], so);
    }
    __device__ __forceinline__ float dot(const raw & r, const half2 (&a)[16], const float yd, const float /*ys*/) const {
        half2 se = make_half2(0.0f, 0.0f), so = make_half2(0.0f, 0.0f);
        word<0>(r.q[0], r.ha, r.hb, a, se, so);
        word<1>(r.q[1], r.ha, r.hb, a, se, so);
        word<2>(r.q[2], r.ha, r.hb, a, se, so);
        word<3>(r.q[3], r.ha, r.hb, a, se, so);
        const uint16_t dbits = r.d;
        const float d = __half2float(*((const half *) &dbits));
        const float scale_e = d*(float) (int8_t) (r.sc & 0xFF);
        const float scale_o = d*(float) (int8_t) (r.sc >> 8);
        const float2 fe = __half22float2(se);
        const float2 fo = __half22float2(so);
        return yd*(scale_e*(fe.x + fe.y) + scale_o*(fo.x + fo.y));
    }
    // split form of dot() (see the top of this file): chunk K = word<K>'s four weights
    struct sc { float e, o; };
    __device__ __forceinline__ sc prep(const raw & r) const {
        const uint16_t dbits = r.d;
        const float d = __half2float(*((const half *) &dbits));
        sc s;
        s.e = d*(float) (int8_t) (r.sc & 0xFF);
        s.o = d*(float) (int8_t) (r.sc >> 8);
        return s;
    }
    template <int K>
    static __device__ __forceinline__ void chunk(const raw & r, half2 (&w)[4]) {
        const half2 magic_lo = __float2half2_rn(1056.0f);   // 1024 + 32
        const half2 magic_hi = __float2half2_rn(  96.0f);   //   64 + 32
        const uint32_t w32  = r.q[K];
        const uint32_t w32s = w32 >> 8;
        const uint32_t t0 = or_and(lop_lo(w32),  shl_s<4 - 4*K>(r.ha), 0x00300030u);
        const uint32_t t2 = or_and(lop_lo(w32s), shl_s<2 - 4*K>(r.ha), 0x00300030u);
        const uint32_t t1 = or_and(lop_hi(w32),  shl_s<8 - 4*K>(r.hb), 0x03000300u);
        const uint32_t t3 = or_and(lop_hi(w32s), shl_s<6 - 4*K>(r.hb), 0x03000300u);
        w[0] = __hsub2(as_h2(t0), magic_lo);
        w[1] = __hsub2(as_h2(t1), magic_hi);
        w[2] = __hsub2(as_h2(t2), magic_lo);
        w[3] = __hsub2(as_h2(t3), magic_hi);
    }
    static __device__ __forceinline__ float fin(const half2 se, const half2 so, const sc & s, const float yd, const float /*ys*/) {
        const float2 fe = __half22float2(se);
        const float2 fo = __half22float2(so);
        return yd*(s.e*(fe.x + fe.y) + s.o*(fo.x + fo.y));
    }
};

// Q5_K, q5kr layout (matvec_pack_body.inc): q4kr-style qs plane + one u32 of high bits per group + header plane.
// Same weights, slots and fp16 order as kq_decode<Q5_K> + mul_mat_vec_kq: bit-identical per group.
template <int LP>
struct dec_q5kr {
    template <int L> using rebind = dec_q5kr<L>;
    uint32_t selV, selQ;
    bool ghi;
    struct raw { uint4 q; uint32_t h; uint4 hd; };
    static __device__ __forceinline__ void prefetch(const char * xr, const int t, const int nb) {
        asm volatile("prefetch.global.L2 [%0];" :: "l"(xr + 16*t));
        (void) nb;
    }
    __device__ __forceinline__ explicit dec_q5kr(const int lane) {
        const int g = lane & 7;
        selV = (g & 3)*0x1111u;
        selQ = selV + 0x4400u;
        ghi  = g >= 4;
    }
    // keep the per-lane constants in registers (matvec_wrm.cu: stops the compiler from re-deriving them every group)
    __device__ __forceinline__ void pin() {
        int h = ghi;
        asm volatile("" : "+r"(selV), "+r"(selQ), "+r"(h));
        ghi = h != 0;
    }
    __device__ __forceinline__ void load(raw & r, const char * xr, const int t, const int nb) const {
        r.q  = wld<LP>((const uint4 *) (xr + 16*t));
        r.h  = wld<LP>((const uint32_t *) (xr + 128*nb + 4*t));
        r.hd = wld<LP>((const uint4 *) (xr + 160*nb + 16*(t >> 3)));
    }
    template <int K>
    static __device__ __forceinline__ void word(const uint32_t w32, const uint32_t H, const half2 (&a)[16],
                                                half2 & se, half2 & so) {
        const half2 magic_lo = __float2half2_rn(1040.0f);   // 1024 + 16
        const half2 magic_hi = __float2half2_rn(  80.0f);   //   64 + 16
        const uint32_t w32s = w32 >> 8;
        const uint32_t s1 = H >> K;
        const uint32_t t0 = or_and(lop_lo(w32),  H << (4 - K), 0x00100010u);
        const uint32_t t2 = or_and(lop_lo(w32s), s1,           0x00100010u);
        const uint32_t t1 = or_and(lop_hi(w32),  s1,           0x01000100u);
        const uint32_t t3 = or_and(lop_hi(w32s), s1 >> 4,      0x01000100u);
        se = __hfma2(__hsub2(as_h2(t0), magic_lo), a[4*K + 0], se);
        so = __hfma2(__hsub2(as_h2(t1), magic_hi), a[4*K + 1], so);
        se = __hfma2(__hsub2(as_h2(t2), magic_lo), a[4*K + 2], se);
        so = __hfma2(__hsub2(as_h2(t3), magic_hi), a[4*K + 3], so);
    }
    __device__ __forceinline__ float dot(const raw & r, const half2 (&a)[16], const float yd, const float ys) const {
        const half2 magic_sc = __float2half2_rn(1024.0f);
        const uint4 h = r.hd;
        const uint32_t Q = __byte_perm(h.y, h.z, selQ);
        const uint32_t V = __byte_perm(h.w, h.w, selV);
        const uint32_t X = ghi ? (((Q >> 2) & 0x00300030u) | (V & 0x0000000Fu) | ((V >> 4) & 0x000F0000u) | 0x64006400u)
                               : ((Q & 0x003F003Fu) | 0x64006400u);
        const float2 scm = __half22float2(__hsub2(as_h2(X), magic_sc));
        const float2 dm  = __half22float2(as_h2(h.x));
        half2 se = make_half2(0.0f, 0.0f), so = make_half2(0.0f, 0.0f);
        word<0>(r.q.x, r.h, a, se, so);
        word<1>(r.q.y, r.h, a, se, so);
        word<2>(r.q.z, r.h, a, se, so);
        word<3>(r.q.w, r.h, a, se, so);
        const float scale_e = dm.x*scm.x;
        const float scale_o = scale_e;
        const float bias    = 16.0f*scale_e - dm.y*scm.y;
        const float2 fe = __half22float2(se);
        const float2 fo = __half22float2(so);
        return yd*(scale_e*(fe.x + fe.y) + scale_o*(fo.x + fo.y) + bias*ys);
    }
    // split form of dot() (see the top of this file): chunk K = word<K>'s four weights
    struct sc { float e, b; };
    __device__ __forceinline__ sc prep(const raw & r) const {
        const half2 magic_sc = __float2half2_rn(1024.0f);
        const uint4 h = r.hd;
        const uint32_t Q = __byte_perm(h.y, h.z, selQ);
        const uint32_t V = __byte_perm(h.w, h.w, selV);
        const uint32_t X = ghi ? (((Q >> 2) & 0x00300030u) | (V & 0x0000000Fu) | ((V >> 4) & 0x000F0000u) | 0x64006400u)
                               : ((Q & 0x003F003Fu) | 0x64006400u);
        const float2 scm = __half22float2(__hsub2(as_h2(X), magic_sc));
        const float2 dm  = __half22float2(as_h2(h.x));
        sc s;
        s.e = dm.x*scm.x;
        s.b = 16.0f*s.e - dm.y*scm.y;
        return s;
    }
    template <int K>
    static __device__ __forceinline__ void chunk(const raw & r, half2 (&w)[4]) {
        const half2 magic_lo = __float2half2_rn(1040.0f);   // 1024 + 16
        const half2 magic_hi = __float2half2_rn(  80.0f);   //   64 + 16
        const uint32_t w32  = K == 0 ? r.q.x : K == 1 ? r.q.y : K == 2 ? r.q.z : r.q.w;
        const uint32_t H    = r.h;
        const uint32_t w32s = w32 >> 8;
        const uint32_t s1 = H >> K;
        const uint32_t t0 = or_and(lop_lo(w32),  H << (4 - K), 0x00100010u);
        const uint32_t t2 = or_and(lop_lo(w32s), s1,           0x00100010u);
        const uint32_t t1 = or_and(lop_hi(w32),  s1,           0x01000100u);
        const uint32_t t3 = or_and(lop_hi(w32s), s1 >> 4,      0x01000100u);
        w[0] = __hsub2(as_h2(t0), magic_lo);
        w[1] = __hsub2(as_h2(t1), magic_hi);
        w[2] = __hsub2(as_h2(t2), magic_lo);
        w[3] = __hsub2(as_h2(t3), magic_hi);
    }
    static __device__ __forceinline__ float fin(const half2 se, const half2 so, const sc & s, const float yd, const float ys) {
        const float scale_e = s.e;
        const float scale_o = scale_e;
        const float2 fe = __half22float2(se);
        const float2 fo = __half22float2(so);
        return yd*(scale_e*(fe.x + fe.y) + scale_o*(fo.x + fo.y) + s.b*ys);
    }
};

// ------------------------------------------------------------------------------------------------- fused quantize
// The a16 activation of one 32-value group, computed from f32 x by ONE thread with exactly the arithmetic of
// quantize_a16 (matvec_a16.cu, G = 8 lanes of E = 4 elements): the same amax, 1/amax, products, the same per-lane
// partial sums combined in the same butterfly order, the same f16 roundings.  So the slots and (amax, sum) are
// bit-identical to what the separate quantize kernel writes, and the GEMV results equal the unfused ones.
static __device__ __forceinline__ void a16_group(const float * __restrict__ xg, uint4 (&q)[4], float2 & ds) {
    float v[32];
    const float4 * xp = (const float4 *) xg;
#pragma unroll
    for (int c = 0; c < 8; ++c) {
        const float4 f = __ldg(xp + c);
        v[4*c + 0] = f.x; v[4*c + 1] = f.y; v[4*c + 2] = f.z; v[4*c + 3] = f.w;
    }
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(v[i]));
    amax = amax < 65520.0f ? amax : 65504.0f;   // f16 scale overflow clamp, as quantize_a16
    const float id = amax > 0.0f ? 1.0f/amax : 0.0f;
    // __fmul_rn / __fadd_rn: no FMA contraction of the scale into the sum, like a16_quant8 (a16q.cuh) and the
    // unfused quantize_a16 (whose separate multiply and add are not contracted) - the bit identity does not hinge on
    // what the compiler decides for this kernel
    float sl[8];
#pragma unroll
    for (int l = 0; l < 8; ++l) {
        float sum = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            v[4*l + i] = __fmul_rn(v[4*l + i], id);
            sum = __fadd_rn(sum, v[4*l + i]);
        }
        sl[l] = sum;
    }
    const float sum = __fadd_rn(__fadd_rn(__fadd_rn(sl[0], sl[4]), __fadd_rn(sl[2], sl[6])),
                                __fadd_rn(__fadd_rn(sl[1], sl[5]), __fadd_rn(sl[3], sl[7])));
    ds = __half22float2(make_half2(__float2half(amax), __float2half(sum)));
    // slot 4k+0 = (v[4k], v[4k+2]), 4k+1 = (v[16+4k], v[18+4k]), 4k+2 = (v[4k+1], v[4k+3]), 4k+3 = (v[17+4k], v[19+4k])
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const half2 s0 = make_half2(__float2half(v[4*k + 0]),  __float2half(v[4*k + 2]));
        const half2 s1 = make_half2(__float2half(v[16 + 4*k]), __float2half(v[18 + 4*k]));
        const half2 s2 = make_half2(__float2half(v[4*k + 1]),  __float2half(v[4*k + 3]));
        const half2 s3 = make_half2(__float2half(v[17 + 4*k]), __float2half(v[19 + 4*k]));
        q[k] = make_uint4(*((const uint32_t *) &s0), *((const uint32_t *) &s1), *((const uint32_t *) &s2), *((const uint32_t *) &s3));
    }
}

} // namespace p100e::fast
