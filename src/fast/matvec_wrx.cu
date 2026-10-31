// p100e fast: mv_wrpx, the persistent warp-rows T = 1 GEMV (mv_wrp / wrq in matvec_wr.cu) with its inner loop written
// as inline PTX: explicit ld.global.nc loads from per-row pointers that are bumped by a constant each step (the C++
// kernel re-derives every address from row*row_bytes, ~25% of its loop on the P100), and the unpack / dequant / half2
// FMA chains as hand-written lop3 / prmt / bfe / sub.rn.f16x2 / fma.rn.f16x2 sequences.
//
// Bit identity: every decoder computes the same fp16 values with the same fp16 ops in the same order as the C++
// decoders in matvec_dec.cuh, and the same fp32 expression with the contractions nvcc chose for them (read from the
// sm_60 PTX/SASS of mv_wrp: scale = d*sc, inner = fma(scale_e, se.x+se.y, scale_o*(so.x+so.y)) [+ fma(ys, bias, .)],
// acc = fma(yd, inner, acc)); the per-row group order is mv_wrp's.  So rows equal mv_wrp's bit for bit.
//
// Switches (one per type, default off): P100E_MVWRP_Q3 / _Q4 / _Q5 / _Q6 = 1 (one group per lane per step) or 2 (two
// groups per lane per step: twice the loads in flight, half the pointer bumps).  Geometries: the default T = 1 choices
// (nw8 rw4, nw12 rw4, nw8 rw2, nw16 rw2); any other variant runs the C++ kernel.
#include "matvec.cuh"
#include "matvec_internal.cuh"
#include "matvec_dec.cuh"
#include <algorithm>
#include <cstdio>
#include <cstdlib>

namespace p100e::fast {

// ------------------------------------------------------------------------------------------------- PTX helpers
template <int OFF>
static __device__ __forceinline__ uint4 ldx4(const char * p) {
    uint4 r;
    asm volatile("ld.global.nc.v4.u32 {%0, %1, %2, %3}, [%4+%5];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p), "n"(OFF));
    return r;
}
template <int OFF>
static __device__ __forceinline__ uint2 ldx2(const char * p) {
    uint2 r;
    asm volatile("ld.global.nc.v2.u32 {%0, %1}, [%2+%3];" : "=r"(r.x), "=r"(r.y) : "l"(p), "n"(OFF));
    return r;
}
template <int OFF>
static __device__ __forceinline__ uint32_t ldx1(const char * p) {
    uint32_t r;
    asm volatile("ld.global.nc.u32 %0, [%1+%2];" : "=r"(r) : "l"(p), "n"(OFF));
    return r;
}
template <int OFF>
static __device__ __forceinline__ uint32_t ldxh(const char * p) {   // u16, zero-extended
    uint32_t r;
    asm volatile("ld.global.nc.u16 %0, [%1+%2];" : "=r"(r) : "l"(p), "n"(OFF));
    return r;
}
static __device__ __forceinline__ uint32_t h2u(const half2 v) { return *((const uint32_t *) &v); }
// p += N in PTX: the add is opaque to NVVM, so the per-row pointers stay loop-carried registers (left to itself, its
// loop strength reduction re-derives them from row*row_bytes every step, the very cost this kernel removes)
template <int N>
static __device__ __forceinline__ void bump(const char * & p) { asm("add.s64 %0, %0, %1;" : "+l"(p) : "n"(N)); }

// fp32 tail shared by all types: acc = fma(yd, fma(sce, se.x + se.y, sco*(so.x + so.y)), acc)
static __device__ __forceinline__ float tail2(const uint32_t se, const uint32_t so, const float sce, const float sco,
                                             const float yd, const float acc) {
    float r;
    asm("{\n\t.reg .f16 a, b, c, d;\n\t.reg .f32 fa, fb, fc, fd, fe, fo, t;\n\t"
        "mov.b32 {a, b}, %1;\n\tmov.b32 {c, d}, %2;\n\t"
        "cvt.f32.f16 fa, a;\n\tcvt.f32.f16 fb, b;\n\tcvt.f32.f16 fc, c;\n\tcvt.f32.f16 fd, d;\n\t"
        "add.rn.f32 fe, fa, fb;\n\tadd.rn.f32 fo, fc, fd;\n\t"
        "mul.rn.f32 t, %4, fo;\n\tfma.rn.f32 t, %3, fe, t;\n\t"
        "fma.rn.f32 %0, %5, t, %6;\n\t}"
        : "=f"(r) : "r"(se), "r"(so), "f"(sce), "f"(sco), "f"(yd), "f"(acc));
    return r;
}
// the same with the q4_K / q5_K min term: acc = fma(yd, fma(ys, bias, fma(s, se.x + se.y, s*(so.x + so.y))), acc)
static __device__ __forceinline__ float tail2b(const uint32_t se, const uint32_t so, const float s, const float bias,
                                              const float yd, const float ys, const float acc) {
    float r;
    asm("{\n\t.reg .f16 a, b, c, d;\n\t.reg .f32 fa, fb, fc, fd, fe, fo, t;\n\t"
        "mov.b32 {a, b}, %1;\n\tmov.b32 {c, d}, %2;\n\t"
        "cvt.f32.f16 fa, a;\n\tcvt.f32.f16 fb, b;\n\tcvt.f32.f16 fc, c;\n\tcvt.f32.f16 fd, d;\n\t"
        "add.rn.f32 fe, fa, fb;\n\tadd.rn.f32 fo, fc, fd;\n\t"
        "mul.rn.f32 t, %3, fo;\n\tfma.rn.f32 t, %3, fe, t;\n\t"
        "fma.rn.f32 t, %6, %4, t;\n\t"
        "fma.rn.f32 %0, %5, t, %7;\n\t}"
        : "=f"(r) : "r"(se), "r"(so), "f"(s), "f"(bias), "f"(yd), "f"(ys), "f"(acc));
    return r;
}

// q4_K / q5_K header (h.x = (d, dmin) f16, h.y..h.w = the 12 packed 6-bit scale/min bytes): s = d*sc, bias = M*s - dmin*m
// (M = 8 for q4_K, 16 for q5_K).  The per-lane constants turn dec_q4kr's ghi branch into one branch-free sequence:
// X = ((Q >> sh) & mQ) | 0x64006400 | (V & m1) | ((V >> 4) & m2).
struct kq_hdr_c { uint32_t selQ, selV, sh, mQ, m1, m2; };
static __device__ __forceinline__ kq_hdr_c kq_hdr_init(const int lane) {
    const int g = lane & 7;
    const bool hi = g >= 4;
    kq_hdr_c c;
    c.selV = (g & 3)*0x1111u;
    c.selQ = c.selV + 0x4400u;
    c.sh = hi ? 2 : 0;
    c.mQ = hi ? 0x00300030u : 0x003F003Fu;
    c.m1 = hi ? 0x0000000Fu : 0u;
    c.m2 = hi ? 0x000F0000u : 0u;
    return c;
}
template <int MUL>   // 0x41000000 = 8.0f, 0x41800000 = 16.0f
static __device__ __forceinline__ void kq_hdr(const uint4 h, const kq_hdr_c & c, float & s, float & bias) {
    asm("{\n\t.reg .b32 q, v, x, y;\n\t.reg .f16 sl, sh2, dl, dh;\n\t.reg .f32 fsl, fsh, fdl, fdh, m, s8;\n\t"
        "prmt.b32 q, %3, %4, %7;\n\t"
        "prmt.b32 v, %5, %5, %8;\n\t"
        "shr.b32 x, q, %9;\n\t"
        "lop3.b32 x, x, %10, 0x64006400, 0xea;\n\t"
        "lop3.b32 x, v, %11, x, 0xea;\n\t"
        "shr.b32 y, v, 4;\n\t"
        "lop3.b32 x, y, %12, x, 0xea;\n\t"
        "mov.b32 y, 0x64006400;\n\t"
        "sub.rn.f16x2 x, x, y;\n\t"
        "mov.b32 {sl, sh2}, x;\n\tmov.b32 {dl, dh}, %2;\n\t"
        "cvt.f32.f16 fsl, sl;\n\tcvt.f32.f16 fsh, sh2;\n\tcvt.f32.f16 fdl, dl;\n\tcvt.f32.f16 fdh, dh;\n\t"
        "mul.rn.f32 %0, fdl, fsl;\n\t"
        "mul.rn.f32 m, fdh, fsh;\n\t"
        "mul.rn.f32 s8, %0, %13;\n\t"
        "sub.rn.f32 %1, s8, m;\n\t}"
        : "=f"(s), "=f"(bias)
        : "r"(h.x), "r"(h.y), "r"(h.z), "r"(h.w), "r"(0u), "r"(c.selQ), "r"(c.selV), "r"(c.sh), "r"(c.mQ), "r"(c.m1),
          "r"(c.m2), "f"(__int_as_float(MUL)));
}

// ------------------------------------------------------------------------------------------------- decoders
// Interface: ctor(lane); init(p, row base, lane, nb); load<J>(r, p) = group t + 32*J of the current step; adv<N>(p)
// = move the pointers N groups-of-32 on; acc = dot(r, a, yd, ys, acc).

// Q4_K, q4kr layout: qs plane (16 B per group) | header plane (16 B per 8 groups)
struct xq4 {
    static constexpr int kStage = 0;   // bytes of shared-memory staging per row (0 = none)
    __device__ __forceinline__ void set_stage(uint32_t) {}
    kq_hdr_c c;
    struct ptr { const char * q; const char * h; };
    struct raw { uint4 q, h; };
    static __device__ __forceinline__ void prefetch(const char * xr, const int t, const int nb) { dec_q4kr<0>::prefetch(xr, t, nb); }
    __device__ __forceinline__ explicit xq4(const int lane) : c(kq_hdr_init(lane)) {}
    __device__ __forceinline__ void init(ptr & p, const char * xr, const int lane, const int nb) const {
        p.q = xr + 16*lane; p.h = xr + 128*nb + 16*(lane >> 3);
    }
    template <int J> static __device__ __forceinline__ void load(raw & r, const ptr & p) {
        r.q = ldx4<512*J>(p.q); r.h = ldx4<64*J>(p.h);
    }
    template <int N> static __device__ __forceinline__ void adv(ptr & p) { bump<512*N>(p.q); bump<64*N>(p.h); }
    __device__ __forceinline__ float dot(const raw & r, const uint32_t (&a)[16], const float yd, const float ys, const float acc, const int) const {
        float s, bias;
        kq_hdr<0x41000000>(r.h, c, s, bias);
        uint32_t se, so;
        asm("{\n\t.reg .b32 ml, mh, s, l, h;\n\t"
            "mov.b32 ml, %22;\n\tmov.b32 mh, %23;\n\tmov.b32 %0, 0;\n\tmov.b32 %1, 0;\n\t"
#define Q4W(QW, A0, A1, A2, A3) \
            "lop3.b32 l, " QW ", 0x000f000f, 0x64006400, 0xea;\n\t" \
            "lop3.b32 h, " QW ", 0x00f000f0, 0x54005400, 0xea;\n\t" \
            "shr.b32 s, " QW ", 8;\n\t" \
            "sub.rn.f16x2 l, l, ml;\n\t" \
            "sub.rn.f16x2 h, h, mh;\n\t" \
            "fma.rn.f16x2 %0, l, " A0 ", %0;\n\t" \
            "fma.rn.f16x2 %1, h, " A1 ", %1;\n\t" \
            "lop3.b32 l, s, 0x000f000f, 0x64006400, 0xea;\n\t" \
            "lop3.b32 h, s, 0x00f000f0, 0x54005400, 0xea;\n\t" \
            "sub.rn.f16x2 l, l, ml;\n\t" \
            "sub.rn.f16x2 h, h, mh;\n\t" \
            "fma.rn.f16x2 %0, l, " A2 ", %0;\n\t" \
            "fma.rn.f16x2 %1, h, " A3 ", %1;\n\t"
            Q4W("%2", "%6", "%7", "%8", "%9")
            Q4W("%3", "%10", "%11", "%12", "%13")
            Q4W("%4", "%14", "%15", "%16", "%17")
            Q4W("%5", "%18", "%19", "%20", "%21")
#undef Q4W
            "}"
            : "=&r"(se), "=&r"(so)
            : "r"(r.q.x), "r"(r.q.y), "r"(r.q.z), "r"(r.q.w),
              "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(a[4]), "r"(a[5]), "r"(a[6]), "r"(a[7]),
              "r"(a[8]), "r"(a[9]), "r"(a[10]), "r"(a[11]), "r"(a[12]), "r"(a[13]), "r"(a[14]), "r"(a[15]),
              "r"(h2u(__float2half2_rn(1032.0f))), "r"(h2u(__float2half2_rn(72.0f))));
        return tail2b(se, so, s, bias, yd, ys, acc);
    }
};

// Q5_K, q5kr layout: qs plane (16 B per group) | qh plane (4 B per group) | header plane (16 B per 8 groups)
struct xq5 {
    static constexpr int kStage = 0;   // bytes of shared-memory staging per row (0 = none)
    __device__ __forceinline__ void set_stage(uint32_t) {}
    kq_hdr_c c;
    struct ptr { const char * q; const char * h; const char * hd; };
    struct raw { uint4 q; uint32_t h; uint4 hd; };
    static __device__ __forceinline__ void prefetch(const char * xr, const int t, const int nb) { dec_q5kr<0>::prefetch(xr, t, nb); }
    __device__ __forceinline__ explicit xq5(const int lane) : c(kq_hdr_init(lane)) {}
    __device__ __forceinline__ void init(ptr & p, const char * xr, const int lane, const int nb) const {
        p.q = xr + 16*lane; p.h = xr + 128*nb + 4*lane; p.hd = xr + 160*nb + 16*(lane >> 3);
    }
    template <int J> static __device__ __forceinline__ void load(raw & r, const ptr & p) {
        r.q = ldx4<512*J>(p.q); r.h = ldx1<128*J>(p.h); r.hd = ldx4<64*J>(p.hd);
    }
    template <int N> static __device__ __forceinline__ void adv(ptr & p) { bump<512*N>(p.q); bump<128*N>(p.h); bump<64*N>(p.hd); }
    __device__ __forceinline__ float dot(const raw & r, const uint32_t (&a)[16], const float yd, const float ys, const float acc, const int) const {
        float s, bias;
        kq_hdr<0x41800000>(r.hd, c, s, bias);
        uint32_t se, so;
        // word K: t0 = lo(w) | ((H << (4-K)) & 0x10), t2 = lo(w >> 8) | ((H >> K) & 0x10),
        //         t1 = hi(w) | ((H >> K) & 0x100),    t3 = hi(w >> 8) | ((H >> (K+4)) & 0x100)   (per 16-bit half)
        asm("{\n\t.reg .b32 ml, mh, s, x, y, k1, k2, l, h;\n\t"
            "mov.b32 ml, %23;\n\tmov.b32 mh, %24;\n\tmov.b32 %0, 0;\n\tmov.b32 %1, 0;\n\t"
#define Q5W(QW, SHL, SHR, SHR4, A0, A1, A2, A3) \
            "shr.b32 s, " QW ", 8;\n\t" \
            "shl.b32 k1, %6, " SHL ";\n\t" \
            "shr.b32 k2, %6, " SHR ";\n\t" \
            "shr.b32 y, %6, " SHR4 ";\n\t" \
            "lop3.b32 l, " QW ", 0x000f000f, 0x64006400, 0xea;\n\t" \
            "lop3.b32 l, k1, 0x00100010, l, 0xea;\n\t" \
            "lop3.b32 h, " QW ", 0x00f000f0, 0x54005400, 0xea;\n\t" \
            "lop3.b32 h, k2, 0x01000100, h, 0xea;\n\t" \
            "sub.rn.f16x2 l, l, ml;\n\t" \
            "sub.rn.f16x2 h, h, mh;\n\t" \
            "fma.rn.f16x2 %0, l, " A0 ", %0;\n\t" \
            "fma.rn.f16x2 %1, h, " A1 ", %1;\n\t" \
            "lop3.b32 l, s, 0x000f000f, 0x64006400, 0xea;\n\t" \
            "lop3.b32 l, k2, 0x00100010, l, 0xea;\n\t" \
            "lop3.b32 h, s, 0x00f000f0, 0x54005400, 0xea;\n\t" \
            "lop3.b32 h, y, 0x01000100, h, 0xea;\n\t" \
            "sub.rn.f16x2 l, l, ml;\n\t" \
            "sub.rn.f16x2 h, h, mh;\n\t" \
            "fma.rn.f16x2 %0, l, " A2 ", %0;\n\t" \
            "fma.rn.f16x2 %1, h, " A3 ", %1;\n\t"
            Q5W("%2", "4", "0", "4", "%7", "%8", "%9", "%10")
            Q5W("%3", "3", "1", "5", "%11", "%12", "%13", "%14")
            Q5W("%4", "2", "2", "6", "%15", "%16", "%17", "%18")
            Q5W("%5", "1", "3", "7", "%19", "%20", "%21", "%22")
#undef Q5W
            "}"
            : "=&r"(se), "=&r"(so)
            : "r"(r.q.x), "r"(r.q.y), "r"(r.q.z), "r"(r.q.w), "r"(r.h),
              "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(a[4]), "r"(a[5]), "r"(a[6]), "r"(a[7]),
              "r"(a[8]), "r"(a[9]), "r"(a[10]), "r"(a[11]), "r"(a[12]), "r"(a[13]), "r"(a[14]), "r"(a[15]),
              "r"(h2u(__float2half2_rn(1040.0f))), "r"(h2u(__float2half2_rn(80.0f))));
        return tail2b(se, so, s, bias, yd, ys, acc);
    }
};

// Q3_K, q3r layout: A, B, H words (12 B per group) | scale words (12 B per 8 groups) | d (2 B per 8 groups)
struct xq3 {
    static constexpr int kStage = 0;   // bytes of shared-memory staging per row (0 = none)
    __device__ __forceinline__ void set_stage(uint32_t) {}
    uint32_t lo0, lo1, hi0, hi1;   // bfe positions of the two 4-bit low scale fields / 2-bit high scale fields
    struct ptr { const char * q; const char * sl; const char * s2; const char * d; };
    struct raw { uint32_t A, B, H, wlo, w2, d; };
    static __device__ __forceinline__ void prefetch(const char * xr, const int t, const int nb) { dec_q3r<0>::prefetch(xr, t, nb); }
    __device__ __forceinline__ explicit xq3(const int lane) {
        const int g = lane & 7;
        lo0 = 16*(g & 1) + (g >= 4 ? 4 : 0); lo1 = lo0 + 8;
        hi0 = 16*(g & 1) + 2*(g >> 1);       hi1 = hi0 + 8;
    }
    __device__ __forceinline__ void init(ptr & p, const char * xr, const int lane, const int nb) const {
        const int g = lane & 7, widx = (g & 3) >> 1;
        p.q  = xr + 12*lane;
        p.sl = xr + 96*nb + 12*(lane >> 3) + 4*widx;
        p.s2 = xr + 96*nb + 12*(lane >> 3) + 8;
        p.d  = xr + 108*nb + 2*(lane >> 3);
    }
    template <int J> static __device__ __forceinline__ void load(raw & r, const ptr & p) {
        r.A = ldx1<384*J>(p.q); r.B = ldx1<384*J + 4>(p.q); r.H = ldx1<384*J + 8>(p.q);
        r.wlo = ldx1<48*J>(p.sl); r.w2 = ldx1<48*J>(p.s2); r.d = ldxh<8*J>(p.d);
    }
    template <int N> static __device__ __forceinline__ void adv(ptr & p) { bump<384*N>(p.q); bump<48*N>(p.sl); bump<48*N>(p.s2); bump<8*N>(p.d); }
    __device__ __forceinline__ float dot(const raw & r, const uint32_t (&a)[16], const float yd, const float, const float acc, const int) const {
        return dot_abh(r.A, r.B, r.H, r.wlo, r.w2, r.d, a, yd, acc);
    }
    __device__ __forceinline__ float dot_abh(const uint32_t A, const uint32_t B, const uint32_t H, const uint32_t wlo,
                                             const uint32_t w2, const uint32_t dd, const uint32_t (&a)[16], const float yd,
                                             const float acc) const {
        float sce, sco;
        asm("{\n\t.reg .b32 x, y, e, o;\n\t.reg .f32 fe, fo, fd;\n\t.reg .b16 dh;\n\t"
            "bfe.u32 x, %2, %4, 4;\n\tbfe.u32 y, %3, %6, 2;\n\tbfi.b32 e, y, x, 4, 2;\n\t"
            "bfe.u32 x, %2, %5, 4;\n\tbfe.u32 y, %3, %7, 2;\n\tbfi.b32 o, y, x, 4, 2;\n\t"
            "sub.s32 e, e, 32;\n\tsub.s32 o, o, 32;\n\t"
            "cvt.rn.f32.s32 fe, e;\n\tcvt.rn.f32.s32 fo, o;\n\t"
            "cvt.u16.u32 dh, %8;\n\tcvt.f32.f16 fd, dh;\n\t"
            "mul.rn.f32 %0, fd, fe;\n\tmul.rn.f32 %1, fd, fo;\n\t}"
            : "=f"(sce), "=f"(sco)
            : "r"(wlo), "r"(w2), "r"(lo0), "r"(lo1), "r"(hi0), "r"(hi1), "r"(dd));
        uint32_t se, so;
        // word k: W = (k < 2 ? A : B) >> 8*(k & 1), V = {H, H >> 1, rotr(H, 8), rotr(H, 9)}[k];
        // slot q: ((W & (3 << 2q)) | M[q] | (V & (4 << 2q))) - sub[q]   (per 16-bit half)
        asm("{\n\t.reg .b32 c0, c1, c2, c3, w, v, x;\n\t"
            "mov.b32 c0, %21;\n\tmov.b32 c1, %22;\n\tmov.b32 c2, %23;\n\tmov.b32 c3, %24;\n\tmov.b32 %0, 0;\n\tmov.b32 %1, 0;\n\t"
#define Q3S(MW, MV, MM, C, A, ACC) \
            "lop3.b32 x, w, " MW ", " MM ", 0xea;\n\t" \
            "lop3.b32 x, v, " MV ", x, 0xea;\n\t" \
            "sub.rn.f16x2 x, x, " C ";\n\t" \
            "fma.rn.f16x2 " ACC ", x, " A ", " ACC ";\n\t"
#define Q3K(A0, A1, A2, A3) \
            Q3S("0x00030003", "0x00040004", "0x64006400", "c0", A0, "%0") \
            Q3S("0x000c000c", "0x00100010", "0x5c005c00", "c1", A1, "%1") \
            Q3S("0x00300030", "0x00400040", "0x54005400", "c2", A2, "%0") \
            Q3S("0x00c000c0", "0x01000100", "0x4c004c00", "c3", A3, "%1")
            "mov.b32 w, %2;\n\tmov.b32 v, %4;\n\t"
            Q3K("%5", "%6", "%7", "%8")
            "shr.b32 w, %2, 8;\n\tshr.b32 v, %4, 1;\n\t"
            Q3K("%9", "%10", "%11", "%12")
            "mov.b32 w, %3;\n\tshf.r.wrap.b32 v, %4, %4, 8;\n\t"
            Q3K("%13", "%14", "%15", "%16")
            "shr.b32 w, %3, 8;\n\tshf.r.wrap.b32 v, %4, %4, 9;\n\t"
            Q3K("%17", "%18", "%19", "%20")
#undef Q3K
#undef Q3S
            "}"
            : "=&r"(se), "=&r"(so)
            : "r"(A), "r"(B), "r"(H),
              "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(a[4]), "r"(a[5]), "r"(a[6]), "r"(a[7]),
              "r"(a[8]), "r"(a[9]), "r"(a[10]), "r"(a[11]), "r"(a[12]), "r"(a[13]), "r"(a[14]), "r"(a[15]),
              "r"(h2u(__float2half2_rn(1028.0f))), "r"(h2u(__float2half2_rn(260.0f))),
              "r"(h2u(__float2half2_rn(68.0f))), "r"(h2u(__float2half2_rn(20.0f))));
        return tail2(se, so, sce, sco, yd, acc);
    }
};

// Q3_K level 2: the same math; the A/B/H words of the warp's 32 groups (384 contiguous bytes) come in with one 16-byte
// ld.global.nc per lane, go through a per-warp, per-row shared slot and are read back as each lane's three words
// (12-byte stride: conflict-free).  Every q byte is requested once instead of by three loads.  q3r rows are only
// 4-byte aligned (110*nb bytes), so the warp reads the 400 bytes from the row's 16-byte aligned-down start (lanes 0..24)
// and the lanes read their words m = (row start & 15) bytes further in.
struct xq3s {
    static constexpr int kStage = 400;
    xq3 m;
    uint32_t sst, sld;   // shared addresses: this lane's 16-byte store slot / 12-byte read slot in row 0's area
    bool qok;            // lane < 25: loads one 16-byte chunk
    struct ptr { const char * q; const char * sl; const char * s2; const char * d; uint32_t m; };
    struct raw { uint4 v; uint32_t wlo, w2, d, m; };
    static __device__ __forceinline__ void prefetch(const char * xr, const int t, const int nb) { dec_q3r<0>::prefetch(xr, t, nb); }
    __device__ __forceinline__ explicit xq3s(const int lane) : m(lane), sst(0), sld(0), qok(lane < 25) {}
    __device__ __forceinline__ void set_stage(const uint32_t warp_base) {
        const int lane = threadIdx.x;
        sst = warp_base + 16*lane; sld = warp_base + 12*lane;
    }
    __device__ __forceinline__ void init(ptr & p, const char * xr, const int lane, const int nb) const {
        const int g = lane & 7, widx = (g & 3) >> 1;
        p.m  = (uint32_t) ((uintptr_t) xr & 15);
        p.q  = xr - p.m + 16*lane;
        p.sl = xr + 96*nb + 12*(lane >> 3) + 4*widx;
        p.s2 = xr + 96*nb + 12*(lane >> 3) + 8;
        p.d  = xr + 108*nb + 2*(lane >> 3);
    }
    template <int J> __device__ __forceinline__ void load(raw & r, const ptr & p) const {
        if (qok) r.v = ldx4<384*J>(p.q); else r.v = make_uint4(0, 0, 0, 0);
        r.wlo = ldx1<48*J>(p.sl); r.w2 = ldx1<48*J>(p.s2); r.d = ldxh<8*J>(p.d); r.m = p.m;
    }
    template <int N> static __device__ __forceinline__ void adv(ptr & p) { bump<384*N>(p.q); bump<48*N>(p.sl); bump<48*N>(p.s2); bump<8*N>(p.d); }
    __device__ __forceinline__ float dot(const raw & r, const uint32_t (&a)[16], const float yd, const float, const float acc, const int i) const {
        // lockstep warp (sm_60): the store of every active lane precedes the reads in program order; sm_70+: explicit sync
        if (qok) asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};" :: "r"(sst + kStage*i), "r"(r.v.x), "r"(r.v.y), "r"(r.v.z), "r"(r.v.w) : "memory");
#if __CUDA_ARCH__ >= 700
        __syncwarp(__activemask());
#endif
        uint32_t A, B, H;
        asm volatile("ld.shared.u32 %0, [%3];\n\tld.shared.u32 %1, [%3+4];\n\tld.shared.u32 %2, [%3+8];"
                     : "=r"(A), "=r"(B), "=r"(H) : "r"(sld + kStage*i + r.m) : "memory");
#if __CUDA_ARCH__ >= 700
        __syncwarp(__activemask());
#endif
        return m.dot_abh(A, B, H, r.wlo, r.w2, r.d, a, yd, acc);
    }
};

// Q6_K, q6r3 layout: ql2 (16 B per group, 8- or 16-byte aligned: VA) | qh (8 B per group) | sc (2 B per group) | d
template <int VA>
struct xq6 {
    static constexpr int kStage = 0;   // bytes of shared-memory staging per row (0 = none)
    __device__ __forceinline__ void set_stage(uint32_t) {}
    struct ptr { const char * q; const char * h; const char * sc; const char * d; };
    struct raw { uint32_t q[4]; uint32_t ha, hb, sc, d; };
    static __device__ __forceinline__ void prefetch(const char * xr, const int t, const int nb) { dec_q6r3<0, VA>::prefetch(xr, t, nb); }
    __device__ __forceinline__ explicit xq6(const int) {}
    __device__ __forceinline__ void init(ptr & p, const char * xr, const int lane, const int nb) const {
        p.q = xr + 16*lane; p.h = xr + 128*nb + 8*lane; p.sc = xr + 192*nb + 2*lane; p.d = xr + 208*nb + 2*(lane >> 3);
    }
    template <int J> static __device__ __forceinline__ void load(raw & r, const ptr & p) {
        if constexpr (VA == 16) {
            const uint4 v = ldx4<512*J>(p.q);
            r.q[0] = v.x; r.q[1] = v.y; r.q[2] = v.z; r.q[3] = v.w;
        } else {
            const uint2 v0 = ldx2<512*J>(p.q), v1 = ldx2<512*J + 8>(p.q);
            r.q[0] = v0.x; r.q[1] = v0.y; r.q[2] = v1.x; r.q[3] = v1.y;
        }
        const uint2 hv = ldx2<256*J>(p.h);
        r.ha = hv.x; r.hb = hv.y;
        r.sc = ldxh<64*J>(p.sc); r.d = ldxh<8*J>(p.d);
    }
    template <int N> static __device__ __forceinline__ void adv(ptr & p) { bump<512*N>(p.q); bump<256*N>(p.h); bump<64*N>(p.sc); bump<8*N>(p.d); }
    __device__ __forceinline__ float dot(const raw & r, const uint32_t (&a)[16], const float yd, const float, const float acc, const int) const {
        float sce, sco;
        asm("{\n\t.reg .b32 e, o;\n\t.reg .f32 fe, fo, fd;\n\t.reg .b16 dh;\n\t"
            "bfe.s32 e, %2, 0, 8;\n\tbfe.s32 o, %2, 8, 8;\n\t"
            "cvt.rn.f32.s32 fe, e;\n\tcvt.rn.f32.s32 fo, o;\n\t"
            "cvt.u16.u32 dh, %3;\n\tcvt.f32.f16 fd, dh;\n\t"
            "mul.rn.f32 %0, fd, fe;\n\tmul.rn.f32 %1, fd, fo;\n\t}"
            : "=f"(sce), "=f"(sco) : "r"(r.sc), "r"(r.d));
        uint32_t se, so;
        // word K: t0 = lo(w) | (shl(ha, 4-4K) & 0x30), t2 = lo(w >> 8) | (shl(ha, 2-4K) & 0x30),
        //         t1 = hi(w) | (shl(hb, 8-4K) & 0x300), t3 = hi(w >> 8) | (shl(hb, 6-4K) & 0x300)   (per 16-bit half)
        asm("{\n\t.reg .b32 ml, mh, s, x, l, h;\n\t"
            "mov.b32 ml, %24;\n\tmov.b32 mh, %25;\n\tmov.b32 %0, 0;\n\tmov.b32 %1, 0;\n\t"
#define Q6W(QW, HA0, HA2, HB1, HB3, A0, A1, A2, A3) \
            "shr.b32 s, " QW ", 8;\n\t" \
            "lop3.b32 l, " QW ", 0x000f000f, 0x64006400, 0xea;\n\t" HA0 \
            "lop3.b32 l, x, 0x00300030, l, 0xea;\n\t" \
            "lop3.b32 h, " QW ", 0x00f000f0, 0x54005400, 0xea;\n\t" HB1 \
            "lop3.b32 h, x, 0x03000300, h, 0xea;\n\t" \
            "sub.rn.f16x2 l, l, ml;\n\t" \
            "sub.rn.f16x2 h, h, mh;\n\t" \
            "fma.rn.f16x2 %0, l, " A0 ", %0;\n\t" \
            "fma.rn.f16x2 %1, h, " A1 ", %1;\n\t" \
            "lop3.b32 l, s, 0x000f000f, 0x64006400, 0xea;\n\t" HA2 \
            "lop3.b32 l, x, 0x00300030, l, 0xea;\n\t" \
            "lop3.b32 h, s, 0x00f000f0, 0x54005400, 0xea;\n\t" HB3 \
            "lop3.b32 h, x, 0x03000300, h, 0xea;\n\t" \
            "sub.rn.f16x2 l, l, ml;\n\t" \
            "sub.rn.f16x2 h, h, mh;\n\t" \
            "fma.rn.f16x2 %0, l, " A2 ", %0;\n\t" \
            "fma.rn.f16x2 %1, h, " A3 ", %1;\n\t"
            Q6W("%2", "shl.b32 x, %6, 4;\n\t", "shl.b32 x, %6, 2;\n\t", "shl.b32 x, %7, 8;\n\t", "shl.b32 x, %7, 6;\n\t", "%8", "%9", "%10", "%11")
            Q6W("%3", "mov.b32 x, %6;\n\t",    "shr.b32 x, %6, 2;\n\t", "shl.b32 x, %7, 4;\n\t", "shl.b32 x, %7, 2;\n\t", "%12", "%13", "%14", "%15")
            Q6W("%4", "shr.b32 x, %6, 4;\n\t", "shr.b32 x, %6, 6;\n\t", "mov.b32 x, %7;\n\t",    "shr.b32 x, %7, 2;\n\t", "%16", "%17", "%18", "%19")
            Q6W("%5", "shr.b32 x, %6, 8;\n\t", "shr.b32 x, %6, 10;\n\t", "shr.b32 x, %7, 4;\n\t", "shr.b32 x, %7, 6;\n\t", "%20", "%21", "%22", "%23")
#undef Q6W
            "}"
            : "=&r"(se), "=&r"(so)
            : "r"(r.q[0]), "r"(r.q[1]), "r"(r.q[2]), "r"(r.q[3]), "r"(r.ha), "r"(r.hb),
              "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(a[4]), "r"(a[5]), "r"(a[6]), "r"(a[7]),
              "r"(a[8]), "r"(a[9]), "r"(a[10]), "r"(a[11]), "r"(a[12]), "r"(a[13]), "r"(a[14]), "r"(a[15]),
              "r"(h2u(__float2half2_rn(1056.0f))), "r"(h2u(__float2half2_rn(96.0f))));
        return tail2(se, so, sce, sco, yd, acc);
    }
};

// ------------------------------------------------------------------------------------------------- kernel
// mv_wrp with the PTX decoders: same staging, tiles, per-row group order (t = lane, lane + 32, ...) and warp tree.
// UX = groups per lane per step (1 or 2; with 2 all 2*RW records are loaded before the first dot); UX = 3: one group per
// step, and the first step of the warp's first tile is loaded before the activation staging and its barrier.
static inline int wrx_ngp(const int ngroups) { return ngroups + ((10 - ngroups % 8) % 8); }
static inline size_t wrx_smem(const int ngroups) { return (size_t) 4*wrx_ngp(ngroups)*16 + (size_t) ngroups*8; }

template <class X, int NW, int RW, int MINB, bool PF, bool PQ, int UX>
static __global__ void __launch_bounds__(WARP_SIZE*NW, MINB)
mv_wrpx(const char * __restrict__ vx, const float * __restrict__ x, const half2 * __restrict__ aq,
        const half2 * __restrict__ ads, float * __restrict__ dst,
        const int ngroups, const int nrows, const int row_bytes, const int ngp) {
    extern __shared__ uint4 wrx_sm[];
    const int lane = threadIdx.x;
    const int tid  = threadIdx.x + WARP_SIZE*threadIdx.y;
    const int ntiles = (nrows + RW - 1)/RW;
    const int wstep  = gridDim.x*NW;
    int tile = blockIdx.x*NW + threadIdx.y;
    if constexpr (PF) {
        if (tile < ntiles) {
            const int nb = ngroups/8;
#pragma unroll
            for (int i = 0; i < RW; ++i) {
                const char * xr = vx + (size_t) min(tile*RW + i, nrows - 1)*row_bytes;
                for (int t = lane; t < ngroups; t += WARP_SIZE) X::prefetch(xr, t, nb);
            }
        }
    }
    X dec(lane);
    typename X::raw rpre[RW];
    typename X::ptr ppre[RW];
    bool have = false;
    if constexpr (UX == 3) {
        if (tile < ntiles) {
#pragma unroll
            for (int i = 0; i < RW; ++i) dec.init(ppre[i], vx + (size_t) min(tile*RW + i, nrows - 1)*row_bytes, lane, ngroups/8);
#pragma unroll
            for (int i = 0; i < RW; ++i) dec.template load<0>(rpre[i], ppre[i]);
            have = true;
        }
    }
    float2 * sds = (float2 *) (wrx_sm + 4*ngp);
    if constexpr (PQ) {
        (void) x;
        const uint4 * ag = (const uint4 *) aq;
        for (int i = tid; i < 4*ngroups; i += WARP_SIZE*NW) wrx_sm[(i & 3)*ngp + (i >> 2)] = ag[i];
        for (int i = tid; i < ngroups; i += WARP_SIZE*NW) sds[i] = __half22float2(ads[i]);
    } else {
        (void) aq; (void) ads;
        for (int g = tid; g < ngroups; g += WARP_SIZE*NW) {
            uint4 q[4];
            float2 ds;
            a16_group(x + 32*g, q, ds);
#pragma unroll
            for (int c = 0; c < 4; ++c) wrx_sm[c*ngp + g] = q[c];
            sds[g] = ds;
        }
    }
    __syncthreads();

    if constexpr (X::kStage > 0) {   // per-warp staging slots after the activation planes (16-byte aligned)
        const uint32_t base = (uint32_t) __cvta_generic_to_shared(wrx_sm) + (uint32_t) ((4*ngp*16 + ngroups*8 + 15) & ~15);
        dec.set_stage(base + threadIdx.y*RW*X::kStage);
    }
    const int nb = ngroups/8;
    for (; tile < ntiles; tile += wstep) {
        const int row0 = tile*RW;
        typename X::ptr p[RW];
        if (UX == 3 && have) {
#pragma unroll
            for (int i = 0; i < RW; ++i) p[i] = ppre[i];
        } else {
#pragma unroll
            for (int i = 0; i < RW; ++i) dec.init(p[i], vx + (size_t) min(row0 + i, nrows - 1)*row_bytes, lane, nb);
        }
        float acc[RW];
#pragma unroll
        for (int i = 0; i < RW; ++i) acc[i] = 0.0f;
        auto step = [&](const typename X::raw (&r)[RW], const int t) {
            uint32_t a[16];
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                const uint4 v = wrx_sm[c*ngp + t];
                a[4*c + 0] = v.x; a[4*c + 1] = v.y; a[4*c + 2] = v.z; a[4*c + 3] = v.w;
            }
            const float2 ds = sds[t];
#pragma unroll
            for (int i = 0; i < RW; ++i) acc[i] = dec.dot(r[i], a, ds.x, ds.y, acc[i], i);
        };
        int t = lane;
        if constexpr (UX == 3) {
            if (have) {
#pragma unroll
                for (int i = 0; i < RW; ++i) X::template adv<1>(p[i]);
                step(rpre, t);
                t += WARP_SIZE;
                have = false;
            }
        }
        if constexpr (UX == 2) {
#pragma unroll 1
            for (; t + WARP_SIZE < ngroups; t += 2*WARP_SIZE) {
                typename X::raw r0[RW], r1[RW];
#pragma unroll
                for (int i = 0; i < RW; ++i) dec.template load<0>(r0[i], p[i]);
#pragma unroll
                for (int i = 0; i < RW; ++i) dec.template load<1>(r1[i], p[i]);
#pragma unroll
                for (int i = 0; i < RW; ++i) X::template adv<2>(p[i]);
                step(r0, t);
                step(r1, t + WARP_SIZE);
            }
        }
#pragma unroll 1
        for (; t < ngroups; t += WARP_SIZE) {
            typename X::raw r0[RW];
#pragma unroll
            for (int i = 0; i < RW; ++i) dec.template load<0>(r0[i], p[i]);
#pragma unroll
            for (int i = 0; i < RW; ++i) X::template adv<1>(p[i]);
            step(r0, t);
        }
#pragma unroll
        for (int i = 0; i < RW; ++i) {
            const float v = warp_reduce_sum(acc[i]);
            if (lane == i && row0 + i < nrows) dst[row0 + i] = v;
        }
    }
}

// ------------------------------------------------------------------------------------------------- launch
template <class X, int NW, int RW, bool PQ, int UX>
static bool wrpx_launch(const char * vx, const float * x, const half2 * aq, const half2 * ads, float * y, int ngroups,
                        int nrows, int row_bytes, cudaStream_t s) {
    const size_t smem = X::kStage > 0 ? ((wrx_smem(ngroups) + 15) & ~(size_t) 15) + (size_t) NW*RW*X::kStage : wrx_smem(ngroups);
    if (smem > 48*1024) return false;
    struct entry { int dev; size_t smem; int nblk; };
    static entry cache[16];
    static int ncache = 0;
    int dev = 0;
    cudaGetDevice(&dev);
    int nblk = 0;
    for (int i = 0; i < ncache; ++i) if (cache[i].dev == dev && cache[i].smem == smem) { nblk = cache[i].nblk; break; }
    if (nblk == 0) {
        int per_sm = 0, nsm = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, mv_wrpx<X, NW, RW, 1, false, PQ, UX>, WARP_SIZE*NW, smem);
        cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
        nblk = std::max(1, per_sm)*nsm;
        if (ncache < 16) cache[ncache++] = entry{dev, smem, nblk};
    }
    const int ntiles = (nrows + RW - 1)/RW;
    const int grid = std::max(1, std::min(nblk, (ntiles + NW - 1)/NW));
    mv_wrpx<X, NW, RW, 1, false, PQ, UX><<<grid, dim3(WARP_SIZE, NW), smem, s>>>(vx, x, aq, ads, y, ngroups, nrows, row_bytes,
                                                                              wrx_ngp(ngroups));
    return true;
}

template <class X, bool PQ, int UX>
static bool wrpx_geo(int geo, const char * vx, const float * x, const half2 * aq, const half2 * ads, float * y, int ngroups,
                     int nrows, int row_bytes, cudaStream_t s) {
    switch (geo) {
        case 303: return wrpx_launch<X,  8, 2, PQ, UX>(vx, x, aq, ads, y, ngroups, nrows, row_bytes, s);
        case 306: return wrpx_launch<X,  8, 4, PQ, UX>(vx, x, aq, ads, y, ngroups, nrows, row_bytes, s);
        case 314: return wrpx_launch<X, 16, 2, PQ, UX>(vx, x, aq, ads, y, ngroups, nrows, row_bytes, s);
        case 320: return wrpx_launch<X, 12, 4, PQ, UX>(vx, x, aq, ads, y, ngroups, nrows, row_bytes, s);
        default:  return false;
    }
}

template <class X>
static bool wrpx_t(int level, int geo, const char * vx, const float * x, const half2 * aq, const half2 * ads, float * y,
                   int ngroups, int nrows, int row_bytes, cudaStream_t s) {
    const bool pq = aq != nullptr;
    if (level == 3) return pq ? wrpx_geo<X, true, 3>(geo, vx, x, aq, ads, y, ngroups, nrows, row_bytes, s)
                              : wrpx_geo<X, false, 3>(geo, vx, x, aq, ads, y, ngroups, nrows, row_bytes, s);
    if (level == 2) return pq ? wrpx_geo<X, true, 2>(geo, vx, x, aq, ads, y, ngroups, nrows, row_bytes, s)
                              : wrpx_geo<X, false, 2>(geo, vx, x, aq, ads, y, ngroups, nrows, row_bytes, s);
    return pq ? wrpx_geo<X, true, 1>(geo, vx, x, aq, ads, y, ngroups, nrows, row_bytes, s)
              : wrpx_geo<X, false, 1>(geo, vx, x, aq, ads, y, ngroups, nrows, row_bytes, s);
}

// P100E_MVWRP_Q3 / _Q4 / _Q5 / _Q6 (index 0..3), read on first use and again by wrx_reload_env(); the bench's
// mv_tune.wrx[] (>= 0) takes precedence
static int wrx_env[4] = {-1, -1, -1, -1};
bool wrx_reload_env() {
    static const char * const names[4] = {"P100E_MVWRP_Q3", "P100E_MVWRP_Q4", "P100E_MVWRP_Q5", "P100E_MVWRP_Q6"};
    bool changed = false;
    for (int k = 0; k < 4; ++k) {
        const char * e = getenv(names[k]);
        const int v = e && *e ? atoi(e) : 0;
        changed |= wrx_env[k] >= 0 && v != wrx_env[k];
        wrx_env[k] = v;
    }
    {   // P100E_MV_WRM_PH (K-phased wrm tile rounds, matvec_wrm.cuh) is baked into captured graphs like the MVWRP levels
        const char * e = getenv("P100E_MV_WRM_PH");
        const int v = e && *e ? atoi(e) : 0;
        changed |= v != mv_tune.wrm_ph;
        mv_tune.wrm_ph = v;
    }
    return changed;
}
int wrx_level(int type_idx) {
    if (wrx_env[0] < 0) wrx_reload_env();
    const int v = mv_tune.wrx[type_idx];
    return v >= 0 ? v : wrx_env[type_idx];
}

static void wrx_note(const int type_idx, const int lv, const int variant) {   // one line per type and level, first launch
    static bool seen[4][4] = {};
    if (type_idx < 0 || type_idx > 3 || lv < 0 || lv > 3 || seen[type_idx][lv]) return;
    seen[type_idx][lv] = true;
    static const char * const nm[4] = {"q3_K", "q4_K", "q5_K", "q6_K"};
    fprintf(stderr, "p100e: inline-PTX mv_wrp in use for %s (level %d, variant %d)\n", nm[type_idx], lv, variant);
}

bool launch_wrpx(int layout, const char * vx, const float * x, const half2 * aq, const half2 * ads, float * y, int ngroups,
                 int nrows, int row_bytes, int variant, cudaStream_t s) {
    const int geo = variant >= 500 ? variant - 200 : variant;
    if (ngroups % 8 != 0) return false;
    const int xi = layout == LAYOUT_Q3R ? 0 : layout == LAYOUT_Q4KR ? 1 : layout == LAYOUT_Q5KR ? 2 : layout == LAYOUT_Q6R3 ? 3 : -1;
    if (xi < 0) return false;
    const int lv = wrx_level(xi);
    if (lv <= 0) return false;
    bool ok = false;
    switch (layout) {
        case LAYOUT_Q3R:
            ok = lv == 2 ? row_bytes % 4 == 0 && wrpx_t<xq3s>(1, geo, vx, x, aq, ads, y, ngroups, nrows, row_bytes, s)   // staged
                         : wrpx_t<xq3>(lv, geo, vx, x, aq, ads, y, ngroups, nrows, row_bytes, s);
            break;
        case LAYOUT_Q4KR: ok = wrpx_t<xq4>(lv, geo, vx, x, aq, ads, y, ngroups, nrows, row_bytes, s); break;
        case LAYOUT_Q5KR: ok = wrpx_t<xq5>(lv, geo, vx, x, aq, ads, y, ngroups, nrows, row_bytes, s); break;
        case LAYOUT_Q6R3:
            if (row_bytes % 16 == 0)     ok = wrpx_t<xq6<16>>(lv, geo, vx, x, aq, ads, y, ngroups, nrows, row_bytes, s);
            else if (row_bytes % 8 == 0) ok = wrpx_t<xq6<8>>(lv, geo, vx, x, aq, ads, y, ngroups, nrows, row_bytes, s);
            break;
        default: break;
    }
    if (ok) wrx_note(xi, lv, variant);
    return ok;
}

} // namespace p100e::fast
