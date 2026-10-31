// p100e fast: dec_q6h, the Q6_K (q6r3 layout) decoder with a shorter fp32 tail
// (multi-column verify), behind P100E_MV_Q6_H16 (default 2; 0 = dec_q6r3 everywhere).
//
// Today's q6r3 path (dec_q6r3, matvec_dec.cuh) is ALREADY an HFMA2 dot: fp16 a16 activations (|a| <= 1 per 32-group),
// fp16 integer weights q - 32, two half2 chains per (row, column, group) - se over the even slots, so over the odd slots,
// 8 products per half lane - and then the fp32 tail fin(): 4 half -> f32 conversions, 2 FADD, 2 FMUL, FFMA, FMUL per
// (row, column, group). On sm_60 a half -> f32 conversion and the int8 -> f32 scale conversions are quarter-rate
// (CUDA guide table 3, cc 6.0: 16 conversions per clock per SM vs 64 FP32 / 128 fp16 lanes), so that tail is a large
// part of the issue cost per column at T >= 2. This decoder keeps the same weights, slots, chains and per-row group
// order, and only changes the tail:
//
//   F = 1 ("lane fold"): the two half lanes of se and of so are added in fp16 (16-term sums), then 2 conversions:
//         acc = acc + yd * (sc_e * f32(se.x + se.y) + sc_o * f32(so.x + so.y))   [explicit _rn ops, no contraction]
//   F = 2 ("scale fold"): the int8 sub-block scales go into fp16 as sc/4 (exact: |sc| <= 128, power-of-2 factor) with
//         one LOP3 + PRMT + HSUB2 + HMUL2 per (row, group) (no I2F), then per (row, column, group)
//         h = so * sco4 + se * sce4 (HMUL2 + HFMA2), one fp16 lane add, ONE conversion:
//         acc = fma(yd * (4 d), f32(h.x + h.y), acc)                                 [explicit _rn ops]
//         Chains of 8 products per half lane + 2 scale products + 1 lane add: at most 32 fp16 terms before fp32.
//
// Overflow (fp16 max 65504): with |a| <= 1, |q - 32| <= 32, so |se.x|, |so.x| <= 8 x 32 = 256; F = 1 lane sum <= 512.
// F = 2: |sc/4| <= 32, so |se * sce4| <= 8192, |h.x| <= 16384, |h.x + h.y| <= 32768 < 65504. |a| <= 1 holds for every
// group with amax < 65520: quantize_a16 divides by the group amax, so the activation's size does not matter (the largest
// seen on a real model is 467). Above that amax is clamped to 65504 and |a| = |x|/65504 passes 1 (matvec_a16.cu):
// F = 2 is finite for any row up to |x| of about 130000 (32768 |a| < 65504), and a row with larger x can reach fp16
// inf where dec_q6r3's tail (|a| x 256 only) stays finite up to about 1.3e8. Such values only come from a diverged run.
// T-invariance: with P100E_MV_Q6_H16 on, EVERY q6r3 GEMV (T = 1 included: matvec, matvec_prequantized, the parts
// launch, the q8 pair launch is not used) runs mv_wrm with this decoder (matvec_wrm_q6h1/2.cu), so a row's value depends
// only on (row, its activation column): column j of a T-column call equals the T = 1 call on column j, bit for bit.
#pragma once
#include "matvec_dec.cuh"

namespace p100e::fast {

template <int LP, int VA, int F>
struct dec_q6h : dec_q6r3<LP, VA> {
    static_assert(F == 1 || F == 2, "dec_q6h: F = 1 (lane fold) or 2 (scale fold)");
    using B = dec_q6r3<LP, VA>;
    template <int L> using rebind = dec_q6h<L, VA, F>;
    using raw = typename B::raw;
    static constexpr bool kAcc = true;   // mv_wrm: acc = fin_acc(acc, ...) instead of acc += fin(...)
    __device__ __forceinline__ explicit dec_q6h(const int lane) : B(lane) {}

    struct sc { float e, o; half2 h; };   // F = 1: e, o = d*sc (as dec_q6r3); F = 2: e = 4d, h = (sc_e/4, sc_o/4)
    __device__ __forceinline__ sc prep(const raw & r) const {
        const uint16_t dbits = r.d;
        const float d = __half2float(*((const half *) &dbits));
        sc s;
        if constexpr (F == 1) {
            s.e = d*(float) (int8_t) (r.sc & 0xFF);
            s.o = d*(float) (int8_t) (r.sc >> 8);
            s.h = make_half2(0.0f, 0.0f);
        } else {
            // int8 pair -> fp16 pair without I2F: bytes b + 128 (xor 0x80) into the low bytes of the halves of
            // 0x6400 (1024), minus 1152 = sc exactly; then x 0.25 (exact)
            const uint32_t u = (uint32_t) r.sc ^ 0x8080u;
            uint32_t p;
            asm("prmt.b32 %0, %1, 0x64006400, 0x7150;" : "=r"(p) : "r"(u));   // bytes {b0', 0x64, b1', 0x64}: lo = 0x64|b0', hi = 0x64|b1'
            const half2 v = __hsub2(as_h2(p), __float2half2_rn(1152.0f));
            s.h = __hmul2(v, __float2half2_rn(0.25f));
            s.e = __fmul_rn(4.0f, d);
            s.o = 0.0f;
        }
        return s;
    }
    static __device__ __forceinline__ float fin_acc(const float acc, const half2 se, const half2 so, const sc & s,
                                                    const float yd, const float /*ys*/) {
        if constexpr (F == 1) {
            const float e = __half2float(__hadd(__low2half(se), __high2half(se)));
            const float o = __half2float(__hadd(__low2half(so), __high2half(so)));
            return __fadd_rn(acc, __fmul_rn(yd, __fadd_rn(__fmul_rn(s.e, e), __fmul_rn(s.o, o))));
        } else {
            const half2 h = __hfma2(so, __high2half2(s.h), __hmul2(se, __low2half2(s.h)));
            const float f = __half2float(__hadd(__low2half(h), __high2half(h)));
            return __fmaf_rn(__fmul_rn(yd, s.e), f, acc);
        }
    }
};

} // namespace p100e::fast
