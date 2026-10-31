// p100e-fast-wire-test: the f16 prefill wire's saturation guard (fast::fold_gemm with fold_opts::out16 + sat)
// and fast::abs_max (the P100E_PF_WIRE_STATS reduction). Any GPU: the out16 launches always run the
// compiled kernel (the SASS cubin only covers f32 outputs), so the GTX 1660 exercises the same code as the P100.
//  1. boundary cases, every kernel variant (T 10 / 40 / 130 -> JN 1 / 2 / 8): outputs 65504 (exact), 65512 (rounds to
//     65504, not counted), 65520 (the first value that would round to inf: clamped, counted), 131072, their negatives,
//     and small values; *sat counts the (token, 4-row group)s with a clamped value.
//  2. random problems with per-token scales up to 2^12 (many partials > 65504): every half equals the f32 output of the
//     same launch rounded to nearest even, clamped to +-65504 where |v| >= 65520; no inf / NaN; *sat as above; also a
//     launch with sat = null.
//  3. abs_max over the f32 and the f16 outputs (and a 1M-element array, accumulated over two calls) equals the host max.
#include "fast/gemm.cuh"

#include <cuda_fp16.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using namespace p100e;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

static int fails = 0;
static void check(bool ok, const char * what) {
    if (!ok) { ++fails; printf("  FAIL: %s\n", what); }
}

template <typename T> static T * dev_copy(const std::vector<T> & h) {
    T * d;
    CK(cudaMalloc(&d, h.size() * sizeof(T)));
    CK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}
template <typename T> static std::vector<T> host_copy(const T * d, size_t n) {
    std::vector<T> h(n);
    CK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
}
static uint16_t hbits(half h) { uint16_t u; memcpy(&u, &h, 2); return u; }

// the wire's rounding: f16 round-to-nearest-even, finite |v| >= 65520 saturated to +-65504
static half wire_round(float v) {
    if (std::isfinite(v) && std::fabs(v) >= 65520.0f) v = std::copysign(65504.0f, v);
    return __float2half_rn(v);
}

struct problem {
    int K = 0, ldx = 0, T = 0;
    std::vector<half> x16;           // [T][ldx]
    std::vector<float> cs;           // [T]
    std::vector<std::vector<half>> w;   // per segment [rows][K]
    std::vector<int> rows;
};

// runs the problem with f32 outputs and with f16 outputs; checks every half against the f32 value and the count
static int run(const problem & P, const char * tag, bool with_null_sat = false) {
    const int ns = (int) P.w.size();
    half * x16 = dev_copy(P.x16);
    float * cs = dev_copy(P.cs);
    std::vector<half *> wd(ns);
    std::vector<float *> y32(ns);
    std::vector<half *> y16(ns);
    for (int i = 0; i < ns; ++i) {
        wd[i] = dev_copy(P.w[i]);
        CK(cudaMalloc(&y32[i], (size_t) P.T * P.rows[i] * 4));
        CK(cudaMalloc(&y16[i], (size_t) P.T * P.rows[i] * 2));
        CK(cudaMemset(y16[i], 0xff, (size_t) P.T * P.rows[i] * 2));   // NaN pattern: unwritten values show up
    }
    int * sat;
    CK(cudaMalloc(&sat, sizeof(int)));
    CK(cudaMemset(sat, 0, sizeof(int)));
    fast::fold_seg seg[4];
    for (int i = 0; i < ns; ++i) { seg[i].w = wd[i]; seg[i].rows = P.rows[i]; seg[i].ldy = P.rows[i]; seg[i].y = y32[i]; }
    fast::fold_opts o;
    fast::fold_gemm(seg, ns, x16, P.ldx, cs, P.T, P.K, 0, o);
    for (int i = 0; i < ns; ++i) { seg[i].y = nullptr; seg[i].y16 = y16[i]; }
    o.out16 = true;
    o.sat = with_null_sat ? nullptr : sat;
    fast::fold_gemm(seg, ns, x16, P.ldx, cs, P.T, P.K, 0, o);
    CK(cudaDeviceSynchronize());
    const int got_sat = host_copy(sat, 1)[0];

    long long bad = 0, nonfinite = 0, groups = 0, clamped = 0, total = 0;
    float amax32 = 0.0f, amax16 = 0.0f;
    for (int i = 0; i < ns; ++i) {
        const std::vector<float> f = host_copy(y32[i], (size_t) P.T * P.rows[i]);
        const std::vector<half> h = host_copy(y16[i], (size_t) P.T * P.rows[i]);
        for (int n = 0; n < P.T; ++n)
            for (int m0 = 0; m0 < P.rows[i]; m0 += 4) {
                bool g = false;
                for (int m = m0; m < m0 + 4; ++m) {
                    const size_t j = (size_t) n * P.rows[i] + m;
                    const float v = f[j];
                    const bool c = std::isfinite(v) && std::fabs(v) >= 65520.0f;
                    g |= c;
                    clamped += c;
                    ++total;
                    if (hbits(h[j]) != hbits(wire_round(v))) {
                        if (bad < 5) printf("  %s seg %d n %d m %d: f32 %.9g -> half %.9g, expected %.9g\n", tag, i, n, m, v,
                                            __half2float(h[j]), __half2float(wire_round(v)));
                        ++bad;
                    }
                    const float hv = __half2float(h[j]);
                    if (!std::isfinite(hv)) ++nonfinite;
                    amax32 = std::max(amax32, std::fabs(v));
                    amax16 = std::max(amax16, std::fabs(hv));
                }
                groups += g;
            }
    }
    // abs_max over the outputs (all segments into one slot each)
    float * am;
    CK(cudaMalloc(&am, 2 * sizeof(float)));
    CK(cudaMemset(am, 0, 2 * sizeof(float)));
    for (int i = 0; i < ns; ++i) {
        fast::abs_max(y32[i], false, (size_t) P.T * P.rows[i], am, 0);
        fast::abs_max(y16[i], true, (size_t) P.T * P.rows[i], am + 1, 0);
    }
    const std::vector<float> amh = host_copy(am, 2);
    printf("%-26s T %4d: %lld values, %lld > f16 range (%lld groups), sat %d, %lld mismatches, %lld non-finite halves; "
           "abs_max f32 %.7g (host %.7g) f16 %.7g (host %.7g)\n", tag, P.T, total, clamped, groups,
           with_null_sat ? -1 : got_sat, bad, nonfinite, amh[0], amax32, amh[1], amax16);
    check(bad == 0, "every half = the f32 output rounded / clamped");
    check(nonfinite == 0, "no inf / NaN on the wire");
    check(with_null_sat ? got_sat == 0 : got_sat == groups, "*sat = number of (token, 4-row group)s with a clamped value");
    check(amh[0] == amax32 && amh[1] == amax16, "abs_max = host max");
    for (int i = 0; i < ns; ++i) { cudaFree(wd[i]); cudaFree(y32[i]); cudaFree(y16[i]); }
    cudaFree(x16); cudaFree(cs); cudaFree(sat); cudaFree(am);
    return got_sat;
}

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    printf("device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);

    // 1. boundaries: x16[n][0] = 8, else 0; w[m][0] = a_m, else 0 -> v = 8 a_m cs[n], exact in f16 chains and in f32
    {
        const int K = 64;
        const float a[8] = {1024, -1024, 64, 0, 64, -64, 3, -3};   // group 0 saturates for cs >= 65520 / 8192
        const float scales[6] = {1.0f, 7.99609375f /*65504*/, 7.9970703125f /*65512*/, 7.998046875f /*65520*/, 16.0f, 0.5f};
        for (int T : {10, 40, 130}) {
            problem P;
            P.K = K; P.ldx = K; P.T = T;
            P.x16.assign((size_t) T * K, __float2half(0.0f));
            P.cs.resize(T);
            for (int n = 0; n < T; ++n) { P.x16[(size_t) n * K] = __float2half(8.0f); P.cs[n] = scales[n % 6]; }
            P.rows = {8};
            P.w.assign(1, std::vector<half>((size_t) 8 * K, __float2half(0.0f)));
            for (int m = 0; m < 8; ++m) P.w[0][(size_t) m * K] = __float2half(a[m]);
            const int got = run(P, "boundaries");
            // by hand: the tokens with scale 65520 / 8192 or 16 saturate group 0 (rows 0-3), nothing else does
            int want = 0;
            for (int n = 0; n < T; ++n) want += (n % 6 == 3 || n % 6 == 4);
            printf("  by hand: %d saturated groups\n", want);
            check(got == want, "boundary count = hand count (65512 not counted, 65520 counted)");
        }
    }
    // 2. random: two segments (one with a partial row tile), per-token scales 2^(n % 13)
    {
        std::mt19937 rng(11);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        std::uniform_real_distribution<float> ud(-16.0f, 16.0f);
        const int K = 256, LDX = K + 64;
        for (int T : {10, 40, 300}) {
            problem P;
            P.K = K; P.ldx = LDX; P.T = T;
            P.x16.resize((size_t) T * LDX);
            for (auto & v : P.x16) v = __float2half(ud(rng));
            P.cs.resize(T);
            for (int n = 0; n < T; ++n) P.cs[n] = std::ldexp(1.0f, n % 13);
            P.rows = {136, 64};
            for (int r : P.rows) {
                std::vector<half> w((size_t) r * K);
                for (auto & v : w) v = __float2half(nd(rng));
                P.w.push_back(w);
            }
            run(P, "random");
            if (T == 300) run(P, "random, sat = null", true);
        }
    }
    // 3. abs_max on a large array, accumulated over two calls (f32 and f16)
    {
        std::mt19937 rng(5);
        std::normal_distribution<float> nd(0.0f, 100.0f);
        const size_t n = 1000003;
        std::vector<float> a(n), b(n);
        std::vector<half> ah(n);
        for (auto & v : a) v = nd(rng);
        for (auto & v : b) v = nd(rng);
        a[777777] = -54321.5f;   // the maximum, negative
        float hm = 0, hh = 0;
        for (size_t i = 0; i < n; ++i) {
            hm = std::max(hm, std::max(std::fabs(a[i]), std::fabs(b[i])));
            ah[i] = __float2half_rn(a[i]);
            hh = std::max(hh, std::fabs(__half2float(ah[i])));
        }
        float * da = dev_copy(a);
        float * db = dev_copy(b);
        half * dh = dev_copy(ah);
        float * am;
        CK(cudaMalloc(&am, 2 * sizeof(float)));
        CK(cudaMemset(am, 0, 2 * sizeof(float)));
        fast::abs_max(da, false, n, am, 0);
        fast::abs_max(db, false, n, am, 0);
        fast::abs_max(dh, true, n, am + 1, 0);
        const std::vector<float> r = host_copy(am, 2);
        printf("abs_max 1M: f32 %.7g (host %.7g), f16 %.7g (host %.7g)\n", r[0], hm, r[1], hh);
        check(r[0] == hm && r[1] == hh, "abs_max (1M elements, two calls) = host max");
        cudaFree(da); cudaFree(db); cudaFree(dh); cudaFree(am);
    }
    printf(fails ? "FAILED (%d failing checks)\n" : "ALL PASSED (%d failing checks)\n", fails);
    return fails ? 1 : 0;
}
