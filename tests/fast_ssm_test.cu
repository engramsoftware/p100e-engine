// p100e-fast-ssm-test: fast::gdn_* vs the reference path (k::gdn_gates, conv1d_silu, gdn_recurrent,
// rmsnorm_gated) on random data with Qwen3.8-27B shapes (the test model.gguf: ssm.group_count 16,
// time_step_rank 48, state_size 128, inner_size 6144, conv_kernel 4, rms eps 1e-6).
#include "kernels.cuh"
#include "fast/ssm.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("CUDA %s at %d\n", cudaGetErrorString(e_), __LINE__); std::exit(1); } } while (0)

using namespace p100e;

static std::mt19937 rng(1234);
static std::vector<float> rnd(size_t n, float lo, float hi) {
    std::uniform_real_distribution<float> d(lo, hi);
    std::vector<float> v(n);
    for (auto & x : v) x = d(rng);
    return v;
}
static float * up(const std::vector<float> & v) {
    float * p; CK(cudaMalloc(&p, v.size() * 4)); CK(cudaMemcpy(p, v.data(), v.size() * 4, cudaMemcpyHostToDevice)); return p;
}
static float * dalloc(size_t n) { float * p; CK(cudaMalloc(&p, n * 4)); CK(cudaMemset(p, 0, n * 4)); return p; }
static std::vector<float> down(const float * p, size_t n) {
    std::vector<float> v(n); CK(cudaMemcpy(v.data(), p, n * 4, cudaMemcpyDeviceToHost)); return v;
}
struct Err { double abs = 0, ref = 0; };
static void acc(Err & e, const float * r, const float * o, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        const double d = std::fabs((double) r[i] - o[i]);
        if (!(d <= e.abs)) e.abs = std::isnan(d) ? INFINITY : std::max(e.abs, d);
        e.ref = std::max(e.ref, std::fabs((double) r[i]));
    }
}
static void pr(const char * what, const Err & e) {
    std::printf("  %-28s max abs %.3e  max|ref| %.3e  rel %.3e\n", what, e.abs, e.ref, e.ref > 0 ? e.abs / e.ref : 0.0);
}

struct Shape { const char * name; int Hk, Hv; };
static const int D = 128;
static const float EPS = 1e-6f;
static int g_rows = 0;   // fast::gdn_scan rows per block (0 = auto)

struct Case {
    int Hk, Hv, C, Tmax;
    float * x, * a, * b, * z, * cw, * dtb, * ssma, * nw;           // inputs
    float * ref_cs, * ref_ss, * ref_a, * ref_b, * ref_y, * ref_o;   // reference state/scratch
    float * f_cs, * f_ss, * f_y, * f_dec, * f_beta, * f_o;          // fast state/scratch
    std::vector<float> cs0, ss0;
    Case(int hk, int hv, int tmax) : Hk(hk), Hv(hv), C(2 * hk * D + hv * D), Tmax(tmax) {
        x = up(rnd((size_t) Tmax * C, -2.f, 2.f));
        a = up(rnd((size_t) Tmax * Hv, -3.f, 3.f));
        b = up(rnd((size_t) Tmax * Hv, -3.f, 3.f));
        z = up(rnd((size_t) Tmax * Hv * D, -3.f, 3.f));
        cw = up(rnd((size_t) C * 4, -0.6f, 0.6f));
        dtb = up(rnd(Hv, -1.f, 1.f));
        ssma = up(rnd(Hv, -8.f, -0.05f));   // ssm_a = -exp(A_log) < 0
        nw = up(rnd(D, 0.5f, 1.5f));
        cs0 = rnd((size_t) C * 3, -2.f, 2.f);
        ss0 = rnd((size_t) Hv * D * D, -0.2f, 0.2f);
        ref_cs = up(cs0); ref_ss = up(ss0); f_cs = up(cs0); f_ss = up(ss0);
        ref_a = dalloc((size_t) Tmax * Hv); ref_b = dalloc((size_t) Tmax * Hv);
        ref_y = dalloc((size_t) Tmax * C); ref_o = dalloc((size_t) Tmax * Hv * D);
        f_y = dalloc((size_t) Tmax * C); f_dec = dalloc((size_t) Tmax * Hv); f_beta = dalloc((size_t) Tmax * Hv);
        f_o = dalloc((size_t) Tmax * Hv * D);
    }
    void reset() {
        CK(cudaMemcpy(ref_cs, cs0.data(), cs0.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(f_cs, cs0.data(), cs0.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(ref_ss, ss0.data(), ss0.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(f_ss, ss0.data(), ss0.size() * 4, cudaMemcpyHostToDevice));
    }
    // tokens [t0, t0+T) of the inputs
    void run_ref(int t0, int T) {
        CK(cudaMemcpyAsync(ref_a, a + (size_t) t0 * Hv, (size_t) T * Hv * 4, cudaMemcpyDeviceToDevice));
        CK(cudaMemcpyAsync(ref_b, b + (size_t) t0 * Hv, (size_t) T * Hv * 4, cudaMemcpyDeviceToDevice));
        ref_core(t0, T);
    }
    void ref_core(int t0, int T) {   // the 4 reference kernels as model.cu runs them
        k::gdn_gates(ref_a, ref_b, dtb, ssma, T, Hv, 0);
        k::conv1d_silu(x + (size_t) t0 * C, ref_cs, cw, ref_y, T, C, 0);
        k::gdn_recurrent(ref_y, ref_a, ref_b, ref_ss, ref_o, T, Hk, Hv, D, D, EPS, 0);
        k::rmsnorm_gated(ref_o, nw, z + (size_t) t0 * Hv * D, ref_o, (int64_t) T * Hv, D, EPS, 0);
    }
    void run_fast(int t0, int T) {
        fast::GdnWeights w{cw, dtb, ssma, nw};
        if (g_rows == 0) {
            fast::gdn_layer(x + (size_t) t0 * C, a + (size_t) t0 * Hv, b + (size_t) t0 * Hv, z + (size_t) t0 * Hv * D, w,
                            f_cs, f_ss, f_y, f_dec, f_beta, f_o, T, Hk, Hv, D, EPS, 0);
        } else {
            fast::gdn_prep(x + (size_t) t0 * C, f_cs, w, a + (size_t) t0 * Hv, b + (size_t) t0 * Hv, f_y, f_dec, f_beta, T, Hk, Hv, D, EPS, 0);
            fast::gdn_scan(f_y, f_dec, f_beta, z + (size_t) t0 * Hv * D, nw, f_ss, f_o, T, Hk, Hv, D, EPS, 0, g_rows);
        }
    }
    void check(const char * label, int T, int steps) {
        reset();
        Err eo;
        const size_t no = (size_t) T * Hv * D;
        for (int st = 0; st < steps; ++st) {
            run_ref(st * T, T); run_fast(st * T, T);
            CK(cudaGetLastError());
            auto r = down(ref_o, no), o = down(f_o, no);
            acc(eo, r.data(), o.data(), no);
        }
        Err es, ec;
        auto rs = down(ref_ss, (size_t) Hv * D * D), fs = down(f_ss, (size_t) Hv * D * D);
        acc(es, rs.data(), fs.data(), rs.size());
        auto rc = down(ref_cs, (size_t) C * 3), fc = down(f_cs, (size_t) C * 3);
        acc(ec, rc.data(), fc.data(), rc.size());
        std::printf("%s (T=%d x %d steps, scan rows/block %s)\n", label, T, steps, g_rows ? std::to_string(g_rows).c_str() : "auto");
        pr("output (all steps)", eo); pr("ssm state (final)", es); pr("conv state (final)", ec);
    }
    template <class F> float time_us(F f, int reps = 100) {
        for (int i = 0; i < 10; ++i) f();
        cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
        CK(cudaEventRecord(e0));
        for (int i = 0; i < reps; ++i) f();
        CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
        float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
        CK(cudaGetLastError());
        return ms * 1000.f / reps;
    }
    void bench(const char * label, int T) {
        reset();
        // reference gates run in place on ref_a/ref_b (values drift, irrelevant for timing)
        CK(cudaMemcpy(ref_a, a, (size_t) T * Hv * 4, cudaMemcpyDeviceToDevice));
        CK(cudaMemcpy(ref_b, b, (size_t) T * Hv * 4, cudaMemcpyDeviceToDevice));
        float tr = time_us([&] { ref_core(0, T); });
        float tf = time_us([&] { run_fast(0, T); });
        fast::GdnWeights w{cw, dtb, ssma, nw};
        float tp = time_us([&] { fast::gdn_prep(x, f_cs, w, a, b, f_y, f_dec, f_beta, T, Hk, Hv, D, EPS, 0); });
        float ts = time_us([&] { fast::gdn_scan(f_y, f_dec, f_beta, z, nw, f_ss, f_o, T, Hk, Hv, D, EPS, 0); });
        std::printf("  time %-22s T=%-4d ref %9.1f us  fast %8.1f us (prep %.1f + scan %.1f)  speedup %.1fx\n",
                    label, T, tr, tf, tp, ts, tr / tf);
        std::printf("    scan by rows/block:");
        for (int r : {128, 64, 32, 16})
            std::printf("  %d: %.1f", r, time_us([&] { fast::gdn_scan(f_y, f_dec, f_beta, z, nw, f_ss, f_o, T, Hk, Hv, D, EPS, 0, r); }));
        std::printf(" us\n");
    }
};

int main() {
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    std::printf("device: %s (sm_%d%d)\n", p.name, p.major, p.minor);
    const Shape shapes[] = {{"full (Hk16 Hv48)", 16, 48}, {"tp/2 per GPU (Hk8 Hv24)", 8, 24}};
    for (const auto & sh : shapes) {
        std::printf("\n=== %s, D=%d, C=%d ===\n", sh.name, D, 2 * sh.Hk * D + sh.Hv * D);
        Case c(sh.Hk, sh.Hv, 512);
        c.check("decode", 1, 64);
        c.check("decode", 2, 32);
        c.check("decode", 4, 32);
        c.check("prefill", 512, 1);
        c.check("prefill odd", 37, 3);
        for (int r : {64, 32, 16}) { g_rows = r; c.check("decode", 1, 64); c.check("prefill", 512, 1); c.check("prefill odd", 37, 3); }
        g_rows = 0;
        for (int T : {1, 2, 4, 64, 512}) c.bench(sh.name, T);
    }
    return 0;
}
