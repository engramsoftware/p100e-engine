// Correctness + timing of p100e::fast fused small kernels vs the reference kernels in src/kernels.cu.
#include "fast/fused.cuh"
#include "kernels.cuh"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <functional>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(2); } } while (0)

namespace F = p100e::fast;
namespace K = p100e::k;

static int g_fail = 0;
static uint32_t rs = 12345;
static float frand() { rs = rs * 1664525u + 1013904223u; return ((rs >> 8) & 0xffff) / 32768.0f - 1.0f; }

static float * dev_rand(size_t n, float scale, float off = 0.0f) {
    std::vector<float> h(n);
    for (auto & v : h) v = (frand() + frand() + frand()) * scale + off;
    float * d; CK(cudaMalloc(&d, n * 4)); CK(cudaMemcpy(d, h.data(), n * 4, cudaMemcpyHostToDevice));
    return d;
}
static std::vector<float> to_host(const float * d, size_t n) {
    std::vector<float> h(n); CK(cudaMemcpy(h.data(), d, n * 4, cudaMemcpyDeviceToHost)); return h;
}
static void cmp(const char * name, const float * a, const float * b, size_t n, double tol) {
    auto ha = to_host(a, n), hb = to_host(b, n);
    double ma = 0, mr = 0, mx = 0;
    for (size_t i = 0; i < n; ++i) { mx = fmax(mx, fabs(hb[i])); }
    for (size_t i = 0; i < n; ++i) {
        const double e = fabs((double) ha[i] - hb[i]);
        ma = fmax(ma, e); mr = fmax(mr, e / (fabs(hb[i]) + 1e-3 * mx + 1e-20));
    }
    const bool ok = ma <= tol * (mx > 1 ? mx : 1);
    if (!ok) ++g_fail;
    printf("  %-34s max_abs=%.3e max_rel=%.3e (ref max %.2e)  %s\n", name, ma, mr, mx, ok ? "OK" : "FAIL");
}

static cudaStream_t g_s;
// per-call GPU time: 100 calls captured into a CUDA graph (removes host launch overhead), replayed, cudaEvents around the replay
static float time_ms(const std::function<void()> & f, int reps = 100) {
    for (int i = 0; i < 10; ++i) f();
    CK(cudaStreamSynchronize(g_s));
    cudaGraph_t g; cudaGraphExec_t ge;
    CK(cudaStreamBeginCapture(g_s, cudaStreamCaptureModeThreadLocal));
    for (int i = 0; i < reps; ++i) f();
    CK(cudaStreamEndCapture(g_s, &g)); CK(cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0));
    CK(cudaGraphLaunch(ge, g_s)); CK(cudaStreamSynchronize(g_s));
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a, g_s);
    CK(cudaGraphLaunch(ge, g_s));
    cudaEventRecord(b, g_s); CK(cudaEventSynchronize(b));
    float ms; cudaEventElapsedTime(&ms, a, b);
    cudaGraphExecDestroy(ge); cudaGraphDestroy(g);
    return ms / reps;
}

int main() {
    cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, 0));
    printf("device: %s (sm_%d%d)\n", pr.name, pr.major, pr.minor);
    const float eps = 1e-6f;
    cudaStream_t s; CK(cudaStreamCreate(&s)); g_s = s;
    const int ns[] = {5120, 256, 128, 1000, 777, 8192, 17408};
    for (int n : ns) for (int T : {1, 4}) {
        printf("n=%d rows=%d\n", n, T);
        const size_t N = (size_t) n * T;
        float * x = dev_rand(N, 1.0f), * r = dev_rand(N, 0.5f), * w = dev_rand(n, 0.3f, 1.0f), * z = dev_rand(N, 1.5f);
        float * xa, * xb, * ya, * yb; CK(cudaMalloc(&xa, N * 4)); CK(cudaMalloc(&xb, N * 4)); CK(cudaMalloc(&ya, N * 4)); CK(cudaMalloc(&yb, N * 4));
        __half * yh; CK(cudaMalloc(&yh, N * 2));
        // rmsnorm
        K::rmsnorm(x, w, ya, T, n, eps, s); F::rmsnorm(x, w, yb, T, n, eps, s); CK(cudaDeviceSynchronize());
        cmp("rmsnorm", yb, ya, N, 1e-5);
        K::rmsnorm(x, nullptr, ya, T, n, eps, s); F::rmsnorm(x, nullptr, yb, T, n, eps, s); CK(cudaDeviceSynchronize());
        cmp("rmsnorm (w=null)", yb, ya, N, 1e-5);
        // in place
        CK(cudaMemcpy(xb, x, N * 4, cudaMemcpyDeviceToDevice)); F::rmsnorm(xb, w, xb, T, n, eps, s);
        K::rmsnorm(x, w, ya, T, n, eps, s); CK(cudaDeviceSynchronize());
        cmp("rmsnorm in-place", xb, ya, N, 1e-5);
        // add_rmsnorm
        CK(cudaMemcpy(xa, x, N * 4, cudaMemcpyDeviceToDevice)); CK(cudaMemcpy(xb, x, N * 4, cudaMemcpyDeviceToDevice));
        K::add(xa, r, N, s); K::rmsnorm(xa, w, ya, T, n, eps, s);
        F::add_rmsnorm(xb, r, w, yb, T, n, eps, s); CK(cudaDeviceSynchronize());
        cmp("add_rmsnorm y", yb, ya, N, 1e-5); cmp("add_rmsnorm x (residual)", xb, xa, N, 1e-6);
        // with half copy
        CK(cudaMemcpy(xb, x, N * 4, cudaMemcpyDeviceToDevice));
        F::add_rmsnorm_h(xb, r, w, yb, yh, T, n, eps, s); CK(cudaDeviceSynchronize());
        { std::vector<__half> hh(N); CK(cudaMemcpy(hh.data(), yh, N * 2, cudaMemcpyDeviceToHost)); auto hy = to_host(yb, N);
          double m = 0; for (size_t i = 0; i < N; ++i) m = fmax(m, fabs(__half2float(hh[i]) - hy[i]) / (fabs(hy[i]) + 1e-2));
          printf("  %-34s max_rel_vs_f32=%.3e  %s\n", "add_rmsnorm_h half copy", m, m < 2e-3 ? "OK" : "FAIL"); if (m >= 2e-3) ++g_fail; }
        // gated
        K::rmsnorm_gated(x, w, z, ya, T, n, eps, s); F::rmsnorm_gated(x, w, z, yb, T, n, eps, s); CK(cudaDeviceSynchronize());
        cmp("rmsnorm_gated", yb, ya, N, 1e-5);
        // elementwise
        CK(cudaMemcpy(xa, x, N * 4, cudaMemcpyDeviceToDevice)); CK(cudaMemcpy(xb, x, N * 4, cudaMemcpyDeviceToDevice));
        K::silu_mul(xa, r, N, s); F::silu_mul(xb, r, N, s); CK(cudaDeviceSynchronize()); cmp("silu_mul", xb, xa, N, 1e-6);
        CK(cudaMemcpy(xa, x, N * 4, cudaMemcpyDeviceToDevice)); CK(cudaMemcpy(xb, x, N * 4, cudaMemcpyDeviceToDevice));
        K::sigmoid_mul(xa, r, N, s); F::sigmoid_mul(xb, r, N, s); CK(cudaDeviceSynchronize()); cmp("sigmoid_mul", xb, xa, N, 1e-6);
        // unaligned pointers (offset by 1 float), scalar path
        if (T == 1) {
            float * xo, * yo, * yr; CK(cudaMalloc(&xo, (N + 4) * 4)); CK(cudaMalloc(&yo, (N + 4) * 4)); CK(cudaMalloc(&yr, (N + 4) * 4));
            CK(cudaMemcpy(xo + 1, x, N * 4, cudaMemcpyDeviceToDevice));
            F::rmsnorm(xo + 1, w, yo + 1, 1, n, eps, s); K::rmsnorm(x, w, ya, 1, n, eps, s); CK(cudaDeviceSynchronize());
            cmp("rmsnorm unaligned ptrs", yo + 1, ya, N, 1e-5);
            cudaFree(xo); cudaFree(yo); cudaFree(yr);
        }
        cudaFree(x); cudaFree(r); cudaFree(w); cudaFree(z); cudaFree(xa); cudaFree(xb); cudaFree(ya); cudaFree(yb); cudaFree(yh);
    }

    // ---- timing: n=5120, T=1 (average over 100 100 calls in a CUDA graph incl. launch cost) ----
    printf("\ntiming n=5120 T=1, 100 reps after 10 warmup, cudaEvents (ms per call; 100 calls in a CUDA graph)\n");
    const int n = 5120;
    float * x = dev_rand(20000, 1.0f), * r = dev_rand(n, 0.5f), * w = dev_rand(n, 0.3f, 1.0f), * y = dev_rand(20000, 1.0f), * z = dev_rand(20000, 1.0f);
    __half * yh; CK(cudaMalloc(&yh, 20000 * 2));
    auto row = [&](const char * name, float ref, float fast) { printf("  %-28s ref %.4f ms   fast %.4f ms   x%.2f\n", name, ref, fast, ref / fast); };
    row("rmsnorm", time_ms([&] { K::rmsnorm(x, w, y, 1, n, 1e-6f, s); }), time_ms([&] { F::rmsnorm(x, w, y, 1, n, 1e-6f, s); }));
    row("add + rmsnorm (ref 2 kernels)", time_ms([&] { K::add(x, r, n, s); K::rmsnorm(x, w, y, 1, n, 1e-6f, s); }),
        time_ms([&] { F::add_rmsnorm(x, r, w, y, 1, n, 1e-6f, s); }));
    row("add_rmsnorm_h (f32+f16 out)", time_ms([&] { K::add(x, r, n, s); K::rmsnorm(x, w, y, 1, n, 1e-6f, s); }),
        time_ms([&] { F::add_rmsnorm_h(x, r, w, y, yh, 1, n, 1e-6f, s); }));
    row("rmsnorm 256 x 48 rows", time_ms([&] { K::rmsnorm(x, w, y, 48, 256, 1e-6f, s); }), time_ms([&] { F::rmsnorm(x, w, y, 48, 256, 1e-6f, s); }));
    row("rmsnorm 128 x 64 rows", time_ms([&] { K::rmsnorm(x, w, y, 64, 128, 1e-6f, s); }), time_ms([&] { F::rmsnorm(x, w, y, 64, 128, 1e-6f, s); }));
    row("rmsnorm_gated 128 x 48 rows", time_ms([&] { K::rmsnorm_gated(x, w, z, y, 48, 128, 1e-6f, s); }), time_ms([&] { F::rmsnorm_gated(x, w, z, y, 48, 128, 1e-6f, s); }));
    row("silu_mul 17408", time_ms([&] { K::silu_mul(y, z, 17408, s); }), time_ms([&] { F::silu_mul(y, z, 17408, s); }));
    row("sigmoid_mul 6144", time_ms([&] { K::sigmoid_mul(y, z, 6144, s); }), time_ms([&] { F::sigmoid_mul(y, z, 6144, s); }));
    printf("\n%s\n", g_fail ? "SOME CHECKS FAILED" : "ALL CHECKS PASSED");
    return g_fail ? 1 : 0;
}
