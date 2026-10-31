// p100e-mrope-test: interleaved M-RoPE (images) on one GPU.
//   1. text rows (t = h = w = p): k::rope_mrope == k::rope_neox and attn_prep(M-RoPE) == attn_prep(plain), bit for bit
//      (q, gate, K / V cache bytes; f16, q8_0, q4_0 caches)
//   2. image rows (t = P, h = P + i / nx, w = P + i % nx): attn_prep(M-RoPE) == split + rmsnorm + rope_mrope + kv_append
//      bit for bit (the fused decode path vs the per-op / prefill path)
//   3. rope_mrope vs a double-precision CPU port of ggml-cuda's rope_multi (is_imrope, sections [11, 11, 10, 0], NEOX
//      pairing, theta_scale = base^(-2 / n_dims)): max |difference| (float rounding only)
#include "fast/fused.cuh"
#include "fast/attn.cuh"
#include "fast/mrope.cuh"
#include "kernels.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(2); } } while (0)

namespace F = p100e::fast;
namespace K = p100e::k;

static int g_fail = 0, g_checks = 0;
static uint32_t rs = 1234567u;
static float frand() { rs ^= rs << 13; rs ^= rs >> 17; rs ^= rs << 5; return (rs >> 8) / 8388608.0f - 1.0f; }
static cudaStream_t g_s;

static float * dev_rand(size_t n, float scale, float off = 0.0f) {
    std::vector<float> h(n);
    for (auto & v : h) v = (frand() + frand()) * scale + off;
    float * d; CK(cudaMalloc(&d, n * 4)); CK(cudaMemcpy(d, h.data(), n * 4, cudaMemcpyHostToDevice));
    CK(cudaDeviceSynchronize());
    return d;
}
static void * dev_zero(size_t bytes) { void * d; CK(cudaMalloc(&d, bytes)); CK(cudaMemset(d, 0, bytes)); CK(cudaDeviceSynchronize()); return d; }
static std::vector<uint8_t> host(const void * d, size_t n) { std::vector<uint8_t> h(n); CK(cudaMemcpy(h.data(), d, n, cudaMemcpyDeviceToHost)); return h; }
static void same(const char * name, const void * a, const void * b, size_t bytes) {
    CK(cudaDeviceSynchronize());
    auto ha = host(a, bytes), hb = host(b, bytes);
    size_t diff = 0;
    for (size_t i = 0; i < bytes; ++i) diff += ha[i] != hb[i];
    ++g_checks;
    if (diff) ++g_fail;
    printf("  %-52s %8zu bytes  %s\n", name, bytes, diff ? "DIFF  FAIL" : "identical");
}

struct grid { int nx, ny, P; };   // image rows i0.. of an nx x ny grid at rotary position P; nx = 0: text rows at P + t

// positions [3][stride] for T rows
static std::vector<int32_t> positions(const grid & g, int T, int i0, int stride) {
    std::vector<int32_t> p(3 * stride, 0);
    for (int t = 0; t < T; ++t) {
        int a, b, c;
        if (g.nx) { const int k = i0 + t; a = g.P; b = g.P + k / g.nx; c = g.P + k % g.nx; }
        else a = b = c = g.P + t;
        p[t] = a; p[stride + t] = b; p[2 * stride + t] = c;
    }
    return p;
}

static void test_prep(const grid & g, int T, uint32_t kvt) {
    const int Hq = 12, Hkv = 2, D = 256, n_rot = 64, n_ctx = 512, stride = 64;
    const float base = 1e7f, eps = 1e-6f;
    const int row0 = 300;   // KV rows (differ from the rotary positions, as after an image)
    printf("%s rows (P=%d nx=%d ny=%d T=%d), %s cache\n", g.nx ? "image" : "text", g.P, g.nx, g.ny, T,
           kvt == F::KV_Q8_0 ? "q8_0" : kvt == F::KV_Q4_0 ? "q4_0" : "f16");
    float * qg = dev_rand((size_t) T * Hq * 2 * D, 1.0f), * k = dev_rand((size_t) T * Hkv * D, 1.0f), * v = dev_rand((size_t) T * Hkv * D, 1.0f);
    float * qn = dev_rand(D, 0.2f, 1.0f), * kn = dev_rand(D, 0.2f, 1.0f);
    std::vector<int32_t> rows(T), hp = positions(g, T, 7, stride);
    for (int t = 0; t < T; ++t) rows[t] = row0 + t;
    int32_t * dpos, * drp, * dtxt;
    CK(cudaMalloc(&dpos, T * 4)); CK(cudaMemcpy(dpos, rows.data(), T * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&drp, 3 * stride * 4)); CK(cudaMemcpy(drp, hp.data(), 3 * stride * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dtxt, T * 4)); CK(cudaMemcpy(dtxt, hp.data(), T * 4, cudaMemcpyHostToDevice));   // the t stream
    F::mrope_pos mr; mr.p = drp; mr.stride = stride; mr.s0 = 11; mr.s1 = 11; mr.s2 = 10; mr.sect = 32;
    const size_t cb = F::kv_row_bytes(kvt, Hkv, D) * n_ctx, qb = (size_t) T * Hq * D * 4;
    F::kv_cache c0 = F::kv_alloc(kvt, n_ctx, Hkv, D), c1 = F::kv_alloc(kvt, n_ctx, Hkv, D);
    for (void * p : {c0.k, c0.v, c1.k, c1.v}) CK(cudaMemset(p, 0, cb));
    float * q0 = (float *) dev_zero(qb), * g0 = (float *) dev_zero(qb), * q1 = (float *) dev_zero(qb), * g1 = (float *) dev_zero(qb);
    float * k0 = (float *) dev_zero((size_t) T * Hkv * D * 4);
    // fused, M-RoPE
    if (!F::attn_prep(qg, k, v, qn, kn, dpos, q1, g1, c1, -1, T, Hq, Hkv, D, n_rot, base, eps, g_s, 0, 0, 0, mr)) {
        printf("  attn_prep(M-RoPE) refused  FAIL\n"); ++g_fail; return;
    }
    if (!g.nx) {
        // text rows: the plain kernel with the t positions as its rotary positions (cache rows: another cache, p0 = row0)
        if (!F::attn_prep(qg, k, v, qn, kn, dtxt, q0, g0, c0, row0, T, Hq, Hkv, D, n_rot, base, eps, g_s)) { printf("  refused FAIL\n"); ++g_fail; }
        same("q: attn_prep M-RoPE (t=h=w) vs plain", q0, q1, qb);
        same("gate", g0, g1, qb);
        same("K cache", c0.k, c1.k, cb);
        same("V cache", c0.v, c1.v, cb);
        // per-op kernels: rope_mrope vs rope_neox
        float * a = dev_rand((size_t) T * Hq * D, 1.0f);
        float * b; CK(cudaMalloc(&b, (size_t) T * Hq * D * 4)); CK(cudaMemcpy(b, a, (size_t) T * Hq * D * 4, cudaMemcpyDeviceToDevice));
        K::rope_neox(a, dtxt, T, Hq, D, n_rot, base, g_s);
        K::rope_mrope(b, mr, T, Hq, D, n_rot, base, g_s);
        same("k::rope_mrope (t=h=w) vs k::rope_neox", a, b, (size_t) T * Hq * D * 4);
        cudaFree(a); cudaFree(b);
    } else {
        // image rows: per-op path (prefill / layer()) = split + rmsnorm + rope_mrope + kv_append
        K::split_q_gate(qg, q0, g0, T, Hq, D, g_s);
        F::rmsnorm(q0, qn, q0, (int64_t) T * Hq, D, eps, g_s);
        CK(cudaMemcpyAsync(k0, k, (size_t) T * Hkv * D * 4, cudaMemcpyDeviceToDevice, g_s));
        F::rmsnorm(k0, kn, k0, (int64_t) T * Hkv, D, eps, g_s);
        K::rope_mrope(q0, mr, T, Hq, D, n_rot, base, g_s);
        K::rope_mrope(k0, mr, T, Hkv, D, n_rot, base, g_s);
        F::kv_append(c0, k0, v, row0, T, g_s);
        same("q: attn_prep M-RoPE vs split+norm+rope_mrope", q0, q1, qb);
        same("gate", g0, g1, qb);
        same("K cache", c0.k, c1.k, cb);
        same("V cache", c0.v, c1.v, cb);
    }
    F::kv_free(c0); F::kv_free(c1);
    for (void * p : {(void *) qg, (void *) k, (void *) v, (void *) qn, (void *) kn, (void *) dpos, (void *) drp, (void *) dtxt,
                     (void *) q0, (void *) g0, (void *) q1, (void *) g1, (void *) k0}) cudaFree(p);
}

// ggml-cuda rope_multi (is_imrope, forward, no freq factors / YaRN: ext_factor 0, freq_scale 1, attn_factor 1), in double
static void ref_imrope(std::vector<double> & x, const std::vector<int32_t> & pos, int stride, int T, int H, int D, int n_dims,
                       double base, const int sections[4]) {
    const int sect_dims = sections[0] + sections[1] + sections[2] + sections[3];
    const double theta_scale = std::pow(base, -2.0 / n_dims);
    for (int t = 0; t < T; ++t)
        for (int h = 0; h < H; ++h)
            for (int i0 = 0; i0 < n_dims; i0 += 2) {
                const int sector = (i0 / 2) % sect_dims;
                double p;
                if (sector % 3 == 1 && sector < 3 * sections[1]) p = pos[stride + t];
                else if (sector % 3 == 2 && sector < 3 * sections[2]) p = pos[2 * stride + t];
                else if (sector % 3 == 0 && sector < 3 * sections[0]) p = pos[t];
                else p = 0;
                const double theta = p * std::pow(theta_scale, i0 / 2.0);
                double * r = &x[((size_t) t * H + h) * D];
                const double a = r[i0 / 2], b = r[i0 / 2 + n_dims / 2];
                r[i0 / 2] = a * std::cos(theta) - b * std::sin(theta);
                r[i0 / 2 + n_dims / 2] = a * std::sin(theta) + b * std::cos(theta);
            }
}

static void test_ref(const grid & g, int T) {
    const int H = 12, D = 256, n_rot = 64, stride = 64;
    const int sections[4] = {11, 11, 10, 0};
    printf("rope_mrope vs CPU ggml imrope reference (P=%d nx=%d ny=%d T=%d)\n", g.P, g.nx, g.ny, T);
    std::vector<int32_t> hp = positions(g, T, 11, stride);
    int32_t * drp; CK(cudaMalloc(&drp, 3 * stride * 4)); CK(cudaMemcpy(drp, hp.data(), 3 * stride * 4, cudaMemcpyHostToDevice));
    F::mrope_pos mr; mr.p = drp; mr.stride = stride; mr.s0 = 11; mr.s1 = 11; mr.s2 = 10; mr.sect = 32;
    const size_t n = (size_t) T * H * D;
    float * x = dev_rand(n, 1.0f);
    std::vector<float> hx(n);
    CK(cudaMemcpy(hx.data(), x, n * 4, cudaMemcpyDeviceToHost));
    std::vector<double> ref(hx.begin(), hx.end());
    ref_imrope(ref, hp, stride, T, H, D, n_rot, 1e7, sections);
    K::rope_mrope(x, mr, T, H, D, n_rot, 1e7f, g_s);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(hx.data(), x, n * 4, cudaMemcpyDeviceToHost));
    double mx = 0, mref = 0;
    for (size_t i = 0; i < n; ++i) { mx = std::max(mx, std::fabs(hx[i] - ref[i])); mref = std::max(mref, std::fabs(ref[i])); }
    ++g_checks;
    // float theta = pos * freq (ggml and k::rope_neox alike) carries ~1.2e-7 relative error, i.e. ~0.03 rad at 250K
    int maxp = 0;
    for (int v : hp) maxp = std::max(maxp, v);
    const double tol = 2e-4 + 1.2e-7 * maxp;
    const bool ok = mx < tol * std::max(1.0, mref);
    if (!ok) ++g_fail;
    printf("  max |gpu - ref| = %.3g (max |ref| %.3g, tolerance %.3g for positions <= %d)  %s\n", mx, mref, tol, maxp,
           ok ? "ok" : "FAIL");
    cudaFree(drp); cudaFree(x);
}

int main() {
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    printf("device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    CK(cudaStreamCreateWithFlags(&g_s, cudaStreamNonBlocking));
    for (uint32_t kvt : {F::KV_F16, F::KV_Q8_0, F::KV_Q4_0}) {
        test_prep(grid{0, 0, 1234}, 1, kvt);
        test_prep(grid{0, 0, 98765}, 4, kvt);
        test_prep(grid{60, 34, 2000}, 4, kvt);
        test_prep(grid{49, 37, 15}, 3, kvt);
    }
    test_ref(grid{60, 34, 2000}, 40);
    test_ref(grid{37, 28, 5}, 64);
    test_ref(grid{0, 0, 250000}, 8);
    printf("%s: %d checks, %d failed\n", g_fail ? "SOME CHECKS FAILED" : "ALL CHECKS PASSED", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
