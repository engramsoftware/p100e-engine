// p100e-fast-fusion-test: the decode fusions against the per-op path they replace. Every check is
// BIT-EXACT (memcmp of f32 outputs, quantized activations, KV cache bytes), plus GPU time per call of both forms.
//   add2_rmsnorm_q      vs k::add x2 + fast::rmsnorm + fast::matvec_quantize
//   ew_quant            vs k::silu_mul / k::sigmoid_mul + fast::matvec_quantize
//   gdn_layer(qbuf)     vs gdn_layer + fast::matvec_quantize
//   attn_prep           vs k::split_q_gate + fast::rmsnorm (q, k) + k::rope_neox (q, k) + fast::kv_append
//   matvec_prequantized vs fast::matvec (q4_0, q4_K, q3_K, q5_K, q6_K, q8_0 random weights)
//   concatenated weights (one launch over A||B) vs separate launches
//   T-invariance: column t of a T-column matvec vs the T = 1 matvec of column t (warp-rows layouts and q8r), for every
//   multi-column kernel geometry (wrcp, wrm staged / K-phased / hybrid / unstaged, q8_wrt, q8_wrm)
//   matvec_prequantized_parts over A||B (per-part [T][n] outputs) vs separate launches on A and B
#include "fast/fused.cuh"
#include "fast/matvec.cuh"
#include "fast/matvec_internal.cuh"
#include "fast/ssm.cuh"
#include "fast/attn.cuh"
#include "kernels.cuh"
#include "gguf.h"

#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(2); } } while (0)

namespace F = p100e::fast;
namespace K = p100e::k;

static int g_fail = 0, g_checks = 0;
static uint32_t rs = 987654321u;
static uint32_t urand() { rs ^= rs << 13; rs ^= rs >> 17; rs ^= rs << 5; return rs; }
static float frand() { return (urand() >> 8) / 8388608.0f - 1.0f; }   // [-1, 1)
static cudaStream_t g_s;

static float * dev_rand(size_t n, float scale, float off = 0.0f) {
    std::vector<float> h(n);
    for (auto & v : h) v = (frand() + frand() + frand()) * scale + off;
    for (size_t i = 0; i < n; i += 97) h[i] *= 8.0f;   // a few outliers per group, like real activations
    float * d; CK(cudaMalloc(&d, n * 4)); CK(cudaMemcpy(d, h.data(), n * 4, cudaMemcpyHostToDevice));
    CK(cudaDeviceSynchronize());   // pageable H2D may return before the DMA lands; the kernels run on a non-blocking stream
    return d;
}
static void * dev_alloc(size_t bytes) { void * d; CK(cudaMalloc(&d, bytes)); CK(cudaMemset(d, 0, bytes)); CK(cudaDeviceSynchronize()); return d; }
static std::vector<uint8_t> bytes_of(const void * d, size_t n) {
    std::vector<uint8_t> h(n); CK(cudaMemcpy(h.data(), d, n, cudaMemcpyDeviceToHost)); return h;
}
static void same(const char * name, const void * a, const void * b, size_t bytes) {
    CK(cudaDeviceSynchronize());
    auto ha = bytes_of(a, bytes), hb = bytes_of(b, bytes);
    size_t diff = 0, first = (size_t) -1;
    for (size_t i = 0; i < bytes; ++i) if (ha[i] != hb[i]) { ++diff; if (first == (size_t) -1) first = i; }
    ++g_checks;
    if (diff) ++g_fail;
    if (diff) printf("  %-58s %8zu bytes  DIFF %zu bytes (first at %zu)  FAIL\n", name, bytes, diff, first);
    else      printf("  %-58s %8zu bytes  identical\n", name, bytes);
}
// D2D cudaMemcpy is async w.r.t. the host and the legacy stream does not order with the non-blocking test stream
static void copy_dev(void * dst, const void * src, size_t bytes) {
    CK(cudaDeviceSynchronize()); CK(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice)); CK(cudaDeviceSynchronize());
}

// per-call GPU time: reps calls captured into a CUDA graph, replayed (no host launch overhead in the number)
static float time_us(const std::function<void()> & f, int reps = 200) {
    for (int i = 0; i < 5; ++i) f();
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
    return ms * 1000.0f / reps;
}

// ---------------------------------------------------------------------------------------------------------------
static void test_add2_rmsnorm(int n, int T) {
    printf("add2_rmsnorm_q n=%d T=%d\n", n, T);
    const size_t N = (size_t) n * T, qb = F::matvec_scratch_bytes(n, T);
    float * x0 = dev_rand(N, 1.0f), * r1 = dev_rand(N, 0.3f), * r2 = dev_rand(N, 0.3f), * w = dev_rand(n, 0.2f, 1.0f);
    float * xa = (float *) dev_alloc(N * 4), * xb = (float *) dev_alloc(N * 4), * ya = (float *) dev_alloc(N * 4), * yb = (float *) dev_alloc(N * 4);
    void * qa = dev_alloc(qb), * qbuf = dev_alloc(qb);
    const float eps = 1e-6f;
    for (int variant = 0; variant < 3; ++variant) {   // 2 partials (TP=2), 1 partial (TP=1), none (layer 0)
        const float * p1 = variant < 2 ? r1 : nullptr, * p2 = variant == 0 ? r2 : nullptr;
        copy_dev(xa, x0, N * 4); copy_dev(xb, x0, N * 4);
        if (p1) K::add(xa, p1, (int64_t) N, g_s);
        if (p2) K::add(xa, p2, (int64_t) N, g_s);
        F::rmsnorm(xa, w, ya, T, n, eps, g_s);
        F::matvec_quantize(ya, n, T, qa, g_s);
        const bool q = F::add2_rmsnorm_q(xb, p1, p2, w, yb, qbuf, T, n, eps, g_s);
        if (!q) { printf("  add2_rmsnorm_q did not quantize  FAIL\n"); ++g_fail; }
        char nm[96];
        snprintf(nm, sizeof nm, "x (residual) [%s]", variant == 0 ? "2 partials" : variant == 1 ? "1 partial" : "no add");
        same(nm, xa, xb, N * 4);
        snprintf(nm, sizeof nm, "y (normed) [%s]", variant == 0 ? "2 partials" : variant == 1 ? "1 partial" : "no add");
        same(nm, ya, yb, N * 4);
        snprintf(nm, sizeof nm, "a16 activation [%s]", variant == 0 ? "2 partials" : variant == 1 ? "1 partial" : "no add");
        same(nm, qa, qbuf, qb);
    }
    {   // P2P-allreduce variant (single GPU here: the "peer" partial is local; checks the spin + cache-volatile path)
        int * flag; CK(cudaMalloc(&flag, 256)); CK(cudaMemset(flag, 0, 256));
        copy_dev(xa, x0, N * 4); copy_dev(xb, x0, N * 4);
        F::add2_rmsnorm_q(xa, r1, r2, w, ya, qa, T, n, eps, g_s);
        F::p2p_signal(flag, 7, g_s);
        if (!F::add2_rmsnorm_q_wait(xb, r1, r2, flag, 7, w, yb, qbuf, T, n, eps, g_s)) { printf("  _wait refused  FAIL\n"); ++g_fail; }
        same("x (residual) [P2P wait variant]", xa, xb, N * 4);
        same("y (normed) [P2P wait variant]", ya, yb, N * 4);
        same("a16 activation [P2P wait variant]", qa, qbuf, qb);
        const float t_wait = time_us([&] { F::p2p_signal(flag, 9, g_s); F::add2_rmsnorm_q_wait(xb, r1, r2, flag, 9, w, yb, qbuf, T, n, eps, g_s); });
        printf("  time: p2p_signal + add2_rmsnorm_q_wait %.2f us (2 kernels, local flag)\n", t_wait);
        cudaFree(flag);
    }
    {   // device-counter variant (CUDA-graph replays): seq = ++flag[2] on the device; captured once, replayed 3 times
        int * flag; CK(cudaMalloc(&flag, 256)); CK(cudaMemset(flag, 0, 256));
        copy_dev(xa, x0, N * 4);
        F::add2_rmsnorm_q(xa, r1, r2, w, ya, qa, T, n, eps, g_s);
        bool ok = true;
        copy_dev(xb, x0, N * 4);
        F::p2p_signal_ctr(flag, flag + 2, g_s);
        ok = ok && F::add2_rmsnorm_q_wait_ctr(xb, r1, r2, flag, flag + 2, w, yb, qbuf, T, n, eps, g_s);
        same("x (residual) [P2P counter variant]", xa, xb, N * 4);
        same("y (normed) [P2P counter variant]", ya, yb, N * 4);
        same("a16 activation [P2P counter variant]", qa, qbuf, qb);
        cudaGraph_t g = nullptr; cudaGraphExec_t ge = nullptr;
        CK(cudaStreamBeginCapture(g_s, cudaStreamCaptureModeThreadLocal));
        CK(cudaMemcpyAsync(xb, x0, N * 4, cudaMemcpyDeviceToDevice, g_s));
        F::p2p_signal_ctr(flag, flag + 2, g_s);
        ok = ok && F::add2_rmsnorm_q_wait_ctr(xb, r1, r2, flag, flag + 2, w, yb, qbuf, T, n, eps, g_s);
        CK(cudaStreamEndCapture(g_s, &g));
        CK(cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0));
        for (int i = 0; i < 3; ++i) {
            CK(cudaMemsetAsync(yb, 0, N * 4, g_s));
            CK(cudaGraphLaunch(ge, g_s));
            CK(cudaStreamSynchronize(g_s));
            same("y (normed) [P2P counter variant, graph replay]", ya, yb, N * 4);
        }
        int h[3] = {0, 0, 0};
        CK(cudaMemcpy(h, flag, sizeof h, cudaMemcpyDeviceToHost));
        const bool cnt_ok = ok && h[0] == 4 && h[2] == 4 && h[1] == 0;
        printf("  counter after 1 eager + 3 replays: flag %d, own %d, timeouts %d  %s\n", h[0], h[2], h[1], cnt_ok ? "OK" : "FAIL");
        if (!cnt_ok) ++g_fail;
        cudaGraphExecDestroy(ge); cudaGraphDestroy(g);
        cudaFree(flag);
    }
    const float t_old = time_us([&] { K::add(xa, r1, (int64_t) N, g_s); K::add(xa, r2, (int64_t) N, g_s);
                                      F::rmsnorm(xa, w, ya, T, n, eps, g_s); F::matvec_quantize(ya, n, T, qa, g_s); });
    const float t_new = time_us([&] { F::add2_rmsnorm_q(xb, r1, r2, w, yb, qbuf, T, n, eps, g_s); });
    printf("  time: add+add+rmsnorm+quantize %.2f us (4 kernels)  ->  add2_rmsnorm_q %.2f us (1 kernel)\n", t_old, t_new);
    for (void * p : {(void *) x0, (void *) r1, (void *) r2, (void *) w, (void *) xa, (void *) xb, (void *) ya, (void *) yb, qa, qbuf}) cudaFree(p);
}

static void test_ew(F::ew_op op, int n, int T) {
    printf("ew_quant %s n=%d T=%d\n", op == F::EW_SILU_MUL ? "silu_mul" : "sigmoid_mul", n, T);
    const size_t N = (size_t) n * T, qb = F::matvec_scratch_bytes(n, T);
    float * a0 = dev_rand(N, 2.0f), * b = dev_rand(N, 2.0f);
    float * aa = (float *) dev_alloc(N * 4), * ab = (float *) dev_alloc(N * 4);
    void * qa = dev_alloc(qb), * qbuf = dev_alloc(qb);
    copy_dev(aa, a0, N * 4); copy_dev(ab, a0, N * 4);
    if (op == F::EW_SILU_MUL) K::silu_mul(aa, b, (int64_t) N, g_s); else K::sigmoid_mul(aa, b, (int64_t) N, g_s);
    F::matvec_quantize(aa, n, T, qa, g_s);
    if (!F::ew_quant(op, ab, b, ab, qbuf, T, n, g_s)) { printf("  ew_quant refused  FAIL\n"); ++g_fail; }
    same("y (in place over a)", aa, ab, N * 4);
    same("a16 activation", qa, qbuf, qb);
    const float t_old = time_us([&] { if (op == F::EW_SILU_MUL) K::silu_mul(aa, b, (int64_t) N, g_s); else K::sigmoid_mul(aa, b, (int64_t) N, g_s);
                                      F::matvec_quantize(aa, n, T, qa, g_s); });
    const float t_new = time_us([&] { F::ew_quant(op, ab, b, ab, qbuf, T, n, g_s); });
    printf("  time: op+quantize %.2f us (2 kernels)  ->  ew_quant %.2f us (1 kernel)\n", t_old, t_new);
    for (void * p : {(void *) a0, (void *) b, (void *) aa, (void *) ab, qa, qbuf}) cudaFree(p);
}

static void test_gdn(int T) {
    const int Hk = 8, Hv = 24, D = 128, C = 2 * Hk * D + Hv * D;
    printf("gdn_layer + quantize epilogue (Hk=%d Hv=%d D=%d T=%d)\n", Hk, Hv, D, T);
    const float eps = 1e-6f;
    float * x = dev_rand((size_t) T * C, 1.0f), * a = dev_rand((size_t) T * Hv, 1.0f), * bb = dev_rand((size_t) T * Hv, 1.0f);
    float * z = dev_rand((size_t) T * Hv * D, 1.0f);
    float * cw = dev_rand((size_t) C * 4, 0.3f), * dtb = dev_rand(Hv, 0.5f), * ssa = dev_rand(Hv, 0.3f, -1.0f), * nw = dev_rand(D, 0.2f, 1.0f);
    F::GdnWeights gw{cw, dtb, ssa, nw};
    float * cs0 = dev_rand((size_t) C * 3, 0.5f), * st0 = dev_rand((size_t) Hv * D * D, 0.05f);
    float * cs[2], * st[2], * aa[2], * ba[2], * y[2], * out[2];
    void * q[2];
    const size_t qb = F::matvec_scratch_bytes((int64_t) Hv * D, T);
    for (int i = 0; i < 2; ++i) {
        cs[i] = (float *) dev_alloc((size_t) C * 3 * 4); st[i] = (float *) dev_alloc((size_t) Hv * D * D * 4);
        aa[i] = (float *) dev_alloc((size_t) T * Hv * 4); ba[i] = (float *) dev_alloc((size_t) T * Hv * 4);
        y[i] = (float *) dev_alloc((size_t) T * C * 4); out[i] = (float *) dev_alloc((size_t) T * Hv * D * 4);
        q[i] = dev_alloc(qb);
        copy_dev(cs[i], cs0, (size_t) C * 3 * 4); copy_dev(st[i], st0, (size_t) Hv * D * D * 4);
        copy_dev(aa[i], a, (size_t) T * Hv * 4); copy_dev(ba[i], bb, (size_t) T * Hv * 4);
    }
    F::gdn_layer(x, aa[0], ba[0], z, gw, cs[0], st[0], y[0], aa[0], ba[0], out[0], T, Hk, Hv, D, eps, g_s);
    F::matvec_quantize(out[0], (int64_t) Hv * D, T, q[0], g_s);
    F::gdn_layer(x, aa[1], ba[1], z, gw, cs[1], st[1], y[1], aa[1], ba[1], out[1], T, Hk, Hv, D, eps, g_s, q[1]);
    same("out (gated-normed)", out[0], out[1], (size_t) T * Hv * D * 4);
    same("state", st[0], st[1], (size_t) Hv * D * D * 4);
    same("a16 activation", q[0], q[1], qb);
    const float t_old = time_us([&] { F::gdn_scan(y[0], aa[0], ba[0], z, nw, st[0], out[0], T, Hk, Hv, D, eps, g_s);
                                      F::matvec_quantize(out[0], (int64_t) Hv * D, T, q[0], g_s); });
    const float t_new = time_us([&] { F::gdn_scan(y[1], aa[1], ba[1], z, nw, st[1], out[1], T, Hk, Hv, D, eps, g_s, 0, q[1]); });
    printf("  time: gdn_scan+quantize %.2f us (2 kernels)  ->  gdn_scan(qbuf) %.2f us (1 kernel)\n", t_old, t_new);
}

static void test_attn_prep(int T, int p0, uint32_t kvt = F::KV_F16) {
    const int Hq = 12, Hkv = 2, D = 256, n_rot = 64;
    const float base = 1e7f, eps = 1e-6f;
    printf("attn_prep (Hq=%d Hkv=%d D=%d n_rot=%d T=%d p0=%d, %s cache)\n", Hq, Hkv, D, n_rot, T, p0,
           kvt == F::KV_Q8_0 ? "q8_0" : kvt == F::KV_Q4_0 ? "q4_0" : "f16");
    float * qg = dev_rand((size_t) T * Hq * 2 * D, 1.0f), * k = dev_rand((size_t) T * Hkv * D, 1.0f), * v = dev_rand((size_t) T * Hkv * D, 1.0f);
    float * qn = dev_rand(D, 0.2f, 1.0f), * kn = dev_rand(D, 0.2f, 1.0f);
    std::vector<int32_t> hp(T);
    for (int t = 0; t < T; ++t) hp[t] = p0 + t;
    int32_t * pos; CK(cudaMalloc(&pos, T * 4)); CK(cudaMemcpy(pos, hp.data(), T * 4, cudaMemcpyHostToDevice));
    const int n_ctx = 256;
    F::kv_cache c0 = F::kv_alloc(kvt, n_ctx, Hkv, D), c1 = F::kv_alloc(kvt, n_ctx, Hkv, D);
    const size_t cb = F::kv_row_bytes(kvt, Hkv, D) * n_ctx;
    CK(cudaMemset(c0.k, 0, cb)); CK(cudaMemset(c0.v, 0, cb)); CK(cudaMemset(c1.k, 0, cb)); CK(cudaMemset(c1.v, 0, cb));
    float * q0 = (float *) dev_alloc((size_t) T * Hq * D * 4), * g0 = (float *) dev_alloc((size_t) T * Hq * D * 4);
    float * k0 = (float *) dev_alloc((size_t) T * Hkv * D * 4);
    float * q1 = (float *) dev_alloc((size_t) T * Hq * D * 4), * g1 = (float *) dev_alloc((size_t) T * Hq * D * 4);
    auto old_path = [&] {
        K::split_q_gate(qg, q0, g0, T, Hq, D, g_s);
        F::rmsnorm(q0, qn, q0, (int64_t) T * Hq, D, eps, g_s);
        copy_dev(k0, k, (size_t) T * Hkv * D * 4);
        F::rmsnorm(k0, kn, k0, (int64_t) T * Hkv, D, eps, g_s);
        K::rope_neox(q0, pos, T, Hq, D, n_rot, base, g_s);
        K::rope_neox(k0, pos, T, Hkv, D, n_rot, base, g_s);
        F::kv_append(c0, k0, v, p0, T, g_s);
    };
    old_path();
    if (!F::attn_prep(qg, k, v, qn, kn, pos, q1, g1, c1, p0, T, Hq, Hkv, D, n_rot, base, eps, g_s)) { printf("  attn_prep refused  FAIL\n"); ++g_fail; }
    same("q (normed + roped)", q0, q1, (size_t) T * Hq * D * 4);
    same("gate", g0, g1, (size_t) T * Hq * D * 4);
    same("K cache", c0.k, c1.k, cb);
    same("V cache", c0.v, c1.v, cb);
    {   // p0 = -1: cache rows from pos[t] on the device (CUDA-graph decode)
        F::kv_cache c2 = F::kv_alloc(kvt, n_ctx, Hkv, D);
        CK(cudaMemset(c2.k, 0, cb)); CK(cudaMemset(c2.v, 0, cb));
        float * q2 = (float *) dev_alloc((size_t) T * Hq * D * 4), * g2 = (float *) dev_alloc((size_t) T * Hq * D * 4);
        if (!F::attn_prep(qg, k, v, qn, kn, pos, q2, g2, c2, -1, T, Hq, Hkv, D, n_rot, base, eps, g_s)) { printf("  attn_prep(p0=-1) refused  FAIL\n"); ++g_fail; }
        same("q [device positions]", q0, q2, (size_t) T * Hq * D * 4);
        same("K cache [device positions]", c0.k, c2.k, cb);
        same("V cache [device positions]", c0.v, c2.v, cb);
        F::kv_free(c2); cudaFree(q2); cudaFree(g2);
    }
    const float t_old = time_us([&] {
        K::split_q_gate(qg, q0, g0, T, Hq, D, g_s);
        F::rmsnorm(q0, qn, q0, (int64_t) T * Hq, D, eps, g_s);
        F::rmsnorm(k, kn, k0, (int64_t) T * Hkv, D, eps, g_s);
        K::rope_neox(q0, pos, T, Hq, D, n_rot, base, g_s);
        K::rope_neox(k0, pos, T, Hkv, D, n_rot, base, g_s);
        F::kv_append(c0, k0, v, p0, T, g_s); });
    const float t_new = time_us([&] { F::attn_prep(qg, k, v, qn, kn, pos, q1, g1, c1, p0, T, Hq, Hkv, D, n_rot, base, eps, g_s); });
    printf("  time: split+norm x2+rope x2+append %.2f us (6 kernels)  ->  attn_prep %.2f us (1 kernel)\n", t_old, t_new);
}

// ---- random quantized weights ----
struct qspec { uint32_t type; const char * name; int bsz; std::vector<int> half_offs; };
static std::vector<uint8_t> rand_weight(const qspec & q, int64_t n_in, int64_t n_out) {
    const p100e::qtype_info & ti = p100e::get_type_info(q.type);
    if ((int) ti.size != q.bsz) { printf("block size mismatch for %s: %d vs %d\n", q.name, (int) ti.size, q.bsz); exit(3); }
    const size_t nb = (size_t) (n_in / ti.blck) * n_out;
    std::vector<uint8_t> h(nb * q.bsz);
    for (auto & b : h) b = (uint8_t) (urand() >> 24);
    for (size_t b = 0; b < nb; ++b)
        for (int off : q.half_offs) {
            const __half hv = __float2half((0.5f + (urand() >> 8) / 16777216.0f) / 256.0f);
            memcpy(&h[b * q.bsz + off], &hv, 2);
        }
    return h;
}
static const qspec g_specs[] = {
    {2,  "q4_0", 18,  {0}},
    {12, "q4_K", 144, {0, 2}},
    {11, "q3_K", 110, {108}},
    {13, "q5_K", 176, {0, 2}},
    {14, "q6_K", 210, {208}},
    {8,  "q8_0", 34,  {0}},
};

static F::packed_weight upload_pack(const std::vector<uint8_t> & h, uint32_t type, int64_t n_in, int64_t n_out, void ** dptr) {
    void * d; CK(cudaMalloc(&d, h.size())); CK(cudaMemcpy(d, h.data(), h.size(), cudaMemcpyHostToDevice));
    // pageable H2D may return before the DMA lands and the in-place repack runs on a non-blocking stream
    CK(cudaDeviceSynchronize());
    *dptr = d;
    F::packed_weight w = F::pack_weight(type, d, n_in, n_out, g_s);
    CK(cudaStreamSynchronize(g_s));
    return w;
}

static void test_matvec(const qspec & q, int64_t n_in, int64_t nA, int64_t nB) {
    printf("%s n_in=%lld: matvec_prequantized and concatenated weights (A %lld rows || B %lld rows)\n", q.name,
           (long long) n_in, (long long) nA, (long long) nB);
    auto hA = rand_weight(q, n_in, nA), hB = rand_weight(q, n_in, nB);
    std::vector<uint8_t> hAB(hA); hAB.insert(hAB.end(), hB.begin(), hB.end());
    void *dA, *dB, *dAB;
    F::packed_weight wA = upload_pack(hA, q.type, n_in, nA, &dA), wB = upload_pack(hB, q.type, n_in, nB, &dB);
    F::packed_weight wAB = upload_pack(hAB, q.type, n_in, nA + nB, &dAB);
    printf("  layouts: A %s, B %s, A||B %s\n", F::kernel_name(wA), F::kernel_name(wB), F::kernel_name(wAB));
    if (wA.layout != wAB.layout || wB.layout != wAB.layout) { printf("  layout differs  FAIL\n"); ++g_fail; }
    // T = 1..8 (batched multi-session decode steps up to 8 rows; P100E_TEST_TMAX overrides)
    static const int Tmax = getenv("P100E_TEST_TMAX") ? atoi(getenv("P100E_TEST_TMAX")) : 8;
    float * x = dev_rand((size_t) Tmax * n_in, 1.0f);
    void * scr = dev_alloc(F::matvec_scratch_bytes(n_in, 8)), * qbuf = dev_alloc(F::matvec_scratch_bytes(n_in, 8));
    float * yA = (float *) dev_alloc((size_t) Tmax * nA * 4), * yB = (float *) dev_alloc((size_t) Tmax * nB * 4);
    float * yA2 = (float *) dev_alloc((size_t) Tmax * nA * 4), * yAB = (float *) dev_alloc((size_t) Tmax * (nA + nB) * 4);
    float * yA1 = (float *) dev_alloc((size_t) Tmax * nA * 4);
    float * yP = (float *) dev_alloc((size_t) Tmax * (nA + nB) * 4);
    // the T = 1 result of every column (reference for the T-invariance checks)
    for (int t = 0; t < Tmax; ++t) F::matvec(wA, x + (size_t) t * n_in, yA1 + (size_t) t * nA, 1, scr, g_s);
    // q4r runs 4 warps at T = 1 and 2 above (legacy kernel): its columns are not T-invariant
    const bool tinv = wA.layout != F::LAYOUT_Q4R;
    for (int T = 1; T <= Tmax; ++T) {
        F::matvec(wA, x, yA, T, scr, g_s);
        F::matvec(wB, x, yB, T, scr, g_s);
        F::matvec_quantize(x, n_in, T, qbuf, g_s);
        F::matvec_prequantized(wA, x, qbuf, yA2, T, g_s);
        F::matvec_prequantized(wAB, x, qbuf, yAB, T, g_s);
        char nm[96];
        snprintf(nm, sizeof nm, "T=%d prequantized vs matvec (A)", T);
        same(nm, yA, yA2, (size_t) T * nA * 4);
        if (T > 1 && tinv) {
            snprintf(nm, sizeof nm, "T=%d all columns vs T=1 per column (A)", T);
            same(nm, yA, yA1, (size_t) T * nA * 4);
        }
        for (int t = 0; t < T; ++t) {
            snprintf(nm, sizeof nm, "T=%d col %d: A||B rows [0,%lld) vs A", T, t, (long long) nA);
            same(nm, yAB + (size_t) t * (nA + nB), yA + (size_t) t * nA, (size_t) nA * 4);
            snprintf(nm, sizeof nm, "T=%d col %d: A||B rows [%lld,%lld) vs B", T, t, (long long) nA, (long long) (nA + nB));
            same(nm, yAB + (size_t) t * (nA + nB) + nA, yB + (size_t) t * nB, (size_t) nB * 4);
        }
        // one launch over A||B writing A's rows to [T][nA] and B's rows to [T][nB] (the MTP verify's concatenated sets)
        CK(cudaMemset(yP, 0xff, (size_t) Tmax * (nA + nB) * 4));
        const F::mv_out parts[2] = {{yP, nA}, {yP + (size_t) T * nA, nB}};
        F::matvec_prequantized_parts(wAB, x, qbuf, parts, 2, T, g_s);
        snprintf(nm, sizeof nm, "T=%d parts launch over A||B: A part vs A", T);
        same(nm, yP, yA, (size_t) T * nA * 4);
        snprintf(nm, sizeof nm, "T=%d parts launch over A||B: B part vs B", T);
        same(nm, yP + (size_t) T * nA, yB, (size_t) T * nB * 4);
    }
    // multi-column kernels (T = 2..4, the MTP verify): every wrm geometry, forced K-phase splits and the older wrcp must
    // give column t bit-identical to the T = 1 call on column t (the warp-rows layouts; q8r / legacy layouts skip)
    const bool wr_layout = wA.layout == F::LAYOUT_Q4KR || wA.layout == F::LAYOUT_Q3R || wA.layout == F::LAYOUT_Q5KR ||
                           wA.layout == F::LAYOUT_Q6R3 || (wA.layout == F::LAYOUT_Q6R2 && wA.bytes / nA % 8 == 0);
    const bool q8_layout = wA.layout == F::LAYOUT_Q8R;
    if (wr_layout || q8_layout) {
        const F::mv_tune_t saved = F::mv_tune;
        struct cfg { int wrm, geo, kpg, ph; };
        std::vector<cfg> cfgs = {{0, 0, 0, 0}, {1, 0, 0, 0}};
        if (wr_layout) {
            for (int g : {1, 2, 3, 4, 6, 9, 10, 11, 12, 13, 21, 22, 23, 24, 25, 26, 27}) cfgs.push_back({1, g, 0, 0});
            const int ng = (int) (n_in / 32);
            for (int k = 32; k < ng; k += 32) cfgs.push_back({1, k % 64 == 0 ? 4 : 1, k, 0});
            for (int ph : {1, 2, 3}) cfgs.push_back({1, 1, 0, ph});   // K-phased tile rounds per pass (P100E_MV_WRM_PH)
            for (int ph : {2, 3, 4}) if (ng > 64) cfgs.push_back({1, 1, 32*((ng/2 + 31)/32), ph});
        } else {
            for (int g = 31; g <= 47; ++g) cfgs.push_back({1, g, 0, 0});
        }
        int ndiff = 0, nrun = 0;
        for (int T = 2; T <= Tmax; ++T) {
            F::matvec_quantize(x, n_in, T, qbuf, g_s);
            for (const cfg & c : cfgs) {
                if (wr_layout) { F::mv_tune.wrm = c.wrm; F::mv_tune.wrm_geo = c.geo; F::mv_tune.wrm_kpg = c.kpg; F::mv_tune.wrm_ph = c.ph; }
                else F::mv_tune.q8_geo = c.wrm == 0 ? -1 : c.geo;
                CK(cudaMemset(yA2, 0xff, (size_t) T * nA * 4));
                F::matvec_prequantized(wA, x, qbuf, yA2, T, g_s);
                CK(cudaDeviceSynchronize());
                auto h2 = bytes_of(yA2, (size_t) T * nA * 4), h1 = bytes_of(yA1, (size_t) T * nA * 4);
                const bool d = memcmp(h2.data(), h1.data(), h1.size()) != 0;
                ++nrun;
                if (d) {
                    ++ndiff;
                    printf("  T=%d wrm %d geo %d (%s) kpg %d: columns differ from T=1  FAIL\n", T, c.wrm, c.geo,
                           c.geo ? (wr_layout ? F::wrm_geo_name(c.geo) : F::q8m_geo_name(c.geo)) : "default", c.kpg);
                }
            }
        }
        ++g_checks;
        if (ndiff) ++g_fail;
        printf("  %-58s %8d runs  %s\n", wr_layout ? "T=2..Tmax every wrm geometry / K split / wrcp vs T=1"
                                                 : "T=2..Tmax every q8_wrm geometry / q8_wrt vs T=1", nrun, ndiff ? "FAIL" : "identical");
        for (int T = 2; T <= Tmax; ++T) {
            F::matvec_quantize(x, n_in, T, qbuf, g_s);
            F::mv_tune = saved;
            if (wr_layout) F::mv_tune.wrm = 0; else F::mv_tune.q8_geo = -1;
            const float t_old = time_us([&] { F::matvec_prequantized(wA, x, qbuf, yA2, T, g_s); });
            F::mv_tune = saved;
            const float t_new = time_us([&] { F::matvec_prequantized(wA, x, qbuf, yA2, T, g_s); });
            printf("  time T=%d (A, prequantized): %s %.2f us -> %s %.2f us\n", T, wr_layout ? "wrcp" : "q8_wrt", t_old,
                   wr_layout ? "wrm" : "q8_wrm", t_new);
        }
        F::mv_tune = saved;
    }
    if (F::matvec_uses_a16(wA)) {
        const float t_old = time_us([&] { F::matvec(wA, x, yA, 1, scr, g_s); F::matvec(wB, x, yB, 1, scr, g_s); });
        const float t_new = time_us([&] { F::matvec_quantize(x, n_in, 1, qbuf, g_s); F::matvec_prequantized(wAB, x, qbuf, yAB, 1, g_s); });
        printf("  time T=1: matvec(A)+matvec(B) %.2f us (4 kernels)  ->  quantize + one A||B launch %.2f us (2 kernels)\n", t_old, t_new);
    } else {
        const float t_old = time_us([&] { F::matvec(wA, x, yA, 1, scr, g_s); F::matvec(wB, x, yB, 1, scr, g_s); });
        const float t_new = time_us([&] { F::matvec_prequantized(wAB, x, qbuf, yAB, 1, g_s); });
        printf("  time T=1: matvec(A)+matvec(B) %.2f us (2 kernels)  ->  one A||B launch %.2f us (1 kernel)\n", t_old, t_new);
    }
    for (void * p : {dA, dB, dAB, (void *) x, scr, qbuf, (void *) yA, (void *) yB, (void *) yA2, (void *) yAB, (void *) yA1, (void *) yP}) cudaFree(p);
}

int main() {
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    printf("device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    CK(cudaStreamCreateWithFlags(&g_s, cudaStreamNonBlocking));
    test_add2_rmsnorm(5120, 1);
    test_add2_rmsnorm(5120, 3);
    test_ew(F::EW_SILU_MUL, 8704, 1);
    test_ew(F::EW_SILU_MUL, 8704, 2);
    test_ew(F::EW_SIGMOID_MUL, 3072, 1);
    test_gdn(1);
    test_gdn(3);
    test_attn_prep(1, 37);
    test_attn_prep(2, 100);
    test_attn_prep(1, 37, F::KV_Q8_0);
    test_attn_prep(4, 100, F::KV_Q8_0);
    test_attn_prep(1, 37, F::KV_Q4_0);
    test_attn_prep(3, 100, F::KV_Q4_0);

    // shapes of the TP=2 Qwen3.8-27B slices: gate|up 8704+8704 (q4_K / q3_K), qkv|gate 5120+3072 (q6_K / q4_K),
    // beta|alpha 24+24 (q8_0 / q4_K), k|v 512+512 (q8_0), ssm_out n_in 3072 (q5_K)
    test_matvec(g_specs[1], 5120, 8704, 8704);
    test_matvec(g_specs[2], 5120, 8704, 8704);
    test_matvec(g_specs[4], 5120, 5120, 3072);
    test_matvec(g_specs[1], 5120, 24, 24);
    test_matvec(g_specs[5], 5120, 24, 24);
    test_matvec(g_specs[5], 5120, 512, 512);
    test_matvec(g_specs[3], 3072, 512, 256);
    test_matvec(g_specs[0], 5120, 256, 128);
    // the fused decode's concatenations with a tiny part: q4_K qkv|gate (8192 rows) || beta|alpha (48), q6_K qkv || gate
    test_matvec(g_specs[1], 5120, 8192, 48);
    test_matvec(g_specs[4], 5120, 8192, 48);
    // q5_K ssm_out shard shape; q6_K with nb % 4 == 2 (n_in 8704: q6r2 with a 4-byte aligned row stride, legacy kernel)
    test_matvec(g_specs[3], 3072, 5120, 128);
    test_matvec(g_specs[4], 8704, 5120, 128);
    // q3_K / q4_K ffn_down shard shape (n_in 8704)
    test_matvec(g_specs[2], 8704, 5120, 128);
    test_matvec(g_specs[1], 8704, 5120, 128);
    printf("%s (%d checks, %d failures; all checks bit-exact)\n", g_fail ? "FAIL" : "PASS", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
