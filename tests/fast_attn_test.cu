// Correctness + timing of p100e::fast::attn_decode (f16 / q8_0 / q4_0 KV) vs reference attention on the dequantized
// cache; T-invariance of the decode attention (row t of a T-row call == the T = 1 call at p0 + t, MTP verify);
// attn_prefill (f16 / q8_0) vs the reference.
#include "fast/attn.cuh"
#include "kernels.cuh"

#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <functional>
#include <algorithm>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(2); } } while (0)

namespace F = p100e::fast;
namespace K = p100e::k;

constexpr int Hq = 24, Hkv = 4, D = 256;
static int g_fail = 0;

__device__ __forceinline__ uint32_t hash32(uint32_t x) { x ^= x >> 16; x *= 0x7feb352d; x ^= x >> 15; x *= 0x846ca68b; x ^= x >> 16; return x; }
__global__ void fill_k(float * p, size_t n, uint32_t seed, float scale) {
    size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint32_t h1 = hash32((uint32_t) i * 2654435761u + seed), h2 = hash32(h1 + 77), h3 = hash32(h2 + 99);
    p[i] = ((h1 & 0xffff) + (h2 & 0xffff) + (h3 & 0xffff)) * (scale / 32768.0f) - 3.0f * scale;
}
static void fill(float * p, size_t n, uint32_t seed, float scale) { fill_k<<<(unsigned) ((n + 255) / 256), 256>>>(p, n, seed, scale); }

// read one cache element (pos, kv head, dim) as float
__device__ __forceinline__ float rd(const char * base, uint32_t type, size_t pos, int hk, int d) {
    if (type == F::KV_F16) return __half2float(reinterpret_cast<const __half *>(base + pos * (Hkv * D * 2))[hk * D + d]);
    if (type == F::KV_Q8_0) {   // per head: 8 f16 scales, then 256 int8
        const char * h = base + pos * (Hkv * 272) + hk * 272;
        return __half2float(reinterpret_cast<const __half *>(h)[d >> 5]) * (float) reinterpret_cast<const int8_t *>(h + 16)[d];
    }
    const char * b = base + pos * (Hkv * 8 * 18) + hk * 144 + (d >> 5) * 18;
    const float sc = __half2float(*reinterpret_cast<const __half *>(b));
    const int i = d & 31; const uint32_t by = (uint8_t) b[2 + (i & 15)];
    return ((int) (i < 16 ? (by & 15) : (by >> 4)) - 8) * sc;
}
__global__ void dequant_k(const char * base, uint32_t type, float * out, size_t npos) {
    size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= npos * Hkv * D) return;
    out[i] = rd(base, type, i / (Hkv * D), (int) ((i / D) % Hkv), (int) (i % D));
}
// independent reference: block per (t, h), thread per dim, online softmax over all positions (slow but cache-size independent)
__global__ void ref_attn_k(const float * q, const char * kc, const char * vc, uint32_t type, float * out, int p0, float scale) {
    __shared__ float sh[8];
    const int t = blockIdx.y, h = blockIdx.x, hk = h / 6, d = threadIdx.x, n = p0 + t + 1;
    const float qd = q[((size_t) t * Hq + h) * D + d] * scale;
    float m = -INFINITY, l = 0.0f, acc = 0.0f;
    for (int j = 0; j < n; ++j) {
        float v = qd * rd(kc, type, j, hk, d);
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
        __syncthreads();
        if ((d & 31) == 0) sh[d >> 5] = v;
        __syncthreads();
        float s = 0; for (int w = 0; w < 8; ++w) s += sh[w];
        const float nm = fmaxf(m, s), f = expf(m - nm), p = expf(s - nm);
        l = l * f + p; acc = acc * f + p * rd(vc, type, j, hk, d); m = nm;
    }
    out[((size_t) t * Hq + h) * D + d] = acc / l;
}

static std::vector<float> to_host(const float * d, size_t n) { std::vector<float> h(n); CK(cudaMemcpy(h.data(), d, n * 4, cudaMemcpyDeviceToHost)); return h; }
static void cmp(const char * name, const float * a, const float * b, size_t n, double tol) {
    auto ha = to_host(a, n), hb = to_host(b, n);
    double ma = 0, mx = 0, mr = 0;
    for (size_t i = 0; i < n; ++i) mx = fmax(mx, fabs(hb[i]));
    for (size_t i = 0; i < n; ++i) { const double e = fabs((double) ha[i] - hb[i]); ma = fmax(ma, e); mr = fmax(mr, e / (fabs(hb[i]) + 1e-2 * mx)); }
    const bool ok = ma <= tol * mx && std::isfinite(ma);
    if (!ok) ++g_fail;
    printf("    %-26s max_abs=%.3e  max_abs/ref_max=%.3e  max_rel(floor 1%% of max)=%.3e  %s\n", name, ma, ma / mx, mr, ok ? "OK" : "FAIL");
}
static const char * tname(uint32_t t) { return t == F::KV_F16 ? "f16 " : t == F::KV_Q8_0 ? "q8_0" : "q4_0"; }
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
    printf("device: %s (sm_%d%d, %d SMs)\n", pr.name, pr.major, pr.minor, pr.multiProcessorCount);
    const float scale = 1.0f / 16.0f;
    cudaStream_t s; CK(cudaStreamCreate(&s)); g_s = s;

    // ---- kv_append quantization round trip ----
    {
        const int T = 4; float * k, * v; CK(cudaMalloc(&k, T * Hkv * D * 4)); CK(cudaMalloc(&v, T * Hkv * D * 4));
        fill(k, T * Hkv * D, 1, 1.0f); fill(v, T * Hkv * D, 2, 2.0f);
        for (uint32_t type : {F::KV_F16, F::KV_Q8_0, F::KV_Q4_0}) {
            F::kv_cache c = F::kv_alloc(type, 8, Hkv, D);
            F::kv_append(c, k, v, 3, T, s);
            float * dq; CK(cudaMalloc(&dq, 8 * Hkv * D * 4));
            dequant_k<<<(8 * Hkv * D + 255) / 256, 256>>>((const char *) c.v, type, dq, 8); CK(cudaDeviceSynchronize());
            auto hd = to_host(dq, 8 * Hkv * D), hv = to_host(v, T * Hkv * D);
            double worst = 0;   // error relative to block amax
            for (int t = 0; t < T; ++t) for (int b = 0; b < Hkv * D / 32; ++b) {
                double amax = 0, e = 0;
                for (int i = 0; i < 32; ++i) { amax = fmax(amax, fabs(hv[(size_t) t * Hkv * D + b * 32 + i])); }
                for (int i = 0; i < 32; ++i) e = fmax(e, fabs(hv[(size_t) t * Hkv * D + b * 32 + i] - hd[(size_t) (3 + t) * Hkv * D + b * 32 + i]));
                worst = fmax(worst, e / amax);
            }
            const double lim = type == F::KV_F16 ? 1e-3 : type == F::KV_Q8_0 ? 0.005 : 0.126;
            const bool ok = worst < lim; if (!ok) ++g_fail;
            printf("kv_append %s round trip: max err / block amax = %.4f (limit %.3f)  %s\n", tname(type), worst, lim, ok ? "OK" : "FAIL");
            F::kv_free(c); cudaFree(dq);
        }
        cudaFree(k); cudaFree(v);
    }

    // ---- device-p0 variants (CUDA graph path): one captured graph replayed for many p0, must be bit-identical ----
    {
        const int n_ctx = 4100;
        int32_t * dp0; CK(cudaMalloc(&dp0, 4));
        float * qq, * a, * b, * kk, * vv;
        CK(cudaMalloc(&qq, 4 * Hq * D * 4)); CK(cudaMalloc(&a, 4 * Hq * D * 4)); CK(cudaMalloc(&b, 4 * Hq * D * 4));
        CK(cudaMalloc(&kk, 4 * Hkv * D * 4)); CK(cudaMalloc(&vv, 4 * Hkv * D * 4));
        fill(qq, 4 * Hq * D, 21, 1.5f); fill(kk, 4 * Hkv * D, 22, 1.0f); fill(vv, 4 * Hkv * D, 23, 1.0f);
        for (uint32_t type : {F::KV_F16, F::KV_Q8_0, F::KV_Q4_0}) {
            F::kv_cache c1 = F::kv_alloc(type, n_ctx, Hkv, D), c2 = F::kv_alloc(type, n_ctx, Hkv, D);
            const size_t cb = F::kv_row_bytes(type, Hkv, D) * (size_t) n_ctx;
            float * st; CK(cudaMalloc(&st, (size_t) n_ctx * Hkv * D * 4));
            fill(st, (size_t) n_ctx * Hkv * D, 31, 1.0f);
            for (int p = 0; p < n_ctx; p += 4) F::kv_append(c1, st + (size_t) p * Hkv * D, st + (size_t) p * Hkv * D, p, 4, s);
            CK(cudaMemcpyAsync(c2.k, c1.k, cb, cudaMemcpyDeviceToDevice, s)); CK(cudaMemcpyAsync(c2.v, c1.v, cb, cudaMemcpyDeviceToDevice, s));
            int bad = 0, cases = 0;
            for (int T : {1, 4}) {
                F::attn_decode(qq, c1, a, T, 0, Hq, scale, s);   // warm-up (scratch)
                CK(cudaStreamSynchronize(s));
                cudaGraph_t g; cudaGraphExec_t ge;
                CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
                F::kv_append_dev(c2, kk, vv, dp0, T, s);
                F::attn_decode_dev(qq, c2, b, T, dp0, n_ctx, Hq, scale, s);
                CK(cudaStreamEndCapture(s, &g)); CK(cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0));
                for (int p0 : {0, 1, 5, 31, 60, 63, 64, 65, 127, 200, 511, 600, 777, 1023, 1500, 2047, 2048, 3000, 4000, 4092, 4096}) {
                    if (p0 + T > n_ctx) continue;
                    F::kv_append(c1, kk, vv, p0, T, s);
                    F::attn_decode(qq, c1, a, T, p0, Hq, scale, s);
                    CK(cudaMemcpyAsync(dp0, &p0, 4, cudaMemcpyHostToDevice, s));
                    CK(cudaGraphLaunch(ge, s));
                    CK(cudaStreamSynchronize(s));
                    auto ha = to_host(a, (size_t) T * Hq * D), hb = to_host(b, (size_t) T * Hq * D);
                    std::vector<char> k1(cb), k2(cb);
                    CK(cudaMemcpy(k1.data(), c1.k, cb, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(k2.data(), c2.k, cb, cudaMemcpyDeviceToHost));
                    const bool same = memcmp(ha.data(), hb.data(), ha.size() * 4) == 0 && memcmp(k1.data(), k2.data(), cb) == 0;
                    ++cases; if (!same) { ++bad; printf("  dev-p0 MISMATCH type %u T=%d p0=%d\n", type, T, p0); }
                }
                cudaGraphExecDestroy(ge); cudaGraphDestroy(g);
            }
            if (bad) ++g_fail;
            printf("dev-p0 graph replay %s: %d cases, attention output + K cache bit-identical to host-p0 path: %s\n",
                   tname(type), cases, bad ? "FAIL" : "OK");
            // T-invariance (MTP verify): row t of attn_decode(T rows at p0) == attn_decode(1 row at p0 + t), bit for bit.
            // The split plan only depends on the visible length through 64-key chunks up to 64 * cap positions (cap =
            // min(64, 3 * SMs / Hkv): 4096 on a P100 with 2 local kv heads), so this holds for p0 + T <= 64 * cap;
            // beyond, the chunk size follows n and rows may differ in the last bits (reported only).
            {
                // v2 (default): T-invariant at every depth; v1 (P100E_ATTN_DEC=1): up to 64 * cap keys
                const bool v1 = getenv("P100E_ATTN_DEC") && atoi(getenv("P100E_ATTN_DEC")) == 1;
                const int lim = v1 ? 64 * std::min(64, 3 * pr.multiProcessorCount / Hkv) : 1 << 30;
                int nbad = 0, ncase = 0, nbad_long = 0, ncase_long = 0;
                double maxd = 0;
                float * o1; CK(cudaMalloc(&o1, Hq * D * 4));
                for (int p0 : {0, 1, 3, 29, 31, 32, 60, 61, 62, 63, 64, 100, 127, 128, 500, 1000, 2047, 2048, 3000, 4000, 4092, 4093}) {
                    for (int T = 2; T <= 4; ++T) {
                        if (p0 + T > n_ctx) continue;
                        F::attn_decode(qq, c1, a, T, p0, Hq, scale, s);
                        for (int t = 0; t < T; ++t) {
                            F::attn_decode(qq + (size_t) t * Hq * D, c1, o1, 1, p0 + t, Hq, scale, s);
                            CK(cudaStreamSynchronize(s));
                            auto ha = to_host(a + (size_t) t * Hq * D, (size_t) Hq * D), hb = to_host(o1, (size_t) Hq * D);
                            const bool same = memcmp(ha.data(), hb.data(), ha.size() * 4) == 0;
                            for (size_t i = 0; i < ha.size(); ++i) maxd = fmax(maxd, fabs((double) ha[i] - hb[i]));
                            if (p0 + T <= lim) { ++ncase; nbad += !same; } else { ++ncase_long; nbad_long += !same; }
                        }
                    }
                }
                cudaFree(o1);
                if (nbad) ++g_fail;
                printf("T-invariance %s (T = 2..4, p0 + T <= %d): %d/%d rows bit-identical to the T = 1 call  %s"
                       "  (p0 + T > %d: %d/%d identical, informational; max |diff| over all %.3e)\n",
                       tname(type), lim, ncase - nbad, ncase, nbad ? "FAIL" : "OK", lim, ncase_long - nbad_long, ncase_long, maxd);


            }
            {   // attn_prefill on this cache vs the independent reference
                for (auto pc : {std::make_pair(0, 37), std::make_pair(1000, 300), std::make_pair(3700, 396)}) {
                    const int p0 = pc.first, T = pc.second;
                    float * qp, * o_a, * o_b;
                    CK(cudaMalloc(&qp, (size_t) T * Hq * D * 4)); CK(cudaMalloc(&o_a, (size_t) T * Hq * D * 4));
                    CK(cudaMalloc(&o_b, (size_t) T * Hq * D * 4));
                    fill(qp, (size_t) T * Hq * D, 41 + p0, 1.5f);
                    const bool ok = F::attn_prefill(qp, c1, o_a, T, p0, Hq, scale, s);
                    ref_attn_k<<<dim3(Hq, T), D, 0, s>>>(qp, (const char *) c1.k, (const char *) c1.v, type, o_b, p0, scale);
                    CK(cudaStreamSynchronize(s));
                    if (!ok) { ++g_fail; printf("  attn_prefill refused %s  FAIL\n", tname(type)); }
                    char nm[64]; snprintf(nm, sizeof nm, "prefill %s p0=%d T=%d", tname(type), p0, T);
                    cmp(nm, o_a, o_b, (size_t) T * Hq * D, 2e-3);
                    cudaFree(qp); cudaFree(o_a); cudaFree(o_b);
                }
            }
            for (int p0 : {64, 576, 2048}) {   // cost of the max_n-sized grid (early-exit blocks) vs the exact grid
                CK(cudaMemcpy(dp0, &p0, 4, cudaMemcpyHostToDevice));
                const float th = time_ms([&] { F::attn_decode(qq, c1, a, 1, p0, Hq, scale, s); });
                const float td = time_ms([&] { F::attn_decode_dev(qq, c2, b, 1, dp0, n_ctx, Hq, scale, s); });
                printf("  T=1 p0=%4d: host-p0 %.4f ms, dev-p0 (grid for n<=%d) %.4f ms\n", p0, th, n_ctx, td);
            }
            cudaFree(st); F::kv_free(c1); F::kv_free(c2);
        }
        cudaFree(dp0); cudaFree(qq); cudaFree(a); cudaFree(b); cudaFree(kk); cudaFree(vv);
    }

    const int p0s[] = {128, 4096, 32768, 131072};
    float * q, * o1, * o2, * ck, * cv;
    CK(cudaMalloc(&q, 4 * Hq * D * 4)); CK(cudaMalloc(&o1, 4 * Hq * D * 4)); CK(cudaMalloc(&o2, 4 * Hq * D * 4));
    fill(q, 4 * Hq * D, 11, 1.5f);
    float * stage_k, * stage_v;
    const int CH = 1024;
    CK(cudaMalloc(&stage_k, (size_t) CH * Hkv * D * 4)); CK(cudaMalloc(&stage_v, (size_t) CH * Hkv * D * 4));
    struct Row { uint32_t type; int p0, T; float fast, ref; double gbs; };
    std::vector<Row> rows;

    for (uint32_t type : {F::KV_F16, F::KV_Q8_0, F::KV_Q4_0}) for (int p0 : p0s) {
        const int n_ctx = p0 + 4;
        size_t freeb, totb; CK(cudaMemGetInfo(&freeb, &totb));
        const size_t need = 2 * F::kv_row_bytes(type, Hkv, D) * (size_t) n_ctx + ((size_t) 400 << 20);
        if (need > freeb) { printf("\n[%s p0=%d] SKIPPED: needs ~%.0f MB, only %.0f MB free\n", tname(type), p0, need / 1048576.0, freeb / 1048576.0); continue; }
        F::kv_cache c = F::kv_alloc(type, n_ctx, Hkv, D);
        if (!c.k) { printf("\n[%s p0=%d] SKIPPED: kv_alloc failed (out of memory)\n", tname(type), p0); continue; }
        for (int p = 0; p < n_ctx; p += CH) {
            const int cnt = n_ctx - p < CH ? n_ctx - p : CH;
            fill(stage_k, (size_t) cnt * Hkv * D, 100 + p, 1.0f); fill(stage_v, (size_t) cnt * Hkv * D, 7000 + p, 1.0f);
            // append in pieces of <= 4 tokens is irrelevant for the kernel (grid.y = T); use T = cnt in chunks of 4 to match decode use
            for (int u = 0; u < cnt; u += 4) F::kv_append(c, stage_k + (size_t) u * Hkv * D, stage_v + (size_t) u * Hkv * D, p + u, (cnt - u) < 4 ? cnt - u : 4, s);
        }
        CK(cudaDeviceSynchronize());
        const bool small = p0 <= 4096;
        if (small) {
            CK(cudaMalloc(&ck, (size_t) n_ctx * Hkv * D * 4)); CK(cudaMalloc(&cv, (size_t) n_ctx * Hkv * D * 4));
            const size_t tot = (size_t) n_ctx * Hkv * D;
            dequant_k<<<(unsigned) ((tot + 255) / 256), 256>>>((const char *) c.k, type, ck, n_ctx);
            dequant_k<<<(unsigned) ((tot + 255) / 256), 256>>>((const char *) c.v, type, cv, n_ctx);
        }
        printf("\n[%s] p0=%d (n_ctx=%d, cache %.1f MB)\n", tname(type), p0, n_ctx, 2 * F::kv_row_bytes(type, Hkv, D) * (double) n_ctx / 1048576.0);
        for (int T : {1, 4}) {
            F::attn_decode(q, c, o1, T, p0, Hq, scale, s);
            ref_attn_k<<<dim3(Hq, T), D>>>(q, (const char *) c.k, (const char *) c.v, type, o2, p0, scale);
            CK(cudaDeviceSynchronize());
            printf("  T=%d\n", T);
            cmp("fast vs indep. reference", o1, o2, (size_t) T * Hq * D, 2e-3);
            const float fast = time_ms([&] { F::attn_decode(q, c, o1, T, p0, Hq, scale, s); });
            float refms = -1;
            if (small) {
                K::attention(q, ck, cv, o2, T, p0, Hq, Hkv, D, scale, s); CK(cudaDeviceSynchronize());
                cmp("fast vs k::attention", o1, o2, (size_t) T * Hq * D, 2e-3);
                refms = time_ms([&] { K::attention(q, ck, cv, o2, T, p0, Hq, Hkv, D, scale, s); });
            }
            const double bytes = 2.0 * F::kv_row_bytes(type, Hkv, D) * (p0 + T);
            rows.push_back({type, p0, T, fast, refms, bytes / (fast * 1e-3) / 1e9});
            if (T == 4) {   // T-invariance at this depth: row t of the T = 4 call vs the T = 1 call at p0 + t
                F::attn_decode(q, c, o1, 4, p0, Hq, scale, s);
                int same = 0;
                for (int t = 0; t < 4; ++t) {
                    F::attn_decode(q + (size_t) t * Hq * D, c, o2, 1, p0 + t, Hq, scale, s);
                    CK(cudaStreamSynchronize(s));
                    auto ha = to_host(o1 + (size_t) t * Hq * D, (size_t) Hq * D), hb = to_host(o2, (size_t) Hq * D);
                    same += memcmp(ha.data(), hb.data(), ha.size() * 4) == 0;
                }
                const bool v1 = getenv("P100E_ATTN_DEC") && atoi(getenv("P100E_ATTN_DEC")) == 1;
                if (!v1 && same != 4) ++g_fail;
                printf("    T-invariance at p0=%d: %d/4 rows bit-identical to T = 1%s\n", p0, same, v1 ? " (v1, informational)" : "");
            }
        }
        if (small) { cudaFree(ck); cudaFree(cv); }
        F::kv_free(c);
    }

    printf("\n=== timing summary (cudaEvents, 100 calls replayed as one CUDA graph after 10 warmup; ms per call; K+V cache read GB/s) ===\n");
    printf("%-5s %7s %2s  %10s %10s %8s %9s\n", "kv", "p0", "T", "fast ms", "ref ms", "speedup", "GB/s");
    for (auto & r : rows) {
        char refs[32] = "n/a (smem)", sp[32] = "-";
        if (r.ref > 0) { snprintf(refs, 32, "%.4f", r.ref); snprintf(sp, 32, "x%.1f", r.ref / r.fast); }
        printf("%-5s %7d %2d  %10.4f %10s %8s %9.1f\n", tname(r.type), r.p0, r.T, r.fast, refs, sp, r.gbs);

    }
    printf("(ref = src/kernels.cu k::attention on the dequantized f32 cache; it needs (p0+T)*4 B of shared memory, so it cannot run at p0 >= ~12k)\n");
    printf("\n%s\n", g_fail ? "SOME CHECKS FAILED" : "ALL CHECKS PASSED");
    return g_fail ? 1 : 0;
}
