// fast::attn_fold (fp16 fold-GEMM prefill attention) vs fast::attn_prefill (fp32) on the same f16 / q8_0 cache:
// error relative to max |ref| and RMS, and per-call time of both. Shapes of one P100 of the TP=2 Qwen3.8-27B split
// (Hkv = 2, Hq = 12, D = 256). usage: p100e-fast-attn-fold-test [max_depth] (default 70000; cases deeper are skipped)
#include "fast/attn.cuh"
#include "fast/attn_fold.cuh"

#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(2); } } while (0)

namespace F = p100e::fast;
constexpr int Hq = 12, Hkv = 2, D = 256;

__device__ __forceinline__ uint32_t hash32(uint32_t x) { x ^= x >> 16; x *= 0x7feb352d; x ^= x >> 15; x *= 0x846ca68b; x ^= x >> 16; return x; }
// roughly normal values (sum of 3 uniforms) * scale; every 37th dim of a row boosted (outlier dims like real q/k)
__global__ void fill_k(float * p, size_t n, uint32_t seed, float scale) {
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        const uint32_t h1 = hash32((uint32_t) i * 2654435761u + seed), h2 = hash32(h1 + 77), h3 = hash32(h2 + 99);
        const float u = ((h1 & 0xffff) + (h2 & 0xffff) + (h3 & 0xffff)) / 65536.0f - 1.5f;
        p[i] = u * scale * ((i % D) % 37 == 5 ? 4.0f : 1.0f);
    }
}

int main(int argc, char ** argv) {
    const int max_depth = argc > 1 ? atoi(argv[1]) : 70000;
    cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, 0));
    printf("device: %s (sm_%d%d, %d SMs)\n", pr.name, pr.major, pr.minor, pr.multiProcessorCount);
    cudaStream_t s; CK(cudaStreamCreate(&s));
    const size_t scratch_bytes = (size_t) 256 << 20;   // tp_model's w_scratch
    void * scratch; CK(cudaMalloc(&scratch, scratch_bytes));
    int fails = 0;
    struct cs { int p0, T, chunk, nsplit; };
    const cs cases[] = {{0, 128, 0, 0}, {0, 300, 0, 0}, {0, 2048, 0, 0}, {37, 1000, 512, 0}, {4096, 512, 0, 0},
                        {4096, 512, 0, 3}, {6000, 2048, 1024, 2}, {16384, 2048, 0, 0}, {65536, 2048, 0, 0}};
    const float scale = 1.0f / 16.0f;
    for (uint32_t kvt : {F::KV_F16, F::KV_Q8_0, F::KV_Q4_0}) {
        printf("\n%s cache\n", kvt == F::KV_F16 ? "f16" : kvt == F::KV_Q8_0 ? "q8_0" : "q4_0");
        for (const cs & c : cases) {
            if (c.p0 > max_depth) continue;
            const int n = c.p0 + c.T;
            F::kv_cache kv = F::kv_alloc(kvt, n, Hkv, D);
            if (!kv.k) { printf("  p0=%d T=%d: kv_alloc failed, skipped\n", c.p0, c.T); continue; }
            float * kf, * vf, * q, * o1, * o2;
            const int CH = 4096;
            CK(cudaMalloc(&kf, (size_t) CH * Hkv * D * 4)); CK(cudaMalloc(&vf, (size_t) CH * Hkv * D * 4));
            for (int p = 0; p < n; p += CH) {
                const int m = std::min(CH, n - p);
                fill_k<<<512, 256, 0, s>>>(kf, (size_t) m * Hkv * D, 1000 + p, 1.0f);
                fill_k<<<512, 256, 0, s>>>(vf, (size_t) m * Hkv * D, 9000 + p, 1.0f);
                F::kv_append(kv, kf, vf, p, m, s);
            }
            CK(cudaMalloc(&q, (size_t) c.T * Hq * D * 4));
            CK(cudaMalloc(&o1, (size_t) c.T * Hq * D * 4)); CK(cudaMalloc(&o2, (size_t) c.T * Hq * D * 4));
            fill_k<<<512, 256, 0, s>>>(q, (size_t) c.T * Hq * D, 77 + c.p0, 2.0f);   // logit std ~ 3-4: peaked softmax
            cudaEvent_t e0, e1, e2; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1)); CK(cudaEventCreate(&e2));
            F::attn_prefill(q, kv, o1, c.T, c.p0, Hq, scale, s);   // warm-up
            CK(cudaEventRecord(e0, s));
            const bool ok1 = F::attn_prefill(q, kv, o1, c.T, c.p0, Hq, scale, s);
            CK(cudaEventRecord(e1, s));
            const bool ok2 = F::attn_fold(q, kv, o2, c.T, c.p0, Hq, scale, scratch, scratch_bytes, s, c.chunk, c.nsplit);
            CK(cudaEventRecord(e2, s));
            CK(cudaStreamSynchronize(s));
            CK(cudaGetLastError());
            float ms1 = 0, ms2 = 0; CK(cudaEventElapsedTime(&ms1, e0, e1)); CK(cudaEventElapsedTime(&ms2, e1, e2));
            std::vector<float> a((size_t) c.T * Hq * D), b(a.size());
            CK(cudaMemcpy(a.data(), o1, a.size() * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(b.data(), o2, b.size() * 4, cudaMemcpyDeviceToHost));
            double md = 0, mr = 0, se = 0, sr = 0; int bad = 0;
            for (size_t i = 0; i < a.size(); ++i) {
                if (!std::isfinite(b[i])) ++bad;
                const double d = (double) a[i] - b[i];
                md = std::max(md, std::fabs(d)); mr = std::max(mr, (double) std::fabs(a[i]));
                se += d * d; sr += (double) a[i] * a[i];
            }
            const double rel = md / (mr + 1e-30), rrms = std::sqrt(se / (sr + 1e-30));
            // fp16 products (Q, K, V, P rounded to f16, 128-long f16 chains): ~0.3-0.6% RMS of the fp32 result here
            const bool pass = ok1 && ok2 && !bad && rel < 3e-2 && rrms < 1e-2;
            const double flop = 4.0 * D * Hq * ((double) c.p0 * c.T + 0.5 * (double) c.T * c.T);
            printf("  p0=%6d T=%5d chunk %4d split %d: max|d|/max|ref| %.2e  rms(d)/rms(ref) %.2e  nonfinite %d  "
                   "fp32 %8.3f ms  fold %8.3f ms (%.2f TFLOPS) x%.1f  %s\n",
                   c.p0, c.T, c.chunk, c.nsplit, rel, rrms, bad, ms1, ms2, flop / (ms2 * 1e-3) / 1e12, ms1 / ms2,
                   pass ? "OK" : "FAIL");
            if (!pass) ++fails;
            for (float * p : {kf, vf, q, o1, o2}) cudaFree(p);
            F::kv_free(kv);
        }
    }
    printf("\n%s (%d failing)\n", fails ? "FAILED" : "ALL PASSED", fails);
    return fails ? 1 : 0;
}
