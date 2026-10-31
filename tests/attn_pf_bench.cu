// attn_prefill variants (src/fast/attn_pf2.inc) vs the original attn_prefill_k: BITWISE comparison of the output and
// per-call time, f16 and q8_0 caches, shapes of one P100 of the TP=2 split (Hkv = 2, Hq = 12, D = 256).
// usage: p100e-attn-pf-bench [max_depth] (default 100000)
#include "fast/attn.cuh"

#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(2); } } while (0)

namespace F = p100e::fast;
constexpr int Hq = 12, Hkv = 2, D = 256;

__device__ __forceinline__ uint32_t hash32(uint32_t x) { x ^= x >> 16; x *= 0x7feb352d; x ^= x >> 15; x *= 0x846ca68b; x ^= x >> 16; return x; }
__global__ void fill_k(float * p, size_t n, uint32_t seed, float scale) {
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        const uint32_t h1 = hash32((uint32_t) i * 2654435761u + seed), h2 = hash32(h1 + 77), h3 = hash32(h2 + 99);
        const float u = ((h1 & 0xffff) + (h2 & 0xffff) + (h3 & 0xffff)) / 65536.0f - 1.5f;
        p[i] = u * scale * ((i % D) % 37 == 5 ? 4.0f : 1.0f);
    }
}

int main(int argc, char ** argv) {
    const int max_depth = argc > 1 ? atoi(argv[1]) : 100000;
    cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, 0));
    printf("device: %s (%d SMs)\n", pr.name, pr.multiProcessorCount);
    cudaStream_t s; CK(cudaStreamCreate(&s));
    struct cs { int p0, T; };
    const cs cases[] = {{100000, 48}, {100000, 16}, {100000, 60}, {100000, 9}, {32768, 48}, {4096, 48}, {0, 48}, {0, 1},
                        {1000, 33}, {5000, 7}, {3, 5}, {777, 2}, {4096, 2048}};
    const float scale = 1.0f / 16.0f;
    int fails = 0;
    for (uint32_t kvt : {F::KV_F16, F::KV_Q8_0}) {
        printf("\n%s cache   (ms per call: v0 = original; v1/v2/v3 = RT 4/2/1)\n", kvt == F::KV_F16 ? "f16" : "q8_0");
        for (const cs & c : cases) {
            if (c.p0 > max_depth) continue;
            const int n = c.p0 + c.T;
            F::kv_cache kv = F::kv_alloc(kvt, n, Hkv, D);
            if (!kv.k) { printf("  p0=%d T=%d: kv_alloc failed, skipped\n", c.p0, c.T); continue; }
            float * kf, * vf, * q, * o[4];
            const int CH = 4096;
            CK(cudaMalloc(&kf, (size_t) CH * Hkv * D * 4)); CK(cudaMalloc(&vf, (size_t) CH * Hkv * D * 4));
            for (int p = 0; p < n; p += CH) {
                const int m = std::min(CH, n - p);
                fill_k<<<512, 256, 0, s>>>(kf, (size_t) m * Hkv * D, 1000 + p, 1.0f);
                fill_k<<<512, 256, 0, s>>>(vf, (size_t) m * Hkv * D, 9000 + p, 1.0f);
                F::kv_append(kv, kf, vf, p, m, s);
            }
            CK(cudaMalloc(&q, (size_t) c.T * Hq * D * 4));
            fill_k<<<512, 256, 0, s>>>(q, (size_t) c.T * Hq * D, 77 + c.p0, 2.0f);
            const size_t ob = (size_t) c.T * Hq * D * 4;
            for (int v = 0; v < 4; ++v) { CK(cudaMalloc(&o[v], ob)); CK(cudaMemset(o[v], 0xff, ob)); }
            cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
            double ms[4];
            bool same[4] = {true, true, true, true};
            std::vector<float> ref(ob / 4), cur(ob / 4);
            for (int v = 0; v < 4; ++v) {
                const int reps = c.T >= 1024 ? 2 : 5;
                F::attn_prefill_variant(q, kv, o[v], c.T, c.p0, Hq, scale, s, v);   // warm-up
                CK(cudaEventRecord(e0, s));
                bool ok = true;
                for (int r = 0; r < reps; ++r) ok = F::attn_prefill_variant(q, kv, o[v], c.T, c.p0, Hq, scale, s, v) && ok;
                CK(cudaEventRecord(e1, s));
                CK(cudaStreamSynchronize(s));
                CK(cudaGetLastError());
                float t = 0; CK(cudaEventElapsedTime(&t, e0, e1)); ms[v] = t / reps;
                if (!ok) { printf("  variant %d refused\n", v); same[v] = false; }
                CK(cudaMemcpy(v == 0 ? ref.data() : cur.data(), o[v], ob, cudaMemcpyDeviceToHost));
                if (v > 0) same[v] = same[v] && memcmp(ref.data(), cur.data(), ob) == 0;
            }
            bool allsame = same[1] && same[2] && same[3];
            printf("  p0=%6d T=%5d: v0 %8.3f  v1 %8.3f  v2 %8.3f  v3 %8.3f ms   x%.2f x%.2f x%.2f   bitwise %s%s%s  %s\n", c.p0, c.T,
                   ms[0], ms[1], ms[2], ms[3], ms[0] / ms[1], ms[0] / ms[2], ms[0] / ms[3], same[1] ? "v1 " : "V1-DIFF ",
                   same[2] ? "v2 " : "V2-DIFF ", same[3] ? "v3" : "V3-DIFF", allsame ? "OK" : "FAIL");
            if (!allsame) ++fails;
            for (float * p : {kf, vf, q, o[0], o[1], o[2], o[3]}) cudaFree(p);
            F::kv_free(kv);
        }
    }
    printf("\n%s (%d failing)\n", fails ? "FAILED" : "ALL BITWISE EQUAL", fails);
    return fails ? 1 : 0;
}
