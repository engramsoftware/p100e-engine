// Decode attention timing at depth for the per-GPU shape of the TP=2 Qwen3.8-27B split (Hkv = 2, Hq = 12, D = 256):
// v1 (fast::attn_decode with P100E_ATTN_DEC=1 semantics is not switchable at runtime, so v1 is timed through a
// separate run) vs fast::attn_decode2 with a sweep of split counts. usage: p100e-attn-dec-bench [depths...]
#include "fast/attn.cuh"
#include "fast/attn_dec2.cuh"

#include <cuda_fp16.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(2); } } while (0)

namespace F = p100e::fast;
constexpr int Hq = 12, Hkv = 2, D = 256;

__global__ void fill_k(float * p, size_t n, uint32_t seed) {
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        uint32_t h = (uint32_t) i * 2654435761u ^ seed;
        h ^= h >> 16; h *= 0x7feb352d; h ^= h >> 15;
        p[i] = (float) (h & 0xffff) / 32768.0f - 1.0f;
    }
}

static cudaStream_t g_s;
static float time_ms(const std::function<void()> & f, int reps = 50) {
    if (getenv("MRV_ONCE")) {   // profiler mode (nvprof): warm-up + one plain launch, no graph
        f(); f(); CK(cudaStreamSynchronize(g_s)); return 0.0f;
    }
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
    return ms / reps;
}

int main(int argc, char ** argv) {
    std::vector<int> depths;
    for (int i = 1; i < argc; ++i) depths.push_back(atoi(argv[i]));
    if (depths.empty()) depths = {8192, 32768, 65536, 131072};
    cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, 0));
    printf("device: %s (%d SMs), Hkv %d Hq %d; default split %d; %s\n", pr.name, pr.multiProcessorCount, Hkv, Hq,
           F::attn_dec2_nsplit(Hkv), getenv("P100E_ATTN_DEC") ? "P100E_ATTN_DEC set (v1 path for attn_decode)" : "");
    CK(cudaStreamCreate(&g_s));
    const int maxd = depths.back() + 8;
    float * q, * o, * kf, * vf;
    CK(cudaMalloc(&q, 4 * Hq * D * 4)); CK(cudaMalloc(&o, 4 * Hq * D * 4));
    const int CH = 4096;
    CK(cudaMalloc(&kf, (size_t) CH * Hkv * D * 4)); CK(cudaMalloc(&vf, (size_t) CH * Hkv * D * 4));
    fill_k<<<256, 256>>>(q, 4 * Hq * D, 7);
    // MRV_T="1,2,3,4": T values (default 1,4); MRV_Q8ONLY=1: q8_0 cache only; MRV_DEFSPLIT=1: default split only
    std::vector<int> Ts = {1, 4};
    if (const char * e = getenv("MRV_T")) { Ts.clear(); for (const char * c = e; *c; ++c) if (*c >= '1' && *c <= '8') Ts.push_back(*c - '0'); }
    std::vector<int> splits = {0, 28, 42, 56, 84, 112, 168};
    if (getenv("MRV_DEFSPLIT")) splits = {0, F::attn_dec2_nsplit(Hkv)};
    std::vector<uint32_t> types = {F::KV_Q8_0, F::KV_Q4_0, F::KV_F16};
    if (getenv("MRV_Q8ONLY")) types = {F::KV_Q8_0};
    for (uint32_t type : types) {
        F::kv_cache c = F::kv_alloc(type, maxd, Hkv, D);
        if (!c.k) { printf("alloc failed\n"); return 1; }
        for (int p = 0; p < maxd; p += CH) {
            const int m = std::min(CH, maxd - p);
            fill_k<<<256, 256>>>(kf, (size_t) m * Hkv * D, 100 + p);
            fill_k<<<256, 256>>>(vf, (size_t) m * Hkv * D, 900 + p);
            F::kv_append(c, kf, vf, p, m, g_s);
        }
        CK(cudaStreamSynchronize(g_s));
        const double rowb = 2.0 * F::kv_row_bytes(type, Hkv, D);
        for (int dpt : depths) {
            for (int T : Ts) {
                printf("%-4s depth %6d T=%d:", type == F::KV_F16 ? "f16" : type == F::KV_Q8_0 ? "q8_0" : "q4_0", dpt, T);
                const float t1 = getenv("MRV_DEFSPLIT") ? 0.0f : time_ms([&] { F::attn_decode(q, c, o, T, dpt, Hq, 1.0f / 16, g_s); });
                printf("  attn_decode %.4f ms (%.0f GB/s) |", t1, rowb * dpt / (t1 * 1e-3) / 1e9);
                for (int sp : splits) {
                    if (sp == 0) continue;
                    F::attn_dec2_set_nsplit(sp);
                    const float t2 = time_ms([&] { F::attn_decode2(q, c, o, T, dpt, nullptr, Hq, 1.0f / 16, g_s); });
                    printf(" s%d %.4f", sp, t2);
                }
                F::attn_dec2_set_nsplit(0);
                printf("\n");
            }
        }
        F::kv_free(c);
    }
    return 0;
}
