// k::attention_fast (online softmax, warps over keys; P100E_FN_ATTN_FAST=1) vs the reference k::attention on f32
// caches: max abs / rel error per shape, T-invariance (row t of a T-row call == the T = 1 call at p0 + t, bit for bit),
// timing of both, and an FNV hash of the reference output (compare across builds: unchanged for contexts that fit
// the reference's shared-memory score buffer). Long contexts check that attention() routes instead of failing.
// Usage: p100e-ref-attn-fast-test [device]
#include "kernels.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <functional>
#include <stdexcept>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(2); } } while (0)

namespace K = p100e::k;
static int g_fail = 0;
#ifdef OLD_KERNELS   // build against the previous src/kernels.cu (no attention_fast) to compare the reference hashes
namespace p100e::k { bool attention_fast(const float *, const float *, const float *, float *, int, int, int, int, int, float, cudaStream_t) { return false; } }
#endif

__device__ __forceinline__ uint32_t hash32(uint32_t x) { x ^= x >> 16; x *= 0x7feb352d; x ^= x >> 15; x *= 0x846ca68b; x ^= x >> 16; return x; }
__global__ void fill_k(float * p, size_t n, uint32_t seed, float scale) {
    size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint32_t h1 = hash32((uint32_t) i * 2654435761u + seed), h2 = hash32(h1 + 77), h3 = hash32(h2 + 99);
    p[i] = ((h1 & 0xffff) + (h2 & 0xffff) + (h3 & 0xffff)) * (scale / 32768.0f) - 3.0f * scale;
}
static void fill(float * p, size_t n, uint32_t seed, float scale) { fill_k<<<(unsigned) ((n + 255) / 256), 256>>>(p, n, seed, scale); }

static float time_ms(const std::function<void()> & f, int it = 20) {
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    f(); CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(a));
    for (int i = 0; i < it; ++i) f();
    CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
    float ms = 0; CK(cudaEventElapsedTime(&ms, a, b));
    cudaEventDestroy(a); cudaEventDestroy(b);
    return ms / it;
}

static uint64_t fnv(const std::vector<float> & v) {
    uint64_t h = 1469598103934665603ull;
    const unsigned char * p = (const unsigned char *) v.data();
    for (size_t i = 0; i < v.size() * 4; ++i) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}

static void run(int Hq, int Hkv, int D, int T, int p0, bool timing) {
    const int n = p0 + T;
    const size_t nq = (size_t) T * Hq * D, nc = (size_t) n * Hkv * D;
    float * q, * kc, * vc, * a, * b, * o1;
    CK(cudaMalloc(&q, nq * 4)); CK(cudaMalloc(&kc, nc * 4)); CK(cudaMalloc(&vc, nc * 4));
    CK(cudaMalloc(&a, nq * 4)); CK(cudaMalloc(&b, nq * 4)); CK(cudaMalloc(&o1, (size_t) Hq * D * 4));
    fill(q, nq, 1 + D, 1.0f); fill(kc, nc, 2 + p0, 1.0f); fill(vc, nc, 3 + T, 1.0f);
    CK(cudaMemset(a, 0xff, nq * 4)); CK(cudaMemset(b, 0xff, nq * 4));
    const float scale = 1.0f / sqrtf((float) D);
    bool routed = false;
    try { K::attention(q, kc, vc, a, T, p0, Hq, Hkv, D, scale, 0); }
    catch (const std::exception & e) { printf("  attention threw: %s\n", e.what()); routed = true; }
    CK(cudaGetLastError());
    const bool ok = K::attention_fast(q, kc, vc, b, T, p0, Hq, Hkv, D, scale, 0);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    std::vector<float> ha(nq), hb(nq);
    CK(cudaMemcpy(ha.data(), a, nq * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(hb.data(), b, nq * 4, cudaMemcpyDeviceToHost));
    double mx = 0, mref = 0; bool fin = true;
    for (size_t i = 0; i < nq; ++i) {
        if (!std::isfinite(ha[i]) || !std::isfinite(hb[i])) fin = false;
        mx = fmax(mx, fabs((double) ha[i] - hb[i])); mref = fmax(mref, fabs((double) ha[i]));
    }
    // T-invariance of the fast kernel (MTP verify): row t == the T = 1 call at p0 + t
    bool tinv = true;
    if (ok) for (int t = 0; t < T; ++t) {
        K::attention_fast(q + (size_t) t * Hq * D, kc, vc, o1, 1, p0 + t, Hq, Hkv, D, scale, 0);
        std::vector<float> h1((size_t) Hq * D);
        CK(cudaMemcpy(h1.data(), o1, h1.size() * 4, cudaMemcpyDeviceToHost));
        if (memcmp(h1.data(), hb.data() + (size_t) t * Hq * D, h1.size() * 4)) tinv = false;
    }
    const bool pass = ok && fin && !routed && mx <= 2e-5 * fmax(1.0, mref) && tinv;
    printf("Hq=%2d Hkv=%d D=%3d T=%d p0=%6d  fast=%d max|ref-fast| %.3g (max|ref| %.3g) T-inv %s  ref hash %016llx  %s\n",
           Hq, Hkv, D, T, p0, ok, mx, mref, tinv ? "ok" : "DIFF", (unsigned long long) fnv(ha), pass ? "ok" : "FAIL");
    if (!pass) ++g_fail;
    if (timing) {
        const float tr = time_ms([&] { K::attention(q, kc, vc, a, T, p0, Hq, Hkv, D, scale, 0); }, 5);
        const float tf = time_ms([&] { K::attention_fast(q, kc, vc, b, T, p0, Hq, Hkv, D, scale, 0); });
        printf("    time: reference %.4f ms, fast %.4f ms (%.1fx)\n", tr, tf, tr / tf);
    }
    cudaFree(q); cudaFree(kc); cudaFree(vc); cudaFree(a); cudaFree(b); cudaFree(o1);
}

int main(int argc, char ** argv) {
    const int dev = argc > 1 ? atoi(argv[1]) : 0;
    CK(cudaSetDevice(dev));
    cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, dev));
    printf("device %d: %s (sm_%d%d, opt-in smem %zu B)\n", dev, pr.name, pr.major, pr.minor, pr.sharedMemPerBlockOptin);
    const int Ds[] = {64, 96, 128, 256, 512};
    for (int D : Ds)
        for (int T : {1, 2, 4})
            for (int p0 : {0, 1, 37, 300, 2047}) run(8, 2, D, T, p0, false);
    run(16, 16, 128, 1, 999, false);   // no GQA
    run(6, 1, 256, 3, 511, false);      // GQA 6
    // timing at decode shapes (D 256, GQA 4 / 8): 2K keys (the QSA window) and 8K
    for (int p0 : {511, 2047, 8191}) { run(16, 4, 256, 1, p0, true); run(16, 2, 128, 1, p0, true); }
    run(16, 4, 256, 4, 2047, true);
    // contexts past the default 48 KB score buffer: the reference raises its opt-in limit or routes to the fast kernel
    for (int p0 : {12300, 20000, 30000}) run(8, 2, 256, 1, p0, false);
    printf("%s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
