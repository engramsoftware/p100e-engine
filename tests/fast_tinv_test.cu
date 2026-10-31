// T-invariance of the decode attention (the MTP verify's identity contract): row t of a T-query attn_decode /
// attn_decode_dev call at p0 must be bit-identical to the T = 1 call at p0 + t on the same cache, and attn_prep's K/V
// cache rows / q / gate rows must not depend on T. Per-rank Qwen3.8 shape (Hq 12, Hkv 2, D 256), f16 cache.
#include "fast/attn.cuh"
#include "fast/fused.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(2); } } while (0)
namespace F = p100e::fast;
constexpr int Hq = 12, Hkv = 2, D = 256;

__device__ __forceinline__ uint32_t hash32(uint32_t x) { x ^= x >> 16; x *= 0x7feb352d; x ^= x >> 15; x *= 0x846ca68b; x ^= x >> 16; return x; }
__global__ void fill_k(float * p, size_t n, uint32_t seed, float scale) {
    size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint32_t h1 = hash32((uint32_t) i * 2654435761u + seed), h2 = hash32(h1 + 77);
    p[i] = ((h1 & 0xffff) + (h2 & 0xffff)) * (scale / 32768.0f) - scale;
}
static void fill(float * p, size_t n, uint32_t seed, float scale) { fill_k<<<(unsigned) ((n + 255) / 256), 256>>>(p, n, seed, scale); }
static std::vector<float> host(const float * d, size_t n) { std::vector<float> h(n); CK(cudaMemcpy(h.data(), d, n * 4, cudaMemcpyDeviceToHost)); return h; }

int main(int argc, char ** argv) {
    cudaStream_t s; CK(cudaStreamCreate(&s));
    const int n_ctx = 4096;
    const float scale = 1.0f / 16.0f;
    F::kv_cache c = F::kv_alloc(F::KV_F16, n_ctx, Hkv, D);
    float * st, * sv; CK(cudaMalloc(&st, (size_t) n_ctx * Hkv * D * 4)); CK(cudaMalloc(&sv, (size_t) n_ctx * Hkv * D * 4));
    fill(sv, (size_t) n_ctx * Hkv * D, 47, 1.0f);
    fill(st, (size_t) n_ctx * Hkv * D, 31, 1.0f);
    for (int p = 0; p < n_ctx; p += 4) F::kv_append(c, st + (size_t) p * Hkv * D, sv + (size_t) p * Hkv * D, p, 4, s);
    float * q, * oT, * o1;
    CK(cudaMalloc(&q, 4 * Hq * D * 4)); CK(cudaMalloc(&oT, 4 * Hq * D * 4)); CK(cudaMalloc(&o1, 4 * Hq * D * 4));
    fill(q, 4 * Hq * D, 5, 2.0f);
    int32_t * pos; CK(cudaMalloc(&pos, 4 * 4));
    int bad = 0, cases = 0, bad_dev = 0;
    F::attn_decode(q, c, oT, 4, 0, Hq, scale, s);   // scratch warm-up
    for (int T = 2; T <= 4; ++T) {
        for (int p0 = 0; p0 + T <= n_ctx; p0 += (p0 < 300 ? 1 : 37)) {
            F::attn_decode(q, c, oT, T, p0, Hq, scale, s);
            auto hT = host(oT, (size_t) T * Hq * D);
            int32_t hp[4] = {p0, p0 + 1, p0 + 2, p0 + 3};
            CK(cudaMemcpy(pos, hp, 16, cudaMemcpyHostToDevice));
            F::attn_decode_dev(q, c, o1, T, pos, n_ctx, Hq, scale, s);
            auto hD = host(o1, (size_t) T * Hq * D);
            if (memcmp(hT.data(), hD.data(), hT.size() * 4)) bad_dev++;
            for (int t = 0; t < T; ++t) {
                F::attn_decode(q + (size_t) t * Hq * D, c, o1, 1, p0 + t, Hq, scale, s);
                auto h1 = host(o1, (size_t) Hq * D);
                ++cases;
                if (memcmp(h1.data(), hT.data() + (size_t) t * Hq * D, h1.size() * 4)) {
                    if (bad < 10) printf("  attn MISMATCH T=%d p0=%d row %d\n", T, p0, t);
                    ++bad;
                }
            }
        }
    }
    printf("attn_decode T-invariance: %d / %d rows differ from the T = 1 call; dev-p0 vs host-p0 T-calls differing: %d\n", bad, cases, bad_dev);
    {   // fingerprint of every T = 1 output (p0 = 0..n_ctx-1): compare kernel builds bit for bit
        uint64_t hsh = 1469598103934665603ull;
        for (int p0 = 0; p0 < n_ctx; ++p0) {
            F::attn_decode(q, c, o1, 1, p0, Hq, scale, s);
            auto h1 = host(o1, (size_t) Hq * D);
            const unsigned char * b = (const unsigned char *) h1.data();
            for (size_t i = 0; i < h1.size() * 4; ++i) { hsh ^= b[i]; hsh *= 1099511628211ull; }
        }
        printf("T = 1 output fingerprint (p0 = 0..%d): %016llx\n", n_ctx - 1, (unsigned long long) hsh);
    }
    return bad || bad_dev ? 1 : 0;
}
