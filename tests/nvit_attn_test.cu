// nvit::attn72 (D = 72 maskless attention) vs a double-precision CPU reference on the same f16 inputs, and its speed.
// usage: p100e-nvit-attn-test [N ...]   (default 1024 4144 8160; nq = N / 2 per call like one GPU of the 2-GPU split)
#include "nvit/nvit_kernels.cuh"

#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(2); } } while (0)

using namespace p100e::nvit;

static uint32_t hash32(uint32_t x) { x ^= x >> 16; x *= 0x7feb352d; x ^= x >> 15; x *= 0x846ca68b; x ^= x >> 16; return x; }
static float rnd(uint32_t i, uint32_t seed) {   // roughly normal (sum of 3 uniforms), std ~0.5
    const uint32_t h1 = hash32(i * 2654435761u + seed), h2 = hash32(h1 + 77), h3 = hash32(h2 + 99);
    return ((h1 & 0xffff) + (h2 & 0xffff) + (h3 & 0xffff)) / 65536.0f - 1.5f;
}

int main(int argc, char ** argv) {
    std::vector<int> Ns;
    for (int i = 1; i < argc; ++i) Ns.push_back(atoi(argv[i]));
    if (Ns.empty()) Ns = {1024, 4144, 8160};
    cudaDeviceProp pr;
    CK(cudaGetDeviceProperties(&pr, 0));
    printf("device: %s (sm_%d%d, %d SMs)\n", pr.name, pr.major, pr.minor, pr.multiProcessorCount);
    cudaStream_t s;
    CK(cudaStreamCreate(&s));
    int fails = 0;
    for (int N : Ns) {
        const int nq = N / 2, qrow0 = 0;
        const int64_t hs = (int64_t) N * HD;
        std::vector<half> hq((size_t) NH * nq * HD), hk((size_t) NH * hs), hv((size_t) NH * hs);
        // q pre-scaled into log2 units: logit std ~3 (peaked rows) with a few outlier dims, like real ViT q/k
        for (size_t i = 0; i < hq.size(); ++i) hq[i] = __float2half(rnd((uint32_t) i, 11) * 1.1f * ((i % HD) % 23 == 3 ? 3.0f : 1.0f));
        for (size_t i = 0; i < hk.size(); ++i) hk[i] = __float2half(rnd((uint32_t) i, 22) * 1.0f * ((i % HD) % 23 == 3 ? 3.0f : 1.0f));
        for (size_t i = 0; i < hv.size(); ++i) hv[i] = __float2half(rnd((uint32_t) i, 33) * 4.0f);
        // tiled device layouts (nvit_kernels.cuh)
        const int nqt = (nq + TILE - 1) / TILE, ntk = (N + TILE - 1) / TILE;
        std::vector<half> tq((size_t) NH * nqt * TILE_U32 * 2, __float2half(0.0f)), tk((size_t) NH * ntk * TILE_U32 * 2, __float2half(0.0f)),
            tv((size_t) NH * ntk * TILE_U32 * 2, __float2half(0.0f));
        for (int h = 0; h < NH; ++h) {
            for (int q = 0; q < nq; ++q)
                for (int d = 0; d < HD; ++d)
                    tq[2 * (((size_t) h * nqt + q / TILE) * TILE_U32 + (d / 2) * TILE + q % TILE) + d % 2] = hq[((size_t) h * nq + q) * HD + d];
            for (int k = 0; k < N; ++k)
                for (int d = 0; d < HD; ++d) {
                    tk[2 * (((size_t) h * ntk + k / TILE) * TILE_U32 + (d / 2) * TILE + k % TILE) + d % 2] = hk[(size_t) h * hs + (size_t) k * HD + d];
                    tv[2 * (((size_t) h * ntk + k / TILE) * TILE_U32 + ((k % TILE) / 2) * HD + d) + k % 2] = hv[(size_t) h * hs + (size_t) k * HD + d];
                }
        }
        uint32_t *dq, *dk, *dv;
        float * dout;
        CK(cudaMalloc(&dq, tq.size() * 2)); CK(cudaMalloc(&dk, tk.size() * 2)); CK(cudaMalloc(&dv, tv.size() * 2));
        CK(cudaMalloc(&dout, (size_t) nq * E * 4));
        CK(cudaMemcpy(dq, tq.data(), tq.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dk, tk.data(), tk.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dv, tv.data(), tv.size() * 2, cudaMemcpyHostToDevice));
        const int64_t qhs = (int64_t) nqt * TILE_U32, khs = (int64_t) ntk * TILE_U32;
        attn72(dq, qhs, qrow0, nq, dk, dv, khs, N, NH, dout, E, s);
        CK(cudaStreamSynchronize(s));
        std::vector<float> out((size_t) nq * E);
        CK(cudaMemcpy(out.data(), dout, out.size() * 4, cudaMemcpyDeviceToHost));
        // reference for 48 queries spread over the range (incl. the last), every head
        double max_err = 0, max_ref = 0, se = 0;
        size_t cnt = 0;
        std::vector<double> w(N);
        for (int qi = 0; qi < 48; ++qi) {
            const int q = qi == 47 ? nq - 1 : (int) ((int64_t) qi * nq / 47);
            for (int h = 0; h < NH; ++h) {
                const half * qp = &hq[((size_t) h * nq + q) * HD];
                double mx = -1e300;
                for (int k = 0; k < N; ++k) {
                    const half * kp = &hk[(size_t) h * hs + (size_t) k * HD];
                    double d = 0;
                    for (int c = 0; c < HD; ++c) d += (double) __half2float(qp[c]) * __half2float(kp[c]);
                    w[k] = d;
                    mx = std::max(mx, d);
                }
                double sum = 0;
                for (int k = 0; k < N; ++k) { w[k] = std::exp2(w[k] - mx); sum += w[k]; }
                for (int c = 0; c < HD; ++c) {
                    double o = 0;
                    for (int k = 0; k < N; ++k) o += w[k] * __half2float(hv[(size_t) h * hs + (size_t) k * HD + c]);
                    o /= sum;
                    const double g = out[(size_t) q * E + h * HD + c];
                    max_err = std::max(max_err, std::fabs(g - o));
                    max_ref = std::max(max_ref, std::fabs(o));
                    se += (g - o) * (g - o);
                    ++cnt;
                }
            }
        }
        const double rel = max_err / max_ref, rms = std::sqrt(se / cnt);
        const bool ok = rel < 2e-2 && rms / max_ref < 2e-3;
        if (!ok) ++fails;
        // split key range (two passes with the carried softmax state) == one pass, up to rounding
        {
            float * dout2, * st;
            CK(cudaMalloc(&dout2, (size_t) nq * E * 4));
            CK(cudaMalloc(&st, (size_t) nq * NH * 2 * 4));
            const int mid = ntk / 2 + 1;
            attn72(dq, qhs, qrow0, nq, dk, dv, khs, N, NH, dout2, E, s, mid, ntk, 1, st);   // second half first
            attn72(dq, qhs, qrow0, nq, dk, dv, khs, N, NH, dout2, E, s, 0, mid, 2, st);
            CK(cudaStreamSynchronize(s));
            std::vector<float> out2((size_t) nq * E);
            CK(cudaMemcpy(out2.data(), dout2, out2.size() * 4, cudaMemcpyDeviceToHost));
            double md = 0;
            for (size_t i = 0; i < out2.size(); ++i) md = std::max(md, (double) std::fabs(out2[i] - out[i]));
            const bool ok2 = md < 1e-2 * max_ref;
            if (!ok2) ++fails;
            printf("N %5d split key range [%d, %d) + [0, %d): max |split - one pass| %.3e  %s\n", N, mid, ntk, mid, md, ok2 ? "OK" : "FAIL");
            CK(cudaFree(dout2)); CK(cudaFree(st));
        }
        // timing
        cudaEvent_t e0, e1;
        CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
        const int reps = N >= 8000 ? 5 : 20;
        CK(cudaEventRecord(e0, s));
        for (int r = 0; r < reps; ++r) attn72(dq, qhs, qrow0, nq, dk, dv, khs, N, NH, dout, E, s);
        CK(cudaEventRecord(e1, s));
        CK(cudaEventSynchronize(e1));
        float ms = 0;
        CK(cudaEventElapsedTime(&ms, e0, e1));
        ms /= reps;
        const double tf = 4.0 * nq * (double) N * HD * NH / (ms * 1e-3) / 1e12;
        printf("N %5d nq %5d x %d heads: max err %.3e (rel to max|ref| %.3e: %.2e), rms %.3e  %s   %.3f ms  %.2f TFLOPS\n",
               N, nq, NH, max_err, max_ref, rel, rms, ok ? "OK" : "FAIL", ms, tf);
        CK(cudaFree(dq)); CK(cudaFree(dk)); CK(cudaFree(dv)); CK(cudaFree(dout));
    }
    printf(fails ? "SOME CHECKS FAILED\n" : "ALL CHECKS PASSED\n");
    return fails ? 1 : 0;
}
