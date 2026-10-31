// p100e-fast-sass-test: the register-bank-fixed sm_60 cubins of fold_seg_kernel<8, false> (src/fast/gemm_sass_fold8.h,
// tools/sass/fold_bankfix.py) against the compiled kernel, through fast::fold_gemm (fold_opts::sass):
// bit-exact outputs on a 2-segment problem (T 300, raster 0 and 4), then the time of an ffn_gate|up-sized launch
// (17408 x 5120, T 512) for the compiled kernel and each cubin. Needs a cc 6.0 GPU (elsewhere the cubins are not
// loaded and every arm runs the compiled kernel). Variants: 3 = unpatched cubin (loader / launch path only),
// 1 = bank-fix variant 0, 2 = variant 1 (wide). Progress is printed before each variant (a crash names it).
#include "fast/gemm.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using namespace p100e;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    printf("device: %s (sm_%d%d)%s\n", prop.name, prop.major, prop.minor,
           prop.major == 6 && prop.minor == 0 ? "" : "  (not sm_60: every arm runs the compiled kernel)");
    cudaStream_t s;
    CK(cudaStreamCreate(&s));
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    auto rand_half = [&](size_t n, float sc) {
        std::vector<half> h(n);
        for (auto & v : h) v = __float2half(nd(rng) * sc);
        half * d;
        CK(cudaMalloc(&d, n * 2));
        CK(cudaMemcpy(d, h.data(), n * 2, cudaMemcpyHostToDevice));
        return d;
    };
    int fails = 0;
    {   // exactness
        const int K = 5120, LDX = 5120 + 192, T = 300, M0 = 384, M1 = 256;
        half *w0 = rand_half((size_t) M0 * K, 0.05f), *w1 = rand_half((size_t) M1 * K, 0.05f), *x = rand_half((size_t) T * LDX, 4.0f);
        std::vector<float> hcs(T);
        for (auto & v : hcs) v = ldexpf(1.0f, (int) (rng() % 7) - 3);
        float *cs, *y0, *y1, *r0, *r1;
        CK(cudaMalloc(&cs, T * 4));
        CK(cudaMemcpy(cs, hcs.data(), T * 4, cudaMemcpyHostToDevice));
        for (float ** p : {&y0, &r0}) CK(cudaMalloc(p, (size_t) T * M0 * 4));
        for (float ** p : {&y1, &r1}) CK(cudaMalloc(p, (size_t) T * M1 * 4));
        auto run = [&](int sass, int raster, float * a, float * b) {
            fast::fold_seg segs[2];
            segs[0].w = w0; segs[0].y = a; segs[0].ldy = M0; segs[0].rows = M0;
            segs[1].w = w1; segs[1].y = b; segs[1].ldy = M1; segs[1].rows = M1;
            fast::fold_opts o;
            o.sass = sass;
            o.raster = raster;
            CK(cudaMemsetAsync(a, 0xff, (size_t) T * M0 * 4, s));
            CK(cudaMemsetAsync(b, 0xff, (size_t) T * M1 * 4, s));
            fast::fold_gemm(segs, 2, x, LDX, cs, T, K, s, o);
            CK(cudaStreamSynchronize(s));
        };
        for (int raster : {0, 4}) {
            run(0, raster, r0, r1);
            std::vector<float> ra((size_t) T * M0), rb((size_t) T * M1), a(ra.size()), b(rb.size());
            CK(cudaMemcpy(ra.data(), r0, ra.size() * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(rb.data(), r1, rb.size() * 4, cudaMemcpyDeviceToHost));
            for (int v : {3, 1, 2}) {
                printf("  raster %d, variant %d: running ...", raster, v);
                run(v, raster, y0, y1);
                CK(cudaMemcpy(a.data(), y0, a.size() * 4, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(b.data(), y1, b.size() * 4, cudaMemcpyDeviceToHost));
                long long m = 0;
                for (size_t i = 0; i < a.size(); ++i) m += memcmp(&a[i], &ra[i], 4) != 0;
                for (size_t i = 0; i < b.size(); ++i) m += memcmp(&b[i], &rb[i], 4) != 0;
                printf(" %lld / %zu differ from the compiled kernel %s\n", m, a.size() + b.size(), m ? "FAIL" : "EXACT");
                if (m) ++fails;
            }
        }
        for (void * p : {(void *) w0, (void *) w1, (void *) x, (void *) cs, (void *) y0, (void *) y1, (void *) r0, (void *) r1}) cudaFree(p);
    }
    {   // timing: ffn gate|up shard (2 x 8704 rows), K 5120, T 512 (4 token tiles), raster 4
        const int K = 5120, T = 512, M = 17408;
        half *w = rand_half((size_t) M * K, 0.05f), *x = rand_half((size_t) T * K, 4.0f);
        std::vector<float> hcs(T, 1.0f);
        float *cs, *y;
        CK(cudaMalloc(&cs, T * 4));
        CK(cudaMemcpy(cs, hcs.data(), T * 4, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&y, (size_t) T * M * 4));
        cudaEvent_t e0, e1;
        CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
        for (int v : {0, 3, 1, 2, 0, 3, 1, 2}) {
            fast::fold_seg seg;
            seg.w = w; seg.y = y; seg.ldy = M; seg.rows = M;
            fast::fold_opts o;
            o.sass = v;
            o.raster = 4;
            for (int i = 0; i < 3; ++i) fast::fold_gemm(&seg, 1, x, K, cs, T, K, s, o);
            CK(cudaEventRecord(e0, s));
            for (int i = 0; i < 20; ++i) fast::fold_gemm(&seg, 1, x, K, cs, T, K, s, o);
            CK(cudaEventRecord(e1, s));
            CK(cudaEventSynchronize(e1));
            float ms = 0;
            CK(cudaEventElapsedTime(&ms, e0, e1));
            const double us = ms * 1000 / 20, tf = 2.0 * M * K * (double) T / (us * 1e6);
            printf("  time %-26s %8.1f us  %.2f TFLOPS\n", v == 0 ? "compiled" : v == 3 ? "cubin, unpatched" : v == 1 ? "cubin, bank-fix" : "cubin, bank-fix wide",
                   us, tf);
        }
        cudaFree(w); cudaFree(x); cudaFree(cs); cudaFree(y);
    }
    printf("%s (%d failing checks)\n", fails ? "FAILED" : "ALL PASSED", fails);
    return fails ? 2 : 0;
}
