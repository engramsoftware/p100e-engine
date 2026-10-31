// p100e-gdn-chunk-zero-decay-test: the chunked gated delta net (fast::gdn_scan_chunked, both modes) with some decays
// exactly 0 (gdn_prep's expf(sp * ssm_a) underflows once sp * |A| > ~104): the output and state must stay finite and
// match the sequential scan (fast::gdn_scan), which handles decay 0 as a state reset. A chunked path that took
// log(0) = -inf would turn that head's output and state into NaN.
#include "fast/gdn_chunked.cuh"
#include "fast/ssm.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using namespace p100e;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

template <typename T> static T * up(const std::vector<T> & h) {
    T * d; CK(cudaMalloc(&d, h.size() * sizeof(T)));
    CK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}
template <typename T> static std::vector<T> dl(const T * d, size_t n) {
    std::vector<T> h(n); CK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost)); return h;
}

int main() {
    cudaStream_t s;
    CK(cudaStreamCreate(&s));
    const int Hk = 8, Hv = 24, D = 128, C = 2 * Hk * D + Hv * D;
    const float eps = 1e-6f;
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_real_distribution<float> ud(0.0f, 1.0f);
    int fails = 0;
    for (int T : {64, 100, 300}) {
        std::vector<float> y((size_t) T * C), dec((size_t) T * Hv), bet((size_t) T * Hv), z((size_t) T * Hv * D), nw(D),
            st0((size_t) Hv * D * D);
        for (int t = 0; t < T; ++t) {
            for (int part = 0; part < 2; ++part)
                for (int h = 0; h < Hk; ++h) {
                    float * p = y.data() + (size_t) t * C + part * Hk * D + h * D;
                    double ss = 0;
                    for (int i = 0; i < D; ++i) { p[i] = nd(rng); ss += (double) p[i] * p[i]; }
                    const float sc = (float) (1.0 / std::sqrt(ss + 1e-6)) * (part == 0 ? 1.0f / std::sqrt((float) D) : 1.0f);
                    for (int i = 0; i < D; ++i) p[i] *= sc;
                }
            for (int i = 2 * Hk * D; i < C; ++i) y[(size_t) t * C + i] = nd(rng);
            for (int h = 0; h < Hv; ++h) {
                dec[(size_t) t * Hv + h] = std::exp(-0.5f * ud(rng) * ud(rng));
                bet[(size_t) t * Hv + h] = 0.05f + 0.9f * ud(rng);
            }
        }
        // hard resets: decay exactly 0 at a few (token, head) cells, chunk starts / middles / ends included
        for (int t : {0, 5, 31, 63, 64, 99}) if (t < T) for (int h : {0, 3, 23}) dec[(size_t) t * Hv + h] = 0.0f;
        for (auto & v : z) v = nd(rng);
        for (auto & v : nw) v = 1.0f + 0.1f * nd(rng);
        for (auto & v : st0) v = 0.1f * nd(rng);
        float *dy = up(y), *ddec = up(dec), *dbet = up(bet), *dz = up(z), *dnw = up(nw);
        const size_t nout = (size_t) T * Hv * D, nst = st0.size();
        void * scr;
        const size_t sb = fast::gdn_chunked_scratch_bytes(T, Hk, Hv);
        CK(cudaMalloc(&scr, sb));
        std::vector<float> o0, s0;
        for (int variant = 0; variant < 3; ++variant) {
            float * dst = up(st0), *dout;
            CK(cudaMalloc(&dout, nout * 4));
            if (variant == 0) fast::gdn_scan(dy, ddec, dbet, dz, dnw, dst, dout, T, Hk, Hv, D, eps, s);
            else if (!fast::gdn_scan_chunked(dy, ddec, dbet, dz, dnw, dst, dout, T, Hk, Hv, D, eps, scr, sb, s, variant)) {
                fprintf(stderr, "gdn_scan_chunked refused T=%d\n", T); exit(1);
            }
            CK(cudaStreamSynchronize(s));
            const std::vector<float> o = dl(dout, nout), st = dl(dst, nst);
            size_t bad = 0; double mo = 0, ms = 0;
            for (float v : o) bad += !std::isfinite(v);
            for (float v : st) bad += !std::isfinite(v);
            if (variant == 0) { o0 = o; s0 = st; }
            for (size_t i = 0; i < nout; ++i) mo = fmax(mo, fabs((double) o[i] - o0[i]));
            for (size_t i = 0; i < nst; ++i) ms = fmax(ms, fabs((double) st[i] - s0[i]));
            const bool ok = bad == 0 && mo < 1e-3 && ms < 1e-3;
            printf("  T=%4d %-16s non-finite %zu  max|d out| vs seq %.2e  max|d state| %.2e  %s\n", T,
                   variant == 0 ? "sequential scan" : variant == 1 ? "chunked fp32" : "chunked fp64", bad, mo, ms, ok ? "PASS" : "FAIL");
            if (!ok) ++fails;
            cudaFree(dst); cudaFree(dout);
        }
        for (void * p : {(void *) dy, (void *) ddec, (void *) dbet, (void *) dz, (void *) dnw, scr}) cudaFree(p);
    }
    printf("%s (%d failing checks)\n", fails ? "FAILED" : "ALL PASSED", fails);
    return fails ? 2 : 0;
}
