// p100e-fast-gdn-chunk-test: chunked gated delta net (fast::gdn_scan_chunked) vs the sequential scan
// (fast::gdn_scan) and an fp64 sequential reference, on the tensor-parallel shard shape (Hk 8, Hv 24, D 128).
// Random inputs shaped like gdn_prep's outputs (q, k L2-normalized, q scaled by 1/sqrt(D); decay in (0.6, 1]; beta in
// (0.05, 0.95)), a random start state. Per T: NMSE of the gated-normed output and of the final state against the fp64
// reference for the sequential scan and both chunked modes, and the time of each. Passes when the chunked NMSE is at
// most 4x the sequential scan's (or below 1e-12).
#include "fast/gdn_chunked.cuh"
#include "fast/ssm.cuh"

#include <algorithm>
#include <chrono>
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
static double nmse(const std::vector<float> & a, const std::vector<double> & ref) {
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); ++i) { const double d = a[i] - ref[i]; num += d * d; den += ref[i] * ref[i]; }
    return den > 0 ? num / den : num;
}

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    printf("device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    cudaStream_t s;
    CK(cudaStreamCreate(&s));
    const int Hk = 8, Hv = 24, D = 128, C = 2 * Hk * D + Hv * D;
    const float eps = 1e-6f;
    std::mt19937 rng(99);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_real_distribution<float> ud(0.0f, 1.0f);
    int fails = 0;
    for (int T : {64, 100, 512, 2048}) {
        std::vector<float> y((size_t) T * C), dec((size_t) T * Hv), bet((size_t) T * Hv), z((size_t) T * Hv * D), nw(D),
            st0((size_t) Hv * D * D);
        for (int t = 0; t < T; ++t) {
            for (int part = 0; part < 2; ++part)
                for (int h = 0; h < Hk; ++h) {   // q (part 0, scaled) and k heads, L2-normalized
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
        for (auto & v : z) v = nd(rng);
        for (auto & v : nw) v = 1.0f + 0.1f * nd(rng);
        for (auto & v : st0) v = 0.1f * nd(rng);
        float *dy = up(y), *ddec = up(dec), *dbet = up(bet), *dz = up(z), *dnw = up(nw);
        const size_t nout = (size_t) T * Hv * D, nst = st0.size();
        // fp64 reference: raw output + state, then the gated norm on the host in double
        float *rst = up(st0), *rout;
        CK(cudaMalloc(&rout, nout * 4));
        fast::gdn_scan_ref_f64(dy, ddec, dbet, rst, rout, T, Hk, Hv, s);
        CK(cudaStreamSynchronize(s));
        const std::vector<float> ro = dl(rout, nout), rs = dl(rst, nst);
        std::vector<double> ref_out(nout), ref_st(rs.begin(), rs.end());
        for (size_t row = 0; row < (size_t) T * Hv; ++row) {
            double ss = 0;
            for (int i = 0; i < D; ++i) ss += (double) ro[row * D + i] * ro[row * D + i];
            const double sc = 1.0 / std::sqrt(ss / D + eps);
            const size_t t = row / Hv, h = row % Hv;
            for (int i = 0; i < D; ++i) {
                const double zz = z[t * Hv * D + h * D + i];
                ref_out[row * D + i] = ro[row * D + i] * sc * nw[i] * (zz / (1.0 + std::exp(-zz)));
            }
        }
        void * scr;
        const size_t sb = fast::gdn_chunked_scratch_bytes(T, Hk, Hv);
        CK(cudaMalloc(&scr, sb));
        double nm_seq_o = 0, nm_seq_s = 0;
        for (int variant = 0; variant < 3; ++variant) {
            float * dst = up(st0), *dout;
            CK(cudaMalloc(&dout, nout * 4));
            auto run = [&]() {
                if (variant == 0) fast::gdn_scan(dy, ddec, dbet, dz, dnw, dst, dout, T, Hk, Hv, D, eps, s);
                else if (!fast::gdn_scan_chunked(dy, ddec, dbet, dz, dnw, dst, dout, T, Hk, Hv, D, eps, scr, sb, s, variant)) {
                    fprintf(stderr, "gdn_scan_chunked refused T=%d\n", T); exit(1);
                }
            };
            run();
            CK(cudaStreamSynchronize(s));
            const std::vector<float> o = dl(dout, nout), st = dl(dst, nst);
            const double no = nmse(o, ref_out), ns = nmse(st, ref_st);
            // timing (the state is overwritten each call; values do not matter for the time)
            cudaEvent_t e0, e1;
            CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
            const int reps = T >= 1024 ? 5 : 20;
            CK(cudaEventRecord(e0, s));
            for (int r = 0; r < reps; ++r) run();
            CK(cudaEventRecord(e1, s));
            CK(cudaEventSynchronize(e1));
            float ms = 0;
            CK(cudaEventElapsedTime(&ms, e0, e1));
            if (variant == 0) { nm_seq_o = no; nm_seq_s = ns; }
            const bool ok = variant == 0 || ((no <= 4 * nm_seq_o || no < 1e-12) && (ns <= 4 * nm_seq_s || ns < 1e-12));
            printf("  T=%4d %-16s out NMSE %.2e  state NMSE %.2e  %8.3f ms  %s\n", T,
                   variant == 0 ? "sequential scan" : variant == 1 ? "chunked fp32" : "chunked fp64", no, ns, ms / reps,
                   ok ? (variant ? "PASS" : "") : "FAIL");
            if (!ok) ++fails;
            cudaFree(dst); cudaFree(dout);
        }
        for (void * p : {(void *) dy, (void *) ddec, (void *) dbet, (void *) dz, (void *) dnw, (void *) rst, (void *) rout, scr}) cudaFree(p);
    }
    printf("%s (%d failing checks)\n", fails ? "FAILED" : "ALL PASSED", fails);
    return fails ? 2 : 0;
}
