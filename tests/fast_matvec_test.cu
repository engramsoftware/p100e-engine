// p100e-fast-matvec-test: p100e::fast::matvec vs dequant + cublasSgemm reference on real GGUF tensors.
// usage: p100e-fast-matvec-test file.gguf [file2.gguf ...]    (one tensor per quant type, types seen once)
#include "fast/matvec.cuh"
#include "gguf.h"
#include "convert.cuh"
#include <cublas_v2.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA error %s at %s:%d: %s\n", \
    cudaGetErrorName(e_), __FILE__, __LINE__, cudaGetErrorString(e_)); exit(2); } } while (0)

using namespace p100e;

static const char * tname(uint32_t t) {
    switch (t) {
        case 2: return "q4_0"; case 3: return "q4_1"; case 6: return "q5_0"; case 7: return "q5_1"; case 8: return "q8_0";
        case 10: return "q2_K"; case 11: return "q3_K"; case 12: return "q4_K"; case 13: return "q5_K"; case 14: return "q6_K";
        case 16: return "iq2_xxs"; case 17: return "iq2_xs"; case 18: return "iq3_xxs"; case 19: return "iq1_s"; case 20: return "iq4_nl";
        case 21: return "iq3_s"; case 22: return "iq2_s"; case 23: return "iq4_xs"; case 29: return "iq1_m"; case 39: return "mxfp4";
        default: return "?";
    }
}

static int blck_of(uint32_t t) {
    switch (t) { case 2: case 3: case 6: case 7: case 8: case 20: case 39: return 32; default: return 256; }
}

struct result { double max_rel; double ms; };

template <class F> static float time_ms(cudaStream_t s, int reps, F && fn) {
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    for (int i = 0; i < 10; ++i) fn();
    CK(cudaStreamSynchronize(s));
    CK(cudaEventRecord(a, s));
    for (int i = 0; i < reps; ++i) fn();
    CK(cudaEventRecord(b, s));
    CK(cudaEventSynchronize(b));
    float ms = 0; CK(cudaEventElapsedTime(&ms, a, b));
    CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
    return ms / reps;
}

static double max_rel(const std::vector<float> & y, const std::vector<float> & ref) {
    double mx = 0, me = 0;
    for (size_t i = 0; i < ref.size(); ++i) { mx = std::max(mx, (double) std::fabs(ref[i])); me = std::max(me, (double) std::fabs(y[i] - ref[i])); }
    return mx > 0 ? me / mx : me;
}

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s file.gguf [...]\n", argv[0]); return 1; }
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    printf("device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    cudaStream_t s; CK(cudaStreamCreate(&s));
    cublasHandle_t cb; cublasCreate(&cb); cublasSetStream(cb, s); cublasSetMathMode(cb, CUBLAS_PEDANTIC_MATH);
    const int REPS = 100;

    std::set<uint32_t> done;
    int fails = 0;
    printf("%-9s %-34s %7s %7s %-9s %-9s | %9s %9s %9s %9s | %8s %8s %8s | %8s %8s\n", "type", "tensor", "n_in", "n_out", "layout", "kernel",
           "rel T=1", "rel T=2", "rel T=4", "rel T=8", "GB/s T1", "GB/s T4", "GB/s T8", "gen rel", "gen GB/s");
    for (int ai = 1; ai < argc; ++ai) {
        gguf_file g(argv[ai]);
        std::set<uint32_t> types;
        for (auto & t : g.tensors) if (t.n_dims == 2 && t.type != 0 && t.type != 1 && t.type != 30) types.insert(t.type);
        for (uint32_t ty : types) {
            if (done.count(ty) || !fast::has_fast_path(ty)) continue;
            // pick a tensor: n_in % 256 == 0, n_out % 4 == 0, f32 matrix <= 100M elements, prefer the largest in range
            const gguf_tensor * best = nullptr;
            for (auto & t : g.tensors) {
                if (t.type != ty || t.n_dims != 2) continue;
                if (t.ne[0] % blck_of(ty) != 0 || t.nelements() > 100000000LL) continue;
                const bool good = t.ne[0] % 256 == 0 && t.ne[1] % 4 == 0;
                const bool bgood = best && best->ne[0] % 256 == 0 && best->ne[1] % 4 == 0;
                if (!best || (good && !bgood) || (good == bgood && t.nbytes > best->nbytes)) best = &t;
            }
            if (!best) continue;
            done.insert(ty);
            const int64_t n_in = best->ne[0], n_out = best->ne[1];
            const size_t bytes = best->nbytes;

            void *d_raw, *d_pack; float *d_w, *d_x, *d_ref, *d_y; void * d_scr;
            CK(cudaMalloc(&d_raw, bytes)); CK(cudaMalloc(&d_pack, bytes));
            CK(cudaMemcpy(d_raw, best->data, bytes, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(d_pack, d_raw, bytes, cudaMemcpyDeviceToDevice));
            CK(cudaMalloc(&d_w, (size_t) n_in * n_out * 4));
            CK(cudaMalloc(&d_x, (size_t) 8 * n_in * 4)); CK(cudaMalloc(&d_ref, (size_t) 8 * n_out * 4)); CK(cudaMalloc(&d_y, (size_t) 8 * n_out * 4));
            const size_t scr = fast::matvec_scratch_bytes(n_in, 8);
            CK(cudaMalloc(&d_scr, scr));

            std::vector<float> hx((size_t) 8 * n_in);
            srand(1234 + ty);
            for (auto & v : hx) { float u = 0; for (int k = 0; k < 4; ++k) u += rand() / (float) RAND_MAX - 0.5f; v = u * 1.7f; }
            CK(cudaMemcpy(d_x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice));

            to_fp32_cuda_t f = ggml_get_to_fp32_cuda((ggml_type) ty);
            f(d_raw, d_w, n_in * n_out, s);
            const float one = 1.0f, zero = 0.0f;
            fast::packed_weight pw = fast::pack_weight(ty, d_pack, n_in, n_out, s);
            fast::packed_weight gw = pw; gw.data = d_raw; gw.layout = fast::LAYOUT_RAW;
            CK(cudaStreamSynchronize(s));

            double rel[4]; const int Ts[4] = {1, 2, 4, 8};
            for (int k = 0; k < 4; ++k) {
                const int T = Ts[k];
                cublasSgemm(cb, CUBLAS_OP_T, CUBLAS_OP_N, (int) n_out, T, (int) n_in, &one, d_w, (int) n_in, d_x, (int) n_in, &zero, d_ref, (int) n_out);
                CK(cudaMemset(d_y, 0xff, (size_t) 8 * n_out * 4));
                fast::matvec(pw, d_x, d_y, T, d_scr, s);
                CK(cudaStreamSynchronize(s)); CK(cudaGetLastError());
                std::vector<float> y((size_t) T * n_out), r((size_t) T * n_out);
                CK(cudaMemcpy(y.data(), d_y, y.size() * 4, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(r.data(), d_ref, r.size() * 4, cudaMemcpyDeviceToHost));
                rel[k] = max_rel(y, r);
                if (!(rel[k] < 0.03)) ++fails;   // NaN also fails
            }
            // generic kernel on the untouched bytes (T = 4)
            double grel; float gms;
            {
                const int T = 4;
                cublasSgemm(cb, CUBLAS_OP_T, CUBLAS_OP_N, (int) n_out, T, (int) n_in, &one, d_w, (int) n_in, d_x, (int) n_in, &zero, d_ref, (int) n_out);
                CK(cudaMemset(d_y, 0xff, (size_t) 8 * n_out * 4));
                fast::matvec(gw, d_x, d_y, T, d_scr, s);
                CK(cudaStreamSynchronize(s)); CK(cudaGetLastError());
                std::vector<float> y((size_t) T * n_out), r((size_t) T * n_out);
                CK(cudaMemcpy(y.data(), d_y, y.size() * 4, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(r.data(), d_ref, r.size() * 4, cudaMemcpyDeviceToHost));
                grel = max_rel(y, r);
                if (!(grel < 1e-4)) ++fails;
                gms = time_ms(s, REPS, [&] { fast::matvec(gw, d_x, d_y, 1, d_scr, s); });
            }
            double gbs[3]; const int tt[3] = {1, 4, 8};
            for (int k = 0; k < 3; ++k) {
                const float ms = time_ms(s, REPS, [&] { fast::matvec(pw, d_x, d_y, tt[k], d_scr, s); });
                gbs[k] = (double) bytes / (ms * 1e-3) / 1e9;
            }
            const double ggbs = (double) bytes / (gms * 1e-3) / 1e9;
            printf("%-9s %-34s %7lld %7lld %-9s %-9s | %9.2e %9.2e %9.2e %9.2e | %8.1f %8.1f %8.1f | %8.1e %8.1f\n", tname(ty), best->name.c_str(),
                   (long long) n_in, (long long) n_out, fast::kernel_name(pw) == std::string("generic") ? "raw" : "repacked", fast::kernel_name(pw),
                   rel[0], rel[1], rel[2], rel[3], gbs[0], gbs[1], gbs[2], grel, ggbs);
            fflush(stdout);
            cudaFree(d_raw); cudaFree(d_pack); cudaFree(d_w); cudaFree(d_x); cudaFree(d_ref); cudaFree(d_y); cudaFree(d_scr);
        }
    }
    printf("%s (%d failures; pass = fast rel < 3e-2, generic rel < 1e-4)\n", fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}
