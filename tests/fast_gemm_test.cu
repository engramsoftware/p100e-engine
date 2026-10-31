// p100e-fast-gemm-test <model.gguf> [reps]
// Real quantized tensors from a GGUF: fast::gemm_q (all modes) vs a dequant-f32 + cublasSgemm reference.
// For each tensor (first rows: n_out halved like one tensor-parallel rank) and T in {16,48,128,512,2048}:
// accuracy (max |err| / max |ref|, relative RMS) and time / TFLOPS of each mode. mode cublas32 = current
// tp_model::matmul prefill path (dequant f16 + GemmEx COMPUTE_32F).
#include "fast/gemm.cuh"
#include "gguf.h"
#include "convert.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

using namespace p100e;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

static int run(int argc, char ** argv);
int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    try {
        return run(argc, argv);
    } catch (const std::exception & e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}

static int run(int argc, char ** argv) {
    const char * path = argc > 1 ? argv[1] : "model.gguf";
    const int reps = argc > 2 ? atoi(argv[2]) : 10;
    gguf_file g(path);
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    printf("device: %s (sm_%d%d)  model: %s\n", prop.name, prop.major, prop.minor, path);

    // test set: specific tensors + first one of each priority type
    std::vector<const gguf_tensor *> ts;
    auto add = [&](const gguf_tensor * t) { if (t) for (auto * u : ts) if (u == t) return; if (t) ts.push_back(t); };
    add(g.find("blk.3.ffn_gate.weight"));
    add(g.find("blk.3.ffn_down.weight"));
    add(g.find("blk.3.attn_q.weight"));
    add(g.find("blk.4.ssm_out.weight"));
    add(g.find("blk.3.attn_k.weight"));
    for (uint32_t ty : {12u /*q4_K*/, 14u /*q6_K*/, 13u /*q5_K*/, 11u /*q3_K*/, 8u /*q8_0*/}) {
        for (const auto & t : g.tensors) if (t.type == ty && t.n_dims == 2 && t.ne[1] >= 1024) {
            bool have = false;
            for (auto * u : ts) have |= u->type == ty;
            if (!have) add(&t);
            break;
        }
    }

    cublasHandle_t cb;
    cublasCreate(&cb);
    cudaStream_t s;
    CK(cudaStreamCreate(&s));
    cublasSetStream(cb, s);
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));

    const int Ts[] = {16, 48, 128, 512, 2048};
    constexpr int NT = 5;
    const int TMAX = 2048;
    const fast::gemm_mode modes[] = {fast::gemm_mode::cublas32, fast::gemm_mode::fold, fast::gemm_mode::cublas16,
                                     fast::gemm_mode::cublas16a6, fast::gemm_mode::fold32};
    const char * mname[] = {"cublas32(base)", "fold(new)", "cublas16", "cublas16-algo6", "fold32"};
    constexpr int NM = 5;

    double tot_ms[5][NT] = {};
    for (const gguf_tensor * t : ts) {
        const int64_t n_in = t->ne[0];
        const int64_t n_out = std::max<int64_t>(256, t->ne[1] / 2);   // one TP rank's rows
        const qtype_info & ti = get_type_info(t->type);
        const size_t rb = (size_t) (n_in / ti.blck * ti.size);
        printf("\n== %s  %s  n_in %lld  n_out %lld (of %lld)\n", t->name.c_str(), ti.name, (long long) n_in,
               (long long) n_out, (long long) t->ne[1]);
        void * w;
        CK(cudaMalloc(&w, rb * n_out));
        CK(cudaMemcpy(w, t->data, rb * n_out, cudaMemcpyHostToDevice));
        float * wf;
        CK(cudaMalloc(&wf, (size_t) n_in * n_out * 4));
        ggml_get_to_fp32_cuda((ggml_type) t->type)(w, wf, n_in * n_out, s);

        // activations: N(0,1) with a few outlier channels (x30), as in real hidden states
        std::vector<float> hx((size_t) TMAX * n_in);
        std::mt19937 rng(1234);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        for (int64_t i = 0; i < (int64_t) hx.size(); ++i) hx[i] = nd(rng) * (((i % n_in) % 997 == 5) ? 30.0f : 1.0f);
        float *x, *y, *yr;
        CK(cudaMalloc(&x, hx.size() * 4));
        CK(cudaMemcpy(x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&y, (size_t) TMAX * n_out * 4));
        CK(cudaMalloc(&yr, (size_t) TMAX * n_out * 4));
        const size_t sb = fast::gemm_scratch_bytes(n_in, n_out, TMAX) + (size_t) TMAX * n_out * 2 + 4096;
        void * scr;
        CK(cudaMalloc(&scr, sb));
        {   // cost of the f16 dequant alone (part of every mode)
            auto to16 = ggml_get_to_fp16_cuda((ggml_type) t->type);
            to16(w, (half *) scr, n_in * n_out, s);
            CK(cudaEventRecord(e0, s));
            for (int i = 0; i < reps; ++i) to16(w, (half *) scr, n_in * n_out, s);
            CK(cudaEventRecord(e1, s));
            CK(cudaEventSynchronize(e1));
            float ms = 0;
            CK(cudaEventElapsedTime(&ms, e0, e1));
            ms /= reps;
            printf("  dequant->f16 alone: %.3f ms (%.0f GB/s written)\n", ms, n_in * n_out * 2.0 / (ms * 1e6));
        }

        for (int T : Ts) {
            const float one = 1.0f, zero = 0.0f;
            cublasSgemm(cb, CUBLAS_OP_T, CUBLAS_OP_N, (int) n_out, T, (int) n_in, &one, wf, (int) n_in, x, (int) n_in,
                        &zero, yr, (int) n_out);
            std::vector<float> ref((size_t) T * n_out), out((size_t) T * n_out);
            CK(cudaMemcpyAsync(ref.data(), yr, ref.size() * 4, cudaMemcpyDeviceToHost, s));
            CK(cudaStreamSynchronize(s));
            double rmax = 0, r2 = 0;
            for (float v : ref) { rmax = std::max(rmax, (double) fabsf(v)); r2 += (double) v * v; }
            printf("  T=%-5d", T);
            for (int mi = 0; mi < NM; ++mi) {
                CK(cudaMemsetAsync(y, 0xff, (size_t) T * n_out * 4, s));
                fast::gemm_q_mode(modes[mi], t->type, w, n_in, n_out, x, y, T, scr, sb, s, cb);
                CK(cudaMemcpyAsync(out.data(), y, out.size() * 4, cudaMemcpyDeviceToHost, s));
                CK(cudaStreamSynchronize(s));
                double emax = 0, e2 = 0;
                for (size_t i = 0; i < out.size(); ++i) {
                    const double d = std::isfinite(out[i]) ? fabs((double) out[i] - ref[i]) : INFINITY;
                    emax = std::max(emax, d);
                    e2 += d * d;
                }
                for (int i = 0; i < 2; ++i)
                    fast::gemm_q_mode(modes[mi], t->type, w, n_in, n_out, x, y, T, scr, sb, s, cb);
                CK(cudaEventRecord(e0, s));
                for (int i = 0; i < reps; ++i)
                    fast::gemm_q_mode(modes[mi], t->type, w, n_in, n_out, x, y, T, scr, sb, s, cb);
                CK(cudaEventRecord(e1, s));
                CK(cudaEventSynchronize(e1));
                float ms = 0;
                CK(cudaEventElapsedTime(&ms, e0, e1));
                ms /= reps;
                const double tf = 2.0 * T * n_in * n_out / (ms * 1e-3) / 1e12;
                int ti2 = 0;
                while (Ts[ti2] != T) ++ti2;
                tot_ms[mi][ti2] += ms;
                printf("  | %s %.3f ms %.2f TF maxrel %.1e rms %.1e", mname[mi], ms, tf, emax / rmax, sqrt(e2 / r2));
            }
            printf("\n");
            if (T == 128) {   // row-chunked path: minimum scratch (one 128-row tile of f16 weights per chunk)
                const size_t smin = fast::gemm_scratch_min_bytes(n_in, T);
                CK(cudaMemsetAsync(y, 0xff, (size_t) T * n_out * 4, s));
                fast::gemm_q(t->type, w, n_in, n_out, x, y, T, scr, smin, s, cb);
                CK(cudaMemcpyAsync(out.data(), y, out.size() * 4, cudaMemcpyDeviceToHost, s));
                CK(cudaStreamSynchronize(s));
                double emax = 0;
                for (size_t i = 0; i < out.size(); ++i)
                    emax = std::max(emax, std::isfinite(out[i]) ? fabs((double) out[i] - ref[i]) : INFINITY);
                printf("  T=128 chunked (scratch %.2f MiB): maxrel %.1e\n", smin / 1048576.0, emax / rmax);
            }
        }
        cudaFree(w); cudaFree(wf); cudaFree(x); cudaFree(y); cudaFree(yr); cudaFree(scr);
    }
    printf("\n== total over tensors (ms):\n");
    for (int ti2 = 0; ti2 < NT; ++ti2) {
        printf("  T=%-5d", Ts[ti2]);
        for (int mi = 0; mi < NM; ++mi) printf("  %s %.3f", mname[mi], tot_ms[mi][ti2]);
        printf("   speedup fold/base %.2fx\n", tot_ms[0][ti2] / tot_ms[1][ti2]);
    }
    return 0;
}
