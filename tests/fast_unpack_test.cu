// p100e-fast-unpack-test <model.gguf> [reps]
// Real tensors of each quant type (q3_K q4_K q5_K q6_K q8_0) cut to tensor-parallel-like shapes, plus synthetic
// q4_0 rows (the test model has none) and q6_K at n_in 8704 (the ffn_down shard: padded rows, q6r3; q6r2 with P100E_Q6_PAD=0):
//  (a) fast::packed_to_f16 on the packed weight, over several row chunks (odd starts), vs ggml to-f16 of the raw
//      bytes: exact compare of the f16 bit patterns (mismatch count), for both implementations (UNPACK_SMEM and
//      UNPACK_DIRECT, the fork-style direct kernels) and their timings;
//  (b) fast::gemm_packed vs fast::gemm_q on the raw bytes for T = 16, 128, 512 (max rel diff, timings), plus a
//      row-chunked gemm_packed run with minimum scratch.
#include "fast/gemm.cuh"
#include "fast/unpack.cuh"
#include "fast/matvec.cuh"
#include "gguf.h"
#include "convert.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

    cublasHandle_t cb;
    cublasCreate(&cb);
    cudaStream_t s;
    CK(cudaStreamCreate(&s));
    cublasSetStream(cb, s);
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));

    const uint32_t types[] = {2u /*q4_0, synthetic*/, 11u /*q3_K*/, 12u /*q4_K*/, 13u /*q5_K*/, 14u /*q6_K*/, 8u /*q8_0*/};
    struct shape { int64_t n_in, n_out; };
    const shape shapes[] = {{5120, 8704}, {8704, 2560}, {3072, 24}, {5120, 2560}, {8704, 24}};
    const int Ts[] = {16, 128, 512};
    const int TMAX = 512;

    long long total_mism = 0, total_vals = 0;
    double worst_gemm = 0;
    int cases = 0;
    for (uint32_t ty : types) {
        const qtype_info & ti = get_type_info(ty);
        for (const shape sh : shapes) {
            // a real tensor of this type that contains the shape (rows x leading columns)
            const gguf_tensor * t = nullptr;
            for (const auto & u : g.tensors)
                if (u.type == ty && u.n_dims == 2 && u.ne[0] >= sh.n_in && u.ne[1] >= sh.n_out) { t = &u; break; }
            // synthetic rows where the test model has no tensor: q4_0 (any shape), q6_K at n_in 8704 (the q6r2 shard layout)
            const bool synth = !t && (ty == 2u || (ty == 14u && sh.n_in == 8704));
            if (!t && !synth) { printf("\n== %s n_in %lld n_out %lld: no tensor of that size, skipped\n", ti.name, (long long) sh.n_in, (long long) sh.n_out); continue; }
            const int64_t n_in = sh.n_in, n_out = sh.n_out;
            const size_t rb = (size_t) (n_in / ti.blck * ti.size);
            std::vector<uint8_t> hw(rb * n_out);
            if (synth) {   // random quant bytes, d (q4_0: offset 0, q6_K: offset 208) a random f16 in +-[2^-12, 2^-4]
                std::mt19937 rq(77 + (unsigned) n_in + (unsigned) n_out);
                const size_t bs = ti.size, doff = ty == 2u ? 0 : 208;
                for (size_t b = 0; b < hw.size() / bs; ++b) {
                    uint8_t * blk = hw.data() + bs * b;
                    for (size_t i = 0; i < bs; ++i) blk[i] = (uint8_t) rq();
                    const half d = __float2half(ldexpf(1.0f + (rq() % 1000) / 1000.0f, -4 - (int) (rq() % 9)) * (rq() & 1 ? -1.0f : 1.0f));
                    memcpy(blk + doff, &d, 2);
                }
            } else {
                const size_t rb_src = (size_t) (t->ne[0] / ti.blck * ti.size);
                // rows [0, n_out) of a 1-row-offset window, leading n_in columns (a column slice keeps whole blocks)
                const int64_t rs = t->ne[1] > n_out ? 1 : 0;
                for (int64_t r = 0; r < n_out; ++r)
                    memcpy(hw.data() + r * rb, (const uint8_t *) t->data + (r + rs) * rb_src, rb);
            }

            // the packed copy uses the loader's row stride (tp_model::pad_slice: q6_K rows at n_in 8704 padded to 7144 B
            // for q6r3, P100E_Q6_PAD); the raw copy keeps the GGUF rows for the ggml references
            const size_t st = fast::pack_row_stride(ty, n_in, n_out);
            std::vector<uint8_t> hp(st * n_out, 0);
            for (int64_t r = 0; r < n_out; ++r) memcpy(hp.data() + r * st, hw.data() + r * rb, rb);
            void *wraw, *wpk;
            CK(cudaMalloc(&wraw, hw.size()));
            CK(cudaMalloc(&wpk, hp.size()));
            CK(cudaMemcpy(wraw, hw.data(), hw.size(), cudaMemcpyHostToDevice));
            CK(cudaMemcpy(wpk, hp.data(), hp.size(), cudaMemcpyHostToDevice));
            fast::packed_weight w = fast::pack_weight(ty, wpk, n_in, n_out, s, st);
            printf("\n== %s (%s)  n_in %lld  n_out %lld  layout %d (%s)  row stride %zu (raw %zu)\n", synth ? "synthetic" : t->name.c_str(), ti.name,
                   (long long) n_in, (long long) n_out, w.layout, fast::kernel_name(w), fast::row_stride(w), rb);

            // (a) exactness over row chunks
            const size_t nval = (size_t) n_in * n_out;
            half *ref16, *out16;
            CK(cudaMalloc(&ref16, nval * 2));
            CK(cudaMalloc(&out16, nval * 2));
            ggml_get_to_fp16_cuda((ggml_type) ty)(wraw, ref16, (int64_t) nval, s);
            std::vector<int64_t> cuts = {0};
            for (int64_t c : {(int64_t) 5, (int64_t) 133, (int64_t) 1000, (int64_t) 2048, (int64_t) 5001})
                if (c < n_out) cuts.push_back(c);
            cuts.push_back(n_out);
            std::vector<uint16_t> a(nval), b(nval);
            CK(cudaMemcpyAsync(a.data(), ref16, nval * 2, cudaMemcpyDeviceToHost, s));
            for (int impl : {fast::UNPACK_SMEM, fast::UNPACK_DIRECT}) {
                CK(cudaMemsetAsync(out16, 0xff, nval * 2, s));
                for (size_t i = 0; i + 1 < cuts.size(); ++i)
                    fast::packed_to_f16(w, cuts[i], cuts[i + 1] - cuts[i], out16 + cuts[i] * n_in, s, impl);
                CK(cudaMemcpyAsync(b.data(), out16, nval * 2, cudaMemcpyDeviceToHost, s));
                CK(cudaStreamSynchronize(s));
                long long mism = 0;
                for (size_t i = 0; i < nval; ++i) mism += a[i] != b[i];
                total_mism += mism;
                total_vals += (long long) nval;
                printf("  (a) packed_to_f16 %s in %zu chunks vs ggml to_f16(raw): %lld / %zu f16 mismatches %s\n",
                       impl == fast::UNPACK_DIRECT ? "direct" : "smem  ", cuts.size() - 1, mism, nval, mism ? "FAIL" : "EXACT");
            }
            {   // dequant cost: ggml on raw, packed_to_f16 smem / direct (whole weight)
                float ms[3] = {0, 0, 0};
                for (int k = 0; k < 3; ++k) {
                    CK(cudaEventRecord(e0, s));
                    for (int i = 0; i < reps; ++i) {
                        if (k == 0) ggml_get_to_fp16_cuda((ggml_type) ty)(wraw, ref16, (int64_t) nval, s);
                        else fast::packed_to_f16(w, 0, n_out, out16, s, k == 1 ? fast::UNPACK_SMEM : fast::UNPACK_DIRECT);
                    }
                    CK(cudaEventRecord(e1, s));
                    CK(cudaEventSynchronize(e1));
                    CK(cudaEventElapsedTime(&ms[k], e0, e1));
                }
                const double gb = (double) hw.size() + nval * 2.0;   // bytes read + written per call
                printf("      dequant time: ggml raw %.3f ms, smem %.3f ms (%.0f GB/s), direct %.3f ms (%.0f GB/s)\n",
                       ms[0] / reps, ms[1] / reps, gb / (ms[1] / reps * 1e6), ms[2] / reps, gb / (ms[2] / reps * 1e6));
            }
            cudaFree(ref16); cudaFree(out16);

            // (b) gemm_packed vs gemm_q(raw)
            std::vector<float> hx((size_t) TMAX * n_in);
            std::mt19937 rng(1234);
            std::normal_distribution<float> nd(0.0f, 1.0f);
            for (int64_t i = 0; i < (int64_t) hx.size(); ++i) hx[i] = nd(rng) * (((i % n_in) % 997 == 5) ? 30.0f : 1.0f);
            float *x, *y, *yr;
            CK(cudaMalloc(&x, hx.size() * 4));
            CK(cudaMemcpy(x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice));
            CK(cudaMalloc(&y, (size_t) TMAX * n_out * 4));
            CK(cudaMalloc(&yr, (size_t) TMAX * n_out * 4));
            const size_t sb = fast::gemm_packed_scratch_bytes(n_in, n_out, TMAX);
            void * scr;
            CK(cudaMalloc(&scr, sb));
            for (int T : Ts) {
                if (synth && T != 128) continue;
                fast::gemm_q(ty, wraw, n_in, n_out, x, yr, T, scr, sb, s, cb);
                CK(cudaMemsetAsync(y, 0xff, (size_t) T * n_out * 4, s));
                fast::gemm_packed(w, x, y, T, scr, sb, s, cb);
                std::vector<float> ref((size_t) T * n_out), out((size_t) T * n_out);
                CK(cudaMemcpyAsync(ref.data(), yr, ref.size() * 4, cudaMemcpyDeviceToHost, s));
                CK(cudaMemcpyAsync(out.data(), y, out.size() * 4, cudaMemcpyDeviceToHost, s));
                CK(cudaStreamSynchronize(s));
                double rmax = 0, emax = 0;
                for (size_t i = 0; i < out.size(); ++i) {
                    rmax = std::max(rmax, (double) fabsf(ref[i]));
                    emax = std::max(emax, std::isfinite(out[i]) ? fabs((double) out[i] - ref[i]) : INFINITY);
                }
                const double rel = emax / rmax;
                worst_gemm = std::max(worst_gemm, rel);
                float msq = 0, msp = 0;
                CK(cudaEventRecord(e0, s));
                for (int i = 0; i < reps; ++i) fast::gemm_q(ty, wraw, n_in, n_out, x, yr, T, scr, sb, s, cb);
                CK(cudaEventRecord(e1, s));
                CK(cudaEventSynchronize(e1));
                CK(cudaEventElapsedTime(&msq, e0, e1));
                CK(cudaEventRecord(e0, s));
                for (int i = 0; i < reps; ++i) fast::gemm_packed(w, x, y, T, scr, sb, s, cb);
                CK(cudaEventRecord(e1, s));
                CK(cudaEventSynchronize(e1));
                CK(cudaEventElapsedTime(&msp, e0, e1));
                printf("  (b) T=%-4d gemm_packed vs gemm_q(raw): maxrel %.1e | gemm_q %.3f ms  gemm_packed %.3f ms\n", T,
                       rel, msq / reps, msp / reps);
                if (T == 128) {   // row-chunked: minimum scratch
                    const size_t smin = fast::gemm_scratch_min_bytes(n_in, T);
                    CK(cudaMemsetAsync(y, 0xff, (size_t) T * n_out * 4, s));
                    fast::gemm_packed(w, x, y, T, scr, smin, s, cb);
                    CK(cudaMemcpyAsync(out.data(), y, out.size() * 4, cudaMemcpyDeviceToHost, s));
                    CK(cudaStreamSynchronize(s));
                    double e2 = 0;
                    for (size_t i = 0; i < out.size(); ++i)
                        e2 = std::max(e2, std::isfinite(out[i]) ? fabs((double) out[i] - ref[i]) : INFINITY);
                    worst_gemm = std::max(worst_gemm, e2 / rmax);
                    printf("      T=128 row-chunked gemm_packed (scratch %.2f MiB): maxrel %.1e\n", smin / 1048576.0, e2 / rmax);
                }
            }
            cudaFree(x); cudaFree(y); cudaFree(yr); cudaFree(scr); cudaFree(wraw); cudaFree(wpk);
            ++cases;
        }
    }
    printf("\nSUMMARY: %d cases, packed_to_f16 mismatches %lld of %lld values; worst gemm_packed vs gemm_q maxrel %.1e  => %s\n",
           cases, total_mism, total_vals, worst_gemm, total_mism == 0 && worst_gemm == 0 ? "PASS" : "FAIL");
    return total_mism == 0 && worst_gemm == 0 ? 0 : 2;
}
