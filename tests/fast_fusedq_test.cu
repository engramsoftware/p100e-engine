// p100e-fast-fusedq-test <model.gguf> [reps]
// P100E_PF_FUSEDQ gate: fast::fold_gemm with fold_opts::q4k (fold_seg_q4k_kernel, dequant in the GEMM loads) vs
// fast::packed_to_f16 (direct) + fast::fold_gemm on the f16 copy, exact compare of every output bit, on every q4_K
// tensor shape of the model and its tensor-parallel halves (n_out / 2: column split, n_in / 2: row split), real
// weight bytes; T = 7, 16, 50, 128, 300 (all three tile widths, partial tiles), f32 and f16 (out16) outputs, one and
// three segments. Plus timings of both paths at T = 512.
#include "fast/gemm.cuh"
#include "fast/unpack.cuh"
#include "fast/matvec.cuh"
#include "gguf.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <set>
#include <utility>
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
    const int reps = argc > 2 ? atoi(argv[2]) : 5;
    gguf_file g(path);
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    printf("device: %s (sm_%d%d)  model: %s\n", prop.name, prop.major, prop.minor, path);
    cudaStream_t s;
    CK(cudaStreamCreate(&s));
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    const qtype_info & ti = get_type_info(12u);

    // shapes: (n_in, n_out) of each q4_K tensor and its halves, cut from that tensor
    std::set<std::pair<int64_t, int64_t>> seen;
    struct shape { int64_t n_in, n_out; const gguf_tensor * t; };
    std::vector<shape> shapes;
    for (const auto & u : g.tensors) {
        if (u.type != 12u || u.n_dims != 2) continue;
        const int64_t a = u.ne[0], b = u.ne[1];
        for (auto sh : {std::make_pair(a, b), std::make_pair(a, b / 2), std::make_pair(a / 2, b)})
            if (sh.first % 256 == 0 && sh.second % 4 == 0 && sh.second > 0 && seen.insert(sh).second)
                shapes.push_back({sh.first, sh.second, &u});
    }
    const int Ts[] = {7, 16, 50, 128, 300};
    const int TMAX = 512;
    long long mism_total = 0, vals_total = 0;
    int cases = 0;
    for (const shape & sh : shapes) {
        const int64_t n_in = sh.n_in, n_out = sh.n_out;
        const size_t rb = (size_t) (n_in / 256 * ti.size), rb_src = (size_t) (sh.t->ne[0] / 256 * ti.size);
        std::vector<uint8_t> hw(rb * n_out);
        for (int64_t r = 0; r < n_out; ++r) memcpy(hw.data() + r * rb, (const uint8_t *) sh.t->data + r * rb_src, rb);
        void * wpk;
        CK(cudaMalloc(&wpk, hw.size()));
        CK(cudaMemcpy(wpk, hw.data(), hw.size(), cudaMemcpyHostToDevice));
        fast::packed_weight w = fast::pack_weight(12u, wpk, n_in, n_out, s);
        if (w.layout != fast::LAYOUT_Q4KR) {
            printf("== %s %lld x %lld: layout %s, skipped\n", sh.t->name.c_str(), (long long) n_in, (long long) n_out,
                   fast::kernel_name(w));
            cudaFree(wpk);
            continue;
        }
        half * w16;
        CK(cudaMalloc(&w16, (size_t) n_in * n_out * 2));
        fast::packed_to_f16(w, 0, n_out, w16, s, fast::UNPACK_DIRECT);
        // activations: random halves in the prescaled range (|x| < 16), per-token scales
        std::mt19937 rng(99 + (unsigned) n_in);
        std::normal_distribution<float> nd(0.0f, 3.0f);
        std::vector<half> hx((size_t) TMAX * n_in);
        for (auto & v : hx) v = __float2half(fmaxf(-15.9f, fminf(15.9f, nd(rng))));
        std::vector<float> hcs(TMAX);
        for (auto & v : hcs) v = ldexpf(1.0f, (int) (rng() % 9) - 4);
        half * x16;
        float *cs, *ya, *yb;
        CK(cudaMalloc(&x16, hx.size() * 2));
        CK(cudaMalloc(&cs, TMAX * 4));
        CK(cudaMalloc(&ya, (size_t) TMAX * n_out * 4));
        CK(cudaMalloc(&yb, (size_t) TMAX * n_out * 4));
        CK(cudaMemcpy(x16, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(cs, hcs.data(), TMAX * 4, cudaMemcpyHostToDevice));
        long long mism = 0, vals = 0;
        // segment sets: the whole weight; three row ranges (when n_out allows)
        std::vector<std::vector<std::pair<int64_t, int64_t>>> sets = {{{0, n_out}}};
        if (n_out >= 384) {
            const int64_t a = (n_out / 3) / 4 * 4, b = (2 * n_out / 3) / 4 * 4;
            sets.push_back({{0, a}, {a, b - a}, {b, n_out - b}});
        }
        for (const auto & set : sets) {
            for (int T : Ts) {
                for (int o16 = 0; o16 < 2; ++o16) {
                    fast::fold_seg A[4], B[4];
                    for (size_t i = 0; i < set.size(); ++i) {
                        const int64_t r0 = set[i].first, nr = set[i].second;
                        A[i].ldy = B[i].ldy = n_out;
                        A[i].rows = B[i].rows = (int) nr;
                        if (o16) { A[i].y16 = (half *) ya + r0; B[i].y16 = (half *) yb + r0; }
                        else { A[i].y = ya + r0; B[i].y = yb + r0; }
                        A[i].w = w16 + r0 * n_in;
                        B[i].w = (const half *) ((const char *) w.data + r0 * rb);
                    }
                    fast::fold_opts oa, ob;
                    oa.out16 = ob.out16 = o16 != 0;
                    oa.raster = ob.raster = 4;
                    ob.q4k = true;
                    const size_t es = o16 ? 2 : 4;
                    const size_t nbytes = (size_t) T * n_out * es;
                    CK(cudaMemsetAsync(ya, 0x7f, nbytes, s));
                    CK(cudaMemsetAsync(yb, 0x7f, nbytes, s));
                    fast::fold_gemm(A, (int) set.size(), x16, n_in, cs, T, (int) n_in, s, oa);
                    fast::fold_gemm(B, (int) set.size(), x16, n_in, cs, T, (int) n_in, s, ob);
                    std::vector<uint8_t> ha(nbytes), hb(nbytes);
                    CK(cudaMemcpyAsync(ha.data(), ya, nbytes, cudaMemcpyDeviceToHost, s));
                    CK(cudaMemcpyAsync(hb.data(), yb, nbytes, cudaMemcpyDeviceToHost, s));
                    CK(cudaStreamSynchronize(s));
                    for (size_t i = 0; i < nbytes; i += es) mism += memcmp(&ha[i], &hb[i], es) != 0;
                    vals += (long long) (nbytes / es);
                }
            }
        }
        // timing at T = 512, f32 out: unpack, fold on f16, fused
        float ms[3] = {0, 0, 0};
        for (int k = 0; k < 3; ++k) {
            fast::fold_seg sg;
            sg.ldy = n_out;
            sg.rows = (int) n_out;
            sg.y = ya;
            sg.w = k == 2 ? (const half *) w.data : w16;
            fast::fold_opts o;
            o.raster = 4;
            o.q4k = k == 2;
            CK(cudaEventRecord(e0, s));
            for (int i = 0; i < reps; ++i) {
                if (k == 0) fast::packed_to_f16(w, 0, n_out, w16, s, fast::UNPACK_DIRECT);
                else fast::fold_gemm(&sg, 1, x16, n_in, cs, TMAX, (int) n_in, s, o);
            }
            CK(cudaEventRecord(e1, s));
            CK(cudaEventSynchronize(e1));
            CK(cudaEventElapsedTime(&ms[k], e0, e1));
        }
        printf("== %-26s n_in %5lld n_out %5lld: %lld / %lld outputs differ %s | T=512 unpack %.3f + fold %.3f ms, fused %.3f ms\n",
               sh.t->name.c_str(), (long long) n_in, (long long) n_out, mism, vals, mism ? "FAIL" : "EXACT",
               ms[0] / reps, ms[1] / reps, ms[2] / reps);
        mism_total += mism;
        vals_total += vals;
        ++cases;
        cudaFree(x16); cudaFree(cs); cudaFree(ya); cudaFree(yb); cudaFree(w16); cudaFree(wpk);
    }
    printf("\nSUMMARY: %d q4_K shapes, %lld / %lld outputs differ => %s\n", cases, mism_total, vals_total,
           mism_total == 0 && cases > 0 ? "PASS" : "FAIL");
    return mism_total == 0 && cases > 0 ? 0 : 2;
}
