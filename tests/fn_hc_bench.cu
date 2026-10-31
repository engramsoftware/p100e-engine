// p100e-fn-hc-bench: GPU time of the Flash-Next hyper-connection mixer's two matvecs (blk.<il>.hc_attn_down: Q8_0
// [10240 -> 320], hc_attn_up: [320 -> 10240] dequantized to f16 like fn_model::upload_raw) for T = 1..4, back to back
// on one stream with the weights rotated over enough layers to miss L2.
//   p100e-fn-hc-bench <flash-next.gguf> [--dev 0] [--reps 200] [--layers 8]
#include "fast/matvec.cuh"
#include "fn_hc_kernels.cuh"
#include <cmath>
#include "gguf.h"
#include "convert.cuh"
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(2); } } while (0)
using namespace p100e;

__global__ void f32_to_f16(const float * s, half * d, int64_t n) { const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; if (i < n) d[i] = __float2half(s[i]); }

static fast::packed_weight upload(const gguf_file & g, const gguf_tensor * t, cudaStream_t s) {
    const int64_t n_in = t->ne[0], n_out = t->ne[1];
    void * d = nullptr;
    const bool fl = t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_F16;
    if (!fl && n_in % 256 != 0) {   // narrow rows: f16 copy (fn_model::upload_raw)
        void * raw = nullptr; float * f = nullptr;
        CK(cudaMalloc(&raw, t->nbytes)); CK(cudaMalloc(&f, n_in * n_out * 4)); CK(cudaMalloc(&d, n_in * n_out * 2));
        CK(cudaMemcpy(raw, t->data, t->nbytes, cudaMemcpyHostToDevice));
        to_fp32_cuda_t fn = ggml_get_to_fp32_cuda((ggml_type) t->type);
        if (!fn) { fprintf(stderr, "no dequant for type %u\n", t->type); exit(2); }
        fn(raw, f, n_in * n_out, s);
        f32_to_f16<<<(unsigned) ((n_in * n_out + 255) / 256), 256, 0, s>>>(f, (half *) d, n_in * n_out);
        CK(cudaStreamSynchronize(s));
        cudaFree(raw); cudaFree(f);
        return fast::pack_weight(GGML_TYPE_F16, d, n_in, n_out, s);
    }
    CK(cudaMalloc(&d, t->nbytes));
    CK(cudaMemcpy(d, t->data, t->nbytes, cudaMemcpyHostToDevice));
    return fast::pack_weight(t->type, d, n_in, n_out, s);
}

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: p100e-fn-hc-bench model.gguf [--dev 0] [--reps 200] [--layers 8]\n"); return 2; }
    int dev = 0, reps = 200, nl = 8;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--dev") dev = atoi(argv[++i]); else if (a == "--reps") reps = atoi(argv[++i]); else if (a == "--layers") nl = atoi(argv[++i]);
    }
    CK(cudaSetDevice(dev));
    cudaDeviceProp p{}; CK(cudaGetDeviceProperties(&p, dev));
    gguf_file g(argv[1]);
    cudaStream_t s; CK(cudaStreamCreate(&s));
    std::vector<fast::packed_weight> down, up;
    for (int il = 0; il < nl; ++il) {
        down.push_back(upload(g, g.find("blk." + std::to_string(il) + ".hc_attn_down.weight"), s));
        up.push_back(upload(g, g.find("blk." + std::to_string(il) + ".hc_attn_up.weight"), s));
    }
    const int64_t NI = down[0].n_in, NO = down[0].n_out;
    printf("%s: hc down type %u layout %d [%lld -> %lld] (%s), up type %u layout %d [%lld -> %lld] (%s)\n", p.name, down[0].type, down[0].layout,
           (long long) NI, (long long) NO, fast::kernel_name(down[0]), up[0].type, up[0].layout, (long long) up[0].n_in, (long long) up[0].n_out, fast::kernel_name(up[0]));
    float *x, *y, *x2, *y2; void * scr;
    CK(cudaMalloc(&x, 8 * NI * 4)); CK(cudaMalloc(&y, 8 * NO * 4)); CK(cudaMalloc(&x2, 8 * NO * 4)); CK(cudaMalloc(&y2, 8 * NI * 4));
    CK(cudaMalloc(&scr, fast::matvec_scratch_bytes(NI, 8)));
    std::vector<float> h(8 * NI); for (auto & v : h) v = (float) rand() / RAND_MAX - 0.5f;
    CK(cudaMemcpy(x, h.data(), h.size() * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(x2, h.data(), 8 * NO * 4, cudaMemcpyHostToDevice));
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    float * yr; CK(cudaMalloc(&yr, 8 * NI * 4));
    float * part; CK(cudaMalloc(&part, fnhc::q8r_tile_scratch_bytes(NO)));
    std::vector<float> ha(8 * NI), hb(8 * NI);
    for (int T = 1; T <= 4; ++T) {
        for (int w = 0; w < 2; ++w) {
            for (int alt = 0; alt < 3; ++alt) {   // 0 = fast::matvec (the engine today), 1 = fn_hc_kernels.cuh (P100E_FN_HC_FAST)
                auto call = [&](int i, float * out) {
                    const fast::packed_weight & pk = w ? up[i % nl] : down[i % nl];
                    const float * in = w ? x2 : x;
                    if (alt == 0) fast::matvec(pk, in, out, T, scr, s);
                    else if (w) { if (alt == 1) fnhc::f16_narrow(pk.data, in, out, T, pk.n_in, pk.n_out, s); else fast::matvec(pk, in, out, T, scr, s); }
                    else if (alt == 1) fnhc::q8r_splitk(pk.data, in, out, T, pk.n_in, pk.n_out, s);
                    else fnhc::q8r_tile(pk.data, in, out, part, T, pk.n_in, pk.n_out, s);
                };
                for (int i = 0; i < 10; ++i) call(i, w ? y2 : y);
                CK(cudaEventRecord(a, s));
                for (int i = 0; i < reps; ++i) call(i, w ? y2 : y);
                CK(cudaEventRecord(b, s)); CK(cudaEventSynchronize(b));
                float ms = 0; CK(cudaEventElapsedTime(&ms, a, b));
                const double us = 1e3 * ms / reps, bytes = (double) (w ? up[0].bytes : down[0].bytes);
                // accuracy vs fast::matvec on layer 0
                const int64_t no = w ? up[0].n_out : down[0].n_out;
                call(0, yr);
                fast::matvec(w ? up[0] : down[0], w ? x2 : x, w ? y2 : y, T, scr, s);
                CK(cudaStreamSynchronize(s));
                CK(cudaMemcpy(ha.data(), yr, (size_t) T * no * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(hb.data(), w ? y2 : y, (size_t) T * no * 4, cudaMemcpyDeviceToHost));
                double md = 0, mr = 0; for (int64_t i = 0; i < T * no; ++i) { md = std::max(md, (double) fabs(ha[i] - hb[i])); mr = std::max(mr, (double) fabs(hb[i])); }
                printf("T=%d %-4s %-12s %8.1f us per call  %6.1f GB/s   max|diff| vs fast %.3g (max|y| %.3g)\n", T, w ? "up" : "down", alt == 0 ? "fast::matvec" : alt == 1 ? "fn_hc splitk" : "fn_hc tile", us, bytes / us / 1e3, md, mr);
            }
        }
    }
    return 0;
}
