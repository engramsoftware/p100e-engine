// p100e-dequant <model.gguf> <out_dir> [rows=64]
// For each distinct tensor type in the model, picks the largest tensor of that type, dequantizes its
// first <rows> rows on the GPU (vendored ggml convert kernels, f32 and f16 outputs) and writes
// <out_dir>/<tensor>.f32 / .f16 plus manifest.tsv. tests/verify_dequant.py checks them against
// gguf-py's numpy reference.
#include "../src/gguf.h"
#include "../third_party/ggml/convert.cuh"

#include <algorithm>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

using namespace p100e;

#define CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

static std::string safe(std::string s) {
    for (char & c : s) if (c == '/' || c == '\\' || c == ':') c = '_';
    return s;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: p100e-dequant <model.gguf> <out_dir> [rows=64]\n");
        return 1;
    }
    const std::string out = argv[2];
    const int64_t want_rows = argc > 3 ? atoll(argv[3]) : 64;
    gguf_file g(argv[1]);

    std::map<uint32_t, const gguf_tensor *> pick;
    for (const auto & t : g.tensors) {
        auto & p = pick[t.type];
        if (!p || t.nbytes > p->nbytes) p = &t;
    }

    FILE * man = fopen((out + "/manifest.tsv").c_str(), "w");
    if (!man) { fprintf(stderr, "cannot write %s/manifest.tsv\n", out.c_str()); return 1; }
    fprintf(man, "tensor\ttype\tne0\trows\tfile\tgpu_ms\n");

    cudaEvent_t e0, e1;
    CHECK(cudaEventCreate(&e0));
    CHECK(cudaEventCreate(&e1));
    int failures = 0;
    for (const auto & [type, t] : pick) {
        const qtype_info & ti = get_type_info(type);
        const int64_t rows_total = t->ne[1] * t->ne[2] * t->ne[3];
        const int64_t rows = std::min(want_rows, rows_total);
        const int64_t n = t->ne[0] * rows;
        const size_t row_bytes = (size_t)(t->ne[0] / ti.blck) * ti.size;
        const size_t src_bytes = row_bytes * rows;

        to_fp32_cuda_t f32 = ggml_get_to_fp32_cuda((ggml_type) type);
        to_fp16_cuda_t f16 = ggml_get_to_fp16_cuda((ggml_type) type);
        const bool is_f32 = type == GGML_TYPE_F32;   // same-type outputs are plain copies, no converter
        const bool is_f16 = type == GGML_TYPE_F16;
        if ((!f32 && !is_f32) || (!f16 && !is_f16)) {
            printf("%-8s %-40s NO GPU DEQUANT\n", ti.name, t->name.c_str());
            fprintf(man, "%s\t%s\t%lld\t%lld\t-\t-\n", t->name.c_str(), ti.name, (long long) t->ne[0], (long long) rows);
            failures++;
            continue;
        }

        void * d_src; float * d_f32; half * d_f16;
        CHECK(cudaMalloc(&d_src, src_bytes));
        CHECK(cudaMalloc(&d_f32, n * sizeof(float)));
        CHECK(cudaMalloc(&d_f16, n * sizeof(half)));
        CHECK(cudaMemcpy(d_src, t->data, src_bytes, cudaMemcpyHostToDevice));
        CHECK(cudaEventRecord(e0));
        if (is_f32) {
            CHECK(cudaMemcpyAsync(d_f32, d_src, n * sizeof(float), cudaMemcpyDeviceToDevice, 0));
        } else {
            f32(d_src, d_f32, n, 0);
        }
        CHECK(cudaEventRecord(e1));
        if (is_f16) {
            CHECK(cudaMemcpyAsync(d_f16, d_src, n * sizeof(half), cudaMemcpyDeviceToDevice, 0));
        } else {
            f16(d_src, d_f16, n, 0);
        }
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());
        float ms = 0;
        CHECK(cudaEventElapsedTime(&ms, e0, e1));

        std::vector<float> h32(n);
        std::vector<uint16_t> h16(n);
        CHECK(cudaMemcpy(h32.data(), d_f32, n * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(h16.data(), d_f16, n * sizeof(half), cudaMemcpyDeviceToHost));
        const std::string base = safe(t->name);
        FILE * f = fopen((out + "/" + base + ".f32").c_str(), "wb");
        fwrite(h32.data(), sizeof(float), n, f);
        fclose(f);
        f = fopen((out + "/" + base + ".f16").c_str(), "wb");
        fwrite(h16.data(), sizeof(uint16_t), n, f);
        fclose(f);
        fprintf(man, "%s\t%s\t%lld\t%lld\t%s\t%.4f\n", t->name.c_str(), ti.name, (long long) t->ne[0], (long long) rows,
                base.c_str(), ms);
        printf("%-8s %-40s rows %lld x %lld  %.3f ms\n", ti.name, t->name.c_str(), (long long) rows, (long long) t->ne[0], ms);
        cudaFree(d_src); cudaFree(d_f32); cudaFree(d_f16);
    }
    fclose(man);
    return failures ? 2 : 0;
}
