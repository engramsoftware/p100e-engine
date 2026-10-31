// p100e-fn-cpu-pool-test: the CPU expert pool (src/fn_cpu_pool.cpp) on synthetic experts of the Flash-Next shapes
// (gate / up 2560 -> 640, down 640 -> 2560), in the layout pairs the expert-split files use: q4kr + q4r (Q4_K / Q4_0 hot
// experts), iq2sr + q20 (IQ2_S / Q2_0 cold experts), q4r + q4r. The raw GGUF rows are random; the GPU packs them
// (fast::pack_weight, the bytes fn_model keeps in host RAM), the pool computes from the packed bytes. Checks:
//  - CPU expert output vs an fp64 reference on the dequantized raw rows: max |err| / max |ref| <= 5e-3 (int16 activations)
//  - the same for the GPU path (fast::matvec gate, up, silu*up, down) and the CPU-vs-GPU max |diff| (reported)
//  - several tokens per expert and two experts per job; the result does not depend on the worker count (0 / 3 workers)
// Exit code 0 only if every check passes.
#include "fn_cpu_pool.h"
#include "fast/matvec.cuh"
#include "gguf.h"
#include "kernels.cuh"
#include "convert.cuh"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include <cuda_fp16.h>

using namespace p100e;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

static int fails = 0;
static std::mt19937 rng(1234);

static uint16_t f2h(float f) { __half h = __float2half(f); uint16_t u; memcpy(&u, &h, 2); return u; }
static float h2f(uint16_t u) { __half h; memcpy(&h, &u, 2); return __half2float(h); }

static int block_bytes(uint32_t ty) { return ty == GGML_TYPE_Q4_0 ? 18 : ty == GGML_TYPE_Q4_K ? 144 : ty == GGML_TYPE_IQ2_S ? 82 : 18; }
static int block_vals(uint32_t ty) { return ty == GGML_TYPE_Q4_0 ? 32 : ty == GGML_TYPE_Q2_0 ? 64 : 256; }

// random raw row set [n_out][n_in] with sane fp16 scales
static std::vector<uint8_t> rand_rows(uint32_t ty, int n_in, int n_out) {
    const int bb = block_bytes(ty), nb = n_in / block_vals(ty);
    std::vector<uint8_t> v((size_t) n_out * nb * bb);
    std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_real_distribution<float> sc(0.002f, 0.02f);
    for (auto & b : v) b = (uint8_t) byte(rng);
    for (int r = 0; r < n_out; ++r)
        for (int b = 0; b < nb; ++b) {
            uint8_t * p = v.data() + ((size_t) r * nb + b) * bb;
            const float dv = ty == GGML_TYPE_Q4_K ? 0.05f * sc(rng) : sc(rng);   // Q4_K: 6-bit sub-scales on top
            const uint16_t d = f2h(dv); memcpy(p, &d, 2);
            if (ty == GGML_TYPE_Q4_K) { const uint16_t m = f2h(dv * 0.5f); memcpy(p + 2, &m, 2); }
        }
    return v;
}

// fp64-exact dequantization of one raw row (ggml's definitions)
static void deq_row(uint32_t ty, const uint8_t * row, int n_in, float * y) {
    const int nb = n_in / block_vals(ty), bb = block_bytes(ty);
    for (int b = 0; b < nb; ++b) {
        const uint8_t * p = row + (size_t) b * bb;
        float * o = y + (size_t) b * block_vals(ty);
        uint16_t dh; memcpy(&dh, p, 2); const float d = h2f(dh);
        if (ty == GGML_TYPE_Q4_0) {
            for (int j = 0; j < 16; ++j) { o[j] = ((p[2 + j] & 15) - 8) * d; o[j + 16] = ((p[2 + j] >> 4) - 8) * d; }
        } else if (ty == GGML_TYPE_Q2_0) {
            for (int i = 0; i < 64; ++i) o[i] = (((p[2 + i / 4] >> (2 * (i % 4))) & 3) - 1) * d;
        } else if (ty == GGML_TYPE_Q4_K) {
            uint16_t mh; memcpy(&mh, p + 2, 2); const float dmin = h2f(mh);
            const uint8_t * s = p + 4, * q = p + 16;
            auto gsm = [&](int j, int & sc, int & m) {
                if (j < 4) { sc = s[j] & 63; m = s[j + 4] & 63; }
                else { sc = (s[j + 4] & 0xF) | ((s[j - 4] >> 6) << 4); m = (s[j + 4] >> 4) | ((s[j] >> 6) << 4); }
            };
            for (int j = 0; j < 4; ++j) {
                int sc1, m1, sc2, m2; gsm(2 * j, sc1, m1); gsm(2 * j + 1, sc2, m2);
                for (int l = 0; l < 32; ++l) o[64 * j + l] = d * sc1 * (q[32 * j + l] & 15) - dmin * m1;
                for (int l = 0; l < 32; ++l) o[64 * j + 32 + l] = d * sc2 * (q[32 * j + l] >> 4) - dmin * m2;
            }
        } else {   // IQ2_S
            const uint8_t * qs = p + 2, * signs = p + 2 + 32, * qh = p + 66, * scales = p + 74;
            for (int ib = 0; ib < 8; ++ib) {
                const float db0 = d * (0.5f + (scales[ib] & 15)) * 0.25f, db1 = d * (0.5f + (scales[ib] >> 4)) * 0.25f;
                for (int l = 0; l < 4; ++l) {
                    const float dl = l < 2 ? db0 : db1;
                    const uint8_t * grid = (const uint8_t *) (iq2s_grid + (qs[4 * ib + l] | ((qh[ib] << (8 - 2 * l)) & 0x300)));
                    for (int j = 0; j < 8; ++j) o[32 * ib + 8 * l + j] = dl * grid[j] * ((signs[4 * ib + l] >> j) & 1 ? -1.0f : 1.0f);
                }
            }
        }
    }
}

struct wt { uint32_t ty; int n_in, n_out; std::vector<uint8_t> raw, packed; fast::packed_weight pk; void * d = nullptr; size_t rb; std::vector<float> deq; };

static void make(wt & w, uint32_t ty, int n_in, int n_out, cudaStream_t s) {
    w.ty = ty; w.n_in = n_in; w.n_out = n_out;
    w.raw = rand_rows(ty, n_in, n_out);
    w.rb = w.raw.size() / n_out;
    CK(cudaMalloc(&w.d, w.raw.size()));
    CK(cudaMemcpy(w.d, w.raw.data(), w.raw.size(), cudaMemcpyHostToDevice));
    w.pk = fast::pack_weight(ty, w.d, n_in, n_out, s);
    CK(cudaStreamSynchronize(s));
    w.packed.resize(w.raw.size());
    CK(cudaMemcpy(w.packed.data(), w.d, w.raw.size(), cudaMemcpyDeviceToHost));
    w.deq.resize((size_t) n_in * n_out);
    for (int r = 0; r < n_out; ++r) deq_row(ty, w.raw.data() + (size_t) r * w.rb, n_in, w.deq.data() + (size_t) r * n_in);
}

static bool check(const char * what, const std::vector<float> & got, const std::vector<double> & ref, double tol) {
    double md = 0, mr = 0;
    for (size_t i = 0; i < ref.size(); ++i) { md = std::max(md, std::fabs(got[i] - ref[i])); mr = std::max(mr, std::fabs(ref[i])); }
    const bool ok = md <= tol * mr && std::isfinite(md);
    printf("  %-34s max|err| %.4g  max|ref| %.4g  (%.3f%%) %s\n", what, md, mr, 100.0 * md / mr, ok ? "ok" : "FAIL");
    if (!ok) ++fails;
    return ok;
}

int main() {
    if (!fn_cpu_pool::cpu_ok()) { printf("no AVX2/FMA/F16C: skipped\n"); return 0; }
    cudaStream_t s; CK(cudaStreamCreate(&s));
    const int E = 2560, F = 640, T = 5;
    struct combo { const char * name; uint32_t gu, dn; };
    const combo combos[] = {{"q4kr gate/up + q4r down", GGML_TYPE_Q4_K, GGML_TYPE_Q4_0},
                            {"iq2sr gate/up + q20 down", GGML_TYPE_IQ2_S, GGML_TYPE_Q2_0},
                            {"q4r gate/up + q4r down", GGML_TYPE_Q4_0, GGML_TYPE_Q4_0}};
    std::vector<float> x((size_t) T * E);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto & v : x) v = nd(rng);
    fn_cpu_pool pool0(0), pool3(3);
    float *dx, *dg, *du, *dy; void * scr;
    CK(cudaMalloc(&dx, (size_t) T * E * 4)); CK(cudaMalloc(&dg, (size_t) T * F * 4)); CK(cudaMalloc(&du, (size_t) T * F * 4)); CK(cudaMalloc(&dy, (size_t) T * E * 4));
    CK(cudaMalloc(&scr, fast::matvec_scratch_bytes(E, 8)));
    CK(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
    for (const combo & c : combos) {
        printf("%s\n", c.name);
        wt g[2], u[2], d[2];   // two experts
        for (int k = 0; k < 2; ++k) { make(g[k], c.gu, E, F, s); make(u[k], c.gu, E, F, s); make(d[k], c.dn, F, E, s); }
        printf("  layouts: gate %s down %s\n", fast::kernel_name(g[0].pk), fast::kernel_name(d[0].pk));
        if (!fn_cpu_pool::supports(g[0].pk.layout, d[0].pk.layout)) { printf("  layouts not supported by the pool: FAIL\n"); ++fails; continue; }
        // expert 0: tokens 0, 2, 3, 4; expert 1: tokens 1, 3
        const std::vector<int32_t> tok0 = {0, 2, 3, 4}, tok1 = {1, 3};
        const std::vector<int32_t> * toks[2] = {&tok0, &tok1};
        std::vector<double> ref[2];
        for (int k = 0; k < 2; ++k) {
            ref[k].assign(toks[k]->size() * E, 0.0);
            for (size_t i = 0; i < toks[k]->size(); ++i) {
                const float * xi = x.data() + (size_t) (*toks[k])[i] * E;
                std::vector<double> h(F);
                for (int f = 0; f < F; ++f) {
                    double a = 0, b = 0;
                    for (int e = 0; e < E; ++e) { a += (double) g[k].deq[(size_t) f * E + e] * xi[e]; b += (double) u[k].deq[(size_t) f * E + e] * xi[e]; }
                    h[f] = a / (1.0 + std::exp(-a)) * b;
                }
                for (int e = 0; e < E; ++e) { double y = 0; for (int f = 0; f < F; ++f) y += (double) d[k].deq[(size_t) e * F + f] * h[f]; ref[k][i * E + e] = y; }
            }
        }
        std::vector<float> y0[2], y3[2];
        for (int run = 0; run < 2; ++run) {
            fn_cpu_pool & P = run == 0 ? pool0 : pool3;
            std::vector<float> * ys = run == 0 ? y0 : y3;
            fn_cpu_job j[2];
            for (int k = 0; k < 2; ++k) {
                ys[k].assign(toks[k]->size() * E, 0.0f);
                j[k].gate = g[k].packed.data(); j[k].up = u[k].packed.data(); j[k].down = d[k].packed.data();
                j[k].rb_gu = g[k].rb; j[k].rb_d = d[k].rb; j[k].lay_gu = g[k].pk.layout; j[k].lay_d = d[k].pk.layout;
                j[k].n = (int) toks[k]->size(); j[k].tok = toks[k]->data(); j[k].y = ys[k].data();
            }
            P.submit(x.data(), T, E, F, j, 2);
            P.wait();
        }
        for (int k = 0; k < 2; ++k) {
            char nm[64]; snprintf(nm, sizeof nm, "CPU expert %d (%zu tokens)", k, toks[k]->size());
            check(nm, y0[k], ref[k], 5e-3);
            const bool same = y0[k] == y3[k];
            printf("  CPU expert %d, 0 vs 3 workers: %s\n", k, same ? "bit-identical" : "DIFFERENT (FAIL)");
            if (!same) ++fails;
        }
        // GPU path on the same packed weights (T rows of every token), then the routed rows
        for (int k = 0; k < 2; ++k) {
            fast::packed_weight pg = g[k].pk, pu = u[k].pk, pd = d[k].pk;
            for (int t0 = 0; t0 < T; t0 += 1) {
                fast::matvec(pg, dx + (size_t) t0 * E, dg + (size_t) t0 * F, 1, scr, s);
                fast::matvec(pu, dx + (size_t) t0 * E, du + (size_t) t0 * F, 1, scr, s);
            }
            k::silu_mul(dg, du, (int64_t) T * F, s);
            for (int t0 = 0; t0 < T; ++t0) fast::matvec(pd, dg + (size_t) t0 * F, dy + (size_t) t0 * E, 1, scr, s);
            std::vector<float> yall((size_t) T * E);
            CK(cudaMemcpyAsync(yall.data(), dy, yall.size() * 4, cudaMemcpyDeviceToHost, s));
            CK(cudaStreamSynchronize(s));
            std::vector<float> yg(toks[k]->size() * E);
            for (size_t i = 0; i < toks[k]->size(); ++i) memcpy(yg.data() + i * E, yall.data() + (size_t) (*toks[k])[i] * E, (size_t) E * 4);
            char nm[64]; snprintf(nm, sizeof nm, "GPU expert %d", k);
            check(nm, yg, ref[k], 1e-2);
            double md = 0, mr = 0;
            for (size_t i = 0; i < yg.size(); ++i) { md = std::max(md, (double) std::fabs(yg[i] - y0[k][i])); mr = std::max(mr, (double) std::fabs(yg[i])); }
            printf("  CPU vs GPU expert %d: max|diff| %.4g of max %.4g (%.3f%%)\n", k, md, mr, 100.0 * md / mr);
        }
        for (int k = 0; k < 2; ++k) { cudaFree(g[k].d); cudaFree(u[k].d); cudaFree(d[k].d); }
    }
    printf(fails ? "FAILED: %d check(s)\n" : "all checks passed\n", fails);
    return fails ? 1 : 0;
}
