// p100e-fast-moe-gemv-test: the grouped expert GEMV (launch_q10_grouped, MoE "mul_mat_id") against the single-expert
// kernel it is built from, on Flash-Next expert shapes.
//   - correctness: every (token, expert) group's output column is bitwise the single-expert matvec_prequantized (T = 1)
//     of that expert on that token's a16 column
//   - timing: one grouped launch over G = tokens x top-k groups vs G single launches, repeated for the 48 layers x 3
//     projections of one token step (the launches that grouping saves)
// usage: p100e-fast-moe-gemv-test [--type q1_0|q2_0|q4_k|q4_0] [--experts E] [--topk K] [--tokens T] [--reps N]
#include "fast/matvec.cuh"
#include "matvec_internal.cuh"
#include "gguf.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
using namespace p100e;
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

struct proj { const char * name; int64_t n_in, n_out; };

int main(int argc, char ** argv) {
    int E = 16, K = 10, T = 1, reps = 48*3;
    uint32_t ty = GGML_TYPE_Q1_0;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--type") && i + 1 < argc) {
            const char * t = argv[++i];
            ty = !strcmp(t, "q4_k") ? GGML_TYPE_Q4_K : !strcmp(t, "q4_0") ? GGML_TYPE_Q4_0 : !strcmp(t, "q2_0") ? GGML_TYPE_Q2_0 : !strcmp(t, "iq2_s") ? GGML_TYPE_IQ2_S : GGML_TYPE_Q1_0;
        }
        else if (!strcmp(argv[i], "--experts") && i + 1 < argc) E = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--topk") && i + 1 < argc) K = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--tokens") && i + 1 < argc) T = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
    }
    const proj projs[] = {{"exp_gate|up", 2560, 640}, {"exp_down", 640, 2560}};
    std::mt19937 rng(7);
    cudaStream_t s;
    CK(cudaStreamCreate(&s));
    int fails = 0;
    const char * tname = ty == GGML_TYPE_Q4_K ? "q4_k" : ty == GGML_TYPE_Q4_0 ? "q4_0" : ty == GGML_TYPE_Q2_0 ? "q2_0" : ty == GGML_TYPE_IQ2_S ? "iq2_s" : "q1_0";
    for (const proj & p : projs) {
        const int64_t n_in = p.n_in, n_out = p.n_out;
        const qtype_info & ti = get_type_info(ty);
        if (n_in % ti.blck) { printf("%-12s %s: n_in %lld not a multiple of the %lld-block, skipped\n", p.name, tname, (long long) n_in, (long long) ti.blck); continue; }
        const size_t nb = (size_t) n_in / ti.blck, bs = ti.size, rb = nb * bs, wbytes = rb * (size_t) n_out;
        // E experts of random blocks with valid fp16 scales (Q1_0 / Q4_0: d at 0; Q4_K: d, dmin at 0 / 2)
        std::vector<uint8_t> hw(wbytes * E);
        for (size_t b = 0; b < (size_t) n_out * nb * E; ++b) {
            uint8_t * blk = hw.data() + b * bs;
            for (size_t i = 0; i < bs; ++i) blk[i] = (uint8_t) rng();
            const half d = __float2half(0.002f + 0.01f * (rng() % 1000) / 1000.0f);
            memcpy(blk, &d, 2);
            if (ty == GGML_TYPE_Q4_K) { const half dm = __float2half(0.001f + 0.005f * (rng() % 1000) / 1000.0f); memcpy(blk + 2, &dm, 2); }
        }
        char * dw;
        CK(cudaMalloc(&dw, hw.size()));
        CK(cudaMemcpy(dw, hw.data(), hw.size(), cudaMemcpyHostToDevice));
        std::vector<fast::packed_weight> pw(E);
        const int want_layout = ty == GGML_TYPE_Q4_K ? 2 : ty == GGML_TYPE_Q4_0 ? 1 : ty == GGML_TYPE_Q2_0 ? 18 : ty == GGML_TYPE_IQ2_S ? 14 : 17;   // LAYOUT_Q4KR / Q4R / Q20 / IQ2SR / Q10
        for (int e = 0; e < E; ++e) {
            pw[e] = fast::pack_weight(ty, dw + (size_t) e * wbytes, n_in, n_out, s);
            if (!pw[e].data || pw[e].layout != want_layout) { printf("pack_weight: layout %d for %s expert %d (wanted %d)\n", pw[e].layout, tname, e, want_layout); return 1; }
        }
        CK(cudaStreamSynchronize(s));   // repacks (q4kr / q4r are in-place layouts) before the rows are read
        auto grouped = [&](const fast::mv_group * tab, int G, const fast::a16_args & aa) {
            if (ty == GGML_TYPE_Q4_K) fast::launch_q4kr_grouped(tab, G, aa, s);
            else if (ty == GGML_TYPE_Q4_0) fast::launch_q4r_grouped(tab, G, aa, s);
            else if (ty == GGML_TYPE_Q2_0) fast::launch_q20_grouped(tab, G, aa, s);
            else if (ty == GGML_TYPE_IQ2_S) fast::launch_iq2sr_grouped(tab, G, aa, s);
            else fast::launch_q10_grouped(tab, G, aa, s);
        };
        auto single = [&](const fast::packed_weight & w, const fast::a16_args & aa, int nc = 1) {
            if (ty == GGML_TYPE_Q4_K) fast::launch_q4kr((const char *) w.data, aa, nc, s);
            else if (ty == GGML_TYPE_Q4_0) fast::launch_q4r((const char *) w.data, aa, nc, s);
            else if (ty == GGML_TYPE_Q2_0) fast::launch_q20((const char *) w.data, aa, nc, s);
            else if (ty == GGML_TYPE_IQ2_S) fast::launch_iq2sr((const char *) w.data, aa, nc, s);
            else fast::launch_q10((const char *) w.data, aa, nc, s);
        };
        // T token activations, a16-quantized once (the decode path)
        std::vector<float> hx((size_t) T * n_in);
        for (auto & v : hx) v = (float) ((int) (rng() % 2001) - 1000) / 1000.0f;
        float * dx; void * scr; float * y1; float * yg;
        CK(cudaMalloc(&dx, hx.size() * 4));
        CK(cudaMemcpy(dx, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&scr, fast::matvec_scratch_bytes(n_in, T)));
        fast::matvec_quantize(dx, n_in, T, scr, s);
        const int G = T * K;
        CK(cudaMalloc(&y1, (size_t) G * n_out * 4));
        CK(cudaMalloc(&yg, (size_t) G * n_out * 4));
        // routing: token t picks K distinct experts
        std::vector<fast::mv_group> hg(G);
        std::vector<int> pick(G);
        for (int t = 0; t < T; ++t) {
            std::vector<int> ids(E);
            for (int e = 0; e < E; ++e) ids[e] = e;
            std::shuffle(ids.begin(), ids.end(), rng);
            for (int k = 0; k < K; ++k) {
                const int g = t * K + k;
                pick[g] = ids[k];
                hg[g].w = pw[ids[k]].data;
                hg[g].col = t;
                hg[g].out = g;
            }
        }
        fast::mv_group * dg;
        CK(cudaMalloc(&dg, G * sizeof(fast::mv_group)));
        CK(cudaMemcpy(dg, hg.data(), G * sizeof(fast::mv_group), cudaMemcpyHostToDevice));
        // the a16 layout matvec_quantize wrote: aq = T * ng * 16 half2, then ads = T * ng half2
        const int ng = (int) (n_in / 32);
        fast::a16_args a;
        a.aq = (const half2 *) scr;
        a.ads = (const half2 *) scr + (size_t) T * ng * 16;
        a.dst = yg;
        a.ngroups = ng; a.nrows = (int) n_out; a.row_bytes = (int) rb;
        // reference: the plain (single-expert, T = 1) launch of the SAME kernel family on the token's own a16 column.
        // (matvec_prequantized would pick the wrp decode family for Q4_K at T = 1: a different summation order.)
        std::vector<fast::a16_args> as(G, a);
        for (int g = 0; g < G; ++g) {
            as[g].aq = a.aq + (size_t) hg[g].col * ng * 16; as[g].ads = a.ads + (size_t) hg[g].col * ng;
            as[g].dst = y1 + (size_t) g * n_out;
        }
        CK(cudaMemsetAsync(y1, 0xff, (size_t) G * n_out * 4, s));
        for (int g = 0; g < G; ++g) single(pw[pick[g]], as[g]);
        CK(cudaStreamSynchronize(s));
        CK(cudaGetLastError());
        CK(cudaMemsetAsync(yg, 0xff, (size_t) G * n_out * 4, s));
        grouped(dg, G, a);
        CK(cudaStreamSynchronize(s));
        CK(cudaGetLastError());
        std::vector<float> h1((size_t) G * n_out), hgo(h1.size());
        CK(cudaMemcpy(h1.data(), y1, h1.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(hgo.data(), yg, hgo.size() * 4, cudaMemcpyDeviceToHost));
        long long diff = 0;
        for (size_t i = 0; i < h1.size(); ++i) diff += memcmp(&h1[i], &hgo[i], 4) != 0;
        // T invariance: one NC = T launch over the T a16 columns of expert pick[0] == the T single-column launches
        if (T > 1 && T <= 4) {
            long long tdiff = 0;
            CK(cudaMemsetAsync(yg, 0xff, (size_t) G * n_out * 4, s));
            single(pw[pick[0]], a, T);           // columns 0..T-1 -> yg columns 0..T-1 (stride n_out)
            for (int t = 0; t < T; ++t) {
                fast::a16_args a1 = a;
                a1.aq = a.aq + (size_t) t * ng * 16; a1.ads = a.ads + (size_t) t * ng; a1.dst = y1 + (size_t) t * n_out;
                single(pw[pick[0]], a1);
            }
            CK(cudaStreamSynchronize(s));
            std::vector<float> hm((size_t) T * n_out), hs((size_t) T * n_out);
            CK(cudaMemcpy(hm.data(), yg, hm.size() * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(hs.data(), y1, hs.size() * 4, cudaMemcpyDeviceToHost));
            for (size_t i = 0; i < hm.size(); ++i) tdiff += memcmp(&hm[i], &hs[i], 4) != 0;
            // Q4_0's plain launch picks nwarps by the column count (4 at NC = 1, 2 above; q4r_launch), so its NC = 4
            // sum order differs on rows with 2*ngroups >= 64 (gate/up): a pre-existing property of the Q4_0 path,
            // reported but not failed. The grouped MoE launch is always NC = 1, so it is unaffected.
            const bool info_only = ty == GGML_TYPE_Q4_0;
            printf("%-12s %s T invariance (NC=%d vs %d x NC=1): %lld of %zu differ  %s\n", p.name, tname, T, T, tdiff, hm.size(),
                   tdiff == 0 ? "PASS" : info_only ? "DIFFERS (q4r picks nwarps by NC; informational)" : "FAIL");
            fails += tdiff != 0 && !info_only;
        }
        // NC = 4 groups: one group = 4 consecutive a16 columns of one expert (T >= 4) vs the 4 single-column launches
        if (T >= 4 && (ty == GGML_TYPE_Q4_K || ty == GGML_TYPE_Q4_0 || ty == GGML_TYPE_IQ2_S)) {
            const int G4 = std::min(E, T / 4);   // group q: expert pick[q], columns 4q..4q+3 -> outputs 4q..4q+3
            std::vector<fast::mv_group> h4(G4);
            for (int q = 0; q < G4; ++q) { h4[q].w = pw[pick[q]].data; h4[q].col = 4 * q; h4[q].out = 4 * q; }
            fast::mv_group * d4;
            CK(cudaMalloc(&d4, G4 * sizeof(fast::mv_group)));
            CK(cudaMemcpy(d4, h4.data(), G4 * sizeof(fast::mv_group), cudaMemcpyHostToDevice));
            CK(cudaMemsetAsync(yg, 0xff, (size_t) G * n_out * 4, s));
            if (ty == GGML_TYPE_Q4_K) fast::launch_q4kr_grouped_nc(d4, G4, a, 4, s);
            else if (ty == GGML_TYPE_Q4_0) fast::launch_q4r_grouped_nc(d4, G4, a, 4, s);
            else fast::launch_iq2sr_grouped_nc(d4, G4, a, 4, s);
            CK(cudaMemsetAsync(y1, 0xff, (size_t) G * n_out * 4, s));
            for (int q = 0; q < G4; ++q)
                for (int c = 0; c < 4; ++c) {
                    fast::a16_args a1 = a;
                    a1.aq = a.aq + (size_t) (4 * q + c) * ng * 16; a1.ads = a.ads + (size_t) (4 * q + c) * ng; a1.dst = y1 + (size_t) (4 * q + c) * n_out;
                    single(pw[pick[q]], a1);
                }
            CK(cudaStreamSynchronize(s));
            CK(cudaGetLastError());
            std::vector<float> h4o((size_t) 4 * G4 * n_out), h1o(h4o.size());
            CK(cudaMemcpy(h4o.data(), yg, h4o.size() * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(h1o.data(), y1, h1o.size() * 4, cudaMemcpyDeviceToHost));
            long long d4iff = 0;
            for (size_t i = 0; i < h4o.size(); ++i) d4iff += memcmp(&h4o[i], &h1o[i], 4) != 0;
            printf("%-12s %s NC=4 groups (%d) vs 4 x single columns: %lld of %zu differ  %s\n", p.name, tname, G4, d4iff, h4o.size(), d4iff == 0 ? "PASS" : "FAIL");
            fails += d4iff != 0;
            CK(cudaFree(d4));
        }
        // timing: grouped (one launch per projection per layer) vs single launches, reps = layers x projections
        auto now = [] { return std::chrono::steady_clock::now(); };
        auto ms = [](auto t0) { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(); };
        for (int i = 0; i < 10; ++i) grouped(dg, G, a);
        CK(cudaStreamSynchronize(s));
        auto t0 = now();
        for (int i = 0; i < reps; ++i) grouped(dg, G, a);
        CK(cudaStreamSynchronize(s));
        const double tg = ms(t0);
        for (int g = 0; g < G; ++g) as[g].dst = yg + (size_t) g * n_out;   // timing only: overwrite the grouped buffer
        for (int g = 0; g < G; ++g) single(pw[pick[g]], as[g]);
        CK(cudaStreamSynchronize(s));
        t0 = now();
        for (int i = 0; i < reps; ++i)
            for (int g = 0; g < G; ++g) single(pw[pick[g]], as[g]);
        CK(cudaStreamSynchronize(s));
        const double ts = ms(t0);
        const double macs = (double) G * n_in * n_out * reps;
        printf("%-12s %s %5lld x %5lld  E %d  G %d (tokens %d x top-%d)  grouped vs single: %lld of %zu outputs differ  %s\n"
               "              %d launches: grouped %.2f ms (%.1f us/launch, %.0f GMAC/s), single %.2f ms (%.1f us/launch) -> x%.1f\n",
               p.name, tname, (long long) n_in, (long long) n_out, E, G, T, K, diff, h1.size(), diff ? "FAIL" : "PASS",
               reps, tg, 1e3 * tg / reps, macs / (tg * 1e-3) / 1e9, ts, 1e3 * ts / (reps * G), ts / tg);
        fails += diff != 0;
        cudaFree(dw); cudaFree(dx); cudaFree(scr); cudaFree(y1); cudaFree(yg); cudaFree(dg);
    }
    printf("MOE GEMV TEST: %s\n", fails ? "FAILED" : "passed");
    return fails ? 1 : 0;
}
