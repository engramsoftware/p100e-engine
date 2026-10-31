// p100e-fast-prefill-test [model.gguf]: the prefill-pipeline building blocks against the classic prefill path.
//  (1) fast::fold_gemm (segmented, strided x16, token chunks) vs fast::gemm_packed per weight: must be bit-identical
//  (2) fast::add_norm_prescale vs k::add x2 + fast::rmsnorm + fast::prescale_rows: bit-identical x, x16, cs
//  (3) fast::glu_prescale vs k::silu_mul + fast::prescale_rows: bit-identical
//  (4) fast::attn_prefill vs fast::attn_decode looped 4 queries at a time: max |diff| / max |ref| (fp32, ~1e-6)
// wire ports: in (1) the raster block order (fold_opts::raster)
// must give the same bits, the f16 epilogue (out16) must equal f16(the f32 result) bit for bit, and the one-read
// prescale_rows must equal the two-read kernel; in (2) add_norm_prescale16 on f16 partials must equal
// add_norm_prescale on the same values widened to f32.
// Exit code 0 only if every check passes.
#include "fast/gemm.cuh"
#include "fast/unpack.cuh"
#include "fast/matvec.cuh"
#include "fast/fused.cuh"
#include "fast/attn.cuh"
#include "kernels.cuh"
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace p100e;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

static int fails = 0;

template <typename T>
static std::vector<T> dl(const T * d, size_t n) {
    std::vector<T> h(n);
    CK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
}
template <typename T>
static T * up(const std::vector<T> & h) {
    T * d;
    CK(cudaMalloc(&d, h.size() * sizeof(T)));
    CK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}
static long long mism_bytes(const void * a, const void * b, size_t n, size_t esz) {
    long long m = 0;
    for (size_t i = 0; i < n; ++i) m += memcmp((const char *) a + i * esz, (const char *) b + i * esz, esz) != 0;
    return m;
}
static void report(const char * what, long long m, size_t n) {
    printf("  %-62s %lld / %zu differ %s\n", what, m, n, m ? "FAIL" : "EXACT");
    if (m) ++fails;
}

struct wt { fast::packed_weight pk; int64_t n_in, n_out; std::string name; };

static wt load_w(const gguf_file & g, const gguf_tensor * t, int64_t n_out, cudaStream_t s) {
    const qtype_info & ti = get_type_info(t->type);
    const int64_t n_in = t->ne[0];
    const size_t rb = (size_t) (n_in / ti.blck * ti.size);
    void * d;
    CK(cudaMalloc(&d, rb * n_out));
    CK(cudaMemcpy(d, t->data, rb * n_out, cudaMemcpyHostToDevice));
    wt w;
    w.pk = fast::pack_weight(t->type, d, n_in, n_out, s);
    CK(cudaStreamSynchronize(s));
    w.n_in = n_in; w.n_out = n_out;
    w.name = t->name + " (" + ti.name + ", layout " + std::to_string(w.pk.layout) + ")";
    return w;
}

static int run(int argc, char ** argv) {
    const char * path = argc > 1 ? argv[1] : "model.gguf";
    gguf_file g(path);
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    printf("device: %s (sm_%d%d)  model: %s\n", prop.name, prop.major, prop.minor, path);
    cudaStream_t s;
    CK(cudaStreamCreate(&s));
    cublasHandle_t cb;
    cublasCreate(&cb);
    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    // ---------------- (1) segmented fold GEMM ----------------
    {
        printf("(1) fold_gemm vs gemm_packed\n");
        // one group with n_in 5120 (qkv / gate / beta / alpha like), one with n_in 8704 (ffn_down like)
        std::vector<std::vector<wt>> groups(2);
        auto pick = [&](const char * name, int64_t rows, int gi) {
            const gguf_tensor * t = g.find(name);
            if (!t) { printf("  missing %s\n", name); return; }
            groups[gi].push_back(load_w(g, t, std::min<int64_t>(rows, t->ne[1]), s));
        };
        pick("blk.0.attn_qkv.weight", 1280, 0);
        pick("blk.0.attn_gate.weight", 768, 0);
        pick("blk.0.ssm_beta.weight", 24, 0);
        pick("blk.0.ssm_alpha.weight", 24, 0);
        pick("blk.3.ffn_down.weight", 640, 1);
        pick("blk.3.attn_k.weight", 512, 0);   // a 5th weight is tested alone below
        const int TMAX = 520;
        for (auto & grp : groups) {
            if (grp.empty()) continue;
            const int64_t K = grp[0].n_in;
            const int64_t LDX = K + 64 * 3;   // strided rows like the pipeline's x16
            std::vector<float> hx((size_t) TMAX * K);
            for (size_t i = 0; i < hx.size(); ++i) hx[i] = nd(rng) * ((i % 991) == 7 ? 25.0f : 1.0f);
            float * x = up(hx);
            half * x16; float * cs;
            CK(cudaMalloc(&x16, (size_t) TMAX * LDX * 2));
            CK(cudaMalloc(&cs, TMAX * 4));
            const size_t sb = fast::gemm_packed_scratch_bytes(K, 8704, TMAX);
            void * scr;
            CK(cudaMalloc(&scr, sb));
            std::vector<half *> w16(grp.size());
            std::vector<float *> yref(grp.size()), y(grp.size());
            for (size_t i = 0; i < grp.size(); ++i) {
                printf("  weight %s %lld x %lld\n", grp[i].name.c_str(), (long long) grp[i].n_out, (long long) K);
                CK(cudaMalloc(&w16[i], (size_t) grp[i].n_out * K * 2));
                fast::packed_to_f16(grp[i].pk, 0, grp[i].n_out, w16[i], s);
                CK(cudaMalloc(&yref[i], (size_t) TMAX * grp[i].n_out * 4));
                CK(cudaMalloc(&y[i], (size_t) TMAX * grp[i].n_out * 4));
            }
            for (int T : {9, 20, 64, 65, 128, 300, 512}) {
                for (size_t i = 0; i < grp.size(); ++i) fast::gemm_packed(grp[i].pk, x, yref[i], T, scr, sb, s, cb);
                fast::prescale_rows(x, K, T, (int) K, x16, LDX, cs, s);
                // in chunks like the pipeline: [0, c1), [c1, T); segments in groups of up to 4
                const int c1 = T > 128 ? 128 : T;
                for (size_t i = 0; i < grp.size(); ++i) CK(cudaMemsetAsync(y[i], 0xff, (size_t) T * grp[i].n_out * 4, s));
                for (size_t g0 = 0; g0 < grp.size(); g0 += 4) {
                    const int ns = (int) std::min<size_t>(4, grp.size() - g0);
                    for (int ch = 0; ch < 2; ++ch) {
                        const int t0 = ch ? c1 : 0, t1 = ch ? T : c1;
                        if (t1 <= t0) continue;
                        fast::fold_seg segs[4];
                        for (int k = 0; k < ns; ++k) {
                            segs[k].w = w16[g0 + k];
                            segs[k].ldy = grp[g0 + k].n_out;
                            segs[k].rows = (int) grp[g0 + k].n_out;
                            segs[k].y = y[g0 + k] + (int64_t) t0 * segs[k].ldy;
                        }
                        fast::fold_gemm(segs, ns, x16 + (int64_t) t0 * LDX, LDX, cs + t0, t1 - t0, (int) K, s);
                    }
                }
                CK(cudaStreamSynchronize(s));
                long long m = 0;
                size_t n = 0;
                for (size_t i = 0; i < grp.size(); ++i) {
                    const size_t ni = (size_t) T * grp[i].n_out;
                    auto a = dl(yref[i], ni), b = dl(y[i], ni);
                    m += mism_bytes(a.data(), b.data(), ni, 4);
                    n += ni;
                }
                char msg[128];
                snprintf(msg, sizeof msg, "K=%lld T=%d, %zu weights, 2 chunks", (long long) K, T, grp.size());
                report(msg, m, n);
                // wire variants on the same x16: raster order (same f32 bits), f16 epilogue (f16 of those bits)
                for (int variant = 0; variant < 3; ++variant) {   // 0 raster, 1 f16 out, 2 bank-fixed SASS (cc 6.0)
                    fast::fold_opts o;
                    o.raster = 4;
                    o.out16 = variant == 1;
                    o.sass = variant == 2 ? 1 : 0;
                    int * sat;
                    CK(cudaMalloc(&sat, 4));
                    CK(cudaMemsetAsync(sat, 0, 4, s));
                    o.sat = sat;
                    std::vector<half *> y16(grp.size(), nullptr);
                    for (size_t i = 0; i < grp.size(); ++i) {
                        CK(cudaMemsetAsync(y[i], 0xff, (size_t) T * grp[i].n_out * 4, s));
                        CK(cudaMalloc(&y16[i], (size_t) T * grp[i].n_out * 2));
                        CK(cudaMemsetAsync(y16[i], 0xff, (size_t) T * grp[i].n_out * 2, s));
                    }
                    for (size_t g0 = 0; g0 < grp.size(); g0 += 4) {
                        const int ns = (int) std::min<size_t>(4, grp.size() - g0);
                        for (int ch = 0; ch < 2; ++ch) {
                            const int t0 = ch ? c1 : 0, t1 = ch ? T : c1;
                            if (t1 <= t0) continue;
                            fast::fold_seg segs[4];
                            for (int k = 0; k < ns; ++k) {
                                segs[k].w = w16[g0 + k];
                                segs[k].ldy = grp[g0 + k].n_out;
                                segs[k].rows = (int) grp[g0 + k].n_out;
                                segs[k].y = y[g0 + k] + (int64_t) t0 * segs[k].ldy;
                                segs[k].y16 = y16[g0 + k] + (int64_t) t0 * segs[k].ldy;
                            }
                            fast::fold_gemm(segs, ns, x16 + (int64_t) t0 * LDX, LDX, cs + t0, t1 - t0, (int) K, s, o);
                        }
                    }
                    CK(cudaStreamSynchronize(s));
                    long long mv = 0;
                    for (size_t i = 0; i < grp.size(); ++i) {
                        const size_t ni = (size_t) T * grp[i].n_out;
                        auto a = dl(yref[i], ni);
                        if (variant != 1) {
                            auto b = dl(y[i], ni);
                            mv += mism_bytes(a.data(), b.data(), ni, 4);
                        } else {
                            auto b = dl(y16[i], ni);
                            std::vector<half> e(ni);
                            for (size_t q = 0; q < ni; ++q) e[q] = __float2half_rn(a[q]);
                            mv += mism_bytes(e.data(), b.data(), ni, 2);
                        }
                        cudaFree(y16[i]);
                    }
                    const int nsat = dl(sat, 1)[0];
                    cudaFree(sat);
                    snprintf(msg, sizeof msg, "  %s, raster 4%s",
                             variant == 1 ? "f16 out = f16(f32 out)" : variant == 2 ? (prop.major == 6 && prop.minor == 0 && T > 64
                                 ? "bank-fixed SASS = compiled" : "SASS arm (compiled here: not sm_60 / T <= 64)")
                                 : "raster 4 = default order",
                             nsat ? " (saturated!)" : "");
                    report(msg, mv + nsat, n);
                }
                {   // one-read prescale_rows = two-read
                    half * x16b;
                    float * csb;
                    CK(cudaMalloc(&x16b, (size_t) T * LDX * 2));
                    CK(cudaMalloc(&csb, T * 4));
                    CK(cudaMemsetAsync(x16b, 0xff, (size_t) T * LDX * 2, s));
                    fast::prescale_rows(x, K, T, (int) K, x16b, LDX, csb, s, false);
                    CK(cudaStreamSynchronize(s));
                    std::vector<uint16_t> pa((size_t) T * K), pb((size_t) T * K);
                    for (int t = 0; t < T; ++t) {
                        CK(cudaMemcpy(pa.data() + (size_t) t * K, x16 + (size_t) t * LDX, K * 2, cudaMemcpyDeviceToHost));
                        CK(cudaMemcpy(pb.data() + (size_t) t * K, x16b + (size_t) t * LDX, K * 2, cudaMemcpyDeviceToHost));
                    }
                    auto ca = dl(cs, T), cbb = dl(csb, T);
                    snprintf(msg, sizeof msg, "  prescale_rows one-read = two-read (K=%lld T=%d)", (long long) K, T);
                    report(msg, mism_bytes(pa.data(), pb.data(), pa.size(), 2) + mism_bytes(ca.data(), cbb.data(), T, 4),
                           pa.size() + T);
                    cudaFree(x16b); cudaFree(csb);
                }
            }
            {   // timing on this device: one big segmented launch vs separate gemm_packed (incl. its unpack)
                const int T = 512;
                cudaEvent_t e0, e1;
                CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
                float ms1 = 0, ms2 = 0;
                CK(cudaEventRecord(e0, s));
                for (int rep = 0; rep < 3; ++rep)
                    for (size_t i = 0; i < grp.size(); ++i) fast::gemm_packed(grp[i].pk, x, yref[i], T, scr, sb, s, cb);
                CK(cudaEventRecord(e1, s));
                CK(cudaEventSynchronize(e1));
                CK(cudaEventElapsedTime(&ms1, e0, e1));
                CK(cudaEventRecord(e0, s));
                for (int rep = 0; rep < 3; ++rep) {
                    fast::prescale_rows(x, K, T, (int) K, x16, LDX, cs, s);
                    for (size_t g0 = 0; g0 < grp.size(); g0 += 4) {
                        const int ns = (int) std::min<size_t>(4, grp.size() - g0);
                        fast::fold_seg segs[4];
                        for (int k = 0; k < ns; ++k) {
                            segs[k].w = w16[g0 + k]; segs[k].ldy = grp[g0 + k].n_out;
                            segs[k].rows = (int) grp[g0 + k].n_out; segs[k].y = y[g0 + k];
                        }
                        fast::fold_gemm(segs, ns, x16, LDX, cs, T, (int) K, s);
                    }
                }
                CK(cudaEventRecord(e1, s));
                CK(cudaEventSynchronize(e1));
                CK(cudaEventElapsedTime(&ms2, e0, e1));
                printf("  time T=512: gemm_packed per weight %.3f ms, fold_gemm (pre-unpacked) %.3f ms\n", ms1 / 3, ms2 / 3);
            }
            for (size_t i = 0; i < grp.size(); ++i) { cudaFree(w16[i]); cudaFree(yref[i]); cudaFree(y[i]); }
            cudaFree(x); cudaFree(x16); cudaFree(cs); cudaFree(scr);
        }
    }

    // ---------------- (2) add + norm + prescale ----------------
    {
        printf("(2) add_norm_prescale vs k::add x2 + fast::rmsnorm + prescale_rows\n");
        const int E = 5120, LDX = 8704;
        for (int T : {1, 20, 128, 200}) {
            std::vector<float> hx((size_t) T * E), hp0(hx.size()), hp1(hx.size()), hw(E);
            for (auto & v : hx) v = nd(rng) * 3.0f;
            for (auto & v : hp0) v = nd(rng);
            for (auto & v : hp1) v = nd(rng) * 0.5f;
            for (auto & v : hw) v = 1.0f + 0.1f * nd(rng);
            float *xa = up(hx), *xb = up(hx), *p0 = up(hp0), *p1 = up(hp1), *w = up(hw), *y;
            CK(cudaMalloc(&y, hx.size() * 4));
            half *h1, *h2; float *c1, *c2;
            CK(cudaMalloc(&h1, (size_t) T * LDX * 2)); CK(cudaMalloc(&h2, (size_t) T * LDX * 2));
            CK(cudaMalloc(&c1, T * 4)); CK(cudaMalloc(&c2, T * 4));
            k::add(xa, p0, (int64_t) T * E, s);
            k::add(xa, p1, (int64_t) T * E, s);
            fast::rmsnorm(xa, w, y, T, E, 1e-6f, s);
            fast::prescale_rows(y, E, T, E, h1, LDX, c1, s);
            fast::add_norm_prescale(xb, p0, p1, w, 1e-6f, T, E, h2, LDX, c2, s);
            CK(cudaStreamSynchronize(s));
            auto xa_h = dl(xa, hx.size()), xb_h = dl(xb, hx.size());
            std::vector<uint16_t> a16((size_t) T * E), b16((size_t) T * E);
            for (int t = 0; t < T; ++t) {
                CK(cudaMemcpy(a16.data() + (size_t) t * E, h1 + (size_t) t * LDX, E * 2, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(b16.data() + (size_t) t * E, h2 + (size_t) t * LDX, E * 2, cudaMemcpyDeviceToHost));
            }
            auto ca = dl(c1, T), cbv = dl(c2, T);
            char msg[128];
            snprintf(msg, sizeof msg, "T=%d x after add", T);
            report(msg, mism_bytes(xa_h.data(), xb_h.data(), xa_h.size(), 4), xa_h.size());
            snprintf(msg, sizeof msg, "T=%d x16 + cs", T);
            report(msg, mism_bytes(a16.data(), b16.data(), a16.size(), 2) + mism_bytes(ca.data(), cbv.data(), T, 4), a16.size() + T);
            // no partials (layer 0 attn_norm) and add-only (last layer)
            float * xc = up(hx);
            fast::rmsnorm(xc, w, y, T, E, 1e-6f, s);
            fast::prescale_rows(y, E, T, E, h1, LDX, c1, s);
            fast::add_norm_prescale(xc, nullptr, nullptr, w, 1e-6f, T, E, h2, LDX, c2, s);
            float * xd = up(hx);
            fast::add_norm_prescale(xd, p0, p1, nullptr, 1e-6f, T, E, nullptr, LDX, nullptr, s);
            CK(cudaStreamSynchronize(s));
            for (int t = 0; t < T; ++t) {
                CK(cudaMemcpy(a16.data() + (size_t) t * E, h1 + (size_t) t * LDX, E * 2, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(b16.data() + (size_t) t * E, h2 + (size_t) t * LDX, E * 2, cudaMemcpyDeviceToHost));
            }
            snprintf(msg, sizeof msg, "T=%d norm only (no partials) x16", T);
            report(msg, mism_bytes(a16.data(), b16.data(), a16.size(), 2), a16.size());
            auto xd_h = dl(xd, hx.size());
            snprintf(msg, sizeof msg, "T=%d add only", T);
            report(msg, mism_bytes(xa_h.data(), xd_h.data(), xa_h.size(), 4), xa_h.size());
            {   // f16 wire: add_norm_prescale16 on f16 partials = add_norm_prescale on the same values as f32
                std::vector<half> q0(hp0.size()), q1(hp1.size());
                std::vector<float> w0(hp0.size()), w1(hp1.size());
                for (size_t i = 0; i < hp0.size(); ++i) {
                    q0[i] = __float2half_rn(hp0[i] * 300.0f); w0[i] = __half2float(q0[i]);
                    q1[i] = __float2half_rn(hp1[i] * 300.0f); w1[i] = __half2float(q1[i]);
                }
                half *d0 = up(q0), *d1 = up(q1);
                float *f0 = up(w0), *f1 = up(w1), *xe = up(hx), *xf = up(hx);
                fast::add_norm_prescale(xe, f0, f1, w, 1e-6f, T, E, h1, LDX, c1, s);
                fast::add_norm_prescale16(xf, d0, d1, w, 1e-6f, T, E, h2, LDX, c2, s);
                CK(cudaStreamSynchronize(s));
                for (int t = 0; t < T; ++t) {
                    CK(cudaMemcpy(a16.data() + (size_t) t * E, h1 + (size_t) t * LDX, E * 2, cudaMemcpyDeviceToHost));
                    CK(cudaMemcpy(b16.data() + (size_t) t * E, h2 + (size_t) t * LDX, E * 2, cudaMemcpyDeviceToHost));
                }
                auto xe_h = dl(xe, hx.size()), xf_h = dl(xf, hx.size());
                auto ce = dl(c1, T), cf = dl(c2, T);
                snprintf(msg, sizeof msg, "T=%d f16 partials (add_norm_prescale16) x, x16, cs", T);
                report(msg, mism_bytes(xe_h.data(), xf_h.data(), xe_h.size(), 4) + mism_bytes(a16.data(), b16.data(), a16.size(), 2) +
                                mism_bytes(ce.data(), cf.data(), T, 4), xe_h.size() + a16.size() + T);
                float *xg = up(hx), *xh = up(hx);   // add only
                fast::add_norm_prescale(xg, f0, f1, nullptr, 1e-6f, T, E, nullptr, LDX, nullptr, s);
                fast::add_norm_prescale16(xh, d0, d1, nullptr, 1e-6f, T, E, nullptr, LDX, nullptr, s);
                CK(cudaStreamSynchronize(s));
                auto xg_h = dl(xg, hx.size()), xh_h = dl(xh, hx.size());
                snprintf(msg, sizeof msg, "T=%d f16 partials, add only", T);
                report(msg, mism_bytes(xg_h.data(), xh_h.data(), xg_h.size(), 4), xg_h.size());
                for (void * p : {(void *) d0, (void *) d1, (void *) f0, (void *) f1, (void *) xe, (void *) xf, (void *) xg, (void *) xh})
                    cudaFree(p);
            }
            for (float * p : {xa, xb, xc, xd, p0, p1, w, y, c1, c2}) cudaFree(p);
            cudaFree(h1); cudaFree(h2);
        }
    }

    // ---------------- (3) SwiGLU + prescale ----------------
    {
        printf("(3) glu_prescale vs k::silu_mul + prescale_rows\n");
        const int K = 8704, LDX = 8704;
        for (int T : {3, 130}) {
            std::vector<float> hg((size_t) T * K), hu(hg.size());
            for (auto & v : hg) v = nd(rng) * 2.0f;
            for (auto & v : hu) v = nd(rng);
            float *ga = up(hg), *gb = up(hg), *u = up(hu);
            half *h1, *h2; float *c1, *c2;
            CK(cudaMalloc(&h1, (size_t) T * LDX * 2)); CK(cudaMalloc(&h2, (size_t) T * LDX * 2));
            CK(cudaMalloc(&c1, T * 4)); CK(cudaMalloc(&c2, T * 4));
            k::silu_mul(ga, u, (int64_t) T * K, s);
            fast::prescale_rows(ga, K, T, K, h1, LDX, c1, s);
            fast::glu_prescale(gb, u, T, K, h2, LDX, c2, s);
            CK(cudaStreamSynchronize(s));
            auto a = dl((uint16_t *) h1, (size_t) T * LDX), b = dl((uint16_t *) h2, (size_t) T * LDX);
            auto ca = dl(c1, T), cbv = dl(c2, T);
            char msg[128];
            snprintf(msg, sizeof msg, "T=%d x16 + cs", T);
            report(msg, mism_bytes(a.data(), b.data(), a.size(), 2) + mism_bytes(ca.data(), cbv.data(), T, 4), a.size() + T);
            for (float * p : {ga, gb, u, c1, c2}) cudaFree(p);
            cudaFree(h1); cudaFree(h2);
        }
    }

    // ---------------- (4) prefill attention ----------------
    {
        printf("(4) attn_prefill vs attn_decode (4 queries at a time)\n");
        const int Hkv = 2, Hq = 12, D = 256, NCTX = 2304;
        for (uint32_t kvt : {fast::KV_F16, fast::KV_Q8_0, fast::KV_Q4_0}) {
        printf(" %s cache\n", kvt == fast::KV_F16 ? "f16" : kvt == fast::KV_Q8_0 ? "q8_0" : "q4_0");
        fast::kv_cache kv = fast::kv_alloc(kvt, NCTX, Hkv, D);
        struct cs_ { int p0, T; } cases[] = {{0, 5}, {0, 20}, {0, 130}, {61, 77}, {0, 517}, {1000, 33}, {0, 2048}};
        for (auto c : cases) {
            const int n = c.p0 + c.T;
            std::vector<float> hk((size_t) n * Hkv * D), hv(hk.size()), hq((size_t) c.T * Hq * D);
            for (auto & v : hk) v = nd(rng);
            for (auto & v : hv) v = nd(rng);
            for (auto & v : hq) v = nd(rng) * 2.0f;
            float *dk = up(hk), *dv = up(hv), *dq = up(hq), *o1, *o2;
            CK(cudaMalloc(&o1, hq.size() * 4)); CK(cudaMalloc(&o2, hq.size() * 4));
            fast::kv_append(kv, dk, dv, 0, n, s);
            const float scale = 1.0f / 16.0f;
            cudaEvent_t e0, e1, e2;
            CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1)); CK(cudaEventCreate(&e2));
            CK(cudaEventRecord(e0, s));
            for (int t0 = 0; t0 < c.T; t0 += 4)
                fast::attn_decode(dq + (size_t) t0 * Hq * D, kv, o1 + (size_t) t0 * Hq * D, std::min(4, c.T - t0), c.p0 + t0, Hq, scale, s);
            CK(cudaEventRecord(e1, s));
            const bool ok = fast::attn_prefill(dq, kv, o2, c.T, c.p0, Hq, scale, s);
            CK(cudaEventRecord(e2, s));
            CK(cudaStreamSynchronize(s));
            float ms1 = 0, ms2 = 0;
            CK(cudaEventElapsedTime(&ms1, e0, e1)); CK(cudaEventElapsedTime(&ms2, e1, e2));
            auto a = dl(o1, hq.size()), b = dl(o2, hq.size());
            double md = 0, mr = 0;
            int bad = 0;
            for (size_t i = 0; i < a.size(); ++i) {
                if (!std::isfinite(b[i])) ++bad;
                md = std::max(md, (double) std::fabs(a[i] - b[i]));
                mr = std::max(mr, (double) std::fabs(a[i]));
            }
            const double rel = md / (mr + 1e-30);
            const bool pass = ok && !bad && rel < 1e-5;
            printf("  p0=%4d T=%4d: max|diff|/max|ref| %.2e nonfinite %d  time decode-loop %.3f ms, prefill %.3f ms %s\n",
                   c.p0, c.T, rel, bad, ms1, ms2, pass ? "OK" : "FAIL");
            if (!pass) ++fails;
            for (float * p : {dk, dv, dq, o1, o2}) cudaFree(p);
        }
        fast::kv_free(kv);
        }
    }

    printf("%s (%d failing checks)\n", fails ? "FAILED" : "ALL PASSED", fails);
    return fails ? 1 : 0;
}

int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    try {
        return run(argc, argv);
    } catch (const std::exception & e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
