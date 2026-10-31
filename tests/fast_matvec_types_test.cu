// p100e-fast-matvec-types-test: the model-coverage GEMVs on synthetic weights.
//  - F32 / F16 / BF16 weights (float_gemv, matvec_generic.cu): fast::has_fast_path is true, matvec() vs an fp64 CPU
//    reference (max |err| / max |ref| <= 1e-5), T = 1..8
//  - Q4_1 (raw blocks, the fork's a16 HFMA2 GEMV, matvec_q41.cu): vs the fp64 reference of the dequantized weights
//    (bounded by the a16 activation rounding, <= 4e-3) and vs the generic f32 kernel it replaces
//  - T invariance: column j of every T-column call is bit-identical to the T = 1 call on that column (MTP verify)
//  - matvec_prequantized (fused decode) gives the bits of matvec()
//  - regressions: the generic GEMV on IQ4_NL / MXFP4 / Q1_0 / Q2_0 rows with n_in % 256 != 0
//    (the partial last unit) vs an fp64 CPU reference; Q4_0 (q4r) T invariance; Q8_0 / F32 with a misaligned x (scalar
//    fallback, same bits as an aligned copy); an a16 group with |x| > 65504 stays finite.
//    --hashes: only FNV hashes of the outputs on shapes the fixes must not change (old vs new build bit identity)
// Exit code 0 only if every check passes.
#include "fast/matvec.cuh"
#include "matvec_internal.cuh"
#include "gguf.h"
#include "fast/unpack.cuh"
#include "convert.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using namespace p100e;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

static int fails = 0;

// ---- regressions ---------------------------------------------------------------------------
static uint32_t fnv(const std::vector<float> & v, uint32_t h = 2166136261u) {
    for (float f : v) { uint32_t u; memcpy(&u, &f, 4); for (int k = 0; k < 4; ++k) { h ^= (u >> (8 * k)) & 0xff; h *= 16777619u; } }
    return h;
}
static const int8_t kv_iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
static const int8_t kv_fp4[16]   = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

// random blocks of type ty (n_out rows of n_in) and their CPU dequantization (ggml's block formats)
static bool make_blocks(uint32_t ty, int64_t n_in, int64_t n_out, std::mt19937 & rng, std::vector<uint8_t> & wb, std::vector<double> & wd) {
    const int blk = ty == GGML_TYPE_Q1_0 ? 128 : ty == GGML_TYPE_Q2_0 ? 64 : 32;
    const int bs = ty == GGML_TYPE_MXFP4 ? 17 : ty == GGML_TYPE_Q8_0 ? 34 : 18;
    if (n_in % blk) return false;
    const int64_t nb = n_in / blk;
    wb.assign((size_t) n_out * nb * bs, 0);
    wd.assign((size_t) n_out * n_in, 0.0);
    for (int64_t r = 0; r < n_out; ++r)
        for (int64_t b = 0; b < nb; ++b) {
            uint8_t * p = wb.data() + ((size_t) r * nb + b) * bs;
            double * o = wd.data() + (size_t) r * n_in + b * blk;
            for (int i = 0; i < bs; ++i) p[i] = (uint8_t) rng();
            if (ty == GGML_TYPE_MXFP4) {
                p[0] = (uint8_t) (118 + rng() % 8);   // e8m0 scale 2^-9 .. 2^-2
                const double d = std::ldexp(1.0, (int) p[0] - 127);
                for (int i = 0; i < 16; ++i) { o[i] = d * kv_fp4[p[1 + i] & 15] * 0.5; o[i + 16] = d * kv_fp4[p[1 + i] >> 4] * 0.5; }
                continue;
            }
            const half h = __float2half(0.002f + 0.01f * (rng() % 1000) / 1000.0f);
            memcpy(p, &h, 2);
            const double d = __half2float(h);
            const uint8_t * q = p + 2;
            if (ty == GGML_TYPE_IQ4_NL)
                for (int i = 0; i < 16; ++i) { o[i] = d * kv_iq4nl[q[i] & 15]; o[i + 16] = d * kv_iq4nl[q[i] >> 4]; }
            else if (ty == GGML_TYPE_Q4_0)
                for (int i = 0; i < 16; ++i) { o[i] = d * ((q[i] & 15) - 8); o[i + 16] = d * ((q[i] >> 4) - 8); }
            else if (ty == GGML_TYPE_Q8_0)
                for (int i = 0; i < 32; ++i) o[i] = d * (int8_t) q[i];
            else if (ty == GGML_TYPE_Q1_0)
                for (int i = 0; i < 128; ++i) o[i] = (q[i >> 3] >> (i & 7)) & 1 ? d : -d;
            else   // Q2_0: weight = (code - 1) * d, 4 codes per byte, low bits first
                for (int i = 0; i < 64; ++i) o[i] = (((q[i >> 2] >> (2 * (i & 3))) & 3) - 1) * d;
        }
    return true;
}

static std::vector<float> d2h(const float * d, size_t n) {
    std::vector<float> h(n);
    CK(cudaMemcpy(h.data(), d, n * 4, cudaMemcpyDeviceToHost));
    return h;
}

static std::vector<double> ref_of(const std::vector<double> & wd, const std::vector<float> & hx, int T, int64_t n_in, int64_t n_out, double & rmax) {
    std::vector<double> ref((size_t) T * n_out);
    rmax = 0;
    for (int j = 0; j < T; ++j)
        for (int64_t r = 0; r < n_out; ++r) {
            double a = 0;
            for (int64_t k = 0; k < n_in; ++k) a += wd[(size_t) r * n_in + k] * hx[(size_t) j * n_in + k];
            ref[(size_t) j * n_out + r] = a;
            rmax = std::max(rmax, std::fabs(a));
        }
    return ref;
}

static double worst_of(const std::vector<float> & h, const std::vector<double> & ref, double rmax) {
    double w = 0;
    for (size_t i = 0; i < h.size(); ++i) w = std::max(w, std::isfinite(h[i]) ? std::fabs(h[i] - ref[i]) / rmax : INFINITY);
    return w;
}

// hashes = true: only the FNV hashes of outputs the fixes must leave bit-identical (run old and new builds, compare)
static void regress(cudaStream_t s, bool hashes) {
    std::mt19937 rng(777);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    const int TMAX = 8;
    // (1) generic GEMV, partial last 256-unit (generic-partial-unit-stale-smem). n_out 37: Q1_0 / Q2_0 stay RAW in pack_weight
    {
        const uint32_t types[] = {GGML_TYPE_IQ4_NL, GGML_TYPE_MXFP4, GGML_TYPE_Q1_0, GGML_TYPE_Q2_0, GGML_TYPE_Q4_0};
        const char * names[] = {"iq4_nl", "mxfp4", "q1_0", "q2_0", "q4_0"};
        const int64_t n_ins[] = {64, 128, 640, 2880, 1088, 2560, 4096};
        for (int ti = 0; ti < 5; ++ti)
            for (int64_t n_in : n_ins) {
                const uint32_t ty = types[ti];
                const int64_t n_out = 37;
                std::vector<uint8_t> wb; std::vector<double> wd;
                if (!make_blocks(ty, n_in, n_out, rng, wb, wd)) continue;
                std::vector<float> hx((size_t) TMAX * n_in);
                for (auto & v : hx) v = nd(rng);
                if (hashes && n_in % 256 != 0) continue;   // (after the rng draws: the same data in both modes)
                const fast::packed_weight w = fast::pack_weight(ty, nullptr, n_in, n_out, s);
                void * dw; float * x; float * y;
                CK(cudaMalloc(&dw, wb.size())); CK(cudaMalloc(&x, hx.size() * 4)); CK(cudaMalloc(&y, (size_t) TMAX * n_out * 4));
                CK(cudaMemcpy(dw, wb.data(), wb.size(), cudaMemcpyHostToDevice));
                CK(cudaMemcpy(x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice));
                const size_t rb = wb.size() / n_out;
                double worst = 0, rmax = 0;
                uint32_t h = 2166136261u;
                std::vector<double> ref;
                if (!hashes) ref = ref_of(wd, hx, TMAX, n_in, n_out, rmax);
                for (int T = 1; T <= TMAX; ++T) {
                    CK(cudaMemsetAsync(y, 0xff, (size_t) T * n_out * 4, s));
                    fast::launch_generic(ty, (const char *) dw, x, y, T, n_in, n_out, rb, s);
                    CK(cudaStreamSynchronize(s)); CK(cudaGetLastError());
                    const std::vector<float> hy = d2h(y, (size_t) T * n_out);
                    h = fnv(hy, h);
                    if (!hashes) worst = std::max(worst, worst_of(hy, std::vector<double>(ref.begin(), ref.begin() + (size_t) T * n_out), rmax));
                }
                if (hashes) printf("  hash generic %-6s %5lld x %lld: %08x\n", names[ti], (long long) n_in, (long long) n_out, h);
                else {
                    const bool ok = worst <= 1e-5 && w.bytes != 0;
                    printf("  generic %-6s %5lld x %lld (n_in %% 256 = %3lld, layout %d): max|err|/max|ref| %.2e (limit 1e-05)  %s\n", names[ti],
                           (long long) n_in, (long long) n_out, (long long) (n_in % 256), w.layout, worst, ok ? "PASS" : "FAIL");
                    if (!ok) ++fails;
                }
                cudaFree(dw); cudaFree(x); cudaFree(y);
            }
    }
    // (2) Q4_0 on q4r (q4r-not-t-invariant): column j of a T-column call = the T = 1 call; T = 1 hash for old vs new.
    // (3) big = 1: one a16 group per column with |x| = 1e5 > 65504 (a16-amax-f16-overflow): finite, near the reference
    for (int big = 0; big < 2; ++big) {
        if (hashes && big) continue;
        const int64_t shp[][2] = {{2560, 64}, {640, 256}, {5120, 48}, {4096, 1024}};
        for (auto & sh : shp) {
            const int64_t n_in = sh[0], n_out = sh[1];
            std::vector<uint8_t> wb; std::vector<double> wd;
            make_blocks(GGML_TYPE_Q4_0, n_in, n_out, rng, wb, wd);
            std::vector<float> hx((size_t) TMAX * n_in);
            for (auto & v : hx) v = nd(rng);
            if (big) for (int j = 0; j < TMAX; ++j) hx[(size_t) j * n_in + 32 * (j + 3) + 5] = j & 1 ? -1.0e5f : 1.0e5f;
            void * dw; float * x; float * y; void * scr;
            CK(cudaMalloc(&dw, wb.size())); CK(cudaMalloc(&x, hx.size() * 4)); CK(cudaMalloc(&y, (size_t) TMAX * n_out * 4));
            CK(cudaMalloc(&scr, fast::matvec_scratch_bytes(n_in, TMAX)));
            CK(cudaMemcpy(dw, wb.data(), wb.size(), cudaMemcpyHostToDevice));
            CK(cudaMemcpy(x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice));
            const fast::packed_weight w = fast::pack_weight(GGML_TYPE_Q4_0, dw, n_in, n_out, s);
            CK(cudaStreamSynchronize(s));
            std::vector<float> h1((size_t) TMAX * n_out);
            for (int j = 0; j < TMAX; ++j) {
                fast::matvec(w, x + (size_t) j * n_in, y, 1, scr, s);
                CK(cudaStreamSynchronize(s));
                const std::vector<float> hy = d2h(y, (size_t) n_out);
                std::copy(hy.begin(), hy.end(), h1.begin() + (size_t) j * n_out);
            }
            if (hashes) {
                printf("  hash q4r T=1 %5lld x %5lld (%s): %08x\n", (long long) n_in, (long long) n_out, fast::kernel_name(w), fnv(h1));
            } else {
                double rmax = 0;
                const std::vector<double> ref = ref_of(wd, hx, TMAX, n_in, n_out, rmax);
                // the columns that must equal T = 1 follow P100E_MV_Q4R_TINV (matvec_q4r.cu): 0 (default, the earlier
                // bit-identical warp choice) none, 1 T = 2..4 (MTP verify), 2 every T; the rest are counted, informational
                static const int tinv_mode = getenv("P100E_MV_Q4R_TINV") ? atoi(getenv("P100E_MV_Q4R_TINV")) : 0;
                const int strict_max = tinv_mode >= 2 ? TMAX : tinv_mode == 1 ? 4 : 1;
                long long tinv = 0, tinv58 = 0;
                double worst = worst_of(h1, ref, rmax);
                for (int T = 2; T <= TMAX; ++T) {
                    CK(cudaMemsetAsync(y, 0xff, (size_t) T * n_out * 4, s));
                    fast::matvec(w, x, y, T, scr, s);
                    CK(cudaStreamSynchronize(s));
                    const std::vector<float> hy = d2h(y, (size_t) T * n_out);
                    for (size_t i = 0; i < hy.size(); ++i) (T <= strict_max ? tinv : tinv58) += memcmp(&hy[i], &h1[i], 4) != 0;
                    worst = std::max(worst, worst_of(hy, ref, rmax));
                }
                const bool ok = worst <= 4e-3 && tinv == 0;
                printf("  q4_0 %s%5lld x %5lld (%s): max|err|/max|ref| %.2e (limit 4e-03), T-invariance %lld differ "
                       "(strict up to T = %d; rest informational: %lld)  %s\n", big ? "|x|=1e5 " : "", (long long) n_in, (long long) n_out,
                       fast::kernel_name(w), worst, tinv, strict_max, tinv58, ok ? "PASS" : "FAIL");
                if (!ok) ++fails;
            }
            cudaFree(dw); cudaFree(x); cudaFree(y); cudaFree(scr);
        }
    }
    // (4) Q8_0 (q8r) and F32 with x offset by one float (q8-fallback-misaligned-x): no fault, the bits of an aligned copy
    if (hashes) return;
    for (int ty : {(int) GGML_TYPE_Q8_0, (int) GGML_TYPE_F32}) {
        const int64_t n_in = ty == GGML_TYPE_F32 ? 516 : 512, n_out = 64;
        std::vector<uint8_t> wb; std::vector<double> wd;
        if (ty == GGML_TYPE_Q8_0) make_blocks(GGML_TYPE_Q8_0, n_in, n_out, rng, wb, wd);
        else {
            wb.resize((size_t) n_in * n_out * 4);
            for (size_t i = 0; i < wb.size() / 4; ++i) { const float v = 0.05f * nd(rng); memcpy(wb.data() + 4 * i, &v, 4); }
        }
        std::vector<float> hx((size_t) TMAX * n_in);
        for (auto & v : hx) v = nd(rng);
        void * dw; float * xa; float * xm; float * y; void * scr;
        CK(cudaMalloc(&dw, wb.size())); CK(cudaMalloc(&xa, hx.size() * 4)); CK(cudaMalloc(&xm, hx.size() * 4 + 16));
        CK(cudaMalloc(&y, (size_t) TMAX * n_out * 4)); CK(cudaMalloc(&scr, fast::matvec_scratch_bytes(n_in, TMAX)));
        CK(cudaMemcpy(dw, wb.data(), wb.size(), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(xa, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(xm + 1, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice));
        const fast::packed_weight w = fast::pack_weight((uint32_t) ty, dw, n_in, n_out, s);
        CK(cudaStreamSynchronize(s));
        long long diff = 0;
        for (int T = 1; T <= TMAX; ++T) {
            fast::matvec(w, xa, y, T, scr, s);
            CK(cudaStreamSynchronize(s));
            const std::vector<float> ha = d2h(y, (size_t) T * n_out);
            CK(cudaMemsetAsync(y, 0xff, (size_t) T * n_out * 4, s));
            fast::matvec(w, xm + 1, y, T, scr, s);
            CK(cudaStreamSynchronize(s)); CK(cudaGetLastError());
            const std::vector<float> hm = d2h(y, (size_t) T * n_out);
            for (size_t i = 0; i < ha.size(); ++i) diff += memcmp(&ha[i], &hm[i], 4) != 0;
        }
        printf("  %s misaligned x (%s): %lld differ from the aligned copy, T = 1..8  %s\n", ty == GGML_TYPE_F32 ? "f32 " : "q8_0",
               fast::kernel_name(w), diff, diff == 0 ? "PASS" : "FAIL");
        if (diff) ++fails;
        cudaFree(dw); cudaFree(xa); cudaFree(xm); cudaFree(y); cudaFree(scr);
    }
    // (5) Q8_0 (q8r) shards of more than 2048 rows at T = 1 (ssm_out-v215: q8wr 215 by default, P100E_MV_Q8_BIG): the
    // default and every q8wr variant give the legacy q8r_gemv's bits (mv_tune.variant = -1), and column j of a T-column
    // call (T = 2..8) is the T = 1 call on column j. n_in > 5120 keeps the legacy default (v0); the q8wr variants still run there
    // (17408 -> 2304 stages 70 KB of x, > 48 KB: the staged ones fall back to legacy).
    {
        const fast::mv_tune_t saved = fast::mv_tune;
        struct shp { int64_t n_in, n_out; };
        for (const shp sh : {shp{3072, 5120}, shp{10240, 5120}, shp{8704, 5120}, shp{5120, 2049}, shp{17408, 2304}}) {
            std::vector<uint8_t> wb; std::vector<double> wd;
            make_blocks(GGML_TYPE_Q8_0, sh.n_in, sh.n_out, rng, wb, wd);
            std::vector<float> hx((size_t) TMAX * sh.n_in);
            for (auto & v : hx) v = nd(rng);
            void * dw; float * x; float * y; void * scr;
            CK(cudaMalloc(&dw, wb.size())); CK(cudaMalloc(&x, hx.size() * 4));
            CK(cudaMalloc(&y, (size_t) TMAX * sh.n_out * 4)); CK(cudaMalloc(&scr, fast::matvec_scratch_bytes(sh.n_in, TMAX)));
            CK(cudaMemcpy(dw, wb.data(), wb.size(), cudaMemcpyHostToDevice));
            CK(cudaMemcpy(x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice));
            const fast::packed_weight w = fast::pack_weight(GGML_TYPE_Q8_0, dw, sh.n_in, sh.n_out, s);
            CK(cudaStreamSynchronize(s));
            auto run = [&](int variant, int T, const float * xx) {
                fast::mv_tune.variant = variant;
                CK(cudaMemsetAsync(y, 0xff, (size_t) T * sh.n_out * 4, s));
                fast::matvec(w, xx, y, T, scr, s);
                CK(cudaStreamSynchronize(s)); CK(cudaGetLastError());
                fast::mv_tune.variant = saved.variant;
                return d2h(y, (size_t) T * sh.n_out);
            };
            std::vector<std::vector<float>> leg(TMAX), def(TMAX);
            for (int j = 0; j < TMAX; ++j) { leg[j] = run(-1, 1, x + (size_t) j * sh.n_in); def[j] = run(0, 1, x + (size_t) j * sh.n_in); }
            long long dlegacy = 0, dvar = 0, tinv = 0;
            for (int j = 0; j < TMAX; ++j)
                for (int64_t i = 0; i < sh.n_out; ++i) dlegacy += memcmp(&def[j][i], &leg[j][i], 4) != 0;
            for (int v = 211; v <= 218; ++v) {
                const std::vector<float> yv = run(v, 1, x);
                for (int64_t i = 0; i < sh.n_out; ++i) dvar += memcmp(&yv[i], &leg[0][i], 4) != 0;
            }
            for (int T = 2; T <= TMAX; ++T) {
                const std::vector<float> yT = run(0, T, x);
                for (int j = 0; j < T; ++j)
                    for (int64_t i = 0; i < sh.n_out; ++i) tinv += memcmp(&yT[(size_t) j * sh.n_out + i], &leg[j][i], 4) != 0;
            }
            const bool ok = dlegacy == 0 && dvar == 0 && tinv == 0;
            printf("  q8_0 %5lld -> %5lld T=1 default v%d: %lld differ from legacy (8 columns), q8wr 211..218 %lld, T = 2..8 vs T = 1 %lld  %s\n",
                   (long long) sh.n_in, (long long) sh.n_out, fast::default_t1_variant(w.layout, w.type, sh.n_in, sh.n_out), dlegacy, dvar,
                   tinv, ok ? "PASS" : "FAIL");
            if (!ok) ++fails;
            cudaFree(dw); cudaFree(x); cudaFree(y); cudaFree(scr);
        }
        fast::mv_tune = saved;
    }
}

static uint16_t f2bf(float f) {   // round to nearest even
    uint32_t u; memcpy(&u, &f, 4);
    u += 0x7fff + ((u >> 16) & 1);
    return (uint16_t) (u >> 16);
}
static float bf2f(uint16_t b) { uint32_t u = (uint32_t) b << 16; float f; memcpy(&f, &u, 4); return f; }

// --bench: GB/s of the quantized GEMVs on the MTP draft block shapes (random bytes; matvec_prequantized = the decode path,
// activation quantize outside the timing), T = 1, 2, 4. usage: p100e-fast-matvec-types-test --bench [reps]
static int bench(int reps) {
    cudaStream_t s;
    CK(cudaStreamCreate(&s));
    struct shape { const char * name; int64_t n_in, n_out; };
    const shape shapes[] = {{"attn_q", 5120, 12288}, {"ffn_gate|up", 5120, 17408}, {"ffn_down", 17408, 5120}, {"eh_proj", 10240, 5120},
                            {"exp_gate|up", 2560, 640}, {"exp_down", 640, 2560}};   // one Flash-Next expert (n_in -> n_out)
    const uint32_t types[] = {GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_M, GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S, GGML_TYPE_Q1_0, GGML_TYPE_Q2_0};
    const char * tn[] = {"iq1_S", "iq1_M", "q2_K", "q3_K", "q4_K", "iq2xxs", "iq2xs", "iq2_s", "iq3_xxs", "iq3_s", "q1_0", "q2_0"};
    const int ntypes = (int) (sizeof(types)/sizeof(types[0]));
    printf("%-12s %-7s %8s %8s %-6s | %9s %9s %9s  (GB/s of weight bytes, matvec_prequantized)\n", "tensor", "type", "n_in", "n_out", "kernel", "T=1", "T=2", "T=4");
    for (const shape & sh : shapes)
        for (int ti = 0; ti < ntypes; ++ti) {
            const uint32_t ty = types[ti];
            if (sh.n_in % get_type_info(ty).blck) continue;   // e.g. 640-wide expert rows and 256-block types
            const size_t bytes = (size_t) sh.n_out * (sh.n_in / get_type_info(ty).blck) * get_type_info(ty).size;
            std::vector<uint8_t> hb(bytes);
            std::mt19937 rng(7);
            for (size_t i = 0; i < bytes; ++i) hb[i] = (uint8_t) rng();
            void * dw; CK(cudaMalloc(&dw, bytes));
            CK(cudaMemcpy(dw, hb.data(), bytes, cudaMemcpyHostToDevice));
            const fast::packed_weight w = fast::pack_weight(ty, dw, sh.n_in, sh.n_out, s);
            CK(cudaStreamSynchronize(s));
            std::vector<float> hx((size_t) 4 * sh.n_in);
            std::normal_distribution<float> nd(0.0f, 1.0f);
            for (auto & v : hx) v = nd(rng);
            float *x, *y; void * scr;
            CK(cudaMalloc(&x, hx.size() * 4)); CK(cudaMemcpy(x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice));
            CK(cudaMalloc(&y, (size_t) 4 * sh.n_out * 4));
            CK(cudaMalloc(&scr, fast::matvec_scratch_bytes(sh.n_in, 8)));
            double gbs[3]; const int Ts[3] = {1, 2, 4};
            for (int k = 0; k < 3; ++k) {
                fast::matvec_quantize(x, sh.n_in, Ts[k], scr, s);
                for (int i = 0; i < 20; ++i) fast::matvec_prequantized(w, x, scr, y, Ts[k], s);
                cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
                CK(cudaStreamSynchronize(s));
                CK(cudaEventRecord(e0, s));
                for (int i = 0; i < reps; ++i) fast::matvec_prequantized(w, x, scr, y, Ts[k], s);
                CK(cudaEventRecord(e1, s)); CK(cudaEventSynchronize(e1)); CK(cudaGetLastError());
                float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1));
                gbs[k] = (double) bytes / (ms / reps * 1e-3) / 1e9;
                cudaEventDestroy(e0); cudaEventDestroy(e1);
            }
            printf("%-12s %-7s %8lld %8lld %-6s | %9.1f %9.1f %9.1f\n", sh.name, tn[ti], (long long) sh.n_in, (long long) sh.n_out, fast::kernel_name(w), gbs[0], gbs[1], gbs[2]);
            cudaFree(dw); cudaFree(x); cudaFree(y); cudaFree(scr);
        }
    return 0;
}

// ---- padded Q6_K rows (P100E_Q6_PAD) -----------------------------------------------------------
// Q6_K shapes whose raw row stride is not 8-byte aligned (nb = n_in/256: 34 = the 2-GPU ffn_down shard, 10, 6, odd 5
// and 17): the rows laid out with fast::pack_row_stride, packed (q6r3 on the warp-rows family), and checked:
//   - accuracy vs the fp64 product of ggml's f32 dequant of the raw bytes (a16 rounding bound 4e-3), next to the raw
//     stride packing (q6r2 legacy kernel / generic) for comparison
//   - T invariance: column j of the T = 2..8 call equals the T = 1 call bit for bit
//   - matvec_prequantized == matvec bit for bit, T = 1..8
//   - two parts concatenated (matvec_prequantized_parts) == each part alone on a row view, T = 1..4
//   - the 16-byte padding (VA 16 decoders) gives the same bits as the 8-byte padding, T = 1 and 3
//   - packed_to_f16 (direct and smem, odd row chunk) == ggml to-f16 of the raw rows, bit for bit
// hashes_only: print an FNV hash of the default-stride (pack_row_stride) T = 1..4 outputs per shape (build A/B).
static void q6pad_checks(cudaStream_t s, bool hashes_only) {
    struct shp { int64_t n_in, n_out; };
    // 17408 x 256: raw stride already 8-byte aligned (q6r3, no padding); the 1-GPU ffn_down width, where 3 columns fit neither
    // in shared memory nor in the hybrid geometry 22, so T = 3 needs mv_wrm's K-phase fallback with P100E_MV_Q6_H16 on
    const shp shapes[] = {{8704, 5120}, {8704, 64}, {2560, 512}, {1536, 256}, {1280, 256}, {4352, 128}, {17408, 256}};
    std::mt19937 rng(9157);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    const int TMAX = 8;
    for (const shp sh : shapes) {
        const int64_t n_in = sh.n_in, n_out = sh.n_out, nb = n_in / 256;
        const size_t rb = (size_t) nb * 210;
        std::vector<uint8_t> wb(rb * n_out);
        for (size_t b = 0; b < (size_t) n_out * nb; ++b) {
            uint8_t * blk = wb.data() + b * 210;
            for (int i = 0; i < 210; ++i) blk[i] = (uint8_t) rng();
            const half d = __float2half((0.0002f + 0.0006f * (rng() % 1000) / 1000.0f) * (rng() & 1 ? -1.0f : 1.0f));
            memcpy(blk + 208, &d, 2);
        }
        // odd nb: the loader keeps the raw stride (pack_row_stride), so the padded packing is checked with an explicit
        // 8-byte stride (the mechanism, not the default policy)
        const size_t ps = nb % 2 ? (rb + 7) / 8 * 8 : fast::pack_row_stride(GGML_TYPE_Q6_K, n_in, n_out);
        auto padded_bytes = [&](size_t st) {
            std::vector<uint8_t> p(st * n_out, 0);
            for (int64_t r = 0; r < n_out; ++r) memcpy(p.data() + r * st, wb.data() + r * rb, rb);
            return p;
        };
        auto upload_pack = [&](size_t st, void *& dptr) {
            const std::vector<uint8_t> p = padded_bytes(st);
            CK(cudaMalloc(&dptr, p.size()));
            CK(cudaMemcpy(dptr, p.data(), p.size(), cudaMemcpyHostToDevice));
            const fast::packed_weight w = fast::pack_weight(GGML_TYPE_Q6_K, dptr, n_in, n_out, s, st);
            CK(cudaStreamSynchronize(s));
            return w;
        };
        void *dp = nullptr, *dr = nullptr, *d16p = nullptr;
        const fast::packed_weight wp = upload_pack(ps, dp);                  // default stride (padded unless P100E_Q6_PAD=0)
        const fast::packed_weight wr = upload_pack(rb, dr);                  // raw stride (the old packing)
        const size_t ps16 = (rb + 15) / 16 * 16;
        const fast::packed_weight w16 = upload_pack(ps16, d16p);             // 16-byte padding

        std::vector<float> hx((size_t) TMAX * n_in);
        for (auto & v : hx) v = nd(rng);
        float *x, *y, *y1, *yq;
        void *scr, *qb;
        CK(cudaMalloc(&x, hx.size() * 4)); CK(cudaMalloc(&y, (size_t) TMAX * n_out * 4)); CK(cudaMalloc(&y1, (size_t) TMAX * n_out * 4));
        CK(cudaMalloc(&yq, (size_t) TMAX * n_out * 4));
        CK(cudaMalloc(&scr, fast::matvec_scratch_bytes(n_in, TMAX))); CK(cudaMalloc(&qb, fast::matvec_scratch_bytes(n_in, TMAX)));
        CK(cudaMemcpy(x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice));
        // T = 1 reference per column, default-stride weight
        for (int j = 0; j < TMAX; ++j) fast::matvec(wp, x + (size_t) j * n_in, y1 + (size_t) j * n_out, 1, scr, s);
        CK(cudaStreamSynchronize(s));
        const std::vector<float> h1 = d2h(y1, (size_t) TMAX * n_out);
        if (hashes_only) {
            uint32_t h = fnv(h1);
            for (int T = 2; T <= 4; ++T) { fast::matvec(wp, x, y, T, scr, s); CK(cudaStreamSynchronize(s)); h = fnv(d2h(y, (size_t) T * n_out), h); }
            printf("Q6PAD-HASH %5lld x %5lld stride %zu layout %d %08x\n", (long long) n_in, (long long) n_out, fast::row_stride(wp), wp.layout, h);
            cudaFree(dp); cudaFree(dr); cudaFree(d16p); cudaFree(x); cudaFree(y); cudaFree(y1); cudaFree(yq); cudaFree(scr); cudaFree(qb);
            continue;
        }
        // accuracy vs fp64 of ggml's f32 dequant (column 0)
        double worst_p = 0, worst_r = 0;
        {
            void * draw; float * df;
            CK(cudaMalloc(&draw, wb.size())); CK(cudaMalloc(&df, (size_t) n_in * n_out * 4));
            CK(cudaMemcpy(draw, wb.data(), wb.size(), cudaMemcpyHostToDevice));
            ggml_get_to_fp32_cuda(GGML_TYPE_Q6_K)(draw, df, n_in * n_out, s);
            CK(cudaStreamSynchronize(s));
            const std::vector<float> hf = d2h(df, (size_t) n_in * n_out);
            std::vector<double> wd(hf.begin(), hf.end());
            double rmax = 0;
            const std::vector<double> ref = ref_of(wd, hx, 1, n_in, n_out, rmax);
            worst_p = worst_of(std::vector<float>(h1.begin(), h1.begin() + n_out), ref, rmax);
            if (wr.bytes) {
                fast::matvec(wr, x, y, 1, scr, s);
                CK(cudaStreamSynchronize(s));
                worst_r = worst_of(d2h(y, n_out), ref, rmax);
            }
            cudaFree(draw); cudaFree(df);
        }
        long long tinv = 0, pq = 0, parts = 0, v16 = 0, unp = 0;
        for (int T = 1; T <= TMAX; ++T) {
            fast::matvec(wp, x, y, T, scr, s);
            fast::matvec_quantize(x, n_in, T, qb, s);
            fast::matvec_prequantized(wp, x, qb, yq, T, s);
            CK(cudaStreamSynchronize(s));
            const std::vector<float> hy = d2h(y, (size_t) T * n_out), hq = d2h(yq, (size_t) T * n_out);
            for (size_t i = 0; i < hy.size(); ++i) {
                tinv += memcmp(&hy[i], &h1[i], 4) != 0;
                pq += memcmp(&hy[i], &hq[i], 4) != 0;
            }
            if (T <= 4 && n_out % 8 == 0) {   // parts: rows [0, n/2) and [n/2, n)
                const int64_t na = n_out / 2;
                float *ya, *yb2;
                CK(cudaMalloc(&ya, (size_t) T * n_out * 4)); CK(cudaMalloc(&yb2, (size_t) T * n_out * 4));
                const fast::mv_out po[2] = {{ya, na}, {ya + (size_t) T * na, n_out - na}};
                fast::matvec_prequantized_parts(wp, x, qb, po, 2, T, s);
                fast::packed_weight va = wp, vb = wp;
                va.n_out = na; va.bytes = fast::row_stride(wp) * na;
                vb.n_out = n_out - na; vb.bytes = fast::row_stride(wp) * (n_out - na);
                vb.data = (char *) wp.data + (size_t) na * fast::row_stride(wp);
                fast::matvec_prequantized(va, x, qb, yb2, T, s);
                fast::matvec_prequantized(vb, x, qb, yb2 + (size_t) T * na, T, s);
                CK(cudaStreamSynchronize(s));
                const std::vector<float> a = d2h(ya, (size_t) T * n_out), b = d2h(yb2, (size_t) T * n_out);
                for (size_t i = 0; i < a.size(); ++i) parts += memcmp(&a[i], &b[i], 4) != 0;
                cudaFree(ya); cudaFree(yb2);
            }
            if ((T == 1 || T == 3) && w16.bytes && ps != rb) {
                fast::matvec(w16, x, yq, T, scr, s);
                CK(cudaStreamSynchronize(s));
                const std::vector<float> h16 = d2h(yq, (size_t) T * n_out);
                for (size_t i = 0; i < h16.size(); ++i) v16 += memcmp(&hy[i], &h16[i], 4) != 0;
            }
        }
        // activations past the a16 amax clamp but inside the bound of matvec_dec_q6h.cuh (|x| = 1.2e5 in one group of every
        // column): every output finite, column j of the T = 3 call = the T = 1 call
        long long big_bad = 0;
        {
            std::vector<float> hb = hx;
            for (int j = 0; j < TMAX; ++j) for (int i = 0; i < 8; ++i) hb[(size_t) j * n_in + 32 + i] = (i & 1) ? -1.2e5f : 1.2e5f;
            CK(cudaMemcpy(x, hb.data(), hb.size() * 4, cudaMemcpyHostToDevice));
            for (int j = 0; j < TMAX; ++j) fast::matvec(wp, x + (size_t) j * n_in, y1 + (size_t) j * n_out, 1, scr, s);
            CK(cudaStreamSynchronize(s));
            const std::vector<float> b1 = d2h(y1, (size_t) TMAX * n_out);
            for (float v : b1) big_bad += !std::isfinite(v);
            fast::matvec(wp, x, y, 3, scr, s);
            CK(cudaStreamSynchronize(s));
            const std::vector<float> b3 = d2h(y, (size_t) 3 * n_out);
            for (size_t i = 0; i < b3.size(); ++i) big_bad += memcmp(&b3[i], &b1[i], 4) != 0;
        }
        if (wp.layout != fast::LAYOUT_RAW) {   // packed_to_f16 vs ggml to-f16 of the raw rows
            void * draw; half *a16, *b16, *c16;
            const size_t nv = (size_t) n_in * n_out;
            CK(cudaMalloc(&draw, wb.size())); CK(cudaMalloc(&a16, nv * 2)); CK(cudaMalloc(&b16, nv * 2)); CK(cudaMalloc(&c16, nv * 2));
            CK(cudaMemcpy(draw, wb.data(), wb.size(), cudaMemcpyHostToDevice));
            ggml_get_to_fp16_cuda(GGML_TYPE_Q6_K)(draw, a16, n_in * n_out, s);
            const int64_t r0 = 5;
            fast::packed_to_f16(wp, 0, r0, b16, s, fast::UNPACK_DIRECT);
            fast::packed_to_f16(wp, r0, n_out - r0, b16 + (size_t) r0 * n_in, s, fast::UNPACK_DIRECT);
            fast::packed_to_f16(wp, 0, r0, c16, s, fast::UNPACK_SMEM);
            fast::packed_to_f16(wp, r0, n_out - r0, c16 + (size_t) r0 * n_in, s, fast::UNPACK_SMEM);
            CK(cudaStreamSynchronize(s));
            std::vector<uint16_t> ha(nv), hb(nv), hc(nv);
            CK(cudaMemcpy(ha.data(), a16, nv * 2, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(hb.data(), b16, nv * 2, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(hc.data(), c16, nv * 2, cudaMemcpyDeviceToHost));
            for (size_t i = 0; i < nv; ++i) unp += (ha[i] != hb[i]) + (ha[i] != hc[i]);
            cudaFree(draw); cudaFree(a16); cudaFree(b16); cudaFree(c16);
        }
        // the default stride must pack to q6r3 when it is padded (and the padding must be off only by P100E_Q6_PAD=0)
        const bool layout_ok = ps == rb || (wp.layout == fast::LAYOUT_Q6R3 && wp.stride == ps && ps % 8 == 0);
        const bool ok = layout_ok && wp.bytes && worst_p <= 4e-3 && tinv == 0 && pq == 0 && parts == 0 && v16 == 0 && unp == 0 && big_bad == 0;
        printf("  q6pad %5lld x %5lld nb %2lld: stride %zu (raw %zu) layout %d (%s) | raw-stride layout %d (%s) | "
               "err %.2e (raw stride %.2e) | T-inv %lld, prequant %lld, parts %lld, pad16 %lld, unpack %lld, big-x %lld differ  %s\n",
               (long long) n_in, (long long) n_out, (long long) nb, fast::row_stride(wp), rb, wp.layout, fast::kernel_name(wp),
               wr.layout, fast::kernel_name(wr), worst_p, worst_r, tinv, pq, parts, v16, unp, big_bad, ok ? "PASS" : "FAIL");
        if (!ok) ++fails;
        cudaFree(dp); cudaFree(dr); cudaFree(d16p); cudaFree(x); cudaFree(y); cudaFree(y1); cudaFree(yq); cudaFree(scr); cudaFree(qb);
    }
}

// ---- every Q6_K shape and tuning must have a kernel at every T (P100E_MV_Q6_H16 != 0 has no wrcp fallback) ---------------
// Wide rows at large n_out take the 3-column hybrid (geo 22), which does not fit shared memory above about n_in 10900; a
// tuning geometry may also refuse a shape. matvec must still answer, with the same bits: column j = the T = 1 call on j.
static void q6h_shape_checks(cudaStream_t s) {
    struct shp { int64_t n_in, n_out; };
    struct tune { const char * name; int geo, kpg; };
    const tune tunes[] = {{"default", 0, 0}, {"geo 2 (not wide)", 2, 0}, {"kpg 8192 (>= ng)", 0, 8192}};
    std::mt19937 rng(6601);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    const int TMAX = 8;
    const fast::mv_tune_t saved = fast::mv_tune;
    for (const shp sh : {shp{17408, 256}, shp{12288, 256}, shp{11264, 128}, shp{8704, 5120}}) {
        const int64_t n_in = sh.n_in, n_out = sh.n_out, nb = n_in / 256;
        const size_t st = fast::pack_row_stride(GGML_TYPE_Q6_K, n_in, n_out);
        std::vector<uint8_t> wb(st * n_out, 0);
        for (int64_t r = 0; r < n_out; ++r)
            for (size_t b = 0; b < (size_t) nb; ++b) {
                uint8_t * blk = wb.data() + r * st + b * 210;
                for (int i = 0; i < 210; ++i) blk[i] = (uint8_t) rng();
                const half d = __float2half((0.0002f + 0.0006f * (rng() % 1000) / 1000.0f) * (rng() & 1 ? -1.0f : 1.0f));
                memcpy(blk + 208, &d, 2);
            }
        std::vector<float> hx((size_t) TMAX * n_in);
        for (auto & v : hx) v = nd(rng);
        void * dw; float * x; float * y; void * scr;
        CK(cudaMalloc(&dw, wb.size())); CK(cudaMalloc(&x, hx.size() * 4));
        CK(cudaMalloc(&y, (size_t) TMAX * n_out * 4)); CK(cudaMalloc(&scr, fast::matvec_scratch_bytes(n_in, TMAX)));
        CK(cudaMemcpy(dw, wb.data(), wb.size(), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice));
        const fast::packed_weight w = fast::pack_weight(GGML_TYPE_Q6_K, dw, n_in, n_out, s, st);
        CK(cudaStreamSynchronize(s));
        auto run = [&](int T, const float * xx) {
            CK(cudaMemsetAsync(y, 0xff, (size_t) T * n_out * 4, s));
            fast::matvec(w, xx, y, T, scr, s);
            CK(cudaStreamSynchronize(s)); CK(cudaGetLastError());
            return d2h(y, (size_t) T * n_out);
        };
        std::vector<std::vector<float>> y1(TMAX);
        for (int j = 0; j < TMAX; ++j) y1[j] = run(1, x + (size_t) j * n_in);
        for (const tune & t : tunes) {
            fast::mv_tune.wrm_geo = t.geo; fast::mv_tune.wrm_kpg = t.kpg;
            long long diff = 0;
            for (int T = 1; T <= TMAX; ++T) {
                const std::vector<float> yT = run(T, x);
                for (int j = 0; j < T; ++j)
                    for (int64_t i = 0; i < n_out; ++i) diff += memcmp(&yT[(size_t) j * n_out + i], &y1[j][i], 4) != 0;
            }
            fast::mv_tune = saved;
            const bool strict = fast::mv_tune.q6h != 0;   // mode 0 falls back to wrcp and legacy K splits: not checked for bits
            const bool ok = !strict || diff == 0;
            printf("  q6h %5lld x %5lld %-18s T = 1..8 vs T = 1: %lld differ  %s\n", (long long) n_in, (long long) n_out, t.name, diff,
                   strict ? (ok ? "PASS" : "FAIL") : "ran");
            if (!ok) ++fails;
        }
        cudaFree(dw); cudaFree(x); cudaFree(y); cudaFree(scr);
    }
}

int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc > 1 && !strcmp(argv[1], "--bench")) return bench(argc > 2 ? atoi(argv[2]) : 200);
    if (argc > 1 && !strcmp(argv[1], "--gen-bench")) {   // generic f32 GEMV T = 1 speed (GB/s of weight bytes) vs q4r
        cudaStream_t bs;
        CK(cudaStreamCreate(&bs));
        std::mt19937 rng(9);
        const uint32_t types[] = {GGML_TYPE_IQ4_NL, GGML_TYPE_MXFP4, GGML_TYPE_Q1_0, GGML_TYPE_Q4_0};
        const char * names[] = {"iq4_nl", "mxfp4", "q1_0", "q4_0"};
        const int64_t shp[][2] = {{2880, 2880}, {5120, 5120}, {2880, 4096}};
        for (auto & sh : shp)
            for (int ti = 0; ti < 4; ++ti) {
                std::vector<uint8_t> wb; std::vector<double> wd;
                if (!make_blocks(types[ti], sh[0], sh[1], rng, wb, wd)) continue;
                void * dw; float * x; float * y; void * scr;
                CK(cudaMalloc(&dw, wb.size())); CK(cudaMalloc(&x, sh[0] * 4)); CK(cudaMalloc(&y, sh[1] * 4));
                CK(cudaMalloc(&scr, fast::matvec_scratch_bytes(sh[0], 1)));
                CK(cudaMemcpy(dw, wb.data(), wb.size(), cudaMemcpyHostToDevice));
                CK(cudaMemset(x, 0, sh[0] * 4));
                const size_t rb = wb.size() / sh[1];
                cudaEvent_t e0, e1;
                CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
                float ms[2] = {0, 0};
                for (int k = 0; k < 2; ++k) {
                    if (k == 1 && types[ti] != GGML_TYPE_Q4_0) break;
                    const fast::packed_weight w = k ? fast::pack_weight(types[ti], dw, sh[0], sh[1], bs) : fast::packed_weight{};
                    for (int i = 0; i < 20; ++i) {
                        if (k) fast::matvec(w, x, y, 1, scr, bs); else fast::launch_generic(types[ti], (const char *) dw, x, y, 1, sh[0], sh[1], rb, bs);
                    }
                    CK(cudaEventRecord(e0, bs));
                    for (int i = 0; i < 200; ++i) {
                        if (k) fast::matvec(w, x, y, 1, scr, bs); else fast::launch_generic(types[ti], (const char *) dw, x, y, 1, sh[0], sh[1], rb, bs);
                    }
                    CK(cudaEventRecord(e1, bs));
                    CK(cudaEventSynchronize(e1));
                    CK(cudaEventElapsedTime(&ms[k], e0, e1));
                }
                printf("  %-6s %5lld x %5lld: generic %.1f us (%.0f GB/s)", names[ti], (long long) sh[0], (long long) sh[1],
                       ms[0] * 5, wb.size() / (ms[0] / 200 * 1e6));
                if (ms[1] > 0) printf(" | q4r a16 %.1f us (%.0f GB/s)", ms[1] * 5, wb.size() / (ms[1] / 200 * 1e6));
                printf("\n");
                cudaFree(dw); cudaFree(x); cudaFree(y); cudaFree(scr);
            }
        return 0;
    }
    if (argc > 1 && !strcmp(argv[1], "--q4r-bench")) {   // Q4_0 (q4r) matvec() us per call, T = 1, 2, 4, 8
        cudaStream_t bs;
        CK(cudaStreamCreate(&bs));
        std::mt19937 rng(5);
        const int64_t shp[][2] = {{5120, 5120}, {5120, 17408}, {17408, 5120}, {2560, 640}, {640, 2560}};
        for (auto & sh : shp) {
            std::vector<uint8_t> wb; std::vector<double> wd;
            make_blocks(GGML_TYPE_Q4_0, sh[0], sh[1], rng, wb, wd);
            void * dw; float * x; float * y; void * scr;
            CK(cudaMalloc(&dw, wb.size())); CK(cudaMalloc(&x, 8 * sh[0] * 4)); CK(cudaMalloc(&y, 8 * sh[1] * 4));
            CK(cudaMalloc(&scr, fast::matvec_scratch_bytes(sh[0], 8)));
            CK(cudaMemcpy(dw, wb.data(), wb.size(), cudaMemcpyHostToDevice));
            CK(cudaMemset(x, 0, 8 * sh[0] * 4));
            const fast::packed_weight w = fast::pack_weight(GGML_TYPE_Q4_0, dw, sh[0], sh[1], bs);
            printf("  q4_0 %5lld x %5lld:", (long long) sh[0], (long long) sh[1]);
            for (int T : {1, 2, 4, 8}) {
                cudaEvent_t e0, e1;
                CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
                for (int i = 0; i < 20; ++i) fast::matvec(w, x, y, T, scr, bs);
                CK(cudaEventRecord(e0, bs));
                for (int i = 0; i < 200; ++i) fast::matvec(w, x, y, T, scr, bs);
                CK(cudaEventRecord(e1, bs));
                CK(cudaEventSynchronize(e1));
                float ms = 0;
                CK(cudaEventElapsedTime(&ms, e0, e1));
                printf("  T=%d %.1f us", T, ms * 5);
            }
            printf("\n");
            cudaFree(dw); cudaFree(x); cudaFree(y); cudaFree(scr);
        }
        return 0;
    }
    if (argc > 1 && !strcmp(argv[1], "--fused-bench")) {   // matvec() T = 1 us per call (fused-quantize kernels), K-quants
        cudaStream_t bs;
        CK(cudaStreamCreate(&bs));
        std::mt19937 rng(3);
        const uint32_t types[] = {GGML_TYPE_Q4_K, GGML_TYPE_Q6_K, GGML_TYPE_Q3_K, GGML_TYPE_Q8_0};
        const char * names[] = {"q4_K", "q6_K", "q3_K", "q8_0"};
        const int64_t shp[][2] = {{5120, 5120}, {5120, 17408}, {17408, 5120}, {5120, 1024}};
        for (int ti = 0; ti < 4; ++ti)
            for (auto & sh : shp) {
                const size_t rb = (size_t) (sh[0] / get_type_info(types[ti]).blck) * get_type_info(types[ti]).size;
                std::vector<uint8_t> wb(rb * sh[1]);
                for (auto & b : wb) b = (uint8_t) (rng() & 0x3b);   // no f16 inf / NaN patterns in the scales
                void * dw; float * x; float * y; void * scr;
                CK(cudaMalloc(&dw, wb.size())); CK(cudaMalloc(&x, sh[0] * 4)); CK(cudaMalloc(&y, sh[1] * 4));
                CK(cudaMalloc(&scr, fast::matvec_scratch_bytes(sh[0], 1)));
                CK(cudaMemcpy(dw, wb.data(), wb.size(), cudaMemcpyHostToDevice));
                std::vector<float> hx(sh[0]);
                for (auto & v : hx) v = (rng() % 2001 - 1000) / 500.0f;
                CK(cudaMemcpy(x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice));
                const fast::packed_weight w = fast::pack_weight(types[ti], dw, sh[0], sh[1], bs);
                float best = 1e30f;
                for (int rep = 0; rep < 5; ++rep) {
                    cudaEvent_t e0, e1;
                    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
                    for (int i = 0; i < 20; ++i) fast::matvec(w, x, y, 1, scr, bs);
                    CK(cudaEventRecord(e0, bs));
                    for (int i = 0; i < 200; ++i) fast::matvec(w, x, y, 1, scr, bs);
                    CK(cudaEventRecord(e1, bs));
                    CK(cudaEventSynchronize(e1));
                    float ms = 0;
                    CK(cudaEventElapsedTime(&ms, e0, e1));
                    best = std::min(best, ms * 5);
                }
                printf("  %-5s %5lld x %5lld (%s): T=1 %.1f us (best of 5)\n", names[ti], (long long) sh[0], (long long) sh[1], fast::kernel_name(w), best);
                cudaFree(dw); cudaFree(x); cudaFree(y); cudaFree(scr);
            }
        return 0;
    }
    if (argc > 1 && !strcmp(argv[1], "--hashes")) {
        cudaStream_t hs;
        CK(cudaStreamCreate(&hs));
        regress(hs, true);
        q6pad_checks(hs, true);
        return 0;
    }
    // --both [reps]: the correctness run, then the bench (one remote run)
    const bool both = argc > 1 && !strcmp(argv[1], "--both");
    const int both_reps = both && argc > 2 ? atoi(argv[2]) : 200;
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    printf("device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    cudaStream_t s;
    CK(cudaStreamCreate(&s));
    std::mt19937 rng(4321);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    struct shape { int64_t n_in, n_out; };
    const shape shapes[] = {{5120, 48}, {5120, 1024}, {8704, 5120}, {3072, 24}, {17408, 64}, {10240, 36}, {2304, 8},
                            {640, 2560}, {2560, 640}};   // Flash-Next expert shapes (ffn_down / ffn_gate|up of one expert)
    const uint32_t types[] = {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_Q4_1, GGML_TYPE_Q2_K, GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_M,
                              GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S, GGML_TYPE_Q1_0, GGML_TYPE_Q2_0};
    // host copies of ggml's IQ3 grids / sign tables (the device tables of this TU)
    static uint32_t h_iq3xxs_grid[256], h_iq3s_grid[512]; static uint8_t h_ksigns[128];
    CK(cudaMemcpyFromSymbol(h_iq3xxs_grid, iq3xxs_grid, sizeof(h_iq3xxs_grid)));
    CK(cudaMemcpyFromSymbol(h_iq3s_grid, iq3s_grid, sizeof(h_iq3s_grid)));
    CK(cudaMemcpyFromSymbol(h_ksigns, ksigns_iq2xs, sizeof(h_ksigns)));
    const int TMAX = 8;
    for (uint32_t ty : types) {
        const char * tn = ty == GGML_TYPE_F32 ? "f32" : ty == GGML_TYPE_F16 ? "f16" : ty == GGML_TYPE_BF16 ? "bf16" : ty == GGML_TYPE_Q4_1 ? "q4_1" : ty == GGML_TYPE_Q2_K ? "q2_K" : ty == GGML_TYPE_IQ1_S ? "iq1_s" : ty == GGML_TYPE_IQ1_M ? "iq1_m" : ty == GGML_TYPE_IQ2_XXS ? "iq2_xxs" : ty == GGML_TYPE_IQ2_XS ? "iq2_xs" : ty == GGML_TYPE_IQ2_S ? "iq2_s" : ty == GGML_TYPE_IQ3_XXS ? "iq3_xxs" : ty == GGML_TYPE_Q1_0 ? "q1_0" : ty == GGML_TYPE_Q2_0 ? "q2_0" : "iq3_s";
        const bool is_iq2 = ty == GGML_TYPE_IQ2_XXS || ty == GGML_TYPE_IQ2_XS || ty == GGML_TYPE_IQ2_S;
        printf("== %s: has_fast_path %d\n", tn, (int) fast::has_fast_path(ty));
        if (!fast::has_fast_path(ty)) { ++fails; continue; }
        for (const shape sh : shapes) {
            const int64_t n_in = sh.n_in, n_out = sh.n_out;
            // weights: bytes + the dequantized values (fp64 reference)
            std::vector<double> wd((size_t) n_in * n_out);
            std::vector<uint8_t> wb;
            if (ty == GGML_TYPE_Q4_1) {
                const int64_t nb = n_in / 32;
                wb.resize((size_t) n_out * nb * 20);
                for (int64_t r = 0; r < n_out; ++r)
                    for (int64_t b = 0; b < nb; ++b) {
                        uint8_t * blk = wb.data() + ((size_t) r * nb + b) * 20;
                        const half d = __float2half(0.01f + 0.02f * (rng() % 1000) / 1000.0f);
                        const half m = __float2half(-0.1f * (rng() % 1000) / 1000.0f);
                        memcpy(blk, &d, 2); memcpy(blk + 2, &m, 2);
                        for (int i = 0; i < 16; ++i) blk[4 + i] = (uint8_t) rng();
                        for (int i = 0; i < 32; ++i) {
                            const int q = i < 16 ? blk[4 + i] & 0xF : blk[4 + i - 16] >> 4;
                            wd[(size_t) r * n_in + b * 32 + i] = (double) __half2float(d) * q + (double) __half2float(m);
                        }
                    }
            } else if (ty == GGML_TYPE_Q2_K) {
                if (n_in % 256) continue;
                const int64_t nb = n_in / 256;
                wb.resize((size_t) n_out * nb * 84);
                for (int64_t r = 0; r < n_out; ++r)
                    for (int64_t b = 0; b < nb; ++b) {
                        uint8_t * blk = wb.data() + ((size_t) r * nb + b) * 84;
                        for (int i = 0; i < 80; ++i) blk[i] = (uint8_t) rng();
                        const half d = __float2half(0.004f + 0.004f * (rng() % 1000) / 1000.0f);
                        const half dmin = __float2half(0.004f + 0.004f * (rng() % 1000) / 1000.0f);
                        memcpy(blk + 80, &d, 2); memcpy(blk + 82, &dmin, 2);
                        for (int i = 0; i < 256; ++i) {   // dequantize_row_q2_K
                            const int n = i >> 7, j = (i >> 5) & 3, l = i & 31;
                            const uint8_t sc = blk[(i >> 4) & 15];
                            const int q = (blk[16 + 32 * n + l] >> (2 * j)) & 3;
                            wd[(size_t) r * n_in + b * 256 + i] = (double) __half2float(d) * (sc & 15) * q - (double) __half2float(dmin) * (sc >> 4);
                        }
                    }
            } else if (ty == GGML_TYPE_IQ1_S || ty == GGML_TYPE_IQ1_M || ty == GGML_TYPE_Q1_0 || ty == GGML_TYPE_Q2_0) {
                // random GGUF blocks (valid fp16 scales); the reference is ggml's own device dequant of the same bytes
                const bool q1 = ty == GGML_TYPE_Q1_0, q2 = ty == GGML_TYPE_Q2_0;
                const int blkw = q1 ? 128 : q2 ? 64 : 256;
                if (n_in % blkw) continue;
                const bool m = ty == GGML_TYPE_IQ1_M;
                const size_t bs = (q1 || q2) ? 18 : m ? 56 : 50, nb = (size_t) (n_in / blkw);
                wb.resize((size_t) n_out * nb * bs);
                for (size_t b = 0; b < (size_t) n_out * nb; ++b) {
                    uint8_t * blk = wb.data() + b * bs;
                    for (size_t i = 0; i < bs; ++i) blk[i] = (uint8_t) rng();
                    const half d = __float2half(0.002f + 0.01f * (rng() % 1000) / 1000.0f);
                    uint16_t db; memcpy(&db, &d, 2);
                    if (!m) memcpy(blk, &d, 2);   // Q1_0 and IQ1_S: the fp16 scale leads the block
                    else {   // iq1_m: the fp16 super-scale is spread over the top nibbles of the four scale words
                        uint16_t sc[4]; memcpy(sc, blk + 48, 8);
                        for (int k = 0; k < 4; ++k) sc[k] = (uint16_t) ((sc[k] & 0x0fff) | (((db >> (4 * k)) & 0xf) << 12));
                        memcpy(blk + 48, sc, 8);
                    }
                }
                void * dr; float * df;
                CK(cudaMalloc(&dr, wb.size())); CK(cudaMalloc(&df, wd.size() * 4));
                CK(cudaMemcpy(dr, wb.data(), wb.size(), cudaMemcpyHostToDevice));
                ggml_get_to_fp32_cuda((ggml_type) ty)(dr, df, n_in * n_out, s);
                CK(cudaStreamSynchronize(s));
                std::vector<float> hf(wd.size());
                CK(cudaMemcpy(hf.data(), df, hf.size() * 4, cudaMemcpyDeviceToHost));
                for (size_t i = 0; i < wd.size(); ++i) wd[i] = hf[i];
                cudaFree(dr); cudaFree(df);
            } else if (is_iq2) {
                // random block bytes (every bit pattern is a valid IQ2 block) with a sane fp16 d; fp64 reference from ggml's
                // own f32 dequantization of the raw bytes
                if (n_in % 256) continue;
                const size_t bsz = get_type_info(ty).size;
                const int64_t nb = n_in / 256;
                wb.resize((size_t) n_out * nb * bsz);
                for (int64_t r = 0; r < n_out; ++r)
                    for (int64_t b = 0; b < nb; ++b) {
                        uint8_t * blk = wb.data() + ((size_t) r * nb + b) * bsz;
                        for (size_t i = 2; i < bsz; ++i) blk[i] = (uint8_t) rng();
                        const half d = __float2half(0.004f + 0.008f * (rng() % 1000) / 1000.0f);
                        memcpy(blk, &d, 2);
                    }
                void * draw; float * d32;
                CK(cudaMalloc(&draw, wb.size())); CK(cudaMalloc(&d32, wd.size() * 4));
                CK(cudaMemcpy(draw, wb.data(), wb.size(), cudaMemcpyHostToDevice));
                ggml_get_to_fp32_cuda((ggml_type) ty)(draw, d32, n_in * n_out, s);
                CK(cudaStreamSynchronize(s));
                std::vector<float> h32(wd.size());
                CK(cudaMemcpy(h32.data(), d32, h32.size() * 4, cudaMemcpyDeviceToHost));
                for (size_t i = 0; i < wd.size(); ++i) wd[i] = h32[i];
                cudaFree(draw); cudaFree(d32);
            } else if (ty == GGML_TYPE_IQ3_XXS || ty == GGML_TYPE_IQ3_S) {
                if (n_in % 256) continue;
                const bool S = ty == GGML_TYPE_IQ3_S;
                const int64_t nb = n_in / 256;
                const int bs = S ? 110 : 98;
                wb.resize((size_t) n_out * nb * bs);
                for (int64_t r = 0; r < n_out; ++r)
                    for (int64_t b = 0; b < nb; ++b) {
                        uint8_t * blk = wb.data() + ((size_t) r * nb + b) * bs;
                        for (int i = 2; i < bs; ++i) blk[i] = (uint8_t) rng();
                        const half dh = __float2half(0.004f + 0.004f * (rng() % 1000) / 1000.0f);
                        memcpy(blk, &dh, 2);
                        const float d0 = __half2float(dh);
                        const uint8_t * qs = blk + 2;
                        for (int ib = 0; ib < 8; ++ib) {          // dequantize_row_iq3_xxs / iq3_s (ggml float arithmetic)
                            float d; uint32_t aux = 0;
                            if (S) d = d0 * (1 + 2 * ((blk[106 + ib / 2] >> (4 * (ib % 2))) & 0xf));
                            else { memcpy(&aux, blk + 66 + 4 * ib, 4); d = d0 * (0.5f + (aux >> 28)) * 0.5f; }
                            for (int il = 0; il < 4; ++il) {
                                uint32_t i1, i2; uint8_t signs;
                                if (S) {
                                    i1 = qs[8 * ib + 2 * il] | ((blk[66 + ib] << (8 - 2 * il)) & 256);
                                    i2 = qs[8 * ib + 2 * il + 1] | ((blk[66 + ib] << (7 - 2 * il)) & 256);
                                    signs = blk[74 + 4 * ib + il];
                                } else {
                                    i1 = qs[8 * ib + 2 * il]; i2 = qs[8 * ib + 2 * il + 1];
                                    signs = h_ksigns[(aux >> (7 * il)) & 127];
                                }
                                const uint8_t * g1 = (const uint8_t *) ((S ? h_iq3s_grid : h_iq3xxs_grid) + i1);
                                const uint8_t * g2 = (const uint8_t *) ((S ? h_iq3s_grid : h_iq3xxs_grid) + i2);
                                for (int j = 0; j < 4; ++j) {
                                    wd[(size_t) r * n_in + b * 256 + 32 * ib + 8 * il + j]     = (double) (d * g1[j] * (signs & (1 << j) ? -1.f : 1.f));
                                    wd[(size_t) r * n_in + b * 256 + 32 * ib + 8 * il + j + 4] = (double) (d * g2[j] * (signs & (1 << (j + 4)) ? -1.f : 1.f));
                                }
                            }
                        }
                    }
            } else {
                const size_t es = ty == GGML_TYPE_F32 ? 4 : 2;
                wb.resize((size_t) n_in * n_out * es);
                for (size_t i = 0; i < wd.size(); ++i) {
                    const float v = 0.05f * nd(rng);
                    if (ty == GGML_TYPE_F32) { memcpy(wb.data() + 4 * i, &v, 4); wd[i] = v; }
                    else if (ty == GGML_TYPE_F16) { const half h = __float2half(v); memcpy(wb.data() + 2 * i, &h, 2); wd[i] = __half2float(h); }
                    else { const uint16_t b = f2bf(v); memcpy(wb.data() + 2 * i, &b, 2); wd[i] = bf2f(b); }
                }
            }
            void * dw;
            CK(cudaMalloc(&dw, wb.size()));
            CK(cudaMemcpy(dw, wb.data(), wb.size(), cudaMemcpyHostToDevice));
            const fast::packed_weight w = fast::pack_weight(ty, dw, n_in, n_out, s);
            CK(cudaStreamSynchronize(s));
            std::vector<float> hx((size_t) TMAX * n_in);
            for (auto & v : hx) v = nd(rng) * 2.0f;
            float *x, *y, *y1, *yg;
            CK(cudaMalloc(&x, hx.size() * 4));
            CK(cudaMemcpy(x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice));
            CK(cudaMalloc(&y, (size_t) TMAX * n_out * 4));
            CK(cudaMalloc(&y1, (size_t) TMAX * n_out * 4));
            CK(cudaMalloc(&yg, (size_t) TMAX * n_out * 4));
            void * scr;
            const size_t sb = fast::matvec_scratch_bytes(n_in, TMAX);
            CK(cudaMalloc(&scr, sb));
            // T = 1 per column (the invariance reference)
            for (int j = 0; j < TMAX; ++j) fast::matvec(w, x + (size_t) j * n_in, y1 + (size_t) j * n_out, 1, scr, s);
            CK(cudaStreamSynchronize(s));
            std::vector<float> h1((size_t) TMAX * n_out);
            CK(cudaMemcpy(h1.data(), y1, h1.size() * 4, cudaMemcpyDeviceToHost));
            // fp64 reference
            std::vector<double> ref((size_t) TMAX * n_out);
            double rmax = 0;
            for (int j = 0; j < TMAX; ++j)
                for (int64_t r = 0; r < n_out; ++r) {
                    double a = 0;
                    const double * wr = wd.data() + (size_t) r * n_in;
                    const float * xr = hx.data() + (size_t) j * n_in;
                    for (int64_t k = 0; k < n_in; ++k) a += wr[k] * xr[k];
                    ref[(size_t) j * n_out + r] = a;
                    rmax = std::max(rmax, std::fabs(a));
                }
            double worst = 0;
            long long tinv = 0;
            for (int T = 1; T <= TMAX; ++T) {
                CK(cudaMemsetAsync(y, 0xff, (size_t) T * n_out * 4, s));
                fast::matvec(w, x, y, T, scr, s);
                CK(cudaStreamSynchronize(s));
                std::vector<float> h((size_t) T * n_out);
                CK(cudaMemcpy(h.data(), y, h.size() * 4, cudaMemcpyDeviceToHost));
                for (size_t i = 0; i < h.size(); ++i) {
                    worst = std::max(worst, std::isfinite(h[i]) ? std::fabs(h[i] - ref[i]) / rmax : INFINITY);
                    tinv += memcmp(&h[i], &h1[i], 4) != 0;
                }
            }
            // prequantized (fused decode path) = matvec, T = 1 and 4
            long long pq = 0;
            for (int T : {1, 4}) {
                if (fast::matvec_uses_a16(w)) fast::matvec_quantize(x, n_in, T, scr, s);
                CK(cudaMemsetAsync(y, 0xff, (size_t) T * n_out * 4, s));
                fast::matvec_prequantized(w, x, scr, y, T, s);
                CK(cudaStreamSynchronize(s));
                std::vector<float> h((size_t) T * n_out);
                CK(cudaMemcpy(h.data(), y, h.size() * 4, cudaMemcpyDeviceToHost));
                for (size_t i = 0; i < h.size(); ++i) pq += memcmp(&h[i], &h1[i], 4) != 0;
            }
            const bool iq3 = ty == GGML_TYPE_IQ3_XXS || ty == GGML_TYPE_IQ3_S;
            const bool q1 = ty == GGML_TYPE_Q1_0 || ty == GGML_TYPE_Q2_0;
            const double lim = ty == GGML_TYPE_Q4_1 || ty == GGML_TYPE_Q2_K || ty == GGML_TYPE_IQ1_S || ty == GGML_TYPE_IQ1_M || is_iq2 || iq3 || q1 ? 4e-3 : 1e-5;
            // Q2_K: packed_to_f16 (both implementations, odd row chunk) = ggml's to-f16 of the raw bytes, bit for bit
            long long unp = 0;
            if (ty == GGML_TYPE_Q2_K || ty == GGML_TYPE_IQ1_S || ty == GGML_TYPE_IQ1_M || is_iq2 || iq3 || q1) {
                void * draw; half *d16a, *d16b, *d16c;
                CK(cudaMalloc(&draw, wb.size()));
                CK(cudaMemcpy(draw, wb.data(), wb.size(), cudaMemcpyHostToDevice));
                CK(cudaMalloc(&d16a, (size_t) n_in * n_out * 2)); CK(cudaMalloc(&d16b, (size_t) n_in * n_out * 2)); CK(cudaMalloc(&d16c, (size_t) n_in * n_out * 2));
                ggml_get_to_fp16_cuda((ggml_type) ty)(draw, d16a, n_in * n_out, s);
                const int64_t r0 = n_out > 3 ? 3 : 0;
                fast::packed_to_f16(w, 0, r0, d16b, s, fast::UNPACK_DIRECT);
                fast::packed_to_f16(w, r0, n_out - r0, d16b + (size_t) r0 * n_in, s, fast::UNPACK_DIRECT);
                fast::packed_to_f16(w, 0, n_out, d16c, s, fast::UNPACK_SMEM);
                CK(cudaStreamSynchronize(s));
                std::vector<uint16_t> ha((size_t) n_in * n_out), hb(ha.size()), hc(ha.size());
                CK(cudaMemcpy(ha.data(), d16a, ha.size() * 2, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(hb.data(), d16b, ha.size() * 2, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(hc.data(), d16c, ha.size() * 2, cudaMemcpyDeviceToHost));
                for (size_t i = 0; i < ha.size(); ++i) unp += (ha[i] != hb[i]) + (ha[i] != hc[i]);
                cudaFree(draw); cudaFree(d16a); cudaFree(d16b); cudaFree(d16c);
                printf("  [unpack %lld bit differences vs ggml to-f16] ", unp);
            }
            const bool ok = worst <= lim && tinv == 0 && pq == 0 && unp == 0;
            printf("  %5lld x %5lld  layout %d (%s): max|err|/max|ref| %.2e (limit %.0e), T-invariance %lld differ, "
                   "prequantized %lld differ  %s",
                   (long long) n_in, (long long) n_out, w.layout, fast::kernel_name(w), worst, lim, tinv, pq, ok ? "PASS" : "FAIL");
            if (ty == GGML_TYPE_Q4_1) {   // the generic f32 kernel it replaces (same weights): accuracy and speed
                fast::launch_generic(ty, (const char *) dw, x, yg, 1, n_in, n_out, (size_t) n_in / 32 * 20, s);
                CK(cudaStreamSynchronize(s));
                std::vector<float> hg(n_out);
                CK(cudaMemcpy(hg.data(), yg, hg.size() * 4, cudaMemcpyDeviceToHost));
                double eg = 0;
                for (int64_t r = 0; r < n_out; ++r) eg = std::max(eg, std::fabs(hg[r] - ref[r]) / rmax);
                cudaEvent_t e0, e1;
                CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
                float ms[2] = {0, 0};
                for (int k = 0; k < 2; ++k) {
                    CK(cudaEventRecord(e0, s));
                    for (int i = 0; i < 50; ++i) {
                        if (k == 0) fast::launch_generic(ty, (const char *) dw, x, yg, 1, n_in, n_out, (size_t) n_in / 32 * 20, s);
                        else fast::matvec(w, x, y, 1, scr, s);
                    }
                    CK(cudaEventRecord(e1, s));
                    CK(cudaEventSynchronize(e1));
                    CK(cudaEventElapsedTime(&ms[k], e0, e1));
                }
                printf(" | generic f32: err %.2e, T=1 %.1f us vs a16 %.1f us (%.0f GB/s)", eg, ms[0] * 20, ms[1] * 20,
                       wb.size() / (ms[1] / 50 * 1e6));
            } else {
                cudaEvent_t e0, e1;
                CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
                float ms = 0;
                CK(cudaEventRecord(e0, s));
                for (int i = 0; i < 50; ++i) fast::matvec(w, x, y, 1, scr, s);
                CK(cudaEventRecord(e1, s));
                CK(cudaEventSynchronize(e1));
                CK(cudaEventElapsedTime(&ms, e0, e1));
                printf(" | T=1 %.1f us (%.0f GB/s)", ms * 20, wb.size() / (ms / 50 * 1e6));
            }
            printf("\n");
            if (!ok) ++fails;
            cudaFree(dw); cudaFree(x); cudaFree(y); cudaFree(y1); cudaFree(yg); cudaFree(scr);
        }
    }
    printf("== regressions\n");
    regress(s, false);
    printf("== padded Q6_K rows (P100E_Q6_PAD)\n");
    q6pad_checks(s, false);
    printf("== Q6_K shapes and tunings at every T (P100E_MV_Q6_H16)\n");
    q6h_shape_checks(s);
    printf("%s (%d failing checks)\n", fails ? "FAILED" : "ALL PASSED", fails);
    if (both && !fails) return bench(both_reps);
    return fails ? 2 : 0;
}
