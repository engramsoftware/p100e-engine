// p100e-fast-gdn-verify-test: the MTP verify's batched delta net against the per-token path it
// replaces. Reference = the old keep_states loop of tp_model::decode_layers: T sequential T = 1 gdn_layer calls, D2D
// copies of the ssm / conv state after every token but the last into the rollback snapshots, then one
// matvec_quantize of the T output rows. New = ONE T-token gdn_layer whose kernels write the snapshots themselves
// (gdn_prep: conv state after token t, gdn_scan: ssm state after token t, straight from the registers).
// Every check is BIT-EXACT (memcmp): out [T][Hv*D], the a16 activation (scan epilogue), decay / beta (written in place
// over alpha / beta), the final conv + ssm state, all T-1 conv + ssm snapshots; plus rollback: snapshot slot n_keep-1
// (what tp_model::rollback restores) equals the state of a fresh sequential run over the first n_keep tokens only.
// Many random states / inputs (scales varied per trial) for T = 2, 3, 4 on the TP=2 shard shape (Hk 8, Hv 24) and the
// full shape (Hk 16, Hv 48); also T = 5..8 (conv snapshots from the y >= 1 prep blocks) and scan rows 64/32/16.
// Then GPU time per verify-layer call of both forms (CUDA graph replay, no host launch overhead).
// Strided inputs (the T > 1 fused step reads the in-projections in place from the concatenated launches' [T][n_tot]
// rows): gdn_layer with qkv|gate and beta|alpha row strides, ew_quant with a gate|up stride, attn_prep with q and k|v
// strides, each bit-exact against the dense call (T = 1..4).
#include "fast/ssm.cuh"
#include "fast/matvec.cuh"
#include "fast/fused.cuh"
#include "fast/attn.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(2); } } while (0)

namespace F = p100e::fast;

static int g_fail = 0, g_checks = 0;
static uint32_t rs = 0x9e3779b9u;
static uint32_t urand() { rs ^= rs << 13; rs ^= rs >> 17; rs ^= rs << 5; return rs; }
static float frand() { return (urand() >> 8) / 8388608.0f - 1.0f; }   // [-1, 1)
static cudaStream_t g_s;
static bool g_verbose = false;

static float * dev_from(const std::vector<float> & h) {
    float * d; CK(cudaMalloc(&d, h.size() * 4)); CK(cudaMemcpy(d, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaDeviceSynchronize());   // pageable H2D may return before the DMA lands; the kernels run on a non-blocking stream
    return d;
}
static std::vector<float> host_rand(size_t n, float scale, float off = 0.0f) {
    std::vector<float> h(n);
    for (auto & v : h) v = (frand() + frand() + frand()) * scale + off;
    for (size_t i = urand() % 97; i < n; i += 97) h[i] *= 8.0f;   // a few outliers, like real activations
    return h;
}
static void * dev_alloc(size_t bytes) { void * d; CK(cudaMalloc(&d, bytes)); CK(cudaMemset(d, 0, bytes)); CK(cudaDeviceSynchronize()); return d; }
static void copy_dev(void * dst, const void * src, size_t bytes) {
    CK(cudaDeviceSynchronize()); CK(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice)); CK(cudaDeviceSynchronize());
}
static std::vector<uint8_t> bytes_of(const void * d, size_t n) {
    std::vector<uint8_t> h(n); CK(cudaMemcpy(h.data(), d, n, cudaMemcpyDeviceToHost)); return h;
}
static bool same(const char * name, const void * a, const void * b, size_t bytes) {
    CK(cudaDeviceSynchronize());
    auto ha = bytes_of(a, bytes), hb = bytes_of(b, bytes);
    size_t diff = 0, first = (size_t) -1;
    for (size_t i = 0; i < bytes; ++i) if (ha[i] != hb[i]) { ++diff; if (first == (size_t) -1) first = i; }
    ++g_checks;
    if (diff) { ++g_fail; printf("    %-52s %9zu bytes  DIFF %zu bytes (first at %zu)  FAIL\n", name, bytes, diff, first); }
    else if (g_verbose) printf("    %-52s %9zu bytes  identical\n", name, bytes);
    return diff == 0;
}

// per-call GPU time: reps calls captured into a CUDA graph, replayed (no host launch overhead in the number)
static float time_us(const std::function<void()> & f, int reps = 100) {
    for (int i = 0; i < 3; ++i) f();
    CK(cudaStreamSynchronize(g_s));
    cudaGraph_t g; cudaGraphExec_t ge;
    CK(cudaStreamBeginCapture(g_s, cudaStreamCaptureModeThreadLocal));
    for (int i = 0; i < reps; ++i) f();
    CK(cudaStreamEndCapture(g_s, &g)); CK(cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0));
    CK(cudaGraphLaunch(ge, g_s)); CK(cudaStreamSynchronize(g_s));
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a, g_s);
    CK(cudaGraphLaunch(ge, g_s));
    cudaEventRecord(b, g_s); CK(cudaEventSynchronize(b));
    float ms; cudaEventElapsedTime(&ms, a, b);
    cudaGraphExecDestroy(ge); cudaGraphDestroy(g);
    cudaEventDestroy(a); cudaEventDestroy(b);
    return ms * 1000.0f / reps;
}

constexpr int D = 128, TMAX = 8;
constexpr float EPS = 1e-6f;

// one shape's buffers; side 0 = reference (sequential), side 1 = batched
struct Case {
    int Hk, Hv, C;
    size_t csz, ssz, qb;
    float *x, *al, *be, *z, *cw, *dtb, *ssa, *nw, *cs0, *st0;    // inputs (per trial)
    float *cs[2], *st[2], *a[2], *b[2], *y[2], *out[2], *csn[2], *ssn[2];
    void * q[2];
    F::GdnWeights gw;
    Case(int hk, int hv) : Hk(hk), Hv(hv), C(2 * hk * D + hv * D) {
        csz = (size_t) C * 3; ssz = (size_t) Hv * D * D;
        qb = F::matvec_scratch_bytes((int64_t) Hv * D, TMAX);
        x = (float *) dev_alloc((size_t) TMAX * C * 4); al = (float *) dev_alloc((size_t) TMAX * Hv * 4);
        be = (float *) dev_alloc((size_t) TMAX * Hv * 4); z = (float *) dev_alloc((size_t) TMAX * Hv * D * 4);
        cw = (float *) dev_alloc(csz / 3 * 4 * 4); dtb = (float *) dev_alloc(Hv * 4); ssa = (float *) dev_alloc(Hv * 4);
        nw = (float *) dev_alloc(D * 4); cs0 = (float *) dev_alloc(csz * 4); st0 = (float *) dev_alloc(ssz * 4);
        for (int i = 0; i < 2; ++i) {
            cs[i] = (float *) dev_alloc(csz * 4); st[i] = (float *) dev_alloc(ssz * 4);
            a[i] = (float *) dev_alloc((size_t) TMAX * Hv * 4); b[i] = (float *) dev_alloc((size_t) TMAX * Hv * 4);
            y[i] = (float *) dev_alloc((size_t) TMAX * C * 4); out[i] = (float *) dev_alloc((size_t) TMAX * Hv * D * 4);
            csn[i] = (float *) dev_alloc((TMAX - 1) * csz * 4); ssn[i] = (float *) dev_alloc((TMAX - 1) * ssz * 4);
            q[i] = dev_alloc(qb);
        }
        gw = F::GdnWeights{cw, dtb, ssa, nw};
    }
    void upload(float * d, const std::vector<float> & h) {
        CK(cudaMemcpy(d, h.data(), h.size() * 4, cudaMemcpyHostToDevice)); CK(cudaDeviceSynchronize());
    }
    // new random inputs and initial state; scales vary per trial so the state / activations cover many exponent ranges
    void randomize(int trial) {
        const float xs = 0.25f + 0.5f * (trial % 5), ss = trial % 3 == 0 ? 0.002f : trial % 3 == 1 ? 0.05f : 0.6f;
        upload(x, host_rand((size_t) TMAX * C, xs));
        auto ha = host_rand((size_t) TMAX * Hv, 2.0f);
        if (trial % 4 == 3) for (size_t i = 0; i < ha.size(); i += 5) ha[i] = 25.0f + (float) (i % 7);   // softplus x > 20 branch
        upload(al, ha);
        upload(be, host_rand((size_t) TMAX * Hv, 2.0f));
        upload(z, host_rand((size_t) TMAX * Hv * D, 1.5f));
        upload(cw, host_rand((size_t) C * 4, 0.3f));
        upload(dtb, host_rand(Hv, 0.5f));
        auto hs = host_rand(Hv, 1.0f);
        for (auto & v : hs) v = -std::fabs(v) - 0.05f;   // ssm_a = -exp(A_log) < 0
        upload(ssa, hs);
        upload(nw, host_rand(D, 0.2f, 1.0f));
        upload(cs0, host_rand(csz, xs));
        upload(st0, host_rand(ssz, ss));
    }
    void reset_side(int i, int T) {
        copy_dev(cs[i], cs0, csz * 4); copy_dev(st[i], st0, ssz * 4);
        copy_dev(a[i], al, (size_t) T * Hv * 4); copy_dev(b[i], be, (size_t) T * Hv * 4);
        CK(cudaMemset(out[i], 0, (size_t) TMAX * Hv * D * 4));
        CK(cudaMemset(csn[i], 0xff, (TMAX - 1) * csz * 4)); CK(cudaMemset(ssn[i], 0xff, (TMAX - 1) * ssz * 4));
        CK(cudaMemset(q[i], 0, qb)); CK(cudaDeviceSynchronize());
    }
    // the old decode_layers keep_states loop (rows: gdn_scan rows per block; 0 = auto, i.e. 128 at T = 1)
    void run_seq(int i, int T, bool quant, int rows = 0) {
        for (int t = 0; t < T; ++t) {
            if (rows == 0) {
                F::gdn_layer(x + (size_t) t * C, a[i] + (size_t) t * Hv, b[i] + (size_t) t * Hv, z + (size_t) t * Hv * D, gw,
                             cs[i], st[i], y[i], a[i] + (size_t) t * Hv, b[i] + (size_t) t * Hv, out[i] + (size_t) t * Hv * D,
                             1, Hk, Hv, D, EPS, g_s);
            } else {
                F::gdn_prep(x + (size_t) t * C, cs[i], gw, a[i] + (size_t) t * Hv, b[i] + (size_t) t * Hv, y[i],
                            a[i] + (size_t) t * Hv, b[i] + (size_t) t * Hv, 1, Hk, Hv, D, EPS, g_s);
                F::gdn_scan(y[i], a[i] + (size_t) t * Hv, b[i] + (size_t) t * Hv, z + (size_t) t * Hv * D, nw, st[i],
                            out[i] + (size_t) t * Hv * D, 1, Hk, Hv, D, EPS, g_s, rows);
            }
            if (t < T - 1) {
                CK(cudaMemcpyAsync(ssn[i] + t * ssz, st[i], ssz * 4, cudaMemcpyDeviceToDevice, g_s));
                CK(cudaMemcpyAsync(csn[i] + t * csz, cs[i], csz * 4, cudaMemcpyDeviceToDevice, g_s));
            }
        }
        if (quant) F::matvec_quantize(out[i], (int64_t) Hv * D, T, q[i], g_s);
    }
    void run_batch(int i, int T, bool quant, int rows = 0) {
        if (rows == 0) {
            F::gdn_layer(x, a[i], b[i], z, gw, cs[i], st[i], y[i], a[i], b[i], out[i], T, Hk, Hv, D, EPS, g_s,
                         quant ? q[i] : nullptr, csn[i], ssn[i]);
        } else {
            F::gdn_prep(x, cs[i], gw, a[i], b[i], y[i], a[i], b[i], T, Hk, Hv, D, EPS, g_s, csn[i]);
            F::gdn_scan(y[i], a[i], b[i], z, nw, st[i], out[i], T, Hk, Hv, D, EPS, g_s, rows, nullptr, ssn[i]);
        }
    }
    // full comparison of both sides after a T-token step; returns true when everything is identical
    bool compare(const char * tag, int T, bool quant) {
        char nm[128];
        bool ok = true;
        auto chk = [&](const char * what, const void * p0, const void * p1, size_t bytes) {
            snprintf(nm, sizeof nm, "%s %s", tag, what);
            ok = same(nm, p0, p1, bytes) && ok;
        };
        chk("out", out[0], out[1], (size_t) T * Hv * D * 4);
        if (quant) chk("a16 activation", q[0], q[1], F::matvec_scratch_bytes((int64_t) Hv * D, T));
        chk("decay (over alpha)", a[0], a[1], (size_t) T * Hv * 4);
        chk("beta (over beta)", b[0], b[1], (size_t) T * Hv * 4);
        chk("conv state (final)", cs[0], cs[1], csz * 4);
        chk("ssm state (final)", st[0], st[1], ssz * 4);
        chk("conv snapshots", csn[0], csn[1], (T - 1) * csz * 4);
        chk("ssm snapshots", ssn[0], ssn[1], (T - 1) * ssz * 4);
        return ok;
    }
};

static void test_shape(const char * name, int Hk, int Hv, int trials) {
    printf("=== %s: Hk=%d Hv=%d D=%d C=%d, %d random trials per T ===\n", name, Hk, Hv, D, 2 * Hk * D + Hv * D, trials);
    Case c(Hk, Hv);
    for (int T = 2; T <= 4; ++T) {
        const int f0 = g_fail, c0 = g_checks;
        for (int tr = 0; tr < trials; ++tr) {
            c.randomize(tr + 100 * T);
            const bool quant = tr % 2 == 0;   // fused path (scan epilogue quantize) / per-op path (no qbuf)
            char tag[64];
            snprintf(tag, sizeof tag, "T=%d trial %d%s:", T, tr, quant ? " (qbuf)" : "");
            c.reset_side(0, T); c.reset_side(1, T);
            c.run_seq(0, T, quant);
            c.run_batch(1, T, quant);
            CK(cudaGetLastError());
            c.compare(tag, T, quant);
            // rollback: slot n_keep-1 of the batched snapshots == a fresh sequential run of the first n_keep tokens;
            // then one more decode token from the restored state on both sides
            for (int nk = 1; nk < T; ++nk) {
                c.reset_side(0, nk);
                c.run_seq(0, nk, false);
                char nm[96];
                snprintf(nm, sizeof nm, "%s rollback n_keep=%d conv", tag, nk);
                same(nm, c.cs[0], c.csn[1] + (size_t) (nk - 1) * c.csz, c.csz * 4);
                snprintf(nm, sizeof nm, "%s rollback n_keep=%d ssm", tag, nk);
                same(nm, c.st[0], c.ssn[1] + (size_t) (nk - 1) * c.ssz, c.ssz * 4);
            }
        }
        printf("  T=%d: %d checks (out, a16, decay, beta, final states, %d snapshots each, rollback slots), %d failures\n",
               T, g_checks - c0, T - 1, g_fail - f0);
    }
    {   // T = 5..8 (gdn_prep's y >= 1 blocks write the later conv snapshots), rows 128 explicitly (auto picks 32 > 4)
        const int f0 = g_fail, c0 = g_checks;
        for (int T = 5; T <= TMAX; ++T) {
            c.randomize(7 * T);
            char tag[64];
            snprintf(tag, sizeof tag, "T=%d rows 128:", T);
            c.reset_side(0, T); c.reset_side(1, T);
            c.run_seq(0, T, false, 128);
            c.run_batch(1, T, false, 128);
            c.compare(tag, T, false);
        }
        printf("  T=5..8 (scan rows 128): %d checks, %d failures\n", g_checks - c0, g_fail - f0);
    }
    {   // the other scan block shapes (separate norm kernel) with snapshots, T = 3, vs the sequential run at the same rows
        const int f0 = g_fail, c0 = g_checks;
        for (int rows : {64, 32, 16}) {
            c.randomize(rows);
            char tag[64];
            snprintf(tag, sizeof tag, "T=3 rows %d:", rows);
            c.reset_side(0, 3); c.reset_side(1, 3);
            c.run_seq(0, 3, false, rows);
            c.run_batch(1, 3, false, rows);
            c.compare(tag, 3, false);
        }
        printf("  T=3 scan rows 64/32/16: %d checks, %d failures\n", g_checks - c0, g_fail - f0);
    }
    // time per verify-layer call: old loop (T x (prep + scan) + 2(T-1) D2D snapshot copies + quantize) vs one call
    for (int T = 2; T <= 4; ++T) {
        c.randomize(1);
        c.reset_side(0, T); c.reset_side(1, T);
        const float t_old = time_us([&] { c.run_seq(0, T, true); });
        const float t_new = time_us([&] { c.run_batch(1, T, true); });
        const float t_one = time_us([&] { c.run_seq(0, 1, true); });
        printf("  time T=%d: per-token loop %.2f us (%d ops)  ->  batched %.2f us (2 kernels)   [T=1 decode call %.2f us]\n",
               T, t_old, 2 * T + 2 * (T - 1) + 1, t_new, t_one);
    }
}

// ---- strided inputs: the T > 1 fused step reads the projections in place from concatenated launches ----
// [T][n_tot] buffer holding rows of `src` ([T][n] dense) at column offset `col`
static void scatter_rows(float * cat, int ld, int col, const float * src, int n, int T) {
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy2D(cat + col, (size_t) ld * 4, src, (size_t) n * 4, (size_t) n * 4, T, cudaMemcpyDeviceToDevice));
    CK(cudaDeviceSynchronize());
}

// gdn_layer with x|z in one [T][C + Hv*D] buffer and beta|alpha in one [T][2*Hv] buffer (the qkv|gate and beta|alpha
// launches of tp_model's fused step) vs the dense call, batched with snapshots + quantize epilogue
static void test_gdn_strided(int Hk, int Hv, int trials) {
    printf("=== gdn_layer strided inputs (qkv|gate, beta|alpha concatenated rows) Hk=%d Hv=%d ===\n", Hk, Hv);
    Case c(Hk, Hv);
    const int ld1 = c.C + Hv * D, ld2 = 2 * Hv;
    float * cat1 = (float *) dev_alloc((size_t) TMAX * ld1 * 4), * cat2 = (float *) dev_alloc((size_t) TMAX * ld2 * 4);
    const int f0 = g_fail, c0 = g_checks;
    for (int T = 1; T <= 4; ++T)
        for (int tr = 0; tr < trials; ++tr) {
            c.randomize(1000 + 10 * T + tr);
            scatter_rows(cat1, ld1, 0, c.x, c.C, T);
            scatter_rows(cat1, ld1, c.C, c.z, Hv * D, T);
            scatter_rows(cat2, ld2, 0, c.be, Hv, T);
            scatter_rows(cat2, ld2, Hv, c.al, Hv, T);
            c.reset_side(0, T); c.reset_side(1, T);
            float * csn = T > 1 ? c.csn[0] : nullptr, * ssn = T > 1 ? c.ssn[0] : nullptr;
            F::gdn_layer(c.x, c.a[0], c.b[0], c.z, c.gw, c.cs[0], c.st[0], c.y[0], c.a[0], c.b[0], c.out[0], T, Hk, Hv, D, EPS,
                         g_s, c.q[0], csn, ssn);
            // strided: decay / beta to the (dense) side-1 a / b buffers, inputs from cat1 / cat2
            F::gdn_layer(cat1, cat2 + Hv, cat2, cat1 + c.C, c.gw, c.cs[1], c.st[1], c.y[1], c.a[1], c.b[1], c.out[1], T, Hk,
                         Hv, D, EPS, g_s, c.q[1], T > 1 ? c.csn[1] : nullptr, T > 1 ? c.ssn[1] : nullptr, ld1, ld2, ld2, ld1);
            CK(cudaGetLastError());
            char tag[64];
            snprintf(tag, sizeof tag, "strided T=%d trial %d:", T, tr);
            if (T > 1) c.compare(tag, T, true);
            else {
                char nm[96];
                snprintf(nm, sizeof nm, "%s out", tag); same(nm, c.out[0], c.out[1], (size_t) Hv * D * 4);
                snprintf(nm, sizeof nm, "%s ssm state", tag); same(nm, c.st[0], c.st[1], c.ssz * 4);
                snprintf(nm, sizeof nm, "%s conv state", tag); same(nm, c.cs[0], c.cs[1], c.csz * 4);
                snprintf(nm, sizeof nm, "%s a16", tag); same(nm, c.q[0], c.q[1], F::matvec_scratch_bytes((int64_t) Hv * D, 1));
            }
        }
    {   // rows 32 (separate norm kernel reads z with its stride), T = 6, no snapshots
        const int T = 6;
        c.randomize(77);
        scatter_rows(cat1, ld1, 0, c.x, c.C, T);
        scatter_rows(cat1, ld1, c.C, c.z, Hv * D, T);
        scatter_rows(cat2, ld2, 0, c.be, Hv, T);
        scatter_rows(cat2, ld2, Hv, c.al, Hv, T);
        c.reset_side(0, T); c.reset_side(1, T);
        F::gdn_prep(c.x, c.cs[0], c.gw, c.a[0], c.b[0], c.y[0], c.a[0], c.b[0], T, Hk, Hv, D, EPS, g_s);
        F::gdn_scan(c.y[0], c.a[0], c.b[0], c.z, c.nw, c.st[0], c.out[0], T, Hk, Hv, D, EPS, g_s, 32);
        F::gdn_prep(cat1, c.cs[1], c.gw, cat2 + Hv, cat2, c.y[1], c.a[1], c.b[1], T, Hk, Hv, D, EPS, g_s, nullptr, ld1, ld2, ld2);
        F::gdn_scan(c.y[1], c.a[1], c.b[1], cat1 + c.C, c.nw, c.st[1], c.out[1], T, Hk, Hv, D, EPS, g_s, 32, nullptr, nullptr, ld1);
        same("strided T=6 rows 32: out", c.out[0], c.out[1], (size_t) T * Hv * D * 4);
        same("strided T=6 rows 32: ssm state", c.st[0], c.st[1], c.ssz * 4);
    }
    printf("  %d checks, %d failures\n", g_checks - c0, g_fail - f0);
    cudaFree(cat1); cudaFree(cat2);
}

// ew_quant(SILU_MUL) reading gate / up rows from one [T][2n] buffer vs dense inputs
static void test_ew_strided(int n) {
    printf("=== ew_quant silu_mul strided gate|up (n=%d) ===\n", n);
    const int f0 = g_fail, c0 = g_checks;
    for (int T = 1; T <= 4; ++T) {
        float * a = dev_from(host_rand((size_t) T * n, 2.0f)), * b = dev_from(host_rand((size_t) T * n, 2.0f));
        float * cat = (float *) dev_alloc((size_t) T * 2 * n * 4);
        scatter_rows(cat, 2 * n, 0, a, n, T);
        scatter_rows(cat, 2 * n, n, b, n, T);
        float * y0 = (float *) dev_alloc((size_t) T * n * 4), * y1 = (float *) dev_alloc((size_t) T * n * 4);
        const size_t qb = F::matvec_scratch_bytes(n, T);
        void * q0 = dev_alloc(qb), * q1 = dev_alloc(qb);
        copy_dev(y0, a, (size_t) T * n * 4);
        const bool ok0 = F::ew_quant(F::EW_SILU_MUL, y0, b, y0, q0, T, n, g_s);           // the dense in-place call
        const bool ok1 = F::ew_quant(F::EW_SILU_MUL, cat, cat + n, y1, q1, T, n, g_s, 2 * n, 2 * n);
        if (!ok0 || !ok1) { printf("  ew_quant refused (T=%d)  FAIL\n", T); ++g_fail; }
        char nm[64];
        snprintf(nm, sizeof nm, "ew strided T=%d: y", T); same(nm, y0, y1, (size_t) T * n * 4);
        snprintf(nm, sizeof nm, "ew strided T=%d: a16", T); same(nm, q0, q1, qb);
        for (void * p : {(void *) a, (void *) b, (void *) cat, (void *) y0, (void *) y1, q0, q1}) cudaFree(p);
    }
    printf("  %d checks, %d failures\n", g_checks - c0, g_fail - f0);
}

// attn_prep reading q|gate from a padded-stride buffer and k / v from one [T][2*Hkv*D] buffer vs dense inputs
static void test_attn_prep_strided() {
    const int Hq = 12, Hkv = 2, AD = 256, n_rot = 64, n_ctx = 64;
    printf("=== attn_prep strided q (padded rows) and k|v (Hq=%d Hkv=%d D=%d) ===\n", Hq, Hkv, AD);
    const int f0 = g_fail, c0 = g_checks;
    for (int T = 1; T <= 4; ++T) {
        const int nq = Hq * 2 * AD, nk = Hkv * AD, ldq = nq + 64, ldkv = 2 * nk;
        float * qg = dev_from(host_rand((size_t) T * nq, 1.0f)), * k = dev_from(host_rand((size_t) T * nk, 1.0f));
        float * v = dev_from(host_rand((size_t) T * nk, 1.0f));
        float * qn = dev_from(host_rand(AD, 0.2f, 1.0f)), * kn = dev_from(host_rand(AD, 0.2f, 1.0f));
        float * qcat = (float *) dev_alloc((size_t) T * ldq * 4), * kvcat = (float *) dev_alloc((size_t) T * ldkv * 4);
        scatter_rows(qcat, ldq, 0, qg, nq, T);
        scatter_rows(kvcat, ldkv, 0, k, nk, T);
        scatter_rows(kvcat, ldkv, nk, v, nk, T);
        std::vector<int32_t> hp(T);
        for (int t = 0; t < T; ++t) hp[t] = 17 + t;
        int32_t * pos; CK(cudaMalloc(&pos, T * 4)); CK(cudaMemcpy(pos, hp.data(), T * 4, cudaMemcpyHostToDevice));
        F::kv_cache c[2] = {F::kv_alloc(F::KV_F16, n_ctx, Hkv, AD), F::kv_alloc(F::KV_F16, n_ctx, Hkv, AD)};
        const size_t cb = F::kv_row_bytes(F::KV_F16, Hkv, AD) * n_ctx;
        float * q[2], * g[2];
        for (int i = 0; i < 2; ++i) {
            CK(cudaMemset(c[i].k, 0, cb)); CK(cudaMemset(c[i].v, 0, cb));
            q[i] = (float *) dev_alloc((size_t) T * Hq * AD * 4); g[i] = (float *) dev_alloc((size_t) T * Hq * AD * 4);
        }
        CK(cudaDeviceSynchronize());
        const bool ok0 = F::attn_prep(qg, k, v, qn, kn, pos, q[0], g[0], c[0], -1, T, Hq, Hkv, AD, n_rot, 1e7f, EPS, g_s);
        const bool ok1 = F::attn_prep(qcat, kvcat, kvcat + nk, qn, kn, pos, q[1], g[1], c[1], -1, T, Hq, Hkv, AD, n_rot, 1e7f,
                                      EPS, g_s, ldq, ldkv, ldkv);
        if (!ok0 || !ok1) { printf("  attn_prep refused (T=%d)  FAIL\n", T); ++g_fail; }
        char nm[64];
        snprintf(nm, sizeof nm, "attn_prep strided T=%d: q", T); same(nm, q[0], q[1], (size_t) T * Hq * AD * 4);
        snprintf(nm, sizeof nm, "attn_prep strided T=%d: gate", T); same(nm, g[0], g[1], (size_t) T * Hq * AD * 4);
        snprintf(nm, sizeof nm, "attn_prep strided T=%d: K cache", T); same(nm, c[0].k, c[1].k, cb);
        snprintf(nm, sizeof nm, "attn_prep strided T=%d: V cache", T); same(nm, c[0].v, c[1].v, cb);
        for (int i = 0; i < 2; ++i) { F::kv_free(c[i]); cudaFree(q[i]); cudaFree(g[i]); }
        for (void * p : {(void *) qg, (void *) k, (void *) v, (void *) qn, (void *) kn, (void *) qcat, (void *) kvcat, (void *) pos}) cudaFree(p);
    }
    printf("  %d checks, %d failures\n", g_checks - c0, g_fail - f0);
}

int main(int argc, char ** argv) {
    g_verbose = argc > 1 && !strcmp(argv[1], "-v");
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    printf("device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    CK(cudaStreamCreateWithFlags(&g_s, cudaStreamNonBlocking));
    test_shape("TP=2 shard (Qwen3.8-27B per GPU)", 8, 24, 24);
    test_shape("full (TP=1)", 16, 48, 8);
    test_gdn_strided(8, 24, 6);
    test_ew_strided(8704);
    test_attn_prep_strided();
    printf("%s (%d checks, %d failures; all checks bit-exact)\n", g_fail ? "FAIL" : "PASS", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
