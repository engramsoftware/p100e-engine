// p100e-fast-sessions-test: the kernels of the multi-session engine against the single-sequence calls
// they generalize. Every check is BIT-EXACT (memcmp):
//  - paged KV cache (pages of 256 positions in a shared pool, random page order, a pool stride with other layers in
//    between) vs a contiguous cache with the same contents, f16 / q8_0 / q4_0:
//      kv_append (cache bytes), attn_decode T = 1..4 (host p0), attn_prefill, attn_fold, attn_prep (p0 = -1)
//  - attn_decode_rows: up to 8 rows from 3 sequences (each its own page table and position) vs the T = 1 attn_decode of
//    each row on its sequence's contiguous cache; attn_prep with rows of 3 sequences vs per-row calls
//  - gdn_layer_multi: several sequences (1..4 tokens each, states in a slot pool, snapshots from a shared snapshot pool)
//    vs one gdn_layer per sequence: out, a16 activation rows, decay / beta, final conv + ssm states, every snapshot
#include "fast/attn.cuh"
#include "fast/attn_fold.cuh"
#include "fast/fused.cuh"
#include "fast/ssm.cuh"
#include "fast/matvec.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(2); } } while (0)

namespace F = p100e::fast;

static int g_fail = 0, g_checks = 0;
static uint32_t rs = 0x2545F491u;
static uint32_t urand() { rs ^= rs << 13; rs ^= rs >> 17; rs ^= rs << 5; return rs; }
static float frand() { return (urand() >> 8) / 8388608.0f - 1.0f; }   // [-1, 1)
static cudaStream_t g_s;

static float * dev_from(const std::vector<float> & h) {
    float * d; CK(cudaMalloc(&d, h.size() * 4)); CK(cudaMemcpy(d, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaDeviceSynchronize());
    return d;
}
static std::vector<float> host_rand(size_t n, float scale, float off = 0.0f) {
    std::vector<float> h(n);
    for (auto & v : h) v = (frand() + frand() + frand()) * scale + off;
    for (size_t i = urand() % 97; i < n; i += 97) h[i] *= 8.0f;
    return h;
}
static float * dev_rand(size_t n, float scale, float off = 0.0f) { return dev_from(host_rand(n, scale, off)); }
static void * dev_alloc(size_t bytes) { void * d; CK(cudaMalloc(&d, bytes)); CK(cudaMemset(d, 0, bytes)); CK(cudaDeviceSynchronize()); return d; }
static std::vector<uint8_t> bytes_of(const void * d, size_t n) {
    CK(cudaDeviceSynchronize());
    std::vector<uint8_t> h(n); CK(cudaMemcpy(h.data(), d, n, cudaMemcpyDeviceToHost)); return h;
}
static void copy_dev(void * dst, const void * src, size_t bytes) {
    CK(cudaDeviceSynchronize()); CK(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice)); CK(cudaDeviceSynchronize());
}
static bool same_h(const char * name, const uint8_t * a, const uint8_t * b, size_t bytes, bool quiet = false) {
    size_t diff = 0, first = (size_t) -1;
    for (size_t i = 0; i < bytes; ++i) if (a[i] != b[i]) { ++diff; if (first == (size_t) -1) first = i; }
    ++g_checks;
    if (diff) { ++g_fail; printf("    %-60s %9zu bytes  DIFF %zu bytes (first at %zu)  FAIL\n", name, bytes, diff, first); }
    else if (!quiet) printf("    %-60s %9zu bytes  identical\n", name, bytes);
    return diff == 0;
}
static bool same(const char * name, const void * a, const void * b, size_t bytes, bool quiet = false) {
    auto ha = bytes_of(a, bytes), hb = bytes_of(b, bytes);
    return same_h(name, ha.data(), hb.data(), bytes, quiet);
}
static const char * tname(uint32_t t) { return t == F::KV_F16 ? "f16" : t == F::KV_Q8_0 ? "q8_0" : "q4_0"; }

// ======================================================================================================== attention
static void test_attention(uint32_t type) {
    printf("paged KV cache, %s\n", tname(type));
    // sequence 1 is long (several 128-key tiles per split stream: the page-table prefetch of attn_decode2)
    const int Hkv = 2, D = 256, Hq = 12, PSH = 8, PS = 1 << PSH, NP = 64, MAXP = 48, NLAY = 2;
    const float scale = 1.0f / 16.0f;
    const size_t rowb = F::kv_row_bytes(type, Hkv, D);
    const size_t pstride = (size_t) NLAY * 2 * PS * rowb;   // [page][layer][K|V][PS][row] as tp_model's pool
    char * pool = (char *) dev_alloc((size_t) NP * pstride);
    const int NS = 3, len[NS] = {700, 11000, 300};
    // random page assignment: a permutation of the pool
    std::vector<int> perm(NP);
    std::iota(perm.begin(), perm.end(), 0);
    for (int i = NP - 1; i > 0; --i) std::swap(perm[i], perm[urand() % (i + 1)]);
    std::vector<int32_t> tabs((size_t) NS * MAXP, -1);
    int used = 0;
    for (int s = 0; s < NS; ++s)
        for (int p = 0; p < (len[s] + PS - 1) / PS + 1; ++p) tabs[(size_t) s * MAXP + p] = perm[used++];
    int32_t * dtab; CK(cudaMalloc(&dtab, tabs.size() * 4));
    CK(cudaMemcpy(dtab, tabs.data(), tabs.size() * 4, cudaMemcpyHostToDevice));
    // layer 1 of the pool (a nonzero offset inside the page)
    auto paged = [&](int s) {
        F::kv_cache c; c.type = type; c.n_ctx = MAXP * PS; c.Hkv = Hkv; c.D = D;
        c.k = pool + (size_t) 1 * 2 * PS * rowb; c.v = pool + (size_t) 1 * 2 * PS * rowb + PS * rowb;
        c.pt = dtab + (size_t) s * MAXP; c.psh = PSH; c.pstride = pstride;
        return c;
    };
    std::vector<F::kv_cache> cc(NS);
    std::vector<float *> ks(NS), vs(NS);
    for (int s = 0; s < NS; ++s) {
        cc[s] = F::kv_alloc(type, MAXP * PS, Hkv, D);
        ks[s] = dev_rand((size_t) len[s] * Hkv * D, 1.0f);
        vs[s] = dev_rand((size_t) len[s] * Hkv * D, 1.0f);
        F::kv_cache pc = paged(s);
        for (int p = 0; p < len[s]; p += 64) {   // the same appends to both
            const int T = std::min(64, len[s] - p);
            F::kv_append(cc[s], ks[s] + (size_t) p * Hkv * D, vs[s] + (size_t) p * Hkv * D, p, T, g_s);
            F::kv_append(pc, ks[s] + (size_t) p * Hkv * D, vs[s] + (size_t) p * Hkv * D, p, T, g_s);
        }
    }
    {   // append bytes: position p of seq s, contiguous row vs its page slot
        bool ok = true;
        auto hpool = bytes_of(pool, (size_t) NP * pstride);
        for (int s = 0; s < NS; ++s) {
            auto hk = bytes_of(cc[s].k, (size_t) len[s] * rowb), hv = bytes_of(cc[s].v, (size_t) len[s] * rowb);
            for (int p = 0; p < len[s]; ++p) {
                const size_t o = (size_t) tabs[(size_t) s * MAXP + (p >> PSH)] * pstride + (size_t) 2 * PS * rowb + (size_t) (p & (PS - 1)) * rowb;
                ok = ok && !memcmp(hk.data() + (size_t) p * rowb, hpool.data() + o, rowb);
                ok = ok && !memcmp(hv.data() + (size_t) p * rowb, hpool.data() + o + PS * rowb, rowb);
            }
        }
        ++g_checks; if (!ok) ++g_fail;
        printf("    %-60s %s\n", "kv_append paged vs contiguous (every position, K and V)", ok ? "identical" : "FAIL");
    }
    float * q = dev_rand((size_t) 256 * Hq * D, 1.5f);
    float * o1 = (float *) dev_alloc((size_t) 256 * Hq * D * 4), * o2 = (float *) dev_alloc((size_t) 256 * Hq * D * 4);
    {   // decode, host p0, T = 1..4
        int bad = 0, n = 0;
        for (int s = 0; s < NS; ++s) {
            const F::kv_cache pc = paged(s);
            for (int T = 1; T <= 4; ++T)
                for (int p0 : {0, 1, 31, 32, 127, 128, 255, 256, 257, 511, 600, 4223, 4224, 5000, 8191, 10000, len[s] - T}) {
                    if (p0 < 0 || p0 + T > len[s]) continue;
                    F::attn_decode(q, cc[s], o1, T, p0, Hq, scale, g_s);
                    F::attn_decode(q, pc, o2, T, p0, Hq, scale, g_s);
                    auto a = bytes_of(o1, (size_t) T * Hq * D * 4), b = bytes_of(o2, (size_t) T * Hq * D * 4);
                    ++n; if (memcmp(a.data(), b.data(), a.size())) ++bad;
                }
        }
        ++g_checks; if (bad) ++g_fail;
        printf("    %-60s %4d cases %s\n", "attn_decode T=1..4 paged vs contiguous", n, bad ? "FAIL" : "identical");
    }
    {   // multi-row: rows of 3 sequences in one call vs T = 1 per row on the contiguous caches
        struct row { int s, p; };
        const std::vector<std::vector<row>> sets = {
            {{0, 699}},
            {{0, 699}, {1, 1499}, {2, 299}},
            {{1, 10999}, {0, 600}, {1, 9000}, {2, 250}, {1, 4300}},
            {{1, 1200}, {1, 1201}, {1, 1202}, {1, 1203}},
            {{0, 650}, {0, 651}, {1, 1496}, {1, 1497}, {1, 1498}, {2, 100}, {2, 101}, {2, 102}},
            {{2, 0}, {0, 0}, {1, 255}, {1, 256}, {0, 31}, {0, 32}},
        };
        int32_t * dpos; CK(cudaMalloc(&dpos, 16 * 4));
        int32_t * drt; CK(cudaMalloc(&drt, 16 * 4));
        int bad = 0, n = 0;
        for (const auto & set : sets) {
            const int T = (int) set.size();
            std::vector<int32_t> hp(T), hr(T);
            for (int t = 0; t < T; ++t) { hp[t] = set[t].p; hr[t] = set[t].s * MAXP; }
            CK(cudaMemcpy(dpos, hp.data(), T * 4, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(drt, hr.data(), T * 4, cudaMemcpyHostToDevice));
            F::kv_cache pc = paged(0);
            pc.pt = dtab; pc.rtab = drt;
            if (!F::attn_decode_rows(q, pc, o2, T, dpos, Hq, scale, g_s)) { printf("    attn_decode_rows refused  FAIL\n"); ++g_fail; continue; }
            auto b = bytes_of(o2, (size_t) T * Hq * D * 4);
            for (int t = 0; t < T; ++t) {
                F::attn_decode(q + (size_t) t * Hq * D, cc[set[t].s], o1, 1, set[t].p, Hq, scale, g_s);
                auto a = bytes_of(o1, (size_t) Hq * D * 4);
                ++n;
                if (memcmp(a.data(), b.data() + (size_t) t * Hq * D * 4, a.size())) {
                    if (++bad <= 6) printf("      row %d (seq %d pos %d) of a %d-row call differs\n", t, set[t].s, set[t].p, T);
                }
            }
        }
        ++g_checks; if (bad) ++g_fail;
        printf("    %-60s %4d rows  %s\n", "attn_decode_rows (3 sequences, 1..8 rows) vs T=1 per row", n, bad ? "FAIL" : "identical");
        cudaFree(dpos); cudaFree(drt);
    }
    {   // prefill attention: fp32 kernel and fold GEMM
        const int s = 1, T = 200, p0 = 1300;
        const F::kv_cache pc = paged(s);
        F::attn_prefill(q, cc[s], o1, T, p0, Hq, scale, g_s);
        F::attn_prefill(q, pc, o2, T, p0, Hq, scale, g_s);
        same("attn_prefill T=200 p0=1300 paged vs contiguous", o1, o2, (size_t) T * Hq * D * 4);
        const size_t scr_b = 256ull << 20;
        void * scr = dev_alloc(scr_b);
        const bool a1 = F::attn_fold(q, cc[s], o1, T, p0, Hq, scale, scr, scr_b, g_s);
        const bool a2 = F::attn_fold(q, pc, o2, T, p0, Hq, scale, scr, scr_b, g_s);
        if (!a1 || !a2) { printf("    attn_fold refused  FAIL\n"); ++g_fail; }
        same("attn_fold T=200 p0=1300 paged vs contiguous", o1, o2, (size_t) T * Hq * D * 4);
        // chunked fold (several key chunks and splits)
        F::attn_fold(q, cc[s], o1, 64, 1400, Hq, scale, scr, scr_b, g_s, 512, 2);
        F::attn_fold(q, pc, o2, 64, 1400, Hq, scale, scr, scr_b, g_s, 512, 2);
        same("attn_fold T=64 p0=1400 chunk 512 x 2 splits paged vs contiguous", o1, o2, (size_t) 64 * Hq * D * 4);
        cudaFree(scr);
    }
    {   // attn_prep: rows of 3 sequences in one call (p0 = -1, positions + row tables) vs one T = 1 call per row
        const int T = 5;
        const int rs_[T] = {0, 1, 1, 2, 0}, rp[T] = {700, 1500, 1501, 300, 701};
        std::vector<int32_t> hp(rp, rp + T), hr(T);
        for (int t = 0; t < T; ++t) hr[t] = rs_[t] * MAXP;
        int32_t * dpos; CK(cudaMalloc(&dpos, 16 * 4)); int32_t * drt; CK(cudaMalloc(&drt, 16 * 4));
        CK(cudaMemcpy(dpos, hp.data(), T * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(drt, hr.data(), T * 4, cudaMemcpyHostToDevice));
        float * qg = dev_rand((size_t) T * Hq * 2 * D, 1.0f), * kk = dev_rand((size_t) T * Hkv * D, 1.0f);
        float * vv = dev_rand((size_t) T * Hkv * D, 1.0f), * qn = dev_rand(D, 0.2f, 1.0f), * kn = dev_rand(D, 0.2f, 1.0f);
        float * qa = (float *) dev_alloc((size_t) T * Hq * D * 4), * ga = (float *) dev_alloc((size_t) T * Hq * D * 4);
        float * qb = (float *) dev_alloc((size_t) T * Hq * D * 4), * gb = (float *) dev_alloc((size_t) T * Hq * D * 4);
        F::kv_cache pc = paged(0);
        pc.pt = dtab; pc.rtab = drt;
        bool ok = F::attn_prep(qg, kk, vv, qn, kn, dpos, qb, gb, pc, -1, T, Hq, Hkv, D, 64, 1e7f, 1e-6f, g_s);
        for (int t = 0; t < T; ++t)
            ok = ok && F::attn_prep(qg + (size_t) t * Hq * 2 * D, kk + (size_t) t * Hkv * D, vv + (size_t) t * Hkv * D, qn, kn,
                                    dpos + t, qa + (size_t) t * Hq * D, ga + (size_t) t * Hq * D, cc[rs_[t]], -1, 1, Hq, Hkv,
                                    D, 64, 1e7f, 1e-6f, g_s);
        if (!ok) { printf("    attn_prep refused  FAIL\n"); ++g_fail; }
        same("attn_prep multi-row q vs per-row", qa, qb, (size_t) T * Hq * D * 4);
        same("attn_prep multi-row gate vs per-row", ga, gb, (size_t) T * Hq * D * 4);
        auto hpool = bytes_of(pool, (size_t) NP * pstride);
        bool okc = true;
        for (int t = 0; t < T; ++t) {
            const int s = rs_[t], p = rp[t];
            auto hk = bytes_of((char *) cc[s].k + (size_t) p * rowb, rowb), hv = bytes_of((char *) cc[s].v + (size_t) p * rowb, rowb);
            const size_t o = (size_t) tabs[(size_t) s * MAXP + (p >> PSH)] * pstride + (size_t) 2 * PS * rowb + (size_t) (p & (PS - 1)) * rowb;
            okc = okc && !memcmp(hk.data(), hpool.data() + o, rowb) && !memcmp(hv.data(), hpool.data() + o + PS * rowb, rowb);
        }
        ++g_checks; if (!okc) ++g_fail;
        printf("    %-60s %s\n", "attn_prep multi-row cache rows (paged) vs per-row (contiguous)", okc ? "identical" : "FAIL");
        for (float * p : {qg, kk, vv, qn, kn, qa, ga, qb, gb}) cudaFree(p);
        cudaFree(dpos); cudaFree(drt);
    }
    for (int s = 0; s < NS; ++s) { F::kv_free(cc[s]); cudaFree(ks[s]); cudaFree(vs[s]); }
    cudaFree(q); cudaFree(o1); cudaFree(o2); cudaFree(pool); cudaFree(dtab);
}

// ======================================================================================================== delta net
static void test_gdn_multi(const std::vector<int> & Ts, bool snap) {
    const int Hk = 8, Hv = 24, D = 128, C = 2 * Hk * D + Hv * D, NG = 3, G = 1, NSLOT = 5, NSNAP = 8;
    const float eps = 1e-6f;
    const int nseq = (int) Ts.size();
    int Ttot = 0, maxT = 0, nsnap = 0;
    for (int t : Ts) { Ttot += t; maxT = std::max(maxT, t); nsnap += t - 1; }
    printf("gdn_layer_multi: %d sequences, T =", nseq);
    for (int t : Ts) printf(" %d", t);
    printf("%s\n", snap ? " (rollback snapshots)" : "");
    const size_t csz = (size_t) C * 3, ssz = (size_t) Hv * D * D;
    // pools [slot][layer][...]; snapshot pool [layer][snap slot][...] (tp_model's layouts); test layer G
    float * cpool = dev_rand((size_t) NSLOT * NG * csz, 0.5f), * spool = dev_rand((size_t) NSLOT * NG * ssz, 0.05f);
    float * csnp = (float *) dev_alloc((size_t) NG * NSNAP * csz * 4), * ssnp = (float *) dev_alloc((size_t) NG * NSNAP * ssz * 4);
    float * cw = dev_rand((size_t) C * 4, 0.3f), * dtb = dev_rand(Hv, 0.5f), * na = dev_rand(Hv, 0.3f, -1.0f), * nw = dev_rand(D, 0.2f, 1.0f);
    const F::GdnWeights w{cw, dtb, na, nw};
    float * x = dev_rand((size_t) Ttot * C, 1.0f), * z = dev_rand((size_t) Ttot * Hv * D, 1.0f);
    float * a0 = dev_rand((size_t) Ttot * Hv, 1.0f), * b0 = dev_rand((size_t) Ttot * Hv, 1.0f);
    // slots in a scrambled order, snapshot ranges back to back
    std::vector<int> slot(nseq), snap0(nseq), row0(nseq);
    {
        std::vector<int> sl(NSLOT); std::iota(sl.begin(), sl.end(), 0);
        for (int i = NSLOT - 1; i > 0; --i) std::swap(sl[i], sl[urand() % (i + 1)]);
        int r = 0, sn = 0;
        for (int i = 0; i < nseq; ++i) { slot[i] = sl[i]; row0[i] = r; snap0[i] = sn; r += Ts[i]; sn += Ts[i] - 1; }
    }
    // ---- reference: one gdn_layer per sequence on copies of its states ----
    float * cref = (float *) dev_alloc((size_t) nseq * csz * 4), * sref = (float *) dev_alloc((size_t) nseq * ssz * 4);
    float * csr = (float *) dev_alloc((size_t) nseq * 4 * csz * 4), * ssr = (float *) dev_alloc((size_t) nseq * 4 * ssz * 4);
    float * aR = (float *) dev_alloc((size_t) Ttot * Hv * 4), * bR = (float *) dev_alloc((size_t) Ttot * Hv * 4);
    float * yR = (float *) dev_alloc((size_t) Ttot * C * 4), * oR = (float *) dev_alloc((size_t) Ttot * Hv * D * 4);
    copy_dev(aR, a0, (size_t) Ttot * Hv * 4); copy_dev(bR, b0, (size_t) Ttot * Hv * 4);
    const int ng = Hv * D / 32;
    std::vector<void *> qref(nseq);
    for (int i = 0; i < nseq; ++i) {
        copy_dev(cref + i * csz, cpool + ((size_t) slot[i] * NG + G) * csz, csz * 4);
        copy_dev(sref + i * ssz, spool + ((size_t) slot[i] * NG + G) * ssz, ssz * 4);
        qref[i] = dev_alloc(F::matvec_scratch_bytes(Hv * D, Ts[i]));
        const int r = row0[i];
        F::gdn_layer(x + (size_t) r * C, aR + (size_t) r * Hv, bR + (size_t) r * Hv, z + (size_t) r * Hv * D, w,
                     cref + i * csz, sref + i * ssz, yR + (size_t) r * C, aR + (size_t) r * Hv, bR + (size_t) r * Hv,
                     oR + (size_t) r * Hv * D, Ts[i], Hk, Hv, D, eps, g_s, qref[i],
                     snap && Ts[i] > 1 ? csr + (size_t) i * 4 * csz : nullptr, snap && Ts[i] > 1 ? ssr + (size_t) i * 4 * ssz : nullptr);
    }
    // ---- multi ----
    float * aM = (float *) dev_alloc((size_t) Ttot * Hv * 4), * bM = (float *) dev_alloc((size_t) Ttot * Hv * 4);
    float * yM = (float *) dev_alloc((size_t) Ttot * C * 4), * oM = (float *) dev_alloc((size_t) Ttot * Hv * D * 4);
    copy_dev(aM, a0, (size_t) Ttot * Hv * 4); copy_dev(bM, b0, (size_t) Ttot * Hv * 4);
    void * qM = dev_alloc(F::matvec_scratch_bytes(Hv * D, 8));
    std::vector<int32_t> hd((size_t) nseq * 4);
    for (int i = 0; i < nseq; ++i) { hd[i * 4] = row0[i]; hd[i * 4 + 1] = Ts[i]; hd[i * 4 + 2] = slot[i]; hd[i * 4 + 3] = snap0[i]; }
    int32_t * dd; CK(cudaMalloc(&dd, hd.size() * 4)); CK(cudaMemcpy(dd, hd.data(), hd.size() * 4, cudaMemcpyHostToDevice));
    F::gdn_multi m;
    m.seqs = dd; m.nseq = nseq; m.Ttot = Ttot; m.maxT = 4;   // the engine sizes the grid for the largest possible T
    m.conv_base = cpool + (size_t) G * csz; m.conv_slot = (size_t) NG * csz;
    m.ssm_base = spool + (size_t) G * ssz; m.ssm_slot = (size_t) NG * ssz;
    if (snap) { m.conv_snap = csnp + (size_t) G * NSNAP * csz; m.ssm_snap = ssnp + (size_t) G * NSNAP * ssz; }
    F::gdn_layer_multi(x, aM, bM, z, w, m, yM, aM, bM, oM, Hk, Hv, D, eps, g_s, qM);
    CK(cudaStreamSynchronize(g_s));
    same("out [Ttot][Hv*D]", oR, oM, (size_t) Ttot * Hv * D * 4);
    same("decay (in place over alpha)", aR, aM, (size_t) Ttot * Hv * 4);
    same("beta (in place over beta)", bR, bM, (size_t) Ttot * Hv * 4);
    {   // a16 rows: row t of sequence i at row0 + t of the Ttot-row buffer vs row t of its own T-row buffer
        auto hm = bytes_of(qM, F::matvec_scratch_bytes(Hv * D, 8));
        bool ok = true;
        for (int i = 0; i < nseq; ++i) {
            auto hr = bytes_of(qref[i], F::matvec_scratch_bytes(Hv * D, Ts[i]));
            for (int t = 0; t < Ts[i]; ++t) {
                const size_t qs = (size_t) ng * 16 * 4;   // half2 per row
                ok = ok && !memcmp(hm.data() + (size_t) (row0[i] + t) * qs, hr.data() + (size_t) t * qs, qs);
                const size_t ab = (size_t) ng * 4;
                ok = ok && !memcmp(hm.data() + (size_t) Ttot * qs + (size_t) (row0[i] + t) * ab,
                                   hr.data() + (size_t) Ts[i] * qs + (size_t) t * ab, ab);
            }
        }
        ++g_checks; if (!ok) ++g_fail;
        printf("    %-60s %s\n", "a16 activation rows (scan epilogue)", ok ? "identical" : "FAIL");
    }
    char nm[96];
    for (int i = 0; i < nseq; ++i) {
        snprintf(nm, sizeof nm, "seq %d (slot %d): final conv state", i, slot[i]);
        same(nm, cref + i * csz, cpool + ((size_t) slot[i] * NG + G) * csz, csz * 4);
        snprintf(nm, sizeof nm, "seq %d (slot %d): final ssm state", i, slot[i]);
        same(nm, sref + i * ssz, spool + ((size_t) slot[i] * NG + G) * ssz, ssz * 4);
        if (snap)
            for (int t = 0; t + 1 < Ts[i]; ++t) {
                snprintf(nm, sizeof nm, "seq %d snapshot %d (pool slot %d): conv + ssm", i, t, snap0[i] + t);
                same(nm, csr + ((size_t) i * 4 + t) * csz, csnp + ((size_t) G * NSNAP + snap0[i] + t) * csz, csz * 4, true);
                same(nm, ssr + ((size_t) i * 4 + t) * ssz, ssnp + ((size_t) G * NSNAP + snap0[i] + t) * ssz, ssz * 4, true);
            }
    }
    for (void * p : qref) cudaFree(p);
    for (void * p : {(void *) cpool, (void *) spool, (void *) csnp, (void *) ssnp, (void *) cw, (void *) dtb, (void *) na, (void *) nw,
                     (void *) x, (void *) z, (void *) a0, (void *) b0, (void *) cref, (void *) sref, (void *) csr, (void *) ssr,
                     (void *) aR, (void *) bR, (void *) yR, (void *) oR, (void *) aM, (void *) bM, (void *) yM, (void *) oM, qM,
                     (void *) dd})
        cudaFree(p);
}

int main() {
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    printf("device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    CK(cudaStreamCreateWithFlags(&g_s, cudaStreamNonBlocking));
    for (uint32_t t : {F::KV_F16, F::KV_Q8_0, F::KV_Q4_0}) test_attention(t);
    for (bool snap : {false, true}) {
        test_gdn_multi({1}, snap);
        test_gdn_multi({1, 1, 1}, snap);
        test_gdn_multi({4, 4}, snap);
        test_gdn_multi({3, 1, 4}, snap);
        test_gdn_multi({2, 3, 1, 2}, snap);
        test_gdn_multi({4}, snap);
    }
    printf("%s (%d checks, %d failures; all checks bit-exact)\n", g_fail ? "FAIL" : "PASS", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
