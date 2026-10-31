// p100e-kld-test: checks src/kld.cpp (the KL-divergence reader / math of p100e-run --kld-base), host only.
//   p100e-kld-test [tmpdir]          synthetic base files written with the perplexity tool's own compression
//                                    (kld_compress_row), read back with kld_base_file, scored with kld_rows:
//                                    header / tokens / rows round trip, an analytic case (known KLD, NLL, same top),
//                                    identical and shifted logits (KLD ~ 0), random logits vs a double-precision KL
//                                    on the unquantized base, 1 vs N threads
//   p100e-kld-test --peek base.kld [--all]   decode a real base file: header, chunk 0's first tokens, row 0 (scale,
//                                    min log-prob, sum of p, top 5, the next token's log-prob), and PPL(base) of
//                                    chunk 0 (--all: every chunk; equals the "Mean PPL(base)" of the tool's KL logs)
#include "kld.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using namespace p100e;

static int n_fail = 0, n_check = 0;
static void check(bool ok, const char * what, double got = 0, double want = 0) {
    ++n_check;
    if (!ok) { ++n_fail; printf("FAIL %s (got %.9g, want %.9g)\n", what, got, want); }
}

// write a base file like the perplexity tool: header, tokens, per chunk the compressed rows of positions first..n_ctx-2
static void write_base(const std::string & path, int n_ctx, int n_vocab, int n_chunk, const std::vector<int32_t> & toks,
                       const std::vector<float> & logits /* [n_chunk][n_scored][n_vocab] */) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) { printf("cannot write %s\n", path.c_str()); exit(1); }
    const uint32_t nc = (uint32_t) n_ctx;
    fwrite("_logits_", 1, 8, f);
    fwrite(&nc, 4, 1, f);
    fwrite(&n_vocab, 4, 1, f);
    fwrite(&n_chunk, 4, 1, f);
    fwrite(toks.data(), 4, toks.size(), f);
    const int first = n_ctx / 2, ns = n_ctx - 1 - first, nv = 2 * ((n_vocab + 1) / 2) + 4;
    std::vector<uint16_t> rows((size_t) ns * nv, 0);
    for (int c = 0; c < n_chunk; ++c) {
        std::fill(rows.begin(), rows.end(), 0);
        for (int i = 0; i < ns; ++i)
            kld_compress_row(n_vocab, logits.data() + ((size_t) c * ns + i) * n_vocab, rows.data() + (size_t) i * nv,
                             toks[(size_t) c * n_ctx + first + i + 1]);
        fwrite(rows.data(), 2, rows.size(), f);
    }
    fclose(f);
}

// exact KL(P_base || Q) in double from unquantized logits
static double kl_exact(int n, const float * lb, const float * lq) {
    double mb = lb[0], mq = lq[0];
    for (int i = 1; i < n; ++i) { mb = std::max(mb, (double) lb[i]); mq = std::max(mq, (double) lq[i]); }
    double sb = 0, sq = 0;
    for (int i = 0; i < n; ++i) { sb += exp(lb[i] - mb); sq += exp(lq[i] - mq); }
    const double zb = mb + log(sb), zq = mq + log(sq);
    double kl = 0;
    for (int i = 0; i < n; ++i) {
        const double lpb = lb[i] - zb, lpq = lq[i] - zq;
        kl += exp(lpb) * (lpb - lpq);
    }
    return kl;
}

static int synthetic(const std::string & dir) {
    const int n_vocab = 1001, n_ctx = 64, n_chunk = 3;   // odd vocab: nv padding
    const int first = n_ctx / 2, ns = n_ctx - 1 - first, nv = 2 * ((n_vocab + 1) / 2) + 4;
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<int32_t> toks((size_t) n_ctx * n_chunk);
    for (auto & t : toks) t = (int32_t) (rng() % n_vocab);
    std::vector<float> lb((size_t) n_chunk * ns * n_vocab);
    for (size_t r = 0; r < (size_t) n_chunk * ns; ++r) {
        float * l = lb.data() + r * n_vocab;
        for (int i = 0; i < n_vocab; ++i) l[i] = 3.0f * nd(rng);
        // a peaked row, like a language model, peaked at the token it predicts (a random target would often lie
        // below the file's 16-nat log-prob floor, where NLL(base) is clamped: checked in the analytic part)
        const size_t c = r / ns, i = r % ns;
        l[toks[c * n_ctx + first + i + 1]] += 8.0f;
    }
    const std::string path = dir + "/kld_test_base.kld";
    write_base(path, n_ctx, n_vocab, n_chunk, toks, lb);

    // 1. reader round trip
    kld_base_file kf(path);
    check(kf.n_ctx == (uint32_t) n_ctx && kf.n_vocab == n_vocab && kf.n_chunk == n_chunk, "header");
    check(kf.nv == nv && kf.first == first && kf.n_scored == ns, "derived sizes", kf.nv, nv);
    check(kf.tokens == toks, "tokens");
    std::vector<uint16_t> rows, want((size_t) ns * nv);
    for (int c = 0; c < n_chunk; ++c) {
        kf.read_chunk(c, rows);
        std::fill(want.begin(), want.end(), 0);
        for (int i = 0; i < ns; ++i)
            kld_compress_row(n_vocab, lb.data() + ((size_t) c * ns + i) * n_vocab, want.data() + (size_t) i * nv,
                             toks[(size_t) c * n_ctx + first + i + 1]);
        check(rows == want, "chunk rows round trip");
    }

    // 2. identical logits: KLD ~ 0 (only the 16-bit log-prob quantization), same top 100 %, PPL(Q) = PPL(base)
    // 3. shifted logits (+5): log-softmax invariant, same per-row values up to float rounding
    // 4. random perturbation: KLD vs the exact double KL on the unquantized base
    for (int mode = 0; mode < 3; ++mode) {
        std::vector<float> lq = lb;
        if (mode == 1) for (auto & v : lq) v += 5.0f;
        if (mode == 2) for (auto & v : lq) v += 0.5f * nd(rng);
        kld_stats st, st1;
        std::vector<float> kv((size_t) n_chunk * ns), pd((size_t) n_chunk * ns), kv1(kv.size()), pd1(pd.size());
        double max_err = 0, sum_exact = 0;
        for (int c = 0; c < n_chunk; ++c) {
            kf.read_chunk(c, rows);
            const float * lqc = lq.data() + (size_t) c * ns * n_vocab;
            kld_rows(n_vocab, lqc, toks.data() + (size_t) c * n_ctx + first, ns, rows.data(), st, kv.data() + (size_t) c * ns,
                     pd.data() + (size_t) c * ns, 4);
            kld_rows(n_vocab, lqc, toks.data() + (size_t) c * n_ctx + first, ns, rows.data(), st1, kv1.data() + (size_t) c * ns,
                     pd1.data() + (size_t) c * ns, 1);
            for (int i = 0; i < ns; ++i) {
                const double ex = kl_exact(n_vocab, lb.data() + ((size_t) c * ns + i) * n_vocab, lqc + (size_t) i * n_vocab);
                sum_exact += ex;
                max_err = std::max(max_err, std::fabs(ex - kv[(size_t) c * ns + i]));
            }
        }
        const double mean = st.sum_kld / st.count, mean_exact = sum_exact / st.count;
        printf("mode %d (%s): rows %zu mean KLD %.8f exact %.8f max |row err| %.2e same top %zu/%zu\n", mode,
               mode == 0 ? "identical" : mode == 1 ? "shift +5" : "noise 0.5", st.count, mean, mean_exact, max_err,
               st.n_same_top, st.count);
        check(st.count == (size_t) n_chunk * ns, "row count");
        check(kv == kv1 && pd == pd1, "1 vs 4 threads: per-row values identical");
        check(st1.n_same_top == st.n_same_top && st1.count == st.count, "1 vs 4 threads: counts");
        check(std::fabs(st1.sum_kld - st.sum_kld) <= 1e-9 * std::max(1.0, std::fabs(st.sum_kld)), "1 vs 4 threads: sum",
              st1.sum_kld, st.sum_kld);
        // 16-bit log-probs: step (max - min) / 65535 <= 16 / 65535 = 2.4e-4 nats, rounding error <= 1.2e-4 per entry
        check(max_err < 5e-4, "per-row KLD vs exact double KL", max_err, 5e-4);
        if (mode < 2) {
            check(std::fabs(mean) < 2e-4, "KLD ~ 0 for identical / shifted logits", mean, 0);
            check(st.n_same_top == st.count, "same top 100 %", (double) st.n_same_top, (double) st.count);
            check(std::fabs(st.sum_nll - st.sum_nll_base) / st.count < 2e-4, "mean NLL(Q) = NLL(base)",
                  st.sum_nll / st.count, st.sum_nll_base / st.count);
        } else {
            check(mean > 0.05 && std::fabs(mean - mean_exact) < 1e-4, "noise: mean KLD matches exact", mean, mean_exact);
        }
        if (mode == 2) { kld_print_final(stdout, st, kv, pd); kld_print_summary(stdout, "synthetic", st, kv); }
    }

    // 5. analytic case: base p = (1/2, 1/2, ~0...), model q = (3/4, 1/4, ~0...): KL = 0.5 ln(2/3) + 0.5 ln 2
    {
        std::vector<float> b(n_vocab, -100.0f), q(n_vocab, -100.0f);
        b[0] = 0.0f; b[1] = 0.0f;
        q[0] = logf(0.75f); q[1] = logf(0.25f);
        const int nv1 = nv;
        std::vector<uint16_t> row(nv1, 0);
        const int tok = 1;
        const double nll_writer = kld_compress_row(n_vocab, b.data(), row.data(), tok);
        kld_stats st;
        const auto v = kld_row(n_vocab, q.data(), row.data(), tok, st);
        const double want = 0.5 * log(0.5 / 0.75) + 0.5 * log(0.5 / 0.25);
        printf("analytic: KLD %.8f want %.8f, nll %.6f (want %.6f), nll_base %.6f (want %.6f, writer %.6f), p_diff %.6f, "
               "same top %zu\n", v.first, want, st.sum_nll, -log(0.25), st.sum_nll_base, log(2.0), nll_writer, v.second,
               st.n_same_top);
        check(std::fabs(v.first - want) < 2e-4, "analytic KLD", v.first, want);
        check(std::fabs(st.sum_nll + log(0.25)) < 1e-5, "analytic NLL", st.sum_nll, -log(0.25));
        check(std::fabs(st.sum_nll_base - log(2.0)) < 3e-4, "analytic NLL(base)", st.sum_nll_base, log(2.0));
        check(std::fabs(v.second - (0.25f - 0.5f)) < 2e-4, "analytic p_diff", v.second, -0.25);
        check(st.n_same_top == 1, "analytic same top");
        // a target far below the floor: the file stores min_log_prob = (max - 16) - max - log(sum) for it, so
        // NLL(base) = 16 + ln 2 (the perplexity tool's clamp), while NLL(Q) is the true -ln q = 100
        kld_stats sc;
        std::vector<uint16_t> row2(nv1, 0);
        kld_compress_row(n_vocab, b.data(), row2.data(), 7);
        kld_row(n_vocab, q.data(), row2.data(), 7, sc);
        printf("clamp: target below the floor: nll_base %.6f (want %.6f), nll %.4f (want %.4f)\n", sc.sum_nll_base,
               16.0 + log(2.0), sc.sum_nll, 100.0);
        check(std::fabs(sc.sum_nll_base - (16.0 + log(2.0))) < 1e-4, "clamped NLL(base)", sc.sum_nll_base, 16.0 + log(2.0));
        check(std::fabs(sc.sum_nll - 100.0) < 1e-3, "unclamped NLL(Q)", sc.sum_nll, 100.0);
    }
    remove(path.c_str());
    printf("%s: %d/%d checks passed\n", n_fail ? "FAILED" : "ALL CHECKS PASSED", n_check - n_fail, n_check);
    return n_fail ? 1 : 0;
}

static int peek(const char * path, bool all) {
    kld_base_file kf(path);
    printf("file %s: %llu bytes, n_ctx %u, n_vocab %d, n_chunk %d, nv %d, scored rows per chunk %d (positions %d..%u)\n",
           path, (unsigned long long) kf.file_size, kf.n_ctx, kf.n_vocab, kf.n_chunk, kf.nv, kf.n_scored, kf.first, kf.n_ctx - 2);
    printf("chunk 0 tokens [0..8):");
    for (int i = 0; i < 8; ++i) printf(" %d", kf.tokens[i]);
    printf("\n");
    std::vector<uint16_t> rows;
    double sum = 0, sum2 = 0;
    size_t cnt = 0;
    for (int c = 0; c < (all ? kf.n_chunk : 1); ++c) {
        kf.read_chunk(c, rows);
        const int32_t * tk = kf.tokens.data() + (size_t) c * kf.n_ctx + kf.first;
        for (int r = 0; r < kf.n_scored; ++r) {
            const uint16_t * row = rows.data() + (size_t) r * kf.nv;
            const float * d = (const float *) row;
            const float scale = d[0], minlp = d[1];
            const uint16_t * q = row + 4;
            const float nll_base = -(scale * q[tk[r + 1]] + minlp);   // kld_row's formula
            sum += nll_base;
            sum2 += nll_base * nll_base;
            ++cnt;
            if (c == 0 && r == 0) {
                double psum = 0;
                std::vector<int> idx(kf.n_vocab);
                std::iota(idx.begin(), idx.end(), 0);
                for (int i = 0; i < kf.n_vocab; ++i) psum += exp((double) scale * q[i] + minlp);
                std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(), [&](int a, int b) { return q[a] > q[b]; });
                printf("chunk 0 row 0 (position %d, predicts token %d): scale %.6g, min log-prob %.4f, sum p %.6f, top 5:",
                       kf.first, tk[1], scale, minlp, psum);
                for (int k = 0; k < 5; ++k) printf(" %d:%.4f", idx[k], scale * q[idx[k]] + minlp);
                printf("; log p(next) %.4f\n", -nll_base);
            }
        }
        if (c == 0 || all) {
            const double m = sum / cnt;
            printf("after chunk %d: PPL(base) %.6f over %zu rows\n", c + 1, exp(m), cnt);
        }
    }
    const double m = sum / cnt, v = sum2 / cnt - m * m;
    printf("PPL(base) %.6f +- %.6f (%zu rows)\n", exp(m), exp(m) * (v > 0 ? sqrt(v / (cnt - 1)) : 0.0), cnt);
    return 0;
}

int main(int argc, char ** argv) {
    try {
        if (argc >= 3 && !strcmp(argv[1], "--peek")) return peek(argv[2], argc >= 4 && !strcmp(argv[3], "--all"));
        return synthetic(argc >= 2 ? argv[1] : ".");
    } catch (const std::exception & e) {
        printf("error: %s\n", e.what());
        return 1;
    }
}
