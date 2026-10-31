// p100e-run --cache-test / --sched-test, included by p100e_run.cpp after sessions_mode.inc.
//
// --cache-test (prompt = --tokens-file, real text): every cached run is compared with a COLD run of the same request
// (cache = false, same split positions) for identical tokens (and with MTP identical draft hashes: the draft head's KV
// and carry restored from the cache are those of the cold run), plus prefilled tokens and times:
//   1. A / A+B: request A (--cache-a tokens), then A+B (--cache-b more) with split position |A| (a message start): only
//      |B| tokens prefilled. Then A+B without split positions: reuse stops at the last boundary both partitions share
//      that has a checkpoint (the exactness rule, src/p100e_sched.h) - still identical to its cold run.
//   2. multi-turn replay (like Claude Code): turn 1 = --cache-turn0 tokens; turn k = turn k-1's prompt + its generated
//      tokens + the next --cache-step document tokens; split positions = every earlier turn / answer start. Per turn:
//      cached / prefilled tokens, time to first token and turn time vs the cold run.
//   1b. mid-segment reuse: A[:c+8] + other tokens with a split at c+38 after the checkpoint c: the cold
//      run's chunks from c, identical tokens.
//   file checks (cache_file_checks): atomic replace of an existing session file, load + continue, invalid headers
//      refused without leftovers, a file with other numerics loads without exact reuse. --cache-dir <dir>: where the
//      session files go (default: the temp directory; removed at the end when empty).
//   3. disk: the last turn's session is saved, continued (submit_continue) and a new request is run on its prefix; then
//      this process frees the GPUs and starts itself as a NEW process (--cache-child) that loads the file and repeats
//      both: identical tokens, cached-token count and draft hashes.
// --sched-test: scheduler regressions (admission with continuations / resumes, release, host
// budget, oversized jobs, the next turn of a kept MTP session drafting from its first step).
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <process.h>
#include <windows.h>
#define p100e_fseek _fseeki64
#define p100e_ftell _ftelli64
#else
#include <sys/stat.h>
#define p100e_fseek fseeko
#define p100e_ftell ftello
#endif
#include <ctime>
#include <fstream>

struct cache_args {
    sessions_args S;                 // model, devices, ctx, ubatch, kv, prompts (the document), mtp, pool, reserve
    int a = 12000, b = 4000;         // --cache-a / --cache-b
    int turns = 6, turn0 = 6000, step = 2500, cgen = 32;   // --cache-turns / --cache-turn0 / --cache-step / --cache-gen
    bool disk = true;                // --cache-no-disk
    bool no_split_variant = true;    // --cache-no-variant: skip A+B without split positions
    bool offload_turns = true;       // --cache-no-offload: no filler jobs before every 2nd turn (they push the idle
                                     // session to host RAM, so that turn reuses an offloaded entry)
    int ckpt_every = -1, min_split = -1;   // --ckpt-every / --min-split (default: sched_opts)
    bool inexact = false;            // --cache-inexact
    int64_t sched_pool = 4096;       // --sched-pool: KV pool tokens of the --sched-test engine (below --ctx)
    int save_ckpts = 2;              // --cache-save-ckpts N: newest checkpoints in the session file (-1 = all)
    std::string dir;                 // --cache-dir: directory of the session files (default: the temp directory)
    int dturns = 4, dgen = 256;      // --cache-dturns / --cache-dgen: decoded-token reuse replay (0 turns: skipped)
    int fair_prompt = 20000;         // --fair-test N: the long prompt's tokens
    std::string only;                // --cache-only ab,file,dec,replay: cache test sections (default all)
    std::vector<double> fair_shares{0.0, 0.5};   // --fair-shares a,b,..: decode_share values compared (0 = the old rule)
    std::string child_sess, child_expect;   // --cache-child <session> <expect>
    std::vector<std::string> argv;   // this process's arguments (the child's command line)
};

static engine_opts cache_engine_opts(const cache_args & A) {
    engine_opts eo;
    eo.model = A.S.model;
    eo.devices = A.S.devices;
    eo.kv_type = A.S.kv_type;
    eo.n_ctx = A.S.ctx;
    eo.ubatch = A.S.ubatch > 0 ? std::max(A.S.ubatch, 16) : 2048;
    eo.max_seqs = 3;
    eo.mtp = A.S.mtp_k > 0;
    eo.mo.k = A.S.mtp_k > 0 ? A.S.mtp_k : 3;
    eo.mo.tau = A.S.tau;
    eo.mo.k_min = A.S.kmin;
    eo.pool_tokens = A.S.pool < 0 ? -1 : A.S.pool > 0 ? A.S.pool : (int64_t) A.S.ctx * 3;
    eo.reserve_mb = A.S.reserve_mb;
    if (A.ckpt_every >= 0) eo.so.ckpt_every = A.ckpt_every;
    if (A.min_split >= 0) eo.so.min_split = A.min_split;
    eo.so.exact = !A.inexact;
    return eo;
}

static sched_opts cache_sched_opts(engine & E) {
    sched_opts so = E.opts().so;
    so.max_active = E.opts().max_seqs;
    so.mo = E.opts().mo;
    return so;
}

// one request to completion
static const sched::job & cache_run1(sched & S, const sched::request & r) {
    const int id = S.submit(r);
    S.run();
    const auto & j = S.get(id);
    if (j.st != sched::job::DONE) throw std::runtime_error("cache test: job " + std::to_string(id) + " failed: " + j.err);
    return j;
}

static void cache_print_job(const char * tag, const sched::job & j) {
    printf("  %-22s prompt %6zu: cached %6d, prefilled %6d in %7.2f s (catch-up %4.0f ms, %d ckpt %.0f ms, reuse %.1f ms%s), "
           "ttft %7.2f s, %3zu tokens, turn %7.2f s",
           tag, j.prompt.size(), j.n_cached, j.n_prefill, j.t_prefill, 1e3 * j.t_catchup, j.ckpts, 1e3 * j.t_ckpt,
           1e3 * j.t_reuse, j.reuse_offloaded ? " from host" : "", j.t_first - j.t_submit, j.out.size(),
           j.t_done - j.t_submit);
    if (j.mtp) printf(", accepted %d/%d", j.accepted, j.drafted);
    printf("\n");
}

static bool cache_same(const sched::job & a, const sched::job & b) {
    return a.out == b.out && (!a.mtp || a.draft_hash == b.draft_hash);
}

static void cache_print_verdict(const sched::job & c, const sched::job & cold) {
    const bool st = c.out == cold.out, sd = !c.mtp || c.draft_hash == cold.draft_hash;
    printf("    -> tokens %s%s, ttft x%.2f, turn x%.2f\n", st ? "IDENTICAL" : "MISMATCH",
           c.mtp ? (sd ? ", drafts IDENTICAL" : ", drafts DIFFER") : "",
           (cold.t_first - cold.t_submit) / std::max(1e-9, c.t_first - c.t_submit),
           (cold.t_done - cold.t_submit) / std::max(1e-9, c.t_done - c.t_submit));
    if (!st)
        for (size_t i = 0; i < std::min(c.out.size(), cold.out.size()); ++i)
            if (c.out[i] != cold.out[i]) { printf("    first diff at %zu: %d vs cold %d\n", i, c.out[i], cold.out[i]); break; }
}

static std::string cache_quote(const std::string & a) {
    if (!a.empty() && a.find_first_of(" \t\"") == std::string::npos) return a;
    return "\"" + a + "\"";
}

static void write_ids(std::ofstream & f, const char * key, const std::vector<int32_t> & v) {
    f << key << " " << v.size();
    for (int32_t x : v) f << " " << x;
    f << "\n";
}
static std::vector<int32_t> read_ids(std::ifstream & f, const char * key) {
    std::string k;
    size_t n = 0;
    f >> k >> n;
    if (k != key) throw std::runtime_error(std::string("expect file: ") + key + " expected, got " + k);
    std::vector<int32_t> v(n);
    for (auto & x : v) f >> x;
    return v;
}

static int run_cache_child(cache_args & A);

// KL(p_a || p_b) of two logit rows (softmax in double), and whether b's argmax is a's
static double cache_kl(const std::vector<float> & a, const std::vector<float> & b) {
    const size_t n = std::min(a.size(), b.size());
    double ma = -1e300, mb = -1e300;
    for (size_t i = 0; i < n; ++i) { ma = std::max(ma, (double) a[i]); mb = std::max(mb, (double) b[i]); }
    double sa = 0, sb = 0;
    for (size_t i = 0; i < n; ++i) { sa += exp((double) a[i] - ma); sb += exp((double) b[i] - mb); }
    const double la = ma + log(sa), lb = mb + log(sb);
    double kl = 0;
    for (size_t i = 0; i < n; ++i) {
        const double lpa = a[i] - la, lpb = b[i] - lb;
        kl += exp(lpa) * (lpa - lpb);
    }
    return std::max(0.0, kl);
}

// Decoded-token reuse on a multi-turn replay like Claude Code's: turn k = turn k-1's prompt + its
// answer (dgen tokens) + the next document tokens, split positions at every answer / message start. Per turn three runs
// of the same prompt: cold (no cache), cached in EXACT mode (reuse stops at the answer's start: the answer is prefilled
// again) and cached with DECODED reuse (the default: the previous turn's state incl. its decoded answer, only the new
// tokens are prefilled). The cached runs are teacher-forced along the cold run's tokens and every step's full logits
// are compared with the cold run's: KL(cold || cached), top-1 agreement (= the greedy output stays identical while
// every pick agrees), bitwise equality (exact mode must be bit-identical). Plain jobs (the MTP path's tokens are the
// plain ones). Returns {exact runs bit-identical, exact runs}.
static std::pair<int, int> cache_decoded_replay(tp_model & m, tp_mtp * d, const sched_opts & so0, const cache_args & A,
                                                const std::vector<int32_t> & doc) {
    sched_opts sx = so0, sd = so0;
    sx.exact = true;
    sd.exact = false;
    sched SX(m, d, sx), SD(m, d, sd);
    std::vector<int32_t> P(doc.begin(), doc.begin() + std::min<size_t>(A.turn0, doc.size()));
    std::vector<int> splits;
    size_t off = P.size();
    std::vector<int32_t> ans;
    int ok = 0, n = 0;
    double sum_ttft[3] = {0, 0, 0};
    long long sum_pf[3] = {0, 0, 0};
    double kl_sum = 0, kl_max = 0;
    long long agree = 0, steps = 0;
    int greedy_same = 0, turns_cmp = 0;
    printf("CACHE decoded-token reuse replay (%d turns: %zu tokens, then + the answer (%d tokens) + %d document tokens per "
           "turn; cold vs cached exact vs cached with decoded reuse, teacher-forced, full logits):\n", A.dturns, P.size(),
           A.dgen, A.step);
    for (int k = 0; k < A.dturns; ++k) {
        if (k > 0) {
            if (off + A.step > doc.size()) { printf("  (document exhausted after %d turns)\n", k); break; }
            splits.push_back((int) P.size());   // the answer starts a message
            P.insert(P.end(), ans.begin(), ans.end());
            splits.push_back((int) P.size());   // the next user message
            P.insert(P.end(), doc.begin() + off, doc.begin() + off + A.step);
            off += A.step;
        }
        sched::request r;
        r.prompt = P;
        r.n_gen = A.dgen;
        r.splits = splits;
        // cold
        std::vector<std::vector<float>> lc, lx, ld;
        sched::request rc = r;
        rc.cache = false;
        const int jc = SX.submit(rc);
        SX.set_trace(jc, &lc, nullptr);
        SX.run();
        const sched::job & c = SX.get(jc);
        if (c.st != sched::job::DONE) throw std::runtime_error("decoded replay: cold run failed: " + c.err);
        const std::vector<int32_t> out_c = c.out;
        // cached, exact
        const int jx = SX.submit(r);
        SX.set_trace(jx, &lx, &out_c);
        SX.run();
        // cached, decoded reuse
        const int jd = SD.submit(r);
        SD.set_trace(jd, &ld, &out_c);
        SD.run();
        const sched::job & x = SX.get(jx), & dd = SD.get(jd);
        if (x.st != sched::job::DONE || dd.st != sched::job::DONE) throw std::runtime_error("decoded replay: a cached run failed");
        // exact: bitwise
        bool bit = lx.size() == lc.size();
        for (size_t i = 0; bit && i < lc.size(); ++i) bit = lc[i] == lx[i];
        ++n;
        ok += bit;
        // decoded: KL, agreement
        double kmax = 0, ksum = 0;
        int ag = 0, first_div = -1;
        for (size_t i = 0; i < lc.size() && i < ld.size(); ++i) {
            const double kl = cache_kl(lc[i], ld[i]);
            ksum += kl;
            kmax = std::max(kmax, kl);
            const bool same = dd.own[i] == out_c[i];
            ag += same;
            if (!same && first_div < 0) first_div = (int) i;
        }
        const double cold_t = c.t_first - c.t_submit, x_t = x.t_first - x.t_submit, d_t = dd.t_first - dd.t_submit;
        printf("  turn %d prompt %6zu: cold ttft %6.2f s | exact: cached %6d, prefilled %5d, ttft %5.2f s, logits %s | decoded: "
               "cached %6d, prefilled %5d, ttft %5.2f s, KL mean %.2e max %.2e, top-1 %d/%zu, greedy %s\n",
               k + 1, P.size(), cold_t, x.n_cached, x.n_prefill, x_t, bit ? "bit-identical" : "DIFFER", dd.n_cached,
               dd.n_prefill, d_t, ksum / std::max<size_t>(1, lc.size()), kmax, ag, lc.size(),
               first_div < 0 ? "identical" : ("diverges at " + std::to_string(first_div)).c_str());
        fflush(stdout);
        sum_ttft[0] += cold_t; sum_ttft[1] += x_t; sum_ttft[2] += d_t;
        sum_pf[0] += c.n_prefill; sum_pf[1] += x.n_prefill; sum_pf[2] += dd.n_prefill;
        if (k > 0) {   // turn 1 reuses nothing
            kl_sum += ksum; kl_max = std::max(kl_max, kmax); agree += ag; steps += (long long) lc.size();
            greedy_same += first_div < 0; ++turns_cmp;
        }
        ans = out_c;
    }
    printf("  DECODED REUSE: prefilled %lld (cold) / %lld (exact) / %lld (decoded) tokens, ttft %.2f / %.2f / %.2f s; turns 2..: "
           "KL(cold || decoded) mean %.2e max %.2e, top-1 agreement %lld/%lld = %.2f %%, greedy output identical in %d/%d "
           "turns; exact runs bit-identical %d/%d\n", sum_pf[0], sum_pf[1], sum_pf[2], sum_ttft[0], sum_ttft[1], sum_ttft[2],
           kl_sum / std::max(1LL, steps), kl_max, agree, steps, 100.0 * agree / std::max(1LL, steps), greedy_same, turns_cmp,
           ok, n);
    fflush(stdout);
    return {ok, n};
}

// the directory for session files (--cache-dir, created when missing; default: the temp directory), with a separator
static std::string cache_dir(const cache_args & A) {
    std::string d = A.dir;
    if (d.empty()) {
        char tmpdir[1024] = ".";
#ifdef _WIN32
        GetTempPathA(sizeof tmpdir, tmpdir);
#else
        snprintf(tmpdir, sizeof tmpdir, "%s", getenv("TMPDIR") ? getenv("TMPDIR") : "/tmp");
#endif
        d = tmpdir;
    }
#ifdef _WIN32
    CreateDirectoryA(d.c_str(), nullptr);
    if (d.back() != '\\' && d.back() != '/') d += '\\';
#else
    mkdir(d.c_str(), 0755);
    if (d.back() != '/') d += '/';
#endif
    return d;
}

// patch n bytes at offset (from the end when < 0) of a file in place; returns the old bytes
static std::vector<char> cache_patch(const std::string & path, long long off, const void * bytes, size_t n) {
    FILE * f = fopen(path.c_str(), "r+b");
    if (!f) throw std::runtime_error("cannot open " + path);
    if (off < 0) { p100e_fseek(f, 0, SEEK_END); off += p100e_ftell(f); }
    std::vector<char> old(n);
    p100e_fseek(f, off, SEEK_SET);
    if (fread(old.data(), 1, n, f) != n) { fclose(f); throw std::runtime_error("short file " + path); }
    p100e_fseek(f, off, SEEK_SET);
    fwrite(bytes, 1, n, f);
    fclose(f);
    return old;
}

// session file checks: a small held session with one checkpoint is saved twice to
// the same path (atomic replace over an existing file), loaded and continued (tokens equal to the original's
// continuation); then the file is patched in place (and restored) so the header is invalid - boundaries not ascending,
// carry position beyond the sequence, a checkpoint beyond the sequence, an unknown flag - and each load must fail with
// nothing left behind (entries, host bytes); last, a different numerics hash loads but its prefix is not reused by
// an exact lookup. Returns {passed, tests}.
static std::pair<int, int> cache_file_checks(tp_model & m, tp_mtp * d, const sched_opts & so0, const std::vector<int32_t> & doc,
                                             const std::string & dir, bool mtp) {
    int ok = 0, n = 0;
    auto check = [&](const char * name, bool pass, const std::string & detail) {
        ++n;
        ok += pass;
        printf("  %-62s %s  %s\n", name, pass ? "PASS" : "FAIL", detail.c_str());
        fflush(stdout);
    };
    sched_opts so = so0;
    so.exact = true;
    sched S(m, d, so);
    const int P = std::min<int>(600, (int) doc.size() - 200);
    sched::request r;
    r.prompt.assign(doc.begin(), doc.begin() + P);
    r.n_gen = 8;
    r.mtp = mtp;
    r.keep = true;
    const int h = S.submit(r);
    S.run();
    if (S.get(h).st != sched::job::DONE) throw std::runtime_error("file checks: the session job failed");
    const std::string path = dir + "p100e_cache_filecheck.p100esess";
    S.save(h, path, -1);
    long long size1 = 0;
    {
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        size1 = (long long) f.tellg();
    }
    bool threw = false;
    std::string msg;
    try { S.save(h, path, -1); } catch (const std::exception & e) { threw = true; msg = e.what(); }
    std::ifstream tf(path + ".tmp");
    check("save over an existing file (atomic replace)", !threw && !tf.good(),
          threw ? msg : std::to_string(size1 >> 20) + " MiB, no .tmp left");
    tf.close();
    // the original continued vs the loaded copy continued
    const int c1 = S.submit_continue(h, 6, mtp);
    S.run();
    const std::vector<int32_t> want = S.get(c1).out;
    S.release(c1);
    const size_t host0 = m.host_bytes();
    const int n_ent0 = S.n_entries();
    int hl = -1;
    try { hl = S.load(path); } catch (const std::exception & e) { msg = e.what(); }
    bool same = false;
    if (hl >= 0) {
        const int c2 = S.submit_continue(hl, 6, mtp);
        S.run();
        same = S.get(c2).out == want;
        S.release(c2);
    }
    check("load + continue == the original session continued", same, hl < 0 ? msg : same ? "identical" : "MISMATCH");
    // header layout: magic 8 | version 4 | h[8] (last, n_pf, n_draft, carry_at, has_carry, keep, n_bnd, exact) | u64 numerics
    // | bnd[n_bnd] ...; checkpoints at the end: per checkpoint i32 pos, i32 carry flag, ckpt_bytes()
    const long long H = 12, NH = 12 + 32, BND = NH + 8;
    const long long ck_rec = 8 + (long long) S.ckpt_bytes();
    struct bad_case { const char * name; long long off; int32_t v; };
    const int32_t big = 1 << 28;
    std::vector<bad_case> cases = {
        {"boundaries not ascending -> refused", BND + 4, 0},                       // bnd[1] = 0
        {"carry position beyond the sequence -> refused", H + 12, big},            // carry_at
        {"n_draft beyond the sequence -> refused", H + 8, big},                    // n_draft
        {"unknown flag value -> refused", H + 28, 2},                              // exact = 2
        {"checkpoint beyond the sequence -> refused", -ck_rec, big},               // the (only / last) checkpoint's pos
    };
    for (const bad_case & c : cases) {
        const std::vector<char> old = cache_patch(path, c.off, &c.v, 4);
        std::string err;
        bool refused = false;
        try { S.load(path); } catch (const std::exception & e) { refused = true; err = e.what(); }
        cache_patch(path, c.off, old.data(), 4);
        const bool clean = m.host_bytes() == host0 && S.n_entries() == n_ent0;
        check(c.name, refused && clean, (refused ? err : std::string("LOADED")) + (clean ? "" : ", NOT CLEANED UP"));
    }
    // another build's numerics: loads (valid KV), but an exact lookup must not reuse its prefix
    {
        S.release(h);
        uint64_t x = 0x5a5a5a5a5a5a5a5aull;
        std::vector<char> old;
        {
            FILE * f = fopen(path.c_str(), "rb");
            uint64_t nh = 0;
            p100e_fseek(f, NH, SEEK_SET);
            fread(&nh, 8, 1, f);
            fclose(f);
            x ^= nh;
            old = cache_patch(path, NH, &x, 8);
        }
        int hx = -1;
        std::string err;
        try { hx = S.load(path); } catch (const std::exception & e) { err = e.what(); }
        cache_patch(path, NH, old.data(), 8);
        int cached = -1;
        if (hx >= 0) {
            sched::request q = r;
            q.prompt.assign(doc.begin(), doc.begin() + P + 100);
            q.keep = false;
            const int jq = S.submit(q);
            S.run();
            cached = S.get(jq).n_cached;
        }
        check("other numerics: loads, no exact reuse of its prefix", hx >= 0 && cached == 0,
              hx < 0 ? err : "a request on its prefix reused " + std::to_string(cached) + " tokens (0 expected)");
    }
    remove(path.c_str());
    return {ok, n};
}

static int run_cache_test(cache_args & A) {
    using clk = std::chrono::steady_clock;
    const auto t0 = clk::now();
    if (A.S.prompts.empty() || A.S.prompts[0].empty()) throw std::runtime_error("--cache-test needs a document (--tokens-file)");
    const std::vector<int32_t> doc = A.S.prompts[0];
    const bool mtp = A.S.mtp_k > 0;
    auto E = std::make_unique<engine>(cache_engine_opts(A));
    tp_model & m = E->model();
    tp_mtp * d = E->drafter();
    const sched_opts so = cache_sched_opts(*E);
    printf("load %.2f s; cache test: document %zu tokens, %s, ubatch %d, ctx %d, pool %lld tokens, checkpoint %.1f MiB "
           "(both GPUs), budget %zu MiB, every %d, tail %d, min split %d, %s\n",
           std::chrono::duration<double>(clk::now() - t0).count(), doc.size(), mtp ? "MTP" : "plain", m.max_batch(), m.n_ctx(),
           (long long) m.pool_tokens(), (m.state_bytes() * m.n_ranks() + m.n_embd() * 4.0) / 1048576.0, so.ckpt_budget >> 20,
           so.ckpt_every, so.ckpt_tail, so.min_split, so.exact ? "exact" : "INEXACT");
    m.print_mem("after load");
    fflush(stdout);
    int n_ok = 0, n_cmp = 0;
    auto tally = [&](const sched::job & c, const sched::job & cold) { ++n_cmp; n_ok += cache_same(c, cold); };
    auto slice = [&](size_t a, size_t n) {
        if (a + n > doc.size()) throw std::runtime_error("cache test: the document is too short for these sizes");
        return std::vector<int32_t>(doc.begin() + a, doc.begin() + a + n);
    };
    const int G = A.S.gen > 0 ? A.S.gen : 32;
    // --cache-only ab,file,dec,replay: the sections to run (default all; replay includes the disk round trip)
    auto want = [&](const char * sec) {
        return A.only.empty() || ("," + A.only + ",").find(std::string(",") + sec + ",") != std::string::npos;
    };
    // ---- 1. A then A+B ----
    if (want("ab")) {
        sched S(m, d, so);
        sched::request ra;
        ra.prompt = slice(0, A.a);
        ra.n_gen = G;
        ra.mtp = mtp;
        printf("CACHE A/A+B (A %d tokens, B %d tokens, %d generated):\n", A.a, A.b, G);
        const sched::job ja = cache_run1(S, ra);
        cache_print_job("A (cold, cached after)", ja);
        sched::request rab = ra;
        rab.prompt = slice(0, A.a + A.b);
        rab.splits = {A.a};
        const sched::job jab = cache_run1(S, rab);
        cache_print_job("A+B cached", jab);
        sched::request cold = rab;
        cold.cache = false;
        const sched::job jc = cache_run1(S, cold);
        cache_print_job("A+B cold", jc);
        cache_print_verdict(jab, jc);
        tally(jab, jc);
        printf("    A+B: %d of %d prompt tokens from the cache (expected %d)\n", jab.n_cached, A.a + A.b, A.a);
        if (A.no_split_variant) {
            sched::request r2 = rab;
            r2.splits.clear();
            const sched::job j2 = cache_run1(S, r2);
            cache_print_job("A+B no splits, cached", j2);
            sched::request c2 = r2;
            c2.cache = false;
            const sched::job jc2 = cache_run1(S, c2);
            cache_print_job("A+B no splits, cold", jc2);
            cache_print_verdict(j2, jc2);
            tally(j2, jc2);
        }
        // reuse at a checkpoint in the middle of a segment (the first ckpt_every one, a max_batch chunk end)
        // with a split position less than min_split after it. The cached entry (A..., no splits up to there) has the
        // checkpoint c; the request is A[:c+8] + other tokens with a message start at c + 38. Its cold partition is
        // ... c | c+38 | ...; a partition restarted at c would skip c + 38 (within min_split) and differ.
        {
            const int mb = m.max_batch();
            const int c = ((so.ckpt_every + mb - 1) / mb) * mb;
            const int lcp = c + 8, sp = c + 38;
            if (so.min_split > 38 && lcp < A.a && sp + 64 < A.a + A.b) {
                sched::request r = ra;
                const int nx = std::min({A.b, 2000, (int) doc.size() - (A.a + A.b)});
                std::vector<int32_t> x = slice(A.a + A.b, std::max(64, nx));   // tokens from elsewhere in the document
                if (x[0] == doc[lcp]) x[0] = (x[0] + 1) % std::max(2, m.vocab());
                r.prompt = slice(0, lcp);
                r.prompt.insert(r.prompt.end(), x.begin(), x.end());
                r.splits = {sp};
                const sched::job j1 = cache_run1(S, r);
                cache_print_job("mid-segment reuse", j1);
                sched::request rc = r;
                rc.cache = false;
                const sched::job jc1 = cache_run1(S, rc);
                cache_print_job("mid-segment cold", jc1);
                cache_print_verdict(j1, jc1);
                tally(j1, jc1);
                // the chunks themselves: the cached run's from c must be the cold run's tail (tokens alone can agree by luck)
                auto ends = [](const sched::job & j, int from) {
                    std::vector<int> b{from};
                    for (int x : j.chunks) b.push_back(b.back() + x);
                    return b;
                };
                std::vector<int> cold_b = ends(jc1, 0), hit_b = ends(j1, j1.n_cached);
                cold_b.erase(cold_b.begin(), std::lower_bound(cold_b.begin(), cold_b.end(), j1.n_cached));
                const bool chunks_ok = cold_b == hit_b;
                std::string cs;
                for (size_t i = 0; i < hit_b.size() && i < 4; ++i) cs += (i ? " | " : "") + std::to_string(hit_b[i]);
                printf("    mid-segment reuse: %d tokens from the cache (expected %d), split at %d, chunk ends %s ...: "
                       "%s\n", j1.n_cached, c, sp, cs.c_str(),
                       j1.n_cached != c ? "UNEXPECTED REUSE POINT" : chunks_ok ? "the cold run's chunks" : "CHUNKS DIFFER FROM COLD");
                if (j1.n_cached != c || !chunks_ok) n_ok -= cache_same(j1, jc1);   // counts as a failure
            } else {
                printf("    mid-segment reuse: skipped (needs --cache-a > %d and min split > 38)\n", lcp);
            }
        }
        printf("  cache: %d entries, %d checkpoints (%.0f MiB pinned), checkpoint evictions %d\n", S.n_entries(), S.n_ckpts(),
               S.ckpt_host_bytes() / 1048576.0, S.ckpt_evictions);
        fflush(stdout);
    }
    // ---- session file checks ----
    if (A.disk && want("file")) {
        printf("CACHE session file checks (%s):\n", cache_dir(A).c_str());
        const auto fc = cache_file_checks(m, d, so, doc, cache_dir(A), mtp);
        n_ok += fc.first == fc.second;
        ++n_cmp;
        printf("  file checks: %d/%d passed\n", fc.first, fc.second);
    }
    // ---- decoded-token reuse vs exact vs cold ----
    if (A.dturns > 0 && want("dec")) {
        const auto dr = cache_decoded_replay(m, d, so, A, doc);
        n_ok += dr.first == dr.second;
        ++n_cmp;
    }
    // ---- 2. multi-turn replay ----
    std::vector<int32_t> P;
    std::vector<int> splits;
    size_t off = 0;
    std::unique_ptr<sched> SP = std::make_unique<sched>(m, d, so);
    sched & S = *SP;
    int last_id = -1;
    double sum_c = 0, sum_cold = 0, sum_ttft_c = 0, sum_ttft_cold = 0;
    long long sum_cached = 0, sum_pf = 0, sum_cold_pf = 0;
    const int turns = want("replay") ? A.turns : 0;
    if (turns > 0)
        printf("CACHE multi-turn replay (%d turns: %d tokens, then + the answer + %d document tokens per turn, %d generated):\n",
               turns, A.turn0, A.step, A.cgen);
    for (int k = 0; k < turns; ++k) {
        if (k == 0) {
            P = slice(0, A.turn0);
            off = A.turn0;
        } else {
            const auto & prev = S.get(last_id);
            splits.push_back((int) P.size());                  // the answer starts a message
            P.insert(P.end(), prev.out.begin(), prev.out.end());
            splits.push_back((int) P.size());                  // the next user message
            const auto nx = slice(off, A.step);
            off += A.step;
            P.insert(P.end(), nx.begin(), nx.end());
        }
        if (A.offload_turns && k % 2 == 1) {
            // one short uncached job per slot: admission must move the idle session to host RAM
            const int nf = m.max_slots();
            for (int i = 0; i < nf; ++i) {
                sched::request f;
                f.prompt = slice((size_t) (97 * (i + 1)), 64);
                f.n_gen = 2;
                f.cache = false;
                S.submit(f);
            }
            S.run();
            printf("  (%d filler jobs: the idle session went to host RAM, offloads so far %d)\n", nf, S.evictions);
        }
        sched::request r;
        r.prompt = P;
        r.n_gen = A.cgen;
        r.mtp = mtp;
        r.splits = splits;
        const sched::job jc = cache_run1(S, r);
        last_id = S.n_jobs() - 1;
        char tag[32];
        snprintf(tag, sizeof tag, "turn %d cached", k + 1);
        cache_print_job(tag, jc);
        sched::request cr = r;
        cr.cache = false;
        const sched::job jo = cache_run1(S, cr);
        snprintf(tag, sizeof tag, "turn %d cold", k + 1);
        cache_print_job(tag, jo);
        cache_print_verdict(jc, jo);
        tally(jc, jo);
        sum_c += jc.t_done - jc.t_submit; sum_cold += jo.t_done - jo.t_submit;
        sum_ttft_c += jc.t_first - jc.t_submit; sum_ttft_cold += jo.t_first - jo.t_submit;
        sum_cached += jc.n_cached; sum_pf += jc.n_prefill; sum_cold_pf += jo.n_prefill;
        fflush(stdout);
    }
    if (turns > 0)
        printf("  REPLAY: %d turns, %lld prompt tokens from the cache, %lld prefilled (cold: %lld); time to first token %.2f s "
               "vs %.2f s cold (x%.2f), turn time %.2f s vs %.2f s (x%.2f)\n",
           turns, sum_cached, sum_pf, sum_cold_pf, sum_ttft_c, sum_ttft_cold, sum_ttft_cold / std::max(1e-9, sum_ttft_c),
           sum_c, sum_cold, sum_cold / std::max(1e-9, sum_c));
    if (turns > 0)
        printf("  cache: %d entries, %d checkpoints (%.0f MiB pinned), checkpoint evictions %d, offloads %d, drops %d\n",
               S.n_entries(), S.n_ckpts(), S.ckpt_host_bytes() / 1048576.0, S.ckpt_evictions, S.evictions, S.drops);
    fflush(stdout);
    // ---- 3. disk ----
    int rc_child = 0;
    if (A.disk && turns > 0) {
        const std::string base = cache_dir(A) + "p100e_cache_test_" + std::to_string((long long) time(nullptr));
        const std::string sess = base + ".p100esess", expect = base + ".expect.txt";
        const int n_ck = S.n_ckpts();
        auto a = clk::now();
        S.save(last_id, sess, A.save_ckpts);
        const double t_save = std::chrono::duration<double>(clk::now() - a).count();
        long long fsz = 0;
        {
            std::ifstream f(sess, std::ios::binary | std::ios::ate);
            fsz = (long long) f.tellg();
        }
        printf("CACHE disk: session of turn %d (%d positions, newest %d of %d checkpoints) saved in %.2f s (%.0f MiB, "
               "%.2f GB/s) -> %s\n", A.turns, m.seq_len(S.get(last_id).seq),
               A.save_ckpts < 0 ? n_ck : std::min(n_ck, A.save_ckpts), n_ck, t_save, fsz / 1048576.0,
               fsz / 1e9 / std::max(1e-9, t_save), sess.c_str());
        // the same session continued here: (a) the next turn without new input, (b) a new request on its prefix
        const int n2 = A.cgen;
        const int jcid = S.submit_continue(last_id, n2, mtp);
        S.run();
        const sched::job jcont = S.get(jcid);
        if (jcont.st != sched::job::DONE) throw std::runtime_error("cache test: continuation failed: " + jcont.err);
        sched::request r3;
        r3.prompt = P;
        const auto x = slice(off, std::min<size_t>(A.step, doc.size() - off));
        r3.prompt.insert(r3.prompt.end(), x.begin(), x.end());
        r3.splits = splits;
        r3.splits.push_back((int) P.size());
        r3.n_gen = A.cgen;
        r3.mtp = mtp;
        const sched::job j3 = cache_run1(S, r3);
        printf("  this process: continuation %zu tokens (accepted %d/%d), new request: cached %d, prefilled %d\n",
               jcont.out.size(), jcont.accepted, jcont.drafted, j3.n_cached, j3.n_prefill);
        {
            std::ofstream f(expect);
            f << "n2 " << n2 << "\n";
            write_ids(f, "cont", jcont.out);
            f << "cont_hash " << jcont.draft_hash << "\n";
            write_ids(f, "r3_prompt", r3.prompt);
            std::vector<int32_t> sp(r3.splits.begin(), r3.splits.end());
            write_ids(f, "r3_splits", sp);
            f << "r3_gen " << r3.n_gen << "\n";
            write_ids(f, "r3_out", j3.out);
            f << "r3_hash " << j3.draft_hash << "\n";
            f << "r3_cached " << j3.n_cached << "\n";
        }
        fflush(stdout);
        // free the GPUs, then the new process
        SP.reset();
        E.reset();
        for (int dv : A.S.devices) { cudaSetDevice(dv); cudaDeviceReset(); }
        std::vector<std::string> av = A.argv;
        av.push_back("--cache-child");
        av.push_back(sess);
        av.push_back(expect);
        std::string exe = av[0];
#ifdef _WIN32
        char mp[MAX_PATH];
        if (GetModuleFileNameA(nullptr, mp, MAX_PATH)) exe = mp;
        std::vector<std::string> q;
        for (auto & s : av) q.push_back(cache_quote(s));
        std::vector<const char *> cv;
        for (auto & s : q) cv.push_back(s.c_str());
        cv.push_back(nullptr);
        printf("CACHE disk: starting a new process: %s ... --cache-child\n", exe.c_str());
        fflush(stdout);
        rc_child = (int) _spawnv(_P_WAIT, exe.c_str(), cv.data());
#else
        throw std::runtime_error("--cache-test disk step: Windows only");
#endif
        printf("CACHE disk: new process exit code %d (%s)\n", rc_child, rc_child == 0 ? "IDENTICAL" : "FAILED");
        remove(sess.c_str());
        remove(expect.c_str());
        ++n_cmp;
        n_ok += rc_child == 0;
    }
#ifdef _WIN32
    if (!A.dir.empty()) RemoveDirectoryA(A.dir.c_str());   // --cache-dir: gone again when the test left it empty
#endif
    printf("CACHE identity: %d/%d cached runs identical to their cold runs%s\n", n_ok, n_cmp,
           A.disk && turns > 0 ? " (incl. the disk round trip in a new process)" : "");
    return n_ok == n_cmp ? 0 : 2;
}

static int run_cache_child(cache_args & A) {
    using clk = std::chrono::steady_clock;
    const bool mtp = A.S.mtp_k > 0;
    engine E(cache_engine_opts(A));
    tp_model & m = E.model();
    sched S(m, E.drafter(), cache_sched_opts(E));
    std::ifstream f(A.child_expect);
    if (!f) throw std::runtime_error("cannot open " + A.child_expect);
    std::string k;
    int n2 = 0, r3_gen = 0, r3_cached = 0;
    unsigned long long cont_hash = 0, r3_hash = 0;
    f >> k >> n2;
    const auto cont = read_ids(f, "cont");
    f >> k >> cont_hash;
    const auto r3p = read_ids(f, "r3_prompt");
    const auto r3s = read_ids(f, "r3_splits");
    f >> k >> r3_gen;
    const auto r3o = read_ids(f, "r3_out");
    f >> k >> r3_hash;
    f >> k >> r3_cached;
    auto a = clk::now();
    const int h = S.load(A.child_sess);
    const double t_load = std::chrono::duration<double>(clk::now() - a).count();
    const int seq = S.get(h).seq;
    printf("CHILD: session loaded in %.2f s: %d positions, %d checkpoints, %.0f MiB pinned host (offloaded)\n", t_load,
           m.seq_len(seq), S.n_ckpts(), m.host_bytes() / 1048576.0);
    const int jc = S.submit_continue(h, n2, mtp);
    S.run();
    const auto & c = S.get(jc);
    const bool s1 = c.st == sched::job::DONE && c.out == cont && (!mtp || c.draft_hash == cont_hash);
    printf("CHILD: continuation %zu tokens (restore %.1f ms, accepted %d/%d): %s\n", c.out.size(), 1e3 * c.t_restore,
           c.accepted, c.drafted, s1 ? "IDENTICAL" : "MISMATCH");
    sched::request r;
    r.prompt = r3p;
    r.splits.assign(r3s.begin(), r3s.end());
    r.n_gen = r3_gen;
    r.mtp = mtp;
    const int j3 = S.submit(r);
    S.run();
    const auto & q = S.get(j3);
    const bool s2 = q.st == sched::job::DONE && q.out == r3o && q.n_cached == r3_cached && (!mtp || q.draft_hash == r3_hash);
    cache_print_job("CHILD new request", q);
    printf("CHILD: new request on the loaded prefix: cached %d (expected %d), tokens %s\n", q.n_cached, r3_cached,
           s2 ? "IDENTICAL" : "MISMATCH");
    fflush(stdout);
    return s1 && s2 ? 0 : 2;
}

// ---- decode fairness ----
// A session D decodes (short prompt, long answer); then a long prompt P (--fair-test N tokens, default 20000) arrives.
// For each decode_share (--fair-shares, default 0 = one decode step per prefill chunk, the old rule, and 0.5): D's
// decode t/s alone and while P prefills, the share of that window spent in decode steps, P's prefill time vs alone.
// The chunks never change: P's tokens must be identical in every run, and D's tokens identical across runs.
static int run_fair_test(cache_args & A) {
    if (A.S.prompts.empty() || A.S.prompts[0].empty()) throw std::runtime_error("--fair-test needs a document (--tokens-file)");
    const std::vector<int32_t> & doc = A.S.prompts[0];
    const bool mtp = A.S.mtp_k > 0;
    engine E(cache_engine_opts(A));
    tp_model & m = E.model();
    tp_mtp * d = E.drafter();
    const sched_opts so = cache_sched_opts(E);
    auto make = [&](size_t start, int n) {
        std::vector<int32_t> p;
        while ((int) p.size() < n) p.push_back(doc[(start + p.size()) % doc.size()]);
        return p;
    };
    const std::vector<int32_t> pd = make(0, 512), pp = make(512, A.fair_prompt);
    const int gd = std::min(m.n_ctx() - 600, 12000), gp = 8;
    printf("FAIR test: session D (prompt %zu, decoding, %s) while prompt P (%zu tokens, ubatch %d) prefills; decode_share %s\n",
           pd.size(), mtp ? "MTP" : "plain", pp.size(), m.max_batch(), "compared below");
    // P alone
    double pf_alone = 0;
    std::vector<int32_t> out_p;
    {
        sched S(m, d, so);
        sched::request r;
        r.prompt = pp; r.n_gen = gp; r.mtp = mtp; r.cache = false;
        const auto & j = cache_run1(S, r);
        pf_alone = j.t_first - j.t_admit;
        out_p = j.out;
        printf("  P alone: prefill %.2f s (%.1f t/s)\n", pf_alone, pp.size() / std::max(1e-9, pf_alone));
    }
    bool same = true;
    std::vector<int32_t> out_d;
    for (double sh : A.fair_shares) {
        sched_opts s1 = so;
        s1.decode_share = sh;
        s1.decode_per_chunk = 1;
        sched S(m, d, s1);
        sched::request rd;
        rd.prompt = pd; rd.n_gen = gd; rd.mtp = mtp; rd.cache = false;
        const int jd = S.submit(rd);
        while (S.get(jd).out.size() < 32 && S.step()) {}
        // D alone for ~3 s
        double t0 = S.now();
        size_t n0 = S.get(jd).out.size();
        while (S.now() - t0 < 3.0 && S.step()) {}
        const double r_alone = (S.get(jd).out.size() - n0) / (S.now() - t0);
        // P arrives
        sched::request rp;
        rp.prompt = pp; rp.n_gen = gp; rp.mtp = mtp; rp.cache = false;
        const int jp = S.submit(rp);
        const double td0 = S.t_dec_wall, tw0 = S.now();
        const size_t nd0 = S.get(jd).out.size();
        while ((S.get(jp).st == sched::job::QUEUED || S.get(jp).st == sched::job::PREFILL) && S.step()) {}
        const double tw = S.now() - tw0, tdec = S.t_dec_wall - td0;
        const size_t nd = S.get(jd).out.size() - nd0;
        const auto & P = S.get(jp);
        const double pf = P.t_first - P.t_admit;
        while (S.get(jp).st != sched::job::DONE && S.get(jp).st != sched::job::FAILED && S.step()) {}
        same = same && S.get(jp).out == out_p;
        S.cancel(jd, true);
        const std::vector<int32_t> & od = S.get(jd).out;
        if (out_d.empty()) out_d = od;
        const size_t nc = std::min(out_d.size(), od.size());
        same = same && std::equal(od.begin(), od.begin() + nc, out_d.begin());
        if (od.size() > out_d.size()) out_d = od;
        printf("  decode_share %.2f: D alone %.1f t/s; while P prefills (%.1f s): D %zu tokens = %.1f t/s, decode steps %.0f %% of "
               "the wall time; P prefill %.2f s = x%.2f of alone; P tokens %s\n", sh, r_alone, tw, nd, nd / std::max(1e-9, tw),
               100.0 * tdec / std::max(1e-9, tw), pf, pf / std::max(1e-9, pf_alone), S.get(jp).out == out_p ? "identical" : "DIFFER");
        fflush(stdout);
    }
    printf("FAIR: P's tokens identical in every run and D's tokens identical across runs: %s\n", same ? "yes" : "NO");
    return same ? 0 : 2;
}

// ---- scheduler regressions ----
static int run_sched_test(cache_args & A) {
    if (A.S.prompts.empty()) throw std::runtime_error("--sched-test needs prompts");
    const bool mtp = A.S.mtp_k > 0;
    engine_opts eo = cache_engine_opts(A);
    eo.pool_tokens = A.sched_pool;   // small: the tests need jobs that exceed / fill the pool within the context
    engine E(eo);
    tp_model & m = E.model();
    tp_mtp * d = E.drafter();
    sched_opts so = cache_sched_opts(E);
    if (m.pool_tokens() > m.n_ctx() - 128) throw std::runtime_error("--sched-test: needs --sched-pool below --ctx");
    const int ps = m.page_size();
    int n_pass = 0, n_test = 0;
    auto check = [&](const char * name, bool ok, const std::string & detail = "") {
        ++n_test;
        n_pass += ok;
        printf("  %-64s %s%s%s\n", name, ok ? "PASS" : "FAIL", detail.empty() ? "" : "  ", detail.c_str());
        fflush(stdout);
    };
    auto prompt_of = [&](int i, int n) {   // distinct prompts: the given ones rotated by 997 i
        std::vector<int32_t> p = A.S.prompts[i % A.S.prompts.size()];
        std::rotate(p.begin(), p.begin() + (997 * (size_t) i) % p.size(), p.end());
        while ((int) p.size() < n) p.insert(p.end(), p.begin(), p.end());
        p.resize(n);
        return p;
    };
    printf("SCHED tests (pool %d pages of %d, %d slots, %s):\n", m.pool_pages(), ps, m.max_slots(), mtp ? "MTP" : "plain");
    // references alone
    auto ref_of = [&](const std::vector<int32_t> & p, int n, bool mt) {
        sched S(m, d, so);
        sched::request r;
        r.prompt = p; r.n_gen = n; r.mtp = mt; r.cache = false;
        return cache_run1(S, r).out;
    };
    // a job larger than the pool is rejected at submit
    {
        sched S(m, d, so);
        bool threw = false;
        std::string msg;
        try {
            const int n = std::min(m.n_ctx() - 64, (int) m.pool_tokens() + ps);
            S.submit(prompt_of(0, 16), n, false);
        } catch (const std::exception & e) { threw = true; msg = e.what(); }
        check("job larger than the KV pool rejected at submit", threw, msg);
    }
    // a continuation submitted while a running job holds a large reservation must wait, not starve it
    {
        const int big = std::max(64, (int) (m.pool_tokens() * 0.70));
        const int g1 = 8, g2 = 24;
        const auto pa = prompt_of(1, std::min(big, m.n_ctx() - g2 - 64));
        const auto ps1 = prompt_of(2, std::max(16, (int) (m.pool_tokens() * 0.35)));
        const auto ra = ref_of(pa, g2, false);
        const auto rs = ref_of(ps1, g1 + g2, mtp);
        sched S(m, d, so);
        sched::request r;
        r.prompt = ps1; r.n_gen = g1; r.mtp = mtp; r.keep = true;
        const int hs = cache_run1(S, r).id;
        // held session S idle + offloaded by a big job A that needs its pages, then S's next turn while A runs
        sched::request ra_;
        ra_.prompt = pa; ra_.n_gen = g2; ra_.cache = false;
        const int ja = S.submit(ra_);
        int guard = 0;
        while (S.get(ja).st == sched::job::QUEUED && guard++ < 100000) S.step();   // A admitted (S offloaded), prefilling
        bool threw = false;
        std::string msg;
        int jc = -1;
        try {
            jc = S.submit_continue(hs, g2, mtp);
            S.run();
        } catch (const std::exception & e) { threw = true; msg = e.what(); }
        bool ok = !threw && S.get(ja).st == sched::job::DONE && jc >= 0 && S.get(jc).st == sched::job::DONE;
        std::vector<int32_t> all;
        if (ok) {
            all = S.get(hs).out;
            all.insert(all.end(), S.get(jc).out.begin(), S.get(jc).out.end());
            ok = S.get(ja).out == ra && all == rs;
        }
        check("continuation under a running job's reservation waits", ok,
              threw ? msg : (ok ? "both identical to their runs alone" : "MISMATCH"));
        if (jc >= 0 && mtp) {
            const auto & c = S.get(jc);
            check("next turn of a kept MTP session drafts from its first step", c.first_rows > 1,
                  "first step " + std::to_string(c.first_rows) + " rows, accepted " + std::to_string(c.accepted) + "/" +
                  std::to_string(c.drafted) + " in " + std::to_string(c.steps) + " steps");
        }
    }
    // a continuation queued BEHIND a job that needs the session's pages: the session is offloaded for that job (its
    // reservation does not pin it to the GPU), the continuation restores it afterwards
    {
        const int g1 = 6, g2 = 12;
        const auto pa = prompt_of(13, std::max(64, (int) (m.pool_tokens() * 0.70)));
        const auto ps1 = prompt_of(14, std::max(16, (int) (m.pool_tokens() * 0.35)));
        const auto ra = ref_of(pa, g2, false);
        const auto rs = ref_of(ps1, g1 + g2, mtp);
        sched S(m, d, so);
        sched::request r;
        r.prompt = ps1; r.n_gen = g1; r.mtp = mtp; r.keep = true;
        const int hs = cache_run1(S, r).id;
        sched::request rx;
        rx.prompt = pa; rx.n_gen = g2; rx.cache = false;
        const int jx = S.submit(rx);
        const int jc = S.submit_continue(hs, g2, mtp);   // queued after jx
        bool threw = false;
        std::string msg;
        try { S.run(); } catch (const std::exception & e) { threw = true; msg = e.what(); }
        bool ok = !threw && S.get(jx).st == sched::job::DONE && S.get(jc).st == sched::job::DONE;
        if (ok) {
            std::vector<int32_t> all = S.get(hs).out;
            all.insert(all.end(), S.get(jc).out.begin(), S.get(jc).out.end());
            ok = S.get(jx).out == ra && all == rs;
        }
        check("queued continuation does not pin its session against an earlier job", ok,
              threw ? msg : S.get(jx).st == sched::job::FAILED ? S.get(jx).err : (ok ? "both identical" : "MISMATCH"));
    }
    // release() frees a finished session; later admissions never touch its (reused) sequence id
    {
        sched S(m, d, so);
        sched::request r;
        r.prompt = prompt_of(3, 40); r.n_gen = 6; r.keep = true;
        const int h = cache_run1(S, r).id;
        const int seq = S.get(h).seq;
        S.release(h);
        bool ok = S.get(h).seq == -1 && !m.seq_resident(seq) && m.free_slots() == m.max_slots();
        std::string msg;
        try {
            for (int i = 0; i < m.max_slots() + 1; ++i) S.submit(prompt_of(4 + i, 40), 4, false);
            S.run();
            bool cont_threw = false;
            try { S.submit_continue(h, 4, false); } catch (const std::exception &) { cont_threw = true; }
            ok = ok && cont_threw;
        } catch (const std::exception & e) { ok = false; msg = e.what(); }
        check("release() frees the session; no stale sequence id afterwards", ok, msg);
    }
    // offload failures (host budget) during admission do not escape step(): cache entries are dropped instead
    {
        sched S(m, d, so);
        const size_t old = m.host_budget();
        m.set_host_budget(1);   // nothing can be offloaded
        bool ok = true;
        std::string msg;
        try {
            for (int i = 0; i < m.max_slots() + 2; ++i) {
                sched::request r;
                r.prompt = prompt_of(5 + i, 48); r.n_gen = 4;
                cache_run1(S, r);   // each leaves a cache entry; admission must drop old ones
            }
        } catch (const std::exception & e) { ok = false; msg = e.what(); }
        m.set_host_budget(old);
        check("host budget exhausted: admission drops cache entries, no exception", ok && S.drops > 0,
              msg.empty() ? "drops " + std::to_string(S.drops) : msg);
    }
    // a held session that cannot be offloaded and a job that needs its slot: the job fails cleanly when nothing else runs
    {
        sched_opts s1 = so;
        sched S(m, d, s1);
        const size_t old = m.host_budget();
        m.set_host_budget(1);
        std::vector<int> held;
        for (int i = 0; i < m.max_slots(); ++i) {
            sched::request r;
            r.prompt = prompt_of(6 + i, 32); r.n_gen = 2; r.keep = true;
            held.push_back(cache_run1(S, r).id);
        }
        sched::request r;
        r.prompt = prompt_of(9, 32); r.n_gen = 2;
        const int j = S.submit(r);
        bool ok = true;
        try { S.run(); } catch (...) { ok = false; }
        m.set_host_budget(old);
        check("no slot for a job behind non-offloadable held sessions: FAILED, no hang", ok && S.get(j).st == sched::job::FAILED,
              S.get(j).err);
        for (int h : held) S.release(h);
    }
    // resume of a preempted job goes through admission: it waits for a slot instead of throwing
    {
        const int g = 16;
        const auto p0 = prompt_of(10, 48), p1 = prompt_of(11, 48);
        const auto r0 = ref_of(p0, g, mtp);
        sched_opts s1 = so;
        s1.max_active = m.max_slots();
        sched S(m, d, s1);
        const int j0 = S.submit(p0, g, mtp);
        while (S.get(j0).out.size() < 4 && S.step()) {}
        S.preempt(j0);
        std::vector<int> fill;
        for (int i = 0; i < m.max_slots(); ++i) fill.push_back(S.submit(prompt_of(12 + i, 48), 24, false));
        S.step();   // the fillers take every slot
        bool threw = false;
        std::string msg;
        try { S.resume(j0); S.run(); } catch (const std::exception & e) { threw = true; msg = e.what(); }
        check("resume() waits for a slot taken meanwhile", !threw && S.get(j0).out == r0,
              threw ? msg : (S.get(j0).out == r0 ? "identical" : "MISMATCH"));
    }
    // release() of a session whose next turn is still queued fails that turn ('session released')
    // instead of wedging admission (the queued continuation used to read sequence -1 inside step())
    {
        const auto pa = prompt_of(16, std::max(64, (int) (m.pool_tokens() * 0.70)));
        sched S(m, d, so);
        sched::request r;
        r.prompt = prompt_of(17, 48); r.n_gen = 4; r.keep = true;
        const int hs = cache_run1(S, r).id;
        sched::request rx;
        rx.prompt = pa; rx.n_gen = 4; rx.cache = false;
        const int jx = S.submit(rx);                      // ahead of the continuation
        const int jc = S.submit_continue(hs, 8, mtp);     // queued behind jx
        bool threw = false;
        std::string msg;
        try {
            S.release(hs);                                // while jc is still queued
            S.run();
            S.submit(prompt_of(18, 48), 4, false);        // admission still works afterwards
            S.run();
        } catch (const std::exception & e) { threw = true; msg = e.what(); }
        const auto & c = S.get(jc);
        const bool ok = !threw && c.st == sched::job::FAILED && c.err == "session released" &&
                        S.get(jx).st == sched::job::DONE && S.get(S.n_jobs() - 1).st == sched::job::DONE &&
                        S.n_entries() == 1;   // only the last job's cache entry
        check("release() with the next turn queued fails that turn, admission goes on", ok,
              threw ? msg : c.err + ", entries left " + std::to_string(S.n_entries()));
    }
    // a preempted job whose resume can never be admitted fails AND gives back its host copy / sequence
    {
        sched_opts s1 = so;
        s1.max_active = m.max_slots();
        sched S(m, d, s1);
        const size_t host0 = m.host_bytes(), old = m.host_budget();
        const int j0 = S.submit(prompt_of(19, 48), 24, mtp);
        while (S.get(j0).out.size() < 4 && S.step()) {}
        S.preempt(j0);
        const size_t host1 = m.host_bytes();
        std::vector<int> held;   // every slot taken by an idle held session
        for (int i = 0; i < m.max_slots(); ++i) {
            sched::request r;
            r.prompt = prompt_of(20 + i, 32); r.n_gen = 2; r.keep = true;
            held.push_back(cache_run1(S, r).id);
        }
        m.set_host_budget(m.host_bytes());   // nothing more fits in host RAM: the held sessions cannot move
        bool threw = false;
        std::string msg;
        try { S.resume(j0); S.run(); } catch (const std::exception & e) { threw = true; msg = e.what(); }
        m.set_host_budget(old);
        const auto & j = S.get(j0);
        const bool ok = !threw && j.st == sched::job::FAILED && j.seq < 0 && m.host_bytes() == host0;
        check("a resume that can never be admitted fails and frees its host copy", ok,
              threw ? msg : "host copy " + std::to_string(host1 >> 20) + " MiB -> " + std::to_string((m.host_bytes() - host0) >> 20) +
              " MiB after the failure; " + j.err.substr(0, 40));
        for (int h : held) S.release(h);
    }
    // offloaded sequences and checkpoints share the model's pinned budget; a new checkpoint takes
    // its room back from an offloaded cache-only entry; freed checkpoint buffers beyond ckpt_spare are given back
    {
        const size_t old = m.host_budget(), extra0 = m.host_extra();
        bool ok = true;
        std::string msg;
        {
            sched S(m, d, so);
            sched::request ra;
            ra.prompt = prompt_of(24, 96); ra.n_gen = 4; ra.mtp = mtp;
            const int ja = cache_run1(S, ra).id;
            const int sa = S.get(ja).seq;
            const size_t ck = S.ckpt_bytes();
            const size_t a_off = ((size_t) m.seq_pages(sa) * m.page_bytes() + m.state_bytes()) * m.n_ranks();
            const int ck_a = S.n_ckpts();
            // room for A's host copy and the checkpoints so far, not for one more checkpoint
            m.set_host_budget(m.host_used() + a_off + ck / 2);
            for (int i = 0; i < m.max_slots(); ++i) {   // fillers (no cache): A goes to host RAM
                sched::request f;
                f.prompt = prompt_of(25 + i, 40); f.n_gen = 2; f.cache = false;
                S.submit(f);
            }
            S.run();
            const bool a_off_ok = S.get(ja).offloaded && m.host_used() <= m.host_budget();
            const int drops0 = S.drops;
            sched::request rb;
            rb.prompt = prompt_of(30, 96); rb.n_gen = 4; rb.mtp = mtp;
            const auto & jb = cache_run1(S, rb);
            const bool b_ok = jb.ckpts >= 1 && S.drops == drops0 + 1 && S.get(ja).seq < 0 && m.host_used() <= m.host_budget();
            m.set_host_budget(old);
            // free list: dropping every entry gives the buffers back down to ckpt_spare
            S.release(jb.id);
            const size_t spare_max = (size_t) std::max(0, so.ckpt_spare) * ck;
            const bool shrink_ok = S.n_ckpts() == 0 && S.ckpt_host_bytes() <= spare_max;
            ok = a_off_ok && b_ok && shrink_ok;
            msg = "A offloaded " + std::string(a_off_ok ? "within" : "OVER") + " the budget, B's checkpoint " +
                  (b_ok ? "dropped A" : "DID NOT TAKE A's ROOM") + " (ckpts A " + std::to_string(ck_a) + ", B " +
                  std::to_string(jb.ckpts) + "), spare pinned " + std::to_string(S.ckpt_host_bytes() >> 20) + " MiB" +
                  (shrink_ok ? "" : " (MORE THAN ckpt_spare)");
        }
        const bool back = m.host_extra() == extra0;
        check("offload + checkpoints share one pinned budget, spares shrink", ok && back,
              msg + (back ? "" : ", host_extra NOT BACK after the scheduler"));
    }
    // ---- server API ----
    // (a) next turn with new input == the same conversation as a new request reusing the decoded tokens (inexact
    // lookup at the entry's current state: the same chunks from the same state), plain and MTP; MTP == plain
    {
        const auto p0 = prompt_of(31, 80), x = prompt_of(32, 40);
        const int g1 = 6, g2 = 10;
        sched_opts si = so;
        si.exact = false;
        auto turn2_append = [&](bool mt, bool keep) {
            sched S(m, d, si);
            sched::request r;
            r.prompt = p0; r.n_gen = g1; r.mtp = mt; r.keep = keep;
            const int h = cache_run1(S, r).id;
            sched::request c;
            c.prompt = x; c.n_gen = g2; c.mtp = mt; c.keep = keep;
            c.splits = {(int) (p0.size() + g1)};   // the new message starts after the answer's last token
            const int jc = S.submit_continue(h, c);
            S.run();
            const auto & j = S.get(jc);
            if (j.st != sched::job::DONE) throw std::runtime_error("append continuation failed: " + j.err);
            return std::make_pair(S.get(h).out, j);
        };
        const auto a_plain = turn2_append(false, true);
        std::vector<int32_t> conv = p0;
        conv.insert(conv.end(), a_plain.first.begin(), a_plain.first.end());
        conv.insert(conv.end(), x.begin(), x.end());
        std::vector<int32_t> viaq;
        int q_cached = -1;
        {
            sched S(m, d, si);
            sched::request r;
            r.prompt = p0; r.n_gen = g1;
            cache_run1(S, r);   // a cache entry: prompt + the answer's decoded tokens
            sched::request q;
            q.prompt = conv; q.n_gen = g2; q.splits = {(int) (p0.size() + g1)};
            const auto & jq = cache_run1(S, q);
            viaq = jq.out;
            q_cached = jq.n_cached;
        }
        const auto & ja = a_plain.second;
        bool ok = ja.out == viaq && ja.n_cached == (int) (p0.size() + g1 - 1) && q_cached == ja.n_cached &&
                  ja.n_prefill == (int) x.size() + 1;
        std::string det = "cached " + std::to_string(ja.n_cached) + ", prefilled " + std::to_string(ja.n_prefill) +
                          "; new request with decoded reuse: cached " + std::to_string(q_cached) + ", tokens " +
                          (ja.out == viaq ? "identical" : "DIFFER");
        if (mtp) {
            const auto a_mtp = turn2_append(true, true);
            const bool same = a_mtp.first == a_plain.first && a_mtp.second.out == ja.out;
            ok = ok && same;
            det += std::string("; MTP turn 2 ") + (same ? "== plain" : "DIFFERS FROM PLAIN");
        }
        check("next turn with new input (submit_continue + input) == decoded-token reuse", ok, det);
    }
    // (b) stop tokens: generation ends at the first stop token (inside an MTP step: rows rolled back to it); the
    // session then continues exactly as the uninterrupted run
    {
        const auto p0 = prompt_of(33, 64);
        const int G = 24;
        const auto ref = ref_of(p0, G, false);
        // the stop token: the first one from index 7 on that did not occur before (so the turn runs several steps)
        int k1 = 7;
        while (k1 < G - 3 && std::find(ref.begin(), ref.begin() + k1, ref[k1]) != ref.begin() + k1) ++k1;
        const int32_t stop_tok = ref[k1];
        k1 = (int) (std::find(ref.begin(), ref.end(), stop_tok) - ref.begin());
        sched S(m, d, so);
        sched::request r;
        r.prompt = p0; r.n_gen = G; r.mtp = mtp; r.keep = true; r.stop = {stop_tok};
        const int h = S.submit(r);
        if (mtp) S.set_oracle(h, &ref);   // drafts accepted: the stop can fall inside a multi-row step
        S.run();
        const auto & j = S.get(h);
        const bool cut = j.stopped && (int) j.out.size() == k1 + 1 &&
                         std::equal(j.out.begin(), j.out.end(), ref.begin());
        sched::request c;
        c.n_gen = G - (int) j.out.size(); c.mtp = mtp; c.keep = true;   // no stop: the rest of the reference
        const int jc = S.submit_continue(h, c);
        if (mtp) S.set_oracle(jc, nullptr);
        S.run();
        std::vector<int32_t> all = j.out;
        all.insert(all.end(), S.get(jc).out.begin(), S.get(jc).out.end());
        const bool ok = cut && all == ref;
        check("stop token ends the turn, the session continues exactly", ok,
              "stopped after " + std::to_string(j.out.size()) + " tokens (expected " + std::to_string(k1 + 1) + "), " +
              std::to_string(j.steps) + " steps, accepted " + std::to_string(j.accepted) + "/" + std::to_string(j.drafted) +
              "; continued: " + (all == ref ? "identical" : "MISMATCH"));
    }
    // (c) cancel: queued -> FAILED; decoding -> DONE at a step boundary and the next turn continues exactly (MTP: drafts
    // from its first step); prefilling -> the prefilled part is reused by the next request; drop frees it
    {
        const auto p0 = prompt_of(34, 64), p1 = prompt_of(35, 64);
        const int G = 24;
        const auto ref0 = ref_of(p0, G, false);
        sched_opts s1 = so;
        s1.max_active = 1;
        sched S(m, d, s1);
        sched::request r;
        r.prompt = p0; r.n_gen = G; r.mtp = mtp; r.keep = true;
        const int j0 = S.submit(r);
        if (mtp) S.set_oracle(j0, &ref0);
        const int jq = S.submit(p1, 8, false);   // waits (max_active 1)
        while (S.get(j0).out.size() < 9 && S.step()) {}
        S.cancel(jq);
        const bool q_ok = S.get(jq).st == sched::job::FAILED && S.get(jq).err == "cancelled";
        S.cancel(j0);
        const auto & a = S.get(j0);
        const bool d_ok = a.st == sched::job::DONE && a.cancelled && a.seq >= 0;
        sched::request c;
        c.n_gen = G - (int) a.out.size(); c.mtp = mtp; c.keep = true;
        const int jc = S.submit_continue(j0, c);
        if (mtp) S.set_oracle(jc, &ref0);
        S.run();
        std::vector<int32_t> all = a.out;
        all.insert(all.end(), S.get(jc).out.begin(), S.get(jc).out.end());
        const bool c_ok = all == ref0 && (!mtp || S.get(jc).first_rows > 1);
        // prefill: a prompt of at least two chunks, cancelled after its first chunk (the step that admits it runs one);
        // the same request afterwards reuses the prefilled part
        const int mb = m.max_batch();
        const auto p2 = prompt_of(36, std::min(mb + 600, (int) m.pool_tokens() - 512));
        const auto ref2 = ref_of(p2, 6, false);
        const int n_ent0 = S.n_entries();
        const int jp = S.submit(p2, 6, false);
        while (S.get(jp).st == sched::job::QUEUED && S.step()) {}
        S.cancel(jp);
        const bool p_kept = S.get(jp).st == sched::job::FAILED && S.n_entries() == n_ent0 + 1;
        sched::request rr;
        rr.prompt = p2; rr.n_gen = 6;
        const auto & jr = cache_run1(S, rr);
        const bool p_ok = p_kept && jr.n_cached > 0 && jr.out == ref2;
        // drop
        const int jd = S.submit(prompt_of(37, 64), 12, false);
        while (S.get(jd).out.size() < 3 && S.step()) {}
        const int n_ent1 = S.n_entries();
        S.cancel(jd, true);
        const bool drop_ok = S.get(jd).st == sched::job::DONE && S.get(jd).seq < 0 && S.n_entries() == n_ent1 - 1;
        check("cancel: queued / decoding (+ exact next turn) / prefilling (reused) / drop",
              q_ok && d_ok && c_ok && p_ok && drop_ok,
              std::string("queued ") + (q_ok ? "ok" : "BAD") + ", decode " + (d_ok ? "ok" : "BAD") + " after " +
              std::to_string(a.out.size()) + " tokens, next turn " + (all == ref0 ? "identical" : "MISMATCH") +
              (mtp ? " (first step " + std::to_string(S.get(jc).first_rows) + " rows)" : "") + ", prefill reused " +
              std::to_string(jr.n_cached) + (jr.out == ref2 ? " identical" : " MISMATCH") + ", drop " +
              (drop_ok ? "ok" : "BAD"));
    }
    // (c2) cancel() during the PREFILL of an append continuation gives the held session back
    // to its finished job at the turn's end (a later continuation == the same turn without the cancelled one), and
    // submit_continue(done, request) inherits keep when the request does not set it
    {
        const auto p0 = prompt_of(38, 64), xs = prompt_of(39, 24);
        const int mb = m.max_batch();
        const auto xl = prompt_of(40, std::min(mb + 300, (int) m.pool_tokens() / 2));
        const int g1 = 6, g2 = 8;
        auto first_turn = [&](sched & S) {
            sched::request r;
            r.prompt = p0; r.n_gen = g1; r.keep = true;
            return cache_run1(S, r).id;
        };
        std::vector<int32_t> ref;
        {
            sched S(m, d, so);
            const int h = first_turn(S);
            sched::request c;
            c.prompt = xs; c.n_gen = g2; c.keep = true; c.keep_set = true;
            const int jc = S.submit_continue(h, c);
            S.run();
            ref = S.get(jc).out;
            S.release(jc);
        }
        sched S(m, d, so);
        const int h = first_turn(S);
        sched::request cl;
        cl.prompt = xl; cl.n_gen = 4;   // keep not set: inherited
        const int jl = S.submit_continue(h, cl);
        const bool inh = S.get(jl).keep_seq;
        while (S.get(jl).st == sched::job::QUEUED && S.step()) {}
        const bool in_pf = S.get(jl).st == sched::job::PREFILL;
        S.cancel(jl);
        const bool back = S.get(h).seq >= 0;
        sched::request c;
        c.prompt = xs; c.n_gen = g2;
        int jc = -1;
        std::vector<int32_t> got;
        std::string err;
        try {
            jc = S.submit_continue(h, c);
            S.run();
            got = S.get(jc).out;
            if (S.get(jc).st != sched::job::DONE) err = S.get(jc).err;
        } catch (const std::exception & e) { err = e.what(); }
        const bool ok = inh && in_pf && back && got == ref && err.empty();
        check("cancel of an append continuation's prefill returns the session; continue inherits keep", ok,
              std::string("keep inherited ") + (inh ? "yes" : "NO") + ", cancelled in prefill " + (in_pf ? "yes" : "NO") +
              ", session back " + (back ? "yes" : "NO") + ", next turn " + (got == ref ? "identical" : "MISMATCH " + err));
        if (jc >= 0 && S.get(jc).st == sched::job::DONE) S.release(jc);
    }
    // (d) get() references stay valid across submits (deque); forget() shrinks the record, the sequence stays as an
    // anonymous cache entry that the next request reuses
    {
        sched S(m, d, so);
        const auto p0 = prompt_of(38, 72);
        const auto ref = ref_of(p0, 6, false);
        const int h = S.submit(p0, 6, false);
        S.run();
        const sched::job & rj = S.get(h);
        const sched::job * addr = &rj;
        for (int i = 0; i < 64; ++i) {
            sched::request f;
            f.prompt = prompt_of(39 + i, 16); f.n_gen = 1; f.cache = false;
            S.submit(f);
        }
        S.run();
        const bool stable = &S.get(h) == addr && rj.out == ref;
        S.forget(h);
        const bool shrunk = S.get(h).out.empty() && S.get(h).prompt.empty() && S.get(h).seq < 0;
        sched::request r;
        r.prompt = p0;
        r.prompt.insert(r.prompt.end(), ref.begin(), ref.end() - 1);
        r.splits = {(int) p0.size()};   // the answer starts a message: its checkpoint is reusable exactly
        r.n_gen = 4;
        const auto & jr = cache_run1(S, r);
        const bool reused = jr.n_cached > 0;
        check("get() references survive submits; forget() keeps the sequence cached", stable && shrunk && reused,
              std::string(stable ? "stable" : "MOVED") + ", " + (shrunk ? "record freed" : "NOT FREED") + ", next request reused " +
              std::to_string(jr.n_cached));
    }
    printf("SCHED: %d/%d tests passed\n", n_pass, n_test);
    return n_pass == n_test ? 0 : 2;
}
