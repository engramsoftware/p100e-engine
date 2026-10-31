// p100e-run --sessions / --offload-test, included by p100e_run.cpp.
//
// --sessions n1[,n2..]: for every prompt (derived by rotation when fewer than the largest n are given) a plain greedy
// reference run ALONE through the scheduler; then for each mode (plain, and MTP with --mtp K [--mtp-adaptive tau]) and
// each session count n: the first n prompts as n concurrent jobs (prefill chunks interleaved with batched decode steps;
// --sessions-sync: all prefills first). Prints per-session decode t/s, the aggregate, rows per step, and an IDENTITY
// check of every session's tokens against its plain run alone.
// --offload-test n1: job A generates n1 tokens, is offloaded to host RAM, job B (kept resident) runs to the end, A is
// restored into another slot / other pages and finishes: both must equal their runs alone.
// --sessions-pool N|auto (KV pool tokens; default: largest n * --ctx), --reserve-mb a,b (auto: VRAM kept free per GPU,
// default 0,1024 = the vision encoder on GPU 1), --max-active N, --draft-rows N (rows per step with MTP drafts, default 4).
#include "../src/engine.h"

struct sessions_args {
    std::string model;
    std::vector<int> devices;
    int ctx = 4096, ubatch = 0, gen = 32;
    uint32_t kv_type = fast::KV_Q8_0;
    std::vector<std::vector<int32_t>> prompts;
    std::vector<int> counts;
    int mtp_k = 0;          // 0: plain only
    float tau = 0.0f;
    int kmin = 1;
    bool sync = false;
    int offload_n = -1;     // --offload-test
    int64_t pool = 0;       // 0: default, < 0: auto
    std::vector<size_t> reserve_mb{0, 1024};
    int max_active = 0;
    int draft_rows = 4;     // --draft-rows (sched_opts::draft_rows)
    int decode_per_chunk = 1;   // --decode-per-chunk (sched_opts::decode_per_chunk)
    double decode_share = 0.5;  // --decode-share (sched_opts::decode_share)
};

static int run_sessions(sessions_args & A) {
    using clk = std::chrono::steady_clock;
    const auto t0 = clk::now();
    int nmax = 1;
    for (int n : A.counts) nmax = std::max(nmax, n);
    if (A.offload_n >= 0) nmax = std::max(nmax, 2);
    if (A.prompts.empty()) throw std::runtime_error("--sessions needs prompts (--tokens / --tokens-file(s))");
    // more sessions than prompts: rotate the given prompts (a different start, the same length)
    const size_t ng = A.prompts.size();
    for (size_t i = ng; i < (size_t) nmax; ++i) {
        std::vector<int32_t> p = A.prompts[i % ng];
        const size_t sh = (997 * (i / ng)) % p.size();
        std::rotate(p.begin(), p.begin() + sh, p.end());
        A.prompts.push_back(p);
    }
    int maxp = 0;
    for (auto & p : A.prompts) maxp = std::max<int>(maxp, (int) p.size());
    // the engine object a server holds (src/engine.h): model, draft head, snapshots, prefill buffers, then the pool
    engine_opts eo;
    eo.model = A.model;
    eo.devices = A.devices;
    eo.kv_type = A.kv_type;
    eo.n_ctx = A.ctx;
    eo.ubatch = A.ubatch > 0 ? std::max(A.ubatch, 16) : std::max(64, maxp);
    eo.max_seqs = nmax + (A.offload_n >= 0 ? 1 : 0);
    eo.mtp = A.mtp_k > 0;
    eo.mo.k = A.mtp_k > 0 ? A.mtp_k : 3;
    eo.mo.tau = A.tau;
    eo.mo.k_min = A.kmin;
    eo.pool_tokens = A.pool < 0 ? -1 : A.pool > 0 ? A.pool : (int64_t) A.ctx * eo.max_seqs;
    eo.reserve_mb = A.reserve_mb;
    engine E(eo);
    tp_model & m = E.model();
    tp_mtp * d = E.drafter();
    sched_opts so;
    so.max_active = A.max_active > 0 ? A.max_active : nmax;
    so.prefill_first = A.sync;
    so.draft_rows = A.draft_rows;
    so.decode_per_chunk = A.decode_per_chunk;
    so.decode_share = A.decode_share;
    so.mo = eo.mo;
    printf("load %.2f s; sessions: max %d, pool %lld tokens (%d pages), page %d, ubatch %d, %s\n",
           std::chrono::duration<double>(clk::now() - t0).count(), nmax, (long long) m.pool_tokens(), m.pool_pages(),
           m.page_size(), m.max_batch(), A.sync ? "prefill first" : "prefill chunks interleaved with decode steps");
    m.print_mem("after load");
    fflush(stdout);
    auto run_jobs = [&](const std::vector<int> & which, bool mtp, const char * tag, std::vector<std::vector<int32_t>> * outs,
                        const std::vector<std::vector<int32_t>> * ref, int * n_same) {
        sched S(m, d, so);
        std::vector<int> ids;
        static const bool oracle = getenv("P100E_SESS_ORACLE") && atoi(getenv("P100E_SESS_ORACLE")) != 0;
        for (int i : which) {
            ids.push_back(S.submit(A.prompts[i], A.gen, mtp));
            if (oracle && mtp && ref) S.set_oracle(ids.back(), &(*ref)[i]);   // test only: drafts = the expected tokens
        }
        S.run();
        double tf = 1e30, td = 0, tp = 0;
        long long ntok = 0;
        for (size_t k = 0; k < ids.size(); ++k) {
            const auto & j = S.get(ids[k]);
            const double t0d = j.t_dec0 >= 0 ? j.t_dec0 : j.t_done;   // its first decode step
            tf = std::min(tf, t0d);
            td = std::max(td, j.t_done);
            tp += j.t_prefill;
            ntok += (long long) j.out.size() - 1;
            const double dt = j.t_done - t0d;
            bool same = true;
            if (ref) {
                same = j.out == (*ref)[which[k]];
                if (n_same) *n_same += same;
            }
            printf("  %s s%d: prompt %zu tokens, prefill %.2f s (+ catch-up %.0f ms), first token at %.2f s, %zu tokens, "
                   "decode %.2f t/s",
                   tag, which[k], A.prompts[which[k]].size(), j.t_prefill, 1e3 * j.t_catchup, j.t_first, j.out.size(),
                   dt > 0 ? (j.out.size() - 1) / dt : 0.0);
            if (mtp) printf(", accepted %d/%d, %d verify steps, drafts hash %016llx", j.accepted, j.drafted, j.steps,
                            (unsigned long long) j.draft_hash);
            if (ref) printf("  %s", same ? "IDENTICAL" : "MISMATCH");
            printf("\n");
            if (ref && !same) {
                const auto & r = (*ref)[which[k]];
                for (size_t i = 0; i < std::min(r.size(), j.out.size()); ++i)
                    if (r[i] != j.out[i]) { printf("    first diff at %zu: %d vs ref %d\n", i, j.out[i], r[i]); break; }
            }
            if (outs) (*outs)[which[k]] = j.out;
        }
        // decode wall time: from the first decode step to the last token (interleaved prefill chunks of other jobs
        // inside it count, as they do for a server's clients)
        printf("  %s AGGREGATE: %zu sessions, %lld tokens in %.3f s of decode wall time = %.2f t/s (whole run %.2f s); "
               "%d batched steps (%.2f ms/step, rows/step:", tag, ids.size(), ntok, td - tf, ntok / std::max(1e-9, td - tf),
               td, S.decode_steps, 1e3 * S.t_decode / std::max(1, S.decode_steps));
        for (int r = 1; r <= tp_model::MAX_ROWS; ++r) if (S.rows_hist[r]) printf(" [%d]%d", r, S.rows_hist[r]);
        printf("), drafts %.2f ms/step, rollback %.3f ms/step, prefill %.2f s total\n",
               1e3 * S.t_drafts / std::max(1, S.decode_steps), 1e3 * S.t_rollback / std::max(1, S.decode_steps), S.t_prefill);
        fflush(stdout);
    };
    // ---- references: every prompt alone, plain greedy ----
    std::vector<std::vector<int32_t>> ref(A.prompts.size());
    printf("reference runs (alone, plain greedy):\n");
    for (int i = 0; i < nmax; ++i) run_jobs({i}, false, "ref", &ref, nullptr, nullptr);
    int n_same = 0, n_cmp = 0;
    std::vector<bool> modes{false};
    if (d) modes.push_back(true);
    for (bool mtp : modes) {
        char tag[64];
        for (int n : A.counts) {
            std::vector<int> which;
            for (int i = 0; i < n; ++i) which.push_back(i);
            snprintf(tag, sizeof tag, "%s n=%d", mtp ? "mtp" : "plain", n);
            printf("SESSIONS %s%s:\n", tag, mtp ? (A.tau > 0 ? " (adaptive)" : " (fixed k)") : "");
            if (n == 1 && !mtp) { printf("  (= the reference runs above)\n"); continue; }
            n_cmp += n;
            run_jobs(which, mtp, tag, nullptr, &ref, &n_same);
        }
        if (mtp) {   // every prompt alone with MTP too (per-session numbers alone)
            for (int i = 1; i < nmax; ++i) {
                snprintf(tag, sizeof tag, "mtp alone");
                n_cmp += 1;
                run_jobs({i}, true, tag, nullptr, &ref, &n_same);
            }
        }
    }
    // ---- offload / restore ----
    if (A.offload_n >= 0) {
        const bool mtp = (bool) d;
        printf("OFFLOAD test (%s): job A = s0 generates %d tokens, offload, job B = s1 runs (kept resident), restore A, "
               "finish\n", mtp ? "mtp" : "plain", A.offload_n);
        sched S(m, d, so);
        const int ja = S.submit(A.prompts[0], A.gen, mtp);
        static const bool oracle = getenv("P100E_SESS_ORACLE") && atoi(getenv("P100E_SESS_ORACLE")) != 0;
        if (oracle && mtp) S.set_oracle(ja, &ref[0]);
        while ((int) S.get(ja).out.size() < std::max(1, A.offload_n) && S.get(ja).st != sched::job::DONE) S.step();
        const int seq_a = S.get(ja).seq, pages_a = m.seq_pages(seq_a);
        S.preempt(ja);
        printf("  A offloaded after %zu tokens: %d pages, %.1f MiB pinned host, %.1f ms; GPU pool free %d pages\n",
               S.get(ja).out.size(), pages_a, m.seq_host_bytes(seq_a) / 1048576.0, 1e3 * S.get(ja).t_offload, m.free_pages());
        const int jb = S.submit(A.prompts[1], A.gen, mtp, true);
        if (oracle && mtp) S.set_oracle(jb, &ref[1]);
        while (S.get(jb).st != sched::job::DONE) S.step();
        S.resume(ja);   // queued: admission restores it with the next step
        S.run();
        printf("  A restored: %.1f ms (B still holds its slot and pages)\n", 1e3 * S.get(ja).t_restore);
        const bool sa = S.get(ja).out == ref[0], sb = S.get(jb).out == ref[1];
        printf("  A: %zu tokens %s; B: %zu tokens %s\n", S.get(ja).out.size(), sa ? "IDENTICAL" : "MISMATCH",
               S.get(jb).out.size(), sb ? "IDENTICAL" : "MISMATCH");
        S.release(jb);
        n_cmp += 2;
        n_same += sa + sb;
        // idle session: A' = s0 for half the tokens (kept), then one short job per sequence slot - admission must move the
        // idle A' to host RAM; the next turn of A' (submit_continue) restores it and must continue s0's tokens exactly
        sched_opts so2 = so;
        so2.max_active = eo.max_seqs;
        sched S2(m, d, so2);
        const int g1 = std::max(1, A.gen / 2);
        const int ja2 = S2.submit(A.prompts[0], g1, mtp, true);
        S2.run();
        for (int i = 0; i < eo.max_seqs; ++i) {   // short fillers: they only need the slots
            const auto & p = A.prompts[(1 + i) % A.prompts.size()];
            S2.submit(std::vector<int32_t>(p.begin(), p.begin() + std::min<size_t>(64, p.size())), 4, false);
        }
        S2.run();
        const bool was_off = S2.get(ja2).offloaded;
        const int jc = S2.submit_continue(ja2, A.gen - g1, mtp);
        S2.run();
        std::vector<int32_t> all = S2.get(ja2).out;
        all.insert(all.end(), S2.get(jc).out.begin(), S2.get(jc).out.end());
        const bool sc = all == ref[0];
        printf("IDLE test (%s): s0 %d tokens kept, %d jobs filled every slot -> s0 %s (%.1f ms), next turn: restored %.1f ms, "
               "%d more tokens: %s\n", mtp ? "mtp" : "plain", g1, eo.max_seqs, was_off ? "offloaded to host" : "NOT offloaded",
               1e3 * S2.get(ja2).t_offload, 1e3 * S2.get(ja2).t_restore, A.gen - g1, sc ? "IDENTICAL" : "MISMATCH");
        S2.release(jc);
        n_cmp += 1;
        n_same += sc && was_off;
    }
    printf("SESSIONS identity: %d/%d session runs identical to their plain run alone\n", n_same, n_cmp);
    m.print_mem("end");
    return n_same == n_cmp ? 0 : 2;
}
