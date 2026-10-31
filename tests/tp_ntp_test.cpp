// p100e-tp-ntp-test <model.gguf> --configs "0|0,0|0,0,0,0|0,0,0@2,1,1" [--prompt P] [--gen G] [--ctx C] [--ubatch U]
//   [--kv q8_0|f16|q4_0] [--mtp K] [--arms default,event,perop] [--expect-refuse "0,0,0@4,1,1"]
// A config is a device list, optionally "@a,b,c": the tensor-parallel weights per rank (tp_model::opts::split, tag
// 'tp-uneven'; without it the split is equal, as even as the model allows). The split table is printed per config.
// Tensor parallel over N ranks: the same model, prompt and teacher-forced decode on every device list
// of --configs (ranks may share a GPU: "0,0,0,0" = four virtual ranks on GPU 0, needs P100E_P2P_SAME_DEV=1 for the P2P
// allreduce), every logit vector compared with the first config's (max |diff|, max |diff| / max |logit|, top-1, bitwise).
// Per config and arm (default = as the environment says; event = P100E_P2P_AR=0, the copy allreduce; perop =
// P100E_FUSE=0, the per-op decode): a prompt of P synthetic tokens (forward_n: the prefill pipeline when P > 8), G
// decode steps fed the reference config's greedy ids, then the same again after reset() (must be bitwise identical:
// determinism), then (--mtp K, 2+ ranks, default arm) free greedy decoding vs MTP speculative decoding with K
// drafts from the same prompt (must give identical ids), plus G chained draft-head runs after the prompt's KV-only
// catch-up, (token, p) vs the first MTP config. --expect-refuse: constructing on that list must throw.
// --kernel-check: the R > 2 allreduce epilogue kernel alone (see kernel_check).
// --seq-file P: per config (default arm) the sequence after the decode is written to P.<config index>
// (tp_model::seq_write), then every earlier config's file is read back (seq_read): a file from a config with the same
// rank count and split must load, one from another split must be refused (the session fingerprint).
// Exit 0 = every determinism / MTP / refusal / kernel check passed (the diffs vs the reference are reported, not judged).
#include "../src/tp_model.h"
#include "../src/tp_mtp.h"
#include "../src/kernels.cuh"
#include "../src/fast/fused.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace p100e;

static std::vector<std::string> split(const std::string & s, char c) {
    std::vector<std::string> v;
    std::stringstream ss(s);
    std::string t;
    while (std::getline(ss, t, c)) if (!t.empty()) v.push_back(t);
    return v;
}
static std::vector<int> ints(const std::string & s) {
    std::vector<int> v;
    for (const auto & t : split(s, ',')) v.push_back(std::stoi(t));
    return v;
}
// "0,0,0@2,1,1" -> devices {0, 0, 0}, weights {2, 1, 1} (no '@': empty weights)
struct config { std::vector<int> devs; std::vector<double> w; std::string text; };
static config parse_config(const std::string & s) {
    config c;
    c.text = s;
    const size_t at = s.find('@');
    c.devs = ints(s.substr(0, at));
    if (at != std::string::npos) c.w = tp_parse_split(s.substr(at + 1));
    return c;
}
static int32_t argmax(const std::vector<float> & v) { return (int32_t) (std::max_element(v.begin(), v.end()) - v.begin()); }
static uint64_t fnv(const std::vector<std::vector<float>> & vv) {
    uint64_t h = 1469598103934665603ull;
    for (const auto & v : vv)
        for (float x : v) { uint32_t u; memcpy(&u, &x, 4); h = (h ^ u) * 1099511628211ull; }
    return h;
}

struct run_out { std::vector<std::vector<float>> logits; };

// --kernel-check: the R > 2 allreduce epilogue fast::addn_rmsnorm_q_wait_ctr (P2P; flags pre-set so nothing waits) vs
// fast::addn_rmsnorm_q (the event path, no wait) on the same random rows (mixed magnitudes), T = 1 / 3 / 8, n = 1024 /
// 5120, 3 / 4 partials: x, y and the a16 buffer must be bitwise equal; row t of a T-row call must equal a T = 1 call on
// row t (MTP verify). Also reported (not judged): the same rows through k::add + fast::add2_rmsnorm_q (the R = 2
// kernel; its sum of squares is contracted per instantiation, so a row's scale can differ in the last bit).
// Returns the number of failing cases.
static int kernel_check() {
    uint64_t st = 0x9e3779b97f4a7c15ull;
    auto rnd = [&]() { st = st * 6364136223846793005ull + 1442695040888963407ull; return (double) (st >> 11) / 9007199254740992.0; };
    auto val = [&]() { return (float) ((rnd() * 2 - 1) * std::pow(10.0, rnd() * 6 - 3)); };
    int bad = 0, cases = 0, ref_same = 0;
    for (int n : {1024, 5120})
        for (int T : {1, 3, 8})
            for (int np : {3, 4})
                for (int trial = 0; trial < 8; ++trial) {
                    const size_t N = (size_t) T * n;
                    std::vector<float> hx(N), hw(n), hp((size_t) np * N);
                    for (auto & v : hx) v = val();
                    for (auto & v : hw) v = (float) (0.5 + rnd());
                    for (auto & v : hp) v = val();
                    const size_t qb = fast::matvec_scratch_bytes(n, 8);
                    float * xs[4], * ys[4], * w = nullptr, * p = nullptr;
                    void * qs[4];
                    int * flag = nullptr;
                    for (int k = 0; k < 4; ++k) {
                        cudaMalloc(&xs[k], N * 4); cudaMalloc(&ys[k], N * 4); cudaMalloc(&qs[k], qb);
                        cudaMemcpy(xs[k], hx.data(), N * 4, cudaMemcpyHostToDevice);
                        cudaMemset(ys[k], 0, N * 4); cudaMemset(qs[k], 0, qb);
                    }
                    cudaMalloc(&w, (size_t) n * 4); cudaMalloc(&p, hp.size() * 4); cudaMalloc(&flag, 256);
                    cudaMemcpy(w, hw.data(), (size_t) n * 4, cudaMemcpyHostToDevice);
                    cudaMemcpy(p, hp.data(), hp.size() * 4, cudaMemcpyHostToDevice);
                    int words[64] = {};
                    words[2] = 5;   // own counter = the sequence the wait looks for; every sender word already there
                    words[fast::P2P_FLAG_TIMEOUT_MS] = 2000;
                    for (int q = 0; q < np; ++q) words[fast::P2P_FLAG_PEER0 + q] = 5;
                    cudaMemcpy(flag, words, sizeof words, cudaMemcpyHostToDevice);
                    fast::p2p_ptrs pp{};
                    for (int q = 0; q < np; ++q) pp.p[q] = p + q * N;
                    bool ok = fast::addn_rmsnorm_q_wait_ctr(xs[0], pp, np, np - 1, flag, flag + 2, w, ys[0], qs[0], T, n, 1e-6f, nullptr);
                    ok = fast::addn_rmsnorm_q(xs[1], pp, np, w, ys[1], qs[1], T, n, 1e-6f, nullptr) && ok;
                    for (int t = 0; t < T; ++t) {   // [2]: row by row, T = 1 (its own a16 layout: x / y compared)
                        fast::p2p_ptrs pt{};
                        for (int q = 0; q < np; ++q) pt.p[q] = pp.p[q] + (size_t) t * n;
                        ok = fast::addn_rmsnorm_q(xs[2] + (size_t) t * n, pt, np, w, ys[2] + (size_t) t * n, qs[2], 1, n, 1e-6f,
                                                  nullptr) && ok;
                    }
                    for (int q = 0; q < np - 2; ++q) k::add(xs[3], pp.p[q], (int64_t) N, nullptr);   // [3]: the R = 2 kernel
                    const bool q3 = fast::add2_rmsnorm_q(xs[3], pp.p[np - 2], pp.p[np - 1], w, ys[3], qs[3], T, n, 1e-6f, nullptr);
                    ok = cudaDeviceSynchronize() == cudaSuccess && ok && q3;
                    std::vector<char> hxo[4], hyo[4], hqo[4];
                    for (int k = 0; k < 4; ++k) {
                        hxo[k].resize(N * 4); hyo[k].resize(N * 4); hqo[k].resize(qb);
                        cudaMemcpy(hxo[k].data(), xs[k], N * 4, cudaMemcpyDeviceToHost);
                        cudaMemcpy(hyo[k].data(), ys[k], N * 4, cudaMemcpyDeviceToHost);
                        cudaMemcpy(hqo[k].data(), qs[k], qb, cudaMemcpyDeviceToHost);
                    }
                    const bool same01 = hxo[0] == hxo[1] && hyo[0] == hyo[1] && hqo[0] == hqo[1];
                    const bool same02 = hxo[0] == hxo[2] && hyo[0] == hyo[2] && (T > 1 || hqo[0] == hqo[2]);
                    const bool same03 = hxo[0] == hxo[3] && hyo[0] == hyo[3] && hqo[0] == hqo[3];
                    ++cases;
                    ref_same += same03;
                    if (!(ok && same01 && same02)) {
                        ++bad;
                        printf("KERNEL n %d T %d np %d trial %d: launch %s, wait vs no-wait %s, T rows vs row by row %s FAIL\n", n, T,
                               np, trial, ok ? "ok" : "FAILED", same01 ? "same" : "DIFF", same02 ? "same" : "DIFF");
                    }
                    for (int k = 0; k < 4; ++k) { cudaFree(xs[k]); cudaFree(ys[k]); cudaFree(qs[k]); }
                    cudaFree(w); cudaFree(p); cudaFree(flag);
                }
    printf("KERNEL addn_rmsnorm_q(_wait_ctr): %d/%d cases bitwise identical (P2P wait vs event no-wait, T rows vs row by row: "
           "x, y, a16); k::add + add2_rmsnorm_q (R = 2 kernel) also bitwise in %d/%d (x always; the rest: last bit of a "
           "row's norm scale)\n", cases - bad, cases, ref_same, cases);
    return bad;
}

static void set_env(const char * k, const char * v) {
#ifdef _WIN32
    _putenv_s(k, v ? v : "");
#else
    if (v) setenv(k, v, 1); else unsetenv(k);
#endif
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: p100e-tp-ntp-test <model.gguf> --configs \"0|0,0|0,0,0,0\" [--prompt P] [--gen G] [--ctx C] "
                        "[--ubatch U] [--kv q8_0|f16|q4_0] [--mtp K] [--arms default,event,perop] [--expect-refuse 0,0,0]\n");
        return 2;
    }
    const std::string path = argv[1];
    std::vector<config> configs, refuse;
    std::vector<std::string> arms = {"default"};
    int P = 300, G = 16, ctx = 1024, ub = 512, mtp_k = 0;
    std::string seq_file;
    std::vector<std::string> seq_keys;   // per config written: "R|split" (the layout a file must match), "" = none
    uint32_t kv = fast::KV_Q8_0;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--configs" && i + 1 < argc) { for (const auto & c : split(argv[++i], '|')) configs.push_back(parse_config(c)); }
        else if (a == "--expect-refuse" && i + 1 < argc) {
            // a weight list that does not parse is a refusal of its own (tp_parse_split throws): record it as one
            for (const auto & c : split(argv[++i], '|')) {
                try { refuse.push_back(parse_config(c)); }
                catch (const std::exception & e) { printf("REFUSE %s: refused while parsing: %s\n", c.c_str(), e.what()); }
            }
        }
        else if (a == "--arms" && i + 1 < argc) arms = split(argv[++i], ',');
        else if (a == "--prompt" && i + 1 < argc) P = atoi(argv[++i]);
        else if (a == "--gen" && i + 1 < argc) G = atoi(argv[++i]);
        else if (a == "--ctx" && i + 1 < argc) ctx = atoi(argv[++i]);
        else if (a == "--ubatch" && i + 1 < argc) ub = atoi(argv[++i]);
        else if (a == "--mtp" && i + 1 < argc) mtp_k = atoi(argv[++i]);
        else if (a == "--seq-file" && i + 1 < argc) seq_file = argv[++i];
        else if (a == "--kernel-check") { const int b = kernel_check(); if (b) return 1; }
        else if (a == "--kv" && i + 1 < argc) {
            const std::string k = argv[++i];
            kv = k == "f16" ? fast::KV_F16 : k == "q4_0" ? fast::KV_Q4_0 : fast::KV_Q8_0;
        } else { fprintf(stderr, "unknown argument %s\n", argv[i]); return 2; }
    }
    int fails = 0;
    for (const auto & d : refuse) {
        try {
            tp_model::opts o;
            o.n_ctx = ctx; o.max_batch = ub; o.kv_type = kv; o.split = d.w;
            tp_model m(path, d.devs, o);
            printf("REFUSE %s: constructed (expected a refusal) FAIL\n", d.text.c_str());
            ++fails;
        } catch (const std::exception & e) {
            printf("REFUSE %s: refused: %s\n", d.text.c_str(), e.what());
        }
    }
    if (configs.empty()) return fails ? 1 : 0;

    std::vector<int32_t> prompt(P);
    run_out ref;
    std::vector<int32_t> ref_ids;   // reference greedy ids (teacher forcing for every other config)
    std::vector<std::pair<int32_t, float>> ref_drafts;   // the first MTP config's chained draft-head (token, p)
    for (size_t ci = 0; ci < configs.size(); ++ci) {
        const std::vector<int> & devs = configs[ci].devs;
        const int R = (int) devs.size();
        for (const std::string & arm : arms) {
            if (arm != "default" && R < 2) continue;
            set_env("P100E_P2P_AR", arm == "event" ? "0" : nullptr);
            set_env("P100E_FUSE", arm == "perop" ? "0" : nullptr);
            const bool do_mtp = mtp_k > 0 && R >= 2 && arm == "default";
            std::string split_desc;
            std::vector<std::vector<float>> pass[2];
            std::vector<int32_t> greedy, mtp_ids;
            std::vector<std::pair<int32_t, float>> drafts;
            mtp_stats st;
            int V = 0;
            try {
                tp_model::opts o;
                o.n_ctx = ctx; o.max_batch = ub; o.kv_type = kv; o.load_mtp = do_mtp; o.split = configs[ci].w;
                tp_model m(path, devs, o);
                V = m.vocab();
                split_desc = m.split().even ? std::string("even") : m.split().str();
                for (int i = 0; i < P; ++i) prompt[i] = (int32_t) ((i * 7919 + 13) % V);
                for (int ps = 0; ps < 2; ++ps) {
                    m.reset();
                    std::vector<float> l = m.forward_n(prompt, 1);
                    pass[ps].push_back(l);
                    if (ci == 0 && arm == arms[0] && ps == 0) ref_ids.assign(1, argmax(l));
                    for (int g = 0; g < G; ++g) {
                        l = m.forward({ref_ids[g]});
                        pass[ps].push_back(l);
                        if (ci == 0 && arm == arms[0] && ps == 0) ref_ids.push_back(argmax(l));
                    }
                }
                if (!seq_file.empty() && arm == "default") {
                    // session file round trip across splits: write this config's sequence, read the earlier ones
                    const std::string key = std::to_string(R) + "|" + split_desc;
                    {
                        FILE * f = fopen((seq_file + "." + std::to_string(ci)).c_str(), "wb");
                        if (!f) throw std::runtime_error("--seq-file: cannot write " + seq_file);
                        m.seq_write(m.seq_active(), f);
                        fclose(f);
                    }
                    seq_keys.resize(ci + 1);
                    seq_keys[ci] = key;
                    for (size_t cj = 0; cj < ci; ++cj) {
                        if (seq_keys[cj].empty()) continue;
                        FILE * f = fopen((seq_file + "." + std::to_string(cj)).c_str(), "rb");
                        if (!f) continue;
                        const bool want = seq_keys[cj] == key;
                        std::string what;
                        bool loaded = false;
                        try {
                            const int id = m.seq_read(f);
                            loaded = true;
                            m.seq_free(id);
                        } catch (const std::exception & e) { what = e.what(); }
                        fclose(f);
                        const bool ok = loaded == want;
                        if (!ok) ++fails;
                        printf("SEQFILE config %zu -> config %zu (%s): %s%s%s\n", cj, ci, want ? "same layout" : "other layout",
                               loaded ? "loaded" : "refused: ", what.c_str(), ok ? "" : " FAIL");
                    }
                }
                if (do_mtp) {
                    m.reset();
                    std::vector<float> l = m.forward_n(prompt, 1);
                    for (int g = 0; g < G; ++g) { greedy.push_back(argmax(l)); l = m.forward({greedy.back()}); }
                    m.reset();
                    tp_mtp dr(m);
                    mtp_ids = generate_mtp(m, dr, prompt, G, mtp_k, st);
                    // the draft head itself (a random model accepts no drafts, so the verify alone does not check it): the
                    // prompt + KV-only catch-up, then G chained draft runs; (token, p) compared across configs
                    m.reset();
                    dr.h_zero(0);
                    int c0 = 0;
                    for (int c : m.ubatch_chunks(P, 1)) {
                        const std::vector<int32_t> ch(prompt.begin() + c0, prompt.begin() + c0 + c);
                        l = m.forward_n(ch, 1);
                        dr.catchup_kv(ch, c0);
                        c0 += c;
                    }
                    int32_t t = dr.run({argmax(l)}, P, tp_mtp::H_READY);
                    drafts.push_back({t, dr.last_p()});
                    for (int g = 1; g < G; ++g) {
                        t = dr.run({t}, P + g, tp_mtp::H_SELF);
                        drafts.push_back({t, dr.last_p()});
                    }
                }
            } catch (const std::exception & e) {
                printf("NTP R=%d arm=%s: error: %s FAIL\n", R, arm.c_str(), e.what());
                ++fails;
                continue;
            }
            if (ci == 0 && arm == arms[0]) ref.logits = pass[0];
            const bool det = pass[0] == pass[1];
            if (!det) ++fails;
            double mabs = 0, mref = 0;
            int top1 = 0, bit = 0;
            for (size_t s = 0; s < pass[0].size() && s < ref.logits.size(); ++s) {
                const auto & a = pass[0][s], & b = ref.logits[s];
                double ma = 0;
                for (int i = 0; i < V; ++i) { mabs = std::max(mabs, (double) std::fabs(a[i] - b[i])); ma = std::max(ma, (double) std::fabs(b[i])); }
                mref = std::max(mref, ma);
                top1 += argmax(a) == argmax(b);
                bit += a == b;
            }
            printf("NTP R=%d devices %s arm=%s: %zu logit rows, hash %016llx; vs ref: bitwise %d/%zu, max|d| %.3e, "
                   "max|d|/max|ref| %.3e, top1 %d/%zu; rerun after reset %s",
                   R, configs[ci].text.c_str(), arm.c_str(), pass[0].size(), (unsigned long long) fnv(pass[0]), bit,
                   pass[0].size(), mabs, mref > 0 ? mabs / mref : 0.0, top1, pass[0].size(),
                   det ? "IDENTICAL" : "DIFFERENT FAIL");
            if (do_mtp) {
                const bool same = greedy == mtp_ids;
                if (!same) ++fails;
                printf("; MTP k=%d vs own greedy %s (accepted %d/%d drafts)", mtp_k, same ? "IDENTICAL" : "MISMATCH FAIL",
                       st.accepted, st.drafted);
                if (ref_drafts.empty()) {
                    ref_drafts = drafts;
                    printf("; draft head: %zu chained runs (reference for the other even configs)", drafts.size());
                } else {
                    int agree = 0;
                    double dp = 0;
                    for (size_t j = 0; j < drafts.size() && j < ref_drafts.size(); ++j) {
                        agree += drafts[j].first == ref_drafts[j].first;
                        dp = std::max(dp, (double) std::fabs(drafts[j].second - ref_drafts[j].second));
                    }
                    printf("; draft head vs the first MTP config: tokens %d/%zu, max |p diff| %.3e", agree, drafts.size(), dp);
                }
            }
            printf("; split %s\n", split_desc.c_str());
            fflush(stdout);
        }
    }
    set_env("P100E_P2P_AR", nullptr);
    set_env("P100E_FUSE", nullptr);
    printf("NTP %s (%d failure(s))\n", fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}
