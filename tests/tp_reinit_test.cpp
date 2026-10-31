// p100e-reinit-test <model.gguf> --devices 0,1 [--loads N] [--prompt T] [--gen G] [--ubatch U] [--ctx C]
//   [--kv q8_0|f16|q4_0] [--disable-peer-before K[,K2..]] [--pending-error-before K[,K2..]]
// Re-creating tp_model in one process (regression: every instance after the first gave non-finite logits).
// Each load: construct, prefill a T-token synthetic prompt (the prefill pipeline when T > 8), the LM head again through
// logits_rows (the KLD path) on the last row, greedy-decode G tokens (fused T = 1 path, CUDA graphs), destroy.
// PASS = every load finite and bit-identical to load 1 (FNV-1a over all logits, the decoded ids).
// --disable-peer-before K: before load K turn peer access off both ways, as in a fresh process (a control).
// --pending-error-before K: before load K leave a (non-sticky) CUDA error pending, as a component that forgets to clear
// an expected one would; the load must still match (fast::pack_weight clears and reports it). Pair it with
// --disable-peer-before K: else the constructor's AlreadyEnabled overwrites the error slot and clears it.
#include "../src/tp_model.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace p100e;

static std::vector<int> parse_list(const char * s) {
    std::vector<int> v;
    std::stringstream ss(s);
    std::string t;
    while (std::getline(ss, t, ',')) if (!t.empty()) v.push_back(std::stoi(t));
    return v;
}

struct load_result {
    bool ran = false;
    uint64_t hash = 1469598103934665603ull;
    size_t nonfinite = 0, n = 0;
    double head_diff = 0;   // logits_rows vs the forward's own last-row logits
    std::vector<int32_t> ids;
};

static void add(load_result & r, const std::vector<float> & v) {
    for (float x : v) {
        uint32_t u;
        memcpy(&u, &x, 4);
        r.hash = (r.hash ^ u) * 1099511628211ull;
        r.nonfinite += !std::isfinite(x);
    }
    r.n += v.size();
}

static int32_t argmax(const std::vector<float> & v) {
    return (int32_t) (std::max_element(v.begin(), v.end()) - v.begin());
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: p100e-reinit-test <model.gguf> --devices 0,1 [--loads N] [--prompt T] [--gen G] [--ubatch U] "
                        "[--ctx C] [--kv q8_0|f16|q4_0] [--disable-peer-before K[,K2..]] [--pending-error-before K[,K2..]]\n");
        return 2;
    }
    std::vector<int> devices, peer_off, pend_err;
    int loads = 3, T = 512, gen = 8, ubatch = 0, ctx = 0;
    uint32_t kv = fast::KV_Q8_0;
    for (int i = 2; i < argc; ++i) {
        if (!strcmp(argv[i], "--devices") && i + 1 < argc) devices = parse_list(argv[++i]);
        else if (!strcmp(argv[i], "--loads") && i + 1 < argc) loads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--prompt") && i + 1 < argc) T = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--gen") && i + 1 < argc) gen = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ubatch") && i + 1 < argc) ubatch = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ctx") && i + 1 < argc) ctx = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--disable-peer-before") && i + 1 < argc) peer_off = parse_list(argv[++i]);
        else if (!strcmp(argv[i], "--pending-error-before") && i + 1 < argc) pend_err = parse_list(argv[++i]);
        else if (!strcmp(argv[i], "--kv") && i + 1 < argc) {
            const std::string k = argv[++i];
            kv = k == "f16" ? fast::KV_F16 : k == "q4_0" ? fast::KV_Q4_0 : fast::KV_Q8_0;
        } else { fprintf(stderr, "unknown argument %s\n", argv[i]); return 2; }
    }
    if (devices.empty() || loads < 1 || T < 1 || gen < 0) { fprintf(stderr, "need --devices, loads >= 1, prompt >= 1\n"); return 2; }
    if (ubatch <= 0) ubatch = T;
    ctx = std::max(ctx, T + gen + 1);

    using clk = std::chrono::steady_clock;
    auto secs = [](clk::time_point a, clk::time_point b) { return std::chrono::duration<double>(b - a).count(); };
    std::vector<load_result> res(loads);
    for (int l = 0; l < loads; ++l) {
        if (std::find(peer_off.begin(), peer_off.end(), l + 1) != peer_off.end()) {
            for (int a : devices) for (int b : devices) {
                if (a == b) continue;
                cudaSetDevice(a);
                const cudaError_t e = cudaDeviceDisablePeerAccess(b);
                printf("load %d: disable peer access GPU %d -> GPU %d: %s\n", l + 1, a, b, cudaGetErrorString(e));
            }
            cudaGetLastError();
        }
        if (std::find(pend_err.begin(), pend_err.end(), l + 1) != pend_err.end()) {
            int n = 0;
            cudaGetDeviceCount(&n);
            const cudaError_t e = cudaSetDevice(n + 7);   // invalid ordinal: cudaErrorInvalidDevice, left pending
            printf("load %d: left pending before construction: %s\n", l + 1, cudaGetErrorString(e));
        }
        load_result & r = res[l];
        try {
            const auto t0 = clk::now();
            auto m = std::make_unique<tp_model>(argv[1], devices, ctx, ubatch, false, kv);
            const auto t1 = clk::now();
            // synthetic prompt (same every load): LCG ids in [1000, vocab - 1000)
            std::vector<int32_t> prompt(T);
            uint32_t s = 12345;
            for (auto & t : prompt) { s = s * 1664525u + 1013904223u; t = 1000 + (int32_t) ((s >> 8) % (uint32_t) (m->vocab() - 2000)); }
            std::vector<float> lg = m->forward_n(prompt, 1, false);
            const auto t2 = clk::now();
            add(r, lg);
            std::vector<float> chk(m->vocab());
            m->logits_rows(m->last_tokens() - 1, 1, chk.data());
            add(r, chk);
            for (size_t v = 0; v < chk.size(); ++v)
                if (std::isfinite(lg[v]) && std::isfinite(chk[v])) r.head_diff = std::max(r.head_diff, (double) std::fabs(lg[v] - chk[v]));
            const auto t3 = clk::now();
            for (int g = 0; g < gen; ++g) {
                r.ids.push_back(argmax(lg));
                lg = m->forward({r.ids.back()});
                add(r, lg);
            }
            const auto t4 = clk::now();
            r.ran = true;
            printf("load %d: load %.1f s, prefill %d %.2f s, decode %d %.2f t/s | non-finite %zu of %zu, logits_rows max |diff| %g, "
                   "hash %016llx, ids", l + 1, secs(t0, t1), T, secs(t1, t2), gen, gen / std::max(1e-9, secs(t3, t4)),
                   r.nonfinite, r.n, r.head_diff, (unsigned long long) r.hash);
            for (int32_t id : r.ids) printf(" %d", id);
            printf("\n");
        } catch (const std::exception & e) {
            printf("load %d: FAILED: %s\n", l + 1, e.what());
        }
        fflush(stdout);
    }
    int bad = 0;
    for (int l = 0; l < loads; ++l) {
        const load_result & r = res[l];
        const bool ok = r.ran && r.nonfinite == 0 && r.head_diff == 0 && r.hash == res[0].hash && r.ids == res[0].ids;
        if (!ok) {
            ++bad;
            printf("load %d differs from load 1 or is not finite\n", l + 1);
        }
    }
    printf("REINIT %s: %d of %d loads OK\n", bad ? "FAIL" : "PASS", loads - bad, loads);
    return bad ? 1 : 0;
}
