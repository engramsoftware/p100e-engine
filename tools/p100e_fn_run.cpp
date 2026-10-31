// p100e-fn-run: Qwen3.8-Flash-Next (qwen4exp expert-split GGUF) through fn_model.
//   p100e-fn-run <model.gguf> --devices 0,1 --tokens 760,6511,314,9338,369 [--gen N] [--ctx 4096] [--hot-per-rank 80]
//                [--ref <dir>] [--chunk T] [--max-t N]   (--max-t: activation buffers for up to N rows; default max(64, chunk))
//                [--synth N]: N synthetic prompt ids 760 + (i*7919) % 200000 (long prompts without a long command line)
// --tokens: the prompt as token ids (the reference dump's tokens.txt); fed as chunks of --chunk tokens (default 1 = the
//   decode path; larger = the prefill path) and then --gen greedy tokens.
// --ref <dir>: compare the per-layer activations of every prompt token against a llama.cpp fn-ref-dump directory
//   (t<idx>_<name>.bin raw f32): prints max |diff| / max |ref| per tensor family and the top-5
//   logits of both. The reference runs one token per decode, so --ref implies --chunk 1 unless --chunk is given.
// Hybrid pipeline: --net-nodes host0:port,host1:port --net-node 0|1 --pipe-layers k|auto (or
//   --pipe-split a,b): node 0 loads layers [0, k) and serves them (this program then only waits for commands), node 1
//   loads [k, L) + head + draft block and runs the prompt / generation as usual. --slots a,b|auto: resident experts per
//   local GPU (auto: what fits it). --pipe-plan "F0a,F0b:H0/F1a,F1b:H1": offline byte plan (MiB of free VRAM per GPU and
//   available RAM per node), no GPU work. --ram-reserve-mb N (P100E_RAM_RESERVE_MB): RAM the plan leaves free.
//   --layers-mtp: keep the draft block under --layers (debug). P100E_FN_RUN_HASH=1: hash of every returned logit row.
// Remote expert tier: --rexp-listen host:port (node 0 waits for p100e-fn-expert-server --connect),
//   --rexp-max-t N (forwards of <= N rows use it, default 4 = decode / MTP verify / drafts), --rexp-sum (per-row sums
//   back, fewer bytes, not bit-equal), --rexp-connect-s S (how long to wait for node 1).
// --plan "F0,F1:H": print the one-process byte plan for F0, F1 MiB of free VRAM per GPU and H MiB of available RAM
//   (with --slots a,b|auto, --ctx, --max-t, P100E_FN_KV / _PREFETCH_T as for a real run), no GPU work.
#include "fn_model.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <exception>
#include <map>
#include <string>
#include <vector>

using namespace p100e;

static std::vector<float> read_bin(const std::string & p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) return {};
    f.seekg(0, std::ios::end); const size_t n = (size_t) f.tellg() / 4; f.seekg(0);
    std::vector<float> v(n);
    f.read((char *) v.data(), n * 4);
    return v;
}

static void top5(const char * tag, const std::vector<float> & l, size_t off, int V) {
    std::vector<int> idx(V);
    for (int i = 0; i < V; ++i) idx[i] = i;
    std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(), [&](int a, int b) { return l[off + a] > l[off + b]; });
    printf("%s top5:", tag);
    for (int i = 0; i < 5; ++i) printf(" %d:%.4f", idx[i], l[off + idx[i]]);
    printf("\n");
}

static int run(int argc, char ** argv);
int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    try { return run(argc, argv); }
    catch (const std::exception & e) { fprintf(stderr, "p100e-fn-run: %s\n", e.what()); return 1; }
}

static int run(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: p100e-fn-run <model.gguf> --devices 0,1 --tokens a,b,c [--gen N] [--ref dir] [--chunk T]\n"); return 2; }
    std::string path = argv[1], ref, logits_out;
    std::vector<int> devices = {0};
    std::vector<int32_t> toks;
    int gen = 0, ctx = 4096, hpr = -1, chunk = 1, layers = -1, max_t = FN_MAX_T, mtp_k = 0;
    bool verbose = false;
    // hybrid pipeline: two processes / machines; node 0 serves layers [0, k), node 1 runs this program
    std::vector<std::string> net_nodes;
    int net_node = -1, pipe_layers = 0, pipe_total = 0;
    std::vector<int> slots;            // --slots a,b|auto: resident expert slots per local rank (auto = fill the GPU)
    bool layers_mtp = false;           // --layers-mtp: keep the draft block under --layers (debug)
    int ram_reserve = -1;              // --ram-reserve-mb N (default P100E_RAM_RESERVE_MB, else 12288 Windows / 6144 Linux)
    std::string plan1;                 // --plan "F0,F1:H": the one-process byte plan for that free VRAM / RAM, no GPU work
    std::string rexp_listen;           // --rexp-listen host:port: remote expert tier node 0 (p100e-fn-expert-server connects)
    int rexp_max_t = -1, rexp_connect_s = -1;
    bool rexp_sum = false;
    std::string pipe_plan;             // --pipe-plan "F0a,F0b:H0/F1a,F1b:H1" (MiB): offline byte plan, no GPU work
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return std::string(i + 1 < argc ? argv[++i] : ""); };
        if (a == "--devices") { std::string d = next(); devices.clear(); for (char * p = strtok(&d[0], ","); p; p = strtok(nullptr, ",")) devices.push_back(atoi(p)); }
        else if (a == "--tokens") { std::string s = next(); for (char * p = strtok(&s[0], ","); p; p = strtok(nullptr, ",")) toks.push_back(atoi(p)); }
        else if (a == "--gen") gen = atoi(next().c_str());
        else if (a == "--ctx") ctx = atoi(next().c_str());
        else if (a == "--hot-per-rank") hpr = atoi(next().c_str());
        else if (a == "--ref") ref = next();
        else if (a == "--chunk") chunk = atoi(next().c_str());
        else if (a == "--verbose") verbose = true;
        else if (a == "--layers") layers = atoi(next().c_str());
        else if (a == "--max-t") max_t = atoi(next().c_str());
        else if (a == "--drafts" || a == "--mtp") mtp_k = atoi(next().c_str());   // speculative decoding: guesses (drafts) per step, 1..3
        else if (a == "--tokens-file") {   // whitespace / comma separated token ids (prompts beyond the command-line limit)
            std::ifstream f(next());
            toks.clear();
            std::string s;
            while (f >> s) { for (char & ch : s) if (ch == ',') ch = ' '; std::istringstream ss(s); int v; while (ss >> v) toks.push_back(v); }
        }
        else if (a == "--rexp-listen") rexp_listen = next();   // remote expert tier: host:port node 0 listens on
        else if (a == "--rexp-max-t") rexp_max_t = atoi(next().c_str());
        else if (a == "--rexp-sum") rexp_sum = true;
        else if (a == "--rexp-connect-s") rexp_connect_s = atoi(next().c_str());
        else if (a == "--logits-out") logits_out = next();   // f32 logits of the prompt's last row, then of every greedy step
        else if (a == "--net-nodes") { std::string s = next(); for (char * p = strtok(&s[0], ","); p; p = strtok(nullptr, ",")) net_nodes.push_back(p); }
        else if (a == "--net-node") net_node = atoi(next().c_str());
        else if (a == "--pipe-layers") { const std::string v = next(); pipe_layers = v == "auto" ? -1 : atoi(v.c_str()); }
        else if (a == "--pipe-split") { const std::string v = next(); const int x = atoi(v.c_str()); const size_t c = v.find(','); const int y = c == std::string::npos ? 0 : atoi(v.c_str() + c + 1); pipe_layers = x; pipe_total = x + y; }
        else if (a == "--slots") { std::string s = next(); for (char * p = strtok(&s[0], ","); p; p = strtok(nullptr, ",")) slots.push_back(!strcmp(p, "auto") ? -1 : atoi(p)); }
        else if (a == "--layers-mtp") layers_mtp = true;
        else if (a == "--pipe-plan") pipe_plan = next();
        else if (a == "--plan") plan1 = next();   // one-process byte plan offline: "F0,F1:H" MiB free VRAM per GPU : available RAM
        else if (a == "--ram-reserve-mb" || a == "--p100e-ram-reserve-mb") ram_reserve = atoi(next().c_str());   // RAM kept free (P100E_RAM_RESERVE_MB)
        else if (a == "--synth") { const int n = atoi(next().c_str()); toks.clear(); for (int i = 0; i < n; ++i) toks.push_back(760 + (int) (((int64_t) i * 7919) % 200000)); }
    }
    fn_model::opts o; o.n_ctx = ctx; o.hot_per_rank = hpr; o.verbose = verbose; o.layer_limit = layers;
    o.max_t = std::max({max_t, chunk, 1});
    o.mtp_with_limit = layers_mtp;
    o.ram_reserve_mb = ram_reserve;
    if (!plan1.empty()) {
        o.plan_only = true;
        fn_model m(path, devices, o);
        std::vector<double> fr;
        const size_t c = plan1.find(':');
        std::string gs = plan1.substr(0, c);
        for (char * q = strtok(&gs[0], ","); q; q = strtok(nullptr, ",")) fr.push_back(atof(q) * 1048576.0);
        const double host = c == std::string::npos ? 0 : atof(plan1.c_str() + c + 1) * 1048576.0;
        std::vector<int> s1 = slots;
        if (!s1.empty() && s1.size() != fr.size()) s1.assign(fr.size(), s1[0]);
        const fn_model::node_plan p = m.plan_node(0, m.n_layer(), true, fr, s1, host);
        m.print_plan("plan", 0, m.n_layer(), p);
        return p.fits ? 0 : 1;
    }
    if (!pipe_plan.empty()) {
        // offline plan: "F0a,F0b:H0/F1a,F1b:H1" = free VRAM per rank (MiB) and available host RAM (MiB) of each node
        o.plan_only = true;
        fn_model m(path, devices, o);
        std::vector<double> fr[2]; double hf[2] = {0, 0};
        const size_t sl = pipe_plan.find('/');
        for (int j = 0; j < 2; ++j) {
            const std::string part = j == 0 ? pipe_plan.substr(0, sl) : pipe_plan.substr(sl + 1);
            const size_t c = part.find(':');
            std::string g = part.substr(0, c);
            for (char * q = strtok(&g[0], ","); q; q = strtok(nullptr, ",")) fr[j].push_back(atof(q) * 1048576.0);
            hf[j] = atof(part.c_str() + c + 1) * 1048576.0;
        }
        std::vector<int> s0, s1;
        if (!slots.empty()) { s0.assign(fr[0].size(), slots[0]); s1.assign(fr[1].size(), slots[0]); }
        // reserves: node 0 Linux 6144 MiB, node 1 Windows 12288 MiB (P100E_RAM_RESERVE_MB / --ram-reserve-mb sets both)
        const char * re = getenv("P100E_RAM_RESERVE_MB");
        const double r0 = (ram_reserve >= 0 ? ram_reserve : re ? atof(re) : 6144) * 1048576.0, r1 = (ram_reserve >= 0 ? ram_reserve : re ? atof(re) : 12288) * 1048576.0;
        const int k = m.plan_split(fr[0], hf[0], s0, fr[1], hf[1], s1, pipe_layers > 0 ? pipe_layers : -1, true, r0, r1);
        printf("pipe plan: %s k = %d\n", pipe_layers > 0 ? "requested" : "auto", k);
        return k > 0 ? 0 : 1;
    }
    if (!net_nodes.empty()) {
        if (net_node < 0) net_node = getenv("P100E_NET_NODE") ? atoi(getenv("P100E_NET_NODE")) : 0;
        if (pipe_layers == 0) { fprintf(stderr, "error: --net-nodes needs --pipe-layers k|auto or --pipe-split a,b\n"); return 2; }
        o.net_nodes = net_nodes; o.net_node = net_node; o.pipe_layers = pipe_layers; o.pipe_total = pipe_total;
    }
    o.rank_slots = slots;
    o.rexp_listen = rexp_listen; o.rexp_sum = rexp_sum;
    if (rexp_max_t > 0) o.rexp_max_t = rexp_max_t;
    if (rexp_connect_s > 0) o.rexp_connect_s = rexp_connect_s;
    auto t0 = std::chrono::steady_clock::now();
    fn_model m(path, devices, o);
    printf("loaded in %.1f s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    if (m.pipe_worker()) {   // node 0: serve the driver's commands until it quits
        m.pipe_serve();
        return 0;
    }
    const int V = m.vocab();
    m.set_dump(!ref.empty());
    // per-family max relative error over the prompt tokens
    std::map<std::string, std::pair<double, double>> err;   // family -> (max |diff|, max |ref|)
    auto cmp = [&](int ti, const std::string & name, const std::vector<float> & ours) {
        std::string fname = name;
        if (name.rfind("hc_mixed_ffn-", 0) == 0) fname = "hc_mixed-" + name.substr(13) + "_dup1";
        std::vector<float> r = read_bin(ref + "/t" + std::to_string(ti) + "_" + fname + ".bin");
        if (r.empty() && name == "model.input_embed") r = read_bin(ref + "/t" + std::to_string(ti) + "_model_input_embed.bin");
        if (r.empty() || r.size() != ours.size()) { if (ti == 0) printf("  no ref for %s (ours %zu, ref %zu)\n", name.c_str(), ours.size(), r.size()); return; }
        double md = 0, mr = 0;
        for (size_t i = 0; i < r.size(); ++i) { md = std::max(md, (double) fabs(r[i] - ours[i])); mr = std::max(mr, (double) fabs(r[i])); }
        std::string fam = name.substr(0, name.find('-'));
        auto & e = err[fam]; e.first = std::max(e.first, md); e.second = std::max(e.second, mr);
        if (md > 0.05 * mr + 1e-4) printf("  t%d %-18s max|diff| %.4g  max|ref| %.4g  (ours[0..2] %.4g %.4g %.4g ref %.4g %.4g %.4g)\n", ti, name.c_str(), md, mr, ours[0], ours[1], ours[2], r[0], r[1], r[2]);
    };
    std::vector<float> last_logits;
    // P100E_FN_RUN_HASH=1: FNV-1a over every logit row the run returns (prompt chunks, then each greedy step): a bitwise
    // fingerprint of the numerics for identity checks between builds / configurations
    const bool hash_on = getenv("P100E_FN_RUN_HASH") && atoi(getenv("P100E_FN_RUN_HASH")) != 0;
    uint64_t lhash = 1469598103934665603ull;
    auto hash_f = [](uint64_t h, const std::vector<float> & v) { const unsigned char * p = (const unsigned char *) v.data(); for (size_t k = 0; k < v.size() * 4; ++k) h = (h ^ p[k]) * 1099511628211ull; return h; };
    for (size_t i = 0; i < toks.size();) {
        const int T = std::min<int>(chunk, (int) (toks.size() - i));
        std::vector<int32_t> ch(toks.begin() + i, toks.begin() + i + T);
        auto ts = std::chrono::steady_clock::now();
        std::vector<float> l = m.forward_n(ch, 1);
        last_logits = l;
        if (hash_on) { lhash = hash_f(lhash, l); printf("logits hash after prompt row %zu: %016llx\n", i + T - 1, (unsigned long long) lhash); }
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ts).count();
        printf("prompt tokens %zu..%zu (%d): %.1f ms\n", i, i + T - 1, T, ms);
        if (!ref.empty() && T == 1) {
            for (auto & [n, v] : m.dump()) cmp((int) i, n, v);
            std::vector<float> rl = read_bin(ref + "/t" + std::to_string(i) + "_result_output.bin");
            top5("  ours", l, 0, V);
            if (!rl.empty()) top5("  ref ", rl, 0, V);
        }
        i += T;
    }
    if (!ref.empty()) {
        printf("max |diff| / max |ref| per family over the prompt:\n");
        for (auto & [f, e] : err) printf("  %-18s %.4g / %.4g  (%.2f%%)\n", f.c_str(), e.first, e.second, e.second > 0 ? 100.0 * e.first / e.second : 0.0);
    }
    m.set_dump(false);
    // greedy generation: the first token is the argmax of the prompt's last logits, then one forward per token
    if (mtp_k > 0) {
        // MTP: the prompt again in `chunk`-row forwards on a fresh context (with the draft block's catch-up), then gen tokens
        m.reset();
        fn_model::mtp_stats st;
        auto tg = std::chrono::steady_clock::now();
        const std::vector<int32_t> out = m.generate_mtp(toks, gen, chunk, mtp_k, st);
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - tg).count() - st.t_prompt - st.t_catchup;
        printf("mtp gen:");
        for (int32_t t : out) printf(" %d", t);
        printf("\n%zu tokens in %.2f s = %.2f t/s (drafts=%d: %d steps, %d guessed, %d accepted = %.2f tok/step; draft %.2f s, verify %.2f s; prompt %.2f s + catch-up %.2f s)\n",
               out.size(), s, out.size() / s, mtp_k, st.steps, st.drafted, st.accepted, st.steps ? (double) out.size() / st.steps : 0.0, st.t_draft, st.t_verify, st.t_prompt, st.t_catchup);
        return 0;
    }
    if (gen > 0 && !last_logits.empty()) {
        int32_t cur = 0;
        for (int i = 1; i < V; ++i) if (last_logits[i] > last_logits[cur]) cur = i;
        printf("gen: %d", cur); fflush(stdout);
        FILE * lf = logits_out.empty() ? nullptr : fopen(logits_out.c_str(), "wb");
        if (lf) fwrite(last_logits.data(), 4, V, lf);
        auto tg = std::chrono::steady_clock::now();
        for (int g = 1; g < gen; ++g) {
            if (lf || hash_on) {   // the full logits (forward1) and the same greedy pick as forward1_argmax (ties: lower index)
                const std::vector<float> l = m.forward1(cur);
                if (lf) fwrite(l.data(), 4, V, lf);
                if (hash_on) lhash = hash_f(lhash, l);
                cur = 0;
                for (int k = 1; k < V; ++k) if (l[k] > l[cur]) cur = k;
            } else cur = m.forward1_argmax(cur);
            printf(" %d", cur); fflush(stdout);
        }
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - tg).count();
        if (lf) fclose(lf);
        printf("\n%d tokens in %.2f s = %.2f t/s (decode, T = 1)\n", gen - 1, s, (gen - 1) / s);
        if (hash_on) printf("logits hash final: %016llx\n", (unsigned long long) lhash);
    }
    return 0;
}
