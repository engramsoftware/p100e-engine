// p100e-nvit: the native vision encoder (src/nvit) on its own - encode timing, VRAM, and parity against the fork's
// mtmd encoder run in the same process (--ref, needs the fork DLLs).
#include "nvit.h"
#include "nvit_kernels.cuh"
#include "../src/lend_gate.h"
#ifdef P100E_NVIT_REF
#include "p100e_nvit_ref.h"
#endif

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/wait.h>
#include <unistd.h>
#endif

using namespace p100e;

static const char * usage =
    "p100e-nvit --mmproj F --image IMG [--image IMG ...] [--devices 0,1] [--min-tokens 1024] [--max-tokens 2048]\n"
    "           [--runs N] [--resident] [--chunks N] [--prio 0|1] [--borrow MB [--borrow-busy] [--borrow-race]]\n"
    "  --borrow MB  scratch lent like tp_model::idle_scratch (a lend_gate lease per encode); --borrow-busy: odd runs\n"
    "               encode while the owner is busy (no lease: own scratch, same hash); --borrow-race: one more encode\n"
    "               on a thread while this thread enters the owner's busy scope (it must wait for the lease)\n"
    "           [--ref --vocab LLM.gguf [--ref-dev N] [--ref-runs N] [--dlls DIR]] [--dump-layers 0,13,26] [--no-native]\n"
    "  --ref        also encode with the fork's mtmd (same process) and compare: preprocessed image (bit-exact),\n"
    "               per-layer tensors of --dump-layers (native run in dump mode), final embeddings\n"
    "  --oracle     also run the fp32 oracle path (f32 weights, SGEMM, fp32 attention) on the first device and compare\n"
    "               native and mtmd against it (ORA = native vs oracle, ORR = mtmd vs oracle)\n"
    "p100e-nvit --attn-bench N1,N2,... [--devices d]   (attention kernel alone: nq = N/2 queries x 16 heads vs N keys)\n"
    "p100e-nvit --seq <args> ::: <args> ::: ...        (each group as its own process, in order)\n";

static std::string read_file(const std::string & p) {
    FILE * f = fopen(p.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open " + p);
    std::string s;
    char buf[1 << 16];
    size_t n;
    while ((n = fread(buf, 1, sizeof buf, f)) > 0) s.append(buf, n);
    fclose(f);
    return s;
}

struct cmp_res { double max_abs = 0, max_ref = 0, rms_d = 0, rms_r = 0, cos = 0, cos_min = 1, cos_mean = 0; int worst = -1; };
static cmp_res compare(const float * a, const float * b, int rows, int cols) {
    cmp_res r;
    double se = 0, sr = 0, dot = 0, na = 0, nb = 0, csum = 0;
    for (int i = 0; i < rows; ++i) {
        double d1 = 0, a1 = 0, b1 = 0;
        for (int j = 0; j < cols; ++j) {
            const double x = a[(size_t) i * cols + j], y = b[(size_t) i * cols + j];
            r.max_abs = std::max(r.max_abs, std::fabs(x - y));
            r.max_ref = std::max(r.max_ref, std::fabs(y));
            se += (x - y) * (x - y);
            sr += y * y;
            d1 += x * y; a1 += x * x; b1 += y * y;
        }
        dot += d1; na += a1; nb += b1;
        const double c = a1 > 0 && b1 > 0 ? d1 / std::sqrt(a1 * b1) : (a1 == b1 ? 1.0 : 0.0);
        csum += c;
        if (c < r.cos_min) { r.cos_min = c; r.worst = i; }
    }
    const double n = (double) rows * cols;
    r.rms_d = std::sqrt(se / n);
    r.rms_r = std::sqrt(sr / n);
    r.cos = na > 0 && nb > 0 ? dot / std::sqrt(na * nb) : 0;
    r.cos_mean = csum / rows;
    return r;
}
static void print_cmp(const char * tag, const std::string & name, const cmp_res & c) {
    printf("%s %-14s max|d| %.4e (max|ref| %.4e, rel %.3e)  rms(d)/rms(ref) %.3e  cos %.7f  row cos min %.6f (row %d) mean %.7f\n",
           tag, name.c_str(), c.max_abs, c.max_ref, c.max_ref > 0 ? c.max_abs / c.max_ref : 0.0,
           c.rms_r > 0 ? c.rms_d / c.rms_r : 0.0, c.cos, c.cos_min, c.worst, c.cos_mean);
}

static std::vector<size_t> vram_used(const std::vector<int> & devs) {
    std::vector<size_t> u;
    for (int d : devs) {
        size_t fr = 0, tot = 0;
        cudaSetDevice(d);
        cudaMemGetInfo(&fr, &tot);
        u.push_back(tot - fr);
    }
    return u;
}

static int attn_bench(const std::vector<int> & Ns, int dev) {
    cudaSetDevice(dev);
    cudaStream_t s;
    cudaStreamCreate(&s);
    for (int N : Ns) {
        const int nq = N / 2;
        const int nqt = (nq + nvit::TILE - 1) / nvit::TILE, ntk = (N + nvit::TILE - 1) / nvit::TILE;
        const int64_t qhs = (int64_t) nqt * nvit::TILE_U32, khs = (int64_t) ntk * nvit::TILE_U32;
        uint32_t *q, *k, *v;
        float * o;
        cudaMalloc(&q, (size_t) nvit::NH * qhs * 4); cudaMalloc(&k, (size_t) nvit::NH * khs * 4);
        cudaMalloc(&v, (size_t) nvit::NH * khs * 4); cudaMalloc(&o, (size_t) nq * nvit::E * 4);
        // small random-ish f16 values (0x2c00..0x2fff ~ 0.06..0.12, signs mixed): finite logits
        std::vector<uint32_t> h((size_t) nvit::NH * khs);
        uint32_t x = 12345;
        for (auto & w : h) { x = x * 1664525u + 1013904223u; w = ((x >> 8) & 0x83ff83ffu) | 0x2c002c00u; }
        cudaMemcpy(k, h.data(), (size_t) nvit::NH * khs * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(v, h.data(), (size_t) nvit::NH * khs * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(q, h.data(), (size_t) nvit::NH * qhs * 4, cudaMemcpyHostToDevice);
        nvit::attn72(q, qhs, 0, nq, k, v, khs, N, nvit::NH, o, nvit::E, s);
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0); cudaEventCreate(&e1);
        const int reps = 5;
        cudaEventRecord(e0, s);
        for (int r = 0; r < reps; ++r) nvit::attn72(q, qhs, 0, nq, k, v, khs, N, nvit::NH, o, nvit::E, s);
        cudaEventRecord(e1, s);
        cudaEventSynchronize(e1);
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        ms /= reps;
        printf("ATTN N %5d nq %5d x 16 heads: %.3f ms, %.2f TFLOPS (x27 layers: %.1f ms)\n", N, nq, ms,
               4.0 * nq * (double) N * nvit::HD * nvit::NH / (ms * 1e-3) / 1e12, 27 * ms);
        cudaFree(q); cudaFree(k); cudaFree(v); cudaFree(o);
    }
    const cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { printf("attn bench: %s\n", cudaGetErrorString(e)); return 1; }
    return 0;
}

// run this executable once per ':::'-separated group, in order (several configurations in one command)
static int run_seq(int argc, char ** argv) {
#ifdef _WIN32
    char self[MAX_PATH * 2];
    GetModuleFileNameA(nullptr, self, sizeof self);
#else
    char self[4096] = {};
    if (readlink("/proc/self/exe", self, sizeof self - 1) <= 0) snprintf(self, sizeof self, "%s", argv[0]);
#endif
    std::vector<std::vector<std::string>> groups(1);
    for (int i = 2; i < argc; ++i) {
        if (!strcmp(argv[i], ":::")) { groups.emplace_back(); continue; }
        groups.back().push_back(argv[i]);
    }
    int rc_all = 0;
    for (size_t g = 0; g < groups.size(); ++g) {
        std::string cmd = std::string("\"") + self + "\"";
        for (const auto & a : groups[g]) {
            if (a.find_first_of(" \t\"") != std::string::npos) cmd += " \"" + a + "\"";
            else cmd += " " + a;
        }
        printf("\n=== seq %zu/%zu: %s\n", g + 1, groups.size(), cmd.c_str());
        fflush(stdout);
#ifdef _WIN32
        STARTUPINFOA si{};
        si.cb = sizeof si;
        PROCESS_INFORMATION pi{};
        std::vector<char> cl(cmd.begin(), cmd.end());
        cl.push_back('\0');
        DWORD rc = 1;
        if (CreateProcessA(nullptr, cl.data(), nullptr, nullptr, TRUE, 0, nullptr, nullptr, &si, &pi)) {
            WaitForSingleObject(pi.hProcess, INFINITE);
            GetExitCodeProcess(pi.hProcess, &rc);
            CloseHandle(pi.hProcess);
            CloseHandle(pi.hThread);
        } else {
            printf("seq: CreateProcess failed (%lu)\n", GetLastError());
        }
#else
        std::string sh = "'" + std::string(self) + "'";   // POSIX shell words: single quotes, ' written as '\''
        for (const auto & a : groups[g]) {
            std::string q;
            for (char c : a) q += c == '\'' ? std::string("'\\''") : std::string(1, c);
            sh += " '" + q + "'";
        }
        const int st = system(sh.c_str());
        const unsigned long rc = st == -1 ? 1 : WIFEXITED(st) ? (unsigned long) WEXITSTATUS(st) : 1;
#endif
        printf("=== seq %zu exit %lu\n", g + 1, rc);
        fflush(stdout);
        if (rc && !rc_all) rc_all = (int) rc;
    }
    return rc_all;
}

int main(int argc, char ** argv) {
    if (argc > 1 && !strcmp(argv[1], "--seq")) return run_seq(argc, argv);
    if (argc > 2 && !strcmp(argv[1], "--attn-bench")) {
        std::vector<int> Ns;
        std::stringstream ss(argv[2]);
        std::string t;
        while (std::getline(ss, t, ',')) if (!t.empty()) Ns.push_back(std::stoi(t));
        const int dev = argc > 4 && !strcmp(argv[3], "--devices") ? atoi(argv[4]) : 0;
        cudaDeviceProp pr;
        cudaGetDeviceProperties(&pr, dev);
        printf("device %d: %s (sm_%d%d, %d SMs, %d MHz)\n", dev, pr.name, pr.major, pr.minor, pr.multiProcessorCount, pr.clockRate / 1000);
        return attn_bench(Ns, dev);
    }
    std::vector<std::string> images;
    nvit::opts o;
    std::string vocab, dlls;
    int min_tok = 1024, runs = 3, ref_dev = -1, ref_runs = 2;
    bool ref = false, native = true, oracle = false;
    int borrow_mb = 0;
    bool borrow_busy = false, borrow_race = false;
    std::vector<int> dump_layers = {0, 13, 26};
    try {
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            auto next = [&]() -> std::string {
                if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
                return argv[++i];
            };
            auto ints = [&](const std::string & s) {
                std::vector<int> v;
                std::stringstream ss(s);
                std::string t;
                while (std::getline(ss, t, ',')) if (!t.empty()) v.push_back(std::stoi(t));
                return v;
            };
            if (a == "--mmproj") o.mmproj = next();
            else if (a == "--image") images.push_back(next());
            else if (a == "--devices") o.devices = ints(next());
            else if (a == "--min-tokens") min_tok = std::stoi(next());
            else if (a == "--max-tokens") o.max_tokens = std::stoi(next());
            else if (a == "--runs") runs = std::stoi(next());
            else if (a == "--resident") o.host_weights = false;
            else if (a == "--chunks") o.chunks = std::stoi(next());
            else if (a == "--prio") o.stream_priority = std::stoi(next());
            else if (a == "--ref") ref = true;
            else if (a == "--vocab") vocab = next();
            else if (a == "--ref-dev") ref_dev = std::stoi(next());
            else if (a == "--ref-runs") ref_runs = std::stoi(next());
            else if (a == "--dlls") dlls = next();
            else if (a == "--dump-layers") dump_layers = ints(next());
            else if (a == "--no-native") native = false;
            else if (a == "--oracle") oracle = true;
            else if (a == "--borrow") borrow_mb = std::stoi(next());
            else if (a == "--borrow-busy") borrow_busy = true;
            else if (a == "--borrow-race") borrow_race = true;
            else if (a == "-h" || a == "--help") { printf("%s", usage); return 0; }
            else throw std::runtime_error("unknown argument " + a);
        }
        if (o.mmproj.empty() || images.empty()) { printf("%s", usage); return 1; }
#ifndef P100E_NVIT_REF
        if (ref) throw std::runtime_error("--ref: built without the llama.cpp fork (cmake/nvit.cmake)");
#endif
        for (int d : o.devices) {
            cudaDeviceProp pr;
            cudaGetDeviceProperties(&pr, d);
            printf("device %d: %s (sm_%d%d, %d SMs, %.0f MiB)\n", d, pr.name, pr.major, pr.minor, pr.multiProcessorCount,
                   pr.totalGlobalMem / 1048576.0);
        }
        std::vector<int> all_devs = o.devices;
        if (ref_dev < 0) ref_dev = o.devices.back();
        if (std::find(all_devs.begin(), all_devs.end(), ref_dev) == all_devs.end()) all_devs.push_back(ref_dev);

        using clk = std::chrono::steady_clock;
        auto ms = [](clk::time_point a, clk::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); };

        lend_gate gate;   // --borrow: the owner / borrower protocol of tp_model's prefill ring (outlives the encoder)
        std::unique_ptr<nvit::encoder> enc;
        if (native) {
            const auto u0 = vram_used(o.devices);
            const auto t0 = clk::now();
            enc = std::make_unique<nvit::encoder>(o);
            if (borrow_mb > 0) {   // stand-in for tp_model::idle_scratch (the LLM's prefill ring): lent at every encode
                std::vector<nvit::scratch_view> lend;
                for (int d : o.devices) {
                    nvit::scratch_view v;
                    v.dev = d;
                    v.bytes = (size_t) borrow_mb << 20;
                    cudaSetDevice(d);
                    if (cudaMalloc(&v.p, v.bytes) != cudaSuccess) throw std::runtime_error("--borrow: cudaMalloc failed");
                    lend.push_back(v);
                }
                enc->scratch_source = [lend, &gate]() {
                    std::shared_ptr<void> lease = gate.try_lend();
                    if (!lease) return std::vector<nvit::scratch_view>{};   // owner busy: the encoder uses its own
                    std::vector<nvit::scratch_view> v = lend;
                    for (auto & x : v) x.hold = lease;
                    return v;
                };
                printf("native: scratch lent by the caller (%d MiB per GPU, like the prefill ring)\n", borrow_mb);
            }
            const auto u1 = vram_used(o.devices);
            printf("native: load %.2f s, %d GPU(s), weights %s, max_tokens %d\n", ms(t0, clk::now()) / 1e3, enc->n_ranks(),
                   o.host_weights ? "in pinned host memory (uploaded per layer)" : "resident in VRAM", o.max_tokens);
            for (size_t r = 0; r < o.devices.size(); ++r)
                printf("native VRAM GPU%d: +%.1f MiB after load (permanent %.1f MiB), scratch for %d tokens %.1f MiB\n",
                       o.devices[r], (u1[r] - u0[r]) / 1048576.0, enc->permanent_bytes((int) r) / 1048576.0, o.max_tokens,
                       enc->scratch_bytes(o.max_tokens) / 1048576.0);
        }
        std::unique_ptr<nvit::encoder> ora;
        if (oracle) {
            nvit::opts oo = o;
            oo.devices = {o.devices[0]};
            oo.oracle = true;
            ora = std::make_unique<nvit::encoder>(oo);
            printf("oracle: fp32 path on device %d\n", oo.devices[0]);
        }
#ifdef P100E_NVIT_REF
        std::unique_ptr<nvit_ref> rf;
        if (ref) {
            if (vocab.empty()) throw std::runtime_error("--ref needs --vocab (any GGUF of the text model; vocab-only load)");
            const auto u0 = vram_used({ref_dev});
            const auto t0 = clk::now();
            rf = std::make_unique<nvit_ref>(o.mmproj, vocab, ref_dev, min_tok, o.max_tokens, dlls, false);
            const auto u1 = vram_used({ref_dev});
            printf("ref: mtmd on CUDA%d (DLLs %s), load + warmup %.2f s, VRAM +%.1f MiB\n", ref_dev, rf->dll_dir().c_str(),
                   ms(t0, clk::now()) / 1e3, (u1[0] - u0[0]) / 1048576.0);
        }
#endif
        fflush(stdout);

        for (const auto & path : images) {
            const std::string bytes = read_file(path);
            const auto a0 = clk::now();
            nvit::rgb_image rgb;
            std::string err;
            if (!nvit::decode_image(bytes.data(), bytes.size(), rgb, &err)) throw std::runtime_error(path + ": " + err);
            const auto a1 = clk::now();
            nvit::image im = nvit::preprocess(rgb, min_tok, o.max_tokens);
            const auto a2 = clk::now();
            const int ntok = im.n_tokens();
            printf("\nIMAGE %s: %dx%d -> %dx%d px, %d tokens (%dx%d), %d patches; decode %.1f ms, preprocess %.1f ms\n",
                   path.c_str(), rgb.w, rgb.h, im.nx, im.ny, ntok, im.nx / 32, im.ny / 32, 4 * ntok, ms(a0, a1), ms(a1, a2));
            fflush(stdout);
            std::map<std::string, std::vector<float>> mine;
            std::map<std::string, std::pair<int, int>> mshape;
            std::vector<float> emb;
            int tx = 0, ty = 0;
            if (native) {
                for (int r = 0; r < std::max(1, runs); ++r) {
                    nvit::timing t;
                    {
                        std::unique_ptr<lend_gate::busy_scope> busy;   // --borrow-busy: a prefill holds the ring
                        if (borrow_mb > 0 && borrow_busy && (r & 1)) busy = std::make_unique<lend_gate::busy_scope>(gate);
                        enc->encode(im, emb, tx, ty, &t);
                    }
                    uint64_t h = 1469598103934665603ull;
                    for (float v : emb) { uint32_t u; memcpy(&u, &v, 4); h = (h ^ u) * 1099511628211ull; }
                    printf("NATIVE run %d: %d tokens total %.1f ms (embed+qkv0 %.1f, blocks %.1f, merger %.1f GPU ms) hash %016llx%s\n",
                           r, ntok, t.total_ms, t.prep_ms, t.layers_ms, t.merger_ms, (unsigned long long) h,
                           borrow_mb > 0 ? (borrow_busy && (r & 1) ? " [owner busy: nothing lent]" : " [scratch offered]") : "");
                    if (borrow_mb > 0 && gate.lent()) printf("BORROW: lease still out after the encode returned (BUG)\n");
                    fflush(stdout);
                }
                if (borrow_mb > 0 && borrow_race) {   // the owner (a prefill) arrives while an encode holds the lease
                    std::vector<float> e2;
                    int x2, y2;
                    double enc_ms = 0;
                    const auto r0 = clk::now();
                    std::thread th([&] { nvit::timing t; enc->encode(im, e2, x2, y2, &t); enc_ms = t.total_ms; });
                    while (!gate.lent() && ms(r0, clk::now()) < 2000) std::this_thread::yield();
                    const bool leased = gate.lent();
                    const auto b0 = clk::now();
                    double waited = 0;
                    {
                        lend_gate::busy_scope busy(gate);
                        waited = ms(b0, clk::now());
                    }
                    th.join();
                    printf("BORROW-RACE: encode %.1f ms; the owner arrived %.1f ms after the start %s the lease was out and "
                           "waited %.1f ms for it; result %s\n", enc_ms, ms(r0, b0), leased ? "while" : "but NOT while", waited,
                           e2 == emb ? "identical" : "DIFFERENT");
                }
                double s = 0, s2 = 0;
                uint64_t h = 1469598103934665603ull;   // FNV-1a over the f32 bits: bit-identity across configurations
                for (float v : emb) {
                    s += v; s2 += (double) v * v;
                    uint32_t u;
                    memcpy(&u, &v, 4);
                    h = (h ^ u) * 1099511628211ull;
                }
                printf("native embd: %dx%d tokens x %d, sum %.6f rms %.6f first %.6f %.6f %.6f hash %016llx\n", tx, ty,
                       enc->n_embd(), s, std::sqrt(s2 / emb.size()), emb[0], emb[1], emb[2], (unsigned long long) h);
            }
            std::map<std::string, std::vector<float>> orv;
            std::vector<float> oemb;
            auto keyname = [](const std::string & name, int layer) {
                if (name == "post_ln") return std::string("norm_b-27");
                if (name == "inp_pos_emb" || name == "embd") return name;
                return name + "-" + std::to_string(layer);
            };
            if (ora) {
                ora->dump_layers = dump_layers;
                ora->dump = [&](const std::string & name, int layer, int rows, int cols, const float * d) {
                    orv[keyname(name, layer)].assign(d, d + (size_t) rows * cols);
                };
                int otx, oty;
                const auto o0 = clk::now();
                ora->encode(im, oemb, otx, oty, nullptr);
                printf("ORACLE: %dx%d tokens in %.1f ms\n", otx, oty, ms(o0, clk::now()));
                if (native) print_cmp("ORA", "embd", compare(emb.data(), oemb.data(), tx * ty, enc->n_embd()));
            }
#ifdef P100E_NVIT_REF
            if (rf) {
                std::vector<std::string> names = {"inp_pos_emb", "norm_b-27"};
                for (int l : dump_layers)
                    for (const char * n : {"ln1", "Qcur_rope", "Kcur_rope", "kqv_out", "ffn_inp", "layer_out"})
                        names.push_back(std::string(n) + "-" + std::to_string(l));
                nvit_ref_result rr;
                rf->encode(bytes, ref_runs, names, rr);
                printf("REF: %dx%d tokens, encode", rr.tx, rr.ty);
                for (double v : rr.encode_ms) printf(" %.1f", v);
                printf(" ms\n");
                // preprocessing: bit-exact?
                if (rr.inp_raw.size() == im.data.size()) {
                    size_t ndiff = 0;
                    double mx = 0;
                    for (size_t i = 0; i < im.data.size(); ++i)
                        if (im.data[i] != rr.inp_raw[i]) { ++ndiff; mx = std::max(mx, (double) std::fabs(im.data[i] - rr.inp_raw[i])); }
                    printf("PREPROCESS vs mtmd: %zu values, %zu differ (max |d| %.3g) -> %s\n", im.data.size(), ndiff, mx,
                           ndiff == 0 ? "BIT-EXACT" : "DIFFERENT");
                } else {
                    printf("PREPROCESS vs mtmd: size mismatch (mine %zu, mtmd %zu)\n", im.data.size(), rr.inp_raw.size());
                }
                if (native) {
                    if (rr.tx != tx || rr.ty != ty) printf("GRID MISMATCH: native %dx%d, mtmd %dx%d\n", tx, ty, rr.tx, rr.ty);
                    else print_cmp("CMP", "embd", compare(emb.data(), rr.embd.data(), tx * ty, enc->n_embd()));
                    // per-layer: one dump-mode run of the native encoder
                    enc->dump_layers = dump_layers;
                    enc->dump = [&](const std::string & name, int layer, int rows, int cols, const float * d) {
                        const std::string key = keyname(name, layer);
                        mine[key].assign(d, d + (size_t) rows * cols);
                        mshape[key] = {rows, cols};
                    };
                    std::vector<float> emb2;
                    int tx2, ty2;
                    enc->encode(im, emb2, tx2, ty2, nullptr);
                    enc->dump = nullptr;
                    print_cmp("CMP", "embd(dump run)", compare(emb2.data(), rr.embd.data(), tx * ty, enc->n_embd()));
                    for (const auto & n : names) {
                        auto it = rr.t.find(n);
                        auto jt = mine.find(n);
                        if (it == rr.t.end() || jt == mine.end()) { printf("CMP %-14s missing (%s)\n", n.c_str(), it == rr.t.end() ? "ref" : "native"); continue; }
                        if ((size_t) it->second.rows * it->second.cols != jt->second.size()) {
                            printf("CMP %-14s shape mismatch ref %dx%d native %zu\n", n.c_str(), it->second.rows, it->second.cols, jt->second.size());
                            continue;
                        }
                        print_cmp("CMP", n, compare(jt->second.data(), it->second.data.data(), it->second.rows, it->second.cols));
                        auto ot = orv.find(n);
                        if (ot != orv.end() && ot->second.size() == jt->second.size()) {
                            print_cmp("ORA", n, compare(jt->second.data(), ot->second.data(), it->second.rows, it->second.cols));
                            print_cmp("ORR", n, compare(it->second.data.data(), ot->second.data(), it->second.rows, it->second.cols));
                        }
                    }
                    if (!oemb.empty()) print_cmp("ORR", "embd", compare(rr.embd.data(), oemb.data(), tx * ty, enc->n_embd()));
                }
            }
#endif
            fflush(stdout);
        }
        if (native) {
            const auto u = vram_used(o.devices);
            for (size_t r = 0; r < o.devices.size(); ++r)
                printf("native VRAM GPU%d at exit: process total %.1f MiB, encoder permanent %.1f MiB (own scratch included)\n",
                       o.devices[r], u[r] / 1048576.0, enc->permanent_bytes((int) r) / 1048576.0);
        }
    } catch (const std::exception & e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
