#include <chrono>
// p100e-fast-matvec-bench: p100e::fast::matvec on the exact tensor-parallel decode shards of a qwen35 GGUF.
//
// usage: p100e-fast-matvec-bench model.gguf [--rank 0] [--ranks 2] [--reps 40] [--tmax 4] [--sweep] [--only str]
//                                [--skip-big] [--variant v]
// Slices every decode weight of rank `rank` exactly like tp_model::load_rows (same row / input-column ranges),
// groups the slices by (tensor class, type, n_in, n_out), and for one real slice per group (the first layer that
// has it) checks matvec against dequant + cublasSgemm and times it for T = 1..tmax.  Times are per matvec() call
// (activation quantize + GEMV, what the engine pays), with the weight rotated over enough copies that every call
// reads it from DRAM (> 6x the 4 MB P100 L2), back to back on one stream.  The summary weights each group by its
// count per token: total us per token and the average GB/s over the decode weights.
// --sweep additionally times every applicable T = 1 kernel variant (matvec_wr.cu) per group.
// --csweep: per group with a warp-rows layout and T = 2..tmax, the multi-column GEMV alone (prequantized, what the MTP
// verify pays) for wrcp and every wrm geometry / K-phase split (matvec_wrm.cu), with the number of outputs whose bits
// differ from the T = 1 call on that column (CSWEEP lines; any difference is a failure), and per T the step totals for
// wrcp, the default wrm and the best wrm geometry per shape (CSUM lines). --wrm 0 runs the main table on wrcp.
// Per group also (INV line): tinv = outputs of the T = tmax call whose column j differs (bits) from the T = 1 call on
// column j, pqbits = outputs of matvec_prequantized (T = 1) that differ from matvec (T = 1), pq us = the prequantized
// GEMV alone (what the fused decode pays, its producers write the a16 activation). CAT lines: the concatenated weights
// of the fused decode (tp_model::load_set: parts of one input with the same type and n_in), prequantized, T = 1; with
// --sweep every wrq (501..524) and wr (1..28) variant on them (bits vs the default). FUSED TOTAL = the fused decode's
// matvec time per token (concatenated launches + the remaining single weights, prequantized).
#include "fast/matvec.cuh"
#include "fast/matvec_internal.cuh"
#include "gguf.h"
#include "convert.cuh"
#include <cublas_v2.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <array>
#include <map>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA error %s at %s:%d: %s\n", \
    cudaGetErrorName(e_), __FILE__, __LINE__, cudaGetErrorString(e_)); exit(2); } } while (0)

using namespace p100e;
using ranges = std::vector<std::pair<int64_t, int64_t>>;

struct entry { std::string cls, name; ranges rows, cols; int set = -1; };   // set: weights reading the same input (-1: alone)
struct group { entry e; uint32_t type; int64_t n_in, n_out; size_t bytes; int count; };

static std::string bn(int il, const char * s) { return "blk." + std::to_string(il) + "." + s; }

// host slice exactly like tp_model::load_rows
static std::vector<uint8_t> slice(const gguf_tensor * t, const ranges & rows, const ranges & cols, int64_t & n_in, int64_t & n_out) {
    const qtype_info & ti = get_type_info(t->type);
    const int64_t tin = t->ne[0], tout = t->ne[1] * t->ne[2] * t->ne[3];
    const size_t src_row = (size_t) (tin / ti.blck) * ti.size;
    ranges cr = cols.empty() ? ranges{{0, tin}} : cols;
    ranges rr = rows.empty() ? ranges{{0, tout}} : rows;
    n_in = 0; for (auto & c : cr) n_in += c.second - c.first;
    n_out = 0; for (auto & p : rr) n_out += p.second - p.first;
    const size_t dst_row = (size_t) (n_in / ti.blck) * ti.size;
    std::vector<uint8_t> h((size_t) n_out * dst_row);
    size_t o = 0;
    for (auto & p : rr) for (int64_t row = p.first; row < p.second; ++row) {
        const uint8_t * src = t->data + (size_t) row * src_row;
        for (auto & c : cr) {
            const size_t b0 = (size_t) (c.first / ti.blck) * ti.size, b1 = (size_t) (c.second / ti.blck) * ti.size;
            memcpy(h.data() + o, src + b0, b1 - b0);
            o += b1 - b0;
        }
    }
    return h;
}

template <class F> static float time_ms(cudaStream_t s, int reps, F && fn) {
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    for (int i = 0; i < 5; ++i) fn();
    CK(cudaStreamSynchronize(s));
    CK(cudaEventRecord(a, s));
    for (int i = 0; i < reps; ++i) fn();
    CK(cudaEventRecord(b, s));
    CK(cudaEventSynchronize(b));
    float ms = 0; CK(cudaEventElapsedTime(&ms, a, b));
    CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
    return ms / reps;
}

// pure streaming read (calibration of the achievable DRAM read bandwidth): grid-stride uint4 loads, U in flight
template <int U>
static __global__ void __launch_bounds__(256) memread_k(const uint4 * __restrict__ p, const size_t n16, unsigned * out) {
    unsigned acc = 0;
    const size_t stride = (size_t) gridDim.x * blockDim.x;
    size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    for (; i + (U - 1) * stride < n16; i += U * stride) {
        uint4 v[U];
#pragma unroll
        for (int u = 0; u < U; ++u) v[u] = __ldg(p + i + u * stride);
#pragma unroll
        for (int u = 0; u < U; ++u) acc ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
    }
    for (; i < n16; i += stride) { const uint4 v = __ldg(p + i); acc ^= v.x ^ v.y ^ v.z ^ v.w; }
    if (acc == 0x12345678u) out[0] = acc;
}

static void memread_calibration(cudaStream_t s, int reps, int nsm) {
    for (size_t mb : {25, 521}) {
        const size_t bytes = mb << 20;
        const int C = mb < 200 ? 3 : 1;
        std::vector<uint4 *> buf(C);
        for (auto & b : buf) { CK(cudaMalloc(&b, bytes)); CK(cudaMemset(b, 1, bytes)); }
        unsigned * out; CK(cudaMalloc(&out, 64));
        for (int bpsm : {4, 8, 16}) for (int U : {1, 4}) {
            int k = 0;
            const unsigned grid = (unsigned) (nsm * bpsm);
            const float ms = time_ms(s, reps, [&] {
                if (U == 1) memread_k<1><<<grid, 256, 0, s>>>(buf[k], bytes / 16, out);
                else        memread_k<4><<<grid, 256, 0, s>>>(buf[k], bytes / 16, out);
                k = (k + 1) % C;
            });
            printf("MEMREAD %4zu MB  blocks/SM %2d  unroll %d : %8.2f us  %6.1f GB/s\n", mb, bpsm, U, ms * 1000.0, bytes / (ms * 1e-3) / 1e9);
        }
        for (auto & b : buf) cudaFree(b);
        cudaFree(out);
    }
}


// in-situ ceiling: a plain streaming read of exactly `bytes` (same rotation over copies as the GEMV timing, so every
// call reads DRAM), 8 blocks/SM x 256 threads, uint4 loads.  Returns us per call.
static double memread_same_us(cudaStream_t s, int reps, size_t bytes) {
    int dev = 0, nsm = 0;
    CK(cudaGetDevice(&dev)); CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev));
    const int C = std::max<int>(1, (int) ((24u << 20) / bytes) + 1);
    std::vector<uint4 *> buf(C);
    for (auto & b : buf) { CK(cudaMalloc(&b, bytes)); CK(cudaMemset(b, 1, bytes)); }
    unsigned * out; CK(cudaMalloc(&out, 64));
    int k = 0;
    const float ms = time_ms(s, reps, [&] { memread_k<1><<<(unsigned) (nsm * 8), 256, 0, s>>>(buf[k], bytes / 16, out); k = (k + 1) % C; });
    for (auto & b : buf) cudaFree(b);
    cudaFree(out);
    return ms * 1000.0;
}

static double max_rel(const std::vector<float> & y, const std::vector<float> & ref) {
    double mx = 0, me = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        mx = std::max(mx, (double) std::fabs(ref[i]));
        const double e = std::fabs((double) y[i] - ref[i]);
        me = std::isfinite(y[i]) ? std::max(me, e) : INFINITY;
    }
    return mx > 0 ? me / mx : me;
}

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s model.gguf [--rank r] [--ranks R] [--reps N] [--tmax T] [--sweep] [--only s] [--skip-big] [--variant v]\n", argv[0]); return 1; }
    int rank = 0, R = 2, reps = 40, tmax = 4, force_variant = 0;
    bool sweep = false, skip_big = false, calib = false, csweep = false, ptx_ab = false, ptx_env = false;
    int ptx_rounds = 5;
    double warm_s = 0;
    int wrm_main = 1;
    std::string only;
    for (int i = 2; i < argc; ++i) {
        if (!strcmp(argv[i], "--rank") && i + 1 < argc) rank = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ranks") && i + 1 < argc) R = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--tmax") && i + 1 < argc) tmax = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--sweep")) sweep = true;
        else if (!strcmp(argv[i], "--csweep")) csweep = true;
        else if (!strcmp(argv[i], "--wrm") && i + 1 < argc) wrm_main = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--skip-big")) skip_big = true;
        else if (!strcmp(argv[i], "--calib")) calib = true;
        else if (!strcmp(argv[i], "--only") && i + 1 < argc) only = argv[++i];
        else if (!strcmp(argv[i], "--variant") && i + 1 < argc) force_variant = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ptx")) ptx_ab = true;           // A/B the inline-PTX mv_wrp (levels 0 / 1 / 2) per shape
        else if (!strcmp(argv[i], "--ptx-env")) ptx_env = true;      // leave P100E_MVWRP_Q* from the environment in force
        else if (!strcmp(argv[i], "--ptx-rounds") && i + 1 < argc) ptx_rounds = std::max(1, atoi(argv[++i]));
        else if (!strcmp(argv[i], "--warm") && i + 1 < argc) warm_s = atof(argv[++i]);
    }
    tmax = std::max(1, std::min(tmax, 8));

    gguf_file g(argv[1]);
    const std::string a = g.get_str("general.architecture", "");
    const int E = (int) g.get_int(a + ".embedding_length", 0);
    const int L = (int) (g.get_int(a + ".block_count", 0) - g.get_int(a + ".nextn_predict_layers", 0));
    const int FF = (int) g.get_int(a + ".feed_forward_length", 0);
    const int H = (int) g.get_int(a + ".attention.head_count", 0), HKV = (int) g.get_int(a + ".attention.head_count_kv", 0);
    const int HD = (int) g.get_int(a + ".attention.key_length", 0);
    const int HK = (int) g.get_int(a + ".ssm.group_count", 0), HV = (int) g.get_int(a + ".ssm.time_step_rank", 0);
    const int DK = (int) g.get_int(a + ".ssm.state_size", 0);
    const int DV = (int) (g.get_int(a + ".ssm.inner_size", 0) / std::max(1, HV));
    const int fai = (int) g.get_int(a + ".full_attention_interval", 4);
    const int VOCAB = (int) g.find("token_embd.weight")->ne[1];
    const int Hl = H / R, HKVl = HKV / R, HKl = HK / R, FFl = FF / R, VL = VOCAB / R;
    const int r = rank;
    (void) E;

    // ---- decode weights of rank r, as tp_model loads them ----
    std::vector<entry> ents;
    auto rows_of = [](int64_t b, int64_t n) { return ranges{{b, b + n}}; };
    ents.push_back({"output.weight", "output.weight", rows_of((int64_t) r * VL, VL), {}});
    for (int il = 0; il < L; ++il) {
        ents.push_back({"ffn_gate.weight", bn(il, "ffn_gate.weight"), rows_of((int64_t) r * FFl, FFl), {}, 2*il});
        ents.push_back({"ffn_up.weight", bn(il, "ffn_up.weight"), rows_of((int64_t) r * FFl, FFl), {}, 2*il});
        ents.push_back({"ffn_down.weight", bn(il, "ffn_down.weight"), {}, {{(int64_t) r * FFl, (int64_t) (r + 1) * FFl}}});
        if ((il + 1) % fai == 0) {
            ents.push_back({"attn_q.weight", bn(il, "attn_q.weight"), rows_of((int64_t) r * Hl * 2 * HD, (int64_t) Hl * 2 * HD), {}, 2*il + 1});
            ents.push_back({"attn_k.weight", bn(il, "attn_k.weight"), rows_of((int64_t) r * HKVl * HD, (int64_t) HKVl * HD), {}, 2*il + 1});
            ents.push_back({"attn_v.weight", bn(il, "attn_v.weight"), rows_of((int64_t) r * HKVl * HD, (int64_t) HKVl * HD), {}, 2*il + 1});
            ents.push_back({"attn_output.weight", bn(il, "attn_output.weight"), {}, {{(int64_t) r * Hl * HD, (int64_t) (r + 1) * Hl * HD}}});
        } else {
            ranges qkv, vh, vrows;
            qkv.push_back({(int64_t) r * HKl * DK, (int64_t) (r + 1) * HKl * DK});
            qkv.push_back({(int64_t) HK * DK + r * HKl * DK, (int64_t) HK * DK + (r + 1) * HKl * DK});
            for (int gi = 0; gi < HV / HK; ++gi) {
                const int64_t h0 = (int64_t) gi * HK + r * HKl;
                qkv.push_back({2LL * HK * DK + h0 * DV, 2LL * HK * DK + (h0 + HKl) * DV});
                vh.push_back({h0, h0 + HKl});
                vrows.push_back({h0 * DV, (h0 + HKl) * DV});
            }
            ents.push_back({"attn_qkv.weight", bn(il, "attn_qkv.weight"), qkv, {}, 2*il + 1});
            ents.push_back({"attn_gate.weight", bn(il, "attn_gate.weight"), vrows, {}, 2*il + 1});
            ents.push_back({"ssm_beta.weight", bn(il, "ssm_beta.weight"), vh, {}, 2*il + 1});
            ents.push_back({"ssm_alpha.weight", bn(il, "ssm_alpha.weight"), vh, {}, 2*il + 1});
            ents.push_back({"ssm_out.weight", bn(il, "ssm_out.weight"), {}, vrows});
        }
    }

    std::map<std::string, group> groups;
    std::vector<std::string> order;
    std::vector<std::string> ent_key;
    std::vector<int64_t> ent_nin, ent_nout;
    std::vector<uint32_t> ent_type;
    for (auto & e : ents) {
        const gguf_tensor * t = g.find(e.name);
        if (!t) { fprintf(stderr, "missing %s\n", e.name.c_str()); return 1; }
        const qtype_info & ti = get_type_info(t->type);
        int64_t n_in = 0, n_out = 0;
        for (auto & c : (e.cols.empty() ? ranges{{0, t->ne[0]}} : e.cols)) n_in += c.second - c.first;
        for (auto & p : (e.rows.empty() ? ranges{{0, t->ne[1]}} : e.rows)) n_out += p.second - p.first;
        const std::string key = e.cls + "|" + ti.name + "|" + std::to_string(n_in) + "x" + std::to_string(n_out);
        ent_key.push_back(key); ent_nin.push_back(n_in); ent_nout.push_back(n_out); ent_type.push_back(t->type);
        auto it = groups.find(key);
        if (it == groups.end()) {
            groups[key] = group{e, t->type, n_in, n_out, (size_t) (n_in / ti.blck) * ti.size * (size_t) n_out, 1};
            order.push_back(key);
        } else {
            it->second.count++;
        }
    }

    // the fused decode's launches (tp_model::load_set): per input set, parts with the same type and n_in (n_out % 4 == 0,
    // 16-byte aligned offsets) concatenated; everything else one launch per weight
    struct cat_group { std::vector<size_t> parts; uint32_t type; int64_t n_in, n_out; int count; };
    std::map<std::string, cat_group> cats;
    std::vector<std::string> cat_order;
    std::map<std::string, int> single_count;   // group key -> launches of that weight alone in the fused decode
    {
        std::map<int, std::vector<size_t>> sets;
        for (size_t i = 0; i < ents.size(); ++i) {
            if (ents[i].set < 0) single_count[ent_key[i]]++;
            else sets[ents[i].set].push_back(i);
        }
        for (auto & [sid, idx] : sets) {
            std::vector<std::vector<size_t>> gs;
            for (size_t i : idx) {
                int gi = -1;
                const bool can = fast::has_fast_path(ent_type[i]) && ent_nout[i] % 4 == 0;
                for (int k = 0; can && k < (int) gs.size() && gi < 0; ++k) {
                    const size_t a0 = gs[k][0];
                    if (ent_type[a0] == ent_type[i] && ent_nin[a0] == ent_nin[i] && ent_nout[a0] % 4 == 0) gi = k;
                }
                if (gi >= 0) {
                    size_t off = 0;
                    const qtype_info & ti = get_type_info(ent_type[i]);
                    for (size_t j : gs[gi]) off += (size_t) (ent_nin[j] / ti.blck) * ti.size * (size_t) ent_nout[j];
                    if (off % 16) gi = -1;
                }
                if (gi >= 0) gs[gi].push_back(i); else gs.push_back({i});
            }
            for (auto & gp : gs) {
                if (gp.size() == 1) { single_count[ent_key[gp[0]]]++; continue; }
                std::string key = "cat:";
                int64_t n_tot = 0;
                for (size_t j : gp) { key += (j == gp[0] ? "" : "+") + ents[j].cls.substr(0, ents[j].cls.find('.')); n_tot += ent_nout[j]; }
                key += std::string("|") + get_type_info(ent_type[gp[0]]).name + "|" + std::to_string(ent_nin[gp[0]]) + "x" + std::to_string(n_tot);
                auto it = cats.find(key);
                if (it == cats.end()) { cats[key] = cat_group{gp, ent_type[gp[0]], ent_nin[gp[0]], n_tot, 1}; cat_order.push_back(key); }
                else it->second.count++;
            }
        }
    }
    std::map<std::string, double> pq_us_of;
    std::map<std::string, std::array<double, 9>> pqT_def;   // --csweep: default T-column GEMV alone, us

    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    printf("device: %s (sm_%d%d, %d SMs)  rank %d of %d  reps %d\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount, r, R, reps);
    cudaStream_t s; CK(cudaStreamCreate(&s));
    cublasHandle_t cb; cublasCreate(&cb); cublasSetStream(cb, s); cublasSetMathMode(cb, CUBLAS_PEDANTIC_MATH);

    if (calib) memread_calibration(s, reps, prop.multiProcessorCount);
    auto smi = [](const char * tag) {
        printf("SMI %s\n", tag); fflush(stdout);
        system("nvidia-smi --query-gpu=index,power.limit,clocks.sm,clocks.mem,clocks_event_reasons.active --format=csv,noheader");
        fflush(stdout);
    };
    if (warm_s > 0) {   // streaming reads until the clocks are up (the P100-IdleWatcher idles the cards at 125 W, low clocks)
        const size_t wb = (size_t) 256 << 20;
        uint4 * wbuf; unsigned * wout;
        CK(cudaMalloc(&wbuf, wb)); CK(cudaMemset(wbuf, 1, wb)); CK(cudaMalloc(&wout, 64));
        const auto t0 = std::chrono::steady_clock::now();
        while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < warm_s) {
            for (int k = 0; k < 20; ++k) memread_k<1><<<(unsigned) (prop.multiProcessorCount * 8), 256, 0, s>>>(wbuf, wb / 16, wout);
            CK(cudaStreamSynchronize(s));
        }
        cudaFree(wbuf); cudaFree(wout);
    }
    smi("start");
    printf("%-18s %-5s %6s %6s %3s %8s %-8s | %9s %9s %9s %9s | %8s %8s %8s %8s | %7s\n", "class", "type", "n_in", "n_out", "cnt", "MB",
           "kernel", "rel T=1", "rel T=2", "rel T=3", "rel T=4", "us T=1", "us T=2", "us T=3", "us T=4", "GB/s T1");
    double tot_bytes = 0, tot_us[9] = {0}, tot_q_us = 0, tot_kus = 0;
    double csum_wrcp[9] = {0}, csum_def[9] = {0}, csum_best[9] = {0};   // --csweep totals (warp-rows and q8 groups)
    double cat_sep[9] = {0}, cat_par[9] = {0};                          // --csweep: concatenated sets at T >= 2
    fast::mv_tune.wrm = wrm_main;
    if (!ptx_env) for (int k = 0; k < 4; ++k) fast::mv_tune.wrx[k] = 0;   // baseline = the C++ kernels unless asked
    // inline-PTX mv_wrp type index of a layout (q3_K, q4_K, q5_K, q6_K), -1 = none
    auto wrx_idx = [](int L) { return L == fast::LAYOUT_Q3R ? 0 : L == fast::LAYOUT_Q4KR ? 1 : L == fast::LAYOUT_Q5KR ? 2 :
                                      L == fast::LAYOUT_Q6R3 ? 3 : -1; };
    auto bits_diff = [](const std::vector<float> & a, const std::vector<float> & b) {
        long long n = 0;
        for (size_t i = 0; i < a.size(); ++i) n += memcmp(&a[i], &b[i], 4) != 0;
        return n;
    };
    int fails = 0;
    std::vector<std::string> sweep_lines;
    float quant_us_5120 = 0;
    for (auto & key : order) {
        group & G = groups[key];
        const gguf_tensor * t = g.find(G.e.name);
        const qtype_info & ti = get_type_info(G.type);
        if (!only.empty() && G.e.cls.find(only) == std::string::npos && std::string(ti.name).find(only) == std::string::npos) continue;
        if (skip_big && G.bytes > 256u * 1024 * 1024) continue;
        int64_t n_in, n_out;
        std::vector<uint8_t> h = slice(t, G.e.rows, G.e.cols, n_in, n_out);
        const size_t bytes = h.size();
        const int C = bytes >= (size_t) 200 << 20 ? 1 : (int) std::min<size_t>(512, std::max<size_t>(2, ((size_t) 48 << 20) / bytes + 1));

        void * d_raw; float *d_w, *d_x, *d_ref, *d_y; void * d_scr;
        CK(cudaMalloc(&d_raw, bytes));
        CK(cudaMemcpy(d_raw, h.data(), bytes, cudaMemcpyHostToDevice));
        // packed copies laid out with the loader's row stride (tp_model::pad_slice: padded Q6_K rows, P100E_Q6_PAD);
        // d_raw keeps the raw rows for the reference. GB/s below count the raw weight bytes.
        const size_t rb0 = bytes / (size_t) n_out, st = fast::pack_row_stride(G.type, n_in, n_out);
        std::vector<uint8_t> hpad;
        if (st != rb0) {
            hpad.assign(st * (size_t) n_out, 0);
            for (int64_t i = 0; i < n_out; ++i) memcpy(hpad.data() + (size_t) i * st, h.data() + (size_t) i * rb0, rb0);
        }
        std::vector<void *> d_pack(C);
        std::vector<fast::packed_weight> pw(C);
        for (int c = 0; c < C; ++c) {
            if (st != rb0) {
                CK(cudaMalloc(&d_pack[c], hpad.size()));
                CK(cudaMemcpy(d_pack[c], hpad.data(), hpad.size(), cudaMemcpyHostToDevice));
            } else {
                CK(cudaMalloc(&d_pack[c], bytes));
                CK(cudaMemcpy(d_pack[c], d_raw, bytes, cudaMemcpyDeviceToDevice));
            }
            pw[c] = fast::pack_weight(G.type, d_pack[c], n_in, n_out, s, st);
        }
        CK(cudaMalloc(&d_w, (size_t) n_in * n_out * 4));
        CK(cudaMalloc(&d_x, (size_t) 8 * n_in * 4)); CK(cudaMalloc(&d_ref, (size_t) 8 * n_out * 4)); CK(cudaMalloc(&d_y, (size_t) 8 * n_out * 4));
        CK(cudaMalloc(&d_scr, fast::matvec_scratch_bytes(n_in, 8)));
        std::vector<float> hx((size_t) 8 * n_in);
        srand(1234 + (unsigned) G.type);
        for (auto & v : hx) { float u = 0; for (int k = 0; k < 4; ++k) u += rand() / (float) RAND_MAX - 0.5f; v = u * 1.7f; }
        CK(cudaMemcpy(d_x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice));
        // f32 weights (a Q6_K model's ssm_beta / ssm_alpha): ggml has no to-fp32 converter for F32 (nullptr; calling
        // it was the whole-shard bench's segfault, rc 139, right after attn_gate), the raw rows are the reference already
        if (G.type == GGML_TYPE_F32) CK(cudaMemcpyAsync(d_w, d_raw, (size_t) n_in * n_out * 4, cudaMemcpyDeviceToDevice, s));
        else if (const auto to_f32 = ggml_get_to_fp32_cuda((ggml_type) G.type)) to_f32(d_raw, d_w, n_in * n_out, s);
        else { fprintf(stderr, "%s: no ggml to-fp32 converter for type %s\n", G.e.cls.c_str(), ti.name); return 1; }
        CK(cudaStreamSynchronize(s));
        const float one = 1.0f, zero = 0.0f;

        auto check = [&](int T) {
            cublasSgemm(cb, CUBLAS_OP_T, CUBLAS_OP_N, (int) n_out, T, (int) n_in, &one, d_w, (int) n_in, d_x, (int) n_in, &zero, d_ref, (int) n_out);
            CK(cudaMemset(d_y, 0xff, (size_t) 8 * n_out * 4));
            fast::matvec(pw[0], d_x, d_y, T, d_scr, s);
            CK(cudaStreamSynchronize(s)); CK(cudaGetLastError());
            std::vector<float> y((size_t) T * n_out), rf((size_t) T * n_out);
            CK(cudaMemcpy(y.data(), d_y, y.size() * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(rf.data(), d_ref, rf.size() * 4, cudaMemcpyDeviceToHost));
            return max_rel(y, rf);
        };
        auto timeit = [&](int T) {
            int k = 0;
            return 1000.0 * time_ms(s, reps, [&] { fast::matvec(pw[k], d_x, d_y, T, d_scr, s); k = (k + 1) % C; });
        };

        fast::mv_tune.variant = force_variant;
        double rel[9] = {0}, us[9] = {0};
        for (int T = 1; T <= tmax; ++T) {
            rel[T] = check(T);
            if (!(rel[T] < 0.03)) ++fails;
            us[T] = timeit(T);
        }
        // GEMV alone at T = 1 (activation quantized once, then reused)
        fast::matvec(pw[0], d_x, d_y, 1, d_scr, s);
        fast::mv_tune.skip_quant = pw[0].layout != fast::LAYOUT_Q8R && pw[0].layout != fast::LAYOUT_RAW;
        const double kus = timeit(1);
        fast::mv_tune.skip_quant = false;
        tot_kus += kus * G.count;
        const char * kn = fast::kernel_name(pw[0]);
        printf("%-18s %-5s %6lld %6lld %3d %8.2f %-8s |", G.e.cls.c_str(), ti.name, (long long) n_in, (long long) n_out, G.count, bytes / 1e6, kn);
        for (int T = 1; T <= 4; ++T) printf(" %9.2e", T <= tmax ? rel[T] : 0.0);
        printf(" |");
        for (int T = 1; T <= 4; ++T) printf(" %8.2f", T <= tmax ? us[T] : 0.0);
        printf(" | %7.1f | gemv %8.2f us %7.1f GB/s\n", bytes / (us[1] * 1e-6) / 1e9, kus, bytes / (kus * 1e-6) / 1e9);
        fflush(stdout);
        tot_bytes += (double) bytes * G.count;
        for (int T = 1; T <= tmax; ++T) tot_us[T] += us[T] * G.count;

        // T-invariance (column j of the T = tmax call vs T = 1 on column j) and matvec_prequantized vs matvec (T = 1)
        {
            std::vector<float> yT((size_t) tmax * n_out), y1((size_t) n_out), yq((size_t) n_out);
            fast::matvec(pw[0], d_x, d_y, tmax, d_scr, s);
            CK(cudaStreamSynchronize(s));
            CK(cudaMemcpy(yT.data(), d_y, yT.size() * 4, cudaMemcpyDeviceToHost));
            long long tinv = 0, pqbits = 0;
            void * d_q; CK(cudaMalloc(&d_q, fast::matvec_scratch_bytes(n_in, 1)));
            for (int j = 0; j < tmax; ++j) {
                fast::matvec(pw[0], d_x + (size_t) j*n_in, d_y, 1, d_scr, s);
                CK(cudaStreamSynchronize(s));
                CK(cudaMemcpy(y1.data(), d_y, y1.size() * 4, cudaMemcpyDeviceToHost));
                for (int64_t i = 0; i < n_out; ++i) tinv += memcmp(&y1[i], &yT[(size_t) j*n_out + i], 4) != 0;
                if (j == 0) {
                    fast::matvec_quantize(d_x, n_in, 1, d_q, s);
                    fast::matvec_prequantized(pw[0], d_x, d_q, d_y, 1, s);
                    CK(cudaStreamSynchronize(s));
                    CK(cudaMemcpy(yq.data(), d_y, yq.size() * 4, cudaMemcpyDeviceToHost));
                    for (int64_t i = 0; i < n_out; ++i) pqbits += memcmp(&y1[i], &yq[i], 4) != 0;
                }
            }
            int k = 0;
            const double pq = 1000.0 * time_ms(s, reps, [&] { fast::matvec_prequantized(pw[k], d_x, d_q, d_y, 1, s); k = (k + 1) % C; });
            pq_us_of[key] = pq;
            cudaFree(d_q);
            const int L0 = pw[0].layout;   // q4r (4 warps at T = 1, 2 above), kq raw, generic: legacy kernels, not checked
            const bool inv_expected = L0 == fast::LAYOUT_Q4KR || L0 == fast::LAYOUT_Q3R || L0 == fast::LAYOUT_Q5KR ||
                                      L0 == fast::LAYOUT_Q6R2 || L0 == fast::LAYOUT_Q6R3 || L0 == fast::LAYOUT_Q8R;
            if (inv_expected && (tinv || pqbits)) ++fails;
            printf("INV %-18s %-5s %6lld %6lld  tinv(T=%d) %lld  pqbits %lld  pq us %8.2f  %7.1f GB/s\n", G.e.cls.c_str(), ti.name,
                   (long long) n_in, (long long) n_out, tmax, tinv, pqbits, pq, bytes / (pq * 1e-6) / 1e9);
            const double cu = memread_same_us(s, reps, bytes);
            printf("CEIL %-18s %-5s %6lld %6lld  pq us %8.2f  ceil %8.2f us %7.1f GB/s  pq/ceil %5.1f%%  pq of611 %5.1f%%\n", G.e.cls.c_str(), ti.name,
                   (long long) n_in, (long long) n_out, pq, cu, bytes / (cu * 1e-6) / 1e9, 100.0 * cu / pq, 100.0 * bytes / (pq * 1e-6) / 1e9 / 611.0);
            fflush(stdout);
            // inline-PTX mv_wrp vs the C++ one: prequantized (wrq) and fused-quantize (wrp, matvec T = 1) outputs must be
            // bit-identical; GB/s and % of the in-situ ceiling per level, the levels timed back to back
            const int xi = wrx_idx(L0);
            if (ptx_ab && xi >= 0) {
                void * d_q2; CK(cudaMalloc(&d_q2, fast::matvec_scratch_bytes(n_in, 1)));
                fast::matvec_quantize(d_x, n_in, 1, d_q2, s);
                std::vector<float> ref_pq((size_t) n_out), ref_f((size_t) n_out), yv((size_t) n_out);
                auto run_pq = [&](std::vector<float> & out) {
                    CK(cudaMemset(d_y, 0xff, (size_t) n_out * 4));
                    fast::matvec_prequantized(pw[0], d_x, d_q2, d_y, 1, s);
                    CK(cudaStreamSynchronize(s)); CK(cudaGetLastError());
                    CK(cudaMemcpy(out.data(), d_y, out.size() * 4, cudaMemcpyDeviceToHost));
                };
                auto run_f = [&](std::vector<float> & out) {
                    CK(cudaMemset(d_y, 0xff, (size_t) n_out * 4));
                    fast::matvec(pw[0], d_x, d_y, 1, d_scr, s);
                    CK(cudaStreamSynchronize(s)); CK(cudaGetLastError());
                    CK(cudaMemcpy(out.data(), d_y, out.size() * 4, cudaMemcpyDeviceToHost));
                };
                double u[4] = {1e30, 1e30, 1e30, 1e30};
                long long bq[4] = {0}, bf[4] = {0};
                for (int lv = 0; lv <= 3; ++lv) {
                    fast::mv_tune.wrx[xi] = lv;
                    if (lv == 0) { run_pq(ref_pq); run_f(ref_f); }
                    else { run_pq(yv); bq[lv] = bits_diff(yv, ref_pq); run_f(yv); bf[lv] = bits_diff(yv, ref_f); }
                }
                for (int rd = 0; rd < ptx_rounds; ++rd)   // interleaved rounds, min per level
                    for (int lv = 0; lv <= 3; ++lv) {
                        fast::mv_tune.wrx[xi] = lv;
                        int k = 0;
                        u[lv] = std::min(u[lv], 1000.0 * time_ms(s, reps, [&] { fast::matvec_prequantized(pw[k], d_x, d_q2, d_y, 1, s); k = (k + 1) % C; }));
                    }
                fast::mv_tune.wrx[xi] = 0;
                if (bq[1] || bq[2] || bq[3] || bf[1] || bf[2] || bf[3]) ++fails;
                for (int lv = 1; lv <= 3; ++lv)
                    printf("PTXAB %-18s %-5s %6lld %6lld L%d  us %8.2f -> %8.2f  GB/s %6.1f -> %6.1f  ceil%% %5.1f -> %5.1f  gain %+5.1f%%  bits pq %lld fused %lld  x%d\n",
                           G.e.cls.c_str(), ti.name, (long long) n_in, (long long) n_out, lv, u[0], u[lv], bytes / (u[0] * 1e-6) / 1e9,
                           bytes / (u[lv] * 1e-6) / 1e9, 100.0 * cu / u[0], 100.0 * cu / u[lv], 100.0 * (u[0] / u[lv] - 1.0), bq[lv], bf[lv],
                           single_count[key]);
                fflush(stdout);
                cudaFree(d_q2);
            }
        }

        // multi-column GEMV alone (prequantized, T = 2..tmax): wrcp / q8_wrt vs every wrm / q8_wrm geometry and K-phase
        // split, bits vs the T = 1 call per column
        if (csweep) {
            const int L0 = pw[0].layout;
            const size_t rb0 = bytes / (size_t) n_out;
            const bool wrl = L0 == fast::LAYOUT_Q4KR || L0 == fast::LAYOUT_Q3R || L0 == fast::LAYOUT_Q5KR ||
                             L0 == fast::LAYOUT_Q6R3 || (L0 == fast::LAYOUT_Q6R2 && rb0 % 8 == 0);
            const bool q8l = L0 == fast::LAYOUT_Q8R;
            if (wrl || q8l) {
                const fast::mv_tune_t saved = fast::mv_tune;
                fast::mv_tune.variant = 0;
                const int ng = (int) (n_in / 32);
                struct cfg { int kind, geo, kpg; };   // kind 0: wrcp / q8_wrt, 1: wrm / q8_wrm
                std::vector<cfg> cfgs = {{0, 0, 0}, {1, 0, 0}};
                if (wrl) {
                    for (int g2 : {1, 2, 3, 4, 6, 9, 10, 21, 22, 23, 24, 25, 26, 27}) cfgs.push_back({1, g2, 0});
                    if (n_out <= 4096) for (int g2 : {11, 12, 13}) cfgs.push_back({1, g2, 0});
                    if (n_in > 5120) for (int k = 96; k < ng; k += 32) cfgs.push_back({1, 1, k});
                } else {
                    for (int g2 = 31; g2 <= 47; ++g2) cfgs.push_back({1, g2, 0});
                }
                auto cname = [&](const cfg & c) -> const char * {
                    if (c.kind == 0) return wrl ? "wrcp" : "q8_wrt";
                    if (c.geo == 0) return wrl ? "wrm default" : "q8_wrm default";
                    return wrl ? fast::wrm_geo_name(c.geo) : fast::q8m_geo_name(c.geo);
                };
                auto apply = [&](const cfg & c) {
                    if (wrl) { fast::mv_tune.wrm = c.kind; fast::mv_tune.wrm_geo = c.geo; fast::mv_tune.wrm_kpg = c.kpg; }
                    else fast::mv_tune.q8_geo = c.kind == 0 ? -1 : c.geo;
                };
                std::vector<float> y1((size_t) tmax * n_out);   // T = 1 per column (reference bits)
                for (int j = 0; j < tmax; ++j) {
                    fast::matvec(pw[0], d_x + (size_t) j * n_in, d_y, 1, d_scr, s);
                    CK(cudaStreamSynchronize(s));
                    CK(cudaMemcpy(y1.data() + (size_t) j * n_out, d_y, (size_t) n_out * 4, cudaMemcpyDeviceToHost));
                }
                void * d_q; CK(cudaMalloc(&d_q, fast::matvec_scratch_bytes(n_in, 8)));
                for (int T = 2; T <= tmax; ++T) {
                    fast::matvec_quantize(d_x, n_in, T, d_q, s);
                    double best = 1e30; int best_i = -1;
                    for (size_t ci = 0; ci < cfgs.size(); ++ci) {
                        const cfg & c = cfgs[ci];
                        apply(c);
                        CK(cudaMemset(d_y, 0xff, (size_t) T * n_out * 4));
                        fast::matvec_prequantized(pw[0], d_x, d_q, d_y, T, s);
                        CK(cudaStreamSynchronize(s)); CK(cudaGetLastError());
                        std::vector<float> yT((size_t) T * n_out);
                        CK(cudaMemcpy(yT.data(), d_y, yT.size() * 4, cudaMemcpyDeviceToHost));
                        long long nbits = 0;
                        for (size_t i = 0; i < yT.size(); ++i) nbits += memcmp(&yT[i], &y1[i], 4) != 0;
                        if (nbits) ++fails;
                        int k = 0;
                        const double u = 1000.0 * time_ms(s, reps, [&] { fast::matvec_prequantized(pw[k], d_x, d_q, d_y, T, s); k = (k + 1) % C; });
                        printf("CSWEEP %-18s %-5s %6lld %6lld T=%d %-34s kpg %3d  us %8.2f  GB/s %7.1f  bits %lld\n", G.e.cls.c_str(),
                               ti.name, (long long) n_in, (long long) n_out, T, cname(c), c.kpg, u, bytes / (u * 1e-6) / 1e9, nbits);
                        fflush(stdout);
                        if (ci == 0) csum_wrcp[T] += u * G.count;
                        if (ci == 1) { csum_def[T] += u * G.count; pqT_def[key][T] = u; }
                        if (c.kind == 1 && u < best) { best = u; best_i = (int) ci; }
                    }
                    csum_best[T] += best * G.count;
                    const cfg & cb2 = cfgs[best_i];
                    printf("CBEST %-18s %-5s %6lld %6lld T=%d %-34s kpg %3d  us %8.2f\n", G.e.cls.c_str(), ti.name, (long long) n_in,
                           (long long) n_out, T, cname(cb2), cb2.kpg, best);
                }
                cudaFree(d_q);
                fast::mv_tune = saved;
            }
        }

        // activation quantize alone (once per n_in), for reference
        if (pw[0].layout != fast::LAYOUT_Q8R && pw[0].layout != fast::LAYOUT_RAW) {
            const int ng = (int) (n_in / 32);
            half2 * aq = (half2 *) d_scr; half2 * ads = aq + (size_t) ng * 16;
            const float qus = 1000.0f * time_ms(s, reps, [&] { fast::a16_quantize(d_x, aq, ads, n_in, ng, 1, true, s); });
            tot_q_us += qus * G.count;
            if (n_in == 5120) quant_us_5120 = qus;
        }

        if (sweep) {
            std::vector<int> vs;
            const int L0 = pw[0].layout;
            vs.push_back(-1);
            if (L0 == fast::LAYOUT_Q4KR || L0 == fast::LAYOUT_Q3R || L0 == fast::LAYOUT_Q6R2 || L0 == fast::LAYOUT_Q6R3 ||
                L0 == fast::LAYOUT_Q5KR) {
                for (int v = 1; v <= fast::wr_variant_count(); ++v) vs.push_back(v);
                if (n_out <= 2048) for (int v = 101; v <= 104; ++v) vs.push_back(v);
                vs.push_back(105);
                for (int v = 301; v <= 324; ++v) vs.push_back(v);
                for (int v = 501; v <= 524; ++v) vs.push_back(v);   // prequantized: timed as the GEMV alone
                if (n_out <= 2048) for (int v = 401; v <= 404; ++v) vs.push_back(v);
            } else if (L0 == fast::LAYOUT_Q8R) {
                for (int v : {201, 202, 203, 204, 211, 212, 213, 214, 215, 216, 217, 218}) vs.push_back(v);
            }
            // warp-rows variants (1..99 and the fused 301..399) sum each row in the same order: their outputs must be
            // bit-identical to variant 1's; "bits" reports the number of differing outputs (-1 = not compared).
            // Q8R: the q8wr variants (211..218) must give the legacy q8r_gemv's bits (v-1, the first in vs).
            std::vector<float> y_wr1, y_q8leg;
            for (int v : vs) {
                fast::mv_tune.variant = v;
                const double rl = check(1);
                long long nbits = -1;
                const bool wr_family = (v >= 1 && v < 100) || (v >= 300 && v < 400) || (v >= 500 && v < 600);
                if (L0 == fast::LAYOUT_Q8R && (v == -1 || (v >= 211 && v <= 218))) {
                    std::vector<float> yv((size_t) n_out);
                    CK(cudaMemcpy(yv.data(), d_y, yv.size() * 4, cudaMemcpyDeviceToHost));
                    if (v == -1) y_q8leg = yv;
                    else {
                        nbits = 0;
                        for (size_t i = 0; i < yv.size(); ++i) nbits += memcmp(&yv[i], &y_q8leg[i], 4) != 0;
                        if (nbits) ++fails;
                    }
                }
                if (wr_family) {
                    std::vector<float> yv((size_t) n_out);
                    CK(cudaMemcpy(yv.data(), d_y, yv.size() * 4, cudaMemcpyDeviceToHost));
                    if (v == 1) y_wr1 = yv;
                    if (!y_wr1.empty()) {
                        nbits = 0;
                        for (size_t i = 0; i < yv.size(); ++i) nbits += memcmp(&yv[i], &y_wr1[i], 4) != 0;
                        if (nbits) ++fails;
                    }
                }
                fast::mv_tune.skip_quant = v >= 500;
                const double u = timeit(1);
                fast::mv_tune.skip_quant = false;
                const char * nm = v < 0 ? "legacy" : v >= 300 ? fast::fused_variant_name(v) : v >= 200 ? fast::q8_variant_name(v) : v >= 100 ? fast::ks_variant_name(v) : fast::wr_variant_name(v);
                char line[256];
                snprintf(line, sizeof line, "SWEEP %-18s %-5s %6lld %6lld v%-4d %-22s rel %9.2e  us %8.2f  GB/s %7.1f  bits %lld", G.e.cls.c_str(), ti.name,
                         (long long) n_in, (long long) n_out, v, nm, rl, u, bytes / (u * 1e-6) / 1e9, nbits);
                printf("%s\n", line); fflush(stdout);
            }
            fast::mv_tune.variant = force_variant;
        }

        cudaFree(d_raw); for (void * p : d_pack) cudaFree(p);
        cudaFree(d_w); cudaFree(d_x); cudaFree(d_ref); cudaFree(d_y); cudaFree(d_scr);
    }
    // ---- the fused decode's concatenated weights (prequantized, T = 1) ----
    double fused_us = 0;
    for (auto & key : cat_order) {
        cat_group & CG = cats[key];
        const qtype_info & ti = get_type_info(CG.type);
        if (!only.empty() && key.find(only) == std::string::npos) continue;
        std::vector<uint8_t> h;
        for (size_t j : CG.parts) {
            int64_t a, b;
            std::vector<uint8_t> hp = slice(g.find(ents[j].name), ents[j].rows, ents[j].cols, a, b);
            h.insert(h.end(), hp.begin(), hp.end());
        }
        const int64_t n_in = CG.n_in, n_out = CG.n_out;
        const size_t bytes = h.size();
        const int C = (int) std::min<size_t>(512, std::max<size_t>(2, ((size_t) 48 << 20) / bytes + 1));
        void * d_raw; float *d_w, *d_x, *d_ref, *d_y; void * d_q;
        CK(cudaMalloc(&d_raw, bytes));
        CK(cudaMemcpy(d_raw, h.data(), bytes, cudaMemcpyHostToDevice));
        std::vector<void *> d_pack(C);
        std::vector<fast::packed_weight> pw(C);
        for (int c = 0; c < C; ++c) {
            CK(cudaMalloc(&d_pack[c], bytes));
            CK(cudaMemcpy(d_pack[c], d_raw, bytes, cudaMemcpyDeviceToDevice));
            pw[c] = fast::pack_weight(CG.type, d_pack[c], n_in, n_out, s);
        }
        CK(cudaMalloc(&d_w, (size_t) n_in * n_out * 4));
        CK(cudaMalloc(&d_x, (size_t) n_in * 4)); CK(cudaMalloc(&d_ref, (size_t) n_out * 4)); CK(cudaMalloc(&d_y, (size_t) n_out * 4));
        CK(cudaMalloc(&d_q, fast::matvec_scratch_bytes(n_in, 1)));
        std::vector<float> hx((size_t) n_in);
        srand(4321 + (unsigned) CG.type);
        for (auto & v : hx) { float u = 0; for (int k = 0; k < 4; ++k) u += rand() / (float) RAND_MAX - 0.5f; v = u * 1.7f; }
        CK(cudaMemcpy(d_x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice));
        // f32 concatenations (cat:ssm_beta+ssm_alpha on a Q6_K model): no ggml to-fp32 converter, as above
        if (CG.type == GGML_TYPE_F32) CK(cudaMemcpyAsync(d_w, d_raw, (size_t) n_in * n_out * 4, cudaMemcpyDeviceToDevice, s));
        else if (const auto to_f32 = ggml_get_to_fp32_cuda((ggml_type) CG.type)) to_f32(d_raw, d_w, n_in * n_out, s);
        else { fprintf(stderr, "%s: no ggml to-fp32 converter for type %s\n", key.c_str(), ti.name); return 1; }
        const float one = 1.0f, zero = 0.0f;
        cublasSgemm(cb, CUBLAS_OP_T, CUBLAS_OP_N, (int) n_out, 1, (int) n_in, &one, d_w, (int) n_in, d_x, (int) n_in, &zero, d_ref, (int) n_out);
        fast::matvec_quantize(d_x, n_in, 1, d_q, s);
        CK(cudaStreamSynchronize(s));
        std::vector<float> rf((size_t) n_out), y0((size_t) n_out), yv((size_t) n_out);
        CK(cudaMemcpy(rf.data(), d_ref, rf.size() * 4, cudaMemcpyDeviceToHost));
        auto run = [&](std::vector<float> & out) {
            CK(cudaMemset(d_y, 0xff, (size_t) n_out * 4));
            fast::matvec_prequantized(pw[0], d_x, d_q, d_y, 1, s);
            CK(cudaStreamSynchronize(s)); CK(cudaGetLastError());
            CK(cudaMemcpy(out.data(), d_y, out.size() * 4, cudaMemcpyDeviceToHost));
        };
        auto timepq = [&] {
            int k = 0;
            return 1000.0 * time_ms(s, reps, [&] { fast::matvec_prequantized(pw[k], d_x, d_q, d_y, 1, s); k = (k + 1) % C; });
        };
        fast::mv_tune.variant = force_variant;
        run(y0);
        const double rl = max_rel(y0, rf);
        if (!(rl < 0.03)) ++fails;
        const double u0 = timepq();
        fused_us += u0 * CG.count;
        printf("CAT %-40s %-5s %6lld %6lld %3d %8.2f %-6s | rel %9.2e | pq us %8.2f  %7.1f GB/s  (default v%d)\n", key.c_str(), ti.name,
               (long long) n_in, (long long) n_out, CG.count, bytes / 1e6, fast::kernel_name(pw[0]), rl, u0, bytes / (u0 * 1e-6) / 1e9,
               fast::default_pq_variant(pw[0].layout, CG.type, n_in, n_out));
        const double cu = memread_same_us(s, reps, bytes);
        printf("CATCEIL %-40s ceil %8.2f us %7.1f GB/s  pq/ceil %5.1f%%  pq of611 %5.1f%%\n", key.c_str(), cu, bytes / (cu * 1e-6) / 1e9,
               100.0 * cu / u0, 100.0 * bytes / (u0 * 1e-6) / 1e9 / 611.0);
        fflush(stdout);
        if (ptx_ab && wrx_idx(pw[0].layout) >= 0) {
            const int xi = wrx_idx(pw[0].layout);
            double u[4] = {1e30, 1e30, 1e30, 1e30};
            long long bq[4] = {0};
            for (int lv = 1; lv <= 3; ++lv) {
                fast::mv_tune.wrx[xi] = lv;
                run(yv); bq[lv] = bits_diff(yv, y0);
            }
            for (int rd = 0; rd < ptx_rounds; ++rd)
                for (int lv = 0; lv <= 3; ++lv) {
                    fast::mv_tune.wrx[xi] = lv;
                    u[lv] = std::min(u[lv], timepq());
                }
            fast::mv_tune.wrx[xi] = 0;
            if (bq[1] || bq[2] || bq[3]) ++fails;
            for (int lv = 1; lv <= 3; ++lv)
                printf("PTXAB %-40s %-5s L%d  us %8.2f -> %8.2f  GB/s %6.1f -> %6.1f  ceil%% %5.1f -> %5.1f  gain %+5.1f%%  bits pq %lld  x%d\n",
                       key.c_str(), ti.name, lv, u[0], u[lv], bytes / (u[0] * 1e-6) / 1e9, bytes / (u[lv] * 1e-6) / 1e9, 100.0 * cu / u[0],
                       100.0 * cu / u[lv], 100.0 * (u[0] / u[lv] - 1.0), bq[lv], CG.count);
            fflush(stdout);
        }
        if (csweep) {
            // T = 2..tmax: the verify's launch over the concatenation with per-part outputs vs separate launches on the
            // parts (row views); bits must match
            CK(cudaFree(d_x));
            CK(cudaMalloc(&d_x, (size_t) 8 * n_in * 4));
            std::vector<float> hx8((size_t) 8 * n_in);
            for (auto & v : hx8) { float u = 0; for (int k = 0; k < 4; ++k) u += rand() / (float) RAND_MAX - 0.5f; v = u * 1.7f; }
            CK(cudaMemcpy(d_x, hx8.data(), hx8.size() * 4, cudaMemcpyHostToDevice));
            void * d_q8; CK(cudaMalloc(&d_q8, fast::matvec_scratch_bytes(n_in, 8)));
            float * d_ys; CK(cudaMalloc(&d_ys, (size_t) 8 * n_out * 4));
            float * d_yp; CK(cudaMalloc(&d_yp, (size_t) 8 * n_out * 4));
            const size_t rbc = bytes / (size_t) n_out;
            for (int T = 2; T <= tmax; ++T) {
                fast::matvec_quantize(d_x, n_in, T, d_q8, s);
                std::vector<fast::mv_out> outs_s, outs_p;
                int64_t o = 0;
                for (size_t j : CG.parts) {   // part p's [T][n_p] block at offset T * (rows before p)
                    outs_s.push_back({d_ys + (size_t) T * o, ent_nout[j]});
                    outs_p.push_back({d_yp + (size_t) T * o, ent_nout[j]});
                    o += ent_nout[j];
                }
                auto separate = [&](int k) {
                    int64_t r0 = 0;
                    for (size_t p = 0; p < outs_s.size(); ++p) {
                        fast::packed_weight v = pw[k];
                        v.data = (char *) pw[k].data + (size_t) r0 * rbc; v.n_out = outs_s[p].n; v.bytes = rbc * (size_t) outs_s[p].n;
                        fast::matvec_prequantized(v, d_x, d_q8, outs_s[p].y, T, s);
                        r0 += outs_s[p].n;
                    }
                };
                CK(cudaMemset(d_ys, 0xff, (size_t) 8 * n_out * 4)); CK(cudaMemset(d_yp, 0xfe, (size_t) 8 * n_out * 4));
                separate(0);
                fast::matvec_prequantized_parts(pw[0], d_x, d_q8, outs_p.data(), (int) outs_p.size(), T, s);
                CK(cudaStreamSynchronize(s)); CK(cudaGetLastError());
                std::vector<float> ys((size_t) T * n_out), yp((size_t) T * n_out);
                CK(cudaMemcpy(ys.data(), d_ys, ys.size() * 4, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(yp.data(), d_yp, yp.size() * 4, cudaMemcpyDeviceToHost));
                long long nb2 = 0;
                for (size_t i = 0; i < ys.size(); ++i) nb2 += memcmp(&ys[i], &yp[i], 4) != 0;
                if (nb2) ++fails;
                int k = 0;
                const double us_s = 1000.0 * time_ms(s, reps, [&] { separate(k); k = (k + 1) % C; });
                k = 0;
                const double us_p = 1000.0 * time_ms(s, reps, [&] {
                    fast::matvec_prequantized_parts(pw[k], d_x, d_q8, outs_p.data(), (int) outs_p.size(), T, s); k = (k + 1) % C; });
                cat_sep[T] += us_s * CG.count; cat_par[T] += us_p * CG.count;
                printf("CATT %-40s T=%d  separate %8.2f us  parts launch %8.2f us  bits %lld\n", key.c_str(), T, us_s, us_p, nb2);
                fflush(stdout);
            }
            cudaFree(d_q8); cudaFree(d_ys); cudaFree(d_yp);
        }
        if (sweep && fast::matvec_uses_a16(pw[0])) {
            std::vector<int> vs;
            for (int v = 501; v <= 524; ++v) vs.push_back(v);
            for (int v = 1; v <= fast::wr_variant_count(); ++v) vs.push_back(v);
            for (int v : vs) {
                fast::mv_tune.variant = v;
                run(yv);
                long long nbits = 0;
                for (int64_t i = 0; i < n_out; ++i) nbits += memcmp(&yv[i], &y0[i], 4) != 0;
                if (nbits) ++fails;
                const double u = timepq();
                printf("CATSWEEP %-40s v%-4d %-22s us %8.2f  GB/s %7.1f  bits %lld\n", key.c_str(), v,
                       v >= 500 ? fast::fused_variant_name(v) : fast::wr_variant_name(v), u, bytes / (u * 1e-6) / 1e9, nbits);
                fflush(stdout);
            }
            fast::mv_tune.variant = force_variant;
        }
        cudaFree(d_raw); for (void * p : d_pack) cudaFree(p);
        cudaFree(d_w); cudaFree(d_x); cudaFree(d_ref); cudaFree(d_y); cudaFree(d_q);
    }
    if (only.empty()) {
        double singles = 0;
        int missing = 0;
        for (auto & [k, n] : single_count) {
            auto it = pq_us_of.find(k);
            if (it == pq_us_of.end()) { missing += n; continue; }
            singles += it->second * n;
        }
        printf("FUSED TOTAL T=1 (prequantized): %.3f ms per token = concatenated %.3f + single weights %.3f%s\n",
               (fused_us + singles) / 1000.0, fused_us / 1000.0, singles / 1000.0, missing ? " (some weights skipped)" : "");
    }

    printf("TOTAL per token (rank %d): weights %.1f MB\n", r, tot_bytes / 1e6);
    for (int T = 1; T <= tmax; ++T)
        printf("TOTAL T=%d: %.3f ms per step, %.1f GB/s average (matvec incl. activation quantize)\n", T, tot_us[T] / 1000.0, tot_bytes / (tot_us[T] * 1e-6) / 1e9);
    printf("TOTAL activation quantize alone (T=1): %.3f ms (%.2f us per call at n_in 5120)\n", tot_q_us / 1000.0, quant_us_5120);
    if (csweep)
        for (int T = 2; T <= tmax; ++T)
            printf("CSUM T=%d (every weight alone, GEMV per step): wrcp/q8_wrt %.3f ms, default %.3f ms, best per shape %.3f ms\n",
                   T, csum_wrcp[T] / 1000.0, csum_def[T] / 1000.0, csum_best[T] / 1000.0);
    if (csweep && only.empty()) {
        for (int T = 2; T <= tmax; ++T) {
            double singles = 0;
            for (auto & [k2, n] : single_count) {
                auto it = pqT_def.find(k2);
                if (it != pqT_def.end()) singles += it->second[T] * n;
            }
            printf("VERIFY TOTAL T=%d (fused verify matvecs per step, default kernels): sets as separate launches %.3f + singles %.3f = %.3f ms;"
                   " sets as parts launches %.3f + singles = %.3f ms\n", T, cat_sep[T] / 1000.0, singles / 1000.0,
                   (cat_sep[T] + singles) / 1000.0, cat_par[T] / 1000.0, (cat_par[T] + singles) / 1000.0);
        }
    }
    printf("TOTAL GEMV alone T=1: %.3f ms, %.1f GB/s\n", tot_kus / 1000.0, tot_bytes / (tot_kus * 1e-6) / 1e9);
    smi("end");
    printf("%s (%d failures; pass = rel < 3e-2)\n", fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}
