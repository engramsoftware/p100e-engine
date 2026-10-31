// p100e-fn-expert-server: node 1 of fn_model's remote expert tier (protocol in src/fn_rexp.h).
//   p100e-fn-expert-server <model.gguf> --connect host:port [--devices 0,1] [--slots a,b|auto] [--margin-mb 384]
//                          [--connect-s 3600] [--once] [--swap-gpus 0|0,1|none] [--no-keepalive] [--no-threads] [--no-one-copy] [--merge-gu] [--spin-sync] [--verbose]
// --swap-gpus: positions in --devices whose experts follow node 0's routing counts (default 0 = the first GPU only;
// the others hold a fixed set loaded once: the 1660 sits on a PCIe x4 link)
// Connects to node 0 (p100e-fn-run / fn_model with --rexp-listen), reports its GPUs, loads the Q4 routed experts node 0
// asks for (gate / up / down slices of blk.<il>.ffn_*_exps.weight, packed in place by fast::pack_weight exactly as
// fn_model's resident slots) and then answers one FFN request per layer per forward: each (row, expert) pair through the
// same grouped a16 kernels (quads of 4 rows per expert where the layout has them, then singles; the fold GEMM for
// experts with fold_min..fold_max rows), so a pair's output has the bits node 0's own grouped path would give on the
// same GPU architecture. No dense weights, no KV, nothing in host RAM but a bounce buffer and the request / reply rows.
// After node 0 quits it frees everything and waits for the next connection (--once: exits).
#include "fn_rexp.h"
#include "gguf.h"
#include "kernels.cuh"
#include "fast/matvec.cuh"
#include "fast/matvec_internal.cuh"
#include "fast/gemm.cuh"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <atomic>
#include <functional>
#include <memory>
#include <exception>

using namespace p100e;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) throw std::runtime_error(std::string("CUDA ") + cudaGetErrorString(e_) + " at " __FILE__ ":" + std::to_string(__LINE__)); } while (0)
#define CB(x) do { cublasStatus_t e_ = (x); if (e_ != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLAS error " + std::to_string((int) e_) + " at line " + std::to_string(__LINE__)); } while (0)

namespace {

__global__ void gather_rows_k(const float * src, const int32_t * idx, float * dst, int n) {   // = fn_model's
    const int j = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[(int64_t) j * n + i] = src[(int64_t) idx[j] * n + i];
}
unsigned nb256(int64_t n) { return (unsigned) ((n + 255) / 256); }
// keep-alive (--keepalive, default on): one block on a lowest-priority stream spins for ~us microseconds and touches a
// small buffer, relaunched by a host thread while serving. Without it a GeForce card between requests idles in P8
// (measured on the 3070: 210 MHz core / 405 MHz memory during a run), so every request ran at idle clocks
// (~2 ms of GPU time for ~25 (row, expert) pairs).
__global__ void keepalive_k(float * buf, long long cycles) {
    const long long t0 = clock64();
    float acc = 0.0f;
    int i = threadIdx.x;
    while (clock64() - t0 < cycles) { acc += buf[i & 4095]; i += 977; }
    if (acc == 1234.5f) buf[threadIdx.x] = acc;   // never true: keeps the loads
}

bool grouped_layout(int l) { return l == fast::LAYOUT_Q4KR || l == fast::LAYOUT_Q4R || l == fast::LAYOUT_Q10 || l == fast::LAYOUT_Q20 || l == fast::LAYOUT_IQ2SR; }
bool grouped_nc_layout(int l) { return l == fast::LAYOUT_Q4KR || l == fast::LAYOUT_Q4R || l == fast::LAYOUT_IQ2SR; }
void launch_grouped(int layout, const fast::mv_group * tab, int G, const fast::a16_args & a, cudaStream_t s, int nc = 1) {   // = fn_model's
    if (G <= 0) return;
    switch (layout) {
        case fast::LAYOUT_Q4KR:  fast::launch_q4kr_grouped_nc(tab, G, a, nc, s); break;
        case fast::LAYOUT_Q4R:   fast::launch_q4r_grouped_nc(tab, G, a, nc, s); break;
        case fast::LAYOUT_IQ2SR: fast::launch_iq2sr_grouped_nc(tab, G, a, nc, s); break;
        case fast::LAYOUT_Q10:   fast::launch_q10_grouped(tab, G, a, s); break;
        default:                 fast::launch_q20_grouped(tab, G, a, s); break;
    }
}
double gib(double b) { return b / 1073741824.0; }
std::string bn(int il, const char * n) { return "blk." + std::to_string(il) + "." + n; }

struct proj { uint32_t type = 0; size_t bytes = 0; int layout = 0; };   // one projection of one expert
struct layer_set {      // the experts of one served layer on one GPU
    int n = 0;
    char * base[3] = {};   // device [n][bytes] gate, up, down
    proj p[3];
    std::vector<int> ids;
};
struct gpu {
    int dev = 0;
    std::string name;
    size_t free0 = 0;
    cudaStream_t s = nullptr;
    cudaEvent_t e0 = nullptr, e1 = nullptr, e2 = nullptr; double t_x = 0, t_k = 0, t_out = 0; int n_t = 0;   // GPU-time profile (--verbose)
    cublasHandle_t cb = nullptr;
    float *x = nullptr, *gath = nullptr, *a = nullptr, *b = nullptr, *gout = nullptr;
    int32_t * gidx = nullptr;
    void *qa = nullptr, *q2 = nullptr, *tab = nullptr, *gemm = nullptr;
    size_t gemm_bytes = 0;
    int32_t * gidx_pin = nullptr; fast::mv_group * tab_pin = nullptr; float * out_pin = nullptr;
    // one-copy staging (default; --no-one-copy: three copies as before): [x rows | gather index | tables] in one pinned
    // block and one device block, so a request costs one H2D per GPU instead of three
    char * stg = nullptr; char * stg_pin = nullptr; size_t off_g = 0, off_t = 0;
    char * bounce = nullptr;
    int slots = 0;            // experts per served layer
    bool fixed = false;       // its set is loaded once and never swapped (--swap-gpus)
    std::vector<layer_set> ls;
    // per request
    std::vector<int> jobs;    // global job indices on this GPU
    int np = 0;
};

// a spinning worker thread (one per extra GPU): run() hands it a job, wait() returns when it is done (rethrows)
struct worker {
    std::thread th;
    std::atomic<uint64_t> go{0}, fin{0};
    std::atomic<bool> quit{false};
    std::function<void()> fn;
    std::string err;
    worker() {
        th = std::thread([this] {
            uint64_t seen = 0;
            while (!quit.load(std::memory_order_acquire)) {
                const uint64_t g = go.load(std::memory_order_acquire);
                if (g == seen) { std::this_thread::yield(); continue; }
                try { fn(); } catch (const std::exception & e) { err = e.what(); }
                seen = g;
                fin.store(g, std::memory_order_release);
            }
        });
    }
    ~worker() { quit = true; if (th.joinable()) th.join(); }
    void run(std::function<void()> f) { fn = std::move(f); err.clear(); go.fetch_add(1, std::memory_order_acq_rel); }
    void wait() {
        const uint64_t g = go.load(std::memory_order_acquire);
        while (fin.load(std::memory_order_acquire) != g) std::this_thread::yield();
        if (!err.empty()) throw std::runtime_error(err);
    }
};

struct server {
    gguf_file & g;
    std::vector<gpu> G;
    int E = 0, FFE = 0, NEXP = 0;
    rexp_serve sv;
    std::vector<int> layers;                  // file layer per served layer
    std::vector<int> nexp_of;                 // [served layer] experts its tensors hold (ids node 0 may name: < this)
    std::vector<std::vector<std::pair<int, int>>> where;   // [served layer][expert] = (gpu, slot) or (-1, -1)
    size_t fixed = 0, margin = 0;
    bool verbose = false;
    char * req = nullptr; size_t req_cap = 0;   // pinned request payload
    float * rep = nullptr; size_t rep_cap = 0;  // pinned reply rows
    // stats
    uint64_t n_req = 0, n_pairs = 0, bytes_in = 0, bytes_out = 0; double t_compute = 0;
    uint64_t n_swaps = 0, n_landed = 0;
    double t_ph[5] = {};   // ms: payload recv, swap landing, GPU issue, GPU wait, reply build + send
    using clk = std::chrono::steady_clock;
    static double ms(clk::time_point a, clk::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); }

    // ---- swaps (node 0's SWAP): a loader thread reads the new expert's three Q4 slices from the file into a pinned
    // ring entry; the request loop lands ready entries (copy into the freed slot + pack, queued on that GPU's stream
    // ahead of any later request) and reports them in its next reply
    struct load_job { int li, q, slot, e, ring; };
    struct ring_e { char * buf = nullptr; cudaEvent_t ev[REXP_MAX_GPU] = {}; int ev_q = -1; bool busy = false; };
    std::vector<ring_e> ring;
    size_t rec_bytes = 0;
    std::thread loader;
    std::mutex mu; std::condition_variable cv;
    std::deque<load_job> todo, ready;
    bool stop = false;
    std::vector<int32_t> landed;   // (served layer, expert) pairs landed since the last reply
    bool threads = true;           // --no-threads: issue the GPUs one after the other on the request thread
    bool one_copy = true;          // --no-one-copy: x, index and tables in three copies per GPU (as before)
    bool merge_gu = false;         // --merge-gu: gate + up groups in one grouped launch per column width (same bits)
    std::vector<std::unique_ptr<worker>> workers;

    explicit server(gguf_file & gf) : g(gf) {}

    static size_t a256(size_t b) { return (b + 255) & ~(size_t) 255; }

    // ---- keep-alive (see keepalive_k) ----
    bool keepalive = true;
    std::thread ka_thread; std::atomic<bool> ka_stop{false};
    void start_keepalive() {
        if (!keepalive) return;
        ka_stop = false;
        ka_thread = std::thread([this] {
            std::vector<cudaStream_t> st(G.size()); std::vector<float *> buf(G.size()); std::vector<long long> cyc(G.size());
            for (size_t q = 0; q < G.size(); ++q) {
                cudaSetDevice(G[q].dev);
                int lo = 0, hi = 0; cudaDeviceGetStreamPriorityRange(&lo, &hi);
                cudaStreamCreateWithPriority(&st[q], cudaStreamNonBlocking, lo);   // lo = the lowest priority
                cudaMalloc(&buf[q], 4096 * 4); cudaMemset(buf[q], 0, 4096 * 4);
                int khz = 0; cudaDeviceGetAttribute(&khz, cudaDevAttrClockRate, G[q].dev);
                cyc[q] = (long long) khz * 2;   // ~2 ms at the boost clock
            }
            while (!ka_stop) {
                for (size_t q = 0; q < G.size(); ++q) { cudaSetDevice(G[q].dev); keepalive_k<<<1, 32, 0, st[q]>>>(buf[q], cyc[q]); }
                for (size_t q = 0; q < G.size(); ++q) cudaStreamSynchronize(st[q]);
            }
            for (size_t q = 0; q < G.size(); ++q) { cudaSetDevice(G[q].dev); cudaStreamSynchronize(st[q]); cudaFree(buf[q]); cudaStreamDestroy(st[q]); }
        });
    }
    void stop_keepalive() { ka_stop = true; if (ka_thread.joinable()) ka_thread.join(); }

    void start_loader() {
        rec_bytes = 0;
        for (int fil : layers) {
            size_t b = 0;
            for (const char * n : {"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"}) {
                const gguf_tensor * t = g.find(bn(fil, n));
                b += a256((size_t) tensor_nbytes(t->type, t->ne) / (size_t) t->ne[2]);
            }
            rec_bytes = std::max(rec_bytes, b);
        }
        ring.assign(32, ring_e{});
        for (ring_e & r : ring) {
            CK(cudaMallocHost((void **) &r.buf, rec_bytes));
            for (size_t q = 0; q < G.size(); ++q) { CK(cudaSetDevice(G[q].dev)); CK(cudaEventCreateWithFlags(&r.ev[q], cudaEventDisableTiming)); }
        }
        stop = false;
        loader = std::thread([this] { load_loop(); });
    }
    void stop_loader() {
        { std::lock_guard<std::mutex> lk(mu); stop = true; }
        cv.notify_all();
        if (loader.joinable()) loader.join();
        for (ring_e & r : ring) {
            if (r.buf) cudaFreeHost(r.buf);
            for (size_t q = 0; q < G.size(); ++q) if (r.ev[q]) { cudaSetDevice(G[q].dev); cudaEventDestroy(r.ev[q]); }
        }
        ring.clear(); todo.clear(); ready.clear(); landed.clear();
    }
    void load_loop() {
        static const char * names[3] = {"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"};
        while (true) {
            load_job j;
            {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [&] { return stop || !todo.empty(); });
                if (stop) return;
                j = todo.front(); todo.pop_front();
            }
            int ri = -1;
            while (ri < 0) {   // a ring entry whose last landing copy is done
                {
                    std::lock_guard<std::mutex> lk(mu);
                    if (stop) return;
                    for (int i = 0; i < (int) ring.size() && ri < 0; ++i) {
                        ring_e & r = ring[i];
                        if (r.busy && r.ev_q >= 0 && cudaEventQuery(r.ev[r.ev_q]) == cudaSuccess) { r.busy = false; r.ev_q = -1; }
                        if (!r.busy) { r.busy = true; ri = i; }
                    }
                }
                if (ri < 0) std::this_thread::sleep_for(std::chrono::microseconds(200));
            }
            size_t off = 0;
            for (int k = 0; k < 3; ++k) {
                const gguf_tensor * t = g.find(bn(layers[j.li], names[k]));
                const size_t b = (size_t) tensor_nbytes(t->type, t->ne) / (size_t) t->ne[2];
                g.read_at(ring[ri].buf + off, t->data + (size_t) j.e * b, b);
                off += a256(b);
            }
            j.ring = ri;
            std::lock_guard<std::mutex> lk(mu);
            ready.push_back(j);
        }
    }
    // SWAP: drop each old expert now (node 0 no longer sends it), queue the new one into its slot
    void swap(rexp_sock & k, const rexp_hdr & h) {
        if (h.a < 0 || h.bytes != (uint64_t) h.a * 12 || h.bytes > ((uint64_t) 64 << 20)) throw std::runtime_error("expert server: bad SWAP size");
        std::vector<int32_t> v((size_t) h.a * 3);
        k.recv_all(v.data(), h.bytes, 60000);
        std::lock_guard<std::mutex> lk(mu);
        for (int i = 0; i < h.a; ++i) {
            const int li = v[3 * i], eo = v[3 * i + 1], en = v[3 * i + 2];
            if (li < 0 || li >= (int) layers.size() || eo < 0 || eo >= NEXP || en < 0 || en >= nexp_of[li]) throw std::runtime_error("expert server: SWAP out of range");
            const auto w = where[li][eo];
            if (w.first < 0 || where[li][en].first >= 0 || G[w.first].fixed) continue;   // not held / already held / fixed GPU: never lands
            where[li][eo] = {-1, -1};
            todo.push_back({li, w.first, w.second, en, -1});
            ++n_swaps;
        }
        cv.notify_all();
    }
    // ready loads -> their slots (copy + pack on the GPU's stream, before this request's work)
    void land() {
        std::deque<load_job> rd;
        { std::lock_guard<std::mutex> lk(mu); rd.swap(ready); }
        for (const load_job & j : rd) {
            gpu & d = G[j.q];
            CK(cudaSetDevice(d.dev));
            layer_set & L = d.ls[j.li];
            size_t off = 0;
            for (int k = 0; k < 3; ++k) {
                const gguf_tensor * t = g.find(bn(layers[j.li], k == 0 ? "ffn_gate_exps.weight" : k == 1 ? "ffn_up_exps.weight" : "ffn_down_exps.weight"));
                char * dst = L.base[k] + (size_t) j.slot * L.p[k].bytes;
                CK(cudaMemcpyAsync(dst, ring[j.ring].buf + off, L.p[k].bytes, cudaMemcpyHostToDevice, d.s));
                const fast::packed_weight pk = fast::pack_weight(t->type, dst, t->ne[0], t->ne[1], d.s);
                if (pk.data != dst || pk.layout != L.p[k].layout) throw std::runtime_error("expert server: swapped expert packs differently");
                off += a256(L.p[k].bytes);
            }
            {
                std::lock_guard<std::mutex> lk(mu);
                CK(cudaEventRecord(ring[j.ring].ev[j.q], d.s));
                ring[j.ring].ev_q = j.q;   // busy until the copy is done (the loader polls the event)
            }
            L.ids[j.slot] = j.e;
            where[j.li][j.e] = {j.q, j.slot};
            landed.push_back(j.li); landed.push_back(j.e);
            ++n_landed;
        }
    }

    void grow_pinned(char ** p, size_t * cap, size_t need) {
        if (*cap >= need) return;
        if (*p) CK(cudaFreeHost(*p));
        *cap = std::max(need, (size_t) 1 << 20);
        CK(cudaMallocHost((void **) p, *cap));
    }

    void free_all() {
        workers.clear();
        stop_keepalive();
        stop_loader();
        for (gpu & q : G) {
            cudaSetDevice(q.dev);
            cudaDeviceSynchronize();
            for (layer_set & l : q.ls) for (int k = 0; k < 3; ++k) if (l.base[k]) cudaFree(l.base[k]);
            q.ls.clear();
            if (merge_gu) q.b = nullptr;   // a view into q.a
            for (void * p : {(void *) q.x, (void *) q.gath, (void *) q.a, (void *) q.b, (void *) q.gout, (void *) q.gidx, q.qa, q.q2, q.tab, q.gemm}) if (p) cudaFree(p);
            for (void * p : {(void *) q.gidx_pin, (void *) q.tab_pin, (void *) q.out_pin, (void *) q.bounce, (void *) q.stg_pin}) if (p) cudaFreeHost(p);
            if (q.stg) cudaFree(q.stg);
            if (q.cb) cublasDestroy(q.cb);
            if (q.s) cudaStreamDestroy(q.s);
            const int dev = q.dev; const std::string nm = q.name; const size_t f0 = q.free0;
            q = gpu{}; q.dev = dev; q.name = nm; q.free0 = f0;
        }
        if (req) { cudaFreeHost(req); req = nullptr; req_cap = 0; }
        if (rep) { cudaFreeHost(rep); rep = nullptr; rep_cap = 0; }
        where.clear(); layers.clear(); nexp_of.clear();
    }

    size_t expert_bytes(int fil) const {
        size_t b = 0;
        for (const char * n : {"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"}) {
            const gguf_tensor * t = g.find(bn(fil, n));
            if (!t) throw std::runtime_error("expert server: missing " + bn(fil, n));
            b += (size_t) tensor_nbytes(t->type, t->ne) / (size_t) t->ne[2];
        }
        return b;
    }

    // byte plan of one GPU: the request buffers at max_pairs rows + cuBLAS / fold scratch + margin; the rest = slots
    void plan(const std::vector<int> & slots_req) {
        const size_t NP = (size_t) std::max(sv.max_pairs, 1), NX = (size_t) std::max(sv.max_x, 1);
        const bool fold = sv.fold_min > 0 && sv.max_x >= sv.fold_min;
        fixed = NX * E * 4 + NP * E * 4 * 2 + NP * FFE * 4 * 2 + NP * 4 + fast::matvec_scratch_bytes(E, (int) NP) +
                fast::matvec_scratch_bytes(FFE, (int) NP) + 6 * NP * sizeof(fast::mv_group) + (fold ? ((size_t) 96 << 20) : 0);
        size_t per = 0;
        for (int fil : layers) per = std::max(per, expert_bytes(fil));
        fixed += layers.size() * 3 * ((size_t) 2 << 20);   // 2 MiB page rounding of each layer's 3 slot arrays
        const double MiB = 1048576.0;
        printf("expert server: byte plan for %zu served layers, %.2f MiB per expert (Q4 gate + up + down), requests <= %d rows / %d pairs:\n",
               layers.size(), per / MiB, sv.max_x, sv.max_pairs);
        for (size_t q = 0; q < G.size(); ++q) {
            gpu & d = G[q];
            const double avail = (double) d.free0 - (double) fixed - (double) margin;
            const int fit = avail > 0 ? (int) std::min<double>(NEXP, std::floor(avail / ((double) per * layers.size()))) : 0;
            const int req = q < slots_req.size() ? slots_req[q] : -1;
            d.slots = req < 0 ? fit : std::min(req, NEXP);
            const double tot = (double) fixed + (double) d.slots * per * layers.size();
            printf("expert server:   GPU %zu (dev %d, %s): experts %.2f GiB (%d per layer x %zu layers) + buffers %.2f GiB + margin %.2f GiB = %.2f GiB of %.2f GiB free: %s\n",
                   q, d.dev, d.name.c_str(), gib((double) d.slots * per * layers.size()), d.slots, layers.size(), gib((double) fixed), gib((double) margin),
                   gib(tot + margin), gib((double) d.free0), tot + margin <= d.free0 ? "fits" : "DOES NOT FIT (explicit --slots)");
        }
        fflush(stdout);
    }

    void alloc_buffers(gpu & d) {
        CK(cudaSetDevice(d.dev));
        CK(cudaStreamCreateWithFlags(&d.s, cudaStreamNonBlocking));
        CK(cudaEventCreate(&d.e0)); CK(cudaEventCreate(&d.e1)); CK(cudaEventCreate(&d.e2));
        CB(cublasCreate(&d.cb)); CB(cublasSetStream(d.cb, d.s)); CB(cublasSetMathMode(d.cb, CUBLAS_PEDANTIC_MATH));
        const size_t NP = (size_t) std::max(sv.max_pairs, 1), NX = (size_t) std::max(sv.max_x, 1);
        CK(cudaMalloc(&d.x, NX * E * 4)); CK(cudaMalloc(&d.gath, NP * E * 4)); CK(cudaMalloc(&d.gout, NP * E * 4));
        if (merge_gu) { CK(cudaMalloc(&d.a, 2 * NP * FFE * 4)); d.b = d.a + NP * FFE; }   // gate rows then up rows: one launch writes both
        else { CK(cudaMalloc(&d.a, NP * FFE * 4)); CK(cudaMalloc(&d.b, NP * FFE * 4)); }
        CK(cudaMalloc(&d.gidx, NP * 4));
        CK(cudaMalloc(&d.qa, fast::matvec_scratch_bytes(E, (int) NP))); CK(cudaMalloc(&d.q2, fast::matvec_scratch_bytes(FFE, (int) NP)));
        CK(cudaMalloc(&d.tab, 6 * NP * sizeof(fast::mv_group)));
        if (sv.fold_min > 0 && sv.max_x >= sv.fold_min) { d.gemm_bytes = (size_t) 96 << 20; CK(cudaMalloc(&d.gemm, d.gemm_bytes)); }
        CK(cudaMallocHost(&d.gidx_pin, NP * 4)); CK(cudaMallocHost(&d.tab_pin, 6 * NP * sizeof(fast::mv_group)));
        CK(cudaMallocHost(&d.out_pin, NP * E * 4));
        CK(cudaMallocHost(&d.bounce, (size_t) 8 << 20));
        d.off_g = (NX * E * 4 + 255) & ~(size_t) 255;
        d.off_t = d.off_g + ((NP * 4 + 255) & ~(size_t) 255);
        const size_t sb = d.off_t + 6 * NP * sizeof(fast::mv_group);
        CK(cudaMalloc(&d.stg, sb)); CK(cudaMallocHost(&d.stg_pin, sb));
    }

    // GPU d's experts: per served layer its slice of the list (fn_model::load_experts' load_device: raw bytes read with
    // ReadFile through a pinned bounce, copied, packed in place)
    void load_gpu(gpu & d, const std::vector<std::vector<int>> & want) {
        CK(cudaSetDevice(d.dev));
        static const char * names[3] = {"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"};
        d.ls.resize(layers.size());
        for (size_t li = 0; li < layers.size(); ++li) {
            layer_set & L = d.ls[li];
            L.ids = want[li];
            L.n = (int) L.ids.size();
            if (L.n == 0) continue;
            for (int k = 0; k < 3; ++k) {
                const gguf_tensor * t = g.find(bn(layers[li], names[k]));
                proj & p = L.p[k];
                p.type = t->type;
                p.bytes = (size_t) tensor_nbytes(t->type, t->ne) / (size_t) t->ne[2];
                if (p.bytes > ((size_t) 8 << 20)) throw std::runtime_error("expert server: expert slice above the 8 MiB bounce");
                CK(cudaMalloc(&L.base[k], p.bytes * L.n));
                for (int j = 0; j < L.n; ++j) {
                    char * dst = L.base[k] + (size_t) j * p.bytes;
                    g.read_at(d.bounce, t->data + (size_t) L.ids[j] * p.bytes, p.bytes);
                    CK(cudaMemcpyAsync(dst, d.bounce, p.bytes, cudaMemcpyHostToDevice, d.s));
                    CK(cudaStreamSynchronize(d.s));
                    const fast::packed_weight pk = fast::pack_weight(t->type, dst, t->ne[0], t->ne[1], d.s);
                    if (!pk.data || pk.data != dst) throw std::runtime_error("expert server: no in-place packed layout for the expert type");
                    p.layout = pk.layout;
                }
                if (!grouped_layout(p.layout)) throw std::runtime_error("expert server: layout without a grouped kernel");
            }
            if (L.p[0].layout != L.p[1].layout) throw std::runtime_error("expert server: gate / up layouts differ");
        }
        CK(cudaStreamSynchronize(d.s));
    }

    void setup(rexp_sock & k, const std::vector<int> & slots_req) {
        // every size, count and index node 0 sends is checked against this node's file and buffers before it is used
        rexp_hdr h = k.recv_hdr(REXP_SERVE, 600000);
        if (h.bytes < sizeof sv || h.bytes > sizeof sv + ((uint64_t) 1 << 18)) throw std::runtime_error("expert server: bad SERVE size");
        std::vector<char> pl(h.bytes);
        k.recv_all(pl.data(), pl.size(), 60000);
        memcpy(&sv, pl.data(), sizeof sv);
        if (sv.n_layers < 0 || h.bytes != sizeof sv + (uint64_t) sv.n_layers * 4 || sv.max_x < 0 || sv.max_x > (1 << 24) ||
            sv.max_pairs < 0 || sv.max_pairs > (1 << 24))
            throw std::runtime_error("expert server: bad SERVE fields");
        layers.assign((const int32_t *) (pl.data() + sizeof sv), (const int32_t *) (pl.data() + sizeof sv) + sv.n_layers);
        // each served layer: its three expert tensors with blk.0's shapes (the buffers and kernels are sized from blk.0)
        nexp_of.assign(layers.size(), 0);
        for (size_t li = 0; li < layers.size(); ++li) {
            int ne = NEXP;
            for (const char * n : {"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"}) {
                const gguf_tensor * t = layers[li] < 0 ? nullptr : g.find(bn(layers[li], n));
                const gguf_tensor * t0 = g.find(bn(0, n));
                if (!t || !t0 || t->ne[0] != t0->ne[0] || t->ne[1] != t0->ne[1] || t->ne[2] < 1)
                    throw std::runtime_error("expert server: node 0 serves layer " + std::to_string(layers[li]) + ", which has no " + n + " like blk.0's here");
                ne = (int) std::min<int64_t>(ne, t->ne[2]);
            }
            nexp_of[li] = ne;
        }
        plan(slots_req);
        std::vector<int32_t> caps;
        for (gpu & d : G) caps.push_back(d.slots);
        for (gpu & d : G) caps.push_back(d.fixed ? 1 : 0);
        rexp_hdr c; c.op = REXP_CAPS; c.a = (int) G.size();
        k.send_msg(c, caps.data(), caps.size() * 4, 60000);
        h = k.recv_hdr(REXP_SET, 600000);
        if (h.bytes % 4 || h.bytes > (uint64_t) layers.size() * G.size() * ((uint64_t) NEXP + 1) * 4) throw std::runtime_error("expert server: bad SET size");
        pl.resize(h.bytes);
        k.recv_all(pl.data(), pl.size(), 60000);
        const int32_t * p = (const int32_t *) pl.data();
        std::vector<std::vector<std::vector<int>>> want(G.size(), std::vector<std::vector<int>>(layers.size()));
        where.assign(layers.size(), std::vector<std::pair<int, int>>(NEXP, {-1, -1}));
        const int32_t * pend = (const int32_t *) (pl.data() + pl.size());
        for (size_t li = 0; li < layers.size(); ++li)
            for (size_t q = 0; q < G.size(); ++q) {
                if (p >= pend) throw std::runtime_error("expert server: short SET");
                const int n = *p++;
                if (n < 0 || n > pend - p) throw std::runtime_error("expert server: short SET");
                for (int i = 0; i < n; ++i) {
                    const int e = *p++;
                    if ((int) want[q][li].size() >= G[q].slots || e < 0 || e >= nexp_of[li] || where[li][e].first >= 0) continue;
                    where[li][e] = {(int) q, (int) want[q][li].size()};
                    want[q][li].push_back(e);
                }
            }
        const auto t0 = std::chrono::steady_clock::now();
        for (gpu & d : G) alloc_buffers(d);
        std::vector<std::thread> th;
        std::vector<std::string> err(G.size());
        for (size_t q = 0; q < G.size(); ++q)
            th.emplace_back([&, q] { try { load_gpu(G[q], want[q]); } catch (const std::exception & e) { err[q] = e.what(); } });
        for (auto & t : th) t.join();
        for (auto & e : err) if (!e.empty()) throw std::runtime_error(e);
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        size_t tot = 0;
        for (size_t q = 0; q < G.size(); ++q) {
            size_t n = 0; for (auto & v : want[q]) n += v.size();
            tot += n;
            size_t fr = 0, all = 0; CK(cudaSetDevice(G[q].dev)); CK(cudaMemGetInfo(&fr, &all));
            printf("expert server: GPU %zu (dev %d): %zu experts loaded, %.0f MiB free\n", q, G[q].dev, n, fr / 1048576.0);
        }
        printf("expert server: %zu experts over %zu layers loaded in %.1f s; serving\n", tot, layers.size(), s);
        fflush(stdout);
        start_loader();
        start_keepalive();
        workers.clear();
        if (threads) for (size_t q = 1; q < G.size(); ++q) workers.push_back(std::make_unique<worker>());
        rexp_hdr r; r.op = REXP_READY; r.a = 0; r.b = (int) tot;
        k.send_msg(r, nullptr, 0, 60000);
    }

    // one request: per GPU its jobs' pairs as rows 0..np-1 (job order), the grouped batch of fn_model::ffn_grouped
    void ffn(rexp_sock & k, const rexp_hdr & h) {
        const auto t0 = std::chrono::steady_clock::now();
        const int li = h.a, nx = h.b, nj = h.c, npairs = h.d;
        if (li < 0 || li >= (int) layers.size() || nx > sv.max_x || npairs > sv.max_pairs || nx < 0 || nj < 0 || npairs < 0 || nj > npairs)
            throw std::runtime_error("expert server: request out of range");
        // exactly x rows + jobs + xrow + w: the pointers below are computed from nx / nj / npairs, not from h.bytes
        if (h.bytes != (uint64_t) nx * E * 4 + (uint64_t) nj * 8 + (uint64_t) npairs * 8) throw std::runtime_error("expert server: bad FFN size");
        grow_pinned(&req, &req_cap, h.bytes);
        k.recv_all(req, h.bytes, 120000);
        const auto tA = clk::now();
        land();
        const auto tB = clk::now();
        bytes_in += h.bytes + sizeof h;
        const float * x = (const float *) req;
        const int32_t * jobs = (const int32_t *) (req + (size_t) nx * E * 4);
        const int32_t * xrow = jobs + 2 * nj;
        const float * w = (const float *) (xrow + npairs);
        std::vector<int> joff(nj + 1, 0);
        for (int j = 0; j < nj; ++j) {   // each count in 0..pairs left: no mix of signs can reach past npairs
            const int n = jobs[2 * j + 1];
            if (n < 0 || n > npairs - joff[j]) throw std::runtime_error("expert server: pair count mismatch");
            joff[j + 1] = joff[j] + n;
        }
        if (joff[nj] != npairs) throw std::runtime_error("expert server: pair count mismatch");
        for (int i = 0; i < npairs; ++i)
            if (xrow[i] < 0 || xrow[i] >= nx) throw std::runtime_error("expert server: pair row out of range");
        for (gpu & d : G) { d.jobs.clear(); d.np = 0; }
        for (int j = 0; j < nj; ++j) {
            const int e = jobs[2 * j];
            if (e < 0 || e >= NEXP || where[li][e].first < 0) throw std::runtime_error("expert server: expert " + std::to_string(e) + " of layer " + std::to_string(layers[li]) + " is not held here");
            G[where[li][e].first].jobs.push_back(j);
        }
        const int fold_min = sv.fold_min, fold_max = sv.fold_max;
        const int ng = E / 32, ngd = FFE / 32;
        auto issue = [&](gpu & d) {   // one GPU's share of the request, then its stream drained
            CK(cudaSetDevice(d.dev));
            const layer_set & L = d.ls[li];
            int r = 0;
            int32_t * gp = one_copy ? (int32_t *) (d.stg_pin + d.off_g) : d.gidx_pin;
            for (int j : d.jobs) for (int i = joff[j]; i < joff[j + 1]; ++i) gp[r++] = xrow[i];
            d.np = r;
            if (verbose) CK(cudaEventRecord(d.e0, d.s));
            float * dx = one_copy ? (float *) d.stg : d.x;
            int32_t * dg = one_copy ? (int32_t *) (d.stg + d.off_g) : d.gidx;
            if (!one_copy) {
                CK(cudaMemcpyAsync(d.x, x, (size_t) nx * E * 4, cudaMemcpyHostToDevice, d.s));
                CK(cudaMemcpyAsync(d.gidx, d.gidx_pin, (size_t) r * 4, cudaMemcpyHostToDevice, d.s));
                gather_rows_k<<<dim3(nb256(E), r), 256, 0, d.s>>>(d.x, d.gidx, d.gath, E);
                fast::matvec_quantize(d.gath, E, r, d.qa, d.s);
            } else memcpy(d.stg_pin, x, (size_t) nx * E * 4);
            std::vector<fast::mv_group> tg, tu, tg4, tu4, td, td4;
            struct fold_job { int r0, n; const char *wg, *wu, *wd; };
            std::vector<fold_job> folds;
            r = 0;
            for (int j : d.jobs) {
                const int e = jobs[2 * j], n = jobs[2 * j + 1], sl = where[li][e].second;
                const char * wg = L.base[0] + (size_t) sl * L.p[0].bytes;
                const char * wu = L.base[1] + (size_t) sl * L.p[1].bytes;
                const char * wd = L.base[2] + (size_t) sl * L.p[2].bytes;
                const int r0 = r, r1 = r + n;
                r = r1;
                if (fold_min > 0 && n >= fold_min && n <= fold_max) { folds.push_back({r0, n, wg, wu, wd}); continue; }
                int c = r0;
                if (grouped_nc_layout(L.p[0].layout)) for (; c + 4 <= r1; c += 4) { tg4.push_back({wg, c, c}); tu4.push_back({wu, c, c}); }
                for (; c < r1; ++c) { tg.push_back({wg, c, c}); tu.push_back({wu, c, c}); }
                c = r0;
                if (grouped_nc_layout(L.p[2].layout)) for (; c + 4 <= r1; c += 4) td4.push_back({wd, c, c});
                for (; c < r1; ++c) td.push_back({wd, c, c});
            }
            size_t pos = 0;
            fast::mv_group * dp = one_copy ? (fast::mv_group *) (d.stg + d.off_t) : (fast::mv_group *) d.tab;
            fast::mv_group * hp = one_copy ? (fast::mv_group *) (d.stg_pin + d.off_t) : d.tab_pin;
            auto up = [&](const std::vector<fast::mv_group> & v) { memcpy(hp + pos, v.data(), v.size() * sizeof(fast::mv_group)); const fast::mv_group * q = dp + pos; pos += v.size(); return q; };
            const int NPm = std::max(sv.max_pairs, 1);
            if (merge_gu) { for (auto & e : tu) e.out += NPm; for (auto & e : tu4) e.out += NPm; }
            const fast::mv_group *pg = up(tg), *pu = up(tu), *pg4 = up(tg4), *pu4 = up(tu4), *pd = up(td), *pd4 = up(td4);
            if (one_copy) {   // x | index | tables in one copy, then the gather and the a16 quantize
                CK(cudaMemcpyAsync(d.stg, d.stg_pin, d.off_t + pos * sizeof(fast::mv_group), cudaMemcpyHostToDevice, d.s));
                gather_rows_k<<<dim3(nb256(E), d.np), 256, 0, d.s>>>(dx, dg, d.gath, E);
                fast::matvec_quantize(d.gath, E, d.np, d.qa, d.s);
            } else CK(cudaMemcpyAsync(dp, d.tab_pin, pos * sizeof(fast::mv_group), cudaMemcpyHostToDevice, d.s));
            fast::a16_args a;
            a.aq = (const half2 *) d.qa; a.ads = (const half2 *) d.qa + (size_t) d.np * ng * 16;
            a.ngroups = ng; a.nrows = FFE; a.row_bytes = (int) (L.p[0].bytes / (size_t) FFE);
            const int lay = L.p[0].layout;
            if (merge_gu) {   // gate and up groups in one table (up outputs at rows NP.. = d.b): 2 launches instead of 4, same bits
                a.dst = d.a; launch_grouped(lay, pg4, (int) (tg4.size() + tu4.size()), a, d.s, 4); launch_grouped(lay, pg, (int) (tg.size() + tu.size()), a, d.s, 1);
            } else {
                a.dst = d.a; launch_grouped(lay, pg4, (int) tg4.size(), a, d.s, 4); launch_grouped(lay, pg, (int) tg.size(), a, d.s, 1);
                a.dst = d.b; launch_grouped(lay, pu4, (int) tu4.size(), a, d.s, 4); launch_grouped(lay, pu, (int) tu.size(), a, d.s, 1);
            }
            for (const fold_job & f : folds) {
                fast::packed_weight pk; pk.type = L.p[0].type; pk.n_in = E; pk.n_out = FFE; pk.data = (void *) f.wg; pk.bytes = L.p[0].bytes; pk.layout = L.p[0].layout;
                fast::packed_weight pku = pk; pku.data = (void *) f.wu;
                const float * xr = d.gath + (size_t) f.r0 * E;
                fast::gemm_packed(pk, xr, d.a + (size_t) f.r0 * FFE, f.n, d.gemm, d.gemm_bytes, d.s, d.cb);
                fast::gemm_packed(pku, xr, d.b + (size_t) f.r0 * FFE, f.n, d.gemm, d.gemm_bytes, d.s, d.cb);
            }
            k::silu_mul(d.a, d.b, (int64_t) d.np * FFE, d.s);
            fast::matvec_quantize(d.a, FFE, d.np, d.q2, d.s);
            fast::a16_args dd;
            dd.aq = (const half2 *) d.q2; dd.ads = (const half2 *) d.q2 + (size_t) d.np * ngd * 16;
            dd.ngroups = ngd; dd.nrows = E; dd.dst = d.gout; dd.row_bytes = (int) (L.p[2].bytes / (size_t) E);
            launch_grouped(L.p[2].layout, pd4, (int) td4.size(), dd, d.s, 4); launch_grouped(L.p[2].layout, pd, (int) td.size(), dd, d.s, 1);
            for (const fold_job & f : folds) {
                fast::packed_weight pk; pk.type = L.p[2].type; pk.n_in = FFE; pk.n_out = E; pk.data = (void *) f.wd; pk.bytes = L.p[2].bytes; pk.layout = L.p[2].layout;
                fast::gemm_packed(pk, d.a + (size_t) f.r0 * FFE, d.gout + (size_t) f.r0 * E, f.n, d.gemm, d.gemm_bytes, d.s, d.cb);
            }
            if (verbose) CK(cudaEventRecord(d.e1, d.s));
            CK(cudaMemcpyAsync(d.out_pin, d.gout, (size_t) d.np * E * 4, cudaMemcpyDeviceToHost, d.s));
            if (verbose) CK(cudaEventRecord(d.e2, d.s));
            CK(cudaStreamSynchronize(d.s));
        };
        // the GPUs in parallel: GPU q > 0 on its worker thread (--threads, default on), the first on this thread
        // the workers capture this frame (issue, joff, jobs, xrow) by reference: every started one is waited for before
        // an error leaves ffn(), and the first error is rethrown after all of them are done
        std::vector<worker *> started;
        std::exception_ptr err;
        try {
            for (size_t q = 1; q < G.size(); ++q)
                if (!G[q].jobs.empty() && q - 1 < workers.size()) { workers[q - 1]->run([&, q] { issue(G[q]); }); started.push_back(workers[q - 1].get()); }
            for (size_t q = 0; q < G.size(); ++q) if (!G[q].jobs.empty() && !(q >= 1 && q - 1 < workers.size())) issue(G[q]);
        } catch (...) {
            err = std::current_exception();
        }
        for (worker * wk : started) {
            try { wk->wait(); } catch (...) { if (!err) err = std::current_exception(); }
        }
        if (err) std::rethrow_exception(err);
        // reply: pair rows in request order (exact) or per x row weighted sums (sum mode, pair order within each row)
        const size_t nrow = sv.sum_mode ? (size_t) nx : (size_t) npairs;
        const bool f16 = sv.sum_mode && sv.reply_f16;
        char * rp = (char *) rep;
        grow_pinned(&rp, &rep_cap, nrow * E * 4 + landed.size() * 4 + 64);
        rep = (float *) rp;
        if (sv.sum_mode) memset(rep, 0, nrow * E * 4);
        const auto tC = clk::now();
        for (gpu & d : G) { if (!d.jobs.empty()) { CK(cudaSetDevice(d.dev)); CK(cudaStreamSynchronize(d.s)); } }
        const auto tD = clk::now();
        if (verbose)
            for (gpu & d : G) {
                if (d.jobs.empty()) continue;
                float a = 0, b = 0;
                CK(cudaEventElapsedTime(&a, d.e0, d.e1)); CK(cudaEventElapsedTime(&b, d.e1, d.e2));
                d.t_k += a; d.t_out += b; d.n_t += 1;
            }
        for (gpu & d : G) {
            if (d.jobs.empty()) continue;
            int r = 0;
            for (int j : d.jobs) {
                const int n = jobs[2 * j + 1];
                if (!sv.sum_mode) memcpy(rep + (size_t) joff[j] * E, d.out_pin + (size_t) r * E, (size_t) n * E * 4);
                r += n;
            }
        }
        if (sv.sum_mode) {   // per row: pairs in request order (jobs in ascending expert id per rank)
            std::vector<std::pair<int, int>> src(npairs);   // pair -> (gpu, row on that gpu)
            for (size_t q = 0; q < G.size(); ++q) { int r = 0; for (int j : G[q].jobs) for (int i = joff[j]; i < joff[j + 1]; ++i) src[i] = {(int) q, r++}; }
            for (int i = 0; i < npairs; ++i) {
                float * y = rep + (size_t) xrow[i] * E;
                const float * s = G[src[i].first].out_pin + (size_t) src[i].second * E;
                const float wi = w[i];
                for (int c = 0; c < E; ++c) y[c] = std::fma(wi, s[c], y[c]);
            }
        }
        t_compute += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        rexp_hdr o; o.op = REXP_FFN_R; o.seq = h.seq; o.a = sv.sum_mode; o.d = (int) nrow; o.c = (int) landed.size() / 2;
        size_t rb = nrow * E * 4;
        if (f16) {   // sum rows -> f16 in place (the half array is written below its f32 source as it goes)
            __half * hp = (__half *) rep;
            for (size_t i = 0; i < nrow * (size_t) E; ++i) hp[i] = __float2half(rep[i]);
            rb = nrow * E * 2;
        }
        memcpy((char *) rep + rb, landed.data(), landed.size() * 4);
        const size_t nb = rb + landed.size() * 4;
        landed.clear();
        k.send_msg(o, rep, nb, 120000);
        bytes_out += nb + sizeof o;
        n_req += 1; n_pairs += npairs;
        const auto tE = clk::now();
        t_ph[0] += ms(t0, tA); t_ph[1] += ms(tA, tB); t_ph[2] += ms(tB, tC); t_ph[3] += ms(tC, tD); t_ph[4] += ms(tD, tE);
    }
};

int run(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: p100e-fn-expert-server <model.gguf> --connect host:port [--devices 0,1] [--slots a,b|auto] [--margin-mb N] [--once]\n"); return 2; }
    std::string path = argv[1], addr;
    std::vector<int> devices = {0}, slots, swap_gpus = {0};
    int margin_mb = 384, connect_s = 3600;
    bool once = false, verbose = false, keepalive = true, threads = true, one_copy = true, merge_gu = false, spin_sync = false;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return std::string(i + 1 < argc ? argv[++i] : ""); };
        if (a == "--connect") addr = next();
        else if (a == "--devices") { std::string d = next(); devices.clear(); for (char * p = strtok(&d[0], ","); p; p = strtok(nullptr, ",")) devices.push_back(atoi(p)); }
        else if (a == "--slots") { std::string s = next(); for (char * p = strtok(&s[0], ","); p; p = strtok(nullptr, ",")) slots.push_back(!strcmp(p, "auto") ? -1 : atoi(p)); }
        else if (a == "--margin-mb") margin_mb = atoi(next().c_str());
        else if (a == "--swap-gpus") { std::string s = next(); swap_gpus.clear(); if (s != "none") for (char * p = strtok(&s[0], ","); p; p = strtok(nullptr, ",")) swap_gpus.push_back(atoi(p)); }
        else if (a == "--connect-s") connect_s = atoi(next().c_str());
        else if (a == "--once") once = true;
        else if (a == "--no-keepalive") keepalive = false;
        else if (a == "--no-threads") threads = false;
        else if (a == "--no-one-copy") one_copy = false;
        else if (a == "--merge-gu") merge_gu = true;
        else if (a == "--spin-sync") spin_sync = true;
        else if (a == "--verbose") verbose = true;
        else { fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    if (addr.empty()) { fprintf(stderr, "--connect host:port is required (node 0 listens: p100e-fn-run --rexp-listen)\n"); return 2; }
    if ((int) devices.size() > REXP_MAX_GPU) { fprintf(stderr, "at most %d GPUs\n", REXP_MAX_GPU); return 2; }
    std::string host; int port = 0;
    rexp_parse_addr(addr, host, port);
    gguf_file g(path);
    const gguf_tensor * t0 = g.find("blk.0.ffn_gate_exps.weight");
    if (!t0) throw std::runtime_error("expert server: not a Flash-Next expert file (no blk.0.ffn_gate_exps.weight)");
    server sv(g);
    sv.verbose = verbose;
    sv.keepalive = keepalive;
    sv.threads = threads;
    sv.one_copy = one_copy;
    sv.merge_gu = merge_gu;
    sv.E = (int) t0->ne[0]; sv.FFE = (int) t0->ne[1]; sv.NEXP = (int) t0->ne[2];
    sv.margin = (size_t) margin_mb << 20;
    const uint64_t thash = g.tensor_table_hash();
    for (int d : devices) {
        gpu q; q.dev = d;
        CK(cudaSetDevice(d));
        // --spin-sync: the host thread spins in stream / event syncs instead of yielding (lower wake-up latency per
        // request on WDDM; costs CPU while a GPU works). Must precede the context's creation (the memory query below).
        if (spin_sync) CK(cudaSetDeviceFlags(cudaDeviceScheduleSpin));
        cudaDeviceProp pr{}; CK(cudaGetDeviceProperties(&pr, d)); q.name = pr.name;
        size_t fr = 0, tot = 0; CK(cudaMemGetInfo(&fr, &tot)); q.free0 = fr;
        q.fixed = std::find(swap_gpus.begin(), swap_gpus.end(), (int) sv.G.size()) == swap_gpus.end();
        sv.G.push_back(q);
    }
    printf("expert server: %s (%.1f GB), E %d ff %d experts %d; GPUs:", path.c_str(), g.file_size / 1e9, sv.E, sv.FFE, sv.NEXP);
    for (auto & q : sv.G) printf(" [dev %d %s, %.0f MiB free, %s]", q.dev, q.name.c_str(), q.free0 / 1048576.0, q.fixed ? "fixed set" : "swaps");
    printf("\n"); fflush(stdout);
    while (true) {
        printf("expert server: connecting to node 0 at %s (up to %d s)\n", addr.c_str(), connect_s); fflush(stdout);
        rexp_sock k = rexp_sock::connect_to(host, port, connect_s * 1000);
        try {
            rexp_hello he;
            he.n_gpu = (int) sv.G.size(); he.file_bytes = g.file_size; he.table_hash = thash;
            he.E = sv.E; he.FFE = sv.FFE; he.NEXP = sv.NEXP; he.n_blk = (int) g.get_int("qwen4exp.block_count", 0);
            for (size_t q = 0; q < sv.G.size(); ++q) { he.gpu_free[q] = sv.G[q].free0; snprintf(he.gpu_name[q], 64, "%s", sv.G[q].name.c_str()); }
            rexp_hdr h; h.op = REXP_HELLO;
            k.send_msg(h, &he, sizeof he, 60000);
            sv.setup(k, slots);
            while (true) {
                const rexp_hdr r = k.recv_hdr(0, -1);
                if (r.op == REXP_QUIT) break;
                if (r.op == REXP_SWAP) { sv.swap(k, r); continue; }
                if (r.op != REXP_FFN) throw std::runtime_error("expert server: unexpected message " + std::to_string(r.op));
                sv.ffn(k, r);
            }
            printf("expert server: node 0 quit: %llu requests, %llu pairs, %.1f MB in, %.1f MB out, %.1f ms computing (%.3f ms per request); swaps %llu asked, %llu landed\n",
                   (unsigned long long) sv.n_req, (unsigned long long) sv.n_pairs, sv.bytes_in / 1e6, sv.bytes_out / 1e6, sv.t_compute, sv.n_req ? sv.t_compute / sv.n_req : 0.0,
                   (unsigned long long) sv.n_swaps, (unsigned long long) sv.n_landed);
            for (auto & d : sv.G) if (d.n_t) printf("expert server: GPU dev %d: %d requests, GPU time per request: inputs + kernels %.3f ms, output copy %.3f ms\n", d.dev, d.n_t, d.t_k / d.n_t, d.t_out / d.n_t);
            if (sv.n_req) printf("expert server: per request ms: payload recv %.3f, swap landing %.3f, GPU issue %.3f, GPU wait %.3f, reply %.3f\n",
                                 sv.t_ph[0] / sv.n_req, sv.t_ph[1] / sv.n_req, sv.t_ph[2] / sv.n_req, sv.t_ph[3] / sv.n_req, sv.t_ph[4] / sv.n_req);
        } catch (const std::exception & e) {
            printf("expert server: session ended: %s\n", e.what());
        }
        fflush(stdout);
        sv.free_all();
        sv.n_req = sv.n_pairs = sv.bytes_in = sv.bytes_out = 0; sv.t_compute = 0;
        for (double & v : sv.t_ph) v = 0;
        if (once) break;
    }
    return 0;
}

} // namespace

int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    try { return run(argc, argv); }
    catch (const std::exception & e) { fprintf(stderr, "p100e-fn-expert-server: %s\n", e.what()); return 1; }
}
