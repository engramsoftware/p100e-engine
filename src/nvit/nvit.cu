// nvit encoder: weights, scratch and the pipelined forward. See nvit.h for the design.
//
// Pipeline (per GPU r, its patches split into chunks, one stream per chunk; ws = weight stream):
//   embed:   im2col -> SGEMM with (patch_embd.weight + .weight.1) -> + bias + resized position grid -> LN1(0) ->
//            qkv GEMM (layer 0) -> bias + rope -> tiled Q (own rows) / K, V (global rows) -> K/V tiles to the peer
//   layer l: [join: every chunk's K/V on this GPU and the peer's arrived]
//            attention -> prescale -> out GEMM -> x += out + b; LN2 -> up GEMM -> + b, GELU, prescale -> down GEMM ->
//            x += down + b; LN1(l+1) -> qkv GEMM (l+1) -> rope -> K/V to the peer      (last layer: post-LN instead)
//   merger:  [4 patches -> 4608] mm.0 (2 row halves) -> + b, GELU -> mm.2 (2 row halves) -> + b -> host
// Weight jobs (31: 27 layers, 4 merger halves) are dequantized into two f16 slots on ws, job j into slot j % 2, after
// every chunk released the job that used the slot before (events). K / V are double-buffered by layer parity.
#include "nvit.h"
#include "nvit_kernels.cuh"

#include "gguf.h"
#include "fast/gemm.cuh"
#include "convert.cuh"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>
#include <thread>

namespace p100e::nvit {

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) throw std::runtime_error( \
    std::string("nvit: ") + #x + ": " + cudaGetErrorString(e_)); } while (0)
#define CB(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) throw std::runtime_error( \
    std::string("nvit: ") + #x + ": cublas status " + std::to_string((int) s_)); } while (0)

namespace {

constexpr int NL = 27;
constexpr int QKV = 3 * E;
constexpr int FFP = 4352;                 // FF padded to a multiple of 64 (fold GEMM K)
constexpr int MAXCH = 8;
constexpr int NJOB = NL + 4;              // layers + merger halves (mm.0 rows 0..2303, 2304..4607; mm.2 0..2559, 2560..5119)
constexpr uint32_t T_F32 = 0, T_F16 = 1, T_Q8_0 = 8;
constexpr size_t Q8_ROW_E = (size_t) E / 32 * 34;          // q8_0 row bytes for K = 1152
constexpr size_t Q8_ROW_M = (size_t) MERGE_IN / 32 * 34;   // K = 4608
// f16 slot layout of a layer job (bytes)
constexpr size_t SL_QKV = 0;
constexpr size_t SL_OUT = SL_QKV + (size_t) QKV * E * 2;
constexpr size_t SL_UP = SL_OUT + (size_t) E * E * 2;
constexpr size_t SL_DOWN = SL_UP + (size_t) FF * E * 2;
constexpr size_t SL_LAYER = SL_DOWN + (size_t) E * FFP * 2;
constexpr size_t SL_MERGER = (size_t) 2560 * MERGE_IN * 2;  // largest merger half
constexpr size_t SLOT_BYTES = SL_LAYER > SL_MERGER ? SL_LAYER : SL_MERGER;
constexpr size_t RAW_LAYER_Q8 = (size_t) (QKV + E + FF) * Q8_ROW_E;
constexpr size_t RAW_MERGER = (size_t) 2560 * Q8_ROW_M;
constexpr size_t RAW_SLOT_BYTES = RAW_LAYER_Q8 > RAW_MERGER ? RAW_LAYER_Q8 : RAW_MERGER;

inline size_t al(size_t v) { return (v + 255) & ~(size_t) 255; }

// one weight in a job: raw bytes at blob offset `off` (q8_0 -> dequantized into the slot; f16 -> copied as is)
struct piece {
    size_t off = 0, bytes = 0, raw_off = 0, slot_off = 0;
    uint32_t type = 0;
    int64_t n = 0;   // elements
};
struct job { std::vector<piece> pieces; };

struct layer_small {   // device f32 vectors of one block
    float *ln1_w, *ln1_b, *ln2_w, *ln2_b, *qkv_b, *o_b, *up_b, *down_b;
};

struct rank_t {
    int dev = 0;
    cudaStream_t ws = nullptr;
    cudaStream_t cs[MAXCH] = {};
    cublasHandle_t cb = nullptr;              // one handle (no workspace), stream set per call: patch-embed SGEMM
    cudaEvent_t ev_w[2] = {}, ev_wdone[MAXCH] = {}, ev_qkv[MAXCH] = {}, ev_sent[MAXCH] = {}, ev_up = nullptr,
                ev_end[MAXCH] = {};
    cudaEvent_t t_beg = nullptr, t_emb = nullptr, t_lay = nullptr, t_end = nullptr;   // timing (chunk 0)
    float * small = nullptr;                   // all small f32 tensors (one allocation)
    std::vector<layer_small> L;
    float *post_w, *post_b, *mm0_b, *mm2_b, *patch_w, *patch_b, *pos;
    uint8_t * blob_dev = nullptr;              // resident raw weights (host_weights = false)
    void * own = nullptr;                      // own scratch
    size_t own_bytes = 0;
    // carve of the current scratch
    float *img, *x, *y, *cs_, *part, *st;
    half * x16;
    uint32_t *qt, *kvt;                        // tiled attention operands (nvit_kernels.cuh)
    half * slot[2];
    uint8_t * raw[2];
    int n0 = 0, n = 0;                         // owned patches [n0, n0 + n)
    int nch = 1, cb_[MAXCH + 1] = {};          // chunk row bounds (local)
};

void * hmalloc_pinned(size_t n) {
    void * p = nullptr;
    CK(cudaHostAlloc(&p, n, cudaHostAllocPortable));
    return p;
}

} // namespace

struct encoder::impl {
    opts o;
    std::unique_ptr<gguf_file> g;
    int R = 1;
    std::vector<rank_t> rk;
    uint8_t * blob_host = nullptr;             // pinned raw weights (host_weights)
    size_t blob_bytes = 0;
    std::vector<job> jobs;
    size_t small_floats = 0;
    float eps = 1e-6f;
    float * out_h = nullptr;                   // pinned output staging
    float * img_h = nullptr;                   // pinned input staging
    size_t img_h_bytes = 0;
    bool p2p_2d = true;                        // cudaMemcpy2DAsync between the GPUs works (else per-row peer copies)
    size_t out_h_bytes = 0;
    to_fp16_cuda_t q8_to16 = nullptr;

    ~impl() {
        for (auto & K : rk) {
            cudaSetDevice(K.dev);
            cudaDeviceSynchronize();
            if (K.cb) cublasDestroy(K.cb);
            for (int c = 0; c < MAXCH; ++c) {
                if (K.cs[c]) cudaStreamDestroy(K.cs[c]);
                for (cudaEvent_t e : {K.ev_wdone[c], K.ev_qkv[c], K.ev_sent[c], K.ev_end[c]}) if (e) cudaEventDestroy(e);
            }
            for (cudaEvent_t e : {K.ev_w[0], K.ev_w[1], K.ev_up, K.t_beg, K.t_emb, K.t_lay, K.t_end}) if (e) cudaEventDestroy(e);
            if (K.ws) cudaStreamDestroy(K.ws);
            if (K.small) cudaFree(K.small);
            if (K.blob_dev) cudaFree(K.blob_dev);
            if (K.own) cudaFree(K.own);
        }
        if (blob_host) cudaFreeHost(blob_host);
        if (out_h) cudaFreeHost(out_h);
        if (img_h) cudaFreeHost(img_h);
        cudaGetLastError();
    }

    // ---- load ----
    const gguf_tensor * T(const std::string & name, uint32_t type, std::initializer_list<int64_t> ne) const {
        const gguf_tensor * t = g->find(name);
        if (!t) throw std::runtime_error("nvit: tensor missing from the mmproj: " + name);
        if (t->type != type) throw std::runtime_error("nvit: unexpected type of " + name);
        int i = 0;
        for (int64_t v : ne) {
            if (t->ne[i] != v) throw std::runtime_error("nvit: unexpected shape of " + name);
            ++i;
        }
        return t;
    }
    static std::string bn(int il, const char * s) { return "v.blk." + std::to_string(il) + "." + s; }

    void load() {
        g = std::make_unique<gguf_file>(o.mmproj);
        if (g->get_str("clip.projector_type", "") != "qwen3vl_merger")
            throw std::runtime_error("nvit: projector type is not qwen3vl_merger");
        if (g->get_int("clip.vision.block_count", 0) != NL || g->get_int("clip.vision.embedding_length", 0) != E ||
            g->get_int("clip.vision.feed_forward_length", 0) != FF || g->get_int("clip.vision.attention.head_count", 0) != NH ||
            g->get_int("clip.vision.projection_dim", 0) != OUT || g->get_int("clip.vision.patch_size", 0) != 16 ||
            g->get_int("clip.vision.spatial_merge_size", 0) != 2)
            throw std::runtime_error("nvit: unexpected vision hyper-parameters (only the Qwen3.8-27B mmproj is supported)");
        eps = (float) g->get_float("clip.vision.attention.layer_norm_epsilon", 1e-6);
        q8_to16 = ggml_get_to_fp16_cuda(GGML_TYPE_Q8_0);

        // ---- raw blob: per layer [qkv q8][out q8][up q8][down f16 padded to 4352], then mm.0 q8, mm.2 q8 ----
        size_t off = 0;
        jobs.resize(NJOB);
        struct src { const gguf_tensor * t; size_t off; bool pad_down; };
        std::vector<src> srcs;
        for (int il = 0; il < NL; ++il) {
            job & J = jobs[il];
            const gguf_tensor * tq = T(bn(il, "attn_qkv.weight"), T_Q8_0, {E, QKV});
            const gguf_tensor * to = T(bn(il, "attn_out.weight"), T_Q8_0, {E, E});
            const gguf_tensor * tu = T(bn(il, "ffn_up.weight"), T_Q8_0, {E, FF});
            const gguf_tensor * td = T(bn(il, "ffn_down.weight"), T_F16, {FF, E});
            size_t roff = 0;
            for (auto [t, so, n] : {std::tuple{tq, SL_QKV, (int64_t) QKV * E}, std::tuple{to, SL_OUT, (int64_t) E * E},
                                    std::tuple{tu, SL_UP, (int64_t) FF * E}}) {
                piece p;
                p.off = off; p.bytes = t->nbytes; p.raw_off = roff; p.slot_off = so; p.type = T_Q8_0; p.n = n;
                J.pieces.push_back(p);
                srcs.push_back({t, off, false});
                off = al(off + t->nbytes);
                roff = al(roff + t->nbytes);
            }
            piece p;
            p.off = off; p.bytes = (size_t) E * FFP * 2; p.slot_off = SL_DOWN; p.type = T_F16; p.n = (int64_t) E * FFP;
            J.pieces.push_back(p);
            srcs.push_back({td, off, true});
            off = al(off + p.bytes);
        }
        const gguf_tensor * tm0 = T("mm.0.weight", T_Q8_0, {MERGE_IN, MERGE_IN});
        const gguf_tensor * tm2 = T("mm.2.weight", T_Q8_0, {MERGE_IN, OUT});
        const size_t m0 = off;
        srcs.push_back({tm0, off, false});
        off = al(off + tm0->nbytes);
        const size_t m2 = off;
        srcs.push_back({tm2, off, false});
        off = al(off + tm2->nbytes);
        blob_bytes = off;
        auto mjob = [&](job & J, size_t base, int64_t row0, int64_t rows) {
            piece p;
            p.off = base + (size_t) row0 * Q8_ROW_M; p.bytes = (size_t) rows * Q8_ROW_M; p.raw_off = 0; p.slot_off = 0;
            p.type = T_Q8_0; p.n = rows * MERGE_IN;
            J.pieces.push_back(p);
        };
        mjob(jobs[NL + 0], m0, 0, 2304);
        mjob(jobs[NL + 1], m0, 2304, 2304);
        mjob(jobs[NL + 2], m2, 0, 2560);
        mjob(jobs[NL + 3], m2, 2560, 2560);

        // host copy of the blob (pinned when it stays on the host: the uploads read it directly)
        std::vector<uint8_t> tmp;
        uint8_t * hb = nullptr;
        if (o.host_weights) {
            blob_host = (uint8_t *) hmalloc_pinned(blob_bytes);
            hb = blob_host;
        } else {
            tmp.resize(blob_bytes);
            hb = tmp.data();
        }
        for (const src & s : srcs) {
            if (!s.pad_down) {
                memcpy(hb + s.off, s.t->data, s.t->nbytes);
            } else {
                const uint16_t * a = (const uint16_t *) s.t->data;
                uint16_t * d = (uint16_t *) (hb + s.off);
                for (int r = 0; r < E; ++r) {
                    memcpy(d + (size_t) r * FFP, a + (size_t) r * FF, (size_t) FF * 2);
                    memset(d + (size_t) r * FFP + FF, 0, (size_t) (FFP - FF) * 2);
                }
            }
        }

        // ---- small f32 tensors (same on every GPU) ----
        std::vector<float> sm;
        auto put = [&](const std::string & name, int64_t n) {
            const gguf_tensor * t = T(name, T_F32, {n});
            const size_t at = sm.size();
            sm.insert(sm.end(), (const float *) t->data, (const float *) t->data + n);
            sm.resize(al(sm.size() * 4) / 4, 0.0f);
            return at;
        };
        std::vector<size_t> lo(NL * 8);
        for (int il = 0; il < NL; ++il) {
            lo[il * 8 + 0] = put(bn(il, "ln1.weight"), E);
            lo[il * 8 + 1] = put(bn(il, "ln1.bias"), E);
            lo[il * 8 + 2] = put(bn(il, "ln2.weight"), E);
            lo[il * 8 + 3] = put(bn(il, "ln2.bias"), E);
            lo[il * 8 + 4] = put(bn(il, "attn_qkv.bias"), QKV);
            lo[il * 8 + 5] = put(bn(il, "attn_out.bias"), E);
            lo[il * 8 + 6] = put(bn(il, "ffn_up.bias"), FF);
            lo[il * 8 + 7] = put(bn(il, "ffn_down.bias"), E);
        }
        const size_t o_post_w = put("v.post_ln.weight", E), o_post_b = put("v.post_ln.bias", E);
        const size_t o_mm0_b = put("mm.0.bias", MERGE_IN), o_mm2_b = put("mm.2.bias", OUT);
        const size_t o_pb = put("v.patch_embd.bias", E);
        // patch embedding: a still image is two identical temporal frames, conv(w0) + conv(w1) = conv(w0 + w1)
        const size_t o_pw = sm.size();
        {
            const gguf_tensor * w0 = T("v.patch_embd.weight", T_F32, {16, 16, 3, E});
            const gguf_tensor * w1 = T("v.patch_embd.weight.1", T_F32, {16, 16, 3, E});
            const float * a = (const float *) w0->data, * b = (const float *) w1->data;
            for (int64_t i = 0; i < (int64_t) E * PATCH_K; ++i) sm.push_back(a[i] + b[i]);
            sm.resize(al(sm.size() * 4) / 4, 0.0f);
        }
        const size_t o_pos = sm.size();
        {
            const gguf_tensor * t = T("v.position_embd.weight", T_F32, {E, POS_SIDE * POS_SIDE});
            sm.insert(sm.end(), (const float *) t->data, (const float *) t->data + (size_t) E * POS_SIDE * POS_SIDE);
        }
        small_floats = sm.size();

        // ---- per GPU ----
        R = (int) o.devices.size();
        if (R < 1 || R > 2) throw std::runtime_error("nvit: 1 or 2 devices");
        rk.resize(R);
        if (R == 2 && o.devices[0] != o.devices[1]) {   // (devices 0,0: both ranks on one GPU - a test of the split)
            for (int r = 0; r < 2; ++r) {
                int can = 0;
                CK(cudaDeviceCanAccessPeer(&can, o.devices[r], o.devices[1 - r]));
                if (!can) throw std::runtime_error("nvit: no P2P access between the two GPUs");
                CK(cudaSetDevice(o.devices[r]));
                const cudaError_t e = cudaDeviceEnablePeerAccess(o.devices[1 - r], 0);
                if (e == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
                else CK(e);
            }
        }
        for (int r = 0; r < R; ++r) {
            rank_t & K = rk[r];
            K.dev = o.devices[r];
            CK(cudaSetDevice(K.dev));
            // CUDA's least priority is the numerically greatest value of the range, and that is 0 = the default priority
            // (CUDA 12.9, GTX 1660: least 0, greatest -5): nothing ranks below the default, so these streams rank with
            // other default-priority work. Decode stays ahead of an encode because tp_model's main streams use the
            // greatest priority (P100E_STREAM_PRIO), plus the enqueue pacing in encode().
            int lo_p = 0, hi_p = 0;
            CK(cudaDeviceGetStreamPriorityRange(&lo_p, &hi_p));
            const int prio = o.stream_priority == 1 ? lo_p : 0;
            if (o.verbose)
                fprintf(stderr, "nvit: GPU %d streams at priority %d (range: least %d, greatest %d)\n", K.dev, prio, lo_p, hi_p);
            CK(cudaStreamCreateWithPriority(&K.ws, cudaStreamNonBlocking, prio));
            for (int c = 0; c < MAXCH; ++c) {
                CK(cudaStreamCreateWithPriority(&K.cs[c], cudaStreamNonBlocking, prio));
                if (c == 0) { CB(cublasCreate(&K.cb)); CB(cublasSetWorkspace(K.cb, nullptr, 0)); }
                CK(cudaEventCreateWithFlags(&K.ev_wdone[c], cudaEventDisableTiming));
                CK(cudaEventCreateWithFlags(&K.ev_qkv[c], cudaEventDisableTiming));
                CK(cudaEventCreateWithFlags(&K.ev_sent[c], cudaEventDisableTiming));
                CK(cudaEventCreateWithFlags(&K.ev_end[c], cudaEventDisableTiming));
            }
            for (int s = 0; s < 2; ++s) CK(cudaEventCreateWithFlags(&K.ev_w[s], cudaEventDisableTiming));
            CK(cudaEventCreateWithFlags(&K.ev_up, cudaEventDisableTiming));
            CK(cudaEventCreate(&K.t_beg)); CK(cudaEventCreate(&K.t_emb)); CK(cudaEventCreate(&K.t_lay)); CK(cudaEventCreate(&K.t_end));
            CK(cudaMalloc(&K.small, small_floats * 4));
            CK(cudaMemcpy(K.small, sm.data(), small_floats * 4, cudaMemcpyHostToDevice));
            K.L.resize(NL);
            for (int il = 0; il < NL; ++il) {
                layer_small & S = K.L[il];
                S.ln1_w = K.small + lo[il * 8 + 0]; S.ln1_b = K.small + lo[il * 8 + 1];
                S.ln2_w = K.small + lo[il * 8 + 2]; S.ln2_b = K.small + lo[il * 8 + 3];
                S.qkv_b = K.small + lo[il * 8 + 4]; S.o_b = K.small + lo[il * 8 + 5];
                S.up_b = K.small + lo[il * 8 + 6]; S.down_b = K.small + lo[il * 8 + 7];
            }
            K.post_w = K.small + o_post_w; K.post_b = K.small + o_post_b;
            K.mm0_b = K.small + o_mm0_b; K.mm2_b = K.small + o_mm2_b;
            K.patch_w = K.small + o_pw; K.patch_b = K.small + o_pb; K.pos = K.small + o_pos;
            if (!o.host_weights) {
                CK(cudaMalloc(&K.blob_dev, blob_bytes));
                CK(cudaMemcpy(K.blob_dev, hb, blob_bytes, cudaMemcpyHostToDevice));
            }
        }
    }

    // ---- scratch ----
    struct layout { size_t img, x, y, x16, cs, part, q16, kv16, st, slot, raw, total; };
    layout plan(int N, int nmax) const {
        layout L{};
        size_t at = 0;
        auto take = [&](size_t b) { const size_t a = at; at = al(at + b); return a; };
        L.img = take((size_t) 3 * N * 256 * 4);
        L.x = take((size_t) nmax * E * 4);
        L.y = take((size_t) nmax * FF * 4);
        L.x16 = take((size_t) nmax * FFP * 2);
        L.cs = take((size_t) nmax * 4);
        L.part = take((size_t) nmax * E * 4);
        L.q16 = take((size_t) NH * ((nmax + TILE - 1) / TILE) * TILE_U32 * 4);
        L.kv16 = take((size_t) 2 * 2 * NH * ((N + TILE - 1) / TILE) * TILE_U32 * 4);
        L.st = take((size_t) nmax * NH * 2 * 4);
        L.slot = take(2 * SLOT_BYTES);
        L.raw = o.host_weights ? take(2 * RAW_SLOT_BYTES) : at;
        L.total = at;
        return L;
    }
    // patches owned by each rank: rank 0 the first ceil(tiles / 2) 128-patch tiles (attention / K-V tiles of 64 and
    // merge groups of 4 never straddle the ranks), rank 1 the rest. An image of one tile (<= 128 patches = 32 tokens,
    // only with min_tokens <= 32) has nothing for rank 1: it runs on rank 0 alone, exactly the one-GPU path.
    // Returns the number of ranks in use.
    int split(int N, int * n0, int * n) const {
        n0[0] = 0; n[0] = N;
        n0[1] = N; n[1] = 0;
        const int t = (N + 127) / 128;
        if (R == 1 || t < 2) return 1;
        n[0] = 128 * ((t + 1) / 2);   // < N for t >= 2
        n0[1] = n[0]; n[1] = N - n[0];
        return 2;
    }
    // own scratch of >= bytes on ranks [0, RA); false when an allocation fails for lack of memory (error cleared)
    bool alloc_own(int RA, size_t bytes) {
        // test hook (p100e-run --overlap-bench): P100E_NVIT_TEST_OOM=1 pretends the allocation does not fit
        if (const char * t = getenv("P100E_NVIT_TEST_OOM")) if (atoi(t) > 0) return false;
        for (int r = 0; r < RA; ++r) {
            rank_t & K = rk[r];
            if (K.own_bytes >= bytes) continue;
            CK(cudaSetDevice(K.dev));
            if (K.own) { CK(cudaDeviceSynchronize()); CK(cudaFree(K.own)); K.own = nullptr; K.own_bytes = 0; }
            void * p = nullptr;
            const cudaError_t e = cudaMalloc(&p, bytes);
            if (e == cudaErrorMemoryAllocation) { cudaGetLastError(); return false; }
            CK(e);
            K.own = p;
            K.own_bytes = bytes;
        }
        return true;
    }
    // the own scratch of every rank freed (cudaFree waits for the device); never throws (used while unwinding too)
    void free_own() noexcept {
        for (auto & K : rk) {
            if (!K.own) continue;
            cudaSetDevice(K.dev);
            cudaFree(K.own);
            K.own = nullptr;
            K.own_bytes = 0;
        }
    }
    size_t scratch_for(int n_tokens) const {
        const int N = 4 * n_tokens;
        int n0[2], n[2];
        split(N, n0, n);
        return plan(N, n[0]).total;
    }

    // ---- forward ----
    void encode(const image & img, std::vector<float> & out, int & tx, int & ty, timing * tm,
                const std::vector<scratch_view> * borrow, encoder & self);
    void encode_oracle(const image & img, std::vector<float> & out, int & tx, int & ty, encoder & self);
};

encoder::encoder(const opts & o) : p_(new impl) {
    p_->o = o;
    if (p_->o.devices.empty()) p_->o.devices = {0};
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess) ndev = 0;
    cudaGetLastError();
    for (int d : p_->o.devices)
        if (d < 0 || d >= ndev) throw std::runtime_error("nvit: CUDA device " + std::to_string(d) + " not present (" +
                                                         std::to_string(ndev) + " visible)");
    try {
        p_->load();
    } catch (...) {
        cudaGetLastError();   // leave no stale CUDA error behind for whoever runs next (e.g. the mtmd fallback)
        throw;
    }
}

encoder::~encoder() = default;   // impl::~impl frees (also after a constructor that threw)

int encoder::n_embd() const { return OUT; }
int encoder::n_ranks() const { return p_->R; }
size_t encoder::scratch_bytes(int n_tokens) const { return p_->scratch_for(n_tokens); }
size_t encoder::permanent_bytes(int r) const {
    const rank_t & K = p_->rk.at(r);
    return p_->small_floats * 4 + (K.blob_dev ? p_->blob_bytes : 0) + K.own_bytes;
}
void encoder::release_scratch() {
    for (auto & K : p_->rk) {
        if (!K.own) continue;
        CK(cudaSetDevice(K.dev));
        CK(cudaDeviceSynchronize());
        CK(cudaFree(K.own));
        K.own = nullptr;
        K.own_bytes = 0;
    }
}

void encoder::encode(const image & img, std::vector<float> & out, int & tx, int & ty, timing * t,
                     const std::vector<scratch_view> * borrow) {
    if (p_->o.oracle) p_->encode_oracle(img, out, tx, ty, *this);
    else p_->encode(img, out, tx, ty, t, borrow, *this);
}

// fp32 reference path (debug): one GPU, synchronous, own allocations
void encoder::impl::encode_oracle(const image & img, std::vector<float> & out, int & tx, int & ty, encoder & self) {
    rank_t & K = rk[0];
    CK(cudaSetDevice(K.dev));
    const int pw = img.nx / 16, ph = img.ny / 16, N = pw * ph, NT = N / 4;
    tx = img.nx / 32;
    ty = img.ny / 32;
    cudaStream_t s = K.cs[0];
    cublasHandle_t cb = K.cb;
    CB(cublasSetStream(cb, s));
    std::vector<void *> allocs;
    auto dmal = [&](size_t bytes) { void * p = nullptr; CK(cudaMalloc(&p, bytes)); allocs.push_back(p); return (float *) p; };
    float * dimg = dmal(img.data.size() * 4);
    float * x = dmal((size_t) N * E * 4);
    float * ln = dmal((size_t) N * E * 4);
    float * big = dmal((size_t) N * FF * 4);
    float * att = dmal((size_t) N * E * 4);
    float * part = dmal((size_t) N * E * 4);
    float * q = dmal((size_t) N * E * 4), * k = dmal((size_t) N * E * 4), * v = dmal((size_t) N * E * 4);
    float * w32 = dmal((size_t) OUT * MERGE_IN * 4);
    float * dout = dmal((size_t) NT * OUT * 4);
    uint8_t * wraw = (uint8_t *) dmal((size_t) OUT * MERGE_IN * 2);
    std::vector<float> hbuf;
    auto dump = [&](const std::string & name, int layer, const float * d, int rows, int cols) {
        if (!self.dump) return;
        CK(cudaStreamSynchronize(s));
        hbuf.resize((size_t) rows * cols);
        CK(cudaMemcpy(hbuf.data(), d, hbuf.size() * 4, cudaMemcpyDeviceToHost));
        self.dump(name, layer, rows, cols, hbuf.data());
    };
    auto want = [&](int il) { return std::find(self.dump_layers.begin(), self.dump_layers.end(), il) != self.dump_layers.end(); };
    // weight piece -> f32 [rows][K] in w32
    auto wf32 = [&](const piece & p) {
        const uint8_t * src = K.blob_dev ? K.blob_dev + p.off : nullptr;
        if (!src) {
            CK(cudaMemcpyAsync(wraw, blob_host + p.off, p.bytes, cudaMemcpyHostToDevice, s));
            src = wraw;
        }
        ggml_get_to_fp32_cuda(p.type == T_F16 ? GGML_TYPE_F16 : GGML_TYPE_Q8_0)(src, w32, p.n, s);
        CK(cudaGetLastError());
    };
    const float one = 1.0f, zero = 0.0f;
    auto sgemm = [&](int rows, int Kd, int lda, const float * X, int ldx, float * Y, int ldy, int T) {
        CB(cublasSgemm(cb, CUBLAS_OP_T, CUBLAS_OP_N, rows, T, Kd, &one, w32, lda, X, ldx, &zero, Y, ldy));
    };
    CK(cudaMemcpyAsync(dimg, img.data.data(), img.data.size() * 4, cudaMemcpyHostToDevice, s));
    im2col_merge(dimg, img.nx, img.ny, 0, N, big, s);
    CB(cublasSgemm(cb, CUBLAS_OP_T, CUBLAS_OP_N, E, N, PATCH_K, &one, K.patch_w, PATCH_K, big, PATCH_K, &zero, x, E));
    add_bias_pos(x, K.patch_b, K.pos, pw, ph, 0, N, s);
    dump("inp_pos_emb", -1, x, N, E);
    for (int il = 0; il < NL; ++il) {
        const layer_small & L = K.L[il];
        const job & J = jobs[il];
        add_ln_prescale(1, x, nullptr, nullptr, L.ln1_w, L.ln1_b, eps, N, nullptr, 0, nullptr, ln, s);
        if (want(il)) dump("ln1", il, ln, N, E);
        wf32(J.pieces[0]);
        sgemm(QKV, E, E, ln, E, big, QKV, N);
        qkv_rope_f32(big, L.qkv_b, N, 0, pw, q, k, v, N, s);
        if (want(il)) {
            CK(cudaStreamSynchronize(s));
            std::vector<float> hq((size_t) N * E), hk((size_t) N * E), tq((size_t) N * E), tk((size_t) N * E);
            CK(cudaMemcpy(hq.data(), q, hq.size() * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(hk.data(), k, hk.size() * 4, cudaMemcpyDeviceToHost));
            for (int h = 0; h < NH; ++h)
                for (int i = 0; i < N; ++i)
                    for (int d = 0; d < HD; ++d) {
                        tq[(size_t) i * E + h * HD + d] = hq[((size_t) h * N + i) * HD + d];
                        tk[(size_t) i * E + h * HD + d] = hk[((size_t) h * N + i) * HD + d];
                    }
            if (self.dump) { self.dump("Qcur_rope", il, N, E, tq.data()); self.dump("Kcur_rope", il, N, E, tk.data()); }
        }
        attn_f32(q, k, v, N, att, s);
        if (want(il)) dump("kqv_out", il, att, N, E);
        wf32(J.pieces[1]);
        sgemm(E, E, E, att, E, part, E, N);
        add_ln_prescale(1, x, part, L.o_b, L.ln2_w, L.ln2_b, eps, N, nullptr, 0, nullptr, ln, s);
        if (want(il)) dump("ffn_inp", il, x, N, E);
        wf32(J.pieces[2]);
        sgemm(FF, E, E, ln, E, big, FF, N);
        bias_gelu_f32(big, L.up_b, N, FF, s);
        wf32(J.pieces[3]);
        sgemm(E, FF, FFP, big, FF, part, E, N);
        if (il + 1 < NL) {
            add_ln_prescale(1, x, part, L.down_b, nullptr, nullptr, eps, N, nullptr, 0, nullptr, nullptr, s);
        } else {
            add_ln_prescale(4, x, part, L.down_b, K.post_w, K.post_b, eps, N, nullptr, 0, nullptr, ln, s);
        }
        if (want(il)) dump("layer_out", il, x, N, E);
    }
    dump("post_ln", NL, ln, N, E);
    // merger: ln viewed as [NT][4608]
    piece m0 = jobs[NL].pieces[0], m2 = jobs[NL + 2].pieces[0];
    m0.bytes *= 2; m0.n *= 2;   // whole mm.0 (both row halves are contiguous in the blob)
    m2.bytes *= 2; m2.n *= 2;
    wf32(m0);
    sgemm(MERGE_IN, MERGE_IN, MERGE_IN, ln, MERGE_IN, big, MERGE_IN, NT);
    bias_gelu_f32(big, K.mm0_b, NT, MERGE_IN, s);
    wf32(m2);
    sgemm(OUT, MERGE_IN, MERGE_IN, big, MERGE_IN, dout, OUT, NT);
    add_bias(dout, K.mm2_b, NT, OUT, s);
    CK(cudaStreamSynchronize(s));
    out.resize((size_t) NT * OUT);
    CK(cudaMemcpy(out.data(), dout, out.size() * 4, cudaMemcpyDeviceToHost));
    if (self.dump) self.dump("embd", NL + 1, NT, OUT, out.data());
    for (void * p : allocs) cudaFree(p);
}

void encoder::impl::encode(const image & img, std::vector<float> & out, int & tx, int & ty, timing * tm,
                           const std::vector<scratch_view> * borrow, encoder & self) {
    using clk = std::chrono::steady_clock;
    const auto w0 = clk::now();
    if (img.nx % 32 || img.ny % 32 || img.nx <= 0 || img.ny <= 0 || img.data.size() != (size_t) 3 * img.nx * img.ny)
        throw std::runtime_error("nvit::encode: image sides must be positive multiples of 32 (use nvit::preprocess)");
    const int pw = img.nx / 16, ph = img.ny / 16, N = pw * ph, NT = N / 4;
    tx = img.nx / 32;
    ty = img.ny / 32;
    const bool dbg = (bool) self.dump;
    int n0[2], nn[2];
    const int RA = split(N, n0, nn);   // ranks in use (1 for an image of <= 32 tokens, also on two GPUs)
    struct rank_span { rank_t * b, * e; rank_t * begin() const { return b; } rank_t * end() const { return e; } };
    const rank_span act{rk.data(), rk.data() + RA};   // the ranks in use
    const layout LY = plan(N, nn[0]);
    std::vector<scratch_view> lent;
    // With a lender (scratch_source) that lends nothing usable, the own scratch is for this encode only: sized for this
    // image and freed when the encode returns, so the next encode borrows again (instead of keeping it for the process).
    bool own_temp = false;
    // On an exception: finish this encode's kernels before the scratch goes away (the lent memory with its leases, which
    // a waiting prefill takes at once, or the temporary own scratch). Declared after `lent`: runs before it is released.
    struct end_guard {
        impl & I;
        int RA;
        bool & own_temp;
        ~end_guard() {
            if (std::uncaught_exceptions() == 0) return;
            for (int r = 0; r < RA; ++r) { cudaSetDevice(I.rk[r].dev); cudaDeviceSynchronize(); }
            if (own_temp) I.free_own();
            cudaGetLastError();   // leave no error of this cleanup pending
        }
    } guard{*this, RA, own_temp};
    if (!borrow && self.scratch_source) {   // e.g. the LLM's idle prefill ring
        bool busy = false;                  // nothing lent: the lender is busy (a prefill runs or waits to start)
        size_t offered = 0;                 // else the smallest entry it offered
        auto try_lend = [&]() {
            lent.clear();
            const std::vector<scratch_view> got = self.scratch_source();
            for (int r = 0; r < RA; ++r) {
                for (size_t i = 0; i < got.size(); ++i)
                    if ((got[i].dev >= 0 ? got[i].dev == rk[r].dev : (int) i == r) && got[i].p && got[i].bytes >= LY.total) {
                        lent.push_back(got[i]);
                        break;
                    }
            }
            if ((int) lent.size() == RA) return true;   // kept (with its leases, scratch_view::hold) until this encode returns
            lent.clear();   // not used: give the leases back now (a prefill may be waiting for them)
            busy = got.empty();
            offered = 0;
            for (const auto & g : got) if (g.p) offered = offered ? std::min(offered, g.bytes) : g.bytes;
            return false;
        };
        if (try_lend()) {
            borrow = &lent;
        } else {
            own_temp = true;
            if (busy)
                fprintf(stderr, "nvit: no lent scratch (the lender is busy: a prefill is running or waiting): own scratch "
                                "for this %d-token image (%.1f MiB per GPU), freed after it\n", NT, LY.total / 1048576.0);
            else
                fprintf(stderr, "nvit: lent scratch too small for this %d-token image (%.1f MiB per GPU needed, %.1f MiB "
                                "offered): own scratch, freed after it\n", NT, LY.total / 1048576.0, offered / 1048576.0);
            // not enough free VRAM for it: wait for the lender instead of failing (a busy lender returns its memory when
            // its prefill ends); a lender whose memory is too small cannot help, nor can waiting beyond the limit
            static const double wait_s = getenv("P100E_NVIT_SCRATCH_WAIT_S") ? atof(getenv("P100E_NVIT_SCRATCH_WAIT_S")) : 600.0;
            const auto t0 = clk::now();
            bool said = false;
            while (!alloc_own(RA, LY.total)) {
                const double waited = std::chrono::duration<double>(clk::now() - t0).count();
                if (!busy || waited > wait_s) {
                    free_own();
                    throw std::runtime_error("nvit::encode: not enough free VRAM for " + std::to_string(LY.total >> 20) +
                                             " MiB of scratch per GPU (" + (busy ? "the lender stayed busy for " +
                                             std::to_string((int) waited) + " s" : std::string("no lender with enough memory")) + ")");
                }
                if (!said) {
                    fprintf(stderr, "nvit: not enough free VRAM for the own scratch (%.1f MiB per GPU): waiting for the "
                                    "lender (up to %.0f s, P100E_NVIT_SCRATCH_WAIT_S)\n", LY.total / 1048576.0, wait_s);
                    said = true;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                if (try_lend()) {
                    borrow = &lent;
                    free_own();   // a rank that got its own part before the lend
                    own_temp = false;
                    fprintf(stderr, "nvit: lent scratch after waiting %.0f ms\n",
                            std::chrono::duration<double, std::milli>(clk::now() - t0).count());
                    break;
                }
            }
        }
    }

    // ---- scratch and chunks ----
    for (int r = 0; r < RA; ++r) {
        rank_t & K = rk[r];
        CK(cudaSetDevice(K.dev));
        uint8_t * base = nullptr;
        if (borrow) {
            if ((int) borrow->size() < RA || (*borrow)[r].bytes < LY.total)
                throw std::runtime_error("nvit::encode: borrowed scratch too small (" + std::to_string(LY.total) + " bytes needed)");
            base = (uint8_t *) (*borrow)[r].p;
        } else {
            if (K.own_bytes < LY.total) {
                if (K.own) { CK(cudaDeviceSynchronize()); CK(cudaFree(K.own)); K.own = nullptr; K.own_bytes = 0; }
                const size_t want = std::max(LY.total, scratch_for(o.max_tokens));
                CK(cudaMalloc(&K.own, want));
                K.own_bytes = want;
                fprintf(stderr, "nvit: own scratch allocated on GPU %d: %.1f MiB (for %d tokens; kept until "
                                "release_scratch)\n", K.dev, want / 1048576.0, std::max(o.max_tokens, NT));
            }
            base = (uint8_t *) K.own;
        }
        static const bool poison = getenv("NVIT_POISON") && atoi(getenv("NVIT_POISON")) != 0;
        if (poison) {   // debug: every scratch byte 0xff (NaN as f16 and f32) - an uninitialised read shows up as NaN
            CK(cudaMemset(base, 0xff, LY.total));
            CK(cudaDeviceSynchronize());
        }
        K.img = (float *) (base + LY.img);
        K.x = (float *) (base + LY.x);
        K.y = (float *) (base + LY.y);
        K.x16 = (half *) (base + LY.x16);
        K.cs_ = (float *) (base + LY.cs);
        K.part = (float *) (base + LY.part);
        K.qt = (uint32_t *) (base + LY.q16);
        K.kvt = (uint32_t *) (base + LY.kv16);
        K.st = (float *) (base + LY.st);
        K.slot[0] = (half *) (base + LY.slot);
        K.slot[1] = (half *) (base + LY.slot + SLOT_BYTES);
        K.raw[0] = base + LY.raw;
        K.raw[1] = base + LY.raw + RAW_SLOT_BYTES;
        K.n0 = n0[r];
        K.n = nn[r];
        // chunks: ~1024 patches each (whole 128-row GEMM tiles), at most MAXCH; one chunk in dump mode
        int nch = o.chunks > 0 ? o.chunks : std::max(1, (K.n + 1023) / 1024);
        nch = std::min({nch, MAXCH, std::max(1, K.n / 128)});
        if (dbg) nch = 1;
        K.nch = nch;
        const int tiles = (K.n + 127) / 128;
        for (int c = 0; c <= nch; ++c) K.cb_[c] = std::min(K.n, (int) ((int64_t) tiles * c / nch) * 128);
        K.cb_[nch] = K.n;
    }
    // pinned output staging, sized for max_tokens; an image above that gets a buffer of its own size, given back when
    // the encode returns (end of this function) or, after a throw, replaced here by the next encode: never pinned for good
    const size_t out_bytes = (size_t) NT * OUT * 4, out_std = (size_t) o.max_tokens * OUT * 4;
    if (out_h_bytes < out_bytes || out_h_bytes > std::max(out_bytes, out_std)) {
        if (out_h) { CK(cudaFreeHost(out_h)); out_h = nullptr; out_h_bytes = 0; }
        out_h = (float *) hmalloc_pinned(std::max(out_bytes, out_std));
        out_h_bytes = std::max(out_bytes, out_std);   // after the allocation: a failed one leaves 0 bytes, not a stale size
    }
    // pinned input staging (the GPUs' band uploads then run as async DMA, both at once); same sizing
    const size_t img_bytes = img.data.size() * 4, img_std = (size_t) o.max_tokens * 1024 * 3 * 4;
    if (img_h_bytes < img_bytes || img_h_bytes > std::max(img_bytes, img_std)) {
        if (img_h) { CK(cudaFreeHost(img_h)); img_h = nullptr; img_h_bytes = 0; }
        img_h = (float *) hmalloc_pinned(std::max(img_bytes, img_std));
        img_h_bytes = std::max(img_bytes, img_std);
    }
    memcpy(img_h, img.data.data(), img_bytes);

    // ---- weight jobs ----
    auto prep_job = [&](int r, int j) {
        rank_t & K = rk[r];
        const int s = j & 1;
        CK(cudaSetDevice(K.dev));
        for (int c = 0; c < K.nch; ++c) CK(cudaStreamWaitEvent(K.ws, K.ev_wdone[c], 0));
        for (const piece & p : jobs[j].pieces) {
            if (p.type == T_F16) {
                if (o.host_weights)
                    CK(cudaMemcpyAsync((uint8_t *) K.slot[s] + p.slot_off, blob_host + p.off, p.bytes, cudaMemcpyHostToDevice, K.ws));
                continue;   // resident: the GEMM reads the blob
            }
            const uint8_t * src = K.blob_dev ? K.blob_dev + p.off : K.raw[s] + p.raw_off;
            if (o.host_weights) CK(cudaMemcpyAsync(K.raw[s] + p.raw_off, blob_host + p.off, p.bytes, cudaMemcpyHostToDevice, K.ws));
            q8_to16(src, (half *) ((uint8_t *) K.slot[s] + p.slot_off), p.n, K.ws);
            CK(cudaGetLastError());
        }
        CK(cudaEventRecord(K.ev_w[s], K.ws));
    };
    auto wdown = [&](rank_t & K, int il) -> const half * {
        const piece & p = jobs[il].pieces[3];
        return K.blob_dev ? (const half *) (K.blob_dev + p.off) : (const half *) ((uint8_t *) K.slot[il & 1] + SL_DOWN);
    };
    auto wslot = [&](rank_t & K, int j, size_t off) { return (const half *) ((uint8_t *) K.slot[j & 1] + off); };
    auto gemm = [&](rank_t & K, int c, const half * w, int rows_w, int K_, float * y, int64_t ldy, int T) {
        fast::fold_seg sg;
        sg.w = w; sg.y = y; sg.ldy = ldy; sg.rows = rows_w;
        const int a = K.cb_[c];
        fast::fold_gemm(&sg, 1, K.x16 + (int64_t) a * FFP, K_, K.cs_ + a, T, K_, K.cs[c]);
    };
    const float qscale = (1.0f / std::sqrt((float) HD)) * 1.4426950408889634f;
    const int nqt = (nn[0] + TILE - 1) / TILE, ntk = (N + TILE - 1) / TILE;
    const int64_t q_hs = (int64_t) nqt * TILE_U32;   // Q head stride in u32 (sized for the larger share)
    const int64_t kv_hs = (int64_t) ntk * TILE_U32;  // K / V head stride in u32
    auto kvbuf = [&](rank_t & K, int par) { return K.kvt + (size_t) par * 2 * NH * kv_hs; };   // K heads, then V heads

    // debug dumps: sync everything, gather rows of every rank
    std::vector<float> hbuf;
    auto sync_all = [&]() { for (auto & K : act) { CK(cudaSetDevice(K.dev)); CK(cudaDeviceSynchronize()); } };
    auto dump_rows = [&](const std::string & name, int layer, int cols, auto ptr_of) {   // ptr_of(K) -> device f32 [K.n][cols]
        sync_all();
        hbuf.resize((size_t) N * cols);
        for (auto & K : act) {
            CK(cudaSetDevice(K.dev));
            CK(cudaMemcpy(hbuf.data() + (size_t) K.n0 * cols, ptr_of(K), (size_t) K.n * cols * 4, cudaMemcpyDeviceToHost));
        }
        self.dump(name, layer, N, cols, hbuf.data());
    };
    auto want = [&](int il) { return dbg && std::find(self.dump_layers.begin(), self.dump_layers.end(), il) != self.dump_layers.end(); };
    auto dump_qk = [&](int il, int par) {
        sync_all();
        std::vector<half> hq;
        std::vector<float> fq((size_t) N * E), fk((size_t) N * E);
        for (auto & K : act) {
            CK(cudaSetDevice(K.dev));
            hq.resize((size_t) NH * q_hs * 2);
            CK(cudaMemcpy(hq.data(), K.qt, hq.size() * 2, cudaMemcpyDeviceToHost));
            for (int h = 0; h < NH; ++h)
                for (int i = 0; i < K.n; ++i)
                    for (int d = 0; d < HD; ++d)
                        fq[(size_t) (K.n0 + i) * E + h * HD + d] =
                            __half2float(hq[2 * ((size_t) h * q_hs + (size_t) (i / TILE) * TILE_U32 + (d / 2) * TILE + i % TILE) + d % 2]) / qscale;
        }
        self.dump("Qcur_rope", il, N, E, fq.data());
        rank_t & K = rk[0];
        CK(cudaSetDevice(K.dev));
        std::vector<half> hk((size_t) NH * kv_hs * 2);
        CK(cudaMemcpy(hk.data(), kvbuf(K, par), hk.size() * 2, cudaMemcpyDeviceToHost));
        for (int h = 0; h < NH; ++h)
            for (int i = 0; i < N; ++i)
                for (int d = 0; d < HD; ++d)
                    fk[(size_t) i * E + h * HD + d] =
                        __half2float(hk[2 * ((size_t) h * kv_hs + (size_t) (i / TILE) * TILE_U32 + (d / 2) * TILE + i % TILE) + d % 2]);
        self.dump("Kcur_rope", il, N, E, fk.data());
    };

    // ---- start: weight jobs 0 and 1, image upload ----
    for (int r = 0; r < RA; ++r) { prep_job(r, 0); prep_job(r, 1); }
    for (int r = 0; r < RA; ++r) {
        rank_t & K = rk[r];
        CK(cudaSetDevice(K.dev));
        CK(cudaEventRecord(K.t_beg, K.cs[0]));
        {   // only the pixel rows of this GPU's patches (whole 32-px token rows), each of the 3 planes
            const int g0 = K.n0 / 4, g1 = (K.n0 + K.n) / 4;   // merged tokens [g0, g1)
            const int y0 = (g0 / tx) * 32, y1 = std::min(img.ny, ((g1 - 1) / tx + 1) * 32);
            const size_t plane = (size_t) img.nx * img.ny * 4, off = (size_t) y0 * img.nx * 4;
            CK(cudaMemcpy2DAsync((char *) K.img + off, plane, (const char *) img_h + off, plane, (size_t) (y1 - y0) * img.nx * 4,
                                 3, cudaMemcpyHostToDevice, K.cs[0]));
        }
        CK(cudaEventRecord(K.ev_up, K.cs[0]));
        for (int c = 1; c < K.nch; ++c) CK(cudaStreamWaitEvent(K.cs[c], K.ev_up, 0));
    }

    // qkv GEMM of layer il for chunk c (x16 holds LN1(il) of the chunk) + rope + K/V to the peer
    auto qkv_step = [&](int r, int c, int il) {
        rank_t & K = rk[r];
        const int a = K.cb_[c], T = K.cb_[c + 1] - a;
        CK(cudaStreamWaitEvent(K.cs[c], K.ev_w[il & 1], 0));
        gemm(K, c, wslot(K, il, SL_QKV), QKV, E, K.y + (int64_t) a * FF, QKV, T);
        uint32_t * kv = kvbuf(K, il & 1);
        qkv_rope(K.y + (int64_t) a * FF, K.L[il].qkv_b, T, K.n0 + a, pw, qscale, K.qt, q_hs, a, kv, kv + NH * kv_hs, kv_hs, K.cs[c]);
        CK(cudaEventRecord(K.ev_qkv[c], K.cs[c]));
        if (RA == 2) {
            rank_t & P = rk[1 - r];
            uint32_t * pkv = kvbuf(P, il & 1);
            const int tl0 = (K.n0 + a) / TILE, ntl = (T + TILE - 1) / TILE;
            const size_t pitch = (size_t) kv_hs * 4, width = (size_t) ntl * TILE_U32 * 4, off = (size_t) tl0 * TILE_U32;
            cudaError_t e2 = p2p_2d ? cudaMemcpy2DAsync(pkv + off, pitch, kv + off, pitch, width, 2 * NH, cudaMemcpyDefault, K.cs[c])
                                    : cudaErrorNotSupported;
            if (e2 != cudaSuccess) {   // per-row peer copies (32 per chunk) if the 2D peer copy is refused
                cudaGetLastError();
                p2p_2d = false;
                for (int rw = 0; rw < 2 * NH; ++rw)
                    CK(cudaMemcpyPeerAsync(pkv + off + rw * kv_hs, P.dev, kv + off + rw * kv_hs, K.dev, width, K.cs[c]));
            }
            CK(cudaEventRecord(K.ev_sent[c], K.cs[c]));
        }
    };

    // ---- embedding + layer 0 qkv ----
    for (int r = 0; r < RA; ++r) {
        rank_t & K = rk[r];
        CK(cudaSetDevice(K.dev));
        for (int c = 0; c < K.nch; ++c) {
            const int a = K.cb_[c], T = K.cb_[c + 1] - a;
            float * A = K.y + (int64_t) a * FF;
            im2col_merge(K.img, img.nx, img.ny, K.n0 + a, T, A, K.cs[c]);
            const float one = 1.0f, zero = 0.0f;
            CB(cublasSetStream(K.cb, K.cs[c]));
            CB(cublasSgemm(K.cb, CUBLAS_OP_T, CUBLAS_OP_N, E, T, PATCH_K, &one, K.patch_w, PATCH_K, A, PATCH_K, &zero,
                           K.x + (int64_t) a * E, E));
            add_bias_pos(K.x + (int64_t) a * E, K.patch_b, K.pos, pw, ph, K.n0 + a, T, K.cs[c]);
        }
    }
    if (dbg) dump_rows("inp_pos_emb", -1, E, [&](rank_t & K) { return K.x; });
    for (int r = 0; r < RA; ++r) {
        rank_t & K = rk[r];
        CK(cudaSetDevice(K.dev));
        for (int c = 0; c < K.nch; ++c) {
            const int a = K.cb_[c], T = K.cb_[c + 1] - a;
            add_ln_prescale(1, K.x + (int64_t) a * E, nullptr, nullptr, K.L[0].ln1_w, K.L[0].ln1_b, eps, T,
                            K.x16 + (int64_t) a * FFP, E, K.cs_ + a, want(0) ? K.part + (int64_t) a * E : nullptr, K.cs[c]);
        }
    }
    if (want(0)) dump_rows("ln1", 0, E, [&](rank_t & K) { return K.part; });
    for (int r = 0; r < RA; ++r) {
        rank_t & K = rk[r];
        CK(cudaSetDevice(K.dev));
        for (int c = 0; c < K.nch; ++c) qkv_step(r, c, 0);
        CK(cudaEventRecord(K.t_emb, K.cs[0]));
    }
    if (want(0)) dump_qk(0, 0);

    // ---- blocks ----
    // Pacing (P100E_NVIT_PACE, default 1): before enqueueing layer il, the host waits until every chunk finished
    // layer il-1's down GEMM (ev_wdone), so at most ~1 layer of encoder work sits in the GPU queues. Enqueueing the whole
    // encode at once (~3000 launches) let a decode stream that shares a hardware queue with an encoder stream (CUDA's
    // default 8 connections) wait out the entire encode (one 0.69 s decode step).
    static const bool pace = !getenv("P100E_NVIT_PACE") || atoi(getenv("P100E_NVIT_PACE")) != 0;
    for (int il = 0; il < NL; ++il) {
        const bool last = il == NL - 1;
        if (pace && il > 0 && !dbg) {
            for (auto & K : act) {
                CK(cudaSetDevice(K.dev));
                for (int c = 0; c < K.nch; ++c) CK(cudaEventSynchronize(K.ev_wdone[c]));
            }
        }
        // attention in two passes on two GPUs: own key tiles as soon as this GPU's chunks have their K / V (the peer's
        // rows are still in flight), then the peer's tiles from the carried softmax state. One GPU: one pass.
        for (int r = 0; r < RA; ++r) {
            rank_t & K = rk[r];
            CK(cudaSetDevice(K.dev));
            const uint32_t * kv = kvbuf(K, il & 1);
            const int own0 = K.n0 / TILE, own1 = (K.n0 + K.n + TILE - 1) / TILE;
            for (int c = 0; c < K.nch; ++c) {
                for (int c2 = 0; c2 < K.nch; ++c2) if (c2 != c) CK(cudaStreamWaitEvent(K.cs[c], K.ev_qkv[c2], 0));
                const int a = K.cb_[c], T = K.cb_[c + 1] - a;
                float * y = K.y + (int64_t) a * FF;   // chunk region of y (sized for the widest use)
                attn72(K.qt, q_hs, a, T, kv, kv + NH * kv_hs, kv_hs, N, NH, y, E, K.cs[c], own0, own1, RA == 2 ? 1 : 0,
                       K.st + (int64_t) a * NH * 2);
            }
        }
        if (RA == 2) {
            for (int r = 0; r < RA; ++r) {
                rank_t & K = rk[r];
                rank_t & P = rk[1 - r];
                CK(cudaSetDevice(K.dev));
                const uint32_t * kv = kvbuf(K, il & 1);
                const int p0t = P.n0 / TILE, p1t = (P.n0 + P.n + TILE - 1) / TILE;
                for (int c = 0; c < K.nch; ++c) {
                    for (int c2 = 0; c2 < P.nch; ++c2) CK(cudaStreamWaitEvent(K.cs[c], P.ev_sent[c2], 0));
                    const int a = K.cb_[c], T = K.cb_[c + 1] - a;
                    float * y = K.y + (int64_t) a * FF;
                    attn72(K.qt, q_hs, a, T, kv, kv + NH * kv_hs, kv_hs, N, NH, y, E, K.cs[c], p0t, p1t, 2,
                           K.st + (int64_t) a * NH * 2);
                }
            }
        }
        for (int r = 0; r < RA; ++r) {
            rank_t & K = rk[r];
            CK(cudaSetDevice(K.dev));
            for (int c = 0; c < K.nch; ++c) {
                if (want(il)) continue;   // dump mode: prescale below, after the dump (one chunk)
                const int a = K.cb_[c], T = K.cb_[c + 1] - a;
                fast::prescale_rows(K.y + (int64_t) a * FF, E, T, E, K.x16 + (int64_t) a * FFP, E, K.cs_ + a, K.cs[c]);
            }
        }
        if (want(il)) {
            dump_rows("kqv_out", il, E, [&](rank_t & K) { return K.y; });
            for (int r = 0; r < RA; ++r) {
                rank_t & K = rk[r];
                CK(cudaSetDevice(K.dev));
                fast::prescale_rows(K.y, E, K.n, E, K.x16, E, K.cs_, K.cs[0]);
            }
        }
        for (int r = 0; r < RA; ++r) {
            rank_t & K = rk[r];
            CK(cudaSetDevice(K.dev));
            for (int c = 0; c < K.nch; ++c) {
                const int a = K.cb_[c], T = K.cb_[c + 1] - a;
                gemm(K, c, wslot(K, il, SL_OUT), E, E, K.part + (int64_t) a * E, E, T);
                add_ln_prescale(1, K.x + (int64_t) a * E, K.part + (int64_t) a * E, K.L[il].o_b, K.L[il].ln2_w, K.L[il].ln2_b,
                                eps, T, K.x16 + (int64_t) a * FFP, E, K.cs_ + a, nullptr, K.cs[c]);
            }
        }
        if (want(il)) dump_rows("ffn_inp", il, E, [&](rank_t & K) { return K.x; });
        for (int r = 0; r < RA; ++r) {
            rank_t & K = rk[r];
            CK(cudaSetDevice(K.dev));
            for (int c = 0; c < K.nch; ++c) {
                const int a = K.cb_[c], T = K.cb_[c + 1] - a;
                float * y = K.y + (int64_t) a * FF;
                gemm(K, c, wslot(K, il, SL_UP), FF, E, y, FF, T);
                bias_gelu_prescale(y, FF, K.L[il].up_b, T, FF, FFP, K.x16 + (int64_t) a * FFP, FFP, K.cs_ + a, K.cs[c]);
                gemm(K, c, wdown(K, il), E, FFP, K.part + (int64_t) a * E, E, T);
                CK(cudaEventRecord(K.ev_wdone[c], K.cs[c]));
                if (!last) {
                    add_ln_prescale(1, K.x + (int64_t) a * E, K.part + (int64_t) a * E, K.L[il].down_b, K.L[il + 1].ln1_w,
                                    K.L[il + 1].ln1_b, eps, T, K.x16 + (int64_t) a * FFP, E, K.cs_ + a,
                                    want(il + 1) ? K.y + (int64_t) a * FF : nullptr, K.cs[c]);
                } else {
                    // post-LN of each merge group (4 patches) -> x16 [T/4][4608]
                    add_ln_prescale(4, K.x + (int64_t) a * E, K.part + (int64_t) a * E, K.L[il].down_b, K.post_w, K.post_b,
                                    eps, T, K.x16 + (int64_t) a * FFP, MERGE_IN, K.cs_ + a,
                                    dbg ? K.y + (int64_t) a * FF : nullptr, K.cs[c]);
                }
            }
        }
        if (want(il)) dump_rows("layer_out", il, E, [&](rank_t & K) { return K.x; });
        if (!last && want(il + 1)) dump_rows("ln1", il + 1, E, [&](rank_t & K) { return K.y; });
        if (last && dbg) dump_rows("post_ln", NL, E, [&](rank_t & K) { return K.y; });
        // the weight slot of layer il is free once every chunk's down GEMM ran: prepare job il + 2 there
        for (int r = 0; r < RA; ++r) if (il + 2 < NJOB) prep_job(r, il + 2);
        if (!last) {
            for (int r = 0; r < RA; ++r) {
                rank_t & K = rk[r];
                CK(cudaSetDevice(K.dev));
                for (int c = 0; c < K.nch; ++c) qkv_step(r, c, il + 1);
            }
            if (want(il + 1)) dump_qk(il + 1, (il + 1) & 1);
        }
    }
    for (int r = 0; r < RA; ++r) { CK(cudaSetDevice(rk[r].dev)); CK(cudaEventRecord(rk[r].t_lay, rk[r].cs[0])); }

    // ---- merger: mm.0 halves (jobs 27, 28) -> GELU -> mm.2 halves (jobs 29, 30) ----
    for (int r = 0; r < RA; ++r) {
        rank_t & K = rk[r];
        CK(cudaSetDevice(K.dev));
        for (int c = 0; c < K.nch; ++c) {
            const int a = K.cb_[c], G = (K.cb_[c + 1] - a) / 4;
            float * y = K.y + (int64_t) a * FF;
            for (int h = 0; h < 2; ++h) {
                const int j = NL + h;
                CK(cudaStreamWaitEvent(K.cs[c], K.ev_w[j & 1], 0));
                fast::fold_seg sg;
                sg.w = wslot(K, j, 0); sg.y = y + h * 2304; sg.ldy = MERGE_IN; sg.rows = 2304;
                fast::fold_gemm(&sg, 1, K.x16 + (int64_t) a * FFP, MERGE_IN, K.cs_ + a, G, MERGE_IN, K.cs[c]);
                CK(cudaEventRecord(K.ev_wdone[c], K.cs[c]));
                if (h == 0) { /* job 29 waits for this chunk's half-0 use */ }
            }
        }
    }
    // jobs 29 / 30 reuse the slots of 27 / 28: every chunk must be past both mm.0 halves (ev_wdone after half 1)
    for (int r = 0; r < RA; ++r) { prep_job(r, NL + 2); prep_job(r, NL + 3); }
    for (int r = 0; r < RA; ++r) {
        rank_t & K = rk[r];
        CK(cudaSetDevice(K.dev));
        for (int c = 0; c < K.nch; ++c) {
            const int a = K.cb_[c], G = (K.cb_[c + 1] - a) / 4;
            float * y = K.y + (int64_t) a * FF;
            bias_gelu_prescale(y, MERGE_IN, K.mm0_b, G, MERGE_IN, MERGE_IN, K.x16 + (int64_t) a * FFP, MERGE_IN, K.cs_ + a, K.cs[c]);
            float * outd = y;   // [G][5120], the mm.0 output is consumed
            for (int h = 0; h < 2; ++h) {
                const int j = NL + 2 + h;
                CK(cudaStreamWaitEvent(K.cs[c], K.ev_w[j & 1], 0));
                fast::fold_seg sg;
                sg.w = wslot(K, j, 0); sg.y = outd + h * 2560; sg.ldy = OUT; sg.rows = 2560;
                fast::fold_gemm(&sg, 1, K.x16 + (int64_t) a * FFP, MERGE_IN, K.cs_ + a, G, MERGE_IN, K.cs[c]);
            }
            add_bias(outd, K.mm2_b, G, OUT, K.cs[c]);
            CK(cudaMemcpyAsync(out_h + (size_t) (K.n0 + a) / 4 * OUT, outd, (size_t) G * OUT * 4, cudaMemcpyDeviceToHost, K.cs[c]));
            CK(cudaEventRecord(K.ev_end[c], K.cs[c]));
        }
        for (int c = 1; c < K.nch; ++c) CK(cudaStreamWaitEvent(K.cs[0], K.ev_end[c], 0));
        CK(cudaEventRecord(K.t_end, K.cs[0]));
    }
    for (auto & K : act) { CK(cudaSetDevice(K.dev)); CK(cudaStreamSynchronize(K.cs[0])); CK(cudaStreamSynchronize(K.ws)); }
    if (own_temp) {   // the next encode borrows again
        for (auto & K : rk)
            if (K.own) { CK(cudaSetDevice(K.dev)); CK(cudaFree(K.own)); K.own = nullptr; K.own_bytes = 0; }
        own_temp = false;
    }
    out.resize((size_t) NT * OUT);
    memcpy(out.data(), out_h, out_bytes);
    // staging grown past the max_tokens size by this image: back to the host now (the next encode pins the usual size)
    if (out_h_bytes > out_std) { CK(cudaFreeHost(out_h)); out_h = nullptr; out_h_bytes = 0; }
    if (img_h_bytes > img_std) { CK(cudaFreeHost(img_h)); img_h = nullptr; img_h_bytes = 0; }
    if (dbg) self.dump("embd", NL + 1, NT, OUT, out.data());
    if (tm) {
        float a = 0, b = 0, c = 0;
        CK(cudaSetDevice(rk[0].dev));
        CK(cudaEventElapsedTime(&a, rk[0].t_beg, rk[0].t_emb));
        CK(cudaEventElapsedTime(&b, rk[0].t_emb, rk[0].t_lay));
        CK(cudaEventElapsedTime(&c, rk[0].t_lay, rk[0].t_end));
        tm->prep_ms = a;
        tm->layers_ms = b;
        tm->merger_ms = c;
        tm->total_ms = std::chrono::duration<double, std::milli>(clk::now() - w0).count();
    }
}

} // namespace p100e::nvit
