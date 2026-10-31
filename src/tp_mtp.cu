#include "tp_mtp.h"
#include "kernels.cuh"
#include "fast/fused.cuh"

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace p100e {

#include "draft_vocab_ids.inc"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) throw std::runtime_error( \
    std::string("CUDA ") + cudaGetErrorString(e_) + " at " + __FILE__ + ":" + std::to_string(__LINE__)); } while (0)

static std::string bn(int il, const char * s) { return "blk." + std::to_string(il) + "." + s; }
static const char * kDraftHead = "mtp.draft_head";   // S.w name of the subset LM head

int argmax(const float * l, int n) {
    int b = 0;
    for (int i = 1; i < n; ++i) if (l[i] > l[b]) b = i;
    return b;
}

// ---- greedy pick + confidence of one logit row: (max, argmax, sum exp(l - max), second max), one block ----
struct dconf { float m; int32_t i; float s; float m2; };

// merge partial state b into a (a, b: max m (index i, lower index on ties), sum of exp(l - m), second largest m2)
static __device__ __forceinline__ void cmerge(float & m, int & i, float & s, float & m2, float bm, int bi, float bs, float bm2) {
    if (bm == -INFINITY) return;
    if (m == -INFINITY) { m = bm; i = bi; s = bs; m2 = bm2; return; }
    const bool bw = bm > m || (bm == m && bi < i);
    const float M = fmaxf(m, bm);
    s = s * __expf(m - M) + bs * __expf(bm - M);
    m2 = fmaxf(fmaxf(m2, bm2), bw ? m : bm);
    if (bw) { m = bm; i = bi; }
}

static __device__ __forceinline__ void cpush(float & m, int & bi, float & s, float & m2, float v, int i) {
    if (v > m) {   // indices arrive in increasing order per thread: strict > keeps the first max
        s = (m == -INFINITY ? 0.0f : s * __expf(m - v)) + 1.0f;
        m2 = m; m = v; bi = i;
    } else {
        s += __expf(v - m);
        m2 = fmaxf(m2, v);
    }
}

static __device__ __forceinline__ void cwarp(float & m, int & bi, float & s, float & m2) {
    for (int o = 16; o > 0; o >>= 1) {
        const float om = __shfl_xor_sync(0xffffffff, m, o), os = __shfl_xor_sync(0xffffffff, s, o);
        const float om2 = __shfl_xor_sync(0xffffffff, m2, o);
        const int oi = __shfl_xor_sync(0xffffffff, bi, o);
        cmerge(m, bi, s, m2, om, oi, os, om2);
    }
}

constexpr int CONF_MAXB = 64;   // partial blocks per row (conf_d_ holds CONF_MAXB partials + the result)

// block b scans l[b*chunk, min(n, (b+1)*chunk)) -> part[b]; 4 independent loads in flight per thread
static __global__ void __launch_bounds__(256) conf_part_k(const float * __restrict__ l, int n, int chunk, dconf * __restrict__ part) {
    float m = -INFINITY, s = 0.0f, m2 = -INFINITY;
    int bi = INT_MAX;
    const int b0 = blockIdx.x * chunk, b1 = min(n, b0 + chunk), bd = blockDim.x;
    for (int i = b0 + threadIdx.x; i < b1; i += 4 * bd) {
        float v[4];
#pragma unroll
        for (int u = 0; u < 4; ++u) v[u] = i + u * bd < b1 ? l[i + u * bd] : 0.0f;
#pragma unroll
        for (int u = 0; u < 4; ++u) if (i + u * bd < b1) cpush(m, bi, s, m2, v[u], i + u * bd);
    }
    cwarp(m, bi, s, m2);
    __shared__ dconf sh[8];
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = blockDim.x >> 5;
    if (lane == 0) sh[wid] = dconf{m, bi, s, m2};
    __syncthreads();
    if (wid == 0) {
        m = lane < nw ? sh[lane].m : -INFINITY;
        bi = lane < nw ? sh[lane].i : INT_MAX;
        s = lane < nw ? sh[lane].s : 0.0f;
        m2 = lane < nw ? sh[lane].m2 : -INFINITY;
        cwarp(m, bi, s, m2);
        if (lane == 0) part[blockIdx.x] = dconf{m, bi, s, m2};
    }
}

// one warp merges the nb partials -> *out
static __global__ void conf_final_k(const dconf * __restrict__ part, int nb, dconf * __restrict__ out) {
    float m = -INFINITY, s = 0.0f, m2 = -INFINITY;
    int bi = INT_MAX;
    for (int j = threadIdx.x; j < nb; j += 32) cmerge(m, bi, s, m2, part[j].m, part[j].i, part[j].s, part[j].m2);
    cwarp(m, bi, s, m2);
    if (threadIdx.x == 0) *out = dconf{m, bi == INT_MAX ? 0 : bi, s, m2};
}

static void conf_argmax(const float * l, int n, dconf * buf, cudaStream_t s) {
    const int nb = std::max(1, std::min(CONF_MAXB, (n + 2047) / 2048));
    const int chunk = (n + nb - 1) / nb;
    conf_part_k<<<nb, 256, 0, s>>>(l, n, chunk, buf);
    conf_final_k<<<1, 32, 0, s>>>(buf, nb, buf + CONF_MAXB);
}

tp_mtp::tp_mtp(tp_model & m) : m_(m) {
    if (!m.has_nextn()) throw std::runtime_error("model loaded without the nextn (MTP) block");
    static_assert(fast::P2P_MAX_RANKS <= 4, "conf_d_ holds 4 ranks");
    m.net_refuse("mtp (the MTP draft head)");   // (collect_at would need every node's confidence blocks)
    if (m.rk_.size() < 2) throw std::runtime_error("mtp: needs 2 to 4 GPUs");
    halves_ = m.rk_.size() == 2 && m.split_.rk[0].eh.b == 0 && m.split_.rk[0].eh.n == m.E_;
    kern_epoch_ = m.kern_epoch();   // no draft graphs yet; a later reload_env kernel switch drops the ones captured
    for (size_t r = 0; r < m_.rk_.size(); ++r) {
        auto & S = m_.rk_[r];
        CK(cudaSetDevice(S.dev));
        float *a = nullptr, *b = nullptr;
        CK(cudaMalloc(&a, (size_t) m_.max_batch_ * m_.E_ * 4));
        CK(cudaMemset(a, 0, (size_t) m_.max_batch_ * m_.E_ * 4));
        CK(cudaMalloc(&b, (size_t) m_.E_ * 4));
        CK(cudaMemset(b, 0, (size_t) m_.E_ * 4));
        hin_.push_back(a);
        hout_.push_back(b);
        float * hm = nullptr;
        CK(cudaMalloc(&hm, (size_t) tp_model::MAX_ROWS * m_.E_ * 4));
        CK(cudaMemset(hm, 0, (size_t) tp_model::MAX_ROWS * m_.E_ * 4));
        hmul_.push_back(hm);
        // confidence blocks: [0] the single-sequence run, [1 + i] row i of run_multi
        CK(cudaMalloc(&conf_d_[r], (size_t) (1 + tp_model::MAX_ROWS) * (CONF_MAXB + 1) * sizeof(dconf)));
        CK(cudaMemset(conf_d_[r], 0, (size_t) (1 + tp_model::MAX_ROWS) * (CONF_MAXB + 1) * sizeof(dconf)));
        CK(cudaDeviceSynchronize());   // legacy-stream memsets must land before work on the non-blocking stream
    }
    // pinned results: [r] single run, [R + R i + r] row i of run_multi
    CK(cudaMallocHost(&conf_h_, m_.rk_.size() * (size_t) (1 + tp_model::MAX_ROWS) * sizeof(dconf)));
    if (const char * e = getenv("P100E_DRAFT_FUSE")) fuse_ = atoi(e) != 0;
    if (const char * e = getenv("P100E_DRAFT_GRAPH")) graph_ = atoi(e) != 0;
    fuse_ = fuse_ && m_.fused_ok();
    graph_ = graph_ && fuse_ && m_.graph_maxt_ > 0;
    // draft LM head over the 32768 most frequent tokens by default (P100E_DRAFT_VOCAB=N, 0 = the full head); only
    // with the built-in ranking (a GGUF with another vocabulary keeps the full head unless asked)
    bool matched = false;
    const std::vector<int32_t> rank = default_ranking(&matched);
    const char * dv = getenv("P100E_DRAFT_VOCAB");
    if (dv || matched) set_draft_vocab(rank, dv ? atoi(dv) : 32768);
    printf("mtp draft head: %s%s, vocab %d\n", fuse_ ? "fused" : "per-op", graph_ ? " + CUDA graph" : "", draft_vocab());
}

tp_mtp::~tp_mtp() {
    drop_graphs();
    for (auto & [id, v] : carry_)
        for (size_t r = 0; r < v.size(); ++r) if (v[r]) { cudaSetDevice(m_.rk_[r].dev); cudaFree(v[r]); }
    for (size_t r = 0; r < hin_.size(); ++r) {
        cudaSetDevice(m_.rk_[r].dev);
        cudaFree(hin_[r]);
        cudaFree(hout_[r]);
        if (r < hmul_.size()) cudaFree(hmul_[r]);
        if (conf_d_[r]) cudaFree(conf_d_[r]);
        auto it = m_.rk_[r].w.find(kDraftHead);
        if (it != m_.rk_[r].w.end()) { cudaFree(it->second.d); m_.rk_[r].w.erase(it); }
    }
    if (conf_h_) cudaFreeHost(conf_h_);
}

void tp_mtp::drop_graphs() {
    for (auto & D : graphs_) {
        if (D.G.ge) cudaGraphExecDestroy(D.G.ge);
        if (D.G.g) cudaGraphDestroy(D.G.g);
    }
    graphs_.clear();
    for (auto & D : mgraphs_) {
        if (D.G.ge) cudaGraphExecDestroy(D.G.ge);
        if (D.G.g) cudaGraphDestroy(D.G.g);
    }
    mgraphs_.clear();
}

// No draft graph is in flight here: run() and run_multi() sync every rank before they return, and reload_env synced
// the streams when it bumped the epoch.
void tp_mtp::check_epoch() {
    if (kern_epoch_ == m_.kern_epoch()) return;
    kern_epoch_ = m_.kern_epoch();
    const size_t n = graphs_.size() + mgraphs_.size();
    if (n == 0) return;
    drop_graphs();
    ++n_epoch_drops_;
    if (!getenv("P100E_GRAPH_QUIET"))
        printf("mtp: a kernel switch changed (tp_model::reload_env): dropped %zu draft graph(s), re-captured on use\n", n);
}

std::vector<int32_t> tp_mtp::default_ranking(bool * matched) const {
    const gguf_value * tv = m_.g_.get("tokenizer.ggml.tokens");
    uint64_t h = 1469598103934665603ull;
    if (tv) {
        for (const auto & t : tv->arr_str) {
            for (unsigned char c : t) { h ^= c; h *= 1099511628211ull; }
            h ^= 0; h *= 1099511628211ull;
        }
    }
    const int V = m_.vocab();
    const bool ok = tv && (int) tv->arr_str.size() == kDraftVocabTokens && h == kDraftVocabHash && V == kDraftVocabTokens;
    if (matched) *matched = ok;
    std::vector<int32_t> r;
    std::vector<char> seen(V, 0);
    if (ok) {
        for (int i = 0; i < kDraftRankN; ++i) { r.push_back(kDraftRank[i]); seen[kDraftRank[i]] = 1; }
    } else {
        fprintf(stderr, "mtp: GGUF vocab does not match the built-in draft ranking: using token-id order\n");
        const gguf_value * ty = m_.g_.get("tokenizer.ggml.token_type");
        if (ty) for (int i = 0; i < V && i < (int) ty->arr_num.size(); ++i)
            if (ty->arr_num[i] != 1) { r.push_back(i); seen[i] = 1; }   // control / user-defined first
    }
    for (int i = 0; i < V; ++i) if (!seen[i]) r.push_back(i);
    return r;
}

void tp_mtp::set_draft_vocab(const std::vector<int32_t> & ranked, int n) {
    const int R = (int) m_.rk_.size(), V = m_.vocab();
    drop_graphs();   // the graphs bake the head
    for (int r = 0; r < R; ++r) {
        auto & S = m_.rk_[r];
        CK(cudaSetDevice(S.dev));
        CK(cudaStreamSynchronize(S.s));
        auto it = S.w.find(kDraftHead);
        if (it != S.w.end()) { cudaFree(it->second.d); S.w.erase(it); }
    }
    sub_ids_.assign(R, {});
    n_sub_ = 0;
    n = std::min<int>(n, (int) ranked.size());
    n -= n % (64 * R);
    if (n <= 0 || n >= V) return;
    const gguf_tensor * t = m_.g_.find("output.weight");
    if (!t) throw std::runtime_error("mtp: no output.weight");
    const qtype_info & ti = get_type_info(t->type);
    const size_t rb = (size_t) (t->ne[0] / ti.blck) * ti.size;
    const int nl = n / R;
    for (int r = 0; r < R; ++r) {
        // rank r scores ranked[r*nl .. (r+1)*nl): raw rows gathered from the GGUF, uploaded + packed like output.weight
        tp_model::host_slice hs;
        hs.type = t->type; hs.n_in = t->ne[0]; hs.n_out = nl; hs.row_bytes = rb;
        hs.h.resize((size_t) nl * rb);
        sub_ids_[r].assign(ranked.begin() + (size_t) r * nl, ranked.begin() + (size_t) (r + 1) * nl);
        for (int i = 0; i < nl; ++i) {
            const int id = sub_ids_[r][i];
            if (id < 0 || id >= V) throw std::runtime_error("mtp: draft vocab id out of range");
            memcpy(hs.h.data() + (size_t) i * rb, t->data + (size_t) id * rb, rb);
        }
        CK(cudaSetDevice(m_.rk_[r].dev));
        m_.upload(r, kDraftHead, std::move(hs));
    }
    n_sub_ = n;
}

void tp_mtp::h_from_main(int dst0, int src0, int n) {
    const size_t E = m_.E_;
    if (dst0 < 0 || n < 0 || dst0 + n > m_.max_batch_ || src0 + n > m_.max_batch_)
        throw std::runtime_error("mtp: hidden rows out of range (P100E_MTP_KV_ONLY=0 with a prompt longer than max_batch - 1?)");
    // eh_proj is row-parallel: only the h ranks (columns of the hidden half of its input; R = 2: rank 1) read h. hn is
    // replicated
    for (int r = 0; r < (int) m_.rk_.size(); ++r) {
        if (!h_rank(r)) continue;
        auto & S = m_.rk_[r];
        CK(cudaSetDevice(S.dev));
        CK(cudaMemcpyAsync(hin_[r] + dst0 * E, S.hn + src0 * E, n * E * 4, cudaMemcpyDeviceToDevice, S.s));
    }
}

void tp_mtp::h_zero(int row) {
    const size_t E = m_.E_;
    for (int r = 0; r < (int) m_.rk_.size(); ++r) {
        if (!h_rank(r)) continue;
        auto & S = m_.rk_[r];
        CK(cudaSetDevice(S.dev));
        CK(cudaMemsetAsync(hin_[r] + row * E, 0, E * 4, S.s));
    }
}

// per-sequence carry (h input row 0 between runs of one sequence when other sequences' runs come in between): device
// copies on the h ranks' streams (R = 2: rank 1), ordered with the runs; every h rank keeps the same row
void tp_mtp::carry_save(int seq) { carry_save_row(seq, 0); }

void tp_mtp::carry_save_row(int seq, int row) {
    const size_t E = m_.E_;
    if (row < 0 || row >= m_.max_batch_) throw std::runtime_error("mtp: carry row out of range");
    std::vector<float *> & c = carry_[seq];
    c.resize(m_.rk_.size(), nullptr);
    for (int r = 0; r < (int) m_.rk_.size(); ++r) {
        if (!h_rank(r)) continue;
        auto & S = m_.rk_[r];
        CK(cudaSetDevice(S.dev));
        if (!c[r]) CK(cudaMalloc(&c[r], E * 4));
        CK(cudaMemcpyAsync(c[r], hin_[r] + (size_t) row * E, E * 4, cudaMemcpyDeviceToDevice, S.s));
    }
}

void tp_mtp::carry_load(int seq) {
    const size_t E = m_.E_;
    auto it = carry_.find(seq);
    if (it == carry_.end()) { h_zero(0); return; }
    for (int r = 0; r < (int) m_.rk_.size(); ++r) {
        if (!h_rank(r)) continue;
        auto & S = m_.rk_[r];
        CK(cudaSetDevice(S.dev));
        CK(cudaMemcpyAsync(hin_[r], it->second[r], E * 4, cudaMemcpyDeviceToDevice, S.s));
    }
}

bool tp_mtp::carry_to_host(int seq, float * dst) {
    const size_t E = m_.E_;
    auto it = carry_.find(seq);
    if (it == carry_.end()) { memset(dst, 0, E * 4); return false; }
    int r = 0;   // the first h rank (all h ranks hold the same row)
    while (!h_rank(r)) ++r;
    auto & S = m_.rk_[r];
    CK(cudaSetDevice(S.dev));
    CK(cudaMemcpyAsync(dst, it->second[r], E * 4, cudaMemcpyDeviceToHost, S.s));
    CK(cudaStreamSynchronize(S.s));
    return true;
}

void tp_mtp::carry_from_host(int seq, const float * src) {
    const size_t E = m_.E_;
    std::vector<float *> & c = carry_[seq];
    c.resize(m_.rk_.size(), nullptr);
    for (int r = 0; r < (int) m_.rk_.size(); ++r) {
        if (!h_rank(r)) continue;
        auto & S = m_.rk_[r];
        CK(cudaSetDevice(S.dev));
        if (!c[r]) CK(cudaMalloc(&c[r], E * 4));
        CK(cudaMemcpyAsync(c[r], src, E * 4, cudaMemcpyHostToDevice, S.s));
        CK(cudaStreamSynchronize(S.s));
    }
}

void tp_mtp::carry_drop(int seq) {
    auto it = carry_.find(seq);
    if (it == carry_.end()) return;
    for (int r = 0; r < (int) m_.rk_.size(); ++r) {
        if (!h_rank(r)) continue;
        auto & S = m_.rk_[r];
        CK(cudaSetDevice(S.dev));
        CK(cudaStreamSynchronize(S.s));
        if (r < (int) it->second.size() && it->second[r]) cudaFree(it->second[r]);
    }
    carry_.erase(it);
}

const float * tp_mtp::eh_slice(int r, int T, bool fused) {
    auto & S = m_.rk_[r];
    const int E = m_.E_, il = m_.L_;
    const tp_range eh = m_.split_.rk[r].eh;           // this rank's columns of [enorm(e) | hnorm(h)]
    const int W = (int) eh.n;
    if (eh.b >= E || eh.e() <= E) {
        // inside one half (R = 4: quarters, the old layout): that half normed into S.cur, its columns -> S.c
        const bool hr = h_rank(r);
        const size_t c0 = (size_t) (hr ? eh.b - E : eh.b);   // first column inside its half
        m_.norm(hr ? hin_[r] : S.x, S.f[bn(il, hr ? "nextn.hnorm.weight" : "nextn.enorm.weight")], S.cur, T, E, S.s);
        CK(cudaMemcpy2DAsync(S.c, (size_t) W * 4, S.cur + c0, (size_t) E * 4, (size_t) W * 4, T, cudaMemcpyDeviceToDevice, S.s));
    } else {
        // straddling the halves (R = 3): the embedding half's tail [b, E) from S.cur, then the hidden half's head
        // [0, e - E) from S.e (free scratch until the nextn block's projections), side by side in S.c
        const size_t we = (size_t) (E - eh.b), wh = (size_t) (eh.e() - E);
        m_.norm(S.x, S.f[bn(il, "nextn.enorm.weight")], S.cur, T, E, S.s);
        m_.norm(hin_[r], S.f[bn(il, "nextn.hnorm.weight")], S.e, T, E, S.s);
        CK(cudaMemcpy2DAsync(S.c, (size_t) W * 4, S.cur + eh.b, (size_t) E * 4, we * 4, T, cudaMemcpyDeviceToDevice, S.s));
        CK(cudaMemcpy2DAsync(S.c + we, (size_t) W * 4, S.e, (size_t) E * 4, wh * 4, T, cudaMemcpyDeviceToDevice, S.s));
    }
    if (fused) fast::matvec_quantize(S.c, W, T, S.aq, S.s);
    return S.c;
}

// LM head (full or the draft subset) on h, then the greedy pick + confidence -> pinned conf_h_[r]
void tp_mtp::head(int r, const float * h, bool prequant) {
    auto & S = m_.rk_[r];
    const bool sub = n_sub_ > 0;
    const dev_weight & w = S.w[sub ? kDraftHead : "output.weight"];
    if (prequant) fast::matvec_prequantized(w.pk, h, S.aq, S.logits, 1, S.s);
    else m_.matmul(r, sub ? kDraftHead : "output.weight", h, S.logits, 1);
    conf_argmax(S.logits, (int) w.n_out, (dconf *) conf_d_[r], S.s);
    CK(cudaGetLastError());
    CK(cudaMemcpyAsync((dconf *) conf_h_ + r, (dconf *) conf_d_[r] + CONF_MAXB, sizeof(dconf), cudaMemcpyDeviceToHost, S.s));
}

int32_t tp_mtp::collect() { return collect_at((const dconf *) conf_h_, last_p_, last_margin_); }

// combine the ranks' (max, idx, sum, 2nd) entries c[0..R-1] -> token, probability, margin
int32_t tp_mtp::collect_at(const void * cv, float & p, float & margin) const {
    const dconf * c = (const dconf *) cv;
    const int R = (int) m_.rk_.size();
    int b = 0;
    for (int r = 1; r < R; ++r) if (c[r].m > c[b].m) b = r;   // a later slice wins only with a strictly larger value
    const float M = c[b].m;
    double s = 0;
    float second = c[b].m2;
    for (int r = 0; r < R; ++r) {
        s += (double) c[r].s * std::exp((double) c[r].m - M);
        if (r != b) second = std::max(second, c[r].m);
    }
    p = s > 0 ? (float) (1.0 / s) : 0.0f;
    margin = M - second;
    const int i = c[b].i;
    return n_sub_ > 0 ? sub_ids_[b][i] : i + (int) m_.rk_[b].v0;   // the full head: rank b's vocab rows start at v0
}

// row i -> dst[i] for every sequence i: its last row (row0 + T - 1, from the step descriptor's delta-net table S.gseq)
static __global__ void gather_last_k(const float * __restrict__ src, const int32_t * __restrict__ gseq, int E,
                                     float * __restrict__ dst) {
    const int i = blockIdx.x;
    const int row = gseq[4 * i] + gseq[4 * i + 1] - 1;
    const float4 * s = reinterpret_cast<const float4 *>(src + (size_t) row * E);
    float4 * d = reinterpret_cast<float4 *>(dst + (size_t) i * E);
    for (int k = threadIdx.x; k < E / 4; k += blockDim.x) d[k] = s[k];
}

void tp_mtp::h_from_multi(int dst, int i) {
    const size_t E = m_.E_;
    for (int r = 0; r < (int) m_.rk_.size(); ++r) {
        if (!h_rank(r)) continue;
        auto & S = m_.rk_[r];
        CK(cudaSetDevice(S.dev));
        CK(cudaMemcpyAsync(hin_[r] + (size_t) dst * E, hmul_[r] + (size_t) i * E, E * 4, cudaMemcpyDeviceToDevice, S.s));
    }
}

// the fused draft step for several sequences: as body(), with the nextn block's attention per row (positions and page
// tables of each row's sequence: fused_layer(.., nseq)), then every sequence's last row normed (the T-row norm is
// row-wise the single-row one), gathered, quantized and run through the LM head as nseq columns (column-invariant GEMV)
void tp_mtp::body_multi(int T, int nseq) {
    const int R = (int) m_.rk_.size(), E = m_.E_, il = m_.L_;
    for (int r = 0; r < R; ++r) {
        auto & S = m_.rk_[r];
        CK(cudaSetDevice(S.dev));
        const float * xe = S.cur;
        if (!halves_) {
            xe = eh_slice(r, T, true);
        } else {
            float * xin = r == 0 ? S.x : hin_[r];
            const float * w = S.f[bn(il, r == 0 ? "nextn.enorm.weight" : "nextn.hnorm.weight")];
            if (!fast::add2_rmsnorm_q(xin, nullptr, nullptr, w, S.cur, S.aq, T, E, m_.eps_, S.s))
                fast::matvec_quantize(S.cur, E, T, S.aq, S.s);
        }
        fast::matvec_prequantized(S.w[bn(il, "nextn.eh_proj.weight")].pk, xe, S.aq, m_.part_out(r), T, S.s);
        CK(cudaMemsetAsync(S.x, 0, (size_t) T * E * 4, S.s));
    }
    m_.ar_post(T);
    m_.fused_layer(il, T, false, nseq);
    for (int r = 0; r < R; ++r) {
        auto & S = m_.rk_[r];
        CK(cudaSetDevice(S.dev));
        auto it = S.f.find(bn(il, "nextn.shared_head_norm.weight"));
        const float * hw = it != S.f.end() ? it->second : S.f["output_norm.weight"];
        m_.norm_in(r, hw, true, T);   // S.cur = normed rows (x += the pending partials)
        gather_last_k<<<nseq, 256, 0, S.s>>>(S.cur, S.gseq, E, hmul_[r]);
        CK(cudaGetLastError());
        fast::matvec_quantize(hmul_[r], E, nseq, S.aq, S.s);
        const bool sub = n_sub_ > 0;
        const dev_weight & w = S.w[sub ? kDraftHead : "output.weight"];
        fast::matvec_prequantized(w.pk, hmul_[r], S.aq, S.logits, nseq, S.s);
        for (int i = 0; i < nseq; ++i) {
            dconf * cd = (dconf *) conf_d_[r] + (size_t) (1 + i) * (CONF_MAXB + 1);
            conf_argmax(S.logits + (size_t) i * w.n_out, (int) w.n_out, cd, S.s);
            CK(cudaMemcpyAsync((dconf *) conf_h_ + R + R * i + r, cd + CONF_MAXB, sizeof(dconf), cudaMemcpyDeviceToHost, S.s));
        }
        CK(cudaGetLastError());
    }
}

std::vector<int32_t> tp_mtp::run_multi(const std::vector<tp_model::seq_rows> & b, const std::vector<int> & p0) {
    const int nseq = (int) b.size();
    if (nseq < 2 || nseq > tp_model::MAX_ROWS || (int) p0.size() != nseq) throw std::runtime_error("mtp: run_multi needs 2..8 sequences");
    if (!fuse_) throw std::runtime_error("mtp: run_multi needs the fused draft head");
    const int R = (int) m_.rk_.size();
    int T = 0;
    for (const auto & s : b) {
        if (s.tokens.empty() || s.tokens.size() > 4) throw std::runtime_error("mtp: run_multi 1..4 rows per sequence");
        T += (int) s.tokens.size();
    }
    if (T > tp_model::MAX_ROWS) throw std::runtime_error("mtp: run_multi more than 8 rows");
    check_epoch();
    // step descriptor (as tp_model::forward_batch_ids): row positions, page tables, per-sequence rows
    std::vector<int32_t> toks;
    int row = 0;
    bool mr = false;   // a sequence with images: rotary positions per row (S.rpos), as tp_model::forward_batch_ids
    for (int i = 0; i < nseq; ++i) {
        const int Ti = (int) b[i].tokens.size();
        if (p0[i] + Ti > m_.n_ctx_) throw std::runtime_error("mtp: context full");
        m_.ensure_pages(b[i].seq, p0[i] + Ti);
        const int slot = m_.seqs_.at(b[i].seq).slot;
        for (int t = 0; t < Ti; ++t) {
            m_.pos_h_[row + t] = p0[i] + t;
            m_.desc_h_[tp_model::DESC_RT + row + t] = slot * m_.max_pages_;
            mr = m_.stage_rpos_text(row + t, m_.seqs_.at(b[i].seq), p0[i] + t) || mr;
        }
        int32_t * g = m_.desc_h_ + tp_model::DESC_GS + 4 * i;
        g[0] = row; g[1] = Ti; g[2] = slot; g[3] = 0;
        row += Ti;
        toks.insert(toks.end(), b[i].tokens.begin(), b[i].tokens.end());
    }
    for (auto & S : m_.rk_) {
        CK(cudaSetDevice(S.dev));
        CK(cudaMemcpyAsync(S.desc, m_.desc_h_, (size_t) (tp_model::DESC_POS + T) * 4, cudaMemcpyHostToDevice, S.s));
    }
    m_.step_mr_ = mr;
    if (mr) m_.upload_rpos(T);
    m_.embed(toks);
    mgraph * D = nullptr;
    if (graph_ && !graph_failed_) for (auto & g : mgraphs_) if (g.T == T && g.nseq == nseq && g.G.mr == mr && g.G.sig == m_.sig_fuse_ && g.G.q8f == m_.q8_fork_) D = &g;
    if (D) m_.launch_graph(D->G);
    else body_multi(T, nseq);
    for (int r = 0; r < R; ++r) { CK(cudaSetDevice(m_.rk_[r].dev)); CK(cudaStreamSynchronize(m_.rk_[r].s)); }
    std::vector<int32_t> picks(nseq);
    mp_.assign(nseq, 0.0f);
    for (int i = 0; i < nseq; ++i) {
        float margin = 0.0f;
        picks[i] = collect_at((const dconf *) conf_h_ + R + R * i, mp_[i], margin);
    }
    if (!D && graph_ && !graph_failed_) {   // this eager run was the warm-up: capture (records only)
        mgraph G;
        G.T = T; G.nseq = nseq; G.G.mr = mr;
        std::string err;
        if (m_.capture_body([&] { body_multi(T, nseq); }, G.G, err)) {
            mgraphs_.push_back(G);
            ++n_capt_;
            if (!getenv("P100E_GRAPH_QUIET")) printf("CUDA graph draft T=%d (%d sequences): %zu nodes\n", T, nseq, G.G.nodes);
        } else {
            fprintf(stderr, "CUDA graph (draft T=%d, %d sequences): %s, eager draft runs from now on\n", T, nseq, err.c_str());
            graph_failed_ = true;
        }
    }
    return picks;
}

// fused draft step: rank 0 enorm(embedding), rank 1 hnorm(h) -> eh_proj (row-parallel) -> allreduce -> the nextn block
// through tp_model::fused_layer (plan S.lp[L_], KV cache S.kv[L_], positions from S.pos) -> shared_head_norm of the
// last row -> LM head -> pick. Ends with rank 1's hin row 0 = that normed hidden (the next chained run's h).
// R = 4: ranks 0, 1 the embedding half's column quarters, ranks 2, 3 the hidden half's (eh_slice); every h rank keeps h.
// R = 3 (or any split other than the R = 2 halves): each rank its eh columns (tp_split eh), a straddling rank both halves.
void tp_mtp::body(int T, bool from_main) {
    const int R = (int) m_.rk_.size(), E = m_.E_, il = m_.L_;
    for (int r = 0; r < R; ++r) {
        auto & S = m_.rk_[r];
        CK(cudaSetDevice(S.dev));
        if (from_main && h_rank(r)) CK(cudaMemcpyAsync(hin_[r], S.hn, (size_t) T * E * 4, cudaMemcpyDeviceToDevice, S.s));
        const float * xe = S.cur;
        if (!halves_) {
            xe = eh_slice(r, T, true);
        } else {
            float * xin = r == 0 ? S.x : hin_[r];
            const float * w = S.f[bn(il, r == 0 ? "nextn.enorm.weight" : "nextn.hnorm.weight")];
            if (!fast::add2_rmsnorm_q(xin, nullptr, nullptr, w, S.cur, S.aq, T, E, m_.eps_, S.s))
                fast::matvec_quantize(S.cur, E, T, S.aq, S.s);
        }
        fast::matvec_prequantized(S.w[bn(il, "nextn.eh_proj.weight")].pk, xe, S.aq, m_.part_out(r), T, S.s);
        CK(cudaMemsetAsync(S.x, 0, (size_t) T * E * 4, S.s));   // x = 0 + partials (as the per-op allreduce_add)
    }
    m_.ar_post(T);
    m_.fused_layer(il, T, false);
    for (int r = 0; r < R; ++r) {
        auto & S = m_.rk_[r];
        CK(cudaSetDevice(S.dev));
        auto it = S.f.find(bn(il, "nextn.shared_head_norm.weight"));
        const float * hw = it != S.f.end() ? it->second : S.f["output_norm.weight"];
        m_.norm_last(r, hw, T, hout_[r]);
        if (h_rank(r)) CK(cudaMemcpyAsync(hin_[r], hout_[r], (size_t) E * 4, cudaMemcpyDeviceToDevice, S.s));
        head(r, hout_[r], true);
    }
}

void tp_mtp::eh_in(int T) {
    const int R = (int) m_.rk_.size();
    const int il = m_.L_;
    const int E = m_.E_;
    for (int r = 0; r < R; ++r) {
        auto & S = m_.rk_[r];
        CK(cudaSetDevice(S.dev));
        // eh_proj is row-parallel over its input: rank 0 holds the embedding half, rank 1 the hidden half (R = 4: the
        // quarters, R = 3: 256-column blocks, eh_slice)
        if (!halves_) {
            m_.matmul(r, bn(il, "nextn.eh_proj.weight"), eh_slice(r, T, false), S.part, T);
            CK(cudaMemsetAsync(S.x, 0, (size_t) T * E * 4, S.s));
            continue;
        }
        if (r == 0) m_.norm(S.x, S.f[bn(il, "nextn.enorm.weight")], S.cur, T, E, S.s);
        else        m_.norm(hin_[r], S.f[bn(il, "nextn.hnorm.weight")], S.cur, T, E, S.s);
        m_.matmul(r, bn(il, "nextn.eh_proj.weight"), S.cur, S.part, T);
        CK(cudaMemsetAsync(S.x, 0, (size_t) T * E * 4, S.s));
    }
    m_.allreduce_add(T);   // x = eh_proj(concat(enorm(e), hnorm(h)))
}

void tp_mtp::catchup_kv(const std::vector<int32_t> & tokens, int p0) {
    const int T = (int) tokens.size();
    if (T < 1 || T > m_.max_batch_) throw std::runtime_error("mtp: catch-up batch size out of range");
    if (p0 + T > m_.n_ctx_) throw std::runtime_error("mtp: context full");
    m_.reserve_positions(p0 + T);   // the nextn block's KV rows live in the active sequence's pages
    if (T > 1) h_from_main(1, 0, T - 1);   // h rows 1..T-1 = h_{p0}..h_{p0+T-2}; row 0 = the carry
    m_.set_pos(p0, T);
    m_.embed(tokens);
    eh_in(T);
    m_.layer(m_.L_, p0, T, false, true);   // K / V rows only (the same ops as the full per-op block)
    h_from_main(0, T - 1, 1);              // carry = h_{p0+T-1} (after the rows above consumed row 0: same stream)
    for (auto & S : m_.rk_) { CK(cudaSetDevice(S.dev)); CK(cudaStreamSynchronize(S.s)); }
}

void tp_mtp::run_perop(int T, int p0) {
    const int R = (int) m_.rk_.size();
    const int il = m_.L_;
    const int E = m_.E_;
    eh_in(T);
    m_.layer(il, p0, T, false);
    for (int r = 0; r < R; ++r) {
        auto & S = m_.rk_[r];
        CK(cudaSetDevice(S.dev));
        auto it = S.f.find(bn(il, "nextn.shared_head_norm.weight"));
        const float * hw = it != S.f.end() ? it->second : S.f["output_norm.weight"];
        m_.norm(S.x + (size_t) (T - 1) * E, hw, hout_[r], 1, E, S.s);
        if (h_rank(r)) CK(cudaMemcpyAsync(hin_[r], hout_[r], (size_t) E * 4, cudaMemcpyDeviceToDevice, S.s));
        head(r, hout_[r], false);
    }
}

int32_t tp_mtp::run(const std::vector<int32_t> & tokens, int p0, hsrc src) {
    const int T = (int) tokens.size();
    if (T < 1 || T > m_.max_batch_) throw std::runtime_error("mtp: batch size out of range");
    if (p0 + T > m_.n_ctx_) throw std::runtime_error("mtp: context full");
    if (src == H_SELF && T != 1) throw std::runtime_error("mtp: chained run needs T = 1");
    const int R = (int) m_.rk_.size();
    m_.reserve_positions(p0 + T);   // drafts write the nextn KV ahead of the main model's n_past
    m_.set_pos(p0, T);
    m_.embed(tokens);   // S.x = embed(tokens) (rank 0's is read)
    const bool from_main = src == H_MAIN;
    if (fuse_ && T <= 4) {
        check_epoch();
        dgraph * D = nullptr;
        const int slot = m_.sq().slot;
        if (graph_ && !graph_failed_)
            for (auto & g : graphs_) if (g.T == T && g.from_main == from_main && g.slot == slot && g.G.mr == m_.mrope_now() && g.G.sig == m_.sig_fuse_ && g.G.q8f == m_.q8_fork_) D = &g;
        if (D) {
            m_.launch_graph(D->G);
        } else {
            body(T, from_main);
        }
        for (int r = 0; r < R; ++r) { CK(cudaSetDevice(m_.rk_[r].dev)); CK(cudaStreamSynchronize(m_.rk_[r].s)); }
        const int32_t tok = collect();
        if (!D && graph_ && !graph_failed_) {   // this eager run was the warm-up: capture (records only)
            dgraph G;
            G.T = T; G.from_main = from_main; G.slot = slot; G.G.mr = m_.mrope_now();   // M-RoPE attn_prep baked in (images)
            std::string err;
            if (m_.capture_body([&] { body(T, from_main); }, G.G, err)) {
                graphs_.push_back(G);
                ++n_capt_;
                if (!getenv("P100E_GRAPH_QUIET"))
                    printf("CUDA graph draft T=%d%s: %zu nodes\n", T, from_main ? " (h from main)" : "", G.G.nodes);
            } else {
                fprintf(stderr, "CUDA graph (draft T=%d): %s, eager draft runs from now on\n", T, err.c_str());
                graph_failed_ = true;
            }
        }
        return tok;
    }
    if (from_main) h_from_main(0, 0, T);
    run_perop(T, p0);
    for (int r = 0; r < R; ++r) { CK(cudaSetDevice(m_.rk_[r].dev)); CK(cudaStreamSynchronize(m_.rk_[r].s)); }
    return collect();
}

// one implementation for both prompt forms: `pieces` (multimodal, appended to the current context) or `prompt_ids`
// (tokens from position 0, the original path)
static std::vector<int32_t> generate_mtp_impl(tp_model & m, tp_mtp & d, const std::vector<int32_t> & prompt,
                                              const std::vector<mtp_piece> * pieces, int n_gen, const mtp_opts & o,
                                              mtp_stats & st) {
    const int V = m.vocab();
    const int k = o.k;
    if (k < 1 || k > 3) throw std::runtime_error("mtp: k must be 1..3 (verify batch <= 4 for the attention kernel)");
    const bool adaptive = o.tau > 0.0f;
    const int k_min = adaptive ? std::max(0, std::min(o.k_min, k)) : k;
    m.enable_rollback(k + 1);
    std::vector<int32_t> out;
    using clk = std::chrono::steady_clock;
    static const bool kv_only = !getenv("P100E_MTP_KV_ONLY") || atoi(getenv("P100E_MTP_KV_ONLY")) != 0;
    int P = (int) prompt.size();
    int carry_row = -1;   // pieces: the main model's hn row of the context's last row in its latest forward
    const auto t_start = clk::now();
    std::vector<float> L;
    if (pieces) {
        // multimodal: text pieces in ubatch chunks, image pieces through forward_image; each chunk followed by the draft
        // head's KV-only rows of its entries (image rows: the embedding rows as the draft head's input embeddings, at
        // their M-RoPE positions - forward_image calls back with the rows still staged)
        if (!kv_only) throw std::runtime_error("mtp: a multimodal prompt needs the KV-only catch-up (P100E_MTP_KV_ONLY=1)");
        if (pieces->empty() || pieces->back().embd || pieces->back().tokens.empty())
            throw std::runtime_error("mtp: the prompt must end with text");
        // the carry h_{p0-1} (h input row 0): zero at the start of a fresh context; a continued context (the next turn
        // after an earlier generate_mtp(pieces)) keeps the h of its last row, left there by that call (end of this
        // function). Zeroing it there made the first draft rows of every follow-up turn see a wrong h (acceptance only).
        static const bool carry_zero = getenv("P100E_MTP_CARRY_ZERO") && atoi(getenv("P100E_MTP_CARRY_ZERO")) != 0;
        if (m.n_past() == 0 || carry_zero) d.h_zero(0);
        std::vector<tp_model::mm_in> in;
        for (const auto & pc : *pieces) {
            tp_model::mm_in q;
            if (pc.embd) { q.embd = pc.embd; q.nx = pc.nx; q.ny = pc.ny; q.n_embd = pc.n_embd; }
            else { q.tokens = pc.tokens.data(); q.n_tok = (int) pc.tokens.size(); }
            in.push_back(q);
        }
        const auto a = clk::now();
        double tc = 0;
        L = m.forward_mm(in, [&](int p0, const std::vector<int32_t> & ch) {
            const auto b = clk::now();
            d.catchup_kv(ch, p0);
            carry_row = (int) ch.size() - 1;
            tc += std::chrono::duration<double>(clk::now() - b).count();
        });
        st.t_prompt += std::chrono::duration<double>(clk::now() - a).count() - tc;
        st.t_catchup += tc;
        P = m.n_past();
    } else if (kv_only) {
        // the main model's ubatch chunks (forward_n's split: the same numerics as a plain forward of the prompt), each
        // followed by the draft head's KV-only rows of its entries (h from the chunk's hn rows + the carry)
        d.h_zero(0);
        int c0 = 0;
        for (int c : m.ubatch_chunks(P, 1)) {
            const std::vector<int32_t> ch(prompt.begin() + c0, prompt.begin() + c0 + c);
            auto a = clk::now();
            L = m.forward_n(ch, 1);
            st.t_prompt += std::chrono::duration<double>(clk::now() - a).count();
            a = clk::now();
            d.catchup_kv(ch, c0);
            st.t_catchup += std::chrono::duration<double>(clk::now() - a).count();
            c0 += c;
        }
    } else {
        const auto a = clk::now();
        L = m.forward_n(prompt, 1);
        st.t_prompt += std::chrono::duration<double>(clk::now() - a).count();
    }
    int32_t last = argmax(L.data(), V);
    out.push_back(last);
    auto drun = [&](const std::vector<int32_t> & t, int p0, tp_mtp::hsrc src) {   // timed draft-head run (ends with a sync)
        const auto a = clk::now();
        auto r = d.run(t, p0, src);
        st.t_draft += std::chrono::duration<double>(clk::now() - a).count();
        st.n_draft++;
        return r;
    };
    // drafts after a first draft-head run (already done: its pick d0): chain up to k, adaptive: keep draft j while
    // the product of the draft head's top-1 probabilities stays >= tau (at least k_min drafts)
    std::vector<int32_t> drafts;
    std::vector<float> dp, dc, dm;   // per verified draft: top-1 probability, running product, logit margin
    auto chain = [&](int32_t d0, int N) {
        drafts.clear(); dp.clear(); dc.clear(); dm.clear();
        double cum = d.last_p();
        if (0 >= k_min && cum < (o.tau1 > 0 ? o.tau1 : o.tau)) return;
        drafts.push_back(d0); dp.push_back(d.last_p()); dc.push_back((float) cum); dm.push_back(d.last_margin());
        for (int j = 1; j < k; ++j) {
            const int32_t dj = drun({drafts.back()}, N + j, tp_mtp::H_SELF);
            cum *= d.last_p();
            if (j >= k_min && cum < o.tau) break;
            drafts.push_back(dj); dp.push_back(d.last_p()); dc.push_back((float) cum); dm.push_back(d.last_margin());
        }
    };

    // MTP catch-up over the prompt + first draft: entry p = (token p, h_{p-1}); position 0 gets h = 0
    int N = P;   // position of `last`
    {
        const auto a = clk::now();
        int32_t d0;
        if (kv_only || pieces) {
            d0 = drun({last}, N, tp_mtp::H_READY);   // the full row of `last` (h input row 0 = h_{P-1}, the carry)
        } else {
            std::vector<int32_t> mt(prompt);
            mt.push_back(last);
            d.h_zero(0);
            d.h_from_main(1, 0, P);
            d0 = drun(mt, 0, tp_mtp::H_READY);
        }
        st.t_catchup += std::chrono::duration<double>(clk::now() - a).count();
        chain(d0, N);
    }

    while ((int) out.size() < n_gen) {
        std::vector<int32_t> batch{last};
        batch.insert(batch.end(), drafts.begin(), drafts.end());
        const int T = (int) batch.size();
        const auto tv = clk::now();
        std::vector<int32_t> g;
        std::vector<uint64_t> gh;
        if (o.ref_hash) {   // debug: full logits of every row, hashed
            const std::vector<float> lg = m.forward_n(batch, T, true);
            for (int r = 0; r < T; ++r) {
                g.push_back(argmax(lg.data() + (size_t) r * V, V));
                uint64_t h = 1469598103934665603ull;
                const unsigned char * b = (const unsigned char *) (lg.data() + (size_t) r * V);
                for (size_t i = 0; i < (size_t) V * 4; ++i) { h ^= b[i]; h *= 1099511628211ull; }
                gh.push_back(h);
            }
        } else {
            g = m.forward_ids(batch, T, true);   // greedy token after each batch row
        }
        st.t_verify += std::chrono::duration<double>(clk::now() - tv).count();
        int a = 0;
        while (a < (int) drafts.size() && g[a] == drafts[a]) ++a;
        const int32_t nt = g[a];
        st.steps++;
        st.drafted += (int) drafts.size();
        st.accepted += a;
        st.hist[std::min(4, (int) drafts.size())]++;
        if (o.ref_hash) {   // per run (mtp_stats): every mismatching row counted, the first 12 printed
            for (int r = 0; r <= a && r < T; ++r) {
                const size_t ti = out.size() + r;   // generated-token index this row predicts
                if (ti < o.ref_hash->size() && gh[r] != (*o.ref_hash)[ti] && ++st.chk_bad <= 12) {
                    printf("CHECK logits differ: token %zu (step %d, T=%d, row %d, prev T=%d prev a=%d)\n", ti, st.steps, T, r,
                           st.chk_prev_T, st.chk_prev_a);
                }
            }
            st.chk_prev_T = T; st.chk_prev_a = a;
        }
        for (int j = 0; j < (int) drafts.size() && j <= a; ++j) st.cal.push_back({j + 1, dp[j], dc[j], dm[j], j < a});
        std::vector<int32_t> acc(drafts.begin(), drafts.begin() + a);
        acc.push_back(nt);
        // keep the model state consistent with the returned tokens: on the last step emit only the q tokens
        // still needed and keep the state after `last` + the q-1 drafts preceding the final emitted token
        const int q = std::min((int) acc.size(), n_gen - (int) out.size());
        m.rollback(T, q);
        carry_row = q - 1;   // the verify's hn rows: row q - 1 is the context's last row now
        for (int i = 0; i < q; ++i) out.push_back(acc[i]);
        if ((int) out.size() >= n_gen) break;
        // MTP: catch-up rows for the accepted drafts + the full row for nt (positions N+1..N+a+1), h = main hn rows
        const int32_t d0 = drun(acc, N + 1, tp_mtp::H_MAIN);
        N += a + 1;
        last = nt;
        chain(d0, N);
    }
    // pieces: leave the carry = h of the context's last row for a follow-up generate_mtp(pieces) on this context (the
    // chained drafts overwrote it with the draft head's own hidden). Same stream as every later hn write: ordered.
    if (pieces && carry_row >= 0) d.h_from_main(0, carry_row, 1);
    st.t_gen = std::chrono::duration<double>(clk::now() - t_start).count() - st.t_prompt;
    return out;
}

std::vector<int32_t> generate_mtp(tp_model & m, tp_mtp & d, const std::vector<int32_t> & prompt, int n_gen,
                                  const mtp_opts & o, mtp_stats & st) {
    return generate_mtp_impl(m, d, prompt, nullptr, n_gen, o, st);
}

std::vector<int32_t> generate_mtp(tp_model & m, tp_mtp & d, const std::vector<mtp_piece> & prompt, int n_gen,
                                  const mtp_opts & o, mtp_stats & st) {
    return generate_mtp_impl(m, d, {}, &prompt, n_gen, o, st);
}

} // namespace p100e
